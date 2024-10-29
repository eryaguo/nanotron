import sys
sys.path.append('.venv/lib/python3.10/site-packages')

import argparse
import os
from dataclasses import asdict
import json
from pathlib import Path
import torch
from tqdm import tqdm
import yaml
from nanotron import logging
from nanotron.config import Config, LoggingArgs, ParallelismArgs, get_config_from_file
from nanotron.config.models_config import ExistingCheckpointInit, Idefics2VisionConfig, Idefics2Config
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.config.models_config import LlamaConfig as LlamaConfigNanotron
from nanotron.models.base import build_model
from nanotron.models.llama import LlamaForTraining
from nanotron.models.idefics import Idefics2ForTraining, VisionTransformer
from nanotron.parallel.context import ParallelContext
from nanotron.trainer import mark_tied_parameters
from nanotron.parallel.parameters import sanity_check
from nanotron.serialize import load_weights
from nanotron.serialize.weights import save_weights
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.models.llama import LlamaConfig as LlamaConfigHF
from transformers import Idefics2Config as Idefics2ConfigHF
from transformers import SigLIPConfig as SigLIPConfigHF



logger = logging.get_logger(__name__)

DEVICE = torch.device("cpu")
TORCH_DTYPE = torch.bfloat16

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory with a Nanotron Checkpoint",
    )

    group = parser.add_argument_group(title="HuggingFace Model")
    group.add_argument(
        "--hugging-face-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory to store the converted checkpoint",
    )

    args = parser.parse_args()

    return args

def copy_weights_from_nanotron_to_hf_llama(nanotron_model, hf_model, nanotron_llama_config):
    # Copy params from Nanotron to HF
    log_rank("Copying weights from Nanotron model to HF model...", logger=logger, level=logging.INFO, rank=0)
    # Token embeddings
    log_rank("Copying Token Embeddings...", logger=logger, level=logging.INFO, rank=0)
    assert (
        nanotron_model.model.token_position_embeddings.pp_block.token_embedding.weight.shape
        == hf_model.model.embed_tokens.weight.shape
    )
    with torch.no_grad():
        hf_model.model.embed_tokens.weight.copy_(
            nanotron_model.token_position_embeddings.pp_block.token_embedding.weight
        )

    # Decoder layers
    for i in tqdm(
        range(nanotron_llama_config.num_hidden_layers),
        desc="Copying Hidden Layers",
        total=nanotron_llama_config.num_hidden_layers,
    ):
        # Input layer norm
        assert (
            hf_model.model.layers[i].input_layernorm.weight.shape
            == nanotron_model.model.decoder[i].pp_block.input_layernorm.weight.shape
        )
        with torch.no_grad():
            hf_model.model.layers[i].input_layernorm.weight.copy_(
                nanotron_model.decoder[i].pp_block.input_layernorm.weight
            )

        # Self attn
        ## Split Nanotrn qkv projection into q, k, v
        q, k, v = torch.split(
            nanotron_model.model.decoder[i].pp_block.attn.qkv_proj.weight,
            [
                nanotron_llama_config.num_attention_heads * nanotron_model.model.decoder[i].pp_block.attn.d_qk,
                nanotron_llama_config.num_key_value_heads * nanotron_model.model.decoder[i].pp_block.attn.d_qk,
                nanotron_llama_config.num_key_value_heads * nanotron_model.model.decoder[i].pp_block.attn.d_qk,
            ],
        )
        assert q.shape == hf_model.model.layers[i].self_attn.q_proj.weight.shape
        assert k.shape == hf_model.model.layers[i].self_attn.k_proj.weight.shape
        assert v.shape == hf_model.model.layers[i].self_attn.v_proj.weight.shape

        with torch.no_grad():
            hf_model.model.layers[i].self_attn.q_proj.weight.copy_(q)
            hf_model.model.layers[i].self_attn.k_proj.weight.copy_(k)
            hf_model.model.layers[i].self_attn.v_proj.weight.copy_(v)

        # O
        assert (
            hf_model.model.layers[i].self_attn.o_proj.weight.shape
            == nanotron_model.model.decoder[i].pp_block.attn.o_proj.weight.shape
        )
        with torch.no_grad():
            hf_model.model.layers[i].self_attn.o_proj.weight.copy_(
                nanotron_model.model.decoder[i].pp_block.attn.o_proj.weight
            )

        # MLP
        ## Gate Up Proj
        gate_proj, up_proj = torch.split(
            nanotron_model.model.decoder[i].pp_block.mlp.gate_up_proj.weight,
            split_size_or_sections=[nanotron_llama_config.intermediate_size, nanotron_llama_config.intermediate_size],
        )
        assert gate_proj.shape == hf_model.model.layers[i].mlp.gate_proj.weight.shape
        assert up_proj.shape == hf_model.model.layers[i].mlp.up_proj.weight.shape

        with torch.no_grad():
            hf_model.model.layers[i].mlp.gate_proj.weight.copy_(gate_proj)
            hf_model.model.layers[i].mlp.up_proj.weight.copy_(up_proj)

        ## Down Proj
        assert (
            hf_model.model.layers[i].mlp.down_proj.weight.shape
            == nanotron_model.model.decoder[i].pp_block.mlp.down_proj.weight.shape
        )
        with torch.no_grad():
            hf_model.model.layers[i].mlp.down_proj.weight.copy_(
                nanotron_model.model.decoder[i].pp_block.mlp.down_proj.weight
            )

        # Post attn layer norm
        assert (
            hf_model.model.layers[i].post_attention_layernorm.weight.shape
            == nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight.shape
        )
        with torch.no_grad():
            hf_model.model.layers[i].post_attention_layernorm.weight.copy_(
                nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight
            )

    # Last layer norm
    log_rank("Copying Final Layer Norm...", logger=logger, level=logging.INFO, rank=0)
    assert nanotron_model.model.final_layer_norm.pp_block.weight.shape == hf_model.model.norm.weight.shape
    with torch.no_grad():
        hf_model.model.norm.weight.copy_(nanotron_model.model.final_layer_norm.pp_block.weight)

    # LM_Head
    log_rank("Copying LM Head...", logger=logger, level=logging.INFO, rank=0)
    assert nanotron_model.model.lm_head.pp_block.weight.shape == hf_model.lm_head.weight.shape
    with torch.no_grad():
        hf_model.lm_head.weight.copy_(nanotron_model.model.lm_head.pp_block.weight)


def copy_weights_from_nanotron_to_hf_siglip(
    nanotron_model: VisionTransformer,
    hf_model: AutoModel,
    nanotron_vision_config: Idefics2VisionConfig
):
    log_rank("Copying weights from Nanotron SigLIP model to HF model...", logger=logger, level=logging.INFO, rank=0)

    # Vision Embeddings
    log_rank("Copying Vision Embeddings...", logger=logger, level=logging.INFO, rank=0)

    
    assert (
        nanotron_model.embeddings.pp_block.patch_embedding.weight.shape == hf_model.embeddings.patch_embedding.weight.shape
    )
    assert(
        nanotron_model.embeddings.pp_block.patch_embedding.bias.shape == hf_model.embeddings.patch_embedding.bias.shape
    )
    assert (
        nanotron_model.embeddings.pp_block.position_embedding.weight.shape
        == hf_model.embeddings.position_embedding.weight.shape
    )
    with torch.no_grad():
        hf_model.embeddings.patch_embedding.weight.copy_(
            nanotron_model.embeddings.pp_block.patch_embedding.weight
        )
        hf_model.embeddings.patch_embedding.bias.copy_(
            nanotron_model.embeddings.pp_block.patch_embedding.bias
        )
        hf_model.embeddings.position_embedding.weight.copy_(
            nanotron_model.embeddings.pp_block.position_embedding.weight
        )

    log_rank("Copied Vision Embeddings", logger=logger, level=logging.INFO, rank=0)

    for i in tqdm(
        range(nanotron_vision_config.num_hidden_layers),
        desc="Copying Vision Layers",
        total=nanotron_vision_config.num_hidden_layers,
    ):
        assert (
            nanotron_model.encoder[i].pp_block.layer_norm1.weight.shape == hf_model.encoder.layers[i].layer_norm1.weight.shape
        )
        hf_model.encoder.layers[i].layer_norm1.weight.copy_(
            nanotron_model.encoder[i].pp_block.layer_norm1.weight
        )

        tmp_qkv_proj = nanotron_model.encoder[i].pp_block.self_attn.qkv_proj.weight.chunk(3, dim=0)
        
        assert (
            tmp_qkv_proj[0].shape == hf_model.encoder.layers[i].self_attn.k_proj.weight.shape
        )
        assert (
            tmp_qkv_proj[1].shape == hf_model.encoder.layers[i].self_attn.v_proj.weight.shape
        )
        assert (
            tmp_qkv_proj[2].shape == hf_model.encoder.layers[i].self_attn.q_proj.weight.shape
        )
        with torch.no_grad():
            hf_model.encoder.layers[i].self_attn.k_proj.weight.copy_(tmp_qkv_proj[0])
            hf_model.encoder.layers[i].self_attn.v_proj.weight.copy_(tmp_qkv_proj[1])
            hf_model.encoder.layers[i].self_attn.q_proj.weight.copy_(tmp_qkv_proj[2])
    
        tmp_qkv_proj_bias = nanotron_model.encoder[i].pp_block.self_attn.qkv_proj.bias.chunk(3, dim=0)
        
        assert (
            tmp_qkv_proj_bias[0].shape == hf_model.encoder.layers[i].self_attn.k_proj.bias.shape
        )
        assert (
            tmp_qkv_proj_bias[1].shape == hf_model.encoder.layers[i].self_attn.v_proj.bias.shape
        )
        assert (
            tmp_qkv_proj_bias[2].shape == hf_model.encoder.layers[i].self_attn.q_proj.bias.shape
        )
        with torch.no_grad():
            hf_model.encoder.layers[i].self_attn.k_proj.bias.copy_(tmp_qkv_proj_bias[0])
            hf_model.encoder.layers[i].self_attn.v_proj.bias.copy_(tmp_qkv_proj_bias[1])
            hf_model.encoder.layers[i].self_attn.q_proj.bias.copy_(tmp_qkv_proj_bias[2])

        ## O
        assert (
            nanotron_model.encoder[i].pp_block.self_attn.o_proj.weight.shape == hf_model.encoder.layers[i].self_attn.o_proj.weight.shape
        )
        with torch.no_grad():
            hf_model.encoder.layers[i].self_attn.o_proj.weight.copy_(
                nanotron_model.encoder[i].pp_block.self_attn.o_proj.weight
            )

        # Layer norm 2
        assert (
            nanotron_model.encoder[i].pp_block.layer_norm2.weight.shape == hf_model.encoder.layers[i].layer_norm2.weight.shape
        )
        with torch.no_grad():
            hf_model.encoder.layers[i].layer_norm2.weight.copy_(
                nanotron_model.encoder[i].pp_block.layer_norm2.weight
            )

        # MLP
        ## FC1
        assert (
            nanotron_model.encoder[i].pp_block.mlp.fc1.weight.shape == hf_model.encoder.layers[i].mlp.fc1.weight.shape
        )
        with torch.no_grad():
            hf_model.encoder.layers[i].mlp.fc1.weight.copy_(
                nanotron_model.encoder[i].pp_block.mlp.fc1.weight
            )
        
        assert (
            nanotron_model.encoder[i].pp_block.mlp.fc1.bias.shape == hf_model.encoder.layers[i].mlp.fc1.bias.shape
        )
        with torch.no_grad():
            hf_model.encoder.layers[i].mlp.fc1.bias.copy_(
                nanotron_model.encoder[i].pp_block.mlp.fc1.bias
            )

        ## FC2
        assert (
            nanotron_model.encoder[i].pp_block.mlp.fc2.weight.shape == hf_model.encoder.layers[i].mlp.fc2.weight.shape
        )
        with torch.no_grad():
            hf_model.encoder.layers[i].mlp.fc2.weight.copy_(
                nanotron_model.encoder[i].pp_block.mlp.fc2.weight
            )

        assert (
            nanotron_model.encoder[i].pp_block.mlp.fc2.bias.shape == hf_model.encoder.layers[i].mlp.fc2.bias.shape
        )
        with torch.no_grad():   
            hf_model.encoder.layers[i].mlp.fc2.bias.copy_(
                nanotron_model.encoder[i].pp_block.mlp.fc2.bias
            )
    log_rank("Copied Vision Layers", logger=logger, level=logging.INFO, rank=0)

    # Post layer norm
    assert (
        nanotron_model.post_layernorm.pp_block.weight.shape == hf_model.post_layernorm.weight.shape
    )
    with torch.no_grad():
        hf_model.post_layernorm.weight.copy_(nanotron_model.post_layernorm.pp_block.weight)
    
    assert (
        nanotron_model.post_layernorm.pp_block.bias.shape == hf_model.post_layernorm.bias.shape
    )
    with torch.no_grad():
        hf_model.post_layernorm.bias.copy_(nanotron_model.post_layernorm.pp_block.bias)

    log_rank("Copied Post Layer Norm", logger=logger, level=logging.INFO, rank=0)


def main(args):
    # Init Nanotron Parallel Utilities
    parallel_config = ParallelismArgs(dp=1, pp=1, tp=1)

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    set_ranks_logging_level(parallel_context=parallel_context, logging_config=LoggingArgs())

    # Load Nanotron checkpoint config
    log_rank(
        f"Loading Nanotron checkpoint config file: {os.path.join(args.nanotron_checkpoint_path, 'config.yaml')}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    nanotron_config = get_config_from_file(
        os.path.join(args.nanotron_checkpoint_path, "config.yaml"), config_class=Config, model_config_class=None
    )
    nanotron_idefics2_config = nanotron_config.model.model_config
    nanotron_llama_config = nanotron_idefics2_config.llama_config
    nanotron_vision_config = nanotron_idefics2_config.vision_config


    # Init Idefics2 Nanotron model
    log_rank("Init empty Nanotron Idefics2 Model", logger=logger, level=logging.INFO, rank=0)

    nanotron_model = build_model(
        model_builder=lambda: Idefics2ForTraining(
            config=nanotron_idefics2_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        ),
        parallel_context=parallel_context,
        dtype=TORCH_DTYPE,
        device=DEVICE,
    )

    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)
    sanity_check(root_module=nanotron_model)

    # Load Nanotron Checkpoint
    log_rank("Loading Nanotron Idefics2 Model...", logger=logger, level=logging.INFO, rank=0)
    load_weights(
        model=nanotron_model, parallel_context=parallel_context, root_folder=Path(args.nanotron_checkpoint_path)
    )

    # Build empty HF Model
    log_rank("Init empty HF Llama3 Model", logger=logger, level=logging.INFO, rank=0)
    hf_llama_model = AutoModelForCausalLM.from_config(  # WARN This takes a long time
        config=LlamaConfigHF(**asdict(nanotron_llama_config)),
        torch_dtype=TORCH_DTYPE,
        attn_implementation="flash_attention_2",
    ).to(DEVICE)

    log_rank("Init empty HF SigLIP Model", logger=logger, level=logging.INFO, rank=0)
    hf_siglip_model = AutoModel.from_config(
        config=SigLIPConfigHF(**asdict(nanotron_vision_config)),
        torch_dtype=TORCH_DTYPE,
        attn_implementation="flash_attention_2",
    ).to(DEVICE)

    log_rank("Init empty HF Idefics2 Model", logger=logger, level=logging.INFO, rank=0)
    hf_idefics2_model = AutoModel.from_config(
        config=Idefics2ConfigHF(**asdict(nanotron_idefics2_config)),
        torch_dtype=TORCH_DTYPE,
        attn_implementation="flash_attention_2",
    ).to(DEVICE)

    # Copy weights from Nanotron to Hugging Face
    copy_weights_from_nanotron_to_hf_llama(
        nanotron_model=nanotron_model.model.llama,
        hf_model=hf_llama_model,
        nanotron_llama_config=nanotron_llama_config,
        parallel_context=parallel_context,
    )

    log_rank("Copied weights from Nanotron Llama model to HF model!", logger=logger, level=logging.INFO, rank=0)
    
    copy_weights_from_nanotron_to_hf_siglip(
        nanotron_model=nanotron_model.model.vision_model,
        hf_model=hf_siglip_model,
        nanotron_vision_config=nanotron_vision_config,
    )

    log_rank("Copied weights from Nanotron SigLIP model to HF model!", logger=logger, level=logging.INFO, rank=0)


    # Store weights
    log_rank("Saving HF model Checkpoint and Tokenizer!", logger=logger, level=logging.INFO, rank=0)
    hf_llama_model.save_pretrained(args.hugging_face_checkpoint_path_llama, from_pt=True)
    # Store tokenizer
    tokenizer_llama = AutoTokenizer.from_pretrained(nanotron_llama_config.tokenizer.tokenizer_name_or_path)
    tokenizer_llama.save_pretrained(args.hugging_face_checkpoint_path_llama)
    log_rank(
        f"Checkpoint conversion finished, check {args.hugging_face_checkpoint_path_llama}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    
    # Store weights
    hf_siglip_model.save_pretrained(args.hugging_face_checkpoint_path_siglip, from_pt=True)
    # Store tokenizer
    tokenizer_siglip = AutoTokenizer.from_pretrained(nanotron_vision_config.tokenizer.tokenizer_name_or_path)
    tokenizer_siglip.save_pretrained(args.hugging_face_checkpoint_path_siglip)
    log_rank(
        f"Checkpoint conversion finished, check {args.hugging_face_checkpoint_path_siglip}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    

if __name__ == "__main__":
    _args = get_args()
    main(_args)
