# Copy OpenAI's original GPT-2 weights into our own GPTModel.
#
# `download_gpt_weights.download_and_load_gpt2` hands back the checkpoint as a
# nested dict of numpy arrays that mirrors the TensorFlow variable names. This
# module is the bridge from those names to our module attributes.
#
# Two conversions matter throughout:
#   - TF stores the dense layers as (in, out); torch.nn.Linear stores (out, in),
#     so every weight matrix is transposed on the way in.
#   - the three attention projections share one fused `c_attn` tensor in the
#     checkpoint, so it is split into query/key/value thirds along the last axis.

from dataclasses import replace

import numpy as np
import torch

from mini_llm.config import GPT_CONFIG_124M

# emb_dim / n_layers / n_heads for the four checkpoints OpenAI published, keyed
# by the same size strings download_gpt_weights uses
MODEL_CONFIGS = {
    "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def config_for_gpt2(model_size: str = "124M") -> GPT_CONFIG_124M:
    """Return the config describing the pretrained GPT-2 checkpoint `model_size`.

    Two settings are not negotiable here: the context length is 1024 because
    that is the size of the checkpoint's positional embedding table, and
    qkv_bias must be True because OpenAI trained the attention projections with
    biases and `load_weights_into_gpt` has to put them somewhere.
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"model_size must be one of {tuple(MODEL_CONFIGS)}, got {model_size!r}"
        )

    return replace(
        GPT_CONFIG_124M(),
        context_length=1024,
        qkv_bias=True,
        **MODEL_CONFIGS[model_size],
    )


def assign(left: torch.nn.Parameter, right) -> torch.nn.Parameter:
    """Wrap `right` as a Parameter, refusing to load a mismatched tensor.

    The shape check is the whole point: a silent mismatch here would produce a
    model that runs and generates fluent-looking garbage, which is far harder to
    debug than an exception at load time.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params: dict) -> None:
    """Load the checkpoint dict `params` into `gpt` in place."""
    if gpt.trf_blocks[0].att.w_query.bias is None:
        raise ValueError(
            "the checkpoint carries qkv biases but the model was built with "
            "qkv_bias=False; build it from config_for_gpt2() instead"
        )

    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])

    for b, block_params in enumerate(params["blocks"]):
        block = gpt.trf_blocks[b]

        # attention: split the fused qkv tensors into their three thirds
        q_w, k_w, v_w = np.split(block_params["attn"]["c_attn"]["w"], 3, axis=-1)
        block.att.w_query.weight = assign(block.att.w_query.weight, q_w.T)
        block.att.w_key.weight = assign(block.att.w_key.weight, k_w.T)
        block.att.w_value.weight = assign(block.att.w_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(block_params["attn"]["c_attn"]["b"], 3, axis=-1)
        block.att.w_query.bias = assign(block.att.w_query.bias, q_b)
        block.att.w_key.bias = assign(block.att.w_key.bias, k_b)
        block.att.w_value.bias = assign(block.att.w_value.bias, v_b)

        block.att.out_proj.weight = assign(
            block.att.out_proj.weight, block_params["attn"]["c_proj"]["w"].T
        )
        block.att.out_proj.bias = assign(
            block.att.out_proj.bias, block_params["attn"]["c_proj"]["b"]
        )

        # feed forward: layers[0] is the 4x expansion, layers[2] the projection
        # back down (layers[1] is GELU and carries no weights)
        block.ff.layers[0].weight = assign(
            block.ff.layers[0].weight, block_params["mlp"]["c_fc"]["w"].T
        )
        block.ff.layers[0].bias = assign(
            block.ff.layers[0].bias, block_params["mlp"]["c_fc"]["b"]
        )
        block.ff.layers[2].weight = assign(
            block.ff.layers[2].weight, block_params["mlp"]["c_proj"]["w"].T
        )
        block.ff.layers[2].bias = assign(
            block.ff.layers[2].bias, block_params["mlp"]["c_proj"]["b"]
        )

        # our LayerNorm calls the learned parameters scale/shift; TF calls them
        # g (gain) and b (bias)
        block.norm1.scale = assign(block.norm1.scale, block_params["ln_1"]["g"])
        block.norm1.shift = assign(block.norm1.shift, block_params["ln_1"]["b"])
        block.norm2.scale = assign(block.norm2.scale, block_params["ln_2"]["g"])
        block.norm2.shift = assign(block.norm2.shift, block_params["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # GPT-2 ties the output head to the token embeddings, so both read "wte"
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
