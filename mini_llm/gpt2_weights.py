"""Fetch OpenAI's original GPT-2 weights and copy them into our GPTModel.

OpenAI never published PyTorch state dicts -- the weights are TensorFlow v1
checkpoints in a public Azure blob container, which is why tensorflow appears
as a dependency even though the model itself is pure PyTorch. The checkpoint is
read once into a nested dict of numpy arrays that mirrors the GPT-2 variable
names, and `load_weights_into_gpt` is the bridge from those names to our module
attributes.

Two conversions matter throughout:
  - TF stores dense layers as (in, out); nn.Linear stores (out, in), so every
    weight matrix is transposed on the way in.
  - the three attention projections share one fused `c_attn` tensor, which is
    split into query/key/value thirds along the last axis.
"""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mini_llm.config import GPT2_SIZES
from mini_llm.paths import GPT2_DIR

BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"

# every checkpoint directory ships this same set of files: hparams.json has the
# architecture settings, model.ckpt.* are the weights, and encoder.json/vocab.bpe
# are the BPE tokenizer (we use tiktoken instead, but they keep the directory
# self-contained)
FILENAMES = (
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
)


def _download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream `url` to `destination`, skipping it if already complete.

    The 1558M checkpoint is ~6 GB, so a resumed session must not re-fetch what
    is already on disk. We compare against Content-Length rather than merely
    checking existence, because an interrupted run leaves behind a truncated
    file that would otherwise look valid -- and we write to a `.part` name so
    that situation stops arising in the first place.
    """
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))

        if destination.exists() and destination.stat().st_size == total_size:
            print(f"{destination.name} already up to date, skipping")
            return

        partial = destination.with_suffix(destination.suffix + ".part")
        progress = tqdm(
            total=total_size, unit="iB", unit_scale=True, desc=destination.name
        )
        with open(partial, "wb") as f, progress:
            while chunk := response.read(chunk_size):
                f.write(chunk)
                progress.update(len(chunk))

    os.replace(partial, destination)


def download_gpt2_files(size: str, models_dir: Path = GPT2_DIR) -> Path:
    """Fetch every checkpoint file for `size` and return the directory."""
    if size not in GPT2_SIZES:
        raise ValueError(f"size must be one of {tuple(GPT2_SIZES)}, got {size!r}")

    model_dir = Path(models_dir) / size
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILENAMES:
        _download_file(f"{BASE_URL}/{size}/{filename}", model_dir / filename)
    return model_dir


def _params_from_tf_checkpoint(ckpt_path: str, n_layers: int) -> dict:
    """Convert the TF checkpoint into a nested dict of numpy arrays.

    TF stores variables under flat slash-separated names like
    `model/h0/attn/c_attn/w`. We rebuild the nesting those names imply: per
    layer `h<N>` variables go into `params["blocks"][N]`, everything else
    (`wte`, `wpe`, the final `ln_f`) stays at the top level.
    """
    # imported here, not at module scope: tensorflow is slow to import and is
    # only needed to read these checkpoints
    import tensorflow as tf

    params = {"blocks": [{} for _ in range(n_layers)]}

    for name, _ in tf.train.list_variables(ckpt_path):
        # squeeze because TF keeps the conv1d weights as (1, in, out); the
        # matching nn.Linear layers want plain 2-D matrices
        array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # drop the leading "model/" scope, which every variable shares
        parts = name.split("/")[1:]

        target = params
        if parts[0].startswith("h"):
            target = params["blocks"][int(parts[0][1:])]
        # walk/create the intermediate scopes, e.g. "attn" -> "c_attn"
        for key in parts[1:-1]:
            target = target.setdefault(key, {})
        target[parts[-1]] = array

    return params


def download_and_load_gpt2(
    size: str = "124M", models_dir: Path | str = GPT2_DIR
) -> tuple[dict, dict]:
    """Download `size` if needed and return `(settings, params)`.

    `settings` is OpenAI's hparams.json; `params` is the nested numpy dict.
    """
    model_dir = download_gpt2_files(size, Path(models_dir))
    settings = json.loads((model_dir / "hparams.json").read_text(encoding="utf-8"))
    # tf.train wants the checkpoint prefix, not a file that exists on disk
    params = _params_from_tf_checkpoint(
        str(model_dir / "model.ckpt"), settings["n_layer"]
    )
    return settings, params


def _assign(left: torch.nn.Parameter, right) -> torch.nn.Parameter:
    """Wrap `right` as a Parameter, refusing a mismatched shape.

    The check is the whole point: a silent mismatch produces a model that runs
    and emits fluent-looking garbage, far harder to debug than an exception at
    load time.
    """
    if left.shape != right.shape:
        raise ValueError(
            f"shape mismatch: model {left.shape}, checkpoint {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params: dict) -> None:
    """Copy the checkpoint dict `params` into `gpt`, in place."""
    if gpt.trf_blocks[0].att.w_query.bias is None:
        raise ValueError(
            "the checkpoint carries qkv biases but the model was built with "
            "qkv_bias=False; build it from gpt2_config() instead"
        )

    gpt.tok_emb.weight = _assign(gpt.tok_emb.weight, params["wte"])
    gpt.pos_emb.weight = _assign(gpt.pos_emb.weight, params["wpe"])

    for block, block_params in zip(gpt.trf_blocks, params["blocks"]):
        att, ff = block.att, block.ff

        # attention: split the fused qkv tensors into their three thirds
        q_w, k_w, v_w = np.split(block_params["attn"]["c_attn"]["w"], 3, axis=-1)
        q_b, k_b, v_b = np.split(block_params["attn"]["c_attn"]["b"], 3, axis=-1)
        att.w_query.weight = _assign(att.w_query.weight, q_w.T)
        att.w_key.weight = _assign(att.w_key.weight, k_w.T)
        att.w_value.weight = _assign(att.w_value.weight, v_w.T)
        att.w_query.bias = _assign(att.w_query.bias, q_b)
        att.w_key.bias = _assign(att.w_key.bias, k_b)
        att.w_value.bias = _assign(att.w_value.bias, v_b)

        proj = block_params["attn"]["c_proj"]
        att.out_proj.weight = _assign(att.out_proj.weight, proj["w"].T)
        att.out_proj.bias = _assign(att.out_proj.bias, proj["b"])

        # feed forward: layers[0] is the 4x expansion, layers[2] projects back
        # down (layers[1] is GELU and carries no weights)
        for layer_idx, tf_name in ((0, "c_fc"), (2, "c_proj")):
            layer = ff.layers[layer_idx]
            layer.weight = _assign(layer.weight, block_params["mlp"][tf_name]["w"].T)
            layer.bias = _assign(layer.bias, block_params["mlp"][tf_name]["b"])

        # our LayerNorm calls the learned parameters scale/shift; TF calls them
        # g (gain) and b (bias)
        for norm, tf_name in ((block.norm1, "ln_1"), (block.norm2, "ln_2")):
            norm.scale = _assign(norm.scale, block_params[tf_name]["g"])
            norm.shift = _assign(norm.shift, block_params[tf_name]["b"])

    gpt.final_norm.scale = _assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = _assign(gpt.final_norm.shift, params["b"])
    # GPT-2 ties the output head to the token embeddings, so both read "wte"
    gpt.out_head.weight = _assign(gpt.out_head.weight, params["wte"])
