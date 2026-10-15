"""Model geometry: one dataclass, plus the four sizes OpenAI published.

Everything that needs to know how big the model is reads it from here, so
there is exactly one place where `emb_dim` and friends are spelled out.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    # OpenAI trained the attention projections with biases; our from-scratch
    # training runs do not need them.
    qkv_bias: bool = False


# emb_dim / n_layers / n_heads for OpenAI's four GPT-2 checkpoints, keyed by the
# same size strings the download URLs use.
GPT2_SIZES = {
    "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def gpt2_config(size: str = "124M") -> GPTConfig:
    """Config matching the pretrained GPT-2 checkpoint `size`.

    Two settings are not negotiable: context_length is 1024 because that is the
    size of the checkpoint's positional embedding table, and qkv_bias must be
    True because the checkpoint carries attention biases that have to go
    somewhere. drop_rate is 0 because we only ever run these weights for
    inference.
    """
    if size not in GPT2_SIZES:
        raise ValueError(f"size must be one of {tuple(GPT2_SIZES)}, got {size!r}")

    return replace(
        GPTConfig(),
        context_length=1024,
        qkv_bias=True,
        drop_rate=0.0,
        **GPT2_SIZES[size],
    )
