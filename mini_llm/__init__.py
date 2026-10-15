"""A small GPT, written from scratch.

The pieces, roughly in the order they stack up:

    config.py        GPTConfig, and the four GPT-2 sizes
    tokenizer.py     a word-level tokenizer (reference only; we use tiktoken)
    attention.py     CausalAttention, MultiHeadAttention
    model.py         GELU, LayerNorm, FeedForward, TransformerBlock, GPTModel
    generate.py      text <-> token ids, and the sampling loop
    data.py          the pretraining corpus, sliced into training windows
    training.py      losses, the training loop, device selection
    gpt2_weights.py  OpenAI's pretrained checkpoints -> our GPTModel
    spam.py          the spam-classification finetuning dataset
    plotting.py      loss curves
    cli.py           the `mini-llm` command

Importing any of these is free of side effects: nothing downloads, trains or
prints at import time. The step-by-step explorations live in notes/ instead.
"""

from mini_llm.config import GPT2_SIZES, GPTConfig, gpt2_config
from mini_llm.generate import generate, text_to_token_ids, token_ids_to_text
from mini_llm.model import GPTModel

__all__ = [
    "GPT2_SIZES",
    "GPTConfig",
    "GPTModel",
    "generate",
    "gpt2_config",
    "text_to_token_ids",
    "token_ids_to_text",
]
