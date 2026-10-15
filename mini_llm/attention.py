"""Causal self-attention, in the two forms the book builds up to.

`CausalAttention` is one head written out plainly; `MultiHeadAttention` is the
batched version the model actually uses, where all heads live in one set of
projection matrices and are split apart with a reshape.
"""

import torch
from torch import nn


class CausalAttention(nn.Module):
    """A single masked attention head.

    Kept because it is the readable version: `MultiHeadAttention` does the same
    arithmetic with an extra heads dimension threaded through every tensor.
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # upper triangle (excluding the diagonal) = the future, which every
        # position must be stopped from attending to
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        _, num_tokens, _ = x.shape
        queries, keys, values = self.w_query(x), self.w_key(x), self.w_value(x)

        # transpose the token and feature dims, leaving the batch dim at 0
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # scale by sqrt(head dim) to keep the softmax out of its flat regions
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values


class MultiHeadAttention(nn.Module):
    """All heads at once: project to d_out, then split that into num_heads."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError(
                f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"
            )

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # mixes the per-head outputs back together after they are concatenated
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def _split_heads(self, x, b, num_tokens):
        """(b, tokens, d_out) -> (b, heads, tokens, head_dim)."""
        return x.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        b, num_tokens, _ = x.shape

        queries = self._split_heads(self.w_query(x), b, num_tokens)
        keys = self._split_heads(self.w_key(x), b, num_tokens)
        values = self._split_heads(self.w_value(x), b, num_tokens)

        # one dot product per head: (b, heads, tokens, tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # back to (b, tokens, heads, head_dim), then flatten the heads away
        context = (attn_weights @ values).transpose(1, 2)
        context = context.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)
