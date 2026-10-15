"""Attention built up by hand, before any of it becomes an nn.Module.

Covers: dot-product scores, softmax normalization, context vectors, then the
trainable Q/K/V projections and the causal mask.
"""

import torch
from torch import nn

# six three-dimensional "token embeddings" standing in for a real sentence
INPUTS = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)


def unnormalized_attention():
    """Scores for one query, three ways, then the context vector they produce."""
    query = INPUTS[1]

    # a score is just the dot product of the query with every token
    scores = torch.tensor([torch.dot(x_i, query) for x_i in INPUTS])
    print(f"scores (loop):       {scores}")
    print(f"scores (matmul):     {INPUTS @ query}")

    # dividing by the sum normalizes, but softmax is better behaved with
    # negative scores and gives a sharper distribution
    print(f"weights (sum):       {scores / scores.sum()}")
    weights = torch.softmax(scores, dim=0)
    print(f"weights (softmax):   {weights}  (sums to {weights.sum():.4f})")

    # the context vector is the weighted average of every token
    print(f"context vector:      {weights @ INPUTS}")


def all_pairs_attention():
    """The same thing for every query at once: one matmul instead of a loop."""
    scores = INPUTS @ INPUTS.T
    weights = torch.softmax(scores, dim=-1)
    print(f"all weights:\n{weights}")
    print(f"row sums: {weights.sum(dim=-1)}")
    print(f"all context vectors:\n{weights @ INPUTS}")


class SelfAttention(nn.Module):
    """Self-attention with learned projections, written with raw Parameters.

    nn.Linear (as in mini_llm/attention.py) is the same computation with a
    better weight initialization; this version just makes the three matrices
    impossible to miss.
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_query = nn.Parameter(torch.rand(d_in, d_out))
        self.w_key = nn.Parameter(torch.rand(d_in, d_out))
        self.w_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries, keys, values = x @ self.w_query, x @ self.w_key, x @ self.w_value
        scores = queries @ keys.T
        # scale by sqrt(d_k) so the softmax does not saturate as d_k grows
        weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)
        return weights @ values


def causal_masking():
    """Two ways to stop a position from attending to its own future."""
    torch.manual_seed(789)
    d_in, d_out = INPUTS.shape[1], 2

    w_query = nn.Linear(d_in, d_out, bias=False)
    w_key = nn.Linear(d_in, d_out, bias=False)
    scores = w_query(INPUTS) @ w_key(INPUTS).T
    weights = torch.softmax(scores / d_out**0.5, dim=-1)

    n = scores.shape[0]

    # the obvious way: zero the future out, then renormalize the rows
    masked = weights * torch.tril(torch.ones(n, n))
    print(f"masked then renormalized:\n{masked / masked.sum(dim=-1, keepdim=True)}")

    # the efficient way: -inf before the softmax, so it normalizes correctly
    # the first time. this is what MultiHeadAttention does.
    masked = scores.masked_fill(
        torch.triu(torch.ones(n, n), diagonal=1).bool(), -torch.inf
    )
    print(f"-inf before softmax:\n{torch.softmax(masked / d_out**0.5, dim=-1)}")

    # dropout on the weights, applied after the softmax
    torch.manual_seed(123)
    print(
        f"with 50% dropout:\n{nn.Dropout(0.5)(torch.softmax(masked / d_out**0.5, dim=-1))}"
    )


if __name__ == "__main__":
    unnormalized_attention()
    all_pairs_attention()

    torch.manual_seed(123)
    print(f"trainable self-attention:\n{SelfAttention(INPUTS.shape[1], 2)(INPUTS)}")

    causal_masking()
