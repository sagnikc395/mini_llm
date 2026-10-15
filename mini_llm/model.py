"""The GPT model, bottom up: GELU -> FeedForward -> LayerNorm -> block -> model."""

import torch
from torch import nn

from mini_llm.attention import MultiHeadAttention
from mini_llm.config import GPTConfig


class GELU(nn.Module):
    """The tanh approximation of GELU that GPT-2 was trained with.

    torch.nn.GELU defaults to the exact erf form, which is close but not
    bit-identical, so we spell this one out to match the checkpoints.
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class LayerNorm(nn.Module):
    """Layer norm with GPT-2's parameter names (scale/shift, i.e. gain/bias)."""

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False matches GPT-2's original TF implementation
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, GELU, project back down."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Pre-norm attention and feed-forward, each wrapped in a residual."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        # the residual is what keeps gradients alive through 12+ stacked blocks
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    """Token + position embeddings -> N transformer blocks -> vocab logits."""

    def __init__(self, cfg: GPTConfig | None = None):
        super().__init__()
        self.cfg = cfg or GPTConfig()
        cfg = self.cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    @property
    def context_length(self) -> int:
        return self.pos_emb.weight.shape[0]

    def forward(self, in_idx):
        """(batch, tokens) token ids -> (batch, tokens, vocab_size) logits."""
        _, seq_len = in_idx.shape

        # positions are built on the input's device so the model runs wherever
        # the data happens to live
        pos = torch.arange(seq_len, device=in_idx.device)
        x = self.drop_emb(self.tok_emb(in_idx) + self.pos_emb(pos))
        x = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))
