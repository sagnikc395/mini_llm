import torch
from torch import nn
from mini_llm.config import GPT_CONFIG_124M as cfg
from mini_llm.training.mini_llm_gpt import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(
            cfg.context_length,
        )
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(*[])
