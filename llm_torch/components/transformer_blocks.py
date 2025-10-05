from __future__ import annotations
from torch import nn
from dataclasses import asdict

from llm_torch.configs.configs import ModelConfig


class TransformerBlock(nn.Module):

    def __init__(self, model_cfg: ModelConfig, context_length):
        super().__init__()
        self.dropout = nn.Dropout(model_cfg.drop_rate) if model_cfg.drop_rate is not None else None
        self.ff = model_cfg.ff_block_config.instantiate(model_cfg.emb_dim, dtype=model_cfg.dtype)
        self.mha = model_cfg.attention_config.instantiate(
            context_length, d_in=model_cfg.emb_dim, d_out=model_cfg.emb_dim, dtype=model_cfg.dtype
        )
        self.ln1 = model_cfg.normalizer_config.instantiate(model_cfg.emb_dim)
        self.ln2 = model_cfg.normalizer_config.instantiate(model_cfg.emb_dim)

    def forward(self, x, use_cache=False):
        residual_connection = x
        x = self.ln1(x)
        x = self.mha(x, use_cache=use_cache)
        if self.dropout is not None:
            x = self.dropout(x)
        x += residual_connection

        residual_connection = x
        x = self.ln2(x)
        x = self.ff(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x + residual_connection
