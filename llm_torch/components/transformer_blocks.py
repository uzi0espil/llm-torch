from __future__ import annotations
from torch import nn

from llm_torch.configs.configs import ModelConfig


class TransformerBlock(nn.Module):

    def __init__(self, model_cfg: ModelConfig, context_length, attention, ff_block, activation, norm):
        super().__init__()

        self.dropout = nn.Dropout(model_cfg.drop_rate) if model_cfg.drop_rate is not None else None
        self.ff = ff_block(model_cfg.emb_dim, model_cfg.hidden_dim, activation=activation, dtype=model_cfg.dtype)
        self.mha = attention(d_in=model_cfg.emb_dim, d_out=model_cfg.emb_dim,
                             context_length=context_length, dropout_rate=model_cfg.drop_rate,
                             n_heads=model_cfg.n_heads, qkv_bias=model_cfg.qkv_bias, dtype=model_cfg.dtype,
                             kv_window_size=model_cfg.kv_window_size, theta_base=500_000)
        self.ln1 = norm(model_cfg.emb_dim)
        self.ln2 = norm(model_cfg.emb_dim)

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
