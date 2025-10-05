import torch
from torch import nn

from llm_torch.architectures.base import BaseLLMModel
from llm_torch.components.transformer_blocks import TransformerBlock
from llm_torch.components import (
    MultiHeadAttention,
    LayerNorm,
)
from llm_torch.configs import ModelConfig


class GPT2(BaseLLMModel):

    def __init__(self, model_cfg: ModelConfig, vocab_size, context_length):
        super().__init__(model_cfg, vocab_size, context_length)

        self.tok_embedding = nn.Embedding(vocab_size, model_cfg.emb_dim, dtype=model_cfg.dtype)
        self.pos_embedding = nn.Embedding(context_length, model_cfg.emb_dim, dtype=model_cfg.dtype)

        self.blocks = nn.ModuleList([TransformerBlock(
            model_cfg,
            context_length=context_length
        ) for _ in range(model_cfg.n_layers)])

        self.dropout = nn.Dropout(model_cfg.drop_rate)
        self.norm = model_cfg.normalizer_config.instantiate(model_cfg.emb_dim)
        self.output = nn.Linear(model_cfg.emb_dim, vocab_size, dtype=model_cfg.dtype, bias=False)
        self.current_position = 0

    @property
    def transformer_blocks(self) -> torch.nn.ModuleList:
        return self.blocks

    def forward(self, x, use_cache=False):
        batch_size, seq_length = x.shape
        token_emb = self.tok_embedding(x)
        if use_cache:
            pos_ids = torch.arange(self.current_position, self.current_position + seq_length, device=x.device)
            self.current_position += seq_length
        else:
            pos_ids = torch.arange(0, seq_length, device=x.device)
        pos_emb = self.pos_embedding(pos_ids).unsqueeze(0)

        x = token_emb + pos_emb
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.norm(x)
        return self.output(x)

    def reset_kv_cache(self):
        self.current_position = 0
        super().reset_kv_cache()
