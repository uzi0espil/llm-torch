import torch
from torch import nn

from llm_torch.architectures.base import BaseLLMModel
from llm_torch.components.transformer_blocks import TransformerBlock
from llm_torch.components import (
    GELU,
    MultiHeadAttention,
    LayerNorm,
    FFBlock,
)
from llm_torch.configs import ModelConfig


class GPT2(BaseLLMModel):

    def __init__(self, model_cfg: ModelConfig, vocab_size, context_length):
        super().__init__(model_cfg, vocab_size, context_length)

        self.tok_embedding = nn.Embedding(vocab_size, model_cfg.emb_dim)
        self.pos_embedding = nn.Embedding(context_length, model_cfg.emb_dim)

        self.blocks = nn.Sequential(*[TransformerBlock(model_cfg,
                                                       context_length=context_length,
                                                       attention=MultiHeadAttention,
                                                       norm=LayerNorm,
                                                       ff_block=FFBlock,
                                                       activation=GELU) for _ in range(model_cfg.n_layers)])
        self.dropout = nn.Dropout(model_cfg.drop_rate)
        self.norm = LayerNorm(model_cfg.emb_dim)
        self.output = nn.Linear(model_cfg.emb_dim, vocab_size, bias=False)

    def forward(self, x):
        batch_size, seq_length = x.shape
        token_emb = self.tok_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(seq_length, device=x.device))
        x = token_emb + pos_emb
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.output(x)