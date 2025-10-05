from llm_torch.architectures.base import BaseLLMModel
import torch
from torch import nn

from llm_torch.components.transformer_blocks import TransformerBlock
from llm_torch.components.normalizer import RMSNorm


class Qwen3(BaseLLMModel):

    def __init__(self, model_cfg, vocab_size, context_length):
        super().__init__(model_cfg, vocab_size, context_length)
        self.tok_embedding = nn.Embedding(vocab_size, model_cfg.emb_dim, dtype=model_cfg.dtype)

        self.blocks = nn.ModuleList([TransformerBlock(
            model_cfg,
            context_length=context_length
        ) for _ in range(model_cfg.n_layers)])

        # the Norm mean and std are computed with float32 before projected back to original dtype.
        self.norm = model_cfg.normalizer_config.instantiate(model_cfg.emb_dim)
        self.output = nn.Linear(model_cfg.emb_dim, vocab_size, bias=False, dtype=model_cfg.dtype)

    @property
    def transformer_blocks(self) -> torch.nn.ModuleList:
        return self.blocks

    def forward(self, x, use_cache: bool = False):
        x = self.tok_embedding(x)
        for block in self.transformer_blocks:
            x = block(x, use_cache=use_cache)
        x = self.norm(x)
        return self.output(x)
