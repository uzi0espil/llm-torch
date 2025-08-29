from llm_torch.architectures.base import BaseLLMModel
from torch import nn

from llm_torch.components.attention import RoPEMHA, RoPEGOA
from llm_torch.components.normalizers import RMSNorm
from llm_torch.components.feedforward_blocks import SwiGLUBlock
from llm_torch.components.activations import SiLU
from llm_torch.components.transformer_blocks import TransformerBlock


class Llama2(BaseLLMModel):

    def __init__(self, model_cfg, vocab_size, context_length):
        super().__init__(model_cfg, vocab_size, context_length)

        self.tok_embedding = nn.Embedding(vocab_size, model_cfg.emb_dim, dtype=model_cfg.dtype)

        self.blocks = nn.ModuleList([TransformerBlock(model_cfg,
                                                      context_length=context_length,
                                                      attention=RoPEMHA,
                                                      norm=RMSNorm,
                                                      ff_block=SwiGLUBlock,
                                                      activation=SiLU) for _ in range(model_cfg.n_layers)])
        self.norm = RMSNorm(model_cfg.emb_dim)
        self.output = nn.Linear(model_cfg.emb_dim, vocab_size, bias=False, dtype=model_cfg.dtype)

    @property
    def transformer_blocks(self) -> nn.ModuleList:
        return self.blocks

    def forward(self, x, use_cache: bool = False):
        x = self.tok_embedding(x)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.norm(x)
        return self.output(x)


class Llama3(Llama2):

    def __init__(self, model_cfg, vocab_size, context_length):
        super().__init__(model_cfg, vocab_size, context_length)

        self.tok_embedding = nn.Embedding(vocab_size, model_cfg.emb_dim, dtype=model_cfg.dtype)

        self.blocks = nn.ModuleList([TransformerBlock(model_cfg,
                                                      context_length=context_length,
                                                      attention=RoPEGOA,
                                                      norm=RMSNorm,
                                                      ff_block=SwiGLUBlock,
                                                      activation=SiLU) for _ in range(model_cfg.n_layers)])
        self.norm = RMSNorm(model_cfg.emb_dim)
        self.output = nn.Linear(model_cfg.emb_dim, vocab_size, bias=False, dtype=model_cfg.dtype)
