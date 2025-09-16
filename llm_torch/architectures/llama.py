from abc import ABCMeta, abstractmethod

from llm_torch.architectures.base import BaseLLMModel
from torch import nn

from llm_torch.components.attention import RoPEMHA, RoPEGOA, YarnGOA
from llm_torch.components.normalizer import RMSNorm
from llm_torch.components.transformer_blocks import TransformerBlock


class BaseLlamaModel(BaseLLMModel, metaclass=ABCMeta):
    """The main difference between the Llama models are the attention class and configuration."""

    def __init__(self, model_cfg, vocab_size, context_length):
        super().__init__(model_cfg, vocab_size, context_length)

        self.tok_embedding = nn.Embedding(vocab_size, model_cfg.emb_dim, dtype=model_cfg.dtype)

        self.blocks = nn.ModuleList([TransformerBlock(model_cfg,
                                                      context_length=context_length,
                                                      attention=self.attention,
                                                      norm=RMSNorm) for _ in range(model_cfg.n_layers)])
        self.norm = RMSNorm(model_cfg.emb_dim)
        self.output = nn.Linear(model_cfg.emb_dim, vocab_size, bias=False, dtype=model_cfg.dtype)

    @property
    @abstractmethod
    def attention(self):
        raise NotImplemented

    @property
    def transformer_blocks(self) -> nn.ModuleList:
        return self.blocks

    def forward(self, x, use_cache: bool = False):
        x = self.tok_embedding(x)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.norm(x)
        return self.output(x)


class Llama2(BaseLlamaModel):

    @property
    def attention(self):
        return RoPEMHA


class Llama3(BaseLlamaModel):

    @property
    def attention(self):
        return RoPEGOA


class Llama31(BaseLlamaModel):

    @property
    def attention(self):
        return YarnGOA


class Llama32(Llama31):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # tying the embedding and output weight.
        self.tok_embedding.weight = self.output.weight

