from dataclasses import dataclass, fields
from abc import abstractmethod, ABCMeta
from typing import Type
import torch

from llm_torch.components import feedforward_blocks as ff_blocks
from llm_torch.components import activations


@dataclass(kw_only=True)
class FFBaseConfig(metaclass=ABCMeta):
    hidden_dim: int
    activation: torch.nn.Module = activations.GELU

    @abstractmethod
    def block_class(self) -> ff_blocks.FFBaseBlock:
        raise NotImplementedError("Return the block class.")

    def instantiate(self, emb_dim: int, dtype) -> ff_blocks.FFBaseBlock:
        config = {f.name: getattr(self, f.name) for f in fields(self)}
        config.update(dict(emb_dim=emb_dim, dtype=dtype))
        return self.block_class(**config)


@dataclass
class FFBlockConfig(FFBaseConfig):

    @property
    def block_class(self):
        return ff_blocks.FFBlock


@dataclass
class SwiGLUBlockConfig(FFBaseConfig):
    activation: torch.nn.Module = activations.SiLU

    @property
    def block_class(self):
        return ff_blocks.SwiGLUBlock


@dataclass(kw_only=True)
class MoEConfig(FFBaseConfig):
    n_experts: int
    n_experts_per_token: int = 1
    ff_block: Type[ff_blocks.FFBaseBlock] = ff_blocks.SwiGLUBlock
    activation: torch.nn.Module = activations.SiLU

    @property
    def block_class(self):
        return ff_blocks.MoEBlock
