from dataclasses import dataclass, fields
from abc import abstractmethod, ABCMeta
from typing import Type, Optional
import torch

from llm_torch.components import normalizer


@dataclass(kw_only=True)
class NormalizerConfig(metaclass=ABCMeta):
    eps: float = 1e-6

    @property
    @abstractmethod
    def normalizer_cls(self) -> Type[normalizer.Normalizer]:
        raise NotImplementedError

    def instantiate(self, emb_dim: int, **overrides) -> normalizer.Normalizer:
        config = {f.name: getattr(self, f.name) for f in fields(self)}
        config.update(**overrides)
        config.setdefault("emb_dim", emb_dim)
        return self.normalizer_cls(**config)


@dataclass(kw_only=True)
class LayerNormConfig(NormalizerConfig):

    @property
    def normalizer_cls(self) -> Type[normalizer.Normalizer]:
        return normalizer.LayerNorm


@dataclass(kw_only=True)
class RMSNormConfig(NormalizerConfig):
    dtype: Optional[torch.dtype] = None

    @property
    def normalizer_cls(self) -> Type[normalizer.Normalizer]:
        return normalizer.RMSNorm
