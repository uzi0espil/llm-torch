from dataclasses import dataclass, fields
from abc import ABCMeta, abstractmethod
from typing import Optional

from llm_torch.components import attention
from llm_torch.configs.normalizer import NormalizerConfig


@dataclass(kw_only=True)
class AttentionConfig(metaclass=ABCMeta):
    n_heads: int
    dropout_rate: Optional[float] = 0.1
    mask: bool = True
    qkv_bias: bool = False
    qk_norm: Optional[NormalizerConfig] = None
    kv_window_size: Optional[int] = None

    def __post_init__(self):
        if self.qk_norm is not None:
            self.qk_norm = self._build_qk_norm_factory()

    @property
    @abstractmethod
    def attention_cls(self):
        raise NotImplementedError("Return the attention class.")

    def _build_qk_norm_factory(self):
        if self.qk_norm is None:
            return None

        def factory(emb_dim, *, _cfg=self.qk_norm):
            return _cfg.instantiate(emb_dim)

        return factory

    def instantiate(self, context_length: int, **overrides):
        config = {f.name: getattr(self, f.name) for f in fields(self)}
        config.update(overrides)
        config.setdefault("context_length", context_length)
        return self.attention_cls(**config)


@dataclass(kw_only=True)
class MultiHeadAttentionConfig(AttentionConfig):

    @property
    def attention_cls(self):
        return attention.MultiHeadAttention


@dataclass(kw_only=True)
class RoPEMultiHeadAttentionConfig(AttentionConfig):
    theta_base: float = 10_000.0

    @property
    def attention_cls(self):
        return attention.RoPEMHA


@dataclass(kw_only=True)
class RoPEGroupedAttentionConfig(AttentionConfig):
    n_kv_group: int
    theta_base: float = 10_000.0

    @property
    def attention_cls(self):
        return attention.RoPEGOA


@dataclass(kw_only=True)
class YarnGroupedAttentionConfig(AttentionConfig):
    n_kv_group: int
    factor: float
    low_freq: float
    high_freq: float
    original_max_pos_embeddings: Optional[int] = None
    theta_base: float = 10_000.0

    @property
    def attention_cls(self):
        return attention.YarnGOA
