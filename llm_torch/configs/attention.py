from dataclasses import dataclass, fields
from abc import ABCMeta, abstractmethod
from typing import Optional

from llm_torch.components import attention
from llm_torch.configs.normalizer import NormalizerConfig


@dataclass(kw_only=True)
class MaskMixinConfig:
    mask: bool = True


@dataclass(kw_only=True)
class AttentionConfig(metaclass=ABCMeta):
    n_heads: int
    dropout_rate: Optional[float] = 0.1
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

    @property
    @abstractmethod
    def is_rotary(self) -> bool:
        raise NotImplementedError("Return whether rotary position encoding is used.")

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
class MultiHeadAttentionConfig(MaskMixinConfig, AttentionConfig):

    @property
    def attention_cls(self):
        return attention.MultiHeadAttention

    @property
    def is_rotary(self) -> bool:
        return False


@dataclass(kw_only=True)
class RoPEMultiHeadAttentionConfig(MaskMixinConfig, AttentionConfig):
    theta_base: float = 10_000.0

    @property
    def attention_cls(self):
        return attention.RoPEMHA

    @property
    def is_rotary(self) -> bool:
        return True


@dataclass(kw_only=True)
class GroupedKeyAttention(MaskMixinConfig, AttentionConfig):
    n_kv_group: int

    @property
    def attention_cls(self):
        return attention.GroupedKeyAttention

    @property
    def is_rotary(self) -> bool:
        return False


@dataclass(kw_only=True)
class SlidingWindowAttentionConfig(AttentionConfig):
    n_kv_group: int
    window_size: int

    @property
    def attention_cls(self):
        return attention.SlidingWindowAttention

    @property
    def is_rotary(self) -> bool:
        return False


@dataclass(kw_only=True)
class NaiveSWAConfig(AttentionConfig):
    n_kv_group: int
    window_size: int

    @property
    def attention_cls(self):
        return attention.NaiveSWA

    @property
    def is_rotary(self) -> bool:
        return False


@dataclass(kw_only=True)
class RoPEGroupedAttentionConfig(MaskMixinConfig, AttentionConfig):
    n_kv_group: int
    theta_base: float = 10_000.0

    @property
    def attention_cls(self):
        return attention.RoPEGOA

    @property
    def is_rotary(self) -> bool:
        return True


@dataclass(kw_only=True)
class YarnGroupedAttentionConfig(MaskMixinConfig, AttentionConfig):
    n_kv_group: int
    factor: float
    low_freq: float
    high_freq: float
    original_max_pos_embeddings: Optional[int] = None
    theta_base: float = 10_000.0

    @property
    def attention_cls(self):
        return attention.YarnGOA

    @property
    def is_rotary(self) -> bool:
        return True


@dataclass(kw_only=True)
class YarnSWAConfig(AttentionConfig):
    n_kv_group: int
    window_size: int
    factor: float
    low_freq: float
    high_freq: float
    original_max_pos_embeddings: Optional[int] = None
    theta_base: float = 10_000.0

    @property
    def attention_cls(self):
        return attention.YarnSWA

    @property
    def is_rotary(self) -> bool:
        return True
