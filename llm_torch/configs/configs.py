from __future__ import annotations

from abc import abstractmethod, ABCMeta
from typing import Optional
from dataclasses import dataclass, field, fields
from typing import Type, Dict, Any, Callable, List
import torch

from llm_torch.components.callbacks import Callback
from llm_torch.components.normalizer import Normalizer, LayerNorm, RMSNorm
from llm_torch.components.feedforward_blocks import FFBaseBlock, FFBlock, MoEBlock, SwiGLUBlock
from llm_torch.components.attention import MultiHeadAttention, RoPEGOA, RoPEMHA, YarnGOA
from llm_torch.components.activations import GELU, SiLU


@dataclass
class DatasetConfig:
    batch_size: int = 4
    shuffle: bool = True
    max_length: int = 256
    stride: int = 1

# Attention Dataclasses

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
        return MultiHeadAttention


@dataclass(kw_only=True)
class RoPEMultiHeadAttentionConfig(AttentionConfig):
    theta_base: float = 10_000.0

    @property
    def attention_cls(self):
        return RoPEMHA


@dataclass(kw_only=True)
class RoPEGroupedAttentionConfig(AttentionConfig):
    n_kv_group: int
    theta_base: float = 10_000.0

    @property
    def attention_cls(self):
        return RoPEGOA


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
        return YarnGOA

# Normalizers

@dataclass(kw_only=True)
class NormalizerConfig(metaclass=ABCMeta):
    eps: float = 1e-6

    @property
    @abstractmethod
    def normalizer_cls(self) -> Type[Normalizer]:
        raise NotImplementedError

    def instantiate(self, emb_dim: int, **overrides) -> Normalizer:
        config = {f.name: getattr(self, f.name) for f in fields(self)}
        config.update(**overrides)
        config.setdefault("emb_dim", emb_dim)
        return self.normalizer_cls(**config)


@dataclass(kw_only=True)
class LayerNormConfig(NormalizerConfig):

    @property
    def normalizer_cls(self) -> Type[Normalizer]:
        return LayerNorm


@dataclass(kw_only=True)
class RMSNormConfig(NormalizerConfig):
    dtype: Optional[torch.dtype] = None

    @property
    def normalizer_cls(self) -> Type[Normalizer]:
        return RMSNorm

# FeedForward Block Configurations

@dataclass(kw_only=True)
class FFBaseConfig(metaclass=ABCMeta):
    hidden_dim: int
    activation: torch.nn.Module = GELU

    @abstractmethod
    def block_class(self) -> FFBaseBlock:
        raise NotImplementedError("Return the block class.")

    def instantiate(self, emb_dim: int, dtype) -> FFBaseBlock:
        config = {f.name: getattr(self, f.name) for f in fields(self)}
        config.update(dict(emb_dim=emb_dim, dtype=dtype))
        return self.block_class(**config)


@dataclass
class FFBlockConfig(FFBaseConfig):

    @property
    def block_class(self):
        return FFBlock


@dataclass
class SwiGLUBlockConfig(FFBaseConfig):
    activation: torch.nn.Module = SiLU

    @property
    def block_class(self):
        return SwiGLUBlock


@dataclass(kw_only=True)
class MoEConfig(FFBaseConfig):
    n_experts: int
    n_experts_per_token: int = 1
    ff_block: Type[FFBaseBlock] = SwiGLUBlock
    activation: torch.nn.Module = SiLU

    @property
    def block_class(self):
        return MoEBlock


@dataclass
class ModelConfig:
    emb_dim: int
    ff_block_config: FFBlockConfig | SwiGLUBlockConfig | MoEConfig
    attention_config: AttentionConfig
    normalizer_config: NormalizerConfig
    n_layers: int = 12
    drop_rate: Optional[float] = 0.1
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.attention_config is None:
            raise ValueError("ModelConfig requires an attention configuration.")
        if self.normalizer_config is None:
            raise ValueError("ModelConfig requires a normalizer configuration.")
        if self.attention_config.dropout_rate is None:
            self.attention_config.dropout_rate = self.drop_rate
        if hasattr(self.normalizer_config, "dtype") and getattr(self.normalizer_config, "dtype", None) is None:
            self.normalizer_config.dtype = self.dtype
        qk_norm_cfg = getattr(self.attention_config, "qk_norm", None)
        if qk_norm_cfg is not None and hasattr(qk_norm_cfg, "dtype") and getattr(qk_norm_cfg, "dtype", None) is None:
            qk_norm_cfg.dtype = self.dtype


@dataclass
class OptimizerConfig:
    class_name: Type[torch.optim.Optimizer]
    config: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self, params):
        """Create the optimizer instance with provided model parameters."""
        return self.class_name(params, **self.config)


@dataclass
class CallbackConfig:
    class_name: Type[Callback]
    config: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self):
        return self.class_name(**self.config)


@dataclass
class TrainerConfig:
    epochs: int
    eval_freq: int = 5
    eval_iter: int = 5
    max_grad_norm: Optional[float] = 1.0
    optimizer: Optional[OptimizerConfig] = None
    loss: Optional[Callable] = None

    @staticmethod
    def _default_loss(outputs, targets):
        """Default cross-entropy loss function."""
        return torch.nn.functional.cross_entropy(outputs.flatten(0, 1), targets.flatten())

    @staticmethod
    def _default_optimizer_config():
        """Returns a default AdamW optimizer configuration."""
        return OptimizerConfig(class_name=torch.optim.AdamW, config=dict(lr=1e-4))

    def __post_init__(self):
        """Set default optimizer and loss if not provided."""
        if self.optimizer is None:
            self.optimizer = self._default_optimizer_config()
        if self.loss is None:
            self.loss = self._default_loss


@dataclass
class LLMConfig:
    vocab_size: int
    context_length: int
    model_config: ModelConfig
    dataset_config: DatasetConfig
    train_config: TrainerConfig
    callback_configs: Optional[List[CallbackConfig]] = None