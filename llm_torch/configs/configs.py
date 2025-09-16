from __future__ import annotations

from abc import abstractmethod, ABCMeta
from typing import Optional
from dataclasses import dataclass, field, fields
from typing import Type, Dict, Any, Callable, List
import torch

from llm_torch.components.callbacks import Callback
from llm_torch.components.normalizer import Normalizer
from llm_torch.components.feedforward_blocks import FFBaseBlock, FFBlock, MoEBlock, SwiGLUBlock
from llm_torch.components.activations import GELU, SiLU


@dataclass
class DatasetConfig:
    batch_size: int = 4
    shuffle: bool = True
    max_length: int = 256
    stride: int = 1


@dataclass
class RoPEConfig:
    theta_base: float = 10_000


@dataclass
class YarnConfig:
    factor: float
    low_freq: float
    high_freq: float
    original_max_pos_embeddings: Optional[int] = None
    theta_base: float = 10_000


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
    n_heads: int
    ff_block_config: FFBlockConfig | SwiGLUBlockConfig | MoEConfig
    n_layers: int = 12
    drop_rate: Optional[float] = 0.1
    qkv_bias: Optional[bool] = False
    qk_norm: Optional[Type[Normalizer]]= None
    kv_window_size: Optional[int] = None
    n_kv_group: Optional[int] = None
    dtype: torch.dtype = torch.float32
    rope_scaling: Optional[RoPEConfig | YarnConfig] = None

    def __post_init__(self):
        if self.n_kv_group is None:
            self.n_kv_group = self.n_heads


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