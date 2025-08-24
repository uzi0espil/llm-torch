from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
from typing import Type, Dict, Any, Callable, List
import torch

from llm_torch.components.callbacks import Callback


@dataclass
class DatasetConfig:
    batch_size: int = 4
    shuffle: bool = True
    max_length: int = 256
    stride: int = 1


@dataclass
class ModelConfig:
    emb_dim: int
    n_heads: int
    hidden_dim: int
    n_layers: int = 12
    drop_rate: Optional[float] = 0.1
    qkv_bias: Optional[bool] = False
    kv_window_size: Optional[int] = None
    dtype: torch.dtype = torch.float32


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