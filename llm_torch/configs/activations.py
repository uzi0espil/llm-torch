from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, fields
from typing import Type

from llm_torch.components.activations import GELU, SiLU, BaseActivation


@dataclass(kw_only=True)
class ActivationConfig(metaclass=ABCMeta):

    @property
    @abstractmethod
    def activation_cls(self) -> Type[BaseActivation]:
        raise NotImplementedError("Return the activation class.")

    def instantiate(self, **overrides) -> BaseActivation:
        config = {f.name: getattr(self, f.name) for f in fields(self)}
        config.update(overrides)
        return self.activation_cls(**config)


@dataclass(kw_only=True)
class GELUConfig(ActivationConfig):

    @property
    def activation_cls(self) -> Type[BaseActivation]:
        return GELU


@dataclass(kw_only=True)
class SiLUConfig(ActivationConfig):
    alpha: float = 1.0

    @property
    def activation_cls(self) -> Type[BaseActivation]:
        return SiLU