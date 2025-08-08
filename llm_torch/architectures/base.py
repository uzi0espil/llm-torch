from abc import ABCMeta
import torch
from typing import Optional

from llm_torch.configs import ModelConfig


class BaseLLMModel(torch.nn.Module, metaclass=ABCMeta):


    def __init__(self, config: ModelConfig, vocab_size, context_length):
        super(BaseLLMModel, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.context_length = context_length

    def forward(self, x):
        raise NotImplementedError

    def overall_state_dict(self, save_to: Optional[str] = None) -> dict:
        states = dict(
            model_state_dict=self.state_dict(),
            config=self.config,
            vocab_size=self.vocab_size,
            context_length=self.context_length,
        )
        if save_to is not None:
            torch.save(states, save_to)
        return states

    @classmethod
    def load(cls, path: str | dict = "model.pth"):
        checkpoint = path if isinstance(path, dict) else torch.load(path, weights_only=False)
        instance = cls(checkpoint["config"], checkpoint["vocab_size"], checkpoint["context_length"])
        instance.load_state_dict(checkpoint["model_state_dict"])
        return instance