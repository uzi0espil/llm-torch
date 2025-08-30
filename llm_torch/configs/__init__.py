from llm_torch.configs.configs import (LLMConfig, TrainerConfig, DatasetConfig,
                                       CallbackConfig, ModelConfig, YarnConfig)
from llm_torch.configs.gpt import GPT2_CONFIG_124
from llm_torch.configs.llama import LLAMA2_CONFIG_7B, LLAMA3_CONFIG_8B, LLAMA31_CONFIG_8B


__all__ = ["get", "LLMConfig", "GPT2_CONFIG_124", "LLAMA2_CONFIG_7B", "LLAMA3_CONFIG_8B", "LLAMA31_CONFIG_8B",
           "TrainerConfig", "DatasetConfig", "CallbackConfig", "ModelConfig", "YarnConfig"]


def get(name, size):
    config_name = f"{name}_config_{size}".upper()
    if config_name not in globals():
        raise ValueError(f"LLM {name} of {size} is not supported, please provide your own config.")
    return globals()[config_name]