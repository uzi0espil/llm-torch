from llm_torch.configs.configs import (LLMConfig, TrainerConfig, DatasetConfig, RoPEConfig, SwiGLUBlockConfig,
                                       CallbackConfig, ModelConfig, YarnConfig, MoEConfig, LayerNormConfig,
                                       RMSNormConfig)
from llm_torch.configs.gpt import GPT2_CONFIG_124
from llm_torch.configs.llama import LLAMA2_CONFIG_7B, LLAMA3_CONFIG_8B, LLAMA31_CONFIG_8B, LLAMA32_CONFIG_1B
from llm_torch.configs.qwen import QWEN3_CONFIG_30B


__all__ = [
    "get",
    "LLMConfig",
    "GPT2_CONFIG_124",
    "LLAMA2_CONFIG_7B",
    "LLAMA3_CONFIG_8B",
    "LLAMA31_CONFIG_8B",
    "QWEN3_CONFIG_30B",
    "RoPEConfig",
    "SwiGLUBlockConfig",
    "TrainerConfig",
    "DatasetConfig",
    "CallbackConfig",
    "MoEConfig",
    "ModelConfig",
    "YarnConfig",
    "LayerNormConfig",
    "RMSNormConfig",]


def get(name, size):
    config_name = f"{name}_config_{size}".upper()
    if config_name not in globals():
        raise ValueError(f"LLM {name} of {size} is not supported, please provide your own config.")
    return globals()[config_name]