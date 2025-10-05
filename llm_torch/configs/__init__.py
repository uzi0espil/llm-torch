from llm_torch.configs.configs import (LLMConfig, TrainerConfig, DatasetConfig, SwiGLUBlockConfig,
                                       CallbackConfig, ModelConfig, MoEConfig, LayerNormConfig,
                                       RMSNormConfig, MultiHeadAttentionConfig, RoPEMultiHeadAttentionConfig,
                                       YarnGroupedAttentionConfig, RoPEGroupedAttentionConfig)
from llm_torch.configs.gpt import GPT2_CONFIG_124M
from llm_torch.configs.llama import (LLAMA2_CONFIG_7B, LLAMA3_CONFIG_8B, LLAMA31_CONFIG_8B,
                                     LLAMA32_CONFIG_1B, LLAMA32_CONFIG_3B)
from llm_torch.configs.qwen import QWEN3_CONFIG_30B


__all__ = [
    "get",
    "LLMConfig",
    "GPT2_CONFIG_124M",
    "LLAMA2_CONFIG_7B",
    "LLAMA3_CONFIG_8B",
    "LLAMA31_CONFIG_8B",
    "LLAMA32_CONFIG_3B",
    "QWEN3_CONFIG_30B",
    "SwiGLUBlockConfig",
    "TrainerConfig",
    "DatasetConfig",
    "CallbackConfig",
    "MoEConfig",
    "ModelConfig",
    "LayerNormConfig",
    "RMSNormConfig",
    "RoPEMultiHeadAttentionConfig",
    "MultiHeadAttentionConfig",
    "yarnGroupedAttentionConfig",
    "RoPEGroupedAttentionConfig",
]


def get(name, size):
    config_name = f"{name}_config_{size}".upper()
    if config_name not in globals():
        raise ValueError(f"LLM {name} of {size} is not supported, please provide your own config.")
    return globals()[config_name]