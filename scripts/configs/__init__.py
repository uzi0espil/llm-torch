from scripts.configs.gpt import GPT2_CONFIG_124M, GPT_OSS_CONFIG_20B
from scripts.configs.llama import (LLAMA2_CONFIG_7B, LLAMA3_CONFIG_8B, LLAMA31_CONFIG_8B,
                                   LLAMA32_CONFIG_3B)
from scripts.configs.qwen import QWEN3_CONFIG_30B


__all__ = [
    "get",
    "GPT2_CONFIG_124M",
    "GPT_OSS_CONFIG_20B",
    "LLAMA2_CONFIG_7B",
    "LLAMA3_CONFIG_8B",
    "LLAMA31_CONFIG_8B",
    "LLAMA32_CONFIG_3B",
    "QWEN3_CONFIG_30B",
]


def get(name, size):
    config_name = f"{name}_config_{size}".upper()
    if config_name not in globals():
        raise ValueError(f"LLM {name} of {size} is not supported, please provide your own config.")
    return globals()[config_name]