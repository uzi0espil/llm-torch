from llm_torch.configs.configs import LLMConfig, TrainerConfig, DatasetConfig, CallbackConfig, ModelConfig
from llm_torch.configs.gpt import GPT2_CONFIG_124


__all__ = ["get", "LLMConfig", "GPT2_CONFIG_124", "TrainerConfig", "DatasetConfig",
           "CallbackConfig", "ModelConfig"]


def get(name, size):
    config_name = f"{name}_config_{size}".upper()
    if config_name not in globals():
        raise ValueError(f"LLM {name} of {size} is not supported, please provide your own config.")
    return globals()[config_name]