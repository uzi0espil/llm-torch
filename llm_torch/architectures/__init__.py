from llm_torch.architectures.gpt import GPT2
from llm_torch.architectures.llama import Llama2

__all__ = ["GPT2", "Llama2", "get"]


def get(name: str):

    if name.lower() == "gpt2":
        return GPT2
    elif name.lower() == "llama2":
        return Llama2
    else:
        raise ValueError(f"Unknown LLM architecture: {name}.")
