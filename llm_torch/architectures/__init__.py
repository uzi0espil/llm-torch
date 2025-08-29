from llm_torch.architectures.gpt import GPT2
from llm_torch.architectures.llama import Llama2, Llama3

__all__ = ["GPT2", "Llama2", "Llama3", "get"]


def get(name: str):

    if name.lower() == "gpt2":
        return GPT2
    elif name.lower() == "llama2":
        return Llama2
    elif name.lower() == "llama3":
        return Llama3
    else:
        raise ValueError(f"Unknown LLM architecture: {name}.")
