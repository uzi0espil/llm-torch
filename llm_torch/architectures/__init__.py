from llm_torch.architectures.gpt import GPT2
from llm_torch.architectures.llama import Llama2, Llama3, Llama31

__all__ = ["GPT2", "Llama2", "Llama3", "Llama31", "get"]


def get(name: str):

    if name.lower() == "gpt2":
        return GPT2
    elif name.lower() == "llama2":
        return Llama2
    elif name.lower() == "llama3":
        return Llama3
    elif name.lower() == "llama31":
        return Llama31
    else:
        raise ValueError(f"Unknown LLM architecture: {name}.")
