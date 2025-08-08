from llm_torch.architectures.gpt import GPT2

__all__ = ["GPT2", "get"]


def get(name: str):

    if name.lower() == "gpt2":
        return GPT2
    else:
        raise ValueError(f"Unknown LLM architecture: {name}.")
