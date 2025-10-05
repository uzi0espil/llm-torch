from abc import ABCMeta, abstractmethod

from llm_torch.architectures.base import BaseLLMModel


class Llama2(BaseLLMModel):
    pass


class Llama3(BaseLLMModel):
    pass


class Llama31(BaseLLMModel):
    pass


class Llama32(BaseLLMModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # tying the embedding and output weight.
        self.tok_embedding.weight = self.output.weight
