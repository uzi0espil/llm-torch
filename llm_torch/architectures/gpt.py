import torch

from llm_torch.architectures.base import BaseLLMModel
from llm_torch.configs import ModelConfig


class GPT2(BaseLLMModel):

    def __init__(self, model_cfg: ModelConfig, vocab_size, context_length):
        super().__init__(model_cfg, vocab_size, context_length)
        self.pos_embedding = torch.nn.Embedding(context_length, model_cfg.emb_dim, dtype=model_cfg.dtype)
        self.current_position = 0

    def embed(self, x, use_cache=False):
        _, seq_length = x.shape
        token_emb = self.tok_embedding(x)
        if use_cache:
            pos_ids = torch.arange(self.current_position, self.current_position + seq_length, device=x.device)
            self.current_position += seq_length
        else:
            pos_ids = torch.arange(0, seq_length, device=x.device)
        pos_emb = self.pos_embedding(pos_ids).unsqueeze(0)
        return token_emb + pos_emb

    def reset_kv_cache(self):
        self.current_position = 0
        super().reset_kv_cache()
