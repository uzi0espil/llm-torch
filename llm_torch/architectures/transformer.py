import torch
from torch import nn
from typing import Optional

from llm_torch.configs import ModelConfig
from llm_torch.components.transformer_blocks import TransformerBlock


class Transformer(torch.nn.Module):

    def __init__(self, config: ModelConfig, vocab_size, context_length):
        super(Transformer, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.context_length = context_length

        # embedding
        self.current_pos = 0

        self.tok_embedding = nn.Embedding(vocab_size, config.emb_dim, dtype=config.dtype)
        self.pos_embedding = None
        if config.attention_config.is_rotary:
            self.pos_embedding = nn.Embedding(context_length, config.emb_dim, dtype=config.dtype)

        # blocks
        self.blocks = nn.ModuleList([TransformerBlock(
            config,
            context_length=context_length
        ) for _ in range(config.n_layers)])

        # dropout if given
        self.dropout = nn.Dropout(config.drop_rate) if config.drop_rate and config.drop_rate > 0.0 else None
        # final norm
        self.norm = config.normalizer_config.instantiate(config.emb_dim)
        # final layer
        self.output = nn.Linear(config.emb_dim, vocab_size, bias=False, dtype=config.dtype)
        # share output
        if config.tie_weight:
            self.tok_embedding.weight = self.output.weight

    def embed(self, x, use_cache=False):
        if self.pos_embedding is None:
            return self.tok_embedding(x)
        else:
            return self._pos_embed(x, use_cache=use_cache)

    def _pos_embed(self, x, use_cache=False):
        """Embed x with regard to positional encoding, no rotary is used."""
        _, seq_length = x.shape
        token_emb = self.tok_embedding(x)
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_length, device=x.device)
            self.current_pos += seq_length
        else:
            pos_ids = torch.arange(0, seq_length, device=x.device)
        pos_emb = self.pos_embedding(pos_ids).unsqueeze(0)
        return token_emb + pos_emb

    def forward(self, x, use_cache: bool = False):
        x = self.embed(x, use_cache=use_cache)
        if self.dropout is not None:
            x = self.dropout(x)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.norm(x)
        return self.output(x)

    @property
    def transformer_blocks(self) -> torch.nn.ModuleList:
        """Return the List of blocks"""
        return self.blocks

    def overall_state_dict(self, save_to: Optional[str] = None) -> dict:
        states = dict(
            model_state_dict=self.state_dict(),
            config=self.config,
            vocab_size=self.vocab_size,
            context_length=self.context_length,
        )
        if save_to is not None:
            torch.save(states, save_to)
        return states

    @classmethod
    def load(cls, path: str | dict = "model.pth"):
        checkpoint = path if isinstance(path, dict) else torch.load(path, weights_only=False)
        instance = cls(checkpoint["config"], checkpoint["vocab_size"], checkpoint["context_length"])
        instance.load_state_dict(checkpoint["model_state_dict"])
        return instance

    def reset_kv_cache(self):
        self.current_pos = 0
        for block in self.transformer_blocks:
            block.mha.reset_cache()
