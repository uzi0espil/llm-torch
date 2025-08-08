import torch
import logging
from typing import List, Optional


logger = logging.getLogger(__name__)


class Predictor(object):

    def __init__(self, model, tokenizer, eos_id=None, pad_id=0, allowed_special: Optional[set] = None, device='cpu'):
        self.model = model
        self.model.eval()  # always in eval state.
        self.context_size = self.model.pos_embedding.weight.shape[0]
        self.pad_id = pad_id
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.allowed_special = allowed_special or set()
        self.device = device

    def encode(self, text: List[str] | str):
        text = [text] if isinstance(text, str) else text
        encoded_text = [self.tokenizer.encode(t, allowed_special=self.allowed_special) for t in text]

        padded = [
            seq[:self.context_size] + [self.pad_id] * (self.context_size - len(seq)) for seq in encoded_text
        ]

        return torch.tensor(padded)

    def generate_text(self, ids, max_new_tokens, temperature=1., top_k=None):
        # todo: make it keep predicting until eos is predicted, if max_new_tokens are None.
        if temperature <= 0:
            raise ValueError('The `temperature` should be a positive number.')

        if self.eos_id is None and max_new_tokens is None:
            raise ValueError('please set either `max_new_tokens` or eos_id.')

        for _ in range(max_new_tokens):
            ids = ids[:, -self.context_size:]

            with torch.no_grad():
                logits = self.model(ids)

            logits = logits[:, -1, :]

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                top_values, _ = torch.topk(logits, k=k, dim=-1)  # already returned ordered
                min_val = top_values[:, -1].unsqueeze(-1)  # take the minimum value for each item in batch
                logits = torch.where(logits < min_val,
                                     input=torch.tensor(float('-inf')).to(self.device),
                                     other=logits)

            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == self.eos_id:
                break

            ids = torch.cat((ids, idx_next), dim=1)

        return ids

    def decode(self, ids_list: List[torch.LongTensor]):
        return [self.tokenizer.decode(ids.tolist()) for ids in ids_list]

    def predict(self, text: List[str] | str, max_new_tokens, temperature=1., top_k=None):
        ids = self.encode(text).to(self.device)
        token_ids = self.generate_text(ids, max_new_tokens, temperature, top_k)
        decoded_text = self.decode(token_ids)
        return decoded_text