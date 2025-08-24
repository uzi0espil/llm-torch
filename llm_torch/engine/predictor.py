import torch
import logging
from typing import List, Optional


logger = logging.getLogger(__name__)


class Predictor(object):

    def __init__(self,
                 model,
                 tokenizer,
                 eos_id=None,
                 pad_id=0,
                 allowed_special: Optional[set] = None,
                 use_cache: bool = True,
                 device='cpu'):
        self.model = model
        self.model.eval()  # always in eval state.
        self.context_length = self.model.context_length
        self.pad_id = pad_id
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.allowed_special = allowed_special or set()
        self.device = device
        self.use_cache = use_cache

        if use_cache and not hasattr(self.model, 'reset_kv_cache'):
            logger.warning(f"Model {self.model.__class__.__name__} has no reset_kv_cache method. Cache is disabled.")
            self.use_cache = False

    def encode(self, text: List[str] | str):
        if isinstance(text, str):
            return torch.tensor([self.tokenizer.encode(text, allowed_special=self.allowed_special)], dtype=torch.long)

        ids = [self.tokenizer.encode(t, allowed_special=self.allowed_special) for t in text]
        maxlen = max(len(x) for x in ids)  # pad only based on the longest item in the batch.
        padded = [x + [self.pad_id] * (maxlen - len(x)) for x in ids]
        return torch.tensor(padded, dtype=torch.long)

    def generate_text(self, ids, max_new_tokens: int, temperature: float = 1., top_k: int = None):
        # todo: make it keep predicting until eos is predicted, if max_new_tokens are None.
        if temperature <= 0:
            raise ValueError('The `temperature` should be a positive number.')

        if self.eos_id is None and max_new_tokens is None:
            raise ValueError('please set either `max_new_tokens` or eos_id.')

        if self.use_cache:
            self.model.reset_kv_cache()

        idx_next = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # if `use_cache` is enabled, then pass only the next id and not all history.
                x = idx_next if self.use_cache and idx_next is not None else ids[:, -self.context_length:]
                logits = self.model(x, use_cache=self.use_cache)
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

                # todo: instead of exiting the complete batch if any of the items in the batch has EOS, exit when all.
                if self.eos_id is not None and (idx_next == self.eos_id).any().item():
                    logger.info("Model predicted End-Of-End token.")
                    break

                ids = torch.cat((ids, idx_next), dim=1)

        return ids

    def decode(self, ids_list: List[torch.LongTensor]):
        decoded_texts = []
        for ids in ids_list:
            try:
                start_index = (ids != self.pad_id).nonzero(as_tuple=True)[0][0]
            except IndexError:
                start_index = len(ids)
            trimmed_ids = ids[start_index:]
            decoded_texts.append(self.tokenizer.decode(trimmed_ids.tolist()))
        return decoded_texts

    def predict(self, text: List[str] | str, max_new_tokens, temperature=1., top_k=None):
        ids = self.encode(text).to(self.device)
        token_ids = self.generate_text(ids, max_new_tokens, temperature, top_k)
        decoded_text = self.decode(token_ids)
        return decoded_text
