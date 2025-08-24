import torch
from torch import nn
from typing import Optional

from llm_torch.utils.core import make_get_function


class RoPEMixin(object):
    """Requires the consumer (e.g., MultiHeadAttention) to define:
      - self.head_dim (even)
      - self.context_length (max sequence length used for precomputation)"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not issubclass(cls, BaseAttention):
            raise TypeError(f"{cls.__name__} must inherit from BaseAttention.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self, "head_dim"), "RoPEMixin expects self.head_dim to be set before super().__init__()"
        assert hasattr(self, "context_length"), "RoPEMixin expects self.context_length to be set before super().__init__()"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        cos, sin = self.precompute_rope_params()
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def precompute_rope_params(self, theta_base: float = 10_000.0):
        half_dim = self.head_dim // 2
        freqs = 1.0 / (theta_base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))

        positions = torch.arange(self.context_length, dtype=torch.float32)  # shape: (context_length,)

        angles = positions[:, None] * freqs[None, :]  # shape: (context_length, half_dim)
        return torch.cos(angles), torch.sin(angles)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to (b, h, seq, d). Uses even/odd interleaving, not first-half/second-half split."""
        b, h, seq, d = x.shape
        assert d == self.head_dim
        # (1,1,seq,half_dim)
        cos = self.cos[:seq, :].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
        sin = self.sin[:seq, :].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)

        # split into even/odd channels
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # rotate
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        # interleave back
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_rot_even
        x_out[..., 1::2] = x_rot_odd
        return x_out.to(dtype=x.dtype)

    # hooks used by attention
    def transform_keys(self, keys: torch.Tensor) -> torch.Tensor:
        return self._apply_rope(keys)

    def transform_queries(self, queries: torch.Tensor) -> torch.Tensor:
        return self._apply_rope(queries)


class CacheMixin(object):
    """Mixin class the implements caching KV and retrieving from cache when needed."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not issubclass(cls, BaseAttention):
            raise TypeError(f"{cls.__name__} must inherit from BaseAttention.")

    def __init__(self, *args, kv_window_size: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_window_size = kv_window_size or self.context_length
        self.kv_window_size = self.context_length if self.kv_window_size > self.context_length else self.kv_window_size
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.current_pos = 0

    def persist_kv(self, k, v, num_tokens, batch_size):
        # If the incoming sequence is larger than the cache, only cache the most recent `kv_window_size` tokens.
        if num_tokens > self.kv_window_size:
            k = k[:, :, -self.kv_window_size:, :]
            v = v[:, :, -self.kv_window_size:, :]
            num_tokens = self.kv_window_size

        if self.cache_k is None or self.cache_k.size(0) != batch_size:
            self.cache_k = torch.zeros(batch_size, self.n_heads,
                                       self.kv_window_size, self.head_dim,
                                       device=k.device, dtype=k.dtype)
            self.cache_v = torch.zeros_like(self.cache_k)
            self.current_pos = 0  # pointer to next free slot

        # When the cache overflows, shift the existing cache to the left to make room for new tokens
        if self.current_pos + num_tokens > self.kv_window_size:
            overflow = self.current_pos + num_tokens - self.kv_window_size
            # shift everything left by `overflow` (cheap view-copy)
            self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
            self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()
            self.current_pos -= overflow  # pointer after shift

        self.cache_k[:, :, self.current_pos:self.current_pos + num_tokens, :] = k
        self.cache_v[:, :, self.current_pos:self.current_pos + num_tokens, :] = v
        self.current_pos += num_tokens

        keys = self.cache_k[:, :, :self.current_pos, :]
        values = self.cache_v[:, :, :self.current_pos, :]
        return keys, values

    def get_mask_pool(self, num_tokens, num_keys):
        if num_tokens == num_keys:  # no cache.
            return super().get_mask_pool(num_tokens, num_keys)

        # need to offset.
        offset = num_keys - num_tokens
        row_idx = torch.arange(num_tokens).unsqueeze(1)  # (num_tokens, 1)
        col_idx = torch.arange(num_keys).unsqueeze(0)
        return row_idx + offset < col_idx  # True where j > i + offset

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None


class BaseAttention(nn.Module):

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 context_length: int,
                 dropout_rate: Optional[float] = 0.1,
                 n_heads: int = 8,
                 mask: bool = True,
                 qkv_bias: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu'):
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.d_out = d_out
        self.d_in = d_in
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.context_length = context_length
        self.to_mask = mask

        # defer super call until attributes are assigned.
        super().__init__()

        # linear projections
        self.W_q = nn.Linear(d_in, d_out, dtype=dtype, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, dtype=dtype, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, dtype=dtype, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None

    # overridable hooks
    def transform_keys(self, keys: torch.Tensor) -> torch.Tensor:
        return keys

    def transform_queries(self, queries: torch.Tensor) -> torch.Tensor:
        return queries

    def get_mask_pool(self, num_tokens, num_keys):
        return torch.triu(torch.ones(num_tokens, num_keys, dtype=torch.bool, device=self.W_q.weight.device), diagonal=1)

    def persist_kv(self, k, v, num_tokens, batch_size):
        return k, v

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        b, num_tokens, _ = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # reshape to (b, n_heads, seq, head_dim)
        q = q.view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        # optional transforms, such RoPE, etc...
        k = self.transform_keys(k)
        q = self.transform_queries(q)

        if use_cache:
            k, v = self.persist_kv(k, v, num_tokens=num_tokens, batch_size=b)

        # attention scores (b, h, seq, seq)
        attn_scores = q @ k.transpose(2, 3)

        if self.to_mask:
            mask_pool = self.get_mask_pool(num_tokens, attn_scores.size(-1))
            attn_scores.masked_fill_(mask_pool, float("-inf"))

        scale = (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        if self.dropout is not None:  # some architectures don't use dropout
            attn_weights = self.dropout(attn_weights)

        context = attn_weights @ v  # (b, h, seq, head_dim)
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)


class MultiHeadAttention(CacheMixin, BaseAttention):
    pass


class RoPEMHA(RoPEMixin, MultiHeadAttention):
    pass


get = make_get_function(globals())


if __name__ == '__main__':

    mha = MultiHeadAttention(
        d_in=32,
        d_out=32,
        context_length=512,
        dropout_rate=0.1,
        n_heads=2,
        mask=True,
        qkv_bias=False,
        dtype=torch.float32,
        kv_window_size=128,
    )

    x = torch.randn(4, 512, 32, requires_grad=False)

    out = mha(x, use_cache=True)
    print(out.shape)
