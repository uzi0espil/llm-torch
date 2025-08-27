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

    def __init__(self, theta_base: float = 10_000.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self, "head_dim"), "RoPEMixin expects self.head_dim to be set before super().__init__()"
        assert hasattr(self, "context_length"), "RoPEMixin expects self.context_length to be set before super().__init__()"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.theta_base = theta_base

        half_dim = self.head_dim // 2
        self.register_buffer(
            "freqs",
            1.0 / (theta_base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)),
            persistent=False,
        )

    def _apply_rope(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        """Apply RoPE to (b, h, seq, d). Uses even/odd interleaving, not first-half/second-half split."""
        b, h, seq, d = x.shape
        half = d // 2
        pos = torch.arange(start_index, start_index + seq, device=x.device, dtype=torch.float32)
        angles = pos[:, None] * self.freqs[None, :]
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # (1,1,seq,half)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

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
    def transform_keys(self, keys: torch.Tensor, start_index: int = 0) -> torch.Tensor:
        return self._apply_rope(keys, start_index)

    def transform_queries(self, queries: torch.Tensor, start_index: int = 0) -> torch.Tensor:
        return self._apply_rope(queries, start_index)


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
                 n_kv_group: int = 8,
                 mask: bool = True,
                 qkv_bias: bool = False,
                 dtype: torch.dtype = torch.float32):
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        assert n_kv_group <= n_heads, "kv_group should be <= than n_heads"
        assert n_heads % n_kv_group == 0, "n_heads must be divisible by n_kv_group (for grouping)"

        self.d_out = d_out
        self.d_in = d_in
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.context_length = context_length
        self.to_mask = mask
        self.n_kv_group = n_kv_group

        # defer super call until attributes are assigned.
        super().__init__()

        # linear projections
        self.W_q = nn.Linear(d_in, d_out, dtype=dtype, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, n_kv_group * self.head_dim, dtype=dtype, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, n_kv_group * self.head_dim, dtype=dtype, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None
        self.group_size_ = n_heads // n_kv_group

    # overridable hooks
    def transform_keys(self, keys: torch.Tensor, start_index: int = 0) -> torch.Tensor:
        return keys

    def transform_queries(self, queries: torch.Tensor, start_index: int = 0) -> torch.Tensor:
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
        k = k.view(b, num_tokens, self.n_kv_group, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.n_kv_group, self.head_dim).transpose(1, 2)

        # optional transforms, such RoPE, etc...
        start_k = self.current_pos if (use_cache and hasattr(self, "current_pos")) else 0
        start_q = max(0, start_k + k.size(-2) - q.size(-2))

        k = self.transform_keys(k, start_index=start_k)
        q = self.transform_queries(q, start_index=start_q)

        if use_cache:
            k, v = self.persist_kv(k, v, num_tokens=num_tokens, batch_size=b)

        if self.group_size_ > 1:
            k = k.repeat_interleave(self.group_size_, dim=1)
            v = v.repeat_interleave(self.group_size_, dim=1)

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
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 context_length: int,
                 dropout_rate: Optional[float] = 0.1,
                 n_heads: int = 8,
                 mask: bool = True,
                 qkv_bias: bool = False,
                 dtype: torch.dtype = torch.float32,
                 # Mixin init variables
                 kv_window_size: Optional[int] = None):
        """Standard MHA: set n_kv_group == n_heads (i.e., no grouping)."""
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout_rate=dropout_rate,
            n_heads=n_heads,
            n_kv_group=n_heads,
            mask=mask,
            qkv_bias=qkv_bias,
            dtype=dtype,
            kv_window_size=kv_window_size
        )


class GroupedKeyAttention(CacheMixin, BaseAttention):
    pass


class RoPEMHA(RoPEMixin, MultiHeadAttention):
    pass


class RoPEGOA(RoPEMHA, GroupedKeyAttention):
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
