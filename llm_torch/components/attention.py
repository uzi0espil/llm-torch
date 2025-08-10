import torch
from torch import nn
from typing import Optional

from llm_torch.utils.core import make_get_function


class RoPEMixin(object):
    """Requires the consumer (e.g., MultiHeadAttention) to define:
      - self.head_dim (even)
      - self.context_length (max sequence length used for precomputation)"""

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


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 context_length: int,
                 dropout_rate: Optional[float] = 0.1,
                 n_heads: int = 8,
                 mask: bool = True,
                 qkv_bias: bool = False,
                 dtype: torch.dtype = torch.float32,):
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

        if self.to_mask:
            mask_buf = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            self.register_buffer("mask", mask_buf)

    # overridable hooks
    def transform_keys(self, keys: torch.Tensor) -> torch.Tensor:
        return keys

    def transform_queries(self, queries: torch.Tensor) -> torch.Tensor:
        return queries

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # attention scores (b, h, seq, seq)
        attn_scores = q @ k.transpose(2, 3)

        if self.to_mask:
            mask_pool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores = attn_scores.masked_fill(mask_pool, float("-inf"))

        scale = (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        if self.dropout is not None:  # some architectures don't use dropout
            attn_weights = self.dropout(attn_weights)

        context = attn_weights @ v  # (b, h, seq, head_dim)
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)


class RoPEMHA(MultiHeadAttention, RoPEMixin):
    pass


get = make_get_function(globals())