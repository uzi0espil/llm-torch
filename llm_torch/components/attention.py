import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple

from llm_torch.components.normalizer import Normalizer
from llm_torch.utils.core import make_get_function


class RoPEMixin(object):
    """Requires the consumer (e.g., MultiHeadAttention) to define:
      - self.head_dim (even)
      - self.context_length (max sequence length used for precomputation)"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__.endswith("Mixin"):
            return

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

    def _pos_map(self, pos):
        return pos[:, None]

    def _apply_rope(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        """Apply RoPE to (b, h, seq, d). Uses even/odd interleaving, not first-half/second-half split."""
        b, h, seq, d = x.shape
        pos = torch.arange(start_index, start_index + seq, device=x.device, dtype=torch.float32)
        pos = self._pos_map(pos)
        angles = (pos * self.freqs[None, :]).to(x.device)
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


class YarnMixin(RoPEMixin):
    """Standalone implementation of Yarn, ideally you want to extend RoPEMixin,
    but I separate it for learning purposes."""
    def __init__(self,
                 factor: float,
                 low_freq: float,
                 high_freq: float,
                 *args,
                 original_max_pos_embeddings: Optional[int] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        half = self.head_dim // 2
        idx = torch.arange(half, dtype=torch.float32)
        frac = idx / (half - 1) if half > 1 else torch.zeros_like(idx)
        per_dim = factor * (low_freq + (high_freq - low_freq) * frac)  # (half,)
        self.register_buffer("per_dim_scale", per_dim, persistent=False)
        self.orig_max_pos = original_max_pos_embeddings or self.context_length

    def _pos_map(self, pos):
        base = torch.minimum(pos, torch.tensor(float(self.orig_max_pos), device=pos.device))[:, None]
        extra = torch.relu(pos - float(self.orig_max_pos))[:, None]
        return base + extra / self.per_dim_scale[None, :].to(pos.device)


class NTKMixin(RoPEMixin):
    """Applies NTK-aware rotary embeddings with YaRN concentration smoothing."""
    def __init__(self,
                 factor: float,
                 alpha: float,
                 beta: float,
                 *args,
                 original_max_pos_embeddings: Optional[int] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor
        self.alpha = alpha
        self.beta = beta
        self.original_max_pos_embeddings = original_max_pos_embeddings or self.context_length

        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        self.register_buffer(
            "ntk_inv_freq",
            inv_freq,
            persistent=False,
        )
        self.register_buffer(
            "ntk_concentration",
            torch.tensor(concentration, dtype=torch.float32),
            persistent=False,
        )

    def _compute_concentration_and_inv_freq(self) -> Tuple[float, torch.Tensor]:
        freq = self.theta_base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )
        if self.factor > 1.0:
            concentration = 0.1 * math.log(self.factor) + 1.0

            half = self.head_dim // 2
            d_half = float(half)
            low = (
                d_half
                * math.log(self.original_max_pos_embeddings / (self.beta * 2 * math.pi))
                / math.log(self.theta_base)
            )
            high = (
                d_half
                * math.log(self.original_max_pos_embeddings / (self.alpha * 2 * math.pi))
                / math.log(self.theta_base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(half, dtype=torch.float32) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _apply_rope(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        b, h, seq, d = x.shape
        pos = torch.arange(
            start_index, start_index + seq, device=x.device, dtype=torch.float32
        )
        freqs = torch.einsum(
            "i,j->ij", pos, self.ntk_inv_freq.to(device=x.device, dtype=torch.float32)
        )
        concentration = self.ntk_concentration.to(
            device=x.device, dtype=torch.float32
        )
        cos = torch.cos(freqs) * concentration
        sin = torch.sin(freqs) * concentration
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_rot_even
        x_out[..., 1::2] = x_rot_odd
        return x_out.to(dtype=x.dtype)


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
            self.cache_k = torch.zeros(batch_size, self.n_kv_group,
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

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None


class BaseAttention(nn.Module):

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 context_length: int,
                 head_dim: Optional[int] = None,
                 dropout_rate: Optional[float] = 0.1,
                 n_heads: int = 8,
                 n_kv_group: int = 8,
                 mask: bool = True,
                 qkv_bias: bool = False,
                 qk_norm: Optional[Normalizer] = None,
                 dtype: torch.dtype = torch.float32):
        assert n_kv_group <= n_heads, "kv_group should be <= than n_heads"
        assert n_heads % n_kv_group == 0, "n_heads must be divisible by n_kv_group (for grouping)"
        if head_dim is None:
            assert d_out % n_heads == 0, "d_out must be divisible by n_heads"


        self.d_out = d_out
        self.d_in = d_in
        self.n_heads = n_heads
        self.head_dim = head_dim or d_out // n_heads
        self.context_length = context_length
        self.to_mask = mask
        self.n_kv_group = n_kv_group
        self.group_size_ = n_heads // n_kv_group
        attn_inner = n_heads * self.head_dim

        # defer super call until attributes are assigned.
        super().__init__()

        # linear projections
        self.W_q = nn.Linear(d_in, attn_inner, dtype=dtype, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, n_kv_group * self.head_dim, dtype=dtype, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, n_kv_group * self.head_dim, dtype=dtype, bias=qkv_bias)
        self.out_proj = nn.Linear(attn_inner, d_out, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None
        self.q_norm = qk_norm(self.head_dim) if qk_norm is not None else None
        self.k_norm = qk_norm(self.head_dim) if qk_norm is not None else None

    # overridable hooks
    def transform_keys(self, keys: torch.Tensor, start_index: int = 0) -> torch.Tensor:
        return keys

    def transform_queries(self, queries: torch.Tensor, start_index: int = 0) -> torch.Tensor:
        return queries

    def get_mask_pool(self, num_tokens, num_keys, use_cache=False):
        if not use_cache:
            return torch.triu(torch.ones(num_tokens, num_keys, dtype=torch.bool,
                                         device=self.W_q.weight.device), diagonal=1)

        offset = num_keys - num_tokens
        row_idx = torch.arange(num_tokens).unsqueeze(1).to(device=self.W_q.weight.device)  # (num_tokens, 1)
        col_idx = torch.arange(num_keys).unsqueeze(0).to(device=self.W_q.weight.device)
        return row_idx + offset < col_idx  # True where j > i + offset

    def persist_kv(self, k, v, num_tokens, batch_size):
        return k, v

    def compute_attention_scores(self, q, k, num_tokens, use_cache=False):
        attn_scores = q @ k.transpose(2, 3)

        if self.to_mask:
            mask_pool = self.get_mask_pool(num_tokens, attn_scores.size(-1), use_cache=use_cache)
            attn_scores.masked_fill_(mask_pool, float("-inf"))

        return attn_scores

    def compute_attention_weights(self, attention_scores):
        scale = self.head_dim ** 0.5
        attn_weights = torch.softmax(attention_scores / scale, dim=-1)
        return attn_weights

    def compute_context(self, attention_weights, v, num_tokens, use_cache=False):
        context = attention_weights @ v  # (b, h, seq, head_dim)
        context = context.transpose(1, 2).contiguous().view(v.shape[0], num_tokens, self.n_heads * self.head_dim)
        return context

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        b, num_tokens, _ = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # reshape to (b, n_heads, seq, head_dim)
        q = q.view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.n_kv_group, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.n_kv_group, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Optional transforms, such RoPE, etc...
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
        attn_scores = self.compute_attention_scores(q, k, num_tokens=num_tokens, use_cache=use_cache)

        # attention weights
        attn_weights = self.compute_attention_weights(attn_scores)
        if self.dropout is not None:  # some architectures don't use dropout
            attn_weights = self.dropout(attn_weights)

        # context
        context = self.compute_context(attn_weights, v, num_tokens=num_tokens, use_cache=use_cache)
        return self.out_proj(context)


class MultiHeadAttention(CacheMixin, BaseAttention):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 context_length: int,
                 head_dim: Optional[int] = None,
                 dropout_rate: Optional[float] = 0.1,
                 n_heads: int = 8,
                 mask: bool = True,
                 qkv_bias: bool = False,
                 qk_norm: Optional[Normalizer] = False,
                 dtype: torch.dtype = torch.float32,
                 # Mixin init variables
                 kv_window_size: Optional[int] = None):
        """Standard MHA: set n_kv_group == n_heads (i.e., no grouping)."""
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
            n_heads=n_heads,
            n_kv_group=n_heads,
            mask=mask,
            qkv_bias=qkv_bias,
            dtype=dtype,
            kv_window_size=kv_window_size,
            qk_norm=qk_norm,
        )


class NaiveSWA(CacheMixin, BaseAttention):
    """Naive SWA: computes full scores then masks out tokens beyond the window."""

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 context_length: int,
                 window_size: int,
                 head_dim: Optional[int] = None,
                 dropout_rate: Optional[float] = 0.1,
                 n_heads: int = 8,
                 mask: bool = True,
                 n_kv_group: Optional[int] = None,
                 qkv_bias: bool = False,
                 qk_norm: Optional[Normalizer] = None,
                 kv_window_size: Optional[int] = None,
                 dtype: torch.dtype = torch.float32):
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if window_size > context_length:
            raise ValueError("window_size cannot exceed context_length.")

        self.window_size = window_size
        n_kv_group = n_kv_group or n_heads
        kv_window = kv_window_size or self.window_size

        super().__init__(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
            n_heads=n_heads,
            n_kv_group=n_kv_group,
            mask=mask,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            dtype=dtype,
            kv_window_size=kv_window,
        )

    def get_mask_pool(self, num_tokens, num_keys, use_cache: bool = False):
        base_mask = super().get_mask_pool(num_tokens, num_keys, use_cache=use_cache)
        device = self.W_q.weight.device
        row_idx = torch.arange(num_tokens, device=device).unsqueeze(1)  # (tokens, 1)
        col_idx = torch.arange(num_keys, device=device).unsqueeze(0)    # (1, keys)

        if num_keys == num_tokens:
            effective_rows = row_idx
        else:
            offset = num_keys - num_tokens
            effective_rows = row_idx + offset

        window_mask = col_idx + self.window_size <= effective_rows
        return base_mask | window_mask


class SlidingWindowAttention(CacheMixin, BaseAttention):
    """Vectorised SWA that avoids dense matmul by operating on windowed tensors."""
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 context_length: int,
                 window_size: int,
                 head_dim: Optional[int] = None,
                 dropout_rate: Optional[float] = 0.1,
                 n_heads: int = 8,
                 n_kv_group: Optional[int] = None,
                 mask: bool = True,
                 qkv_bias: bool = False,
                 qk_norm: Optional[Normalizer] = None,
                 kv_window_size: Optional[int] = None,
                 dtype: torch.dtype = torch.float32):
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if window_size > context_length:
            raise ValueError("window_size cannot exceed context_length.")

        self.window_size = window_size
        n_kv_group = n_kv_group or n_heads
        kv_window = kv_window_size or self.window_size

        super().__init__(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
            n_heads=n_heads,
            n_kv_group=n_kv_group,
            mask=mask,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            dtype=dtype,
            kv_window_size=kv_window,
        )

    def get_mask_pool(self, num_queries: int, total_keys: int, use_cache: bool = False):
        """Mask out positions that fall outside the causal window."""
        device = self.W_q.weight.device
        w = self.window_size

        # In cache mode, we are processing the *last* num_queries tokens.
        if use_cache:
            # Their positions are [total_keys - num_queries, ..., total_keys - 1]
            start_pos = total_keys - num_queries
            positions = torch.arange(start_pos, total_keys, device=device)
        else:
            # In pre-fill mode, num_queries == total_keys, all positions from 0 to total_keys - 1 are processed.
            assert num_queries == total_keys, "In non-cache mode, num_queries must equal total_keys"
            positions = torch.arange(total_keys, device=device)  # Shape (num_queries,)

        # For each query position `pos`, the number of *valid* causal keys in its window is min(pos + 1, window_size).
        valid_counts = torch.clamp(positions + 1, max=w)  # Shape (num_queries,)

        # The number of "padding" or acausal tokens at the start of the window.
        pad_counts = w - valid_counts  # Shape (num_queries,)
        window_positions = torch.arange(w, device=device).unsqueeze(0)  # Shape (1, w)
        mask = window_positions < pad_counts.unsqueeze(-1)

        return mask.view(1, 1, num_queries, w)  # reshape for broadcasting with attn_scores

    def _apply_sliding_window(self, x, num_tokens, use_cache=False):
        x_padded = F.pad(x, (0, 0, self.window_size - 1, 0))
        x_windows = x_padded.unfold(2, self.window_size, 1)  # (b, h, seq_len, d, w)
        x_windows = x_windows.transpose(-1, -2)  # (b, h, seq_len, w, d)

        if use_cache:
            return x_windows[:, :, -num_tokens:, :, :]
        return x_windows

    def compute_attention_scores(self, q, k, num_tokens, use_cache=False):
        total_keys = k.size(-2)
        k_windows = self._apply_sliding_window(k, num_tokens, use_cache=use_cache)

        q_expanded = q.unsqueeze(-2)  # (b, h, seq, 1, d)
        attn_scores = torch.sum(q_expanded * k_windows, dim=-1)  # (b, h, seq, w)

        if self.to_mask:
            window_mask = self.get_mask_pool(num_queries=num_tokens, total_keys=total_keys, use_cache=use_cache)
            attn_scores.masked_fill_(window_mask, float("-inf"))
        return attn_scores

    def compute_context(self, attention_weights, v, num_tokens, use_cache=False):
        """attention_weights: shape: (b, h, num_tokens, w)"""
        v_windows = self._apply_sliding_window(v, num_tokens, use_cache=use_cache)

        attn_weights_expanded = attention_weights.unsqueeze(-1)  # (b, h, num_tokens, w, 1)
        context = torch.sum(attn_weights_expanded * v_windows, dim=-2)  # (b, h, num_tokens, d)

        context = context.transpose(1, 2).contiguous().view(v.shape[0], num_tokens, self.n_heads * self.head_dim)
        return context


class GroupedKeyAttention(CacheMixin, BaseAttention):
    pass


class RoPEMHA(RoPEMixin, MultiHeadAttention):
    pass


class RoPEGOA(RoPEMixin, GroupedKeyAttention):
    pass


class YarnGOA(YarnMixin, GroupedKeyAttention):
    pass


class YarnSWA(YarnMixin, SlidingWindowAttention):
    pass


class NTKSWA(NTKMixin, SlidingWindowAttention):
    pass


class NTKNaiveSWA(NTKSWA, NaiveSWA):
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
