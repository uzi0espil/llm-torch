from torch import nn
import torch

from llm_torch.utils.core import make_get_function
from llm_torch.configs.activations import GELUConfig, SiLUConfig, ActivationConfig
from llm_torch.configs.feedforward_blocks import FFBaseConfig, SwiGLUBlockConfig

from typing import Optional


class FFBaseBlock(nn.Module):
    pass


class FFBlock(FFBaseBlock):

    def __init__(self, emb_dim, hidden_dim, activation: ActivationConfig = GELUConfig(),
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim, dtype=dtype),
            activation.instantiate(),
            nn.Linear(hidden_dim, emb_dim, dtype=dtype)
        )

    def forward(self, x):
        return self.layers(x)


class SwiGLUBlock(FFBaseBlock):
    """Swish-Gated Linear Unit network

    Instead of a single path followed by an activation, SwiGLU forms two parallel projections and multiples them,
    letting the model gate information token-wise. SiLU keeps negative partially active and pairs well with the gate.
    Empirical work shows that SiGLU matches or beats GELU at the same compute budget.

    if emb_dim == hidden_dim, then a third linear projection isn't necessary as per gpt-oss implementation."""

    def __init__(self, emb_dim, hidden_dim, activation: ActivationConfig = SiLUConfig(),
                 limit: float = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.f1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.f2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.f3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False) if emb_dim != hidden_dim else None
        self.silu = activation.instantiate()
        self.limit = limit

    def forward(self, x):
        x1 = self.f1(x)
        x2 = self.f2(x)
        if self.limit is not None:
            x1 = torch.clamp(x1, -self.limit, self.limit)
            x2 = torch.clamp(x2, -self.limit, self.limit)
        gated = self.silu(x1) * x2
        return self.f3(gated) if self.f3 is not None else gated


class MoEBlock(FFBaseBlock):

    def __init__(self, n_experts, emb_dim, hidden_dim, n_experts_per_token=1,
                 activation: ActivationConfig = SiLUConfig(), ff_block: Optional[FFBaseConfig] = None,
                 dtype=torch.float32):
        super().__init__()

        self.router = nn.Linear(emb_dim, n_experts, bias=False, dtype=dtype)
        self.n_experts_per_token = n_experts_per_token
        self.n_experts = n_experts
        if ff_block is None:
            ff_block = SwiGLUBlockConfig(hidden_dim=hidden_dim, activation=activation)
        self.ffs = nn.ModuleList(
            ff_block.instantiate(emb_dim=emb_dim, dtype=dtype) for _ in range(n_experts)
        )

    def forward(self, x):
        """x of shape (b, seq, emb_dim)"""
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])  # for efficiency join batch and seq. => N, emb_d
        scores = self.router(x)  # N, emb_d
        top_k_scores, top_k_indices = torch.topk(scores, self.n_experts_per_token, dim=-1)  # N, K
        top_k_probs = torch.softmax(top_k_scores, dim=-1)  # N, K
        y = torch.zeros_like(x)

        for i in range(self.n_experts_per_token):
            expert_index = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i].unsqueeze(-1)

            for e in range(self.n_experts):
                mask = (expert_index == e)
                if mask.any():
                    out = self.ffs[e](x[mask])
                    y[mask] += expert_probs[mask] * out

        return y.view(original_shape)


get = make_get_function(globals())
