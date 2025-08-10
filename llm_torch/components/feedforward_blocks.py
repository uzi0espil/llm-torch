from torch import nn
import torch
from llm_torch.components.activations import GELU, SiLU
from llm_torch.utils.core import make_get_function


class FFBlock(nn.Module):

    def __init__(self, emb_dim, hidden_dim, activation=GELU, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim, dtype=dtype),
            activation(),
            nn.Linear(hidden_dim, emb_dim, dtype=dtype)
        )

    def forward(self, x):
        return self.layers(x)


class SwiGLUBlock(nn.Module):
    """Swish-Gated Linear Unit network

    Instead of a single path followed by an activation, SwiGLU forms two parallel projections and multiples them,
    letting the model gate information token-wise. SiLU keeps negative partially active and pairs well with the gate.
    Empirical work shows that SiGLU matches or beats GELU at the same compute budget."""

    def __init__(self, emb_dim, hidden_dim, activation=SiLU, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.f1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.f2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.f3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False)
        self.silu = activation()

    def forward(self, x):
        x1 = self.f1(x)
        x2 = self.f2(x)
        gated = self.silu(x1) * self.silu(x2)
        return self.f3(gated)


get = make_get_function(globals())