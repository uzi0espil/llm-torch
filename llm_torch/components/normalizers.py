from torch import nn
import torch

from llm_torch.utils.core import make_get_function


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.scale_ = nn.Parameter(torch.ones(emb_dim))  # parameters to control std
        self.shift_ = nn.Parameter(torch.zeros(emb_dim))  # parameters to control mean

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)  # unbiased=False for using t-student

        x = (x - mean) / (std + self.eps)
        return self.scale_ * x + self.shift_


get = make_get_function(globals())