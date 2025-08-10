import torch
from llm_torch.utils.core import make_get_function


class GELU(torch.nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2. / torch.pi)) *
            (x + 0.044715 * x**3)
        ))


class SiLU(torch.nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


get = make_get_function(globals())