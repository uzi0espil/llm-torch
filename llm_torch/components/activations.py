import torch
from llm_torch.utils.core import make_get_function


class BaseActivation(torch.nn.Module):
    pass


class GELU(BaseActivation):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2. / torch.pi)) *
            (x + 0.044715 * x**3)
        ))


class SiLU(BaseActivation):

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * torch.sigmoid(self.alpha * x)


get = make_get_function(globals())