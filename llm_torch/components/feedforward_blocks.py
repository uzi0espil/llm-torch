from torch import nn
from llm_torch.components.activations import GELU
from llm_torch.utils.core import make_get_function


class FFBlock(nn.Module):

    def __init__(self, emb_dim, activation=GELU):
        super().__init__()
        expansion = emb_dim * 4
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, expansion),
            activation(),
            nn.Linear(expansion, emb_dim)
        )

    def forward(self, x):
        return self.layers(x)


get = make_get_function(globals())