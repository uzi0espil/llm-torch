import torch
from torch import nn

from llm_torch.utils.core import make_get_function


class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout_rate=0.1, n_heads=8, mask=True, qkv_bias=False):
        super(MultiHeadAttention, self).__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.d_in = d_in
        self.head_dim = d_out // n_heads
        self.n_heads = n_heads
        self.to_mask = mask

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout_rate)

        # register_buffer offer multiple advantages: it automatically move the variable to the correct device along
        # without model, and it is included in the states_dict that will be saved and loaded correctly.
        if self.to_mask:
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # reshape the final value to n_heads, head_dim and then reshape it to (b, n_heads, num_tokens, head_dim)
        q = q.view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = q @ k.transpose(2, 3)  # dot product for each head

        # apply mask
        if self.to_mask:
            # Original mask truncated to the number of tokens and head dims
            mask_pool = self.mask.bool()[:num_tokens, :num_tokens]
            # Use the mast to fill the attention
            attn_scores.masked_fill_(mask_pool, -float('inf'))

        # compute the attention weights
        attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)

        # apply dropout
        attn_output = self.dropout(attn_weights)

        # compute context vector
        context_vec = attn_output @ v  # shape it back to (b, num_tokens, n_heads, head_dims)

        # concatenate and apply the projection layer
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


get = make_get_function(globals())