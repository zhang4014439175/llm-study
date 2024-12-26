import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """
        Compared to the previous SelfAttention_v1 class, we added a dropout layer.
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        We transpose dimensions 1 and 2, keeping the batch dimension at the first position (0).
        In PyTorch, operations with a trailing underscore are performed in-place, avoiding unnecessary memory copies
        """
        # b是批处理大小
        # token的数量，6个单词
        # d_in是每个元素的特征维度
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        # self.W_query(x)等价于x @ self.W_query.weight.T + self.W_query.bias
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
