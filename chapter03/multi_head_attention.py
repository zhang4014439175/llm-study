import torch
from torch import nn

from causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            ) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 首先，它检查输出维度d_out是否可以被头数num_heads整除，这是为了确保每个头都能获得相等的维度。
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        # 每个头的维度
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 一个线性层，用于将最终的上下文向量（context vector）投影回原始的d_out维度。
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # todo 不懂
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # 首先，它获取输入x的形状，其中b是批次大小，num_tokens是序列中的令牌（token）数量，d_in是输入特征维度。
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # 将查询、键和值的张量重塑为包含多个头的形状，即(b, num_tokens, num_heads, head_dim)。
        # 将 keys 张量重塑为一个四维张量，其形状为 [批次大小, 令牌数量, 头数量, 每头的特征维度]。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 将查询、键和值的最后一个维度（即头维度）与令牌维度交换，以便进行批量矩阵乘法。
        # 转换为了[b, self.num_heads, num_tokens, self.head_dim]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # 通过查询和键的矩阵乘法来计算注意力分数。这里使用了@运算符进行矩阵乘法。
        attn_scores = queries @ keys.transpose(2, 3)
        # Masks truncated to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Adds an optional linear projection
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
