import torch.nn as nn
import torch


# (1) 参数定义方式
# v1（nn.Parameter）：
# python
# self.W_query = nn.Parameter(torch.rand(d_in, d_out))
# 直接定义可训练的权重矩阵，通过 @ 运算符手动实现矩阵乘法。
# 缺点：没有内置偏置项，若需偏置需手动添加。
# v2（nn.Linear）：
# python
# self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
# 使用 nn.Linear 封装权重矩阵和偏置项，前向传播时直接调用线性层。
# 优点：支持偏置（bias=True），且代码更简洁（自动处理矩阵乘法和偏置加法）。
# (2) 偏置项（Bias）
# v1：
# 无偏置，计算过程为纯矩阵乘法：x @ W_query。
# v2：
# 通过 qkv_bias 参数控制是否启用偏置：
# python
# self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
# 如果 qkv_bias=True，计算过程为：x @ W_query.T + b_query（nn.Linear 自动处理）。
# (3) 代码可读性与扩展性
# v1：
# 显式写出矩阵乘法，适合教学或底层实现，但扩展性较差（如需添加 Dropout 或 LayerNorm 需手动修改）。
# v2：
# 使用 nn.Linear 更符合 PyTorch 习惯，易于扩展（例如替换为 nn.Sequential 组合其他操作）。

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
