import torch
from torch import nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    """
    A simple placeholder class that will be replaced by a real TransformerBlock later
    """

    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    """
    A simple placeholder class that will be replaced by a real LayerNorm later
    The parameters here are just to mimic the LayerNorm interface.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # is a small constant (epsilon) added to the variance to prevent division by zero
        self.eps = 1e-5
        # 1.The scale and shift are two trainable parameters (of the
        #   same dimension as the input) that the LLM automatically adjusts during training if it
        #   is determined that doing so would improve the model’s performance on its training task.
        # 2.This allows the model to learn appropriate scaling and shifting that best suit the
        #   data it is processing
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # x - mean: x中所有元素减去均值得到中心化数据，他到均值为0
        # 1.举例：[2, 4, 6, 8, 10]
        #   mean：(2 + 4 + 6 + 8 + 10) / 5 = 6 求得 [-4, -2, 0, 2, 4]
        #   方差：(1/n) * Σ(x_i - μ)^2 = (1/5) * [(-4)^2 + (-2)^2 + 0^2 + 2^2 + 4^2]
        #   标准差：√方差 = √8 ≈ 2.83
        #   归一化后的数据 = [-4/2.83, -2/2.83, 0/2.83, 2/2.83, 4/2.83] ≈ [-1.41, -0.71, 0, 0.71, 1.41]
        # 2.归一化后到特点：
        #   均值接近0（由于中心化）。
        #   方差接近1（由于除以了标准差）。
        #   数据点分布在以0为均值、1为标准差的正态分布（或近似正态分布）上。
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
                          GELU())
        ])

    def forward(self, x):
        # 前向传播中执行每一个隐藏层
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


from chapter03.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # initializes the token and positional embedding layers
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # creates a sequential stack of TransformerBlock modules
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 对于GPTModel类中的out_head，它是一个nn.Linear层，用于将Transformer模型的输出x映射到词汇表大小的logits向量上。
        # 这个层的权重（我们可以称之为W_out）在模型初始化时被随机分配，并在训练过程中根据损失函数的梯度进行更新。
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # 通过词嵌入层将token索引转换为嵌入向量
        tok_embeds = self.tok_emb(in_idx)

        # The device setting will allow us to train the model on a CPU or GPU, depending on which
        # device the input data sits on.
        # 生成位置嵌入向量（pos_embeds），其长度与输入序列的长度相同，并将它们添加到嵌入向量上，以结合位置信息。
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        # 应用丢弃层（drop_emb）以减少过拟合。
        x = self.drop_emb(x)
        # 将结合位置信息的嵌入向量传递给Transformer块的序列（trf_blocks），以生成处理后的序列表示
        x = self.trf_blocks(x)
        # 应用层归一化（final_norm）以稳定输出。
        x = self.final_norm(x)
        # 通过输出头（out_head）将处理后的序列表示转换为logits向量，用于预测下一个token。
        logits = self.out_head(x)
        return logits
