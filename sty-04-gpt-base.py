import torch
from torch import nn


def gpt_config():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size 50,257 words
        "context_length": 1024,  # Context length the maximum number of input tokens
        "emb_dim": 768,  # Embedding dimension transforming each token into a 768-dimensional vector.
        "n_heads": 12,  # Number of attention heads the count of attention heads in the multi-head attention mechanism
        "n_layers": 12,  # Number of layers specifies the number of transformer blocks in the model
        "drop_rate": 0.1,  # Dropout rate the intensity of the dropout mechanism to prevent overfitting
        "qkv_bias": False
        # Query-Key-Value bias determines whether to include a bias vector in the Linear layers ofthe multi-head attention for query, key, and value computations
    }

    return GPT_CONFIG_124M


def gpt_base_01():
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    from styc_04_dummy_gpt_model import DummyGPTModel
    torch.manual_seed(123)
    model = DummyGPTModel(gpt_config())
    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)

    # short for rectified linear unit The neural network layer we have coded consists of a Linear layer followed by a
    # non-linear activation function, ReLU (short for rectified linear unit), which is a standard activation function
    # in neural networks. If you are unfamiliar with ReLU, it simply thresholds negative inputs to 0, ensuring that a
    # layer outputs only positive values, which explains why the resulting layer output does not contain any negative
    # values.
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)

    # keepdim: keepdim=True in operations like mean or variance calculation ensures that the output tensor retains
    # the same number of dimensions as the input tensor
    #
    # dim: the operation reduces the tensor along the dimension specified via dim.For instance, without keepdim=True,
    # the returned mean tensor would be a two-dimensional vector [0.1324, 0.2170] instead of a 2 × 1–dimensional
    # matrix [[0.1324], [0.2170]].
    # dim=0：沿着第一个维度（通常是批次大小维度）进行计算。如果张量形状为(N, ...)，则对每一列（即每个样本的所有特征）进行计算。
    # dim=1：沿着第二个维度进行计算。如果张量形状为(N, C, ...)，则对每一个样本的每一个通道（或特征）的所有其他维度进行计算。
    # dim=-1：沿着最后一个维度进行计算。这通常用于处理特征向量或一维数组，其中最后一个维度包含了你想要计算统计量的数据。
    # mean: 均值计算
    # var: 方差计算
    # 例子：
    #   [[0.1, 0.4],
    #   [0.2, 0.3]]
    # 如果我们沿着行（dim=0）计算均值，不使用keepdim=True，结果将是一个一维向量：
    #   [0.15, 0.35]
    # 然而，如果我们使用keepdim=True，结果将是一个2x1的矩阵（二维张量）：
    #   [[0.15],
    #   [0.35]]
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)



if __name__ == '__main__':
    gpt_base_01()
