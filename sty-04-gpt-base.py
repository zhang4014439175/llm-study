import torch
from torch import nn


def GPT_CONFIG_124M():
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
    model = DummyGPTModel(GPT_CONFIG_124M())
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

    torch.set_printoptions(sci_mode=False)

    from styc_04_dummy_gpt_model import LayerNorm
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)


def feed_forward():
    # GELU activations
    import matplotlib.pyplot as plt
    from styc_04_dummy_gpt_model import GELU
    gelu, relu = GELU(), nn.ReLU()
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)
    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    from styc_04_dummy_gpt_model import FeedForward
    ffn = FeedForward(GPT_CONFIG_124M())
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print(out.shape)


def shortcut_connections():
    print(torch.backends.mps.is_available())
    from styc_04_dummy_gpt_model import ExampleDeepNeuralNetwork
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )

    print_gradients(model_without_shortcut, sample_input)

    print("========== open use_shortcut ========== ")
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=True
    )
    print_gradients(model_with_shortcut, sample_input)


def print_gradients(model, x):
    # 112 A.4 and A.7 in appendix A
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        # if 'weight' in name and param.grad is not None:
        #     print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
        # else:
        #     print(f"{name} has no gradient")
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


def transformer_block():
    from styc_04_dummy_gpt_model import TransformerBlock
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M())
    output = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


if __name__ == '__main__':
    # gpt_base_01()
    # feed_forward()
    # shortcut_connections()
    transformer_block()