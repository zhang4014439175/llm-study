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

    # 1.how data flows in and out of a GPT model,
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    # tokenizer.encode(txt1) 将文本转换为 token ID 列表。
    # torch.tensor(...)：将列表转换为 PyTorch 张量。
    # tokenizer 常用于 OpenAI API 或类似 GPT 模型的文本预处理。
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    # torch.stack(batch, dim=0)：将列表中的两个张量沿第 0 维度（行）堆叠，形成一个形状为 (2, n) 的张量，
    # 其中 n 是序列的最大长度（此处为 4，因为 txt2 会被填充到与 txt1 相同的长度）
    # txt1_ids = [1427, 1857, 2276, 13]  # "Every effort moves you"
    # txt2_ids = [1427, 2293, 2769, 9]   # "Every day holds a"
    # tensor([[1427, 1857, 2276,   13],
    #         [1427, 2293, 2769,    9]])
    batch = torch.stack(batch, dim=0)
    print(batch)

    # 2.Next, we initialize a new 124-million-parameter DummyGPTModel instance and feed it
    # the tokenized batch:
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
    # 设置随机种子保证可复现
    # batch_example为生成的[2, 5]张量
    # nn.Sequential定义神经网络层，输入维度为 5，输出维度为 6。添加 ReLU 激活函数（将负数置为 0，正数保持不变）。
    # 将层按顺序组合，形成 输入 → Linear → ReLU → 输出 的流水线。
    # 3.implement a neural network layer with five inputs and six outputs that we apply to two input examples:
    # Let’s now implement layer normalization to improve the stability and efficiency of neural network training.
    # 层归一化背后的主要思想是调整神经网络层的激活（输出），使其均值为0，方差为1，也称为单位方差。
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example) # todo 创建神经网络向前传播
    print(out)

    # keepdim: keepdim=True in operations like mean or variance calculation ensures that the output tensor retains
    # the same number of dimensions as the input tensor
    #
    # dim: the operation reduces the tensor along the dimension specified via dim.For instance, without keepdim=True,
    # the returned mean tensor would be a two-dimensional vector [0.1324, 0.2170] instead of a 2 × 1–dimensional
    # matrix [[0.1324], [0.2170]].
    # dim=0：对每一列的所有行取均值，结果会减少第 0 维度的长度，沿着第一个维度（通常是批次大小维度）进行计算。如果张量形状为(N, ...)，则对每一列（即每个样本的所有特征）进行计算。
    # dim=1：对每一行的所有列取均值，结果会减少第 1 维度的长度（从 (2, 6) 变为 (2, 1)）。沿着第二个维度进行计算。如果张量形状为(N, C, ...)，则对每一个样本的每一个通道（或特征）的所有其他维度进行计算。
    # dim=-1：沿着最后一个维度进行计算。这通常用于处理特征向量或一维数组，其中最后一个维度包含了你想要计算统计量的数据。
    # 使用dim=-1始终用最后一个维度进行计算，避免dim=1改为dim=2
    # mean: 均值计算
    # var: 方差计算
    # 例子：
    #   [[0.1, 0.4],
    #   [0.2, 0.3]]
    # 如果我们沿着列（dim=0）计算均值，不使用keepdim=True，结果将是一个一维向量：
    #   [0.15, 0.35]
    # 如果我们沿着行（dim=1）计算均值，结果为[0.25, 0.25]
    # 然而，如果我们使用keepdim=True，结果将是一个2x1的矩阵（二维张量）：
    #   [[0.15],
    #   [0.35]]
    # 4.层归一化，将层输入的值进行归一，使数据的均值为0，方差为1，减少波动性
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    print("============== LayerNorm =======================")
    # 默认打印方式:
    # tensor([1.0000e-04, 1.0000e+04])
    # 禁用科学计数法后的打印方式:
    # tensor([   0.0001, 10000.0000])
    torch.set_printoptions(sci_mode=False)

    # 5.层归一化封装，实现层归一化
    from styc_04_dummy_gpt_model import LayerNorm
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)


def feed_forward_02():
    # 层归一结束后，开始前反馈
    # GELU activations,绘制图形
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

    # 前反馈由两个线性层和一个GELU激活函数组成
    from styc_04_dummy_gpt_model import FeedForward
    ffn = FeedForward(GPT_CONFIG_124M())
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print(out.shape)


def shortcut_connections_03():
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
    # 用于打印模型中权重参数梯度均值
    # 112 A.4 and A.7 in appendix A
    # 将输入数据 x 传递给模型，获取模型的输出。
    # todo 前向传播
    output = model(x)
    # 创建一个目标张量，这里是一个包含单个元素0的二维张量，用于计算损失函数。
    target = torch.tensor([[0.]])
    # 创建一个均方误差损失（Mean Squared Error Loss）的实例
    loss = nn.MSELoss()
    # 计算模型输出 output 和目标张量 target 之间的均方误差损失。
    loss = loss(output, target)
    # 进行反向传播，计算损失函数关于模型参数的梯度
    loss.backward()
    for name, param in model.named_parameters():
        # if 'weight' in name and param.grad is not None:
        #     print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
        # else:
        #     print(f"{name} has no gradient")
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


def transformer_block_04():
    from styc_04_dummy_gpt_model import TransformerBlock
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M())
    output = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


def gpt_model_test():
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    from styc_04_dummy_gpt_model import GPTModel
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M())
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    # 模型总参数减去输出层参数
    # sum(p.numel() for p in model.out_head.parameters()) 计算输出层所有参数的总数。
    # total_params_gpt2 是计算得到的模型总参数数量减去输出层参数数量后的结果.
    total_params_gpt2 = (
            total_params - sum(p.numel() for p in model.out_head.parameters())
    )
    print(f"Number of trainable parameters "
          f"considering weight tying: {total_params_gpt2:,}"
          )

    # 计算模型需要内存
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 这段代码演示了使用PyTorch为语言模型生成循环的简单实现。它迭代生成指定数量的新令牌，
    # 裁剪当前上下文以适应模型的最大上下文大小，计算预测，然后根据最大概率预测选择下一个令牌。
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text():
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    from styc_04_dummy_gpt_model import GPTModel
    model = GPTModel(GPT_CONFIG_124M())
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M()["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


def generate_next_token():
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    from styc_04_dummy_gpt_model import GPTModel
    model = GPTModel(GPT_CONFIG_124M())
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M()["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


if __name__ == '__main__':
    # gpt_base_01()
    feed_forward_02()
    # shortcut_connections()
    # transformer_block()
    # gpt_model_test()
    # generate_text()
    # generate_next_token()
