import torch


def simple_example_01():
    input_ids = torch.tensor([2, 3, 5, 1])
    vocab_size = 6
    output_dim = 3
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # 打印单词的权重矩阵
    print(embedding_layer.weight)
    # 输入 [3]，输出单词 ID 为 3 的嵌入向量（形状为 (1, 3)）。
    print(embedding_layer(torch.tensor([3])))
    # 批量查询多个词的嵌入向量
    print(embedding_layer(input_ids))


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    import tiktoken
    from chapter01.GPTdatasetV1 import GPTDatasetV1
    from torch.utils.data import DataLoader
    # Initializes the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # Creates dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  # True drops the last batch if it is shorter than the specified batch_size to prevent
        # loss spikes during training.
        num_workers=num_workers  # The number of CPU processes to use for preprocessing
    )
    return dataloader


def positional_embeddings_02():
    # 创建嵌入层
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    # 创建一个数据加载器
    with open("../chapter01/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )

    # 生成一个[8,4]的矩阵
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    # token embedding 创建词嵌入层
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    print(token_embeddings)

    context_length = max_length
    print("==============================")
    print(torch.arange(context_length))
    # 生成位置嵌入
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)

    # 合并词嵌入和位置嵌入
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)


if __name__ == '__main__':
    # simple_example_01()
    positional_embeddings_02()
