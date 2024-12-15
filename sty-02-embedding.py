import re
import torch


def simple_example_01():
    input_ids = torch.tensor([2, 3, 5, 1])
    vocab_size = 6
    output_dim = 3
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    print(embedding_layer(torch.tensor([3])))
    print(embedding_layer(input_ids))


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    import tiktoken
    from GPTdatasetV1 import GPTDatasetV1
    from torch.utils.data import Dataset, DataLoader
    # Initializes the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # Creates dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  # True drops the last  batch if it is shorter than the specified batch_size to prevent
        # loss spikes during training.
        num_workers=num_workers  # The number of CPU processes to use for preprocessing
    )
    return dataloader


def positional_embeddings_02():
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    # token embedding
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    context_length = max_length
    print("==============================")
    print(torch.arange(context_length))
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)


if __name__ == '__main__':
    positional_embeddings_02()
