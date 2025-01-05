# This is a sample Python script.


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def load_file():
    # Use a breakpoint in the code line below to debug your script.
    import urllib.request
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)


def open_file():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])


def split_study():
    import re
    # text = "Hello, world. This, is a test."
    # result = re.split(r'(\s)', text)
    # result = re.split(r'([,.]|\s)', text)
    # print(result)
    # result = [item for item in result if item.strip()]
    # print(result)
    text = "Hello, world. Is this-- a test?"
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item.strip() for item in result if item.strip()]
    print(result)


# 1.encode or decode
def split_file():
    import re

    # 1.print string-to-value
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    # print(len(preprocessed))
    # print(preprocessed[:30])

    all_words = sorted(set(preprocessed))
    # vocab_size = len(all_words)
    # print(vocab_size)

    # 2.print first of 50 items in text
    vocab = {token: integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        # print(item)
        if i >= 50:
            break

    # 3.use tokenizer class to encode or decode text
    from chapter02.simple_tokenizer_v1 import SimpleTokenizerV1
    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know," 
     Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

    # 4.enhance handle unknow words
    # insert two token about unk and endoftext
    # unk:          encounters a word that is not part of the vocabulary
    # endoftext:    add a token between unrelated texts
    print("=================== 4 =====================")
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    print(len(vocab.items()))
    print(vocab)
    print(vocab.items())
    print(list(vocab.items()))
    print(enumerate(list(vocab.items())))
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    from chapter02.simple_tokenizer_v2 import SimpleTokenizerV2
    tokenizer = SimpleTokenizerV2(vocab)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))

    # 5.byte pair encoding
    print("=================== 5 =====================")
    from importlib.metadata import version
    import tiktoken
    print("tiktoken version:", version("tiktoken"))
    tokenizer = tiktoken.get_encoding("gpt2")
    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)

    strings = tokenizer.decode(integers)
    print(strings)


def sliding_window():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))
    enc_sample = enc_text[50:]

    # 1.create the input–target pairs for the next word prediction task
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size + 1]
    print(f"x: {x}")
    print(f"y: {y}")

    # 2.By processing the inputs along with the targets, which are the inputs shifted by one
    # position, we can create the next-word prediction tasks
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)

    # 3.repeat the previous code but convert the token IDs into text
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    import tiktoken
    from GPTdatasetV1 import GPTDatasetV1
    from torch.utils.data import DataLoader
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


def test_dataloader():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)

    # how use the data loader to sample with a batch size greate than 1
    print("===========================")
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4,
        shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_dataloader()
