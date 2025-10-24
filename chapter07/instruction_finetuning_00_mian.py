import json
import os
import urllib
from urllib.request import urlopen

import torch

from chapter05.pretraining import generate, text_to_token_ids, token_ids_to_text, calc_loss_loader
from chapter07.instruction_finetuning_01_mian_instruction_dataset_class import custom_collate_draft_1, format_input, \
    custom_collate_draft_2, custom_collate_fn, get_data_loaders
from chapter07.instruction_finetuning_03_loading_llm import loading_llm


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def prepare_dateset():
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))
    print("Example entry:\n", data[50])
    print("Another example entry:\n", data[999])


def test():
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = download_and_load_file(file_path, url)
    # ### Instruction:
    # Identify the correct spelling of the following word.
    # ### Input:
    # Ocassion
    # ### Response:
    # The correct spelling is 'Occasion.'
    # model_input = format_input(data[50])
    # desired_response = f"\n\n### Response:\n{data[50]['output']}"
    # print(model_input + desired_response)
    #
    # model_input = format_input(data[999])
    # desired_response = f"\n\n### Response:\n{data[999]['output']}"
    # print(model_input + desired_response)

    # 设置数据比例 partitioned the dataset
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Test set length:", len(test_data))
    print("Validation set length:", len(val_data))

    # 加载模型参数
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 测试填充效果
    # inputs_1 = [0, 1, 2, 3, 4]
    # inputs_2 = [5, 6]
    # inputs_3 = [7, 8, 9]
    # batch = (
    #     inputs_1,
    #     inputs_2,
    #     inputs_3
    # )
    # print(custom_collate_draft_1(batch))
    # inputs, targets = custom_collate_draft_2(batch)
    # print(inputs)
    # print(targets)
    #
    # inputs, targets = custom_collate_fn(batch)
    # print(inputs)
    # print(targets)

    train_loader, val_loader, test_loader = get_data_loaders(train_data, val_data, test_data, tokenizer, device)
    print("Train loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)

    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    print(input_text)

    model, BASE_CONFIG = loading_llm()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(generated_text)
    response_text = generated_text[len(input_text):].strip()
    print(response_text)

    model.to(device)

    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=5
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=5
        )
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


def get_dataset():
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = download_and_load_file(file_path, url)
    # ### Instruction:
    # Identify the correct spelling of the following word.
    # ### Input:
    # Ocassion
    # ### Response:
    # The correct spelling is 'Occasion.'
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_response)

    model_input = format_input(data[999])
    desired_response = f"\n\n### Response:\n{data[999]['output']}"
    print(model_input + desired_response)

    # 设置数据比例 partitioned the dataset
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Test set length:", len(test_data))
    print("Validation set length:", len(val_data))
    return train_data, test_data, val_data


if __name__ == '__main__':
    print("start")
    # 1、准备数据
    # prepare_dateset()
    test()
