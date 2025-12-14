import json
import os
import time
from functools import partial

# 导入必要的库
import requests
import tiktoken
import torch

# 导入自定义模块
from chapter04.styc_04_dummy_gpt_model import GPTModel
from chapter05.calculate_loss import calc_loss_loader
from chapter05.get_device import get_torch_device
from chapter05.gpt_download import download_and_load_gpt2
from chapter05.openai import load_weights_into_gpt
from chapter05.pretraining import generate, text_to_token_ids, token_ids_to_text, train_model_simple, \
    plot_losses, text_to_token_ids_mac
from chapter07.instruction_finetuning_01_mian_instruction_dataset_class import InstructionDataset, custom_collate_fn, \
    format_input


def test():
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    # 如果文件不存在，则从 GitHub 下载
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    # 读取 JSON 数据
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 设置数据划分比例：85% 训练，10% 测试，5% 验证
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    # 划分数据集
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    # 加载模型参数
    # 初始化 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    
    # 设置设备 (GPU 或 CPU)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_torch_device()
    print("Device:", device)

    # 自定义 collate_fn，用于 batch 数据处理
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    from torch.utils.data import DataLoader
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    # 创建训练集 DataLoader
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    # 创建验证集 DataLoader
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    # 创建测试集 DataLoader
    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    # print("Train loader:")
    # for inputs, targets in train_loader:
    #     print(inputs.shape, targets.shape)

    # 基础模型配置
    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True  # Query-key-value bias
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    # CHOOSE_MODEL = "gpt2-small (124M)"
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    # 下载并加载 GPT-2 预训练权重
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="/Users/zhangzhiming/PycharmProjects/llm-study/chapter05/gpt2"
        # models_dir="E:\PythonProject\study\llm-study\chapter07\gpt2"
    )
    
    # 初始化模型并加载权重
    model = GPTModel(BASE_CONFIG)
    # 打印加载前的一个权重切片（例如第一层注意力的权重和）
    print("Weight sum before loading:", model.trf_blocks[0].att.W_query.weight.sum().item())
    load_weights_into_gpt(model, params)
    model.to(device)
    # 打印加载后的权重和
    print("Weight sum after loading:", model.trf_blocks[0].att.W_query.weight.sum().item())
    model.eval()
    print(model.tok_emb.weight.std())
    # print(model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr())

    # == 微调前测试 ==
    # 测试一条数据，看看微调前的模型表现
    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    print(input_text)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids_mac(input_text, tokenizer, device),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    print("generated_text: ", generated_text)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    print("====================================")
    print(response_text)


    torch.manual_seed(123)
    
    # 计算初始 loss
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    print("============== before start train ==============")

    # 开始训练
    start_time = time.time()
    torch.manual_seed(123)
    # 使用 AdamW 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )
    num_epochs = 2
    
    # 调用简单的训练循环
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 绘制 Loss 曲线
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # evaluate
    # 在测试集前 3 个样本上进行评估
    torch.manual_seed(123)
    for entry in test_data[:3]:
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")

    # 7.9 Generating test set responses
    # 为所有测试集数据生成回复并保存
    from tqdm import tqdm
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        test_data[i]["model_response"] = response_text
    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)


if __name__ == '__main__':
    test()
    # torch.backends.mps.is_available()