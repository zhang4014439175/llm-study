import os
import urllib.request
import zipfile
from pathlib import Path

import torch

from chapter06.finetuning_02_calc_accuracy import calc_accuracy_loader, calc_loss_loader
from chapter06.finetuning_03_classify_spam import train_classifier_simple, plot_values
from chapter06.finetuning_struct import SpamDataset

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download "
              "and extraction."
              )
        return
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


# 从一个可能不平衡的数据集 df 中创建一个平衡的数据集 balanced_df
# 特别是在“spam”（垃圾邮件）和“ham”（正常邮件，非垃圾）的样本数量上达到平衡。
# 正常邮件中随机抽取与垃圾邮件数量相等的样本，然后将这些样本与所有的垃圾邮件样本合并，从而创建一个在“spam”和“ham”类别上数量平衡的数据集。
# 这种平衡数据集在构建分类模型时特别有用，因为它可以避免模型因某一类样本数量过多而产生偏差。
def create_balanced_dataset(df, pd):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df


# 将数据集分成三部分：70%用于训练，10%用于验证，20%用于测试。（这些比率在机器学习中很常见，用于训练、调整和评估模型。）
def random_split(df, train_frac, validation_frac):
    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df


# 正式创建数据集
def create_dateset():
    # download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    import pandas as pd

    df = pd.read_csv(
        "sms_spam_collection/SMSSpamCollection.tsv", sep="\t", header=None, names=["Label", "Text"]
    )
    print(df["Label"].value_counts())

    balanced_df = create_balanced_dataset(df, pd)
    print(balanced_df["Label"].value_counts())
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)


def tokenizer_padding_tokens():
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    print(train_dataset.max_length)

    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    from torch.utils.data import DataLoader
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    for input_batch, target_batch in train_loader:
        pass
    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    return train_loader, val_loader, test_loader


def initializing_model_with_pretrained_weights():
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # load the downloaded weights into the GPT model.
    from chapter05.gpt_download import download_and_load_gpt2
    from chapter05.openai import load_weights_into_gpt
    from chapter04.styc_04_dummy_gpt_model import GPTModel
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="../chapter05/"
                                          "gpt2"
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    # After loading the model weights into the GPTModel, we reuse the text generation utility function from chapters
    # 4 and 5 to ensure that the model generates coherent text:
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    from chapter05.pretraining import text_to_token_ids, token_ids_to_text, generate
    text_1 = "Every effort moves you"
    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=text_to_token_ids(text_1, tokenizer),
    #     max_new_tokens=15,
    #     context_size=BASE_CONFIG["context_length"]
    # )
    # token_ids = generate(
    #     model=model,
    #     idx=text_to_token_ids(text_1, tokenizer),
    #     max_new_tokens=15,
    #     context_size=BASE_CONFIG["context_length"],
    #     top_k=25,
    #     temperature=1.4
    # )
    # print(token_ids_to_text(token_ids, tokenizer))

    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=text_to_token_ids(text_2, tokenizer),
    #     max_new_tokens=23,
    #     context_size=BASE_CONFIG["context_length"]
    # )

    # token_ids = generate(
    #     model=model,
    #     idx=text_to_token_ids(text_2, tokenizer),
    #     max_new_tokens=15,
    #     context_size=BASE_CONFIG["context_length"],
    #     top_k=25,
    #     temperature=1.4
    # )
    # print(token_ids_to_text(token_ids, tokenizer))
    # print(model)
    # 为了让模型为分类微调做好准备，我们首先冻结模型，这意味着我们使所有层都不可训练
    for param in model.parameters():
        param.requires_grad = False

    # 然后，我们替换输出层（模型）。Out_head)，它最初将层输入映射到50,257个维度，即词汇表的大小（参见图6.9）
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=num_classes
    )

    # 为了使最后的LayerNorm和最后的transformer块可训练，我们将它们各自的requires_grad设置为True
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape)

    with torch.no_grad():
        outputs = model(inputs)
    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape)
    print("Last output token:", outputs[:, -1, :])

    # We can obtain the class label:
    # probas = torch.softmax(outputs[:, -1, :], dim=-1)
    # label = torch.argmax(probas)
    # print("Class label:", label.item())

    # 在本例中，代码返回1，这意味着模型预测输入文本是“垃圾”。这里使用softmax函数是可选的，因为最大的输出直接对应于最高的概率分数。
    # 因此，我们可以简化代码而不使用softmax：
    # 生成任务使用概率化，可以使用概率高的以及相对高的，分类任务是必须使用概率高的所以就没必要将概率归一化处理了
    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    print("Class label:", label.item())

    train_loader, val_loader, test_loader = tokenizer_padding_tokens()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(
        train_loader, model, device, num_batches=10
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device, num_batches=10
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device, num_batches=10
    )
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # 与计算训练精度类似，我们现在计算每个数据集的初始损失：
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=5
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    import time
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = \
        train_classifier_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=50,
            eval_iter=5
        )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 绘制分类损失 Plotting the classification loss
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    # 使用相同的plot_values函数，现在让我们绘制分类精度：
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(
        epochs_tensor, examples_seen_tensor, train_accs, val_accs,
        label="accuracy"
    )


if __name__ == '__main__':
    print("start")
    # create_dateset()
    # tokenizer_padding_tokens()
    initializing_model_with_pretrained_weights()
