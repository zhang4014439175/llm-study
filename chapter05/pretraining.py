# This chapter covers
# 1.Computing the training and validation set losses
# to assess the quality of LLM-generated text
# during training
# 2.Implementing a training function and pretraining
# the LLM
# 3.Saving and loading model weights to continue
# training an LLM
# 4.Loading pretrained weights from OpenAI
import torch

import tiktoken


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def get_model():
    from chapter04.styc_04_dummy_gpt_model import GPTModel
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    return model, GPT_CONFIG_124M


def init_gpt():
    model, GPT_CONFIG_124M = get_model()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def no01_calculating_the_text_generation_loss():
    inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                           [40, 1107, 588]])  # "I really like"]

    targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
                            [1107, 588, 11311]])  # " really like chocolate"]

    model, GPT_CONFIG_124M = get_model()
    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)

    # torch.Size([2, 3, 50257])
    # 2 is the two examples (rows)
    # 3 is each rows have 3 token
    # 50257 is the embedding dimensionality

    # 1、We can complete steps 3 and 4 by applying the argmax function to the probability scores to obtain the
    # corresponding token IDs:
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)

    # Finally, step 5 converts the token IDs back into text:
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1:"
          f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    # 2、we can print the initial softmax probability scores corresponding to the target tokens
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)

    # The three target token ID probabilities for each batch are
    # Text 1: tensor([7.4541e-05, 3.1061e-05, 1.1563e-05])
    # Text 2: tensor([1.0337e-05, 5.6776e-05, 4.7559e-06])

    # 3、Calculating the loss involves several steps
    # 计算损失可以知道生成的结果和预期结果平均相差多少
    # 3.1 计算Log probabilities
    # 3.2 Average log probability
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)

    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)

    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)

    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)


def calc_loss_batch(input_batch, target_batch, model, device):
    # The transfer to a given device allows us to transfer the data to a GPU.
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    # Iteratives over all batches if no fixed num_batches is specified
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduces the number of batches to match the total number of batches in the data loader if num_batches
        # exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def pretrain():
    file_path = "../chapter01/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    model, GPT_CONFIG_124M = get_model()
    tokenizer = tiktoken.get_encoding("gpt2")
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    # 划分训练集和验证集的比重为 90% 和 10%
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    # print(train_data)
    # print(val_data)

    from chapter01.GPTdatasetV1 import create_dataloader_v1
    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If you have a machine with a CUDA-supported GPU, the LLM will train on the GPU without making any changes to the code.
    model.to(device)
    # Disables gradient tracking for efficiency because we are not training yet
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        # Via the “device” setting, we ensure the data is loaded onto the same device as the LLM model.
        val_loss = calc_loss_loader(val_loader, model, device)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


if __name__ == '__main__':
    # init_gpt()
    # no01_calculating_the_text_generation_loss()
    pretrain()
