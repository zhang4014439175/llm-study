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


def get_model(seed=True):
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
    if seed:
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
    # print("Train loader:")
    # for x, y in train_loader:
    #     print(x.shape, y.shape)
    # print("\nValidation loader:")
    # for x, y in val_loader:
    #     print(x.shape, y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If you have a machine with a CUDA-supported GPU, the LLM will train on the GPU without making any changes to
    # the code.
    model.to(device)
    # Disables gradient tracking for efficiency because we are not training yet
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        # Via the “device” setting, we ensure the data is loaded onto the same device as the LLM model.
        val_loss = calc_loss_loader(val_loader, model, device)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    # train start
    torch.manual_seed(123)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # transferring the model back from the GPU to the CPU
    model.to("cpu")
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

def test():
    # 暂时没用
    model, GPT_CONFIG_124M = get_model()
    model.to("cpu")
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # It prints the training and validation set losses after each model update so we can evaluate whether the
    # training improves the model. More specifically, the evaluate_model function calculates the loss over the
    # training and validation set while ensuring the model is in evaluation mode with gradient tracking and dropout
    # disabled when calculating the loss over the training and validation sets:

    # Dropout is disabled during evaluation for stable,reproducible results.
    model.eval()
    # Disables gradient tracking, which is not required during evaluation, to reduce the computational overhead
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    # we use to track whether the model improves during the training. In particular, the generate_and_print_sample
    # function takes a text snippet (start_context) as input, converts it into token IDs, and feeds it to the LLM to
    # generate a text sample using the generate_text_simple function we used earlier:
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter,
                       start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


if __name__ == '__main__':
    # init_gpt()
    # no01_calculating_the_text_generation_loss()
    pretrain()
    # test()
