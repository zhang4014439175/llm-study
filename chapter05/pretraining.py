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


# 上一章
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 1. 提取当前上下文（最后 context_size 个 token）
        idx_cond = idx[:, -context_size:]

        # 2. 禁用梯度计算（推理阶段不需要梯度）
        with torch.no_grad():
            logits = model(idx_cond)  # 模型预测下一个 token 的 logits

        # 3. 取最后一个时间步的 logits（只关心下一个 token 的预测）
        logits = logits[:, -1, :]

        # 4. Softmax 归一化，得到概率分布
        probas = torch.softmax(logits, dim=-1)

        # 5. 选择概率最高的 token ID（贪婪解码）
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # 6. 将新 token 拼接到输入序列
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


# 这个方法 token_ids_to_text 的作用是将一个包含token IDs的Tensor转换为对应的文本字符串。
# 这个过程通常用于自然语言处理（NLP）任务中，特别是在使用基于token的模型（如BERT、GPT等）时。
def token_ids_to_text(token_ids, tokenizer):
    # 移除token_ids的第一个维度（通常是批次大小维度）
    flat = token_ids.squeeze(0)
    # 将Tensor转换为Python列表，然后使用tokenizer的decode方法将列表转换为文本
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


# 1、本章第一步
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


# 接下来，我们将为生成的输出计算损失度量。这种损失作为训练进度的进展和成功指标。此外，在后面的章节中，当我们微调LLM时，我们将回顾评估模型质量的其他方法。
def no01_calculating_the_text_generation_loss():
    inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                           [40, 1107, 588]])  # "I really like"]

    targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
                            [1107, 588, 11311]])  # " really like chocolate"]

    # 1、获取模型生成的结果
    model, GPT_CONFIG_124M = get_model()
    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)

    # torch.Size([2, 3, 50257])
    # 2 is the two examples (rows)
    # 3 is each rows have 3 token
    # 50257 is the embedding dimensionality

    # We can complete steps 3 and 4 by applying the argmax function to the probability scores to obtain the
    # corresponding token IDs:
    # 用于返回指定维度上最大值的索引。在这个上下文中，它被用来找到每个位置上概率最高的token的索引。
    # 2、返回指定维度上的最大索引
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)

    # Finally, step 5 converts the token IDs back into text:
    # 3、输入目标文本以及最后生成文本
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1:"
          f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    # we can print the initial softmax probability scores corresponding to the target tokens
    # 从probas中提取第0个文本样本在位置0, 1, 2上对应于targets[0]指定索引的概率值
    # 这里[0, 1, 2]是一个索引列表，指定了感兴趣的位置
    # targets[text_idx]给出了在这些位置上我们感兴趣的词汇表中的索引
    # 两个例子为了演示loss的计算
    # 4、取出targets中文本索引的概率，effort moves you中第3626个概率,really like chocolate中，第588个概率
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)

    # The three target token ID probabilities for each batch are
    # Text 1: tensor([7.4541e-05, 3.1061e-05, 1.1563e-05])，取出来这个字符的张量
    # Text 2: tensor([1.0337e-05, 5.6776e-05, 4.7559e-06])，取出来这个字符的张量

    # 5、Calculating the loss involves several steps
    # 计算损失可以知道生成的结果和预期结果平均相差多少
    # 5.1 计算Log probabilities
    # 5.2 Average log probability
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)
    # tensor([ -9.5042, -10.3796, -11.3677, -11.4798,  -9.7764, -12.2561])

    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)
    # tensor(-10.7940)

    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)
    # tensor(10.7940)

    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)


# 计算通过训练和验证加载器返回的给定批的交叉熵损失
def calc_loss_batch(input_batch, target_batch, model, device):
    # The transfer to a given device allows us to transfer the data to a GPU.
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


# 计算由给定数据加载器采样的所有批处理的损失
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

    # 创建训练集和验证集各自的加载器
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
        model.parameters(),  # The .parameters() method returns all trainable weight parameters of the model.
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
    # 我们首先将模型从GPU转移回CPU，因为使用相对较小的模型进行推理不需要GPU。
    # 同样，训练结束后，我们将模型置于评估模式，关闭dropout等随机成分
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

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    from chapter04.styc_04_dummy_gpt_model import GPTModel
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    # Using model.eval() switches the model to evaluation mode
    # for inference, disabling the dropout layers of the model
    model.eval()

    # 保存模型信息
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
        "model_and_optimizer.pth"
    )

    # 加载模型信息，
    checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()


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
    train_losses, val_losses, track_tokens_seen = [], [], []  # Initializes lists to track losses and tokens seen
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):  # Starts the main training loop
        model.train()  # 切换模型到训练模式
        for input_batch, target_batch in train_loader:  # 加载训练数据
            optimizer.zero_grad()  # Resets loss gradients from the previous batch iteration # 清除梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前batch的损失
            loss.backward()  # Calculates loss gradients # 反向传播计算梯度
            optimizer.step()  # Updates model weights using loss gradients # 更新模型参数
            tokens_seen += input_batch.numel()  # 累计处理的token数
            global_step += 1  # 更新全局步数
            if global_step % eval_freq == 0:  # Optional evaluation step # 按指定频率评估
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)  # 记录训练损失
                val_losses.append(val_loss)  # 记录验证损失
                track_tokens_seen.append(tokens_seen)  # 记录当前token数
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )
        generate_and_print_sample(model, tokenizer, device, start_context)  # Prints a sample text after each epoch
    return train_losses, val_losses, track_tokens_seen


def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):  # The for loop is the same as before: gets logits and only focuses on the last
        # time step
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:  # Filters logits with top_k sampling
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:  # Applies temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:  # Carries out greedy next-token selection as before when temperature scaling is disabled
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:  # Stops generating early if end-of-sequence token is encountered
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_test():
    torch.manual_seed(123)
    model, GPT_CONFIG_124M = get_model()
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == '__main__':
    # init_gpt()
    # no01_calculating_the_text_generation_loss()
    # pretrain()
    # test()
    generate_test()

# When LLMs generate text, they output one token at a time.
#  By default, the next token is generated by converting the model outputs into
# probability scores and selecting the token from the vocabulary that corresponds
# to the highest probability score, which is known as “greedy decoding.”
#  Using probabilistic sampling and temperature scaling, we can influence the
# diversity and coherence of the generated text.
#  Training and validation set losses can be used to gauge the quality of text generated by LLM during training.
# Pretraining an LLM involves changing its weights to minimize the training loss.
#  The training loop for LLMs itself is a standard procedure in deep learning,
# using a conventional cross entropy loss and AdamW optimizer.
#  Pretraining an LLM on a large text corpus is time- and resource-intensive, so we
# can load openly available weights as an alternative to pretraining the model on
# a large dataset ourselves.

# Exercise 5.5
# Calculate the training and validation set losses of the GPTModel with the pretrained
# weights from OpenAI on the “The Verdict” dataset.

# Exercise 5.6 Experiment with GPT-2 models of different sizes—for example, the largest 1,558 million parameter
# model—and compare the generated text to the 124 million model.
