import tiktoken
import torch

from chapter05.pretraining import token_ids_to_text, get_model


def calculate_loss():
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
    # probas 每个位置上对整个词表的 softmax 概率分布
    # targets（正确的 token ID）。targets（正确的 token ID）。
    # 就相当于同时取出 第0个句子的所有位置上，模型给真实 token 的概率。
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
    # torch.cat把两个拼成一个整体，tensor([7.4541e-05, 3.1061e-05, 1.1563e-05, 1.0337e-05, 5.6776e-05, 4.7559e-06])
    # 然后通过交叉熵损失函数计算： ln(0.000074541) = -9.5042
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)
    # tensor([ -9.5042, -10.3796, -11.3677, -11.4798,  -9.7764, -12.2561])

    # 求平均值
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
    input_batch = input_batch.to(device) # 将输入数据转移到GPU/CPU
    target_batch = target_batch.to(device) # 将标签转移到同一设备
    logits = model(input_batch) # 模型输出未归一化的预测值
    # logits.flatten(0, 1) 合并前两维：(batch_size, seq_len, num_classes) → (batch_size*seq_len, num_classes)
    # target_batch.flatten() 展平标签：(batch_size, seq_len) → (batch_size*seq_len,)
    # 为什么需要展平？
    # 交叉熵损失要求 logits 形状为 (N, C)（N 是样本数，C 是类别数）。
    # 标签形状需为 (N,)（每个样本的类别索引，非 one-hot）。
    # 如果任务是序列标注（如每个 token 分类），需将 batch_size 和 seq_len 合并为一个维度。
    # 输入数据 (input_batch) → 模型 (model) → logits (未归一化预测)
    #   ↓
    # 展平 logits 和标签 → 交叉熵损失计算 → 损失值 (loss)
    #   ↓
    # 反向传播 (loss.backward()) → 优化器更新参数 (optimizer.step())
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