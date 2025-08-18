import torch
from matplotlib import pyplot as plt


def test():
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }
    inverse_vocab = {v: k for k, v in vocab.items()}

    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    # 之前学习的根据最大可能性获取下个token的方式
    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()
    print(inverse_vocab[next_token_id])

    # temperature scaling
    # 多项函数对下一个与它的概率得分成比例的标记进行采样。换句话说，“forward”仍然是最有可能的代币，并且在大多数时间（但不是所有时间）将被多项选择。为了说明这一点，让我们实现一个函数，重复这个采样1000次：
    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])
    print_sampled_tokens(probas, inverse_vocab)

    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i],
                       bar_width, label=f'Temperature = {T}')
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # top_k = 3
    # top_logits, top_pos = torch.topk(next_token_logits, top_k)
    # print("Top logits:", top_logits)
    # print("Top positions:", top_pos)

    # new_logits = torch.where(
    #     condition=next_token_logits < top_logits[-1],
    #     input=torch.tensor(float('-inf')),
    #     other=next_token_logits
    # )
    # print(new_logits)

    # topk_probas = torch.softmax(new_logits, dim=0)
    # print(topk_probas)

    top_k_sampling(next_token_logits)


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


def print_sampled_tokens(probas, inverse_vocab):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
              for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


def top_k_sampling(next_token_logits):
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print("Top logits:", top_logits)
    print("Top positions:", top_pos)

    # 随后，我们应用PyTorch的where函数将低于我们前三个选择中最低logit值的令牌的logit值设置为负无穷（-inf）：
    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],
        input=torch.tensor(float('-inf')),
        other=next_token_logits
    )

    print(new_logits)

    # 最后，让我们应用softmax函数将这些转换为下一个令牌概率：
    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)

    # We can now apply the temperature scaling and multinomial function for probabilistic
    # sampling to select the next token among these three non-zero probability scores to
    # generate the next token. We do this next by modifying the text generation function.
    # 现在，我们可以应用温度缩放和多项式函数进行概率抽样，从这三个非零概率分数中选择下一个标记来生成下一个标记。接下来，我们通过修改文本生成函数来实现这一点。

if __name__ == '__main__':
    test()
