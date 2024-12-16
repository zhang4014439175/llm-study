import torch


def dot_products():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )

    # dot products
    # select a query token
    query = inputs[1]
    # storage the attention score between query token and each input token
    attn_scores_2 = torch.empty(inputs.shape[0])
    # compute dot attention scores
    for i, x_i in enumerate(inputs):
        print(x_i)
        attn_scores_2[i] = torch.dot(x_i, query)
    print(attn_scores_2)

    print("============ compute dot products ==================")
    res = 0.
    for idx, element in enumerate(inputs[0]):
        print(inputs[0][idx])
        print(query[idx])
        res += inputs[0][idx] * query[idx]
    print("============ compute dot products end ==================")
    print(res)
    print(torch.dot(inputs[0], query))

    # 在接下来的步骤中，如图3.9所示，我们对之前计算出的每个注意力分数进行归一化处理。
    # 归一化的主要目的是获得总和为1的注意力权重。
    # 这种归一化是一种惯例，对于大型语言模型（LLM）中的解释和保持训练稳定性非常有用。以下是实现这一归一化步骤的直接方法：
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())

    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())

    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())

    # 确实，当处理非常大或非常小的输入值时，直接实现（或称为“朴素实现”）softmax函数可能会遇到数值不稳定性的问题，
    # 如溢出（overflow）和下溢（underflow）。溢出通常发生在指数函数（exp）的计算中，因为当输入值非常大时，
    # exp的结果可能会超出浮点数的表示范围。而下溢则发生在输入值非常小且为负数时，exp的结果会接近于零，
    # 可能导致在计算归一化分母时出现除零错误（尽管在浮点运算中通常不会真正除零，但结果会非常接近于零，从而导致数值精度问题）。
    #
    # 为了解决这个问题，并在实践中获得更好的性能和数值稳定性，建议使用PyTorch等深度学习框架提供的softmax实现。
    # PyTorch的softmax函数已经过广泛的优化，不仅考虑了性能，还内置了处理数值不稳定性的机制。

    # 现在我们已经计算出了归一化的注意力权重，接下来就可以进行最后一步了，
    # 如图3.10所示：通过将嵌入的输入标记x(i)与相应的注意力权重相乘，然后将得到的向量相加，来计算上下文向量z(2)。
    # 因此，上下文向量z(2)是所有输入向量的加权和，每个输入向量都乘以其对应的注意力权重得到。
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        print(context_vec_2)
        context_vec_2 += attn_weights_2[i] * x_i
    print(context_vec_2)


def softmax_naive(x):
    # Softmax函数是一种将K个实值向量转换为总和为1的K个实值向量的函数。
    # 具体来说，对于输入向量z = [z1, z2, ..., zk]，Softmax函数将其转换为输出向量y = [y1, y2, ..., yk]，
    # 其中每个元素yi的计算公式为：
    #  yi = exp(zi) / Σj exp(zj)
    # 这里，exp(zi)表示zi的指数函数值，Σj exp(zj)表示所有输入元素指数函数值的和，用于归一化，确保输出向量的元素之和为1。
    return torch.exp(x) / torch.exp(x).sum(dim=0)


if __name__ == '__main__':
    dot_products()
