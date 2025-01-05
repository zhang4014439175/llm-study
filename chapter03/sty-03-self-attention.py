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
    res = 0
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

    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    print(attn_scores)

    # 1) Compute attention scores
    # Compute the attention scores as dot products between the inputs.

    # 2) Compute attention weights
    # The attention weights are a normalized version of the attention scores.

    # 3) Compute context vectors
    # The context vectors are computed as a weighted sum over the inputs.

    # However, for loops are generally slow, and we can achieve the same results using matrix multiplication:
    attn_scores = inputs @ inputs.T
    print(attn_scores)

    # normalization, when dim=-1, it represent computing each row
    # because attn_scores is a two-dimension tensor, it's last or second dimension is row,not column
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)

    row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    print("Row 2 sum:", row_2_sum)
    print("All row sums:", attn_weights.sum(dim=-1))

    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)
    print("Previous 2nd context vector:", context_vec_2)


def trainable_weights():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )

    # 这段代码展示了在PyTorch中，如何对一个输入张量（inputs）进行自注意力机制中的关键步骤之一，
    # 即计算单个输入向量（在这个例子中是x_2，代表"journey"）的查询（query）、键（key）和值（value）向量。
    # 自注意力机制是深度学习，特别是在自然语言处理（NLP）和某些图像处理任务中广泛使用的一种技术。下面是代码的详细解释：
    # d_in:
    # 它们的设置决定了模型的特征表示空间，并且直接影响到自注意力机制中查询（Query）、键（Key）和值（Value）向量的计算。
    # d_in 是查询、键、值向量的输入维度，确保输入的特征能够通过权重矩阵投影到不同的表示空间
    # d_out:
    # d_out 是查询（Query）、键（Key）、值（Value）向量的输出维度，即这些向量在注意力机制中使用的特征维度。
    # d_out 决定了计算注意力分数的特征空间大小,
    # 查询向量 query 和键向量 key 的点积发生在 d_out 维度中，因此 d_out 决定了计算注意力分数时特征的抽象程度。
    # 如果 d_out 太小，表示能力有限，可能无法捕捉复杂的特征关系；
    # 如果 d_out 太大，可能导致计算复杂度增加且易过拟合。
    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    # 矩阵乘法操作（@）的结果是将x_2从原始的3维特征空间转换到新的2维空间，分别得到查询、键、值向量的表示。
    # 查询（Query）、键（Key）、值（Value）是自注意力机制（Self-Attention）中的核心概念，
    # 用于捕捉输入序列中不同部分之间的相关性。它们在计算注意力权重和生成上下文向量的过程中起着重要作用。
    # 以下是它们的详细解释：
    # (1) 查询（Query）：
    # 定义：查询向量是用来提出“问题”的向量，表示某个单词或输入在当前计算中“关注的方向”。
    # 作用：它与键向量计算点积，用于衡量当前输入（或单词）与其他输入（或单词）的相似性。
    # 本质：表达输入的“关注需求”或上下文中希望了解的特定关系。
    # (2) 键（Key）：
    # 定义：键向量是对每个输入进行特征化的向量，用来回答查询向量提出的问题。
    # 作用：它与查询向量结合，决定两个输入之间的相关性。
    # 本质：表达输入的“特征”或“标识”。
    # (3) 值（Value）：
    # 定义：值向量是与键向量一一对应的向量，包含输入的实际信息。
    # 作用：它通过注意力权重加权求和，形成输出的上下文向量。
    # 本质：是输入携带的信息，经过注意力机制后被传递到输出。
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(W_query)
    print(W_key)
    print(W_value)
    print(query_2)
    print(key_2)
    print(value_2)

    # 获取到输入字符的所有key值和value值
    # keys 根据实时 W_key 来计算出当前字符的字符特征是什么
    # values 根据实时 W_value 来计算出当前字符的内容是什么
    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)

    # 这个是需要查询的字符 journey 的key，根据 query值 来进行计算注意力分数
    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    print(attn_score_22)

    # 所有的注意力分数,点积计算注意力分数
    attn_scores_2 = query_2 @ keys.T
    print(attn_scores_2)

    # 归一化计算占比
    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
    print(attn_weights_2)
    print(values)

    # 得到 journey 的上下文向量
    # values 提供了句子中每个单词的语义内容。
    # attn_weights_2 是一种“放大镜”，告诉模型需要关注哪些内容。
    # context_vec_2 是一个“总结”，它动态地结合了当前语境中最相关的信息。
    context_vec_2 = attn_weights_2 @ values
    print(context_vec_2)


def softmax_naive(x):
    # Softmax函数是一种将K个实值向量转换为总和为1的K个实值向量的函数。
    # 具体来说，对于输入向量z = [z1, z2, ..., zk]，Softmax函数将其转换为输出向量y = [y1, y2, ..., yk]，
    # 其中每个元素yi的计算公式为：
    #  yi = exp(zi) / Σj exp(zj)
    # 这里，exp(zi)表示zi的指数函数值，Σj exp(zj)表示所有输入元素指数函数值的和，用于归一化，确保输出向量的元素之和为1。
    return torch.exp(x) / torch.exp(x).sum(dim=0)


def causal_attention():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )
    torch.manual_seed(123)
    d_in = inputs.shape[1]
    d_out = 2
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    values = inputs @ W_value

    # SelfAttention_v1 and SelfAttention_v2 give different outputs because
    # they use different initial weights for the weight matrices since nn.Linear uses a more
    # sophisticated weight initialization scheme
    from self_attention_v1 import SelfAttention_v1, SelfAttention_v2
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))

    """
    掩码注意力分数计算为上下文向量
    """
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    print("===================================================")
    print(attn_weights)

    # create a mask where the values above the diagonal are zero
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print("========== mask_simple ===========")
    print(mask_simple)

    # multiply this mask with the attention weights to zero-out the values above the diagonal:
    masked_simple = attn_weights * mask_simple
    print(masked_simple)

    # renormalize the attention weights to sum up to 1
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print(row_sums)
    print(masked_simple_norm)

    # torch.ones(context_length, context_length) 生成一个全1矩阵
    # 使用torch.triu函数生成一个上三角矩阵
    # 应用掩码到注意力分数
    # mask.bool() 将mask矩阵转换为布尔类型，即1变为True，0变为False。
    # attn_scores.masked_fill(..., -torch.inf)：掩码注意力分数
    # 然后，使用masked_fill函数，将attn_scores中对应于mask为True（即原mask矩阵中值为1的位置）的元素替换为-torch.inf。
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(mask)
    print(masked)
    attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
    print(attn_weights)
    print(values)
    context_vec = attn_weights @ values
    print(context_vec)


def causal_attention_class():
    # inputs
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )
    d_in = inputs.shape[1]
    d_out = 2

    # compute attn_scores and attn_weights
    from self_attention_v1 import SelfAttention_v2
    sa_v2 = SelfAttention_v2(d_in, d_out)
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    values = sa_v2.W_value(inputs)
    attn_scores = queries @ keys.T
    context_length = attn_scores.shape[0]
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)

    # create masked
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)

    torch.manual_seed(123)
    attn_weights = dropout(attn_weights)
    attn_score = attn_weights @ values
    print(attn_weights)
    print(attn_score)

    # batch: use causal attention class
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)
    print(batch)

    from causal_attention import CausalAttention
    torch.manual_seed(123)
    context_length = batch.shape[1]
    print(context_length)
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)
    print("context_vecs:", context_vecs)


def multi_head_attention():
    print("multi_head_attention")
    # batch like up
    # inputs
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )
    batch = torch.stack((inputs, inputs), dim=0)
    torch.manual_seed(123)
    context_length = batch.shape[1]  # This is the number of tokens
    d_in, d_out = 3, 2

    from multi_head_attention import MultiHeadAttentionWrapper, MultiHeadAttention
    mha = MultiHeadAttentionWrapper(
        d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)


def mutil_head_test():
    # The shape of this tensor is (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4).
    a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                        [0.8993, 0.0390, 0.9268, 0.7388],
                        [0.7179, 0.7058, 0.9156, 0.4340]],
                       [[0.0772, 0.3565, 0.1479, 0.5331],
                        [0.4066, 0.2318, 0.4545, 0.9737],
                        [0.4606, 0.5159, 0.4220, 0.5786]]]])
    # print(a.transpose(2, 3))
    # print(a @ a.transpose(2, 3))

    # 通过索引a[0, 0, :, :]，我们选择了
    #   批次中的第一个元素（0，假设批次大小为第一维）、
    #   第一个头（0，假设头数量为第二维）的
    #   所有序列长度/令牌（:，表示选择这一维的所有元素）和
    #   所有特征维度（:，同样表示选择这一维的所有元素）。
    first_head = a[0, 0, :, :]
    print(first_head)
    first_res = first_head @ first_head.T
    print("First head:\n", first_res)
    second_head = a[0, 1, :, :]
    second_res = second_head @ second_head.T
    print("\nSecond head:\n", second_res)


if __name__ == '__main__':
    # 91页Summary
    # trainable_weights()
    # use_class()
    # causal_attention_class()
    multi_head_attention()
    # mutil_head_test()
