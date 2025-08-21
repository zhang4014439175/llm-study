# GPT-2模型的权重
import tiktoken
import torch

from chapter05.pretraining import text_to_token_ids, token_ids_to_text


def open_test():
    # 下周gpt_download.py文件
    # import urllib.request
    # url = (
    #     "https://raw.githubusercontent.com/rasbt/"
    #     "LLMs-from-scratch/main/ch05/"
    #     "01_main-chapter-code/gpt_download.py"
    # )
    # filename = url.split('/')[-1]
    # urllib.request.urlretrieve(url, filename)

    # 下载GPT2模型
    from gpt_download import download_and_load_gpt2
    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="gpt2"
    )
    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    model_name = "gpt2-small (124M)"
    GPT_CONFIG_124M_Test = {
        "vocab_size": 50257,  # Vocabulary size 50,257 words
        "context_length": 1024,  # Context length the maximum number of input tokens
        "emb_dim": 768,  # Embedding dimension transforming each token into a 768-dimensional vector.
        "n_heads": 12,  # Number of attention heads the count of attention heads in the multi-head attention mechanism
        "n_layers": 12,  # Number of layers specifies the number of transformer blocks in the model
        "drop_rate": 0.1,  # Dropout rate the intensity of the dropout mechanism to prevent overfitting
        "qkv_bias": False
        # Query-Key-Value bias determines whether to include a bias vector in the Linear layers ofthe multi-head
        # attention for query, key, and value computations
    }
    NEW_CONFIG = GPT_CONFIG_124M_Test.copy()
    NEW_CONFIG.update(model_configs[model_name])

    NEW_CONFIG.update({"context_length": 1024})

    # OpenAI在多头注意力模块的线性层中使用了偏差向量来实现查询、键和值矩阵的计算。偏置向量在llm中不再常用，
    # 因为它们不能提高建模性能，因此是不必要的。然而，由于我们正在使用预训练的权重，我们需要匹配一致性设置并启用这些偏差向量：
    NEW_CONFIG.update({"qkv_bias": True})
    from chapter04.styc_04_dummy_gpt_model import GPTModel
    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    # 默认情况下，GPTModel实例使用随机权重初始化以进行预训练。使用OpenAI模型权重的最后一步是用我们加载到params字典中的权重覆盖这些随机权重。
    # 为此，我们将首先定义一个小的assign实用函数，用于检查两个张量或数组（左和右）是否具有相同的维度或形状，并将右张量作为可训练的PyTorch参数返回：
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_weights_into_gpt(gpt, params)
    gpt.to(device)

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    from pretraining import generate
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                         "Right: {right.shape}"
                         )
    return torch.nn.Parameter(torch.tensor(right))


def GPT_CONFIG_124M():
    return {
        "vocab_size": 50257,  # Vocabulary size 50,257 words
        "context_length": 1024,  # Context length the maximum number of input tokens
        "emb_dim": 768,  # Embedding dimension transforming each token into a 768-dimensional vector.
        "n_heads": 12,  # Number of attention heads the count of attention heads in the multi-head attention mechanism
        "n_layers": 12,  # Number of layers specifies the number of transformer blocks in the model
        "drop_rate": 0.1,  # Dropout rate the intensity of the dropout mechanism to prevent overfitting
        "qkv_bias": False
        # Query-Key-Value bias determines whether to include a bias vector in the Linear layers ofthe multi-head
        # attention for query, key, and value computations
    }


import numpy as np


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
    gpt.trf_blocks[b].att.W_query.weight = assign(
        gpt.trf_blocks[b].att.W_query.weight, q_w.T)
    gpt.trf_blocks[b].att.W_key.weight = assign(
        gpt.trf_blocks[b].att.W_key.weight, k_w.T)
    gpt.trf_blocks[b].att.W_value.weight = assign(
        gpt.trf_blocks[b].att.W_value.weight, v_w.T)
    q_b, k_b, v_b = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
    gpt.trf_blocks[b].att.W_query.bias = assign(
        gpt.trf_blocks[b].att.W_query.bias, q_b)
    gpt.trf_blocks[b].att.W_key.bias = assign(
        gpt.trf_blocks[b].att.W_key.bias, k_b)
    gpt.trf_blocks[b].att.W_value.bias = assign(
        gpt.trf_blocks[b].att.W_value.bias, v_b)
    gpt.trf_blocks[b].att.out_proj.weight = assign(
        gpt.trf_blocks[b].att.out_proj.weight,
        params["blocks"][b]["attn"]["c_proj"]["w"].T)
    gpt.trf_blocks[b].att.out_proj.bias = assign(
        gpt.trf_blocks[b].att.out_proj.bias,
        params["blocks"][b]["attn"]["c_proj"]["b"])
    gpt.trf_blocks[b].ff.layers[0].weight = assign(
        gpt.trf_blocks[b].ff.layers[0].weight,
        params["blocks"][b]["mlp"]["c_fc"]["w"].T)
    gpt.trf_blocks[b].ff.layers[0].bias = assign(
        gpt.trf_blocks[b].ff.layers[0].bias,
        params["blocks"][b]["mlp"]["c_fc"]["b"])
    gpt.trf_blocks[b].ff.layers[2].weight = assign(
        gpt.trf_blocks[b].ff.layers[2].weight,
        params["blocks"][b]["mlp"]["c_proj"]["w"].T)
    gpt.trf_blocks[b].ff.layers[2].bias = assign(
        gpt.trf_blocks[b].ff.layers[2].bias,
        params["blocks"][b]["mlp"]["c_proj"]["b"])
    gpt.trf_blocks[b].norm1.scale = assign(
        gpt.trf_blocks[b].norm1.scale,
        params["blocks"][b]["ln_1"]["g"])
    gpt.trf_blocks[b].norm1.shift = assign(
        gpt.trf_blocks[b].norm1.shift,
        params["blocks"][b]["ln_1"]["b"])
    gpt.trf_blocks[b].norm2.scale = assign(
        gpt.trf_blocks[b].norm2.scale,
        params["blocks"][b]["ln_2"]["g"])
    gpt.trf_blocks[b].norm2.shift = assign(
        gpt.trf_blocks[b].norm2.shift,
        params["blocks"][b]["ln_2"]["b"])


if __name__ == '__main__':
    open_test()


