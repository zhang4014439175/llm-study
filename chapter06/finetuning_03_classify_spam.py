import torch

from chapter06.finetuning_02_calc_accuracy import calc_loss_loader, calc_accuracy_loader, calc_loss_batch
import matplotlib.pyplot as plt


def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    # 创建空列表来记录训练/验证的损失和准确率（train_losses, val_losses, train_accs, val_accs）。
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    # Main training loop
    for epoch in range(num_epochs):
        # 训练模式：model.train() 设置模型为训练模式（影响Dropout和BatchNorm等层）。
        model.train()
        for input_batch, target_batch in train_loader:
            # 清除梯度。
            optimizer.zero_grad()
            # 前向传播计算当前批次的损失（未展示具体实现，但应包含前向传播和损失计算）。
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # 反向传播计算梯度。
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 更新examples_seen（当前批次的样本数）和global_step（总步数）。
            examples_seen += input_batch.shape[0]
            global_step += 1

            # Optional
            # evaluation
            # step
            if global_step % eval_freq == 0:
                # 调用evaluate_model（未展示）计算训练集和验证集的损失，并记录到列表中。
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                # 打印当前步数的训练和验证损失。
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )

        # 调用calc_accuracy_loader（未展示）计算训练集和验证集的准确率（可指定eval_iter限制评估的批次数）。
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


# 然后我们使用Matplotlib绘制训练和验证集的损失函数。
def plot_values(
        epochs_seen, examples_seen, train_values, val_values,
        label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    # Plots training and validation loss against epoch
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.",
        label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    # Creates a second x-axis for examples seen
    ax2 = ax1.twiny()
    # Invisible plot for aligning ticks
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    # Adjusts layout to make room
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


def classify_review(
        text, model, tokenizer, device, max_length=None,
        pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    # Prepares inputs to the model
    supported_context_length = model.pos_emb.weight.shape[1]

    # Truncates sequences if they are too long
    input_ids = input_ids[:min(
        max_length, supported_context_length
    )]

    # Pads sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    # Adds batch dimension
    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)
    # Models inference without gradient tracking
    with torch.no_grad():
        # Logits of the last output token
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Returns the classified result
    return "spam" if predicted_label == 1 else "not spam"
