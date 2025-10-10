import torch
from instruction_finetuning_01_mian_instruction_dataset_class import custom_collate_fn


def replace_effect():
    logits_1 = torch.tensor(
        [[-1.0, 1.0],
         [-0.5, 1.5]]
    )
    targets_1 = torch.tensor([0, 1])  # Correct token indices to generate
    loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    print(loss_1)

    logits_2 = torch.tensor(
        [[-1.0, 1.0],
         [-0.5, 1.5],
         [-0.5, 1.5]]
    )
    targets_2 = torch.tensor([0, 1, 1])
    loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    print(loss_2)

    targets_3 = torch.tensor([0, 1, -100])
    loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
    print(loss_3)
    print("loss_1 == loss_3:", loss_1 == loss_3)

    # Uncomments these two lines to use the GPU on an Apple Silicon chip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.backends.mps.is_available():
    # device = torch.device("mps")"
    print("Device:", device)


if __name__ == '__main__':
    replace_effect()
