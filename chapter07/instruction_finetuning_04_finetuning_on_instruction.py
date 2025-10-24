import tiktoken
import torch

from chapter05.calculate_loss import calc_loss_loader
from chapter07.instruction_finetuning_00_mian import get_dataset
from chapter07.instruction_finetuning_01_mian_instruction_dataset_class import get_data_loaders
from chapter07.instruction_finetuning_03_loading_llm import loading_llm


def calculate_initial_loss():
    train_data, test_data, val_data = get_dataset()
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data_loaders(train_data, val_data, test_data, tokenizer, device)
    model, BASE_CONFIG = loading_llm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=5
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=5
        )
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


if __name__ == '__main__':
    calculate_initial_loss()
