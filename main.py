from pathlib import Path

# import tiktoken
import torch
import torch.nn.functional as F

from mini_llm.config import GPT_CONFIG_124M
from mini_llm.architecture.gpt_model import GPTModel
from mini_llm.pretraining.tokenizer_run import tokenizer
from mini_llm.pretraining.dataset_loader import create_dataloader_v1



PROJECT_ROOT = Path(__file__).resolve().parent
# seed value
torch.manual_seed(123)


def calc_loss_batch(input_batch,target_batch,model,device):
    # device to allow us to transfer to a given device like GPU
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(
        logits.flatten(0,1),target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader,model,device,num_batches=None):
    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))
    for i, (input_batch,target_batch) in enumerate(data_loader):
        # reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch,target_batch,model,device
            )
            # sum the loss over each batch
            total_loss += loss.item()
        else:
            break

    # avg the loss over all batches
    return total_loss / num_batches

def main(DEBUG=False):
    # the_verdict.txt is only ~5k tokens, so a 1024-token context leaves the 10%
    # validation split with zero full-length windows -> empty loader -> nan loss.
    # 256 tokens is short enough that both splits yield batches.
    cfg = GPT_CONFIG_124M(context_length=256)

    file_path = PROJECT_ROOT / "data" / "the_verdict.txt"
    text_data = file_path.read_text(encoding="utf-8")

    # check the number of characters and tokens in the dataset
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print(f"Characters: {total_characters}")
    print(f"Tokens: {total_tokens}")

    ## training part
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    if DEBUG:
        print("=====")
        print("CHECKING if data loaders are created correctly")
        print("Train loader:")
        for x,y in train_loader:
            print(x.shape,y.shape)
        print("\nValidation loader:")
        for x, y in val_loader:
            print(x.shape,y.shape)


    ## applying loss:
    # instantiate the model; eval mode disables dropout so the loss is deterministic
    model = GPTModel(cfg)
    model.eval()

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"DEVICE TYPE: {device}")
    model.to(device)
    with torch.no_grad():
        # disable gradient tacking for efficiency because we are not training yet
        train_loss = calc_loss_loader(train_loader,model,device)
        # via the "device" setting, we ensure the data is loaded onto the same device as the LLM model
        val_loss = calc_loss_loader(val_loader,model,device)

    print(f"Training loss: {train_loss}")
    print(f"Validation loss: {val_loss}")



if __name__ == "__main__":
    main(DEBUG=False)
