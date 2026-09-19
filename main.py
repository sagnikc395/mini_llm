from pathlib import Path

import tiktoken
import torch

from mini_llm.config import GPT_CONFIG_124M
from mini_llm.architecture.gpt_model import GPTModel
from mini_llm.pretraining.dataset_loader import create_dataloader_v1
from mini_llm.loss.calc_loss import calc_loss_loader
from mini_llm.pretraining.train_model_simple import train_model_simple


PROJECT_ROOT = Path(__file__).resolve().parent
# seed value
torch.manual_seed(123)


def main(DEBUG=False):
    # the_verdict.txt is only ~5k tokens, so a 1024-token context leaves the 10%
    # validation split with zero full-length windows -> empty loader -> nan loss.
    # 256 tokens is short enough that both splits yield batches.
    cfg = GPT_CONFIG_124M(context_length=256)

    # the data loaders tokenize with GPT-2 BPE, and the model's out_head has
    # cfg.vocab_size (50257) outputs, so sampling must use the same encoding.
    # SimpleTokenizer has a ~1160-word vocabulary built from the_verdict.txt and
    # would mis-map every id in both directions.
    tokenizer = tiktoken.get_encoding("gpt2")

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
        num_workers=0,
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
        for x, y in train_loader:
            print(x.shape, y.shape)
        print("\nValidation loader:")
        for x, y in val_loader:
            print(x.shape, y.shape)

    ## applying loss:
    # instantiate the model; eval mode disables dropout so the loss is deterministic
    model = GPTModel(cfg)

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"DEVICE TYPE: {device}")
    model.to(device)
    if DEBUG:
        model.eval()
        with torch.no_grad():
            # disable gradient tacking for efficiency because we are not training yet
            train_loss = calc_loss_loader(train_loader, model, device)
            # via the "device" setting, we ensure the data is loaded onto the same device as the LLM model
            val_loss = calc_loss_loader(val_loader, model, device)

        print(f"Training loss: {train_loss}")
        print(f"Validation loss: {val_loss}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,weight_decay=0.1
    )
    num_epochs = 10
    train_losses,val_losses, tokens_seen = train_model_simple(
        model,train_loader,val_loader,optimizer,device,
        num_epochs=num_epochs,eval_freq=5,eval_iter=5,start_context="Every effort moves you",tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main(DEBUG=False)
