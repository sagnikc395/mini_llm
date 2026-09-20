# CLI for the mini LLM: train a GPT from scratch on the_verdict.txt, or load
# OpenAI's pretrained GPT-2 weights and sample from them.

from datetime import datetime
from pathlib import Path

import click
import tiktoken
import torch

from mini_llm.architecture.gpt_model import GPTModel
from mini_llm.config import GPT_CONFIG_124M
from mini_llm.generate import (
    generate,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)
from mini_llm.loss.calc_loss import calc_loss_loader
from mini_llm.loss.plot_loss import plot_losses
from mini_llm.pretraining.dataset_loader import create_dataloader_v1
from mini_llm.pretraining.load_pretrained_weights import (
    MODEL_CONFIGS,
    config_for_gpt2,
    load_weights_into_gpt,
)
from mini_llm.pretraining.train_model_simple import train_model_simple

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "the_verdict.txt"
MODELS_DIR = PROJECT_ROOT / "saved_models"
# where download_gpt_weights puts the OpenAI checkpoints
GPT2_DIR = PROJECT_ROOT / "gpt2"

# the_verdict.txt is only ~5k tokens, so a 1024-token context leaves the 10%
# validation split with zero full-length windows -> empty loader -> nan loss.
# 256 tokens is short enough that both splits yield batches.
CONTEXT_LENGTH = 256
TRAIN_RATIO = 0.9
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.1
START_CONTEXT = "Every effort moves you"

# seed value
torch.manual_seed(123)


def build_dataloaders(text_data, tokenizer, context_length, debug=False):
    """Split `text_data` by character count and wrap each half in a dataloader.

    The split is on characters rather than tokens because it happens before
    tokenization; the exact boundary does not matter for a single-book corpus.
    """
    click.echo(f"Characters: {len(text_data)}")
    click.echo(f"Tokens: {len(tokenizer.encode(text_data))}")

    split_idx = int(TRAIN_RATIO * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=BATCH_SIZE,
        max_length=context_length,
        stride=context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=BATCH_SIZE,
        max_length=context_length,
        stride=context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    if debug:
        click.echo("=====")
        click.echo("CHECKING if data loaders are created correctly")
        for name, loader in (("Train", train_loader), ("Validation", val_loader)):
            click.echo(f"{name} loader:")
            for x, y in loader:
                click.echo(f"{x.shape} {y.shape}")

    return train_loader, val_loader


def get_device():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    click.echo(f"DEVICE TYPE: {device}")
    return device


def print_initial_losses(model, train_loader, val_loader, device):
    """Report the untrained model's loss (~log(vocab_size) if all is well)."""
    model.eval()
    # disable gradient tracking for efficiency because we are not training yet
    with torch.no_grad():
        # via the "device" setting, we ensure the data is loaded onto the same
        # device as the LLM model
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    click.echo(f"Training loss: {train_loss}")
    click.echo(f"Validation loss: {val_loss}")


def sample(
    model,
    tokenizer,
    context_length,
    prompt=START_CONTEXT,
    max_new_tokens=25,
    device=None,
    **generate_kwargs,
):
    """Generate from `model` and print the decoded text.

    Without `generate_kwargs` this uses the plain greedy loop; passing top_k or
    temperature routes to `generate` instead, which samples. eval() mode keeps
    dropout out of the sampled distribution.
    """
    model.eval()
    idx = text_to_token_ids(prompt, tokenizer)
    if device is not None:
        # the prompt ids have to sit on the same device as the model
        idx = idx.to(device)

    sampler = generate if generate_kwargs else generate_text_simple
    token_ids = sampler(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_length,
        **generate_kwargs,
    )
    click.echo(f"Output text:\n{token_ids_to_text(token_ids, tokenizer)}")


def save_checkpoint(model, optimizer):
    """Save model and optimizer state so a run can be resumed or sampled later.

    The timestamp goes down to seconds so several runs on the same day don't
    overwrite each other.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_path = MODELS_DIR / f"gpt_model_{timestamp}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )
    click.echo(f"Model saved to {save_path}")
    return save_path


@click.group()
def cli():
    """Train a small GPT from scratch, or sample from pretrained GPT-2."""


@cli.command()
@click.option(
    "--epochs", default=NUM_EPOCHS, show_default=True, help="training epochs to run"
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="print dataloader batch shapes and the untrained model's loss",
)
@click.option(
    "--save/--no-save", default=True, show_default=True, help="write a checkpoint"
)
def train(epochs, debug, save):
    """Train a GPT from scratch on the_verdict.txt."""
    cfg = GPT_CONFIG_124M(context_length=CONTEXT_LENGTH)

    # the data loaders tokenize with GPT-2 BPE, and the model's out_head has
    # cfg.vocab_size (50257) outputs, so sampling must use the same encoding.
    # SimpleTokenizer has a ~1160-word vocabulary built from the_verdict.txt and
    # would mis-map every id in both directions.
    tokenizer = tiktoken.get_encoding("gpt2")
    text_data = DATA_FILE.read_text(encoding="utf-8")

    train_loader, val_loader = build_dataloaders(
        text_data, tokenizer, cfg.context_length, debug=debug
    )

    model = GPTModel(cfg)
    # what the model produces before any training, for comparison
    sample(model, tokenizer, cfg.context_length)

    device = get_device()
    model.to(device)
    if debug:
        print_initial_losses(model, train_loader, val_loader, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=START_CONTEXT,
        tokenizer=tokenizer,
    )

    # plot the losses and save it under plots/
    epochs_seen = torch.linspace(0, epochs, len(train_losses))
    plot_losses(epochs_seen, tokens_seen, train_losses, val_losses)

    sample(
        model,
        tokenizer,
        cfg.context_length,
        max_new_tokens=15,
        device=device,
        top_k=25,
        temperature=1.4,
    )

    if save:
        save_checkpoint(model, optimizer)


@cli.command()
@click.option(
    "--model-size",
    default="124M",
    type=click.Choice(tuple(MODEL_CONFIGS)),
    show_default=True,
    help="which pretrained GPT-2 checkpoint to load",
)
@click.option(
    "--models-dir",
    default=GPT2_DIR,
    type=click.Path(file_okay=False, path_type=Path),
    show_default=True,
    help="where the downloaded checkpoints live",
)
@click.option("--prompt", default=START_CONTEXT, show_default=True)
@click.option("--max-new-tokens", default=25, show_default=True)
@click.option(
    "--top-k", default=50, show_default=True, help="0 disables top-k filtering"
)
@click.option(
    "--temperature",
    default=1.0,
    show_default=True,
    help="0 makes generation greedy/deterministic",
)
def pretrained(model_size, models_dir, prompt, max_new_tokens, top_k, temperature):
    """Sample from OpenAI's pretrained GPT-2 weights, downloading them if needed."""
    # imported here rather than at module scope because it pulls in tensorflow,
    # which the train command has no use for and which is slow to import
    from mini_llm.pretraining.download_gpt_weights import download_and_load_gpt2

    _, params = download_and_load_gpt2(model_size, models_dir)

    cfg = config_for_gpt2(model_size)
    model = GPTModel(cfg)
    load_weights_into_gpt(model, params)

    device = get_device()
    model.to(device)

    sample(
        model,
        tokenizer=tiktoken.get_encoding("gpt2"),
        context_length=cfg.context_length,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        top_k=top_k or None,
        temperature=temperature,
    )


if __name__ == "__main__":
    cli()
