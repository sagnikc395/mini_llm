"""The one entry point: `uv run python main.py <command>`.

This module is orchestration only -- argument parsing, wiring the pieces
together, printing. Anything reusable belongs in the modules it imports.
"""

from datetime import datetime
from pathlib import Path

import click
import tiktoken
import torch

from mini_llm.config import GPT2_SIZES, GPTConfig, gpt2_config
from mini_llm.data import load_verdict, train_val_loaders
from mini_llm.model import GPTModel
from mini_llm.paths import GPT2_DIR, MODELS_DIR
from mini_llm.plotting import plot_losses
from mini_llm.training import calc_loss_loader, get_device, sample_text, train_model

# Training hyperparameters, in one place so a run's settings are visible at a
# glance rather than scattered across call sites.
#
# the_verdict.txt is only ~5k tokens, so a 1024-token context leaves the 10%
# validation split with zero full-length windows -> empty loader -> nan loss.
# 256 is short enough that both splits actually yield batches.
CONTEXT_LENGTH = 256
TRAIN_RATIO = 0.9
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 0.1
START_CONTEXT = "Every effort moves you"
SEED = 123


def save_checkpoint(model, optimizer) -> Path:
    """Write model + optimizer state so a run can be resumed or sampled later."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    path = MODELS_DIR / f"gpt_model_{timestamp}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    click.echo(f"model saved to {path}")
    return path


@click.group()
def cli():
    """Train a small GPT from scratch, or sample from pretrained GPT-2."""


@cli.command()
@click.option("--epochs", default=NUM_EPOCHS, show_default=True, help="epochs to run")
@click.option(
    "--debug/--no-debug",
    default=False,
    help="also report batch shapes and the untrained model's loss",
)
@click.option(
    "--save/--no-save", default=True, show_default=True, help="write a checkpoint"
)
def train(epochs, debug, save):
    """Train a GPT from scratch on data/the_verdict.txt."""
    torch.manual_seed(SEED)
    cfg = GPTConfig(context_length=CONTEXT_LENGTH)

    # the dataloaders tokenize with GPT-2 BPE and out_head has cfg.vocab_size
    # (50257) outputs, so sampling has to use the same encoding. SimpleTokenizer
    # would mis-map every id in both directions.
    tokenizer = tiktoken.get_encoding("gpt2")
    text = load_verdict()
    click.echo(f"{len(text)} characters, {len(tokenizer.encode(text))} tokens")

    train_loader, val_loader = train_val_loaders(
        text, cfg.context_length, BATCH_SIZE, TRAIN_RATIO, tokenizer
    )

    model = GPTModel(cfg)
    device = get_device()
    model.to(device)

    if debug:
        for name, loader in (("train", train_loader), ("validation", val_loader)):
            shapes = [tuple(x.shape) for x, _ in loader]
            click.echo(f"{name} loader: {len(shapes)} batches of {shapes[0]}")
        # an untrained model should sit near log(vocab_size) ~= 10.8
        with torch.no_grad():
            click.echo(
                f"initial train loss: {calc_loss_loader(train_loader, model, device):.3f}"
            )
            click.echo(
                f"initial val loss:   {calc_loss_loader(val_loader, model, device):.3f}"
            )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_losses, val_losses, tokens_seen = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=epochs,
        tokenizer=tokenizer,
        start_context=START_CONTEXT,
    )

    plot_losses(
        torch.linspace(0, epochs, len(train_losses)),
        tokens_seen,
        train_losses,
        val_losses,
    )

    text_out = sample_text(
        model,
        tokenizer,
        START_CONTEXT,
        device=device,
        max_new_tokens=15,
        top_k=25,
        temperature=1.4,
    )
    click.echo(f"\n{text_out}")

    if save:
        save_checkpoint(model, optimizer)


@cli.command()
@click.option(
    "--size",
    default="124M",
    type=click.Choice(tuple(GPT2_SIZES)),
    show_default=True,
    help="which pretrained GPT-2 checkpoint to use",
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
    "--temperature", default=1.0, show_default=True, help="0 makes generation greedy"
)
def pretrained(size, models_dir, prompt, max_new_tokens, top_k, temperature):
    """Sample from OpenAI's GPT-2 weights, downloading them if needed."""
    # imported here rather than at module scope: it pulls in tensorflow, which
    # the train command never touches and which is slow to import
    from mini_llm.gpt2_weights import download_and_load_gpt2, load_weights_into_gpt

    _, params = download_and_load_gpt2(size, models_dir)

    model = GPTModel(gpt2_config(size))
    load_weights_into_gpt(model, params)

    device = get_device()
    model.to(device)

    text_out = sample_text(
        model,
        tiktoken.get_encoding("gpt2"),
        prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        top_k=top_k or None,
        temperature=temperature,
    )
    click.echo(f"\n{text_out}")


@cli.command("download-weights")
@click.option(
    "--size", default="124M", type=click.Choice(tuple(GPT2_SIZES)), show_default=True
)
@click.option(
    "--models-dir",
    default=GPT2_DIR,
    type=click.Path(file_okay=False, path_type=Path),
    show_default=True,
)
def download_weights(size, models_dir):
    """Download OpenAI's original GPT-2 weights without running the model."""
    from mini_llm.gpt2_weights import download_and_load_gpt2

    settings, params = download_and_load_gpt2(size, models_dir)
    click.echo(f"settings: {settings}")
    click.echo(f"transformer blocks: {len(params['blocks'])}")
    click.echo(f"token embedding shape: {params['wte'].shape}")


@cli.command("spam-data")
@click.option("--batch-size", default=8, show_default=True)
def spam_data(batch_size):
    """Build the SMS spam train/validation/test splits and check the loaders."""
    from mini_llm.spam import build_dataloaders

    torch.manual_seed(SEED)
    train_loader, val_loader, test_loader = build_dataloaders(batch_size=batch_size)

    for name, loader in (
        ("train", train_loader),
        ("validation", val_loader),
        ("test", test_loader),
    ):
        inputs, labels = next(iter(loader))
        click.echo(
            f"{name}: {len(loader)} batches, inputs {tuple(inputs.shape)}, labels {tuple(labels.shape)}"
        )


if __name__ == "__main__":
    cli()
