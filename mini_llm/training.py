"""The training loop and the loss functions it reports."""

import torch
import torch.nn.functional as F

from mini_llm.generate import generate, text_to_token_ids, token_ids_to_text


def get_device(verbose: bool = True) -> torch.device:
    """Pick the best available device: Apple MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"device: {device}")
    return device


def calc_loss_batch(input_batch, target_batch, model, device):
    """Cross entropy over one batch, with both tensors moved onto `device`."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    # flatten (batch, tokens, vocab) -> (batch*tokens, vocab) so every position
    # is just another classification example
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Average loss over `num_batches` batches (all of them when None)."""
    if len(data_loader) == 0:
        return float("nan")

    num_batches = (
        len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    )

    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, train_loader, val_loader, device, eval_iter):
    """Train and validation loss on a few batches, with the model in eval mode."""
    model.eval()
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def sample_text(model, tokenizer, prompt, device=None, max_new_tokens=50, **kwargs):
    """Generate from `model` and return the decoded continuation.

    Runs under eval() so dropout does not perturb the sampled distribution,
    then restores whatever mode the model was in.
    """
    was_training = model.training
    model.eval()

    idx = text_to_token_ids(prompt, tokenizer)
    if device is not None:
        idx = idx.to(device)

    token_ids = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=model.context_length,
        **kwargs,
    )

    if was_training:
        model.train()
    return token_ids_to_text(token_ids, tokenizer)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    tokenizer,
    start_context,
    eval_freq=5,
    eval_iter=5,
):
    """Train `model`, returning `(train_losses, val_losses, tokens_seen)`.

    Losses are recorded every `eval_freq` optimizer steps rather than every
    step, so the plot stays readable and evaluation does not dominate runtime.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"epoch {epoch + 1} (step {global_step:06d}): "
                    f"train loss {train_loss:.3f}, val loss {val_loss:.3f}"
                )

        # one sample per epoch, as a human-readable check on the numbers above
        preview = sample_text(model, tokenizer, start_context, device=device)
        print(preview.replace("\n", " "))

    return train_losses, val_losses, track_tokens_seen
