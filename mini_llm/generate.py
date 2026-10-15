"""Turning text into token ids, sampling the next token, and back to text."""

import torch


def text_to_token_ids(text, tokenizer):
    """Encode `text` and add the batch dimension the model expects."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    """Drop the batch dimension and decode back to a string."""
    return tokenizer.decode(token_ids.squeeze(0).tolist())


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    context_size=None,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    """Autoregressively extend `idx` by `max_new_tokens` token ids.

    The defaults (temperature 0, no top-k) give plain greedy decoding: take the
    single highest-scoring token every step. Raising temperature or setting
    top_k switches to sampling instead.

    Call this under `model.eval()` -- in train mode dropout is live and quietly
    adds noise to the distribution you sample from.
    """
    context_size = context_size or model.context_length

    for _ in range(max_new_tokens):
        # the model can only see context_size tokens, so crop from the left
        logits = model(idx[:, -context_size:])
        # only the last position predicts the next token
        logits = logits[:, -1, :]

        if top_k is not None:
            # keep the top_k scores, push everything else to -inf so softmax
            # gives it zero probability
            top_logits, _ = torch.topk(logits, top_k)
            min_keep = top_logits[:, -1:]
            logits = torch.where(logits < min_keep, -torch.inf, logits)

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
