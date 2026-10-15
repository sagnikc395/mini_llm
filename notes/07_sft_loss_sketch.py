"""The supervised-finetuning loss, written out.

From the RLHF book rather than Raschka's. Nothing in mini_llm/ calls this yet;
it is here so the shape of the computation is written down somewhere. The one
thing that distinguishes it from pretraining loss is `completion_mask`: during
SFT only the tokens the model was supposed to *produce* are scored, so the
prompt contributes nothing to the gradient.
"""

import torch


def sft_loss(model, input_ids, completion_mask):
    """Mean negative log probability over the completion tokens only.

    Args:
        input_ids: (B, L) prompt and completion concatenated.
        completion_mask: (B, L), 1 where the token is part of the completion.
    """
    # (B, L, V): a score for every vocab token at every position
    logits = model(input_ids).logits

    # drop the last position -- its prediction has no next token to score --
    # and shift the labels left, so position i predicts token i+1
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    # the mask shifts the same way so it still lines up with the labels
    mask = completion_mask[:, 1:]

    # normalize over the vocabulary: (B, L-1, V)
    log_probs = logits.log_softmax(dim=-1)
    # pick out the log probability of the token that actually came next
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # average over completion tokens only; the prompt is masked out
    return -(token_log_probs * mask).sum() / mask.sum()
