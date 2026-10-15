"""Size up an untrained GPTModel, then compute its loss by hand.

Two things worth seeing: where the 124M parameters actually are (weight tying
accounts for the gap), and that `F.cross_entropy` is exactly the negative mean
log probability of the correct next token.
"""

import tiktoken
import torch
import torch.nn.functional as F

from mini_llm.config import GPTConfig
from mini_llm.generate import text_to_token_ids, token_ids_to_text
from mini_llm.model import GPTModel
from mini_llm.training import sample_text


def parameter_budget(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total:,}")

    # GPT-2 ties out_head to tok_emb, so the published "124M" count does not
    # include the output head -- the same 50257x768 matrix twice over
    untied = total - sum(p.numel() for p in model.out_head.parameters())
    print(f"minus the tied output head: {untied:,}")
    print(
        f"tok_emb {tuple(model.tok_emb.weight.shape)} "
        f"== out_head {tuple(model.out_head.weight.shape)}"
    )

    # float32 is 4 bytes per parameter
    print(f"float32 footprint: {total * 4 / 1024**2:.2f} MB")


def cross_entropy_by_hand(model, tokenizer):
    """The loss, computed twice: step by step, then with F.cross_entropy."""
    inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
    targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    print(f"\nprobabilities: {tuple(probas.shape)}  (batch, tokens, vocab)")

    predicted = torch.argmax(probas, dim=-1)
    print(f"targets:  {token_ids_to_text(targets[0], tokenizer)!r}")
    print(
        f"predicted:{token_ids_to_text(predicted[0], tokenizer)!r}  (untrained, so: noise)"
    )

    # pick out the probability the model assigned to each *correct* token
    target_probas = torch.cat(
        [
            probas[i, range(targets.shape[1]), targets[i]]
            for i in range(targets.shape[0])
        ]
    )
    print(f"\nprobability of each correct token: {target_probas}")

    # log, mean, negate -- that is the whole loss
    manual_loss = -torch.mean(torch.log(target_probas))
    builtin_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    print(f"by hand:          {manual_loss:.4f}")
    print(f"F.cross_entropy:  {builtin_loss:.4f}")
    # an untrained model spreads mass evenly over the vocabulary, so the loss
    # starts near log(50257)
    print(f"log(vocab_size):  {torch.log(torch.tensor(50257.0)):.4f}")


def main():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPTConfig())

    batch = torch.stack(
        [
            text_to_token_ids("Every effort moves you", tokenizer).squeeze(0),
            text_to_token_ids("Every day holds a", tokenizer).squeeze(0),
        ]
    )
    print(f"input batch: {tuple(batch.shape)} -> logits {tuple(model(batch).shape)}")

    parameter_budget(model)

    print(
        f"\nuntrained output: {sample_text(model, tokenizer, 'Hello, I am', max_new_tokens=6)!r}"
    )

    cross_entropy_by_hand(model, tokenizer)


if __name__ == "__main__":
    main()
