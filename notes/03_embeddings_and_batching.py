"""What comes out of the dataloader, and how it becomes model input.

Token ids -> token embeddings, plus a positional embedding per slot, summed
into the tensor the first transformer block actually sees.
"""

import torch

from mini_llm.data import create_dataloader, load_verdict

VOCAB_SIZE = 50257
OUTPUT_DIM = 256
MAX_LENGTH = 4


def main():
    raw_text = load_verdict()

    # stride 1 overlaps the windows heavily; stride == max_length does not
    overlapping = create_dataloader(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    print(f"stride 1, first batch (input, target): {next(iter(overlapping))}")

    loader = create_dataloader(
        raw_text, batch_size=8, max_length=MAX_LENGTH, stride=MAX_LENGTH, shuffle=False
    )
    inputs, targets = next(iter(loader))
    print(f"\ninputs:\n{inputs}")
    print(f"targets (the same thing shifted one left):\n{targets}")

    # one row of a (50257, 256) table per token id
    token_embedding_layer = torch.nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
    token_embeddings = token_embedding_layer(inputs)
    print(f"\ntoken embeddings: {tuple(token_embeddings.shape)}")

    # absolute positional embeddings: one row per *slot*, not per token, so the
    # same (4, 256) tensor is added to every sequence in the batch
    pos_embedding_layer = torch.nn.Embedding(MAX_LENGTH, OUTPUT_DIM)
    pos_embeddings = pos_embedding_layer(torch.arange(MAX_LENGTH))
    print(f"positional embeddings: {tuple(pos_embeddings.shape)}")

    # broadcasting adds the (4, 256) positions across all 8 sequences
    print(f"model input: {tuple((token_embeddings + pos_embeddings).shape)}")


if __name__ == "__main__":
    main()
