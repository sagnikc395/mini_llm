"""A word-level tokenizer, built by hand before switching to BPE.

Nothing in the training path uses this -- the model's out_head has 50257
outputs and the dataloaders encode with GPT-2 BPE, so `tiktoken` is what the
rest of the codebase reaches for. This is kept because it makes the thing a
tokenizer actually does (string <-> int, plus the unknown-token problem that
BPE exists to solve) visible in about thirty lines.
"""

import re

END_OF_TEXT = "<|endoftext|>"
UNKNOWN = "<|unk|>"

# split on punctuation, em-dashes and whitespace, keeping the delimiters
_SPLIT_PATTERN = r'([,.:;?_!"()\']|--|\s)'


def tokenize(text: str) -> list[str]:
    return [item.strip() for item in re.split(_SPLIT_PATTERN, text) if item.strip()]


def build_vocab(text: str) -> dict[str, int]:
    """Map every distinct token in `text` to an id, plus the two specials."""
    tokens = sorted(set(tokenize(text)))
    tokens.extend([END_OF_TEXT, UNKNOWN])
    return {token: i for i, token in enumerate(tokens)}


class SimpleTokenizer:
    """Word-level encode/decode against a fixed vocabulary."""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        # anything the vocabulary has never seen collapses to <|unk|>, which is
        # exactly the information loss subword tokenizers avoid
        return [
            self.str_to_int.get(token, self.str_to_int[UNKNOWN])
            for token in tokenize(text)
        ]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        # undo the space the join put before punctuation
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)
