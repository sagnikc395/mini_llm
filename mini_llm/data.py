"""The pretraining corpus: download it, slice it into windows, batch it."""

import urllib.request
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from mini_llm.paths import VERDICT_FILE

VERDICT_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch02/01_main-chapter-code/the-verdict.txt"
)


def download_verdict(path: Path = VERDICT_FILE) -> Path:
    """Fetch the-verdict.txt if it isn't already on disk, and return its path."""
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(VERDICT_URL, path)
    return path


def load_verdict(path=VERDICT_FILE) -> str:
    """Read the corpus, downloading it first if needed."""
    return download_verdict(path).read_text(encoding="utf-8")


class GPTDataset(Dataset):
    """Sliding windows over one long token stream.

    Each item is `(input_chunk, target_chunk)` where the target is the input
    shifted one token to the left -- next-token prediction, spelled out.
    `stride` controls how far the window moves, so stride == max_length gives
    non-overlapping chunks and anything smaller overlaps them.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(txt)
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    tokenizer=None,
):
    """Wrap `txt` in a GPTDataset and hand back a DataLoader over it.

    drop_last throws away a final short batch, which otherwise shows up as a
    loss spike during training.
    """
    tokenizer = tokenizer or tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def train_val_loaders(
    text, context_length, batch_size, train_ratio=0.9, tokenizer=None
):
    """Split `text` in two and return `(train_loader, val_loader)`.

    The split is by character count because it happens before tokenization; for
    a single-book corpus the exact boundary does not matter.
    """
    split_idx = int(train_ratio * len(text))

    train_loader = create_dataloader(
        text[:split_idx],
        batch_size=batch_size,
        max_length=context_length,
        stride=context_length,
        shuffle=True,
        drop_last=True,
        tokenizer=tokenizer,
    )
    val_loader = create_dataloader(
        text[split_idx:],
        batch_size=batch_size,
        max_length=context_length,
        stride=context_length,
        shuffle=False,
        drop_last=False,
        tokenizer=tokenizer,
    )
    return train_loader, val_loader
