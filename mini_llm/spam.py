"""The SMS spam dataset, for finetuning the base model into a classifier.

End to end: download the UCI archive, balance it, split it into
train/validation/test csvs under data/spam/, and wrap each split in a
DataLoader of padded token ids.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from mini_llm.paths import SPAM_DIR

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

RAW_FILE = SPAM_DIR / "SMSSpamCollection.tsv"
SPLIT_FILES = {
    "train": SPAM_DIR / "train.csv",
    "validation": SPAM_DIR / "validation.csv",
    "test": SPAM_DIR / "test.csv",
}

# GPT-2's <|endoftext|> id, reused as the padding token
PAD_TOKEN_ID = 50256
# the dataset labels are strings; torch.tensor(..., dtype=torch.long) needs ints
LABEL_MAP = {"ham": 0, "spam": 1}


def download_spam_data(dest: Path = RAW_FILE) -> Path:
    """Download and unzip the SMS spam collection, returning the .tsv path."""
    if dest.exists():
        print(f"{dest} already exists, skipping download")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / "sms_spam_collection.zip"

    with urllib.request.urlopen(URL) as response:
        zip_path.write_bytes(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest.parent)

    # the archive member has no extension; rename it so the .tsv is obvious
    os.replace(dest.parent / "SMSSpamCollection", dest)
    zip_path.unlink()

    print(f"downloaded to {dest}")
    return dest


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Undersample ham down to the spam count.

    ~4800 ham vs ~750 spam means a classifier reaches 87% accuracy by always
    answering "ham", so accuracy would tell us nothing. Balancing *before* the
    split keeps train, validation and test all balanced.
    """
    num_spam = (df["Label"] == "spam").sum()
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])


def random_split(df: pd.DataFrame, train_frac: float, validation_frac: float):
    """Shuffle and cut `df` into train / validation / test (the remainder)."""
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    return df[:train_end], df[train_end:validation_end], df[validation_end:]


def prepare_splits(train_frac: float = 0.7, validation_frac: float = 0.1) -> dict:
    """Download, balance, split and write the three csvs; return their paths.

    Labels are mapped to ints here rather than in `SpamDataset.__getitem__`, so
    the csvs on disk are already model-ready and every split went through the
    same encoding.
    """
    raw = download_spam_data()
    df = pd.read_csv(raw, sep="\t", header=None, names=["Label", "Text"])

    balanced = balance_dataset(df)
    balanced["Label"] = balanced["Label"].map(LABEL_MAP)

    splits = random_split(balanced, train_frac, validation_frac)
    for (name, path), split in zip(SPLIT_FILES.items(), splits):
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=None)
        print(f"{name}: {len(split)} rows -> {path}")

    return dict(SPLIT_FILES)


class SpamDataset(Dataset):
    """One csv split as `(token_ids, label)` pairs, padded to a common length.

    Every sequence in a split has to be the same length to stack into a batch.
    `max_length` defaults to the longest message in this split; validation and
    test must be built with the *train* split's max_length so all three agree.
    """

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=PAD_TOKEN_ID):
        self.data = pd.read_csv(csv_file)
        encoded = [tokenizer.encode(text) for text in self.data["Text"]]

        self.max_length = max_length or max(len(ids) for ids in encoded)
        self.encoded_texts = [
            ids[: self.max_length] + [pad_token_id] * (self.max_length - len(ids))
            for ids in encoded
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_texts[index], dtype=torch.long),
            torch.tensor(self.data.iloc[index]["Label"], dtype=torch.long),
        )


def build_dataloaders(batch_size: int = 8, num_workers: int = 0, tokenizer=None):
    """Return `(train_loader, val_loader, test_loader)` over the written splits."""
    tokenizer = tokenizer or tiktoken.get_encoding("gpt2")
    paths = prepare_splits()

    train_dataset = SpamDataset(paths["train"], tokenizer)
    # the other two splits inherit the train split's padding length
    val_dataset = SpamDataset(paths["validation"], tokenizer, train_dataset.max_length)
    test_dataset = SpamDataset(paths["test"], tokenizer, train_dataset.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    eval_loaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        for dataset in (val_dataset, test_dataset)
    ]

    return (train_loader, *eval_loaders)
