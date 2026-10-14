import pandas as pd
import torch
from mini_llm.finetuning.download_dataset import data_file_path
from torch.utils.data import Dataset


def load_spam_dataset(path=data_file_path):
    return pd.read_csv(path, sep="\t", header=None, names=["Label", "Text"])


# undersample and create a balance dataset
def create_balanced_dataset(df):
    # count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]
    # random sampling of ham instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    # combine hamd subset with spam
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df


# random split function to divide the dataset into 2 parts


def random_split(df, train_frac, validation_frac):
    # random sampling with shuffling the entire dataframe
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # calculate the split index
    train_end = int(len(df) * train_frac)

    validation_end = train_end + int(len(df) * validation_frac)

    # now split the dataframe
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        # GPT-2 endoftext token id
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # here truncate the sequences if they are longer than max length

            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        # pads the sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
