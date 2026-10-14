import tiktoken
from torch.utils.data import DataLoader
import torch

from mini_llm.finetuning.download_dataset import (
    download_and_unzip_spam_data,
    url,
    zip_path,
    extracted_path,
    data_file_path,
)
from mini_llm.finetuning.load_dataset import (
    load_spam_dataset,
    create_balanced_dataset,
    random_split,
    SpamDataset,
)
# create the training, validation and test set data loader that load the text messages and lables in batches of size 8


num_workers = 0
batch_size = 8
torch.manual_seed(123)


if __name__ == "__main__":
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    df = load_spam_dataset()
    # undersample ham so both classes are equally represented
    balanced_df = create_balanced_dataset(df)
    # the labels have to be integers for torch.tensor(..., dtype=torch.long)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    # test size implied to be 0.2 as the remainder
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    # save the dataset as CSV
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer,
    )

    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )

    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # test for dataloaders
    for input_batch, target_batch in train_loader:
        pass

    print(f"Input batch dimensions: {input_batch.shape}")
    print(f"Label batch dimensions: {target_batch.shape}")
