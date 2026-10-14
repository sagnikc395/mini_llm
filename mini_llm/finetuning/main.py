from mini_llm.finetuning.download_dataset import (
    download_and_unzip_spam_data,
    url,
    zip_path,
    extracted_path,
    data_file_path,
)
from mini_llm.finetuning.load_dataset import load_spam_dataset, random_split


if __name__ == "__main__":
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    df = load_spam_dataset()
    # print(df)
    # test size implied to be 0.2 as the remainder
    train_df, validation_df, test_df = random_split(df, 0.7, 0.1)

    # save the dataset as CSV
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)
