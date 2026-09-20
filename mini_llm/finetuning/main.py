from mini_llm.finetuning.download_dataset import (
    download_and_unzip_spam_data,
    url,
    zip_path,
    extracted_path,
    data_file_path,
)
from mini_llm.finetuning.load_dataset import load_spam_dataset

if __name__ =='__main__':
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    df = load_spam_dataset()
    print(df)
