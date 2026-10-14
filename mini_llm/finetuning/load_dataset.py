import pandas as pd

from mini_llm.finetuning.download_dataset import data_file_path


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
