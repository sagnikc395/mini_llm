# random split function to divide the dataset into 2 parts

from pandas import DataFrame as df


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
