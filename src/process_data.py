import ast
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.download_data import download_data


def process_helsinki(path):
    """Processes the Helsinki-NLP/opus_books dataset to have en and es column."""
    df = pd.read_csv(path)
    df["translation"] = df["translation"].apply(ast.literal_eval)
    df[["en", "es"]] = df["translation"].apply(pd.Series)
    df.drop(columns=["translation", "id", "split"], inplace=True)

    return df[~df["en"].str.startswith("Source:")]


def process_englishspanish(path):
    """
    Processes the Kaggle 'English-Spanish Translation Dataset' from Lonnie.

    This dataset will have three columns, en and es.
    """
    df = pd.read_csv(path)
    df.rename(
        columns={
            "english": "en",
            "spanish": "es",
        },
        inplace=True,
    )

    return df


def process_google(path):
    """Processes the google/wmt24pp dataset to have en and es column."""
    df = pd.read_csv(path)
    df = df[~df["is_bad_source"]]  # Only use translations from good sources
    df.drop(
        columns=[
            "lp",
            "domain",
            "document_id",
            "segment_id",
            "is_bad_source",
            "original_target",
            "split",
        ],
        inplace=True,
    )
    df.rename(columns={"source": "en", "target": "es"}, inplace=True)

    return df


def process_kde4(path):
    """Processes the Helsinki-NLP/kde4 dataset to have en, and es column."""
    df = pd.read_csv(path)
    df["translation"] = df["translation"].apply(ast.literal_eval)
    df[["en", "es"]] = df["translation"].apply(pd.Series)
    df.drop(columns=["translation", "id", "split"], inplace=True)

    return df


def process_data_files():
    """
    Helper function for `process_all_data_files`

    Loads each DataFrame in the data directory and process it accordingly.
    """
    dfs = []
    data_files = [
        f for f in os.listdir("./data") if os.path.isfile(os.path.join("./data", f))
    ]
    for data_file in tqdm(data_files, desc="Processing data files"):
        data_path = f"./data/{data_file}"

        if data_file == "Helsinki-NLP-opus_books.data":
            dfs.append(process_helsinki(data_path))
        elif data_file == "englishspanish-translation-dataset.data":
            dfs.append(process_englishspanish(data_path))
        elif data_file == "google-wmt24pp.data":
            dfs.append(process_google(data_path))
        elif data_file == "kde4.data":
            dfs.append(process_kde4(data_path))
        else:
            print(f"Unknown data file '{data_file}'")

    cols = ["en", "es"]
    dfs = [df.reindex(columns=cols) for df in dfs]
    big_df = pd.concat(dfs, ignore_index=True)
    big_df.drop_duplicates(inplace=True)
    big_df.dropna(how="any", ignore_index=True, inplace=True)

    # Create 3 splits for train, validation, test
    #
    # Train = 60%
    # Validation = 20%
    # Test = 20%
    train, test = train_test_split(big_df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    # Recombine the splits
    cols = ["en", "es", "split"]
    dfs = [df.reindex(columns=cols) for df in [train, val, test]]
    big_df = pd.concat(dfs, ignore_index=True)

    print(big_df.split.value_counts())

    return big_df


def process_all_data_files():
    """
    Downloads all data files and then processes them into a single DataFrame.

    The resulting DataFrame will have three columns:

    - 'en': The source English
    - 'es': The expected Spanish translation
    - 'split': Which of the train/validation/test splits that instance belongs to
    """
    download_data()

    df = process_data_files()
    df.to_csv("./data/combined.data", index=False)


if __name__ == "__main__":
    process_all_data_files()
