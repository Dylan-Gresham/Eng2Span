import os
import shutil
from zipfile import ZipFile

import pandas as pd
import requests
from datasets import load_dataset
from tqdm.auto import tqdm

# Define Dataset links/names
HF_DATASETS = [
    ("google/wmt24pp", "en-es_MX"),
    ("Helsinki-NLP/opus_books", "en-es"),
    ("kde4", None),
]
KAGGLE_DATASETS = [
    (
        "https://www.kaggle.com/api/v1/datasets/download/lonnieqin/englishspanish-translation-dataset",
        "englishspanish-translation-dataset",
    ),
    # This dataset is the same as the one above but would require slightly more processing
    # (
    #     "https://www.kaggle.com/api/v1/datasets/download/tejasurya/eng-spanish",
    #     "eng-spanish",
    # ),
]


def delete_existing_data_dir():
    """Removes the existing data directory to ensure data consistency."""
    # Delete all existing files
    try:
        if os.path.exists("./data"):
            print("Data directory already exists. Removing.")
            # Removing existing files and directory
            shutil.rmtree("./data")
            os.mkdir("./data")
            print("Data directory has been deleted and reconstructed.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except OSError as e:
        print(f"Error: {e}")
        exit(2)


def download_data():
    """
    Downloads all of the datasets defined in the HF_DATASETS and KAGGLE_DATASETS constants.

    All downloaded files will be placed in the "./data" directory and have a '.data' file extension.
    """
    delete_existing_data_dir()

    # Download Hugging Face datasets
    for dataset_name, subset in tqdm(
        HF_DATASETS, desc="Downloading datasets from Hugging Face", leave=True
    ):
        # Load dataset from HuggingFace
        if subset is None:
            dataset = load_dataset(
                dataset_name,
                lang1="en",
                lang2="es",
                trust_remote_code=True,
            )
        else:
            dataset = load_dataset(dataset_name, subset)

        # Combine splits into one DataFrame
        df_list = []
        for split_name, split_data in dataset.items():
            df = pd.DataFrame(split_data)
            df["split"] = split_name
            df_list.append(df)

        full_df = pd.concat(df_list, ignore_index=True)

        # Save to CSV
        full_df.to_csv(f"./data/{dataset_name.replace('/', '-')}.data", index=False)

    # Download Kaggle datasets
    for dataset_url, ouput_file_name in tqdm(
        KAGGLE_DATASETS, desc="Downloading datasets from Kaggle", leave=True
    ):
        response = requests.get(dataset_url, allow_redirects=True)
        with open(f"./data/{ouput_file_name}.zip", "wb") as f:
            f.write(response.content)

        with ZipFile(f"./data/{ouput_file_name}.zip", "r") as zObject:
            if dataset_url.endswith("t"):
                zObject.extract("data.csv", "./data/")

                tmp_df = pd.read_csv("./data/data.csv")
                tmp_df.to_csv(f"./data/{ouput_file_name}.data", index=False)
            elif dataset_url.endswith("h"):
                zObject.extract("spa.txt", "./data/")

                tmp_df = pd.read_csv(
                    "./data/spa.txt",
                    delimiter="\t",
                    header=None,
                    names=["en", "es", "License"],
                )
                tmp_df.to_csv(f"./data/{ouput_file_name}.data", index=False)
            else:
                print("Unknown URL downloaded.")
                zObject.close()
                os.remove(f"./data/{ouput_file_name}.zip")
                continue

            zObject.close()
            if dataset_url.endswith("t"):
                try:
                    os.remove("./data/data.csv")
                    os.remove("./data/englishspanish-translation-dataset.zip")
                except OSError as e:
                    print(f"Error removing temporary files: {e}")
            elif dataset_url.endswith("h"):
                try:
                    os.remove("./data/spa.txt")
                    os.remove("./data/eng-spanish.zip")
                except OSError as e:
                    print(f"Error removing temporary files: {e}")


if __name__ == "__main__":
    download_data()
