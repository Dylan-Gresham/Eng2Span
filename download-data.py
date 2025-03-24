import os
import shutil

import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

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
    (
        "https://www.kaggle.com/api/v1/datasets/download/tejasurya/eng-spanish",
        "eng-spanish",
    ),
]

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
    with open(f"./data/{ouput_file_name}.data", "wb") as f:
        f.write(response.content)
