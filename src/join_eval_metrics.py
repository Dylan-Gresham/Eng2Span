import os.path as op

import pandas as pd


def combine_dataframes():
    dfs = []
    if op.exists("data/m2m_baseline.csv"):
        dfs.append(pd.read_csv("data/m2m_baseline.csv", index_col=0))
    if op.exists("data/m2m_ft.csv"):
        dfs.append(pd.read_csv("data/m2m_ft.csv", index_col=0))

    if op.exists("data/mbart_baseline.csv"):
        dfs.append(pd.read_csv("data/mbart_baseline.csv", index_col=0))
    if op.exists("data/mbart_ft.csv"):
        dfs.append(pd.read_csv("data/mbart_ft.csv", index_col=0))

    if op.exists("data/nllb_baseline.csv"):
        dfs.append(pd.read_csv("data/nllb_baseline.csv", index_col=0))
    if op.exists("data/nllb_ft.csv"):
        dfs.append(pd.read_csv("data/nllb_ft.csv", index_col=0))

    if op.exists("data/opus_baseline.csv"):
        dfs.append(pd.read_csv("data/opus_baseline.csv", index_col=0))
    if op.exists("data/opus_ft.csv"):
        dfs.append(pd.read_csv("data/opus_ft.csv", index_col=0))

    return pd.concat(dfs)


if __name__ == "__main__":
    combined = combine_dataframes()
    combined.to_csv("benchmarks.csv")
