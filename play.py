import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.stats import t
from tqdm import tqdm


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return round(100 * m, 3), round(100 * h, 3)


def get_ic(model, dataset):
    stat_path = f"resource/stat/{model}_{dataset}.stat"
    stat_df = pd.read_csv(stat_path, header=0, sep="\t")
    print(mean_confidence_interval(
        stat_df["Mac-F1"].tolist()
    ))

def show_cls (dataset):
    print(dataset)
    with open(f"resource/dataset/{dataset}/samples.pkl", "rb") as samples_file:
        samples_df = pd.DataFrame(pickle.load(samples_file))
        print(samples_df["cls"].unique())
        print(samples_df["cls"].nunique())

def _load_ids(self, ids_path):
    with open(ids_path, "rb") as ids_file:
        return pickle.load(ids_file)

def prediction(model, dataset, fold_idx):

    predictions_paths = sorted(
        Path(f"resource/prediction/{model}_{dataset}S/fold_{fold_idx}/").glob("*.prd")
    )

    with open(f"resource/dataset/{dataset}/fold_{fold_idx}/test.pkl", "rb") as ids_file:
        test_ids = pickle.load(ids_file)

    predictions = []
    for path in tqdm(predictions_paths, desc="Loading predictions"):
        predictions.extend( # only eval over test split
            filter(lambda prediction: prediction["idx"] in test_ids, torch.load(path))
        )

    return predictions


if __name__ == '__main__':
    prediction()
