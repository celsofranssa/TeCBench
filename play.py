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
    return round(m, 2), round(h, 2)


def get_ic(model, dataset):
    stat_path = f"resource/stat/{model}_{dataset}.stat"
    stat_df = pd.read_csv(stat_path, header=0, sep="\t")
    print(f"Model: {model} - Dataset: {dataset}")
    print(f"Mic-F1: {mean_confidence_interval(stat_df['Mic-F1'].tolist())}")
    print(f"Mac-F1: {mean_confidence_interval(stat_df['Mac-F1'].tolist())}")
    print(f"Wei-F1: {mean_confidence_interval(stat_df['Wei-F1'].tolist())}")


def show_cls(dataset):
    print(dataset)
    with open(f"resource/dataset/{dataset}/samples.pkl", "rb") as samples_file:
        samples_df = pd.DataFrame(pickle.load(samples_file))
        print(samples_df["cls"].unique())
        print(samples_df["cls"].nunique())


def _load_ids(self, ids_path):
    with open(ids_path, "rb") as ids_file:
        return pickle.load(ids_file)


# def read_prediction(model, dataset, fold_idx):
#
#     predictions_paths = sorted(
#         Path(f"resource/prediction/{model}_{dataset}S/fold_{fold_idx}/").glob("*.prd")
#     )
#
#     with open(f"resource/dataset/{dataset}/fold_{fold_idx}/test.pkl", "rb") as ids_file:
#         test_ids = pickle.load(ids_file)
#
#     predictions = []
#     for path in tqdm(predictions_paths, desc="Loading predictions"):
#         predictions.extend( # only eval over test split
#             filter(lambda prediction: prediction["idx"] in test_ids, torch.load(path))
#         )
#
#     return predictions

def read_prediction(model, dataset, fold_idx, split):
    """Read model predictions.
    :param str model: model name (BERT, BERTimbau or LaBSE).
    :param dataset: dataset name (such as DIARIOS).
    :param fold_idx: the fold index (0,1,2,3 or 4).
    :param split: the split name (train, val or test).
    """

    predictions_paths = sorted(
        Path(f"resource/prediction/{model}_{dataset}S/fold_{fold_idx}/").glob("*.prd")
    )

    with open(f"resource/dataset/{dataset}/fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
        split_ids = pickle.load(ids_file)

    predictions = []
    for path in tqdm(predictions_paths, desc="Loading predictions"):
        predictions.extend(  # only eval over test split
            filter(lambda prediction: prediction["idx"] in split_ids, torch.load(path))
        )

    return predictions


def inspect_dataset(dataset):
    with open(f"resource/dataset/{dataset}/samples.pkl", "rb") as samples_fle:
        s = pickle.load(samples_fle)
    print(len(s))

def cls_report(model, dataset):
    report_path = f"resource/stat/{model}_{dataset}.rpt"
    with open(report_path, "rb") as rpt_file:
        rpt = pickle.load(rpt_file)
    # print(rpt)
    p, r, f = {}, {}, {}
    for cls in ["lei", "licitação", "orçamento", "pessoal"]:
        p[cls] = []
        r[cls] = []
        f[cls] = []

    for fold_idx in range(5):
        for cls in ["lei", "licitação", "orçamento", "pessoal"]:

            p[cls].append(rpt[fold_idx][cls]["precision"])
            r[cls].append(rpt[fold_idx][cls]["recall"])
            f[cls].append(rpt[fold_idx][cls]["f1-score"])

    for cls in ["lei", "licitação", "orçamento", "pessoal"]:
        print(cls)
        print(f"precision: {mean_confidence_interval(p[cls])}")
        print(f"recall: {mean_confidence_interval(r[cls])}")
        print(f"f1: {mean_confidence_interval(f[cls])}")




if __name__ == '__main__':
    # cls_report("BERT", "DIARIOS")
    # with open(f"resource/dataset/DIARIOS/Extended/BERTimbau_0/samples.pkl", "rb") as samples_fle:
    #     s = pickle.load(samples_fle)
    # print(len(s))
    get_ic("BERT", "DIARIOS")
    # #read_prediction(model="BERTimbau", dataset="DIARIOS", fold_idx=0, split="test")
    # # inspect_dataset("DIARIOS")

