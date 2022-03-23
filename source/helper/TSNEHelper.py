import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from openTSNE import TSNE

class TSNEHelper:
    def __init__(self, params):
        self.params = params
        sns.set_theme(style="darkgrid")
        #sns.color_palette()

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            return pickle.load(ids_file)

    def load_predictions(self, fold_id):

        predictions_paths = sorted(
            Path(f"{self.params.prediction.dir}fold_{fold_id}/").glob("*.prd")
        )

        train_ids = self._load_ids(
            f"{self.params.data.dir}fold_{fold_id}/train.pkl"
        )

        test_ids = self._load_ids(
            f"{self.params.data.dir}fold_{fold_id}/test.pkl"
        )

        train_predictions = []
        test_predictions = []
        for path in tqdm(predictions_paths, desc="Loading predictions"):
            train_predictions.extend(  # only generates tsne over test split
                filter(lambda prediction: prediction["idx"] in train_ids, torch.load(path))
            )
            test_predictions.extend(  # only generates tsne over test split
                filter(lambda prediction: prediction["idx"] in test_ids, torch.load(path))
            )

        return train_predictions, test_predictions

    def tsne(self, train_rpr, test_rpr):
        tsne_encoder = TSNE(
            perplexity=30,
            metric="euclidean",
            n_jobs=self.params.tsne.n_jobs,
            random_state=self.params.tsne.random_state
        ).fit(train_rpr)

        return tsne_encoder.transform(test_rpr)


    def perform_tsne(self):

        for fold_id in self.params.data.folds:

            train_predictions, test_predictions = self.load_predictions(fold_id=fold_id)
            train_predictions_df = pd.DataFrame(train_predictions)
            test_predictions_df = pd.DataFrame(test_predictions)


            tsne_rpr = self.tsne(
                np.array(train_predictions_df["rpr"].tolist()),
                np.array(test_predictions_df["rpr"].tolist()),
                )

            sns.scatterplot(
                tsne_rpr[:, 0],
                tsne_rpr[:, 1],
                hue=test_predictions_df["true_cls"],
                palette="deep"
            )
            Path(self.params.tsne.dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f"{self.params.tsne.dir}{fold_id}.pdf",
                dpi=300
            )
            plt.clf()

