import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


class SeparabilityHelper:
    def __init__(self, params):
        self.params = params

    def checkpoint_stats(self, stats):
        """
        Checkpoints stats on disk.
        :param stats: dataframe
        """
        stats.to_csv(
            self.params.stat.dir + self.params.model.name + "_" + self.params.data.name + ".stat",
            sep='\t', index=False, header=True)

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            return pickle.load(ids_file)

    def load_predictions(self, fold):

        predictions_paths = sorted(
            Path(f"{self.params.prediction.dir}fold_{fold}/").glob("*.prd")
        )

        test_ids = self._load_ids(
            f"{self.params.data.dir}fold_{fold}/test.pkl"
        )

        predictions = []
        for path in tqdm(predictions_paths, desc="Loading predictions"):
            predictions.extend( # only eval over test split
                filter(lambda prediction: prediction["idx"] in test_ids, torch.load(path))
            )

        return predictions

    def perform_eval(self):
        stats = pd.DataFrame(columns=["fold"])

        for fold in self.params.data.folds:
            predictions = self.load_predictions(fold)
            for prediction in predictions:
                pass

            stats.at[fold, "Silhouette"] = self.silhouette()
            stats.at[fold, "Separability-Index"] = self.separability_index()
            stats.at[fold, "Hypothesis-Margin"] = self.hypothesis_margin()

        # update fold colum
        stats["fold"] = stats.index

        self.checkpoint_stats(stats)

    def silhouette(self):
        pass

    def separability_index(self):
        pass

    def hypothesis_margin(self):
        pass

