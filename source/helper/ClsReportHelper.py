import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, silhouette_score, precision_score, recall_score, classification_report
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class EvalHelper:
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
        preds, targets = [], []

        for fold in self.params.data.folds:
            true_classes = []
            pred_classes = []
            rprs = []
            predictions = self.load_predictions(fold)
            for prediction in predictions:
                true_classes.append(prediction["true_cls"])
                pred_classes.append(prediction["pred_cls"])
                rprs.append(prediction["rpr"])


            preds.extend(pred_classes)
            targets.extend(true_classes)



        # update fold colum
        print(classification_report(targets, preds, labels=[0, 1, 2, 3]))
        stats["fold"] = stats.index

        self.checkpoint_stats(stats)


