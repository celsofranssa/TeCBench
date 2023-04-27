import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, silhouette_score, precision_score, recall_score, classification_report, \
    confusion_matrix
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

    def checkpoint_reports(self, reports):
        with open(f"{self.params.stat.dir}{self.params.model.name}_{self.params.data.name}.rpt", "wb") as reports_file:
            pickle.dump(reports, reports_file)

    def checkpoint_confusion_matrices(self, confusion_matrices):
        with open(f"{self.params.stat.dir}{self.params.model.name}_{self.params.data.name}.cfm", "wb") as confusion_matrices_file:
            pickle.dump(confusion_matrices, confusion_matrices_file)

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
        reports = {}
        confusion_matrices = {}

        for fold in self.params.data.folds:
            print(
                f"Evaluating {self.params.model.name} over {self.params.data.name} "
                f"(fold {fold}) with fowling self.params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")
            true_classes = []
            pred_classes = []
            rprs = []
            predictions = self.load_predictions(fold)
            for prediction in predictions:
                true_classes.append(prediction["true_cls"])
                pred_classes.append(np.argmax(prediction["pred_cls"]))
                rprs.append(prediction["rpr"])

            stats.at[fold, "Mic-F1"] = f1_score(true_classes, pred_classes, average='micro')
            stats.at[fold, "Mac-F1"] = f1_score(true_classes, pred_classes, average='macro')
            stats.at[fold, "Wei-F1"] = f1_score(true_classes, pred_classes, average='weighted')
            reports[fold] = classification_report(true_classes, pred_classes,
                                                  target_names=self.params.data.labels, output_dict=True)
            confusion_matrices[fold] = confusion_matrix(true_classes, pred_classes)


        # update fold colum
        stats["fold"] = stats.index

        self.checkpoint_stats(stats)
        self.checkpoint_reports(reports)
        self.checkpoint_confusion_matrices(confusion_matrices)


    def silhouette_score(self, X, y):
        return silhouette_score(X, y, metric='cosine')

    def separability_index(self, X, y, n_neighbors=1):
        # SI metric
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine')
        nn.fit(X)
        distances, idxs = nn.kneighbors(X)
        score = 0
        for i in range(len(y)):
            same_label = sum([1 for j in idxs[i][1:] if y[j] == y[i]]) / n_neighbors
            score += same_label
        return score / len(y)

    def hypothesis_margin(self, X, y, n_neighbors=1):
        # HM metric
        nn = NearestNeighbors(n_neighbors=len(y), metric='cosine')
        nn.fit(X)
        distances, idxs = nn.kneighbors(X)
        scores = []
        for i in range(len(y)):
            j = 0
            intras = []
            while len(intras) < n_neighbors:
                j += 1
                if j >= len(y):
                    break
                if y[i] == y[idxs[i][j]]:
                    intras.append(distances[i][j])
            j = 0
            inters = []
            while len(inters) < n_neighbors:
                j += 1
                if j >= len(y):
                    break
                if y[i] != y[idxs[i][j]]:
                    inters.append(distances[i][j])
            a = np.mean(intras)
            b = np.mean(inters)
            scores.append((b - a) / max(a, b))
        return np.mean(scores)


