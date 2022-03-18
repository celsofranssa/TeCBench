import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

class SeparabilityHelper:
    def __init__(self, params):
        self.params = params
        with open(self.params.dir + f"samples.pkl", "rb") as dataset_file:
            self.samples = pickle.load(dataset_file)

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

    def load_data(self, fold):

        test_ids = self._load_ids(
            f"{self.params.data.dir}fold_{fold}/test.pkl"
        )

        data = self.samples[test_ids]
        X = []
        y = []
        for item in data:
            X.append(data['vec'])
            y.append(data['cls'])

        return np.array(X), y

    def perform_eval(self):
        stats = pd.DataFrame(columns=["fold"])

        for fold in self.params.data.folds:
            X,y = self.load_data(fold)

            stats.at[fold, "Silhouette-Score"] = self.silhouette_score(X, y)
            stats.at[fold, "Separability-Index"] = self.separability_index(X, y)
            stats.at[fold, "Hypothesis-Margin"] = self.hypothesis_margin(X, y)

        # update fold colum
        stats["fold"] = stats.index

        self.checkpoint_stats(stats)

    def silhouette_score(self, X, y):
        return silhouette_score(X, y)

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

