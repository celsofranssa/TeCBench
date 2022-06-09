import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import torch

class SeparabilityHelper:
    def __init__(self, params):
        self.params = params

    def checkpoint_stats(self, stats):
        """
        Checkpoints stats on disk.
        :param stats: dataframe
        """
        stats.to_csv(
            self.params.stat.dir + self.params.model.name + "_" + self.params.data.name + ".sep",
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
            X = []
            y = []
            predictions = self.load_predictions(fold)
            for prediction in predictions:
                X.append(prediction["rpr"])
                y.append(prediction["true_cls"])
            X = np.array(X)
            stats.at[fold, "Silhouette-Score"] = self.silhouette_score(X, y)
            stats.at[fold, "Separability-Index"] = \
                self.separability_index(X, y, n_neighbors=self.params.separability.n_neighbors)
            stats.at[fold, "Hypothesis-Margin"] = \
                self.hypothesis_margin(X, y, n_neighbors=self.params.separability.n_neighbors)

        # update fold column
        stats["fold"] = stats.index
        self.checkpoint_stats(stats)

    def silhouette_score(self, X, y):
        return silhouette_score(X, y, metric='cosine')

    # SI metric
    def separability_index(X, y, n_neighbors=1):
        n = X.shape[0]
        nn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='cosine', n_jobs=-1)
        nn.fit(X)
        if n > 20000: #Fazer uma linha por vez
            score = 0
            for i in tqdm(range(n)):
                row = X[i,:]
                if type(X) == np.ndarray:
                    row = row.reshape(1, -1)
                distances, idxs = nn.kneighbors(row)
                same_label = sum([1 for j in idxs[0,1:] if y[j] == y[i]]) / n_neighbors
                score += same_label
            return score / len(y)
        distances, idxs = nn.kneighbors(X)
        score = 0
        for i in range(len(y)):
            same_label = sum([1 for j in idxs[i,1:] if y[j] == y[i]]) / n_neighbors
            score += same_label
        return score / len(y)

    def hypothesis_margin(X, y, n_neighbors=1, max_neighbors=100):
        n = X.shape[0]
        cutoff = min(max_neighbors, n)
        nn = NearestNeighbors(n_neighbors=cutoff, metric='cosine', n_jobs=-1)
        nn.fit(X)
        max_error = []
        scores = []

        for i in tqdm(range(n)):
            intras = []
            inters = []
            row = X[i,:]
            if type(X) == np.ndarray:
                row = row.reshape(1, -1)
            distances, idxs = nn.kneighbors(row)
            j = 0
            while len(intras) < n_neighbors:
                j += 1
                if j >= cutoff:
                    break
                if y[i] == y[idxs[0,j]]:
                    intras.append(distances[0,j])   
            j = 0
            while len(inters) < n_neighbors:
                j += 1
                if j >= cutoff:
                    break
                if y[i] != y[idxs[0,j]]:
                    inters.append(distances[0,j])
            a = 1 # Se nao encontrou nenhum vizinho da mesma classe, assume que a intra-dist eh maxima
            b = 1 # Se nao encontrou nenhum vizinho de classe diferente, assume que a inter-dist eh maxima
            approx_a = False
            approx_b = False
            if len(intras) > 0:
                a = np.mean(intras)
            else:
                approx_a = True
            if len(inters) > 0:
                b = np.mean(inters)
            else:
                approx_b = True
            sc = 0
            approx = 0
            m = max(a, b)
            if m != 0:
                sc = (b - a)/m    
            assert(not(approx_a and approx_b))
            if approx_a:
                max_error.append(1 - b) #Subestimou
            elif approx_b:
                max_error.append(a - 1) #Superestimou
            else:
                max_error.append(0)
            scores.append(sc)
        res = np.mean(scores)
        error = np.mean(max_error)
        num = math.fabs(res+error)
        den = math.fabs(res)
        if num > den:
            aux = den
            den = num
            num = aux
        approx = num/den
        if approx < 0.9999:
            print(f"Metrica foi aproximada. Fator de aproximacao: {100*approx}%")
            print(f"Intervalo de variação: resultado está entre {res} e {res+error}")
        return res

