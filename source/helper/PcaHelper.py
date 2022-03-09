import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.decomposition import PCA

class PcaHelper:
    def __init__(self, params):
        self.params = params

    def load_prd_files(self, fold):

        predictions_paths = sorted(
            Path(f"{self.params.prediction.dir}fold_{fold}/").glob("*.prd")
        )


        predictions = []
        for path in tqdm(predictions_paths, desc="Loading predictions"):
            predictions.extend( # only eval over test split
                torch.load(path)
            )

        return predictions

    def run_pca(self, train_emb, test_emb):
        pca = PCA(n_components=2, svd_solver="randomized", random_state=42)

        ## attempt 1
        pca.fit(train_emb)
        vectors = pca.transform(test_emb)

        pos_vectors = list()
        neg_vectors = list()
        for idx in range(0, len(test_emb)):
            if test_docs_class[idx] == 1:
                pos_vectors.append(vectors[idx])
            else:
                neg_vectors.append(vectors[idx])

        pos_vectors = np.asarray(pos_vectors)
        neg_vectors = np.asarray(neg_vectors)

        return pos_vectors, neg_vectors
