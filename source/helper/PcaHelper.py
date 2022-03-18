from pathlib import Path

import numpy as np

import pickle

import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class PCAHelper:
    def __init__(self, params):
        self.params = params
        sns.set_theme(style="darkgrid")

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            ids = pickle.load(ids_file)
            
            return ids
        
        return None

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
        
        if test_ids is None:
            raise Exception("No test ids")
        
        trainings = []
        for path in tqdm(predictions_paths, desc="Loading predictions"):
            trainings.extend(  # only generates tsne over test split
                filter(lambda prediction: prediction["idx"] in train_ids, torch.load(path))
            )

        predictions = []
        for path in tqdm(predictions_paths, desc="Loading predictions"):
            predictions.extend(  # only generates tsne over test split
                filter(lambda prediction: prediction["idx"] in test_ids, torch.load(path))
            )

        return trainings, predictions

    def pca(self, train_rpr, test_rpr):
        pca_obj = PCA(n_components=2, 
                svd_solver="randomized", 
                random_state=42).fit(train_rpr)

        return pca_obj.transform(test_rpr)

    def perform_tsne(self):

        for fold_id in self.params.data.folds:

            trainings, predictions = self.load_predictions(fold_id=fold_id)

            train_rprs = []
            for training in trainings:
                train_rprs.append(training["rpr"])

            test_rprs = []
            for prediction in predictions:
                test_rprs.append(prediction["rpr"])
            
            train_rprs = np.asarray(train_rprs, dtype=np.float64)
            test_rprs = np.asarray(test_rprs, dtype=np.float64)

            pca = self.pca(train_rprs, test_rprs)

            sns.scatterplot(
                pca[:, 0],
                pca[:, 1],
                # hue="cls" insert class
            )
            Path(self.params.pca.dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f"{self.params.pca.dir}{fold_id}.pdf",
                dpi=300
            )

