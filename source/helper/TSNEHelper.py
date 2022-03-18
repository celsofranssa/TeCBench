from pathlib import Path

import numpy as np

import pickle

import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class TSNEHelper:
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
        
        test_ids = self._load_ids(
            f"{self.params.data.dir}fold_{fold_id}/test.pkl"
        )
        
        if test_ids is None:
            raise Exception("No test ids")
        

        predictions = []
        for path in tqdm(predictions_paths, desc="Loading predictions"):
            predictions.extend(  # only generates tsne over test split
                filter(lambda prediction: prediction["idx"] in test_ids, torch.load(path))
            )

        return predictions

    def tsne(self, test_rpr):
        return TSNE(
            n_components=2,
            learning_rate=200,
            init = 'random'
        ).fit_transform(test_rpr)


    def perform_tsne(self):

        for fold_id in self.params.data.folds:

            predictions = self.load_predictions(fold_id=fold_id)

            test_rprs = []
            for prediction in predictions:
                test_rprs.append(prediction["rpr"])
            
            test_rprs = np.asarray(test_rprs, dtype=np.float64)

            tsne = self.tsne(test_rprs)

            sns.scatterplot(
                tsne[:, 0],
                tsne[:, 1],
                # hue="cls" insert class
            )
            Path(self.params.tsne.dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f"{self.params.tsne.dir}{fold_id}.pdf",
                dpi=300
            )

