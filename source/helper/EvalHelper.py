import json

import numpy as np
import torch
from sklearn.metrics import f1_score


class EvalHelper:
    def __init__(self, params):
        self.params = params

    def checkpoint_stats(self, stats):
        stats_path = f"{self.params.stat.dir}{self.params.model.name}_{self.params.data.name}.stat"
        with open(stats_path, "w") as stats_file:
            for data in stats.items():
                stats_file.write(f"{json.dumps(data)}\n")

    def load_predictions(self, fold):
        # load predictions
        return torch.load(self.params.model.prediction.dir +
                          f"{self.params.model.name}_{self.params.data.name}_{fold}.prd")


    def summarize_stats(self, stats):
        mic_f1s = []
        wei_f1s = []

        for fold in stats.keys():
            mic_f1s.append(stats[fold]["Mic-F1"])
            wei_f1s.append(stats[fold]["Wei-F1"])

        stats["Avg-Mic-F1"] = np.average(mic_f1s)
        stats["Avg-Wei-F1"] = np.average(wei_f1s)
        stats["Std-Mic-F1"] = np.std(mic_f1s)
        stats["Std-Wei-F1"] = np.std(wei_f1s)

        return stats

    def compute_stats(self):
        stats = {}
        for fold in self.params.data.folds:
            true_classes = []
            pred_classes = []
            predictions = self.load_predictions(fold)
            for prediction in predictions:
                true_classes.append(prediction["true_cls"])
                pred_classes.append(prediction["pred_cls"])


            stats[fold] = {
                "Mic-F1": f1_score(true_classes, pred_classes, average='micro'),
                "Wei-F1": f1_score(true_classes, pred_classes, average='weighted')
            }
        return self.summarize_stats(stats)

    def perform_eval(self):

        stats = self.compute_stats()
        self.checkpoint_stats(stats)
