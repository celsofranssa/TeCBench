from pathlib import Path
from typing import Any, List, Sequence, Optional

import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):

    def __init__(self, params):
        super(PredictionWriter, self).__init__(params.write_interval)
        self.params = params
        self.predictions = []

    def write_on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", predictions: Sequence[Any],
                           batch_indices: Optional[Sequence[Any]]) -> None:
        pass

    def write_on_batch_end(
            self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
            batch_idx: int, dataloader_idx: int
    ):
        predictions = []

        for idx, rpr, true_class, pred_class in zip(
                prediction["idx"].tolist(),
                prediction["rpr"].tolist(),
                prediction["true_class"].tolist(),
                prediction["pred_class"].tolist()):
            predictions.append({
                "idx": idx,
                "rpr": rpr,
                "pred_class": pred_class,
                "true_class": true_class
            })

        self._checkpoint(predictions, dataloader_idx, batch_idx)

    def _checkpoint(self, predictions, dataloader_idx, batch_idx):
        checkpoint_dir = f"{self.params.dir}fold_{self.params.fold}/"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            predictions,
            f"{checkpoint_dir}{dataloader_idx}_{batch_idx}.prd"
        )
