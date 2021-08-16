from typing import Any, List

import torch
from pytorch_lightning.callbacks import BasePredictionWriter

class PredictionWriter(BasePredictionWriter):

    def __init__(self, params):
        super().__init__(params.write_interval)
        self.params=params

    def write_on_batch_end(
        self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
        batch_idx: int, dataloader_idx: int
    ):
        torch.save(prediction, self.params.output_dir + dataloader_idx + f"{batch_idx}.pt")

    def write_on_epoch_end(
        self, trainer, pl_module, predictions: List[Any], batch_indices: List[Any]
    ):
        torch.save(predictions, self.params.output_dir + "predictions.pt")