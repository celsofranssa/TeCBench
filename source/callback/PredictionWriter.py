from typing import Any, List

import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor


class PredictionWriter(BasePredictionWriter):

    def __init__(self, params):
        super(PredictionWriter, self).__init__(params.write_interval)
        self.params=params
        self.predictions = []

    def write_on_batch_end(
        self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
        batch_idx: int, dataloader_idx: int
    ):
        torch.save(prediction, self.params.output_dir + dataloader_idx + f"{batch_idx}.pt")

    def write_on_epoch_end(
        self, trainer, pl_module, dataloaders: List[Any], batch_indices: List[Any]
    ):
        for dataloader in dataloaders:
            for batch in dataloader:

                # Convert any tensor values to list
                batch_items = {k: v if not isinstance(v, Tensor) else v.tolist() for k, v in batch.items()}

                # Switch predictions so each entry has its own dict
                for values in zip(*batch_items.values()):
                    prediction = dict(zip(batch_items.keys(), values))
                    self.predictions.append(prediction)

        self._checkpoint()


    def _checkpoint(self):
        # Write predictions for current file to disk
        torch.save(self.predictions, f"{self.params.dir}{self.params.name}")






    # def to_disk(self) -> None:
    #     """Write predictions to file(s)."""
    #     for filepath, predictions in self.predictions.items():
    #         fs = get_filesystem(filepath)
    #         # normalize local filepaths only
    #         if fs.protocol == "file":
    #             filepath = os.path.realpath(filepath)
    #         if self.world_size > 1:
    #             stem, extension = os.path.splitext(filepath)
    #             filepath = f"{stem}_rank_{self.global_rank}{extension}"
    #         dirpath = os.path.split(filepath)[0]
    #         fs.mkdirs(dirpath, exist_ok=True)
    #
    #         # Convert any tensor values to list
    #         predictions = {k: v if not isinstance(v, Tensor) else v.tolist() for k, v in predictions.items()}
    #
    #         # Check if all features for this file add up to same length
    #         feature_lens = {k: len(v) for k, v in predictions.items()}
    #         if len(set(feature_lens.values())) != 1:
    #             raise ValueError("Mismatching feature column lengths found in stored EvalResult predictions.")
    #
    #         # Switch predictions so each entry has its own dict
    #         outputs = []
    #         for values in zip(*predictions.values()):
    #             output_element = dict(zip(predictions.keys(), values))
    #             outputs.append(output_element)
    #
    #         # Write predictions for current file to disk
    #         with fs.open(filepath, "wb") as fp:
    #             torch.save(outputs, fp)