import json

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch import nn, Tensor
from torchmetrics import MetricCollection, F1
from transformers import get_scheduler


class TeCModel(pl.LightningModule):

    def __init__(self, hparams):
        super(TeCModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoder layer
        self.encoder = instantiate(hparams.encoder)

        # the classification head
        self.cls_head = torch.nn.Sequential(
            torch.nn.Dropout(hparams.dropout),
            torch.nn.Linear(hparams.hidden_size, hparams.num_classes),
            torch.nn.LogSoftmax(dim=-1)
        )

        # validation and test metrics
        self.val_metrics = self._get_metrics(prefix="val_")
        self.test_metrics = self._get_metrics(prefix="test_")

        # loss
        self.loss = instantiate(hparams.loss)

    def _get_metrics(self, prefix):
        return MetricCollection(
            metrics={
                "Mic-F1": F1(num_classes=self.hparams.num_classes, average="micro"),
                "Mac-F1": F1(num_classes=self.hparams.num_classes, average="macro"),
                "Wei-F1": F1(num_classes=self.hparams.num_classes, average="weighted")
            },
            prefix=prefix)

    def forward(self, text):
        return self.encoder(text)

    def training_step(self, batch, batch_idx):
        text, true_cls = batch["text"], batch["cls"]
        pred_cls = self.cls_head(
            self(text)
        )
        train_loss = self.loss(pred_cls, true_cls)

        # log training loss
        self.log('train_loss', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        text, true_cls = batch["text"], batch["cls"]
        pred_cls = self.cls_head(
            self(text)
        )
        val_loss = self.loss(pred_cls, true_cls)

        # log val loss
        self.log('val_loss', val_loss)

        # log val metrics
        self.log_dict(self.val_metrics(pred_cls, true_cls), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.val_metrics.compute()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        idx, text, true_class = batch["idx"], batch["text"], batch["cls"]
        rpr = self.encoder(text)

        return {
            "idx": idx,
            "rpr": rpr,
            "true_cls": true_class,
            "pred_cls": torch.argmax(self.cls_head(rpr), dim=-1)
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(), 
            lr=self.hparams.lr
        )
        scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=self.hparams.n_warmup_steps,
          num_training_steps=self.hparams.n_training_steps
        )

        return dict(
          optimizer=optimizer,
          lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
          )
        )


    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs
