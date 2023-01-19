import json

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torchmetrics import MetricCollection, F1
from transformers import get_scheduler, get_linear_schedule_with_warmup

class TeCModel(pl.LightningModule):

    def __init__(self, hparams):
        super(TeCModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoder layer
        self.encoder = instantiate(hparams.encoder)

        # dropout layer
        self.dropout = torch.nn.Dropout(hparams.dropout)

        # classification head
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(hparams.hidden_size, hparams.num_classes)
        )

        # validation and test metrics
        self.val_metrics = self._get_metrics(prefix="val_")
        self.test_metrics = self._get_metrics(prefix="test_")

        # loss
        self.loss = instantiate(hparams.criterion)


    def _get_metrics(self, prefix):
        return MetricCollection(
            metrics={
                "Mic-F1": F1(num_classes=self.hparams.num_classes, average="micro"),
                "Mac-F1": F1(num_classes=self.hparams.num_classes, average="macro"),
                "Wei-F1": F1(num_classes=self.hparams.num_classes, average="weighted")
            },
            prefix=prefix)

    def forward(self, text):
        rpr = self.encoder(text)
        return rpr, self.cls_head(rpr)


    def training_step(self, batch, batch_idx):
        text, true_cls = batch["text"], batch["cls"]
        rpr = self.encoder(text)
        pred_cls = self.cls_head(
            self.dropout(rpr)
        )

        # log training loss
        train_loss = self.loss(rpr, pred_cls, true_cls)

        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        text, true_cls = batch["text"], batch["cls"]
        rpr, pred_cls = self(text)

        # log val loss
        val_loss = self.loss(rpr, pred_cls, true_cls)
        self.log('val_loss', val_loss)

        # log val metrics
        self.log_dict(self.val_metrics(torch.argmax(pred_cls, dim=-1), true_cls), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.val_metrics.compute()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        idx, text, true_class = batch["idx"], batch["text"], batch["cls"]
        rpr, pred_cls = self(text)

        return {
            "idx": idx,
            "rpr": rpr,
            "true_cls": true_class,
            "pred_cls": torch.nn.functional.softmax(pred_cls, dim=-1)
        }

    def configure_optimizers(self):
        # optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            amsgrad=True)

        # scheduler
        step_size_up = round(0.03 * self.num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=round(0.03 * self.num_training_steps),
            num_training_steps=self.num_training_steps
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs
