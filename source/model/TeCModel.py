import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch import nn
from torchmetrics import MetricCollection

from source.metric.F1 import F1


class TecModel(pl.LightningModule):

    def __init__(self, hparams):
        super(TecModel, self).__init__()
        self.save_hyperparameters(hparams)

        self.encoder = instantiate(hparams.encoder)

        self.cls_head = torch.nn.Sequential(
            torch.nn.Dropout(hparams.dropout),
            torch.nn.Linear(hparams.hidden_size, hparams.num_classes),
            torch.nn.LogSoftmax(dim=-1)
        )

        self.train_metrics = self.get_metrics(prefix="train_")
        self.val_metrics = self.get_metrics(prefix="val_")

        self.loss = nn.NLLLoss()

    def get_metrics(self, prefix):
        return MetricCollection(
            metrics={
                "MicF1": F1(average="micro"),
                "MacF1": F1(average="macro")
            },
            prefix=prefix)

    def forward(self, text):
        return self.encoder(text)

    def training_step(self, batch, batch_idx):
        text, cls = batch["text"], batch["cls"]
        preds = self.cls_head(
            self(text)
        )
        train_loss = self.loss(preds, cls)

        # log training loss
        self.log('train_loss', train_loss)

        ts_metrics = self.train_metrics(preds, cls)

        # print(f"\n cls: \n{cls}")
        # print(f"\n preds: \n{preds}")
        # print(f"\n ts_metrics: \n{ts_metrics}")

        self.log_dict(ts_metrics, prog_bar=True)

        return train_loss

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, batch, batch_idx):
        text, cls = batch["text"], batch["cls"]
        preds = self.cls_head(
            self(text)
        )
        loss = self.loss(preds, cls)

        # log training loss
        self.log('val_loss', loss)

        # log macro f1
        self.log_dict(self.val_metrics(preds, cls), prog_bar=True)

    def validation_epoch_end(self, outs):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        # optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.hparams.weight_decay,
            amsgrad=True)

        # scheduler
        step_size_up = round(0.03 * self.num_training_steps)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            mode='triangular2',
            base_lr=self.hparams.base_lr,
            max_lr=self.hparams.max_lr,
            step_size_up=step_size_up,
            cycle_momentum=False)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs
