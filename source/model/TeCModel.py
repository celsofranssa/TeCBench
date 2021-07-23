
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.metrics import F1
from torch import nn


class TecModel(pl.LightningModule):

    def __init__(self, hparams):
        super(TecModel, self).__init__()
        self.save_hyperparameters(hparams)

        self.encoder = instantiate(hparams.encoder)

        self.cls_head = torch.nn.Sequential(
            torch.nn.Dropout(hparams.dropout),
            torch.nn.Linear(hparams.hidden_size, hparams.num_cls),
            torch.nn.LogSoftmax(dim=-1)
        )

        self.loss = nn.NLLLoss()

        self.macro_f1 = F1(num_classes=self.hparams.num_cls, average="macro")

    def forward(self, text):
        return self.encoder(text)

    def training_step(self, batch, batch_idx):
        text, cls = batch["text"], batch["cls"]
        pred = self.cls_head(
            self(text)
        )
        loss = self.loss(pred, cls)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        text, cls = batch["text"], batch["cls"]
        pred = self.cls_head(
            self(text)
        )
        loss = self.loss(pred, cls)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
