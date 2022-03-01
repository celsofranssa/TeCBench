import torch
from pytorch_lightning import LightningModule


class NoPooling(LightningModule):
    """
    Performs no pooling on the last hidden-states transformer output.
    """

    def __init__(self):
        super(NoPooling, self).__init__()

    def forward(self, attention_mask, encoder_outputs):
        return encoder_outputs.pooler_output
