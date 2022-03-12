import torch
from torch import nn


class CrossEntropyLoss(nn.Module):

    def __init__(self, params):
        super(CrossEntropyLoss, self).__init__()
        self.params = params
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, rpr, pred_cls, true_cls):
        """
        Computes the cross entropy loss between pred_cls and true_cls.
        """
        return self.criterion(pred_cls, true_cls)
