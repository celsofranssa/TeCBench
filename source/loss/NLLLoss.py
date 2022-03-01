import torch
from torch import nn


class NLLLoss(nn.Module):

    def __init__(self, params):
        super(NLLLoss, self).__init__()
        self.params = params
        self.criterion = nn.NLLLoss()

    def forward(self, pred_cls, true_cls):
        """
        Computes the cross entropy loss between pred_cls and true_cls.
        """
        return self.criterion(pred_cls, true_cls)
