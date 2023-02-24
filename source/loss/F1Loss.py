import torch
import torch.nn.functional as F
from torch import nn


class F1Loss(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.epsilon = params.epsilon

    def forward(self, y_pred, y_true):
        y_true = F.one_hot(y_true, -1).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=-1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()
