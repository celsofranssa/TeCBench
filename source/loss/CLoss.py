from torch import nn
from pytorch_metric_learning import miners, losses

class CLoss(nn.Module):

    def __init__(self, params):
        super(CLoss, self).__init__()
        self.miner = miners.MultiSimilarityMiner(epsilon=params.epsilon)
        self.l1 = losses.NTXentLoss(temperature=params.temperature)
        self.l2 = nn.CrossEntropyLoss()

    def forward(self, rpr, pred_cls, true_cls):
        """
        Computes the cross entropy loss between pred_cls and true_cls.
        """
        pairs = self.miner(rpr, true_cls)
        return 0.89*self.l2(pred_cls, true_cls) + 0.11*self.l1(rpr, true_cls, pairs)
