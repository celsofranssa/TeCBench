import torch
from torchmetrics import Metric
from sklearn.metrics import f1_score


class F1(Metric):
    def __init__(self, average, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("prediction", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([]), dist_reduce_fx="cat")
        self.average = average

    def update(self, prediction, target):
        prediction = torch.argmax(prediction, dim=-1)
        assert prediction.shape == target.shape
        self.prediction=torch.cat([self.prediction, prediction])
        self.target=torch.cat([self.target, target])

    def compute(self):
        """
        TODO: https://stackoverflow.com/questions/62265351/measuring-f1-score-for-multiclass-classification-natively-in-pytorch
        :return:
        """
        return f1_score(
            self.target.cpu(),
            self.prediction.cpu(),
            average=self.average
        )
