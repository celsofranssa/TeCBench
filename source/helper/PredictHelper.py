from omegaconf import OmegaConf

from source.callback.PredictionWriter import PredictionWriter

import pytorch_lightning as pl

from source.datamodule.TecDataModule import TeCDataModule
from source.model.TeCModel import TeCModel


class PredictHelper:

    def __init__(self, params):
        self.params = params

    def perform_predict(self):
        for fold_id in self.params.data.folds:
            print(
                f"Predicting {self.params.model.name} over {self.params.data.name} (fold {fold_id}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            # data
            self.params.data.fold_id = fold_id
            self.params.prediction.fold_id = fold_id
            dm = TeCDataModule(self.params.data)
            dm.prepare_data()
            dm.setup("predict")

            # model
            model = TeCModel.load_from_checkpoint(
                checkpoint_path=f"{self.params.model_checkpoint.dir}{self.params.model.name}_{self.params.data.name}_{fold_id}.ckpt"
            )

            # trainer
            trainer = pl.Trainer(
                gpus=self.params.trainer.gpus,
                callbacks=[PredictionWriter(self.params.prediction)]
            )

            trainer.predict(
                model=model,
                datamodule=dm,

            )
