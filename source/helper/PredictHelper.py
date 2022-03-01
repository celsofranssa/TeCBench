from omegaconf import OmegaConf
from transformers import AutoTokenizer

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


            # datamodule
            self.params.prediction.fold_id = fold_id
            datamodule = TeCDataModule(
                params=self.params.data,
                tokenizer=self.get_tokenizer(self.params.model.tokenizer),
                fold=fold_id
            )
            datamodule.prepare_data()
            datamodule.setup(stage="predict")

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
                datamodule=datamodule,

            )

    def get_tokenizer(self, params):
        tokenizer = AutoTokenizer.from_pretrained(
            params.architecture
        )
        if "gpt" in params.architecture:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            params.pad = tokenizer.convert_tokens_to_ids("[PAD]")
        return tokenizer
