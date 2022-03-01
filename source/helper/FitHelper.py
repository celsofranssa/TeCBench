from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer

from source.datamodule.TecDataModule import TeCDataModule
from source.model.TeCModel import TeCModel


class FitHelper:

    def __init__(self, params):
        self.params = params

    def perform_fit(self):
        for fold_id in self.params.data.folds:
            # Initialize a trainer
            trainer = pl.Trainer(
                fast_dev_run=self.params.trainer.fast_dev_run,
                max_epochs=self.params.trainer.max_epochs,
                precision=self.params.trainer.precision,
                gpus=self.params.trainer.gpus,
                progress_bar_refresh_rate=self.params.trainer.progress_bar_refresh_rate,
                logger=self.get_logger(self.params, fold_id),
                callbacks=[
                    self.get_model_checkpoint_callback(self.params, fold_id),  # checkpoint_callback
                    self.get_early_stopping_callback(self.params),  # early_stopping_callback
                ]
            )

            # datamodule
            datamodule = TeCDataModule(
                params=self.params.data,
                tokenizer=self.get_tokenizer(self.params.model.tokenizer),
                fold=fold_id
            )

            # model
            model = TeCModel(self.params.model)

            # Train the ⚡ model
            print(
                f"Fitting {self.params.model.name} over {self.params.data.name} "
                f"(fold {fold_id}) with fowling self.params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")
            trainer.fit(
                model=model,
                datamodule=datamodule
            )

    def get_logger(self, params, fold):
        return loggers.TensorBoardLogger(
            save_dir=params.log.dir,
            name=f"{params.model.name}_{params.data.name}_{fold}_exp"
        )

    def get_model_checkpoint_callback(self, params, fold):
        return ModelCheckpoint(
            monitor="val_Wei-F1",
            dirpath=params.model_checkpoint.dir,
            filename=f"{params.model.name}_{params.data.name}_{fold}",
            save_top_k=1,
            save_weights_only=True,
            mode="max"
        )

    def get_early_stopping_callback(self, params):
        return EarlyStopping(
            monitor="val_Wei-F1",
            patience=params.trainer.patience,
            min_delta=params.trainer.min_delta,
            mode='max'
        )

    def get_tokenizer(self, params):
        tokenizer = AutoTokenizer.from_pretrained(
            params.architecture
        )
        if "gpt" in params.architecture:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            params.pad = tokenizer.convert_tokens_to_ids("[PAD]")
        return tokenizer
