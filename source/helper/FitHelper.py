from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar

from transformers import AutoTokenizer

from source.datamodule.TecDataModule import TeCDataModule
from source.model.TeCModel import TeCModel


class FitHelper:

    def __init__(self, params):
        self.params = params

    def perform_fit(self):
        seed_everything(777, workers=True)

        for fold_idx in self.params.data.folds:
            # Initialize a trainer
            trainer = pl.Trainer(
                accelerator=self.params.trainer.accelerator,
                devices=self.params.trainer.devices,
                max_epochs=self.params.trainer.max_epochs,
                precision=self.params.trainer.precision,
                logger=self.get_logger(fold_idx),
                callbacks=[
                    self.get_model_checkpoint_callback(fold_idx),  # checkpoint_callback
                    self.get_early_stopping_callback(),  # early_stopping_callback
                    self.get_lr_monitor(),
                    self.get_progress_bar_callback()
                ]
            )

            # datamodule
            datamodule = TeCDataModule(
                params=self.params.data,
                tokenizer=self.get_tokenizer(self.params.model.tokenizer),
                fold=fold_idx
            )

            # model
            model = TeCModel(self.params.model)

            # Train the ⚡ model
            print(
                f"Fitting {self.params.model.name} over {self.params.data.name} "
                f"(fold {fold_idx}) with fowling self.params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")
            trainer.fit(
                model=model,
                datamodule=datamodule
            )

    def get_logger(self, fold_idx):
        return loggers.TensorBoardLogger(
            save_dir=self.params.log.dir,
            name=f"{self.params.model.name}_{self.params.data.name}_{fold_idx}_exp"
        )

    def get_model_checkpoint_callback(self, fold_idx):
        return ModelCheckpoint(
            monitor="val_Wei-F1",
            dirpath=self.params.model_checkpoint.dir,
            filename=f"{self.params.model.name}_{self.params.data.name}_{fold_idx}",
            save_top_k=1,
            save_weights_only=True,
            mode="max"
        )

    def get_early_stopping_callback(self):
        return EarlyStopping(
            monitor="val_Mac-F1",
            patience=self.params.trainer.patience,
            min_delta=self.params.trainer.min_delta,
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

    def get_lr_monitor(self):
        return LearningRateMonitor(logging_interval='step')

    def get_progress_bar_callback(self):
        return TQDMProgressBar(
            refresh_rate=self.params.trainer.progress_bar_refresh_rate,
            process_position=0
        )
