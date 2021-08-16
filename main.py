import os

import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from transformers import AutoTokenizer

from source.callback.PredictionWriter import PredictionWriter
from source.datamodule.TecDataModule import TeCDataModule
from source.helper.EvalHelper import EvalHelper
from source.model.TeCModel import TecModel


def get_logger(params, fold):
    return loggers.TensorBoardLogger(
        save_dir=params.log.dir,
        name=f"{params.model.name}_{params.data.name}_{fold}_exp"
    )


def get_model_checkpoint_callback(params, fold):
    return ModelCheckpoint(
        monitor="val_Mic-F1",
        dirpath=params.model_checkpoint.dir,
        filename=f"{params.model.name}_{params.data.name}_{fold}",
        save_top_k=1,
        save_weights_only=True,
        mode="max"
    )


def get_early_stopping_callback(params):
    return EarlyStopping(
        monitor='val_Mic-F1',
        patience=params.trainer.patience,
        min_delta=params.trainer.min_delta,
        mode='max'
    )


def get_tokenizer(hparams):
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.tokenizer.architecture
    )
    if hparams.tokenizer.architecture == "gpt2":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def train(params):

    for fold in params.data.folds:
        print(f"Fitting {params.model.name} over {params.data.name} (fold {fold}) with fowling params\n"
              f"{OmegaConf.to_yaml(params)}\n")
        # Initialize a trainer
        trainer = pl.Trainer(
            fast_dev_run=params.trainer.fast_dev_run,
            max_epochs=params.trainer.max_epochs,
            precision=params.trainer.precision,
            gpus=params.trainer.gpus,
            progress_bar_refresh_rate=params.trainer.progress_bar_refresh_rate,
            logger=get_logger(params, fold),
            callbacks=[
                get_model_checkpoint_callback(params, fold),  # checkpoint_callback
                get_early_stopping_callback(params),  # early_stopping_callback
            ]
        )
        # Train the ⚡ model
        trainer.fit(
            model=TecModel(params.model),
            datamodule=TeCDataModule(params.data, get_tokenizer(params.model), fold=fold)
        )


def test(params):

    for fold in params.data.folds:
        print(f"Predicting {params.model.name} over {params.data.name} (fold {fold}) with fowling params\n"
              f"{OmegaConf.to_yaml(params)}\n")


        # data
        dm = TeCDataModule(params.data, get_tokenizer(params.model), fold=fold)

        # model
        model = TecModel.load_from_checkpoint(
            checkpoint_path=f"{params.model_checkpoint.dir}{params.model.name}_{params.data.name}_{fold}.ckpt"
        )

        params.prediction.name = f"{params.model.name}_{params.data.name}_{fold}.prd"

        # trainer
        trainer = pl.Trainer(
            gpus=params.trainer.gpus,
            callbacks=[PredictionWriter(params.prediction)]
        )

        # testing
        dm.prepare_data()
        dm.setup('test')
        trainer.test(
            model=model,
            datamodule=dm
        )


def eval(params):
    print(f"Evaluating {params.model.name} over {params.data.name} with fowling params\n"
          f"{OmegaConf.to_yaml(params)}\n")
    evaluator = EvalHelper(params)
    evaluator.perform_eval()

def predict(params):

    for fold in params.data.folds:
        print(f"Predicting {params.model.name} over {params.data.name} (fold {fold}) with fowling params\n"
              f"{OmegaConf.to_yaml(params)}\n")


        # data
        dm = TeCDataModule(params.data, get_tokenizer(params.model), fold=fold)

        # model
        model = TecModel.load_from_checkpoint(
            checkpoint_path=f"{params.model_checkpoint.dir}{params.model.name}_{params.data.name}_{fold}.ckpt"
        )

        params.representation.name = f"{params.model.name}_{params.data.name}_{fold}.rpr"

        # trainer
        trainer = pl.Trainer(
            gpus=params.trainer.gpus,
            callbacks=[PredictionWriter(params.representation)]
        )

        # predicting
        dm.prepare_data()
        dm.setup("predict")
        trainer.predict(
            model=model,
            datamodule=dm,

        )


@hydra.main(config_path="settings/", config_name="settings.yaml")
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)
    if "fit" in params.tasks:
        train(params)
    if "test" in params.tasks:
        test(params)
    if "eval" in params.tasks:
        eval(params)
    if "predict" in params.tasks:
        predict(params)


if __name__ == '__main__':
    perform_tasks()
