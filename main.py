import os

import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from transformers import AutoTokenizer

from source.datamodule.TecDataModule import TeCDataModule
from source.model.TeCModel import TecModel

from source.pooling.NoPooling import NoPooling

# def get_logger(hparams, fold):
#     return pl_loggers.TensorBoardLogger(
#         save_dir=hparams.log.dir,
#         name=f"{hparams.model.name}_{hparams.data.name}_{fold}_exp"
#     )
#
#
# def get_model_checkpoint_callback(hparams, fold):
#     return ModelCheckpoint(
#         monitor="val_loss",
#         dirpath=hparams.model_checkpoint.dir,
#         filename=f"{hparams.model.name}_{hparams.data.name}_{fold}",
#         save_top_k=1,
#         save_weights_only=True,
#         mode="min"
#     )
#
#
# def get_early_stopping_callback(hparams):
#     return EarlyStopping(
#         monitor='val_loss',
#         patience=hparams.patience,
#         min_delta=hparams.min_delta,
#         mode='min'
#     )



def get_tokenizer(hparams):
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.tokenizer.architecture
    )
    if hparams.tokenizer.architecture == "gpt2":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def fit(params):
    print(f"Fitting with fowling params\n"
          f"{OmegaConf.to_yaml(params)}"
          )
    # init model
    tec_model = TecModel(params.model)

    # Initialize a trainer
    trainer = pl.Trainer(
        fast_dev_run=params.trainer.fast_dev_run,
        max_epochs=params.trainer.max_epochs,
        precision=params.trainer.precision,
        gpus=params.trainer.gpus,
        enable_pl_optimizer=params.trainer.enable_pl_optimizer,
        progress_bar_refresh_rate=params.trainer.progress_bar_refresh_rate,
        #logger=tb_logger,
        #callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor]
    )

    tokenizer = get_tokenizer(params.model)

    for fold in params.data.folds:
        # load data
        dm = TeCDataModule(params.data, tokenizer, fold=fold)

        # Train the ⚡ model
        trainer.fit(
            model=tec_model,
            datamodule=dm
        )
    pool = NoPooling()


@hydra.main(config_path="settings/", config_name="settings.yaml")
def perform_tasks(params):
    print(hydra.utils.get_original_cwd())
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)
    fit(params)


if __name__ == '__main__':
    perform_tasks()
