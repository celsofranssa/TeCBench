import os

import hydra
from omegaconf import OmegaConf

from source.model.LitAutoEncoder import LitAutoEncoder
from source.datamodule.MNISTDataModule import MNISTDataModule
import pytorch_lightning as pl


def fit(params):
    print(f"Fitting with fowling params"
          f"{OmegaConf.to_yaml(params, resolve=True)}"
          )
    # init model
    ae = LitAutoEncoder(params.model)

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=3,
        progress_bar_refresh_rate=20
    )

    for fold in params.data.folds:
        # load data
        dm = MNISTDataModule(params.data, fold=fold)

    # Train the ⚡ model
    trainer.fit(
        model=ae,
        datamodule=dm
    )


@hydra.main(config_path="settings/", config_name="settings.yaml")
def perform_tasks(params):
    print(hydra.utils.get_original_cwd())
    os.chdir(hydra.utils.get_original_cwd())
    fit(params)


if __name__ == '__main__':
    perform_tasks()
