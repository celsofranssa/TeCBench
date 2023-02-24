import pickle

import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from tqdm import tqdm

from source.dataset.TeCDataset import TeCDataset
from source.dataset.TeCPredictDataset import TeCPredictDataset


class TeCDataModule(pl.LightningDataModule):
    def __init__(self, params, tokenizer, fold):
        super(TeCDataModule, self).__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.fold = fold


    def prepare_data(self):
        with open(self.params.dir + f"samples.pkl", "rb") as dataset_file:
            self.samples = pickle.load(dataset_file)

        with open(f"{self.params.extended_dir}_{self.fold}/samples.pkl", "rb") as dataset_file:
            extended_samples = pickle.load(dataset_file)
            self.cls_samples = {0: [], 1: [], 2: [], 3: []}
            for sample in tqdm(extended_samples, desc="Extending samples"):
                sample["cls"] = np.argmax(sample["cls"])
                self.cls_samples[sample["cls"]].append(sample)

    def setup(self, stage=None):

        if stage == 'fit' or stage is "predict":
            self.train_dataset = TeCDataset(
                samples=self.samples,
                cls_samples=self.cls_samples,
                ids_path=self.params.dir + f"fold_{self.fold}/train.pkl",
                tokenizer=self.tokenizer,
                max_length=self.params.max_length
            )

            self.val_dataset = TeCPredictDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/val.pkl",
                tokenizer=self.tokenizer,
                max_length=self.params.max_length
            )

        if stage == 'test' or stage is "predict":
            self.test_dataset = TeCPredictDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/test.pkl",
                tokenizer=self.tokenizer,
                max_length=self.params.max_length

            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers
        )

    def predict_dataloader(self):
        return self.test_dataloader()
