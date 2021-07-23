import pytorch_lightning as pl

from torch.utils.data import DataLoader

from source.dataset.TeCDataset import TeCDataset


class TeCDataModule(pl.LightningDataModule):

    def __init__(self, params, tokenizer, fold):
        super(TeCDataModule, self).__init__()
        self.params = params
        self.tokenizer=tokenizer
        self.fold = fold

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self.train_dataset = TeCDataset(
                dataset_path=self.params.dir + f"fold_{self.fold}/train.jsonl",
                tokenizer=self.tokenizer,
                max_length=self.params.max_length
            )

            self.val_dataset = TeCDataset(
                dataset_path=self.params.dir + f"fold_{self.fold}/test.jsonl",
                tokenizer=self.tokenizer,
                max_length=self.params.max_length
            )

        if stage == 'test' or stage is None:
            self.test_dataset = TeCDataset(
                dataset_path=self.params.dir + f"fold_{self.fold}/test.jsonl",
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
