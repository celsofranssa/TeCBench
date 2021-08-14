import json
import pickle

import torch
from torch.utils.data import Dataset


class TeCDataset(Dataset):
    """MNIST Dataset.
    """

    def __init__(self, dataset_path, ids_path, tokenizer, max_length):
        super(TeCDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._init_dataset(dataset_path,ids_path)

    def _init_dataset(self, dataset_path, ids_path):
        with open(dataset_path, "rb") as dataset_file:
            self.samples = pickle.load(dataset_file)
        with open(ids_path, "rb") as ids_file:
            self.ids = pickle.load(ids_file)

    def _encode(self, sample):
        return {
            "idx": sample["idx"],
            "text": torch.tensor(
                self.tokenizer.encode(text=sample["text"], max_length=self.max_length, padding="max_length",
                                      truncation=True)
            ),
            "cls": sample["cls"]
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        return self._encode(
            self.samples[sample_id]
        )
