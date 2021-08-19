import json
import pickle

import torch
from torch.utils.data import Dataset


class TeCDataset(Dataset):
    """Text Classification Dataset.
    """

    def __init__(self, samples, ids_path, tokenizer, max_length):
        super(TeCDataset, self).__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._load_ids(ids_path)

    def _load_ids(self, ids_path):
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
