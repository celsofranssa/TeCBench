import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TeCPredictDataset(Dataset):
    """Text Classification Dataset.
    """

    def __init__(self, samples, ids_path, tokenizer, max_length):
        super(TeCPredictDataset, self).__init__()
        self.samples = samples
        self._load_ids(ids_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        sample_idx = self.ids[idx]
        return self._encode(
            self.samples[sample_idx]
        )
