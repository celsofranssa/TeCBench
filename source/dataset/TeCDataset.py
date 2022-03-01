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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.samples[
            self.ids[idx]
        ]
