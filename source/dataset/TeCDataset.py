import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TeCDataset(Dataset):
    """Text Classification Dataset.
    """

    def __init__(self, samples, cls_samples, ids_path, tokenizer, max_length):
        super(TeCDataset, self).__init__()
        self.samples = []
        self._load_ids(ids_path)
        self.cls_weight = {0: 6, 1: 1, 2: 9, 3: 2}
        print(f"\nAug:\n{self.cls_weight}\n")
        self.reshape_samples(samples, cls_samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def reshape_samples(self, samples, cls_samples):
        for idx in tqdm(self.ids, desc="Reshaping samples"):
            sample = samples[idx]
            self.samples.append(sample)
            self.samples.extend(
                random.sample(cls_samples.get(sample["cls"], []), k=self.cls_weight[sample["cls"]])
            )

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
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )
