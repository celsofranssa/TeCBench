import json

import torch
from torch.utils.data import Dataset


class TeCDataset(Dataset):
    """MNIST Dataset.
    """

    def __init__(self, dataset_path, tokenizer, max_length):
        super(TeCDataset, self).__init__()
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length=max_length
        self._init_dataset(dataset_path)

    def _init_dataset(self, dataset_path):
        with open(dataset_path, "r") as dataset_file:
            for line in dataset_file:
                sample = json.loads(line)
                self.samples.append({
                    "idx": sample["idx"],
                    "text": sample["text"],
                    "cls": int(sample["cls"])
                })

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
        return self._encode(self.samples[idx])
