import os
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_EN_SUBSET = "20231101.en"


def encode(text: str):
    return list(text.encode("utf-8", errors="replace"))


def decode(tokens):
    return bytes(tokens).decode("utf-8", errors="replace")


@lru_cache(maxsize=4)
def _load_token_data(cache_path: str):
    if os.path.exists(cache_path):
        return np.load(cache_path)

    from datasets import load_dataset

    ds = load_dataset(WIKIPEDIA_DATASET, WIKIPEDIA_EN_SUBSET, split="train")
    buf = []
    for row in ds:
        buf.extend(encode(row.get("text", "") + "\n"))

    data = np.array(buf, dtype=np.uint8)
    np.save(cache_path, data)
    return data


class WikiCharDataset(Dataset):
    def __init__(self, split="train", block_size=256, val_frac=0.05, cache_path="data_cache_wikipedia_en.npy"):
        self.block_size = block_size
        data = _load_token_data(cache_path)
        n_val = int(len(data) * val_frac)
        self.data = data[:n_val] if split == "val" else data[n_val:]

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_loaders(block_size=256, batch_size=64, num_workers=0):
    train = WikiCharDataset("train", block_size)
    val = WikiCharDataset("val", block_size)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val, batch_size=batch_size, num_workers=num_workers, pin_memory=True),
    )
