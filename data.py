import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def encode(text: str):
    return list(text.encode("utf-8", errors="replace"))


def decode(tokens):
    return bytes(tokens).decode("utf-8", errors="replace")


class WikiCharDataset(Dataset):
    def __init__(self, split="train", block_size=256, val_frac=0.05, cache_path="data_cache.npy"):
        self.block_size = block_size

        if os.path.exists(cache_path):
            data = np.load(cache_path)
        else:
            from datasets import load_dataset

            ds = load_dataset("awinml/wikipedia_simple_1k", split="train")
            buf = []
            for r in ds:
                buf.extend(encode(r.get("text", "") + "\n"))
            data = np.array(buf, dtype=np.uint8)
            np.save(cache_path, data)

        n_val = int(len(data) * val_frac)

        self.data = data[:n_val] if split == "val" else data[n_val:]

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_loaders(block_size=256, batch_size=64):
    train = WikiCharDataset("train", block_size)
    val = WikiCharDataset("val", block_size)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
    )