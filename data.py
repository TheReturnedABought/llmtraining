import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_EN_SUBSET = "20231101.en"
FLUSH_BYTES = 8 * 1024 * 1024  # 8MB


def encode(text: str):
    return list(text.encode("utf-8", errors="replace"))


def decode(tokens):
    return bytes(tokens).decode("utf-8", errors="replace")


def _metadata_path(cache_path: str) -> Path:
    return Path(cache_path).with_suffix(Path(cache_path).suffix + ".len")


def _build_cache_streaming(cache_path: str):
    from datasets import load_dataset

    ds = load_dataset(WIKIPEDIA_DATASET, WIKIPEDIA_EN_SUBSET, split="train", streaming=True)

    total_bytes = 0
    buffer = bytearray()

    with open(cache_path, "wb") as f:
        for row in ds:
            text = row.get("text", "") + "\n"
            buffer.extend(text.encode("utf-8", errors="replace"))

            if len(buffer) >= FLUSH_BYTES:
                f.write(buffer)
                total_bytes += len(buffer)
                buffer.clear()

        if buffer:
            f.write(buffer)
            total_bytes += len(buffer)

    _metadata_path(cache_path).write_text(str(total_bytes), encoding="utf-8")


@lru_cache(maxsize=4)
def _load_token_data(cache_path: str):
    cache_file = Path(cache_path)
    len_file = _metadata_path(cache_path)

    if not cache_file.exists() or not len_file.exists():
        print(f"📥 Cache not found. Building dataset cache at {cache_file}...", flush=True)
        _build_cache_streaming(cache_path)

    total_bytes = int(len_file.read_text(encoding="utf-8").strip())
    if total_bytes <= 0:
        raise ValueError(f"Cache metadata indicates zero bytes: {len_file}")

    return np.memmap(cache_file, dtype=np.uint8, mode="r", shape=(total_bytes,))


class WikiCharDataset(Dataset):
    def __init__(self, split="train", block_size=256, val_frac=0.05, cache_path="data_cache_wikipedia_en.bin"):
        self.block_size = block_size
        data = _load_token_data(cache_path)
        n_val = int(len(data) * val_frac)
        self.data = data[:n_val] if split == "val" else data[n_val:]

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_loaders(block_size=256, batch_size=64, num_workers=0):
    train = WikiCharDataset("train", block_size)
    val = WikiCharDataset("val", block_size)

    if len(train) == 0:
        raise ValueError(
            "Training split has zero samples after block sizing. "
            "Cache may be too small/corrupted for the configured block_size."
        )

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val, batch_size=batch_size, num_workers=num_workers, pin_memory=True),
    )
