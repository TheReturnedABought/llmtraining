"""Utilities for building and loading a byte-level Wikipedia training cache."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ID = "wikimedia/wikipedia"
DATA_PREFIX = "20231101.en/"
CACHE_PATH = "data_cache_wikipedia_en.bin"
FLUSH_BYTES = 8 * 1024 * 1024  # 8 MB


def encode(text: str):
    return list(text.encode("utf-8", errors="replace"))


def decode(tokens):
    return bytes(tokens).decode("utf-8", errors="replace")


def _meta_path(p: str) -> Path:
    return Path(p).with_suffix(Path(p).suffix + ".len")


def _die(msg: str):
    raise RuntimeError(msg)


def _wipe(cache_path: str):
    cache_file = Path(cache_path)
    meta_file = _meta_path(cache_path)
    if cache_file.exists():
        cache_file.unlink()
    if meta_file.exists():
        meta_file.unlink()


def _log(msg: str):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _build_cache(cache_path: str):
    _wipe(cache_path)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        _die(f"huggingface_hub not found. Run: pip install huggingface_hub ({exc})")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        _die(f"pyarrow not found. Run: pip install pyarrow ({exc})")

    _log(f"[data] Listing parquet files in {REPO_ID} ...")
    try:
        all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
    except Exception as exc:
        _die(f"Could not list repo files: {exc}")

    parquet_files = sorted(
        f for f in all_files if f.startswith(DATA_PREFIX) and f.endswith(".parquet")
    )

    if not parquet_files:
        _die(f"No parquet files found under '{DATA_PREFIX}'. Check the dataset path.")

    _log(f"[data] Found {len(parquet_files)} parquet files. Starting download ...")
    _log("[data] Files are cached by huggingface_hub (~20 GB total).")

    total_bytes = 0
    buffer = bytearray()

    with open(cache_path, "wb") as out:
        for i, filename in enumerate(parquet_files):
            _log(f"[data] [{i + 1}/{len(parquet_files)}] {filename}")
            try:
                local = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    repo_type="dataset",
                )
            except Exception as exc:
                _wipe(cache_path)
                _die(f"Download failed for {filename}: {exc}")

            try:
                table = pq.read_table(local, columns=["text"])
                for text in table.column("text").to_pylist():
                    buffer.extend((text + "\n").encode("utf-8", errors="replace"))
                    if len(buffer) >= FLUSH_BYTES:
                        out.write(buffer)
                        total_bytes += len(buffer)
                        buffer.clear()
            except Exception as exc:
                _wipe(cache_path)
                _die(f"Failed reading {filename}: {exc}")

        if buffer:
            out.write(buffer)
            total_bytes += len(buffer)

    if total_bytes == 0:
        _wipe(cache_path)
        _die("Wrote 0 bytes. Something went wrong.")

    _meta_path(cache_path).write_text(str(total_bytes), encoding="utf-8")


@lru_cache(maxsize=4)
def _load_data(cache_path: str):
    cache_file = Path(cache_path).resolve()
    len_file = _meta_path(str(cache_file))

    if not cache_file.exists() or not len_file.exists():
        print(f"📥 Cache not found. Building dataset cache at {cache_file}...", flush=True)
        _build_cache(str(cache_file))

    total_bytes = int(len_file.read_text(encoding="utf-8").strip())
    if total_bytes <= 0:
        raise ValueError(f"Cache metadata indicates zero bytes: {len_file}")

    return np.memmap(cache_file, dtype=np.uint8, mode="r", shape=(total_bytes,))


class WikiCharDataset(Dataset):
    def __init__(self, split="train", block_size=256, val_frac=0.05, cache_path=CACHE_PATH):
        self.block_size = block_size
        data = _load_data(cache_path)
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
    train_ds = WikiCharDataset("train", block_size)
    val_ds = WikiCharDataset("val", block_size)

    if len(train_ds) == 0:
        raise ValueError(
            "Training split has zero samples after block sizing. "
            "Cache may be too small/corrupted for the configured block_size."
        )

    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(num_workers == 0),
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(num_workers == 0),
        ),
    )
