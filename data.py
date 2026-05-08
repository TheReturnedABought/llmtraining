"""
data.py — downloads Wikipedia parquet files directly via huggingface_hub + pyarrow.
No 'datasets' library required. Both huggingface_hub and pyarrow are already installed.
"""
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ID     = "wikimedia/wikipedia"
DATA_PREFIX = "20231101.en/"
CACHE_PATH  = "data_cache_wikipedia_en.bin"
FLUSH_BYTES = 8 * 1024 * 1024  # 8 MB


def encode(text: str):
    return list(text.encode("utf-8", errors="replace"))


def decode(tokens):
    return bytes(tokens).decode("utf-8", errors="replace")


def _meta_path(p: str) -> Path:
    return Path(p).with_suffix(Path(p).suffix + ".len")


def _log(msg: str):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _die(msg: str):
    _log(f"[data] ERROR: {msg}")
    os._exit(1)


def _wipe(cache_path: str):
    Path(cache_path).unlink(missing_ok=True)
    _meta_path(cache_path).unlink(missing_ok=True)


def _cache_valid(cache_path: str) -> bool:
    cf = Path(cache_path)
    mf = _meta_path(cache_path)
    if not cf.exists() or not mf.exists():
        return False
    try:
        return int(mf.read_text(encoding="utf-8").strip()) > 0 and cf.stat().st_size > 0
    except Exception:
        return False


def _build_cache(cache_path: str):
    _wipe(cache_path)

    # ── imports (both confirmed installed) ───────────────────────────────────
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        _die("huggingface_hub not found. Run: pip install huggingface_hub")

    try:
        import pyarrow.parquet as pq
    except ImportError:
        _die("pyarrow not found. Run: pip install pyarrow")

    # ── list parquet files ───────────────────────────────────────────────────
    _log(f"[data] Listing parquet files in {REPO_ID} ...")
    try:
        all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
    except Exception as e:
        _die(f"Could not list repo files: {e}")

    parquet_files = sorted(
        f for f in all_files
        if f.startswith(DATA_PREFIX) and f.endswith(".parquet")
    )

    if not parquet_files:
        _die(f"No parquet files found under '{DATA_PREFIX}'. Check the dataset path.")

    _log(f"[data] Found {len(parquet_files)} parquet files. Starting download ...")
    _log( "[data] Files are cached by huggingface_hub (~20 GB total).")

    # ── download + encode ────────────────────────────────────────────────────
    total_bytes = 0
    buffer      = bytearray()

    with open(cache_path, "wb") as out:
        for i, filename in enumerate(parquet_files):
            _log(f"[data] [{i+1}/{len(parquet_files)}] {filename}")
            try:
                local = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    repo_type="dataset",
                )
            except Exception as e:
                _wipe(cache_path)
                _die(f"Download failed for {filename}: {e}")

            try:
                table = pq.read_table(local, columns=["text"])
                for text in table.column("text").to_pylist():
                    buffer.extend((text + "\n").encode("utf-8", errors="replace"))
                    if len(buffer) >= FLUSH_BYTES:
                        out.write(buffer)
                        total_bytes += len(buffer)
                        buffer.clear()
            except Exception as e:
                _wipe(cache_path)
                _die(f"Failed reading {filename}: {e}")

        if buffer:
            out.write(buffer)
            total_bytes += len(buffer)

    if total_bytes == 0:
        _wipe(cache_path)
        _die("Wrote 0 bytes. Something went wrong.")

    _meta_path(cache_path).write_text(str(total_bytes), encoding="utf-8")
    _log(f"[data] Cache ready: {total_bytes / 1e9:.1f} GB -> {cache_path}")


def _load_data(cache_path: str) -> np.memmap:
    if not _cache_valid(cache_path):
        _log(f"[data] Cache missing or invalid. Building at {cache_path} ...")
        _build_cache(cache_path)
    total_bytes = int(_meta_path(cache_path).read_text(encoding="utf-8").strip())
    return np.memmap(Path(cache_path), dtype=np.uint8, mode="r", shape=(total_bytes,))


class WikiCharDataset(Dataset):
    def __init__(self, split="train", block_size=256, val_frac=0.05,
                 cache_path=CACHE_PATH):
        self.block_size = block_size
        data  = _load_data(cache_path)
        n_val = int(len(data) * val_frac)
        self.data = data[:n_val] if split == "val" else data[n_val:]

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size]
        x = torch.tensor(chunk, dtype=torch.long)
        y = x.clone()
        return x, y


def get_loaders(block_size=256, batch_size=64, num_workers=0):
    train_ds = WikiCharDataset("train", block_size)
    val_ds   = WikiCharDataset("val",   block_size)
    if len(train_ds) == 0:
        _die("Training split has zero samples. Delete cache files and rerun.")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=(num_workers == 0)),
        DataLoader(val_ds,   batch_size=batch_size,
                   num_workers=num_workers, pin_memory=(num_workers == 0)),
    )