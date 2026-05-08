"""
data.py  —  byte-level Wikipedia dataloader for the NextLat Mini Pipeline.

Public API
----------
    get_loaders(block_size, batch_size, num_workers=0) -> (train_loader, val_loader)
    encode(text: str)  -> list[int]   (UTF-8 bytes 0-255)
    decode(ids: list)  -> str
"""

import os
import importlib.util
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── HF caches stay inside the project folder ─────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HF_CACHE   = os.path.join(_SCRIPT_DIR, ".hf_cache")
os.environ["HF_HOME"]           = _HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(_HF_CACHE, "datasets")
os.environ["HF_HUB_CACHE"]      = os.path.join(_HF_CACHE, "hub")
# Disable parallelism in tokenisers (avoids forking issues on Windows)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_CACHE_BIN    = os.path.join(_SCRIPT_DIR, "data_cache_wikipedia_en.bin")
_CACHE_LEN    = _CACHE_BIN + ".len"
_WIKI_DATASET = "wikimedia/wikipedia"
_WIKI_CONFIG  = "20231101.simple"
_VAL_FRACTION = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# encode / decode
# ─────────────────────────────────────────────────────────────────────────────

def encode(text: str) -> list:
    return list(text.encode("utf-8", errors="replace"))

def decode(ids: list) -> str:
    return bytes(ids).decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_cache_len() -> int:
    try:
        with open(_CACHE_LEN) as f:
            n = int(f.read().strip())
        return n if n > 0 else 0
    except Exception:
        return 0


def _build_cache(path: str) -> int:
    """Stream Wikipedia Simple -> flat uint8 binary file."""

    # ---- import check (find_spec: no side-effects) --------------------------
    print("   [data] Checking datasets package ...", flush=True)
    if importlib.util.find_spec("datasets") is None:
        raise ImportError(
            "  'datasets' not found.  Fix:  pip install datasets"
        )

    print("   [data] Importing load_dataset ... (may take 30-60s on first run)", flush=True)
    try:
        from datasets import load_dataset
        import huggingface_hub, httpx
        print("   [data] load_dataset imported OK", flush=True)
        print("   [data]   huggingface_hub=" + huggingface_hub.__version__
              + "  httpx=" + httpx.__version__, flush=True)
    except Exception as e:
        import sys
        print("   [data] IMPORT FAILED: " + type(e).__name__ + ": " + str(e), flush=True)
        print("   [data] Fix: pip install -U huggingface_hub httpx datasets", flush=True)
        sys.exit(1)

    print("   [data] Calling load_dataset (streaming) ...", flush=True)
    print("   [data] Source : " + _WIKI_DATASET + "  config=" + _WIKI_CONFIG, flush=True)
    print("   [data] Output : " + path, flush=True)

    # streaming=True: no full download up-front; data arrives incrementally
    ds = load_dataset(
        _WIKI_DATASET,
        _WIKI_CONFIG,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    print("   [data] Stream opened — writing cache ...", flush=True)

    total    = 0
    articles = 0
    with open(path, "wb") as fout:
        for sample in ds:
            text = sample.get("text", "").strip()
            if not text:
                continue
            raw = text.encode("utf-8", errors="replace")
            fout.write(raw)
            total    += len(raw)
            articles += 1
            if articles % 5000 == 0:
                print(
                    "   [data] " + str(articles) + " articles | "
                    + str(round(total / 1e6, 1)) + " MB",
                    flush=True,
                )

    print(
        "   [data] Cache built: " + str(articles) + " articles, "
        + str(total) + " bytes",
        flush=True,
    )
    return total


def _ensure_cache() -> int:
    """Return corpus byte-length, building the cache if needed."""

    # Fast path: valid cache already on disk
    print("   [data] Reading cache metadata ...", flush=True)
    n = _read_cache_len()

    if n > 0:
        print("   [data] Verifying cache file ...", flush=True)
        if os.path.exists(_CACHE_BIN):
            actual = os.path.getsize(_CACHE_BIN)
            if actual == n:
                print("   [data] Cache OK: " + str(n) + " bytes", flush=True)
                return n
            print(
                "   [data] Size mismatch (disk=" + str(actual)
                + " recorded=" + str(n) + ") — rebuilding.",
                flush=True,
            )
        else:
            print("   [data] Cache file missing — rebuilding.", flush=True)

    # Build from scratch
    print("   [data] Building cache (first run only) ...", flush=True)
    n = _build_cache(_CACHE_BIN)
    with open(_CACHE_LEN, "w") as f:
        f.write(str(n))
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ByteCorpusDataset(Dataset):
    """Non-overlapping byte windows: each item -> (x, y) LongTensors (block_size,)."""

    def __init__(self, data: np.ndarray, block_size: int):
        if len(data) < block_size + 1:
            raise ValueError(
                "Slice too small: " + str(len(data))
                + " bytes for block_size=" + str(block_size)
            )
        self.data       = data
        self.block_size = block_size
        self.n_windows  = (len(data) - 1) // block_size

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        s     = idx * self.block_size
        chunk = self.data[s : s + self.block_size + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


# ─────────────────────────────────────────────────────────────────────────────
# Public
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(block_size: int, batch_size: int, num_workers: int = 0):
    """Return (train_loader, val_loader).  Never calls sys.exit."""

    print("   [data] Reading/building corpus ...", flush=True)
    total_bytes = _ensure_cache()

    print("   [data] Memory-mapping " + str(total_bytes) + " bytes ...", flush=True)
    corpus = np.memmap(_CACHE_BIN, dtype=np.uint8, mode="r", shape=(total_bytes,))

    # 95 % train / 5 % val  (val from tail)
    raw_split = int(total_bytes * (1.0 - _VAL_FRACTION))
    min_slice = block_size + 1
    split_idx = max(min_slice, min(raw_split, total_bytes - min_slice))

    train_ds = ByteCorpusDataset(corpus[:split_idx], block_size)
    val_ds   = ByteCorpusDataset(corpus[split_idx:],  block_size)

    print(
        "   [data] Train: " + str(round(split_idx / 1e6, 1))
        + " MB -> " + str(len(train_ds)) + " windows",
        flush=True,
    )
    print(
        "   [data] Val  : " + str(round((total_bytes - split_idx) / 1e6, 1))
        + " MB -> " + str(len(val_ds)) + " windows",
        flush=True,
    )

    pin = torch.cuda.is_available()
    kw  = dict(batch_size=batch_size, num_workers=num_workers,
               pin_memory=pin, drop_last=True)

    print("   [data] Creating DataLoaders ...", flush=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    return train_loader, val_loader