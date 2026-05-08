
# NextLat Mini Pipeline — Transformer A World Model

A ~1.1M parameter transformer trained with a **Next-Latent Prediction (NextLat)** objective on Wikipedia Simple, upgraded with modern Transformer A components (SDPA attention, RoPE, GQA, stabilized KL, and multi-step latent dynamics).

---

## Overview

This project implements a compact "world model" transformer that learns:

- **Next-token prediction** (standard autoregressive language modeling)
- **Multi-step latent-state prediction** (8-step dynamics rollout used during training)
- **Distributional alignment** via KL divergence between latent-induced token distributions

The key idea is to encourage internal state consistency over *multiple future steps*, rather than only one-step prediction, while maintaining a stable, efficient transformer design.

---

## Architecture

```
Model Configuration:
├─ Vocab size      : 256 (byte-level)
├─ Block size      : 256 tokens
├─ Model dim       : 256
├─ Layers          : 4
├─ Attention heads : 4 (KV heads: 2 - GQA)
├─ Latent steps    : 8 (training) / 16 (config default)
├─ KL temperature  : 0.5
└─ Total params    : ~1.1M

Parameter Breakdown:
├─ Embedding           : ~65,536
├─ Transformer blocks  : (4 layers)
├─ LM Head (tied)      : 0
└─ Latent Dynamics     : (separate MLP)
```

> **Note**: Exact parameter counts are printed in the Initialization Report at startup.

### Transformer (A) Components

- **SDPA Attention** (`torch.nn.functional.scaled_dot_product_attention`)
  - FlashAttention acceleration when available
  - Memory-efficient attention implementation
- **RMSNorm** - Pre-norm stability
- **RoPE** - Rotary positional embeddings for better extrapolation
- **GQA** - Grouped Query Attention (4 heads, 2 KV heads)
- **SwiGLU** - Enhanced MLP expressivity
- **Weight tying** - Embedding ↔ LM head sharing
- **Multi-step latent dynamics** - Learned rollout over K steps

---

## Loss Function

```
L_total = L_ntp + λ_h · L_latent + λ_kl · L_kl
```

### Components

- **L_ntp (Next Token Prediction)**
  - Standard cross-entropy loss for autoregressive language modeling

- **L_latent (Latent Dynamics)**
  - Multi-step latent prediction loss (K = 8 steps during training)
  - Smooth L1 loss over iterative rollout
  - Stop-gradient applied to target states

- **L_kl (KL Divergence)**
  - KL divergence between token distributions induced by latent states
  - Temperature-scaled (T=0.5 default) for training stability
  - Free-bits threshold to prevent posterior collapse
  - Configurable annealing schedule (linear / cosine / cyclical)

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| λ_h | 0.1 | Latent dynamics weight |
| λ_kl | 0.0005 | KL divergence weight |
| KL warmup | 5000 steps | Linear ramp-up from 0 → λ_kl |
| KL temperature | 0.5 | Softens distribution matching |
| KL free bits | 0.7 | Minimum KL floor to prevent collapse |
| KL annealing | linear | Schedule type (linear / cosine / cyclical) |

---

## Dependencies

| Package | Min version | Role |
|---------|-------------|------|
| **Python** | 3.10+ | Runtime |
| **torch** | 2.0.0 | Model, training, CUDA/CPU backend |
| **numpy** | 1.17 | Dataset cache binary I/O |
| **datasets** | 4.8.0 | Wikipedia Simple streaming download |
| **huggingface_hub** | 1.14.0 | HF dataset registry (transitive — must be ≥ 1.14 to export `CommitInfo`) |
| **httpx** | 0.28.0 | HTTP client for HF Hub (must be ≥ 0.28 to expose `TimeoutException`) |
| **wandb** | any | *(optional)* Experiment tracking |

A pinned `requirements.txt` is provided:

```bash
pip install -r requirements.txt
pip install wandb   # optional
```

> **Why the explicit HF pins?** Older `huggingface_hub < 1.14` and `httpx < 0.28` ship missing symbols (`CommitInfo`, `TimeoutException`) that break the `datasets` import chain at startup. The `requirements.txt` locks safe minimums.

---

## Setup

```bash
# (Recommended) create and activate a virtual environment
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt

# — or manually —
pip install torch datasets numpy
pip install -U huggingface_hub httpx   # ensure safe minimum versions

pip install wandb   # optional: for experiment tracking

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}'); print('CUDA:', torch.cuda.is_available())"
```

---

## Quick Start

```bash
# 1) Start a fresh run (default: 100 000 max steps, 10 max epochs)
python train.py
# Expected early output:
# 📚 Preparing datasets and data loaders...
# ✅ Data loaders ready | train batches: ... | val batches: ...

# 2) Short sanity run
python train.py --steps 500 --batch_size 16

# 3) Resume from best checkpoint in checkpoint_dir
python train.py --resume best

# 4) Resume from latest numbered checkpoint in checkpoint_dir
python train.py --resume latest

# 5) Resume from an explicit path
python train.py --resume checkpoints/best.pt
```

> Note: `train.py` supports both `--steps` and `--epochs`. Training stops when **either** limit is reached.

---

## Training

### Advanced Options

```bash
# With wandb logging
python train.py --wandb --project world-model

# Evaluation frequency
python train.py --eval_every 250 --log_every 50

# Full validation (slow)
python train.py --eval_batches 0

# Custom checkpoint directory
python train.py --checkpoint_dir ./my_checkpoints

# KL annealing strategy
python train.py --kl_anneal_type cyclical --kl_warmup 10000
```

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 128 | Batch size (tokens/batch = batch_size × block_size) |
| `--steps` | 100000 | Maximum training steps |
| `--epochs` | 10 | Maximum epochs (step limit still applies) |
| `--lr` | 3e-4 | Peak learning rate |
| `--min_lr` | 1e-5 | Floor for cosine LR schedule |
| `--lambda_h` | 0.1 | Latent prediction loss weight |
| `--lambda_kl` | 0.0005 | KL divergence loss weight |
| `--kl_warmup` | 5000 | Steps to linearly increase KL weight |
| `--kl_temp` | 0.5 | KL temperature scaling |
| `--kl_free_bits` | 0.7 | Free-bits floor to prevent KL collapse |
| `--kl_anneal_type` | `linear` | KL schedule: `linear`, `cosine`, or `cyclical` |
| `--grad_clip` | 1.0 | Gradient norm clipping |
| `--weight_decay` | 0.01 | AdamW weight decay |
| `--beta1` | 0.9 | AdamW β₁ |
| `--beta2` | 0.95 | AdamW β₂ |
| `--seed` | 42 | Random seed |
| `--num_workers` | 0 | DataLoader worker processes |
| `--eval_every` | 500 | Steps between evaluations |
| `--eval_batches` | 64 | Validation batches (0 = full dataset) |
| `--sample_tokens` | 80 | Tokens to generate during evaluation |
| `--log_every` | 100 | Steps between console logging |
| `--resume` | `None` | Resume source: `best`, `latest`, or a checkpoint path |
| `--wandb` | off | Enable Weights & Biases logging |
| `--project` | `world-model` | W&B project name |
| `--checkpoint_dir` | `checkpoints` | Directory for saved checkpoints |

### Checkpoint Strategy

Checkpoints are saved automatically in two ways:

| Type | Filename | When saved |
|------|----------|------------|
| **Best model** | `checkpoints/best.pt` | Every time validation loss improves |
| **Progress snapshots** | `checkpoints/checkpoint_NNNNNN.pt` | At each 10% step milestone (10%, 20%, … 100%) |

### Training Log Format

```
[████░░░░░░░░░░░░░░░░]  20.0% | step 020000/100000 | loss 1.8432 | ntp 1.8350 | lat 0.0120 | kl 12.34 | kl_w 0.000500 | lr 0.000285 | best 1.7900@18500 | tok/s 650000 | etime 300s
```

**KL health indicators** visible in the log:

| Icon | Meaning |
|------|---------|
| 🔴 KL TOO HIGH | KL > 300 — normal during early warmup |
| 🟡 KL HIGH | KL 150–300 — monitor but usually fine |
| ⚪ KL collapsed | KL < 0.1 — posterior collapse; try lower `--kl_temp` or higher `--kl_free_bits` |
| 🔴 LAT HIGH | Latent loss > 1.0 — dynamics head diverging |
| ⚪ LAT collapsed | Latent loss < 0.001 after step 1000 — dead latent head |

---

## Inference

### Command Line Interface

```bash
# Basic generation
python generate.py \
  --ckpt checkpoints/best.pt \
  --prompt "The history of" \
  --max_new_tokens 300

# With sampling parameters
python generate.py \
  --ckpt checkpoints/best.pt \
  --prompt "In the beginning" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_k 40
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'`**
  - Install dependencies: `pip install torch datasets numpy`

- **`ImportError: cannot import name 'CommitInfo' from 'huggingface_hub'`**
  - Stale HuggingFace packages. Fix: `pip install -U huggingface_hub httpx datasets`

- **`module 'httpx' has no attribute 'TimeoutException'`**
  - Same root cause as above. Run the upgrade command above.

- **Resume path not found**
  - Prefer `python train.py --resume best` or `python train.py --resume latest`
  - Verify checkpoint exists under `--checkpoint_dir` (default: `checkpoints/`)

- **Command returns immediately in PowerShell**
  - Ensure you are in the project directory with the correct Python env.

- **Command appears to do nothing / hangs silently**
  - Run unbuffered: `python -u train.py`
  - On first run, cache building streams Wikipedia data — you should see `[data] Stream opened — writing cache ...`

- **HuggingFace cache / symlink warning on Windows**
  - `hf_cache` is forced to `./.hf_cache/` inside the project folder.
  - The symlinks warning is harmless; enable Developer Mode in Windows settings to silence it.
  - Optional: set `HF_TOKEN` env var to avoid anonymous rate-limit warnings during dataset download.

- **`trust_remote_code` warning**
  - Harmless informational message from the updated `datasets` library. Training is unaffected.
