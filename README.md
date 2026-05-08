
# NextLat Mini Pipeline — Transformer A World Model

A ~1.1M parameter transformer trained with a **Next-Latent Prediction (NextLat)** objective on Wikipedia Simple, upgraded with modern Transformer A components (SDPA attention, RoPE, GQA, stabilized KL, and multi-step latent dynamics).

---

## Overview

This project implements a compact "world model" transformer that learns:

- **Next-token prediction** (standard autoregressive language modeling)
- **Multi-step latent-state prediction** (8-step dynamics rollout)
- **Distributional alignment** via KL divergence between latent-induced token distributions

The key idea is to encourage internal state consistency over *multiple future steps*, rather than only one-step prediction, while maintaining a stable, efficient transformer design.

---

## Architecture

```
Model Configuration:
├─ Vocab size      : 256 (byte-level)
├─ Block size      : 256 tokens
├─ Model dim       : 128
├─ Layers          : 4
├─ Attention heads : 4 (KV heads: 2 - GQA)
├─ Latent steps    : 8
├─ KL temperature  : 2.0
└─ Total params    : 1,115,648

Parameter Breakdown:
├─ Embedding           : 32,768
├─ Transformer blocks  : 984,064 (4 layers)
├─ LM Head (tied)      : 0
└─ Latent Dynamics     : 98,688
```

### Transformer (A) Components

- **SDPA Attention** (`torch.nn.functional.scaled_dot_product_attention`)
  - FlashAttention acceleration when available
  - Memory-efficient attention implementation
- **RMSNorm** - Pre-norm stability
- **RoPE** - Rotary positional embeddings for better extrapolation
- **GQA** - Grouped Query Attention (4 heads, 2 KV heads)
- **SwiGLU** - Enhanced MLP expressivity
- **Weight tying** - Embedding ↔ LM head sharing
- **Multi-step latent dynamics** - 8-step learned rollout

---

## Loss Function

```
L_total = L_ntp + λ_h · L_latent + λ_kl · L_kl
```

### Components

- **L_ntp (Next Token Prediction)**
  - Standard cross-entropy loss for autoregressive language modeling

- **L_latent (Latent Dynamics)**
  - Multi-step latent prediction loss (K = 8 steps)
  - Smooth L1 loss over iterative rollout
  - Stop-gradient applied to target states

- **L_kl (KL Divergence)**
  - KL divergence between token distributions induced by latent states
  - Temperature-scaled (T=2.0) for training stability
  - Linear warmup over first 1000 steps

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| λ_h | 0.1 | Latent dynamics weight |
| λ_kl | 0.01 | KL divergence weight |
| KL warmup | 1000 steps | Linear ramp-up from 0 → λ_kl |
| KL temperature | 2.0 | Softens distribution matching |

---

## Setup

```bash
# (Recommended) create and activate a virtual environment
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install torch datasets numpy
pip install wandb   # optional: for experiment tracking

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

## Quick Start (Commands that work with current train.py)

```bash
# 1) Start a fresh run (default: 100000 max steps, 10 max epochs)
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

> Note: `train.py` supports both `--steps` and `--epochs`. Training stops when either limit is reached.

---

## Training

### Advanced Options

```bash
# With wandb logging
python train.py --wandb --project world-model

# Evaluation frequency
python train.py --eval_every 250 --log_every 50

# Full validation (slow, use with --eval_batches 0)
python train.py --eval_batches 0

# Custom checkpoint directory
python train.py --checkpoint_dir ./my_checkpoints
```

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 128 | Batch size (tokens/batch = batch_size × block_size) |
| `--steps` | 100000 | Maximum training steps |
| `--epochs` | 10 | Maximum epochs (step limit still applies) |
| `--lr` | 3e-4 | Learning rate |
| `--min_lr` | 1e-5 | Floor for cosine LR schedule |
| `--lambda_h` | 0.1 | Latent prediction loss weight |
| `--lambda_kl` | 0.0005 | KL divergence loss weight |
| `--kl_warmup` | 5000 | Steps to linearly increase KL weight |
| `--kl_temp` | 0.5 | KL temperature scaling |
| `--eval_every` | 500 | Steps between evaluations |
| `--eval_batches` | 64 | Validation batches (0 = full dataset) |
| `--sample_tokens` | 80 | Tokens to generate during evaluation |
| `--log_every` | 100 | Steps between console logging |
| `--resume` | `None` | Resume source (`best`, `latest`, or checkpoint path) |

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
  - Install dependencies first: `pip install torch datasets numpy`.
- **Resume path not found**
  - Prefer `python train.py --resume best` or `python train.py --resume latest`.
  - Verify checkpoint exists under `--checkpoint_dir` (default: `checkpoints`).
- **Command returns immediately in PowerShell**
  - Ensure you are in the project directory and using the same Python env where dependencies were installed.
- **Command appears to do nothing**
  - Run unbuffered logging: `python -u train.py`.
  - On first run, cache creation can take time; you should now see `📥 Cache not found. Building dataset cache ...` while it streams Wikipedia data.
  - You should see startup logs like `Preparing datasets` and `Initialization Report` before training begins.

