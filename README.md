# Small World Model — NextLat Mini Pipeline (Transformer A)

A ~1M parameter transformer trained with a **Next-Latent Prediction (NextLat)** objective on Wikipedia Simple, upgraded with modern Transformer A components (SDPA attention, stabilized KL, and multi-step latent dynamics).

---

## Overview

This project implements a compact “world model” transformer that learns:

* Next-token prediction (standard autoregressive language modeling)
* Multi-step latent-state prediction (8-step dynamics rollout)
* Distributional alignment via KL divergence between latent-induced token distributions

The key idea is to encourage internal state consistency over *multiple future steps*, rather than only one-step prediction, while maintaining a stable, efficient transformer design.

---

## Architecture

```
vocab_size = 256   (byte-level)
block_size  = 256

n_layer     = 4
n_head      = 4
n_kv_head   = 2
n_embd      = 128

latent_steps = 8
kl_temp      = 2.0

Total ≈ 850K–900K parameters
```

### Transformer (A) components

* SDPA attention (`torch.nn.functional.scaled_dot_product_attention`)

  * FlashAttention acceleration when available
* RMSNorm (pre-norm stability)
* RoPE (rotary positional embeddings)
* GQA (Grouped Query Attention)
* SwiGLU MLP
* Weight tying (embedding ↔ LM head)
* Multi-step latent dynamics model (8-step rollout)

---

## Loss function

```
L_total = L_ntp + λ_h · L_latent + λ_kl · L_kl
```

### Components

* **L_ntp**

  * Standard next-token cross entropy loss

* **L_latent**

  * Multi-step latent prediction loss (K = 8)
  * Smooth L1 loss over iterative rollout
  * Stop-gradient applied to target states

* **L_kl**

  * KL divergence between token distributions induced by latent states
  * Temperature-scaled for stability

---

## Setup

```bash
pip install torch datasets numpy
pip install wandb   # optional logging
```

---

## Training

```bash
# quick sanity run
python train.py --steps 500 --batch_size 16

# full training
python train.py

# logging (optional)
python train.py --wandb --project world-model

# resume
python train.py --resume checkpoints/step_005000.pt

# loss tuning
python train.py --lambda_h 0.2 --lambda_kl 0.05
```

---

## Inference

### CLI

```bash
python generate.py \
  --ckpt checkpoints/final.pt \
  --prompt "The history of" \
  --max_new_tokens 300 \
  --temperature 0.8 \
  --top_k 40
```

---

### Python API

```python
import torch
from model import WorldModel, ModelConfig
from data import decode

ckpt = torch.load("checkpoints/final.pt", map_location="cpu")
args = ckpt["args"]

cfg = ModelConfig(
    vocab_size=args["vocab_size"],
    block_size=args["block_size"],
    n_layer=args["n_layer"],
    n_head=args["n_head"],
    n_kv_head=args.get("n_kv_head", 2),
    n_embd=args["n_embd"],
)

model = WorldModel(cfg)
model.load_state_dict(ckpt["model"])
model.eval()

def generate(prompt):
    idx = torch.tensor([list(prompt.encode())])
    out = model.generate(
        idx,
        max_new_tokens=300,
        temperature=0.8,
        top_k=40,
    )
    return decode(out[0].tolist())

print(generate("In the beginning"))
```

---

## File structure

```
world_model/
├── config.py     # model hyperparameters
├── model.py      # Transformer A + latent dynamics
├── data.py       # dataset + byte encoding
├── train.py      # training loop + eval + checkpoints
├── generate.py   # sampling / inference
└── README.md
```

---

## Key design choices

| Component                 | Reason                                       |
| ------------------------- | -------------------------------------------- |
| Byte-level tokens         | fixed vocab (256), minimal complexity        |
| RMSNorm                   | stable deep training                         |
| RoPE                      | better extrapolation than learned embeddings |
| SDPA attention            | fused + faster + memory efficient            |
| GQA                       | reduces KV cost during inference             |
| SwiGLU                    | stronger MLP expressivity                    |
| Multi-step latent rollout | improves temporal consistency                |
| KL temperature scaling    | prevents divergence during distillation      |
| Stop-gradient targets     | avoids representation collapse               |

---

## Expected training behavior

After ~10k steps (CPU or small GPU):

* coherent short Wikipedia-style paragraphs
* improved local consistency vs 1-step latent models
* reduced KL instability (if λ_kl tuned properly)
* smoother long-range generation due to SDPA + RoPE

Typical ranges:

* `val/ntp`: ~1.5–2.0
* `val/latent`: < 0.1
* `val/kl`: stable if λ_kl ≤ 0.05

---

## Inference notes

* Context window: 256 tokens
* Byte-level decoding (raw UTF-8)
* Recommended sampling:

  * temperature: 0.7–0.9
  * top_k: 20–50
* SDPA improves generation stability vs manual attention

---

## Optional extensions

* KV cache for faster decoding
* FlashAttention v2 integration (if not already enabled)
* Increase latent_steps (8 → 16 for planning-like behavior)
* Curriculum learning on sequence length
* KL warmup schedule (important for stability)
* Replace dynamics MLP with small transformer over latent space

---

## Summary

This is a compact Transformer A system combining:

* modern efficient attention (SDPA + GQA)
* stable normalization and positional encoding (RMSNorm + RoPE)
* improved feedforward design (SwiGLU)
* and a multi-step latent dynamics objective (NextLat)

The result is a small but structurally expressive world model that learns both token-level prediction and short-horizon latent dynamics consistency.

---
