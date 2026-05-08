from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2
    n_embd: int = 256
    latent_steps: int = 16
    kl_temp: float = 0.5          # Lower default for stability
    kl_free_bits: float = 0.5     # Prevent complete collapse
    kl_anneal_type: str = "cyclical"  # Better warmup strategy
    swa_window: int = 256         # Sliding window size (all layers except last)
    batch_size: int = 128         # Per-step VRAM cost; tune this for your GPU
    grad_accum: int = 1           # effective_batch = batch_size × grad_accum
    num_workers: int = 0          # 0 = safe on Windows; set >0 on Linux/WSL only
