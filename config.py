from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2
    n_embd: int = 256
    latent_steps: int = 16
    kl_temp: float = 0.5  # Lower default for stability
    kl_free_bits: float = 0.5  # Prevent complete collapse
    kl_anneal_type: str = "cyclical"  # Better warmup strategy
