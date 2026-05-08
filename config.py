import torch

class ModelConfig:
    def __init__(
        self,
        vocab_size=256,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=256,
        latent_steps=16,
        kl_temp=0.5,  # Lower default for stability
        kl_free_bits=0.5,  # Prevent complete collapse
        kl_anneal_type='cyclical',  # Better warmup strategy
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.latent_steps = latent_steps
        self.kl_temp = kl_temp
        self.kl_free_bits = kl_free_bits
        self.kl_anneal_type = kl_anneal_type