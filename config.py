class ModelConfig:
    def __init__(
        self,
        vocab_size=256,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        dropout=0.0,
        latent_steps=8,
        kl_temp=2.0,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head

        self.n_embd = n_embd
        self.dropout = dropout

        self.latent_steps = latent_steps
        self.kl_temp = kl_temp