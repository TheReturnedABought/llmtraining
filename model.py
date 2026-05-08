import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q, k):
        seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    def _apply_rotary(self, x, cos, sin):
        x_rot = x[..., ::2]
        x_pass = x[..., 1::2]
        x_neg = torch.stack([-x_pass, x_rot], dim=-1).flatten(-2)
        return (x * cos) + (x_neg * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        # GQA: repeat KV heads
        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        # Flash attention or standard
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
        )
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Tokenizer(nn.Module):
    """Byte-level tokenizer with learnable embeddings"""

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, idx):
        return self.ln(self.embed(idx))


class LatentDynamics(nn.Module):
    """Stochastic latent dynamics model (NextLat)"""

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.latent_steps = config.latent_steps
        self.kl_temp = config.kl_temp

        # Encoder: token embeddings -> latent distribution
        self.encoder = nn.Sequential(
            nn.Linear(config.n_embd, 2 * config.n_embd),
            nn.GELU(),
            nn.Linear(2 * config.n_embd, 2 * config.n_embd),
        )

        # Dynamics predictor: predicts next latent state from current
        self.predictor = nn.GRU(
            config.n_embd, config.n_embd,
            num_layers=1, batch_first=True
        )

        # Decoder: latent -> reconstructed token features
        self.decoder = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd),
        )

        # Learnable prior
        self.prior_mean = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.prior_logvar = nn.Parameter(torch.zeros(1, 1, config.n_embd))

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def kl_divergence(self, mean, logvar, prior_mean=None, prior_logvar=None):
        if prior_mean is None:
            prior_mean = self.prior_mean
        if prior_logvar is None:
            prior_logvar = self.prior_logvar

        var = torch.exp(logvar)
        prior_var = torch.exp(prior_logvar)

        kl = 0.5 * (
                prior_logvar - logvar +
                (var + (mean - prior_mean).pow(2)) / prior_var - 1
        )
        return kl.sum(-1).mean()

    def forward(self, h, lambda_kl=0.01):
        B, T, C = h.shape

        # Encode to latent distribution
        enc_out = self.encoder(h)
        mean, logvar = enc_out.chunk(2, dim=-1)

        # Apply KL temperature for smoother latent space
        mean = mean / self.kl_temp
        logvar = logvar - math.log(self.kl_temp)

        # Sample latent state
        z = self.reparameterize(mean, logvar)

        # NEXT LATENT PREDICTION: Predict next latent from current
        if T < self.latent_steps:
            # Short sequence: just decode current latent directly
            z_predicted = z
            latent_pred_loss = torch.tensor(0.0, device=h.device)
        else:
            # Pad sequence to be divisible by latent_steps if needed
            if T % self.latent_steps != 0:
                pad_len = self.latent_steps - (T % self.latent_steps)
                z_padded = F.pad(z, (0, 0, 0, pad_len))
                T_padded = T + pad_len
            else:
                z_padded = z
                T_padded = T
                pad_len = 0

            # Reshape into chunks: [B * num_chunks, latent_steps, C]
            num_chunks = T_padded // self.latent_steps
            z_chunks = z_padded.reshape(B * num_chunks, self.latent_steps, C)

            # PREDICT NEXT LATENT: Use first latent_steps-1 to predict the last one
            # Input: first N-1 latents, Target: last latent
            if self.latent_steps > 1:
                z_input = z_chunks[:, :-1, :]  # First latent_steps-1
                z_target = z_chunks[:, -1, :]  # Last latent to predict

                # Run predictor on first N-1 steps
                z_predicted_seq, _ = self.predictor(z_input)
                z_predicted = z_predicted_seq[:, -1, :]  # Take last prediction

                # LATENT PREDICTION LOSS: MSE between predicted and actual next latent
                latent_pred_loss = F.mse_loss(z_predicted, z_target)

                # Create full predicted sequence for reconstruction
                # Use predicted latents for reconstruction
                z_reconstructed = torch.cat([z_input, z_predicted.unsqueeze(1)], dim=1)
            else:
                z_reconstructed = z_chunks
                latent_pred_loss = torch.tensor(0.0, device=h.device)

            # Reshape back to [B, T, C]
            z_reconstructed = z_reconstructed.reshape(B, T_padded, C)

            # Remove padding if added
            if pad_len > 0:
                z_reconstructed = z_reconstructed[:, :T, :]

            z_predicted = z_reconstructed

        # Decode latent back to feature space
        z_decoded = self.decoder(z_predicted)

        # Calculate KL divergence
        kl_loss = self.kl_divergence(mean, logvar)

        # Return decoded features, KL loss, and latent prediction loss
        return z_decoded, kl_loss, latent_pred_loss


class WorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok = Tokenizer(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.tok.embed.weight

        self.dynamics = LatentDynamics(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, lambda_h=0.1, lambda_kl=0.01):
        B, T = idx.shape

        # Token embeddings
        x = self.tok(idx)

        # Apply latent dynamics (NextLat)
        latent_features, kl_loss, latent_pred_loss = self.dynamics(x, lambda_kl)

        # Combine with original features for transformer input
        x = x + lambda_h * latent_features

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        logs = {}

        if targets is not None:
            # Next token prediction loss
            ntp_loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                targets[:, 1:].contiguous().view(-1),
                ignore_index=-1
            )

            # Apply free bits to KL (prevent complete collapse)
            kl_free_bits = getattr(self.config, 'kl_free_bits', 0.0)
            if kl_free_bits > 0:
                kl_loss = torch.max(kl_loss, torch.tensor(kl_free_bits).to(kl_loss.device))

            # Total loss = NTP + KL + Latent Prediction
            loss = ntp_loss + lambda_kl * kl_loss + lambda_h * latent_pred_loss

            logs = {
                'total': loss.item(),
                'ntp': ntp_loss.item(),
                'latent': latent_pred_loss.item(),  # NextLat prediction loss
                'kl': kl_loss.item(),
            }

        return logits, loss, logs