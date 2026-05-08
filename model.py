import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# SDPA backend selection (PyTorch 2.3+)
# FLASH_ATTENTION  — fastest, O(T) memory, but only works with is_causal=True / no mask
# EFFICIENT_ATTENTION — O(T) memory, handles arbitrary boolean masks (SWA)
# MATH             — fallback, materialises full O(T²) attention matrix
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel as _sdpa_kernel

    _HAS_EFFICIENT_ATTN = True
except ImportError:
    _HAS_EFFICIENT_ATTN = False


class RMSNorm(nn.Module):
    """RMS Normalisation — faster than LayerNorm (no mean subtraction).
    Used in LLaMA, Mistral, Gemma, DeepSeek-V3."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network — same param count as 4× GELU MLP but
    gated, which improves gradient flow and tensor-core utilisation.
    Used in LLaMA, Mistral, PaLM, DeepSeek-V3.
    hidden_dim ≈ 8d/3, rounded to the nearest multiple of 64."""

    def __init__(self, dim: int):
        super().__init__()
        hidden = int(dim * 8 / 3)
        hidden = (hidden + 63) // 64 * 64  # align to 64 for GPU efficiency
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.val = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.gate(x)) * self.val(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        # Base 1_000_000 (Qwen3 style) — higher than LLaMA-3's 500k to compensate
        # for QK-Norm suppressing attention entropy at long contexts.
        inv_freq = 1.0 / (1_000_000 ** (torch.arange(0, dim, 2).float() / dim))
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
    def __init__(self, config, window_size: int | None = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.window_size = window_size  # None = full causal attention

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # QK-Norm (Qwen3): per-head RMSNorm on Q and K before RoPE.
        # Prevents dot-product overflow in BF16 and attention entropy collapse.
        # Especially important with SWA where short windows create sharp distributions.
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.rotary = RotaryEmbedding(self.head_dim)

        # Pre-build the SWA mask once at init (lives on GPU via register_buffer).
        # Global layers (window_size=None) use is_causal=True — no mask needed.
        if self.window_size is not None:
            self._build_mask(config.block_size)

    def _build_mask(self, T: int) -> None:
        """Build and cache [1,1,T,T] SWA boolean mask (True=attend)."""
        idx = torch.arange(T)
        dist = idx.unsqueeze(1) - idx.unsqueeze(0)
        mask = ((dist >= 0) & (dist < self.window_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('_attn_mask', mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # QK-Norm: normalise per head before RoPE (Qwen3 pattern)
        q = self.q_norm(q)  # [B, n_head,    T, head_dim]
        k = self.k_norm(k)  # [B, n_kv_head, T, head_dim]

        q, k = self.rotary(q, k)

        # GQA: repeat KV heads to match query head count
        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        if self.window_size is None:
            # ── Global attention ────────────────────────────────────────────────
            # is_causal=True lets PyTorch dispatch to Flash Attention-2 on Ampere+.
            # No attention matrix ever materialised — O(T) memory.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # ── Sliding-window attention ──────────────────────────────────────
            # Lazy rebuild if sequence length changes (never during fixed training).
            if T != self._attn_mask.shape[-1]:
                self._build_mask(T)
                self._attn_mask = self._attn_mask.to(x.device)
            # EFFICIENT_ATTENTION backend: O(T) chunked memory, handles custom masks.
            # Falls back to math backend (O(T²)) on CPU or FP32 — safe at eval time.
            if (_HAS_EFFICIENT_ATTN
                    and q.is_cuda
                    and q.dtype in (torch.float16, torch.bfloat16)):
                with _sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=self._attn_mask)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=self._attn_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config, window_size: int | None = None):
        super().__init__()
        self.attn = CausalSelfAttention(config, window_size=window_size)
        self.mlp = SwiGLUFFN(config.n_embd)

        # Parallel architecture (PaLM/GPT-J style):
        # Attention and MLP share a single pre-norm.
        self.ln = RMSNorm(config.n_embd)

    def forward(self, x):
        # Normalise once, compute branches concurrently, add to residual
        nx = self.ln(x)
        x = x + self.attn(nx) + self.mlp(nx)
        return x


class Tokenizer(nn.Module):
    """Byte-level tokenizer with learnable embeddings"""

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.ln = RMSNorm(n_embd)

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

        # Hybrid attention: local SWA layers + 1 global layer at the end
        # (Mistral / BLT pattern — best for byte-level local context)
        swa_w = getattr(config, 'swa_window', 256)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config,
                window_size=None if i == config.n_layer - 1 else swa_w
            )
            for i in range(config.n_layer)
        ])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.tok.embed.weight

        self.dynamics = LatentDynamics(config)

        self.use_checkpoint = False  # set via model.use_checkpoint = True

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

        # Transformer blocks (with optional gradient checkpointing)
        for block in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
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
                ignore_index=-1,
                reduction='mean'
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