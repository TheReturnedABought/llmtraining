import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import ModelConfig


# ─────────────────────────────
# RMSNorm
# ─────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x):
        return self.scale * x / (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()


# ─────────────────────────────
# RoPE
# ─────────────────────────────

def rope(q, k):
    B, H, T, D = q.shape
    half = D // 2

    freq = 1.0 / (10000 ** (torch.arange(0, half, device=q.device) / half))
    pos = torch.arange(T, device=q.device)

    ang = torch.einsum("i,j->ij", pos, freq)
    sin, cos = ang.sin()[None, None], ang.cos()[None, None]

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]

    q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q, k


# ─────────────────────────────
# SDPA GQA Attention (Transformer A)
# ─────────────────────────────

class GQAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        repeat = self.n_head // self.n_kv_head
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        q, k = rope(q, k)

        # ── Transformer (A): SDPA replaces manual attention ──
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# ─────────────────────────────
# SwiGLU
# ─────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = 4 * cfg.n_embd

        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ─────────────────────────────
# Block
# ─────────────────────────────

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = GQAAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.ff = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────
# World Model
# ─────────────────────────────

class WorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)

        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight

        self.dynamics = nn.Sequential(
            nn.Linear(2 * cfg.n_embd, 256),
            nn.SiLU(),
            nn.Linear(256, cfg.n_embd),
        )

    def encode(self, idx):
        x = self.drop(self.tok(idx))
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def forward(self, idx, targets=None, lambda_h=0.1, lambda_kl=0.01):
        h = self.encode(idx)
        logits = self.head(h)

        if targets is None:
            return logits, None, {}

        loss_ntp = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        K = self.cfg.latent_steps

        h_in = h[:, :-K]
        h_tgt = h[:, 1:-K+1]

        e = self.tok(idx[:, 1:-K+1])

        h_pred = h_in
        loss_h = 0.0

        for i in range(K):
            h_pred = self.dynamics(torch.cat([h_pred, e], dim=-1))
            loss_h += F.smooth_l1_loss(h_pred, h_tgt)

        loss_h /= K

        # ── Stable KL ──
        T = self.cfg.kl_temp

        log_p = F.log_softmax(self.head(h_pred) / T, dim=-1)
        with torch.no_grad():
            p_tgt = F.softmax(self.head(h_tgt) / T, dim=-1)

        loss_kl = F.kl_div(log_p, p_tgt, reduction="batchmean") * (T * T)

        loss = loss_ntp + lambda_h * loss_h + lambda_kl * loss_kl

        return logits, loss, {
            "ntp": loss_ntp.item(),
            "latent": loss_h.item(),
            "kl": loss_kl.item(),
            "total": loss.item(),
        }