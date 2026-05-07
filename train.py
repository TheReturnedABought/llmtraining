import torch
import os
import time
import argparse
import signal
import sys
from pathlib import Path
from datetime import datetime
from model import WorldModel
from config import ModelConfig
from data import get_loaders, decode

# ─────────────────────────────
# Graceful interrupt handler
# ─────────────────────────────
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    print("\n⚠️  Interrupt received. Finishing current step and saving...")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

# ─────────────────────────────
# Argument Parser
# ─────────────────────────────
parser = argparse.ArgumentParser(description="Train NextLat Mini Pipeline")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--epochs", type=int, default=1, help="Max epochs (ignored if --steps reached first)")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--lambda_h", type=float, default=0.1)
parser.add_argument("--lambda_kl", type=float, default=0.01)
parser.add_argument("--kl_warmup", type=int, default=1000, help="Steps to linearly ramp up KL weight")
parser.add_argument("--kl_temp", type=float, default=5.0, help="KL temperature (higher = smoother)")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--project", type=str, default="world-model")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--eval_every", type=int, default=500)
parser.add_argument("--eval_batches", type=int, default=64, help="Validation batches per eval (0 = full validation set)")
parser.add_argument("--sample_tokens", type=int, default=80, help="Tokens to generate for quick eval sample")
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ─────────────────────────────
# Setup
# ─────────────────────────────
torch.manual_seed(args.seed)
cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"
os.makedirs(args.checkpoint_dir, exist_ok=True)

cfg = ModelConfig(
    latent_steps=8,
    kl_temp=args.kl_temp,
)

model = WorldModel(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

train_loader, val_loader = get_loaders(cfg.block_size, args.batch_size)

# ─────────────────────────────
# Initialization Logging
# ─────────────────────────────
print("\n" + "=" * 60)
print("🚀 NextLat Mini Pipeline — Initialization Report")
print("=" * 60)
print(f"📅 Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"💻 Device       : {device.upper()}")
if not cuda_available:
    print("⚠️  CUDA unavailable — running on CPU. If you expected GPU, install a CUDA-enabled PyTorch build.")
else:
    print(f"🎮 GPU          : {torch.cuda.get_device_name(0)}")
print(f"🧠 Model params : {sum(p.numel() for p in model.parameters()):,}")
print(f"📦 Trainable    : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Count by component
embed_params = sum(p.numel() for p in model.tok.parameters())
block_params = sum(p.numel() for p in model.blocks.parameters())
dyn_params = sum(p.numel() for p in model.dynamics.parameters())

print(f"\n📊 Parameter Breakdown:")
print(f"   ├─ Embedding        : {embed_params:>8,}")
print(f"   ├─ Transformer blocks : {block_params:>8,} ({cfg.n_layer} layers)")
print(f"   ├─ LM Head (tied)     : 0")
print(f"   └─ Latent Dynamics  : {dyn_params:>8,}")
print(f"\n⚙️  Config:")
print(f"   ├─ Model dim    : {cfg.n_embd}")
print(f"   ├─ Layers       : {cfg.n_layer}")
print(f"   ├─ Heads        : {cfg.n_head} (KV: {cfg.n_kv_head})")
print(f"   ├─ Block size   : {cfg.block_size}")
print(f"   ├─ Vocab size   : {cfg.vocab_size}")
print(f"   ├─ Latent steps : {cfg.latent_steps}")
print(f"   └─ KL temp      : {cfg.kl_temp}")
print(f"\n📈 Training Config:")
print(f"   ├─ Batch size   : {args.batch_size}")
print(f"   ├─ Max steps    : {args.steps:,}")
print(f"   ├─ Max epochs   : {args.epochs}")
print(f"   ├─ Learning rate: {args.lr}")
print(f"   ├─ Lambda_h     : {args.lambda_h}")
print(f"   ├─ Lambda_kl    : {args.lambda_kl}")
print(f"   ├─ KL warmup    : {args.kl_warmup} steps")
print(f"   ├─ Eval batches : {args.eval_batches if args.eval_batches else 'full'}")
print(f"   └─ Sample tokens: {args.sample_tokens}")
print(f"\n📊 Dataset:")
print(f"   ├─ Train batches: {len(train_loader)}")
print(f"   ├─ Val batches  : {len(val_loader)}")
print(f"   └─ Tokens/batch : {args.batch_size * cfg.block_size:,}")
print("=" * 60 + "\n")

# ─────────────────────────────
# Resume Logic
# ─────────────────────────────
global_step = 0
start_epoch = 0
best_val_loss = float('inf')

def resolve_resume_path(resume_path: str):
    """Resolve checkpoint path across different launch directories."""
    candidate = Path(resume_path).expanduser()
    if candidate.is_file():
        return candidate

    script_dir = Path(__file__).resolve().parent
    candidate_from_script = script_dir / candidate
    if candidate_from_script.is_file():
        return candidate_from_script

    if not candidate.is_absolute():
        checkpoint_dir = Path(args.checkpoint_dir).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = script_dir / checkpoint_dir
        candidate_from_checkpoint_dir = checkpoint_dir / candidate.name
        if candidate_from_checkpoint_dir.is_file():
            return candidate_from_checkpoint_dir

    return None

if args.resume is not None:
    resolved_resume = resolve_resume_path(args.resume)
    if resolved_resume is None:
        script_dir = Path(__file__).resolve().parent
        checkpoint_dir = Path(args.checkpoint_dir).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = script_dir / checkpoint_dir
        print(f"❌ Could not find checkpoint: {args.resume}")
        print(f"   Tried relative to CWD   : {Path.cwd()}")
        print(f"   Tried relative to script: {script_dir}")
        print(f"   Tried checkpoint dir    : {checkpoint_dir}")
        sys.exit(1)

    print(f"📂 Resuming from {resolved_resume}")
    ckpt = torch.load(str(resolved_resume), map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    global_step = ckpt.get("global_step", 0)
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float('inf'))
    print(f"✅ Resumed — Step {global_step}, Epoch {start_epoch}")
    print(f"📉 Best val loss: {best_val_loss:.4f}\n")

if args.wandb:
    import wandb
    wandb.init(project=args.project, config=vars(args), resume="allow")


# ─────────────────────────────
# KL Warmup Schedule
# ─────────────────────────────
def get_kl_weight(step):
    """Linear warmup of KL weight from 0 to lambda_kl over kl_warmup steps."""
    if args.kl_warmup <= 0 or args.lambda_kl == 0:
        return args.lambda_kl
    if step >= args.kl_warmup:
        return args.lambda_kl
    return args.lambda_kl * (step / args.kl_warmup)


# ─────────────────────────────
# Training Utilities
# ─────────────────────────────
def evaluate(max_batches=0):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            # Use full KL weight for consistent eval
            _, loss, _ = model(x, y, lambda_h=args.lambda_h, lambda_kl=args.lambda_kl)
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def sample(prompt="The", max_new=80):
    model.eval()
    prompt_bytes = prompt.encode("utf-8", errors="replace")
    tokens = list(prompt_bytes)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_new):
            logits, _, _ = model(idx)
            logits = logits[:, -1] / 0.8
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return decode(idx[0].tolist())


def save_checkpoint(step, epoch, is_best=False):
    fname = f"step_{step:06d}.pt"
    path = os.path.join(args.checkpoint_dir, fname)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "global_step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "args": vars(args),
        "config": {
            "vocab_size": cfg.vocab_size,
            "block_size": cfg.block_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_kv_head": cfg.n_kv_head,
            "n_embd": cfg.n_embd,
            "latent_steps": cfg.latent_steps,
            "kl_temp": cfg.kl_temp,
        },
    }, path)
    if is_best:
        best_path = os.path.join(args.checkpoint_dir, "best.pt")
        torch.save({"model": model.state_dict()}, best_path)
    return path


# ─────────────────────────────
# Main Training Loop
# ─────────────────────────────
tokens_per_step = args.batch_size * cfg.block_size
total_tokens = 0
t0 = time.time()
step_times = []

print("⚡ Starting training...\n")

for epoch in range(start_epoch, args.epochs):
    if global_step >= args.steps or interrupted:
        break

    model.train()

    for x, y in train_loader:
        if global_step >= args.steps or interrupted:
            break

        step_start = time.time()

        x, y = x.to(device), y.to(device)

        # Get current KL weight from warmup schedule
        current_kl_weight = get_kl_weight(global_step)

        _, loss, logs = model(
            x, y,
            lambda_h=args.lambda_h,
            lambda_kl=current_kl_weight,
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        global_step += 1
        total_tokens += tokens_per_step

        step_time = time.time() - step_start
        step_times.append(step_time)
        if len(step_times) > 10:
            step_times.pop(0)
        avg_step_time = sum(step_times) / len(step_times)
        tokens_per_sec = tokens_per_step / avg_step_time if avg_step_time > 0 else 0

        # ── Logging ──
        if global_step % args.log_every == 0:
            elapsed = time.time() - t0
            pct = min(global_step / args.steps * 100, 100)
            bar_len = 20
            filled = int(bar_len * min(global_step, args.steps) / args.steps)
            bar = "█" * filled + "░" * (bar_len - filled)

            print(
                f"[{bar}] {pct:5.1f}% | "
                f"step {global_step:06d}/{args.steps:<06d} | "
                f"loss {logs['total']:7.4f} | "
                f"ntp {logs['ntp']:7.4f} | "
                f"lat {logs['latent']:7.4f} | "
                f"kl {logs['kl']:8.2f} | "
                f"kl_w {current_kl_weight:.4f} | "
                f"tok/s {tokens_per_sec:7.0f} | "
                f"etime {elapsed:.0f}s"
            )

            if args.wandb:
                wandb.log({
                    "train/total_loss": logs['total'],
                    "train/ntp_loss": logs['ntp'],
                    "train/latent_loss": logs['latent'],
                    "train/kl_loss": logs['kl'],
                    "train/kl_weight": current_kl_weight,
                    "train/tokens_per_sec": tokens_per_sec,
                }, step=global_step)

        # ── KL Warning ──
        if logs['kl'] > 50 and global_step % 50 == 0:
            print(f"  ⚠️  KL={logs['kl']:.1f} — consider higher --kl_temp or lower --lambda_kl")

        # ── Evaluation & Checkpointing ──
        if global_step % args.eval_every == 0:
            print(f"\n{'=' * 60}")
            print(f"📊 Evaluating at step {global_step}...")
            val_start = time.time()
            val_loss = evaluate(args.eval_batches)
            val_time = time.time() - val_start
            print(f"📊 Eval done in {val_time:.1f}s — val_loss: {val_loss:.4f}")
            print(f"{'=' * 60}")

            # Quick sample
            print("\n--- Sample ---")
            print(sample("The history of", max_new=args.sample_tokens))
            print("--------------\n")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            path = save_checkpoint(global_step, epoch, is_best)
            print(f"💾 Checkpoint saved: {path}\n")

            if args.wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/best_loss": best_val_loss,
                }, step=global_step)

    if global_step >= args.steps:
        print(f"\n✅ Reached target steps ({args.steps}) — stopping.")
        break

    print(f"✅ Epoch {epoch + 1}/{args.epochs} complete (step {global_step})")

# ─────────────────────────────
# Final Summary
# ─────────────────────────────
total_time = time.time() - t0
avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

print("\n" + "=" * 60)
print("🏁 Training Complete")
print("=" * 60)
print(f"   Total steps    : {global_step:,}")
print(f"   Total tokens   : {total_tokens:,}")
print(f"   Total time     : {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"   Avg tokens/s   : {avg_tokens_per_sec:,.0f}")
print(f"   Best val loss  : {best_val_loss:.4f}")
print("=" * 60 + "\n")

final_path = save_checkpoint(global_step, epoch, is_best=False)
print(f"💾 Final model saved to: {final_path}")

if args.wandb:
    wandb.finish()
