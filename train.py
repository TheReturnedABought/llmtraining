import torch
import os
import time
import argparse
import signal
import sys
import math
import traceback
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
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--steps", type=int, default=100000)
parser.add_argument("--epochs", type=int, default=10, help="Max epochs")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate for cosine schedule")
parser.add_argument("--lambda_h", type=float, default=0.1)
parser.add_argument("--lambda_kl", type=float, default=0.0005, help="KL weight (reduced from 0.01)")
parser.add_argument("--kl_warmup", type=int, default=5000, help="Steps to linearly ramp up KL weight")
parser.add_argument("--kl_temp", type=float, default=0.5, help="KL temperature (lower = less collapse)")
parser.add_argument("--kl_free_bits", type=float, default=0.7, help="Free bits for KL to prevent collapse")
parser.add_argument("--kl_anneal_type", type=str, default="linear", choices=["linear", "cosine", "cyclical"])
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint or 'best' to resume from best model")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--project", type=str, default="world-model")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--eval_every", type=int, default=500)
parser.add_argument("--eval_batches", type=int, default=64,
                    help="Validation batches per eval (0 = full validation set)")
parser.add_argument("--sample_tokens", type=int, default=80, help="Tokens to generate for quick eval sample")
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.95)
parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
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
    kl_free_bits=args.kl_free_bits,
    kl_anneal_type=args.kl_anneal_type,
)

model = WorldModel(cfg).to(device)

# Configure optimizer with weight decay
opt = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay
)

print("📚 Preparing datasets and data loaders...")
try:
    train_loader, val_loader = get_loaders(cfg.block_size, args.batch_size, num_workers=args.num_workers)
except Exception as e:
    print(f"❌ Failed to build data loaders: {e}")
    print("   Make sure dependencies are installed: pip install torch datasets numpy")
    print("   If cache may be corrupted, delete: data_cache_wikipedia_en.bin and data_cache_wikipedia_en.bin.len")
    traceback.print_exc()
    sys.exit(1)

print(f"✅ Data loaders ready | train batches: {len(train_loader)} | val batches: {len(val_loader)}")

if len(train_loader) == 0:
    print("❌ Training dataset produced zero batches.")
    print("   Check your cached dataset file and block size settings.")
    print("   Try deleting cache files and rerun: data_cache_wikipedia_en.bin and data_cache_wikipedia_en.bin.len")
    sys.exit(1)

if len(val_loader) == 0:
    print("⚠️  Validation dataset produced zero batches. Evaluation checkpoints may be skipped.")

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
print(f"   ├─ KL temp      : {cfg.kl_temp}")
print(f"   ├─ KL free bits : {cfg.kl_free_bits}")
print(f"   └─ KL anneal    : {cfg.kl_anneal_type}")
print(f"\n📈 Training Config:")
print(f"   ├─ Batch size   : {args.batch_size}")
print(f"   ├─ Max steps    : {args.steps:,}")
print(f"   ├─ Max epochs   : {args.epochs}")
print(f"   ├─ Learning rate: {args.lr}")
print(f"   ├─ Min LR       : {args.min_lr}")
print(f"   ├─ Lambda_h     : {args.lambda_h}")
print(f"   ├─ Lambda_kl    : {args.lambda_kl}")
print(f"   ├─ KL warmup    : {args.kl_warmup} steps")
print(f"   ├─ Weight decay : {args.weight_decay}")
print(f"   ├─ Grad clip    : {args.grad_clip}")
print(f"   ├─ Eval batches : {args.eval_batches if args.eval_batches else 'full'}")
print(f"   ├─ Num workers  : {args.num_workers}")
print(f"   └─ Sample tokens: {args.sample_tokens}")
print(f"\n📊 Dataset:")
print(f"   ├─ Train batches: {len(train_loader)}")
print(f"   ├─ Val batches  : {len(val_loader)}")
print(f"   └─ Tokens/batch : {args.batch_size * cfg.block_size:,}")
print(f"\n💾 Checkpoint Strategy: Every 10% progress + Best models only")
print("=" * 60 + "\n")

# ─────────────────────────────
# Resume Logic (with best model support)
# ─────────────────────────────
global_step = 0
best_val_loss = float('inf')
best_step = 0
current_epoch = 0


def list_checkpoints():
    """List available checkpoints sorted by step number."""
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints = []
    for f in checkpoint_dir.glob("checkpoint_*.pt"):
        try:
            step = int(f.stem.split('_')[1])
            checkpoints.append((step, f))
        except (IndexError, ValueError):
            continue

    # Sort by step number (descending)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints


def resolve_resume_path(resume_path: str):
    """Resolve checkpoint path across different launch directories."""

    # Special case: resume from best model
    if resume_path.lower() == 'best':
        best_path = Path(args.checkpoint_dir) / "best.pt"
        if best_path.is_file():
            print(f"🏆 Resuming from best model: {best_path}")
            return best_path
        else:
            print(f"❌ No best.pt found in {args.checkpoint_dir}")
            return None

    # Special case: resume from latest checkpoint
    if resume_path.lower() == 'latest':
        checkpoints = list_checkpoints()
        if checkpoints:
            latest_step, latest_path = checkpoints[0]
            print(f"📂 Resuming from latest checkpoint: checkpoint_{latest_step}")
            return latest_path
        else:
            print(f"❌ No checkpoints found in {args.checkpoint_dir}")
            return None

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


# Handle resume
resume_from_best = False
if args.resume is not None:
    resolved_resume = resolve_resume_path(args.resume)
    if resolved_resume is None:
        # Try to show available checkpoints
        checkpoints = list_checkpoints()
        print(f"\n❌ Could not find checkpoint: {args.resume}")
        print(f"   Tried relative to CWD   : {Path.cwd()}")
        print(f"   Tried relative to script: {Path(__file__).resolve().parent}")
        print(f"   Tried checkpoint dir    : {Path(args.checkpoint_dir).resolve()}")

        if checkpoints:
            print(f"\n📋 Available checkpoints:")
            for step, path in checkpoints[:10]:
                print(f"   - checkpoint_{step:06d}.pt")
            if len(checkpoints) > 10:
                print(f"   ... and {len(checkpoints) - 10} more")

        print(f"\n💡 Try one of these:")
        print(f"   python train.py --resume best")
        print(f"   python train.py --resume latest")
        for step, path in checkpoints[:3]:
            print(f"   python train.py --resume {path.name}")

        sys.exit(1)

    print(f"📂 Resuming from {resolved_resume}")
    ckpt = torch.load(str(resolved_resume), map_location=device)

    # Check if this is a best.pt checkpoint (might have different format)
    if 'model' in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        print("❌ Invalid checkpoint format - missing 'model' key")
        sys.exit(1)

    # Load optimizer if available
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
        print("✅ Loaded optimizer state")
    else:
        print("⚠️  No optimizer state found - using fresh optimizer")

    global_step = ckpt.get("global_step", 0)
    best_val_loss = ckpt.get("best_val_loss", float('inf'))
    best_step = ckpt.get("best_step")
    if best_step is None or (best_step == 0 and global_step > 0 and best_val_loss != float('inf')):
        best_step = global_step if best_val_loss != float('inf') else 0

    current_epoch = ckpt.get("epoch")
    if current_epoch is None:
        current_epoch = global_step // max(len(train_loader), 1)

    print(f"✅ Resumed — Step {global_step}, Epoch {current_epoch}")
    print(f"📉 Best val loss: {best_val_loss:.4f} (at step {best_step})")

    if args.resume.lower() == 'best':
        print(f"🏆 Continuing from best model!")
        resume_from_best = True
    print()

if args.wandb:
    import wandb

    wandb.init(project=args.project, config=vars(args), resume="allow")

# ─────────────────────────────
# Initial checkpoint handling
# ─────────────────────────────
# Save best.pt if starting fresh (for consistency)
if not args.resume and global_step == 0:
    print("💾 Saving initial best.pt...")
    best_path = os.path.join(args.checkpoint_dir, "best.pt")
    torch.save({
        "model": model.state_dict(),
        "global_step": 0,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "epoch": current_epoch,
        "optimizer": opt.state_dict(),
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
    }, best_path)
    print(f"💾 Initial best.pt saved\n")


# ─────────────────────────────
# KL Warmup Schedule
# ─────────────────────────────
def get_kl_weight(step):
    """KL weight schedule with multiple annealing strategies."""
    if args.kl_warmup <= 0 or args.lambda_kl == 0:
        return args.lambda_kl

    if step >= args.kl_warmup:
        return args.lambda_kl

    progress = step / args.kl_warmup

    if args.kl_anneal_type == "linear":
        weight = progress
    elif args.kl_anneal_type == "cosine":
        weight = 0.5 * (1 - math.cos(math.pi * progress))
    elif args.kl_anneal_type == "cyclical":
        # Cyclical annealing: go through 4 cycles
        cycle = 4
        weight = 0.5 * (1 - math.cos(2 * math.pi * cycle * progress))
        # Only use increasing parts
        weight = min(weight, progress)
    else:
        weight = progress

    return args.lambda_kl * weight


def get_lr(step, total_steps):
    """Cosine learning rate schedule."""
    if step >= total_steps:
        return args.min_lr

    progress = step / total_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return args.min_lr + (args.lr - args.min_lr) * cosine_decay


# ─────────────────────────────
# Training Utilities
# ─────────────────────────────
def evaluate(max_batches=0):
    model.eval()
    total_loss = 0.0
    total_kl = 0.0
    n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss, logs = model(x, y, lambda_h=args.lambda_h, lambda_kl=args.lambda_kl)
            total_loss += loss.item()
            total_kl += logs.get('kl', 0)
            n += 1
    model.train()
    return total_loss / max(n, 1), total_kl / max(n, 1)


def sample(prompt="The", max_new=80, temperature=0.8):
    model.eval()
    prompt_bytes = prompt.encode("utf-8", errors="replace")
    tokens = list(prompt_bytes)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_new):
            logits, _, _ = model(idx)
            logits = logits[:, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return decode(idx[0].tolist())


def should_save_progress_checkpoint(step):
    """Check if current step is at a 10% milestone."""
    if step == 0:
        return False

    progress_pct = (step / args.steps) * 100

    # Check if we've crossed a 10% boundary (10%, 20%, ..., 90%, 100%)
    prev_step = step - 1
    prev_progress_pct = (prev_step / args.steps) * 100

    # Check each 10% milestone
    for milestone in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        if prev_progress_pct < milestone and progress_pct >= milestone:
            return True

    return False


def save_checkpoint(step, epoch, is_best=False, is_progress=False):
    """Save checkpoint. Different naming for best vs progress checkpoints."""

    # Always update best.pt if this is a new best
    if is_best:
        best_path = os.path.join(args.checkpoint_dir, "best.pt")
        torch.save({
            "model": model.state_dict(),
            "global_step": step,
            "best_val_loss": best_val_loss,
            "best_step": best_step,
            "epoch": epoch,
            "optimizer": opt.state_dict(),
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
        }, best_path)
        print(f"🏆 New best model! val_loss: {best_val_loss:.4f} at step {step}")

    # Save progress checkpoint at 10% intervals
    if is_progress:
        progress_pct = int((step / args.steps) * 100)
        fname = f"checkpoint_{step:06d}.pt"  # Using checkpoint_ prefix to distinguish from best.pt
        path = os.path.join(args.checkpoint_dir, fname)

        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "global_step": step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_step": best_step,
            "progress_pct": progress_pct,
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
                "kl_free_bits": cfg.kl_free_bits,
            },
        }, path)
        print(f"💾 Progress checkpoint ({progress_pct}%): {path}")


# ─────────────────────────────
# Main Training Loop
# ─────────────────────────────
tokens_per_step = args.batch_size * cfg.block_size
total_tokens = 0
t0 = time.time()
step_times = []
total_possible_steps = args.steps

print("⚡ Starting training...\n")

training_completed = False
last_progress_checkpoint = 0

# Epoch loop
for epoch in range(current_epoch, args.epochs):
    if global_step >= args.steps or interrupted:
        break

    model.train()
    epoch_loss = 0.0
    epoch_ntp_loss = 0.0
    epoch_latent_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_steps = 0

    # Data iterator
    for x, y in train_loader:
        if global_step >= args.steps or interrupted:
            break

        step_start = time.time()

        x, y = x.to(device), y.to(device)

        # Update learning rate
        current_lr = get_lr(global_step, total_possible_steps)
        for param_group in opt.param_groups:
            param_group['lr'] = current_lr

        # Get current KL weight from warmup schedule
        current_kl_weight = get_kl_weight(global_step)

        # Forward pass
        _, loss, logs = model(
            x, y,
            lambda_h=args.lambda_h,
            lambda_kl=current_kl_weight,
        )

        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  NaN/Inf loss detected at step {global_step}. Skipping batch.")
            continue

        # Backward pass
        opt.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimizer step
        opt.step()

        global_step += 1
        total_tokens += tokens_per_step
        training_completed = True
        epoch_loss += loss.item()
        epoch_ntp_loss += logs.get('ntp', 0)
        epoch_latent_loss += logs.get('latent', 0)
        epoch_kl_loss += logs.get('kl', 0)
        epoch_steps += 1

        step_time = time.time() - step_start
        step_times.append(step_time)
        if len(step_times) > 100:
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

            # KL warning
            kl_value = logs['kl']
            kl_warning = ""
            if kl_value > 300:
                kl_warning = " 🔴 KL TOO HIGH"
            elif kl_value > 150:
                kl_warning = " 🟡 KL HIGH"
            elif kl_value < 0.1 and current_kl_weight > 0:
                kl_warning = " ⚪ KL collapsed"

            # Latent prediction warning
            latent_value = logs.get('latent', 0)
            latent_warning = ""
            if latent_value > 1.0:
                latent_warning = " 🔴 LAT HIGH"
            elif latent_value < 0.001 and global_step > 1000:
                latent_warning = " ⚪ LAT collapsed"

            print(
                f"[{bar}] {pct:5.1f}% | "
                f"step {global_step:06d}/{args.steps:<06d} | "
                f"loss {logs['total']:7.4f} | "
                f"ntp {logs['ntp']:7.4f} | "
                f"lat {latent_value:8.4f}{latent_warning} | "
                f"kl {kl_value:8.2f}{kl_warning} | "
                f"kl_w {current_kl_weight:.6f} | "
                f"lr {current_lr:.6f} | "
                f"best {best_val_loss:.4f}@{best_step} | "
                f"tok/s {tokens_per_sec:7.0f} | "
                f"etime {elapsed:.0f}s"
            )

            if args.wandb:
                wandb.log({
                    "train/total_loss": logs['total'],
                    "train/ntp_loss": logs['ntp'],
                    "train/latent_loss": latent_value,
                    "train/kl_loss": kl_value,
                    "train/kl_weight": current_kl_weight,
                    "train/learning_rate": current_lr,
                    "train/tokens_per_sec": tokens_per_sec,
                }, step=global_step)

        # ── Evaluation & Checkpointing ──
        if global_step % args.eval_every == 0:
            print(f"\n{'=' * 60}")
            print(f"📊 Evaluating at step {global_step}...")
            val_start = time.time()
            val_loss, val_kl = evaluate(args.eval_batches)
            val_time = time.time() - val_start
            print(f"📊 Eval done in {val_time:.1f}s — val_loss: {val_loss:.4f}, val_kl: {val_kl:.2f}")
            print(f"{'=' * 60}")

            # Quick sample
            print("\n--- Sample ---")
            print(sample("The history of", max_new=args.sample_tokens))
            print("--------------\n")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_step = global_step

            # Check if we should save a progress checkpoint
            is_progress = should_save_progress_checkpoint(global_step)

            if is_best or is_progress:
                save_checkpoint(global_step, epoch + 1, is_best=is_best, is_progress=is_progress)

            print()  # Extra newline for readability

            if args.wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/kl": val_kl,
                    "val/best_loss": best_val_loss,
                    "val/best_step": best_step,
                }, step=global_step)

    # End of epoch
    current_epoch = epoch + 1
    if epoch_steps > 0:
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        avg_epoch_ntp = epoch_ntp_loss / max(epoch_steps, 1)
        avg_epoch_lat = epoch_latent_loss / max(epoch_steps, 1)
        avg_epoch_kl = epoch_kl_loss / max(epoch_steps, 1)

        if current_epoch < args.epochs and global_step < args.steps:
            print(f"✅ Epoch {current_epoch}/{args.epochs} complete (step {global_step})")
            print(
                f"   Avg loss: {avg_epoch_loss:.4f} | NTP: {avg_epoch_ntp:.4f} | Lat: {avg_epoch_lat:.4f} | KL: {avg_epoch_kl:.2f}")

# ─────────────────────────────
# Final Summary
# ─────────────────────────────
total_time = time.time() - t0
avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

print("\n" + "=" * 60)
print("🏁 Training Complete")
print("=" * 60)
print(f"   Total steps    : {global_step:,}")
print(f"   Total epochs   : {current_epoch}")
print(f"   Total tokens   : {total_tokens:,}")
print(f"   Total time     : {total_time:.0f}s ({total_time / 60:.1f} min)")
print(f"   Avg tokens/s   : {avg_tokens_per_sec:,.0f}")
print(f"   Best val loss  : {best_val_loss:.4f} (at step {best_step})")
print("=" * 60 + "\n")

# Show saved checkpoints
checkpoint_dir = Path(args.checkpoint_dir)
best_path = checkpoint_dir / "best.pt"
if best_path.exists():
    print(f"🏆 Best model saved to: {best_path}")

progress_ckpts = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
if progress_ckpts:
    print("\n📋 Progress checkpoints saved:")
    for f in progress_ckpts:
        print(f"   - {f.name}")

# Save final checkpoint if training completed and at a milestone
if training_completed and global_step > 0 and should_save_progress_checkpoint(min(global_step, args.steps)):
    final_step = min(global_step, args.steps)
    final_name = checkpoint_dir / f"checkpoint_{final_step:06d}.pt"
    if not final_name.exists():
        save_checkpoint(final_step, current_epoch, is_progress=True)
        print(f"\n💾 Final checkpoint saved at step {final_step}")

if args.wandb:
    wandb.finish()