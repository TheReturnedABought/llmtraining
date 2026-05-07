import torch
import os
import time
import argparse
from datetime import datetime
from model import WorldModel
from config import ModelConfig
from data import get_loaders, decode

# ─────────────────────────────
# Argument Parser
# ─────────────────────────────
parser = argparse.ArgumentParser(description="Train NextLat Mini Pipeline")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--lambda_h", type=float, default=0.1)
parser.add_argument("--lambda_kl", type=float, default=0.01)
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--project", type=str, default="world-model")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--eval_every", type=int, default=500)
parser.add_argument("--log_every", type=int, default=100)
args = parser.parse_args()

# ─────────────────────────────
# Setup
# ─────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.checkpoint_dir, exist_ok=True)

cfg = ModelConfig(
    latent_steps=8,
    kl_temp=5.0,
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
print(f"🧠 Model params : {sum(p.numel() for p in model.parameters()):,}")
print(f"📦 Trainable    : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Count by component
total_params = sum(p.numel() for p in model.parameters())
embed_params = sum(p.numel() for p in model.tok.parameters())
block_params = sum(p.numel() for p in model.blocks.parameters())
head_params = sum(p.numel() for p in model.head.parameters())
dyn_params = sum(p.numel() for p in model.dynamics.parameters())

print(f"\n📊 Parameter Breakdown:")
print(f"   ├─ Embedding       : {embed_params:>8,}")
print(f"   ├─ Transformer blocks: {block_params:>8,} ({cfg.n_layer} layers)")
print(f"   ├─ LM Head (tied)    : 0")
print(f"   └─ Latent Dynamics : {dyn_params:>8,}")
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
print(f"   ├─ Learning rate: {args.lr}")
print(f"   ├─ Lambda_h     : {args.lambda_h}")
print(f"   └─ Lambda_kl    : {args.lambda_kl}")
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

if args.resume is not None:
    print(f"📂 Resuming from {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)

    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    global_step = ckpt.get("global_step", 0)
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float('inf'))

    print(f"✅ Resumed — Step {global_step}, Epoch {start_epoch}")
    print(f"📉 Best val loss: {best_val_loss:.4f}\n")

# Optional: WandB initialization
if args.wandb:
    import wandb

    wandb.init(project=args.project, config=vars(args), resume="allow")


# ─────────────────────────────
# Training Utilities
# ─────────────────────────────
def evaluate():
    model.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, y, lambda_h=args.lambda_h, lambda_kl=args.lambda_kl)
            total_loss += loss.item()
            n += 1

    model.train()
    return total_loss / max(n, 1)


def sample(prompt="The", max_new=200):
    model.eval()
    tokens = [prompt.encode("utf-8", errors="replace")]
    idx = torch.tensor(tokens, dtype=torch.long).to(device)

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
stop_training = False
max_epochs = 10
tokens_per_step = args.batch_size * cfg.block_size
total_tokens = 0
t0 = time.time()
step_times = []  # For rolling average

print("⚡ Starting training...\n")

for epoch in range(start_epoch, max_epochs):
    if stop_training:
        break

    model.train()

    for x, y in train_loader:
        if global_step >= args.steps:
            stop_training = True
            break

        step_start = time.time()

        x, y = x.to(device), y.to(device)

        _, loss, logs = model(
            x, y,
            lambda_h=args.lambda_h,
            lambda_kl=args.lambda_kl,
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        global_step += 1
        total_tokens += tokens_per_step

        step_time = time.time() - step_start
        step_times.append(step_time)

        # Rolling average over last 10 steps
        if len(step_times) > 10:
            step_times.pop(0)
        avg_step_time = sum(step_times) / len(step_times)
        tokens_per_sec = tokens_per_step / avg_step_time if avg_step_time > 0 else 0

        # ── Logging ──
        if global_step % args.log_every == 0:
            elapsed = time.time() - t0

            # Progress bar
            pct = global_step / args.steps * 100
            bar_len = 20
            filled = int(bar_len * global_step / args.steps)
            bar = "█" * filled + "░" * (bar_len - filled)

            print(
                f"[{bar}] {pct:5.1f}% | "
                f"step {global_step:06d}/{args.steps:<06d} | "
                f"loss {logs['total']:7.4f} | "
                f"ntp {logs['ntp']:7.4f} | "
                f"lat {logs['latent']:7.4f} | "
                f"kl {logs['kl']:8.2f} | "
                f"toks/s {tokens_per_sec:7.0f} | "
                f"etime {elapsed:.0f}s"
            )

            if args.wandb:
                wandb.log({
                    "train/total_loss": logs['total'],
                    "train/ntp_loss": logs['ntp'],
                    "train/latent_loss": logs['latent'],
                    "train/kl_loss": logs['kl'],
                    "train/tokens_per_sec": tokens_per_sec,
                }, step=global_step)

        # ── Evaluation & Checkpointing ──
        if global_step % args.eval_every == 0:
            val_start = time.time()
            val_loss = evaluate()
            val_time = time.time() - val_start

            print(f"\n{'=' * 60}")
            print(f"📊 Eval at step {global_step:06d} (took {val_time:.1f}s)")
            print(f"   val_loss: {val_loss:.4f}")
            print(f"{'=' * 60}")

            # Sample generation
            print("\n--- Sample ---")
            print(sample("The history of"))
            print("--------------\n")

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            path = save_checkpoint(global_step, epoch, is_best)
            print(f"💾 Checkpoint saved: {path}")

            if args.wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/best_loss": best_val_loss,
                }, step=global_step)

    print(f"✅ Epoch {epoch} complete (step {global_step})")

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
print(f"   Total time     : {total_time:.0f}s ({total_time / 60:.1f} min)")
print(f"   Avg tokens/s   : {avg_tokens_per_sec:,.0f}")
print(f"   Best val loss  : {best_val_loss:.4f}")
print("=" * 60 + "\n")

# Final save
final_path = save_checkpoint(global_step, epoch, is_best=False)
print(f"💾 Final model saved to: {final_path}")

if args.wandb:
    wandb.finish()