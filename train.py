"""
TKE-Net: Training Script
==========================
Training pipeline with train/val split, early stopping, and checkpoint saving.
Paper training config: Adam, lr=1e-4, batch=20, 500 epochs, MSE loss.

Usage
-----
python train.py --data_root ./data --fmt npy --output_dir ./checkpoints
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.model   import TKENet
from src.loss    import TKENetLoss
from src.dataset import FlowTKEDataset


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train TKE-Net")
    p.add_argument("--data_root",   type=str, default="./data")
    p.add_argument("--fmt",         type=str, default="npy",
                   choices=["npy", "csv"])
    p.add_argument("--output_dir",  type=str, default="./checkpoints")
    p.add_argument("--n_res_lr",    type=int, default=8,
                   help="Residual blocks before upsampling")
    p.add_argument("--n_res_hr",    type=int, default=4,
                   help="Residual blocks after upsampling")
    p.add_argument("--n_filters",   type=int, default=64)
    p.add_argument("--upscale",     type=int, default=2)
    p.add_argument("--epochs",      type=int, default=500)
    p.add_argument("--batch_size",  type=int, default=20)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--val_split",   type=float, default=0.2,
                   help="Fraction of training data used for validation")
    p.add_argument("--patience",    type=int, default=50)
    p.add_argument("--check_every", type=int, default=5)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      type=str, default="")
    return p.parse_args()


def get_device(r):
    if r: return torch.device(r)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for lr_t, hr_t in loader:
        lr_t, hr_t = lr_t.to(device), hr_t.to(device)
        optimizer.zero_grad()
        pred = model(lr_t)
        loss = criterion(pred, hr_t)
        loss.backward()
        optimizer.step()
        total += loss.item() * lr_t.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for lr_t, hr_t in loader:
        lr_t, hr_t = lr_t.to(device), hr_t.to(device)
        pred = model(lr_t)
        total += criterion(pred, hr_t).item() * lr_t.size(0)
    return total / len(loader.dataset)


def main():
    args   = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    full_ds = FlowTKEDataset(args.data_root, split="train",
                              fmt=args.fmt, upscale_factor=args.upscale,
                              augment=True)
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    print(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ---- Model ----
    model     = TKENet(n_res_lr=args.n_res_lr, n_res_hr=args.n_res_hr,
                       n_filters=args.n_filters,
                       upscale_factor=args.upscale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = TKENetLoss()

    best_val   = float("inf")
    best_w     = copy.deepcopy(model.state_dict())
    no_improve = 0
    history    = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history["train"].append(train_loss)

        if epoch % args.check_every == 0:
            val_loss = val_epoch(model, val_loader, criterion, device)
            history["val"].append({"epoch": epoch, "loss": val_loss})
            print(f"Epoch {epoch:>4d}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"[{time.time()-t0:.1f}s]")

            if val_loss < best_val:
                best_val = val_loss
                best_w   = copy.deepcopy(model.state_dict())
                no_improve = 0
                torch.save(best_w, str(out_dir / "best.pth"))
            else:
                no_improve += args.check_every
                if no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    with open(str(out_dir / "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val loss: {best_val:.6f}")
    print(f"Checkpoint saved to {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
