"""
TKE-Net: Evaluation Script
=============================
Computes PSNR and RMSTKE over the full test set and saves
contour maps and a PSNR histogram.

Usage
-----
python evaluate.py --checkpoint ./checkpoints/best.pth \
                   --data_root ./data --split test \
                   --output_dir ./results/eval
"""

import argparse, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model   import TKENet
from src.dataset import FlowTKEDataset
from src.loss    import compute_psnr, compute_rmstke
from src.utils   import (plot_tke_profile, plot_tke_contour,
                          plot_psnr_histogram)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TKE-Net")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--data_root",   type=str, default="./data")
    p.add_argument("--split",       type=str, default="test")
    p.add_argument("--fmt",         type=str, default="npy",
                   choices=["npy", "csv"])
    p.add_argument("--output_dir",  type=str, default="./results/eval")
    p.add_argument("--n_res_lr",    type=int, default=8)
    p.add_argument("--n_res_hr",    type=int, default=4)
    p.add_argument("--n_filters",   type=int, default=64)
    p.add_argument("--upscale",     type=int, default=2)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--device",      type=str, default="")
    return p.parse_args()


def get_device(r):
    if r: return torch.device(r)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    psnr_list, rmstke_list = [], []
    sample_lr, sample_sr, sample_hr = None, None, None

    for lr_t, hr_t in loader:
        lr_t = lr_t.to(device)
        sr_t = model(lr_t)

        for i in range(lr_t.size(0)):
            hr_np = hr_t[i, 0].numpy()
            sr_np = sr_t[i, 0].cpu().numpy()
            lr_np = lr_t[i, 0].cpu().numpy()
            psnr_list.append(compute_psnr(hr_np, sr_np))
            rmstke_list.append(compute_rmstke(hr_np, sr_np))
            if sample_hr is None:
                sample_lr, sample_sr, sample_hr = lr_np, sr_np, hr_np

    return (np.array(psnr_list), np.array(rmstke_list),
            sample_lr, sample_sr, sample_hr)


def main():
    args   = parse_args()
    device = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TKENet(n_res_lr=args.n_res_lr, n_res_hr=args.n_res_hr,
                   n_filters=args.n_filters,
                   upscale_factor=args.upscale).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded: {args.checkpoint}")

    ds = FlowTKEDataset(args.data_root, split=args.split,
                        fmt=args.fmt, upscale_factor=args.upscale,
                        augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating on {len(ds)} samples ({args.split})...")

    psnr_arr, rmstke_arr, slr, ssr, shr = run_eval(model, loader, device)

    results = {
        "n_samples"      : len(ds),
        "PSNR_mean_dB"   : float(psnr_arr.mean()),
        "PSNR_std_dB"    : float(psnr_arr.std()),
        "RMSTKE_mean"    : float(rmstke_arr.mean()),
        "RMSTKE_std"     : float(rmstke_arr.std()),
    }

    print("\n--- Results ---")
    for k, v in results.items():
        print(f"  {k:<22}: {v:.4f}" if isinstance(v, float) else
              f"  {k:<22}: {v}")

    with open(str(out_dir / "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Figures
    plot_psnr_histogram(psnr_arr,
                        save_path=str(out_dir / "psnr_histogram.pdf"))
    if shr is not None:
        plot_tke_profile(shr, ssr, slr,
                         save_path=str(out_dir / "tke_profile.pdf"))
        plot_tke_contour(shr, ssr, slr,
                         save_path=str(out_dir / "tke_contour.pdf"))
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
