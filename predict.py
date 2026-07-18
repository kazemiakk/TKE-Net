"""
TKE-Net: Inference Script
===========================
Usage
-----
python predict.py --checkpoint ./checkpoints/best.pth \
                  --input ./data/test/sample_001.npy --plot

python predict.py --checkpoint ./checkpoints/best.pth \
                  --input_dir ./data/test --output_dir ./results
"""

import argparse, glob
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model   import TKENet
from src.dataset import _load_npy, _load_csv, _normalize_minmax


def parse_args():
    p = argparse.ArgumentParser(description="TKE-Net inference")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--input",       type=str, default="")
    p.add_argument("--input_dir",   type=str, default="")
    p.add_argument("--fmt",         type=str, default="npy",
                   choices=["npy", "csv"])
    p.add_argument("--output",      type=str, default="")
    p.add_argument("--output_dir",  type=str, default="./results")
    p.add_argument("--n_res_lr",    type=int, default=8)
    p.add_argument("--n_res_hr",    type=int, default=4)
    p.add_argument("--n_filters",   type=int, default=64)
    p.add_argument("--upscale",     type=int, default=2)
    p.add_argument("--plot",        action="store_true")
    p.add_argument("--device",      type=str, default="")
    return p.parse_args()


def get_device(r):
    if r: return torch.device(r)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def preprocess(path, fmt):
    if fmt == "npy":
        lr, hr = _load_npy(path)
    else:
        lr, hr = _load_csv(path)
    lr_n = _normalize_minmax(lr)
    hr_n = _normalize_minmax(hr)
    return (torch.from_numpy(lr_n).unsqueeze(0).unsqueeze(0),
            hr_n, lr_n)


@torch.no_grad()
def infer(model, lr_t, device):
    model.eval()
    return model(lr_t.to(device)).squeeze().cpu().numpy()


def main():
    args   = parse_args()
    device = get_device(args.device)
    model  = TKENet(n_res_lr=args.n_res_lr, n_res_hr=args.n_res_hr,
                    n_filters=args.n_filters,
                    upscale_factor=args.upscale).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded: {args.checkpoint}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        lr_t, hr_n, lr_n = preprocess(args.input, args.fmt)
        sr = infer(model, lr_t, device)
        out = args.output or str(out_dir / "prediction.npy")
        np.save(out, sr)
        print(f"Saved SR TKE to {out}  shape={sr.shape}")
        if args.plot:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, img, t in zip(axes,
                                  [lr_n, sr, hr_n],
                                  ["Low-Res (input)", "TKE-Net (SR)", "High-Res (ref)"]):
                ax.imshow(img, cmap="hot"); ax.set_title(t); ax.axis("off")
            plt.tight_layout(); plt.show()
        return

    if args.input_dir:
        pat  = "*.npy" if args.fmt == "npy" else "*_lr.csv"
        files = sorted(glob.glob(str(Path(args.input_dir) / pat)))
        if args.fmt == "csv":
            files = [f[:-len("_lr.csv")] for f in files]
        print(f"Batch: {len(files)} samples")
        for f in files:
            name = Path(f).stem
            lr_t, _, _ = preprocess(f, args.fmt)
            sr = infer(model, lr_t, device)
            np.save(str(out_dir / f"{name}_sr.npy"), sr)
        print(f"Done. Results in {out_dir}")
        return

    print("Provide --input or --input_dir.")


if __name__ == "__main__":
    main()
