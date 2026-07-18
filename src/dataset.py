"""
TKE-Net: Dataset & Data Loader
================================
Generic loader for low-resolution / high-resolution TKE map pairs.
Supports NPY and CSV formats.

Data preparation pipeline (from the paper)
------------------------------------------
1. High-resolution CFD TKE maps are generated.
2. Gaussian noise is added to CFD velocity components.
3. Velocity components are converted to k-space (FFT).
4. High frequencies are truncated (matrix size halved per dimension).
5. IFFT converts back to spatial domain → low-resolution TKE.
6. Training: LR TKE (input) → HR TKE (target).

Supported file formats
----------------------
NPY : single .npy file per sample, shape (2, H, W)
      channel 0 → LR TKE,  channel 1 → HR TKE
CSV : two files per sample
      <prefix>_lr.csv  (H × W)
      <prefix>_hr.csv  (2H × 2W)
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# K-space downsampling (matches paper's data preparation)
# ---------------------------------------------------------------------------

def kspace_downsample(velocity: np.ndarray,
                      sigma_min: float = 0.01,
                      sigma_max: float = 0.30) -> np.ndarray:
    """
    Simulate low-resolution acquisition via k-space truncation.

    Steps (paper Section 2.3):
      1. Add Gaussian noise to velocity field.
      2. FFT → truncate outer half of k-space → IFFT.

    Parameters
    ----------
    velocity : (H, W) velocity component array
    sigma_min, sigma_max : noise SD range

    Returns
    -------
    lr_velocity : (H//2, W//2) low-resolution velocity
    """
    sigma = np.random.uniform(sigma_min, sigma_max)
    v_noisy = velocity + np.random.normal(0, sigma, velocity.shape)

    kspace = np.fft.fft2(v_noisy)
    kspace_shift = np.fft.fftshift(kspace)

    H, W = kspace_shift.shape
    H2, W2 = H // 2, W // 2
    h0, w0 = H // 4, W // 4
    kspace_crop = kspace_shift[h0:h0 + H2, w0:w0 + W2]

    lr = np.fft.ifft2(np.fft.ifftshift(kspace_crop)).real
    return lr.astype(np.float32)


def compute_tke(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                rho: float = 1060.0) -> np.ndarray:
    """
    Compute Turbulent Kinetic Energy from three velocity components.

    TKE = 0.5 * rho * (u'x² + u'y² + u'z²)

    where u' = sqrt(mean((u_i - mean(u))²)) at each spatial location.

    Parameters
    ----------
    vx, vy, vz : (H, W) or (T, H, W) arrays
    rho : blood density (kg/m³, default 1060)

    Returns
    -------
    tke : (H, W) or (T, H, W) TKE array
    """
    def fluctuation(v: np.ndarray) -> np.ndarray:
        mean = v.mean(axis=-1, keepdims=True) if v.ndim > 2 else v.mean()
        return v - mean

    u_x = fluctuation(vx)
    u_y = fluctuation(vy)
    u_z = fluctuation(vz)
    return 0.5 * rho * (u_x ** 2 + u_y ** 2 + u_z ** 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path).astype(np.float32)
    assert data.shape[0] == 2, \
        f"Expected shape (2, H, W) [LR, HR], got {data.shape}"
    return data[0], data[1]


def _load_csv(prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    def rd(suf): return np.loadtxt(f"{prefix}_{suf}.csv",
                                   delimiter=',', dtype=np.float32)
    return rd("lr"), rd("hr")


def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] for tanh output compatibility."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return 2.0 * (arr - lo) / (hi - lo) - 1.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FlowTKEDataset(Dataset):
    """
    PyTorch Dataset for TKE super-resolution.

    Parameters
    ----------
    data_root : str
        Root directory (contains train/, val/, test/).
    split : str
        'train', 'val', or 'test'.
    fmt : str
        'npy' or 'csv'.
    upscale_factor : int
        Expected spatial upscaling factor (default 2).
    augment : bool
        If True, apply random horizontal/vertical flips.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        fmt: str = "npy",
        upscale_factor: int = 2,
        augment: bool = False,
    ):
        super().__init__()
        self.data_root     = Path(data_root)
        self.split         = split
        self.fmt           = fmt.lower()
        self.upscale_factor = upscale_factor
        self.augment       = augment
        self.samples: List[str] = self._discover()

        if not self.samples:
            raise RuntimeError(
                f"No samples found in {self.data_root / split} "
                f"with format '{self.fmt}'."
            )

    def _discover(self) -> List[str]:
        d = self.data_root / self.split
        if not d.exists():
            raise FileNotFoundError(f"Split directory not found: {d}")
        if self.fmt == "npy":
            return sorted(glob.glob(str(d / "*.npy")))
        elif self.fmt == "csv":
            return [p[:-len("_lr.csv")]
                    for p in sorted(glob.glob(str(d / "*_lr.csv")))]
        raise ValueError(f"Unsupported format '{self.fmt}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.fmt == "npy":
            lr, hr = _load_npy(self.samples[idx])
        else:
            lr, hr = _load_csv(self.samples[idx])

        # Normalize both to [-1, 1]
        lr = _normalize_minmax(lr)
        hr = _normalize_minmax(hr)

        # Ensure HR is exactly 2× the LR
        lr_t = torch.from_numpy(lr).unsqueeze(0)   # (1, H, W)
        hr_t = torch.from_numpy(hr).unsqueeze(0)   # (1, 2H, 2W)
        target_h = lr_t.shape[-2] * self.upscale_factor
        target_w = lr_t.shape[-1] * self.upscale_factor
        if hr_t.shape[-2] != target_h or hr_t.shape[-1] != target_w:
            hr_t = F.interpolate(hr_t.unsqueeze(0),
                                 size=(target_h, target_w),
                                 mode='bilinear',
                                 align_corners=True).squeeze(0)

        # Random augmentation
        if self.augment:
            if torch.rand(1) > 0.5:
                lr_t = lr_t.flip(-1)
                hr_t = hr_t.flip(-1)
            if torch.rand(1) > 0.5:
                lr_t = lr_t.flip(-2)
                hr_t = hr_t.flip(-2)

        return lr_t, hr_t


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    root = Path(tempfile.mkdtemp())
    for split in ["train", "val", "test"]:
        (root / split).mkdir()
        for i in range(4):
            lr = np.random.randn(30, 30).astype(np.float32)
            hr = np.random.randn(60, 60).astype(np.float32)
            np.save(str(root / split / f"s{i:03d}.npy"),
                    np.stack([lr, hr]))

    ds = FlowTKEDataset(root, split="train", fmt="npy", augment=True)
    print(f"Dataset length: {len(ds)}")
    lr_t, hr_t = ds[0]
    print(f"  LR: {lr_t.shape}  HR: {hr_t.shape}")
