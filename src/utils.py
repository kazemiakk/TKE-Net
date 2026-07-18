"""
TKE-Net: Visualisation & Utility Functions
============================================
- plot_tke_profile     : Spatially-averaged TKE vs axial distance
- plot_tke_contour     : 2D TKE contour maps (LR vs SR vs HR)
- plot_psnr_histogram  : Histogram of PSNR values over test set
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_tke_profile(
    tke_hr:  np.ndarray,
    tke_sr:  np.ndarray,
    tke_lr:  Optional[np.ndarray] = None,
    axis:    int = 1,
    title:   str = "TKE along the Stenosis Phantom",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Spatial-averaged TKE line plot (replicates paper Fig. 3).

    Parameters
    ----------
    tke_hr : (H, W) high-resolution reference TKE
    tke_sr : (H, W) TKE-Net super-resolved prediction
    tke_lr : (H//2, W//2) optional low-resolution baseline
    axis   : axis along which to average (0=columns, 1=rows)
    """
    hr_line = tke_hr.mean(axis=axis)
    sr_line = tke_sr.mean(axis=axis)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(hr_line))
    ax.plot(x, hr_line, "b-",  linewidth=2, label="High-Resolution (reference)")
    ax.plot(x, sr_line, "r--", linewidth=2, label="TKE-Net")

    if tke_lr is not None:
        lr_line = tke_lr.mean(axis=axis)
        # Interpolate LR to HR size for plotting on the same axis
        from scipy.ndimage import zoom
        lr_up = zoom(lr_line, len(hr_line) / len(lr_line))
        ax.plot(x, lr_up, "g:",  linewidth=1.5, label="Low-Resolution")

    ax.set_xlabel("Axial position (slice index)")
    ax.set_ylabel("Spatially averaged TKE")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_tke_contour(
    tke_hr:    np.ndarray,
    tke_sr:    np.ndarray,
    tke_lr:    Optional[np.ndarray] = None,
    plane:     str = "axial",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side contour plots (replicates paper Fig. 4 & 5).

    Parameters
    ----------
    tke_hr : (H, W)   ground truth HR TKE
    tke_sr : (H, W)   TKE-Net super-resolved
    tke_lr : (H//2, W//2) optional LR baseline
    plane  : 'axial' or 'sagittal' (label only)
    """
    ncols = 3 if tke_lr is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    vmax = tke_hr.max()
    kw   = dict(cmap="hot", vmin=0, vmax=vmax, aspect="auto")

    col = 0
    if tke_lr is not None:
        im = axes[col].imshow(tke_lr, **kw)
        axes[col].set_title("Low Resolution (LR)")
        plt.colorbar(im, ax=axes[col])
        col += 1

    im = axes[col].imshow(tke_sr, **kw)
    axes[col].set_title("TKE-Net (SR)")
    plt.colorbar(im, ax=axes[col])
    col += 1

    im = axes[col].imshow(tke_hr, **kw)
    axes[col].set_title("High Resolution (HR)")
    plt.colorbar(im, ax=axes[col])

    fig.suptitle(f"TKE Contours — {plane} plane  [peak systole]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_psnr_histogram(
    psnr_values: np.ndarray,
    title: str = "PSNR Distribution over Test Set",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of per-sample PSNR values."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(psnr_values, bins=20, color="#3a7abf", edgecolor="white", alpha=0.85)
    ax.axvline(psnr_values.mean(), color="red", linewidth=1.5,
               label=f"Mean = {psnr_values.mean():.1f} dB")
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
