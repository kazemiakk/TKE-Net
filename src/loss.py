"""
TKE-Net: Loss Function & Metrics
==================================
- TKENetLoss   : MSE loss (as in the paper)
- compute_psnr : Peak Signal-to-Noise Ratio (paper Equation 3)
- compute_rmstke : Normalized RMSE (paper Equation 4)
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TKENetLoss(nn.Module):
    """
    Mean Squared Error loss for TKE super-resolution.
    (Used with Adam optimizer, as described in the paper.)
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmstke(tke_ref: np.ndarray, tke_est: np.ndarray) -> float:
    """
    Normalized Root Mean Squared Error for TKE (paper Equation 4).

    RMSTKE = (1 / max(||TKE||)) * sqrt( (1/N) * Σ ||TKE_i - TKE_est_i||² )

    Parameters
    ----------
    tke_ref : ground truth TKE array (any shape)
    tke_est : estimated TKE array (same shape)

    Returns
    -------
    rmstke : float
    """
    max_tke = np.abs(tke_ref).max()
    if max_tke < 1e-8:
        return float("nan")
    rmse = np.sqrt(np.mean((tke_ref - tke_est) ** 2))
    return float(rmse / max_tke)


def compute_psnr(tke_ref: np.ndarray, tke_est: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (paper Equation 3).

    PSNR = 20 * log10(1 / RMSTKE)  [dB]

    Parameters
    ----------
    tke_ref : ground truth TKE
    tke_est : estimated TKE

    Returns
    -------
    psnr : float (dB)
    """
    rmstke = compute_rmstke(tke_ref, tke_est)
    if np.isnan(rmstke) or rmstke < 1e-12:
        return float("inf")
    return float(20.0 * np.log10(1.0 / rmstke))


if __name__ == "__main__":
    ref = np.random.rand(60, 60).astype(np.float32) * 5000
    est = ref + np.random.randn(*ref.shape) * 50
    print(f"RMSTKE : {compute_rmstke(ref, est):.4f}")
    print(f"PSNR   : {compute_psnr(ref, est):.2f} dB")
