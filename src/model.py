"""
TKE-Net: Network Architecture
================================
ResNet-based super-resolution network for estimating high-resolution
Turbulent Kinetic Energy (TKE) maps from low-resolution 4D flow MRI.

Architecture (from the paper):
  - 8 residual blocks BEFORE upsampling (low-resolution denoising space)
  - Bilinear upsampling ×2
  - 4 residual blocks AFTER upsampling (high-resolution refinement space)
  - tanh output activation

Based on:
    Kazemi et al., "TKE-Net: Deep Learning for Estimation of Super-Resolved
    Turbulent Kinetic Energy Maps from 4D-Flow MRI Data",
    IEEE ISBI 2023. DOI: 10.1109/ISBI53787.2023.10230629
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Residual block with symmetric (reflect) padding and LeakyReLU.

    Structure: Conv → LeakyReLU → Conv → + skip
    Symmetric padding avoids border artifacts (as described in the paper).
    """

    def __init__(self, channels: int = 64, kernel_size: int = 3,
                 negative_slope: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(channels, channels, kernel_size, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(channels, channels, kernel_size, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# TKE-Net
# ---------------------------------------------------------------------------

class TKENet(nn.Module):
    """
    TKE-Net: Super-Resolution network for TKE estimation.

    Input
    -----
    tke_lr : torch.Tensor of shape (B, 1, H, W)
        Low-resolution TKE map (normalized to [-1, 1]).

    Output
    ------
    tke_sr : torch.Tensor of shape (B, 1, 2H, 2W)
        Super-resolved (×2) TKE map, values in [-1, 1] via tanh.

    Parameters
    ----------
    n_res_lr : int
        Number of residual blocks before upsampling (paper: 8).
    n_res_hr : int
        Number of residual blocks after upsampling (paper: 4).
    n_filters : int
        Number of feature channels in convolutional layers (default: 64).
    upscale_factor : int
        Upsampling factor (paper: 2).
    """

    def __init__(
        self,
        n_res_lr: int = 8,
        n_res_hr: int = 4,
        n_filters: int = 64,
        upscale_factor: int = 2,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor

        # ---- Input convolution ----
        self.input_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, n_filters, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

        # ---- Low-resolution residual blocks (denoising) ----
        self.res_blocks_lr = nn.Sequential(
            *[ResidualBlock(n_filters, negative_slope=negative_slope)
              for _ in range(n_res_lr)]
        )

        # ---- Bilinear upsampling ×2 ----
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor,
            mode='bilinear',
            align_corners=True,
        )

        # ---- High-resolution residual blocks (refinement) ----
        self.res_blocks_hr = nn.Sequential(
            *[ResidualBlock(n_filters, negative_slope=negative_slope)
              for _ in range(n_res_hr)]
        )

        # ---- Output head ----
        self.output_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_filters, 1, kernel_size=3, padding=0, bias=True),
            nn.Tanh(),                  # ensures output ∈ [-1, 1]
        )

    def forward(self, tke_lr: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(tke_lr)          # feature extraction
        x = self.res_blocks_lr(x)            # low-res denoising
        x = self.upsample(x)                 # ×2 spatial upsampling
        x = self.res_blocks_hr(x)            # high-res refinement
        tke_sr = self.output_conv(x)         # [-1, 1] output
        return tke_sr


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = TKENet(n_res_lr=8, n_res_hr=4, n_filters=64, upscale_factor=2)
    B, H, W = 4, 30, 30
    lr = torch.randn(B, 1, H, W)
    sr = model(lr)
    print(f"Input  (LR): {lr.shape}")
    print(f"Output (SR): {sr.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
