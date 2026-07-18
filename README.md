# TKE-Net: Super-Resolution Turbulent Kinetic Energy Maps from 4D Flow MRI

[![Paper](https://img.shields.io/badge/Paper-IEEE%20ISBI%202023-blue?style=flat&logo=ieee)](https://doi.org/10.1109/ISBI53787.2023.10230629)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?logo=pytorch)](https://pytorch.org)

> **Official PyTorch implementation of TKE-Net**, a deep learning super-resolution network for estimating **Turbulent Kinetic Energy (TKE) maps** directly from low-resolution 4D flow MRI data, with an upsampling factor of 2.

---

## 📄 Abstract

Arterial stenosis requires accurate quantification of hemodynamic parameters for diagnosis and prognosis. Variations in velocity derivatives correlate with pressure gradients — a key marker of hemodynamic significance. While 4D flow MRI provides time-resolved 3D velocity mapping, **low spatial resolution** hinders accurate quantification of velocity fluctuations and TKE.

**TKE-Net** is a ResNet-based convolutional neural network that learns the direct mapping from **noisy low-resolution TKE maps** to **noise-free high-resolution TKE maps**, trained on CFD simulations and validated on in-vitro 4D flow MRI.

| Metric | CFD Data | In-Vitro 4D Flow MRI |
|--------|----------|-----------------------|
| PSNR   | 39.3 ± 0.6 dB | 32.4 ± 0.4 dB |
| RMSTKE | 0.016 ± 0.05  | 0.034 ± 0.09  |

---

## 🏗️ Network Architecture

```
Low-Resolution TKE Input
         │
    [Conv + LeakyReLU]
         │
  ┌──────▼──────────────────────────┐
  │  8 × Residual Blocks            │  ← Low-resolution space (denoising)
  │  (symmetric padding + LeakyReLU)│
  └──────┬──────────────────────────┘
         │
  [Bilinear Upsampling ×2]          ← 2× super-resolution
         │
  ┌──────▼──────────────────────────┐
  │  4 × Residual Blocks            │  ← High-resolution space (refinement)
  └──────┬──────────────────────────┘
         │
    [Conv + tanh]                   ← Output in [-1, 1]
         │
High-Resolution TKE Map (2× size)
```

Each **Residual Block** = Conv → LeakyReLU → Conv → (add skip)

**TKE Definition:**
$$TKE = \frac{1}{2}\rho\left(u'^2_x + u'^2_y + u'^2_z\right)$$

where $u'$ is the fluctuating part of velocity computed as the RMS deviation from the spatial mean.

---

## 📂 Data Pipeline

```
High-Res CFD TKE
       │
  [FFT → truncate high-freq → add noise → IFFT]   ← k-space downsampling
       │
Low-Res TKE (½ matrix size)
       │
   [TKE-Net]
       │
Super-Res TKE (2× output)
```

---

## 📁 Repository Structure

```
TKE-Net/
├── src/
│   ├── model.py      # TKENet ResNet architecture
│   ├── loss.py       # MSE loss + PSNR metric
│   ├── dataset.py    # FlowTKEDataset — NPY/CSV generic loader
│   └── utils.py      # PSNR, RMSTKE metrics + visualisation
├── train.py          # Training script
├── predict.py        # Single and batch inference
├── evaluate.py       # Quantitative evaluation (PSNR, RMSTKE)
├── configs/
│   └── config.yaml   # All hyperparameters
├── data/             # (Place your dataset here)
├── checkpoints/      # Saved model weights
├── results/          # Evaluation outputs
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/kazemiakk/TKE-Net.git
cd TKE-Net

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📂 Data Format

Place your data under `./data/` following this structure:

```
data/
├── train/
│   ├── sample_000.npy     ← shape (2, H, W) : [LR_TKE, HR_TKE]
│   ├── sample_001.npy
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### Supported formats

| Format | Description |
|--------|-------------|
| **NPY** | Single `.npy` file per sample, shape `(2, H, W)` → channels `[LR_TKE, HR_TKE]` |
| **CSV** | Two CSV files per sample: `<prefix>_lr.csv`, `<prefix>_hr.csv` |

- `H × W` = spatial dimensions of the **low-resolution** TKE map
- The network outputs at `2H × 2W` (upsampling factor = 2)

---

## 🚀 Usage

### Training

```bash
python train.py \
  --data_root ./data \
  --fmt npy \
  --output_dir ./checkpoints \
  --epochs 500 \
  --batch_size 20 \
  --lr 1e-4
```

### Inference

```bash
# Single sample
python predict.py \
  --checkpoint ./checkpoints/best.pth \
  --input ./data/test/sample_001.npy \
  --output ./results/pred_001.npy \
  --plot

# Batch
python predict.py \
  --checkpoint ./checkpoints/best.pth \
  --input_dir ./data/test \
  --output_dir ./results
```

### Evaluation

```bash
python evaluate.py \
  --checkpoint ./checkpoints/best.pth \
  --data_root ./data \
  --split test \
  --output_dir ./results/eval
```

Outputs: `metrics.json`, `psnr_histogram.pdf`, `tke_profile.pdf`

---

## 📊 Results (from the paper)

| Method | PSNR (CFD) | RMSTKE (CFD) | PSNR (MRI) | RMSTKE (MRI) |
|--------|------------|--------------|------------|--------------|
| Low-resolution (baseline) | < 30 dB | > 0.05 | < 25 dB | > 0.08 |
| **TKE-Net (ours)** | **39.3 ± 0.6 dB** | **0.016 ± 0.05** | **32.4 ± 0.4 dB** | **0.034 ± 0.09** |

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kazemi2023tkenet,
  title     = {{TKE-Net}: Deep Learning for Estimation of Super-Resolved
               Turbulent Kinetic Energy Maps from {4D}-Flow {MRI} Data},
  author    = {Kazemi, Amirkhosro and Stoddard, Marcus and Amini, Amir A},
  booktitle = {2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  pages     = {1--4},
  year      = {2023},
  publisher = {IEEE},
  doi       = {10.1109/ISBI53787.2023.10230629}
}
```

---

## 🏛️ Acknowledgements

This work was supported by the **National Institutes of Health**  
(Grant No. 5R21HL132263).

Research conducted at the **Medical Imaging Lab, Department of Electrical
and Computer Engineering, University of Louisville** and the **Robley Rex
Veterans Affairs Medical Center, Louisville, KY**.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
