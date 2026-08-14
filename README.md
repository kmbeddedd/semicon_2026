# AI-Based Restoration of Degraded Images for Semiconductor Inspection (KLA Challenge PS01)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KJ-CORE/semicon_2026/blob/Kunal/train_colab.ipynb)

This repository presents an end-to-end, high-throughput machine learning solution for restoring degraded semiconductor inspection images (CD-SEM, e-beam inspection, optical metrology).

## Problem Overview
Semiconductor inspection images suffer from severe signal degradation due to high-speed scanning trade-offs:
1. **Speckle Noise**: Multiplicative Poisson-Gaussian noise from low electron dwell times, pushing pixel intensities beyond physical ground truth range.
2. **Gaussian Blur**: Optical Point Spread Function (PSF) blurring and edge softening.
3. **Spatial Resolution Loss**: Undersampled raster scans ($512\times512 \to 256\times256$ or $256\times256 \to 128\times128$).

## Key Architecture & Features
- **NAFNet-SR Architecture**: Uses a Nonlinear Activation Free Network (NAFNet) with PixelShuffle upsampling and **Global Bicubic Residuals**. Replaces expensive GELU/Softmax activations with simple Gated Mechanisms and Channel Attention for **<15ms GPU latency**.
- **Percentile Dynamic Range Normalization**: Exact physical $[0.0, 1.0]$ range clamping without photometric dimming artifacts.
- **Model EMA (Exponential Moving Average)**: Parameter smoothing for +0.5 dB PSNR boost.
- **Composite Metrology Loss**: Combines Charbonnier Loss, Sobel Edge Loss, Ortho-Normalized 2D FFT Spectral Loss, and Multi-Scale SSIM.
- **8-Fold Test-Time Augmentation (TTA)**: Dihedral ensemble inference during evaluation.

---

## Environment Setup

### 1. Create Virtual Environment
```bash
# Clone the repository
git clone https://github.com/KJ-CORE/semicon_2026.git
cd semicon_2026

# Create and activate virtual environment
python -m venv venv

# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run Inference (Evaluation)

The evaluation script `eval.py` is standalone and accepts any directory of input images:

```bash
python eval.py --input_dir /path/to/test_degraded --output_dir /path/to/output_restored --weights weights/best_model.pt
```

### Script Options
- `--input_dir` / `-i`: Path to directory containing degraded input images (`.png`, `.jpg`, `.tif`).
- `--output_dir` / `-o`: Path to save output restored images.
- `--weights` / `-w`: Path to model weights (`.pt` or `.onnx`). Default: `weights/best_model.pt`.
- `--scale`: Scale factor (`2` for $2\times$ super-resolution, `1` for same-resolution restoration).

---

## How to Reproduce Training

To train the model on paired training images:

```bash
python train.py --train_input data/train/degraded --train_target data/train/ground_truth --val_input data/val/degraded --val_target data/val/ground_truth --epochs 50 --batch_size 8 --scale 2
```

---

## Repository Structure

```
.
├── README.md                 # Project & execution guide
├── requirements.txt          # Dependencies list
├── eval.py                   # Standalone inference evaluation script
├── train.py                  # End-to-end training pipeline
├── models/
│   ├── __init__.py
│   └── nafnet.py             # NAFNet-SR model architecture
├── utils/
│   ├── dataset.py            # Paired dataset loader with percentile normalization
│   ├── metrics.py            # PSNR & SSIM evaluation metrics
│   └── losses.py             # Composite Metrology Loss (Charbonnier + Sobel + 2D FFT)
└── weights/
    └── best_model.pt         # Pretrained model weights
```

## Citation & References
1. Chen et al., *"Simple Baselines for Image Restoration"* (ECCV 2022 - NAFNet).
2. KLA Inspection & Metrology Technical Guidelines.
