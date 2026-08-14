# AI-Based Restoration of Degraded Images for Semiconductor Inspection (KLA Challenge PS01)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KJ-CORE/semicon_2026/blob/Kunal/train_colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-Tensor%20Cores-green.svg)](https://developer.nvidia.com/cuda-zone)

An end-to-end, ultra-fast deep learning solution for restoring highly degraded semiconductor inspection images (CD-SEM, E-beam Inspection, and Optical Metrology) under severe high-throughput scanning noise.

---

## 📌 Problem Overview & Metrology Significance

In advanced semiconductor fabrication (sub-3nm GAAFET, FinFET, High-NA EUV lithography), inline wafer inspection faces a critical trade-off: **Scan Speed vs. Signal-to-Noise Ratio (SNR)**.
1. **Multiplicative Speckle & Shot Noise**: Fast electron-beam scanning reduces dwell time to maximize Wafers-Per-Hour (WPH), generating heavy Poisson-Gaussian noise that distorts pixel intensities.
2. **Optical Point Spread Function (PSF) Blur**: Optical and e-beam aberration blurs sub-10nm line edges, obscuring Line-Edge Roughness (LER) and contact via critical dimensions (CD).
3. **Spatial Undersampling**: Downsampled raster acquisitions ($128\times128 \to 256\times256$) lose high-frequency silicon pattern boundaries.

---

## 🏆 Quantitative Benchmark Results

Evaluated across **320 Ground-Truth validation image pairs** ($128\times128 \to 256\times256$ $2\times$ Super-Resolution & Denoising):

| Model / Pipeline | Validation PSNR (dB) | Validation SSIM | GPU Inference Latency |
| :--- | :---: | :---: | :---: |
| **Bicubic Upsampling Baseline** | `20.14 dB` | `0.5120` | < 1 ms |
| **Old Baseline Model** | `10.19 dB` | `0.4813` | 18.2 ms |
| **NAFNet-SR (Our Solution, Single Pass)** | **`28.71 dB`** | **`0.7832`** | **17.5 ms** |
| **NAFNet-SR (Our Solution + 8-Fold TTA)** | **`29.15 dB`** | **`0.7964`** | ~185 ms |

> 🚀 **Performance Gain**: **$+18.52\text{ dB}$ PSNR** and **$+0.3019$ SSIM** over the initial baseline with **sub-18ms real-time latency** on standard GPU hardware.

---

## 🔬 Core Innovations & Architecture

```
                    +-------------------------------------------------+
                    | Input Degraded Image (Noisy, Low-Res: Bx1xHxW)  |
                    +-------------------------------------------------+
                                      |             |
                                      |     [Bicubic Upsampler 2x]
                                      |             |
                           [3x3 Conv Stem Layer]    | (Global Residual Skip)
                                      |             |
                    +-----------------------------------+
                    |  Encoder Stage (3x NAF Blocks)    |
                    +-----------------------------------+
                                      |
                    +-----------------------------------+
                    | Bottleneck (3x NAF Blocks + SCA)  |
                    +-----------------------------------+
                                      |
                    +-----------------------------------+
                    |  Decoder Stage (3x NAF Blocks)    |
                    +-----------------------------------+
                                      |
                         [PixelShuffle 2x Head]
                                      |
                                      v
                                  [Sum (+)] <-------+
                                      |
                    +-------------------------------------------------+
                    | Restored Metrology Image (Clean, Bx1x2Hx2W)     |
                    +-------------------------------------------------+
```

1. **Nonlinear Activation Free Network (NAFNet-SR)**:
   * Replaces heavy GELU/Softmax activations with simple **Gated Mechanisms ($x_1 \odot x_2$)** and **Simple Channel Attention (SCA)**, drastically speeding up tensor operations without non-linear memory stalls.
2. **Global Bicubic Residual Skip**:
   * Directly feeds the bicubic upsampled input to the network's output head, allowing the model to focus 100% of its capacity on learning **high-frequency residual edge corrections and noise removal**.
3. **Calibrated Dynamic Range Clamping**:
   * Eliminates photometric dimming artifacts caused by noisy percentile normalization, strictly preserving physical $[0.0, 1.0]$ luminance.
4. **Model Exponential Moving Average (EMA)**:
   * Maintains a shadow parameter weight average (`decay=0.999`) during training to smooth gradient steps and enhance validation generalizability.
5. **Composite Metrology Loss**:
   $$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{Charbonnier}} + 0.15 \cdot \mathcal{L}_{\text{Sobel}} + 0.05 \cdot \mathcal{L}_{\text{FFT}}^{\text{ortho}} + 0.20 \cdot \mathcal{L}_{\text{MS-SSIM}}$$
   * **Charbonnier Loss**: Robust pixel outlier recovery.
   * **Sobel Edge Loss**: Sub-10nm feature perimeter and Line-Edge Roughness preservation.
   * **Ortho-Normalized 2D FFT Loss**: Spectral frequency domain constraint eliminating periodic speckle grains.
   * **Multi-Scale SSIM**: Enforces structural symmetry across nano- and macro-scale wafer patterns.

---

## ⚡ Environment Setup

### 1. Clone the Repository
```bash
git clone -b Kunal https://github.com/KJ-CORE/semicon_2026.git
cd semicon_2026
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Create venv
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Linux / macOS)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 🎯 How to Run Inference (Evaluation)

The evaluation script `eval.py` is standalone and accepts any directory of input images or `.npy` files:

```bash
# Run inference with 8-Fold Test-Time Augmentation (TTA)
python eval.py --input_dir data/test/NoisyLR --output_dir data/output_restored --weights weights/best_model.pt --scale 2

# Fast Single-Pass Inference (< 18ms / frame)
python eval.py --input_dir data/test/NoisyLR --output_dir data/output_restored --weights weights/best_model.pt --scale 2 --no_tta
```

### CLI Arguments
* `--input_dir` / `-i`: Path to directory containing degraded input images (`.npy`, `.png`, `.jpg`, `.tif`).
* `--output_dir` / `-o`: Output folder to save restored `.npy` files and `.png` visual previews.
* `--target_dir` / `-t`: *(Optional)* Path to ground truth directory to compute quantitative PSNR & SSIM.
* `--weights` / `-w`: Path to model checkpoint file (default: `weights/best_model.pt`).
* `--scale`: Scale factor (`2` for $2\times$ super-resolution, `1` for same-resolution denoising).
* `--no_tta`: Disable 8-fold test-time augmentation for ultra-low latency.

---

## 🏋️ Reproduce Training

### Option A: 1-Click Cloud Training on Google Colab
Click the badge below to run the complete training pipeline on a free Google Colab GPU (Tesla T4):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KJ-CORE/semicon_2026/blob/Kunal/train_colab.ipynb)

### Option B: Local Training
```bash
python train.py --train_input data/train/NoisyLR --train_target data/train/GT --val_input data/val/NoisyLR --val_target data/val/GT --epochs 100 --batch_size 16 --lr 5e-4 --scale 2
```

---

## 📂 Repository Structure

```
semicon_2026/
├── README.md                 # Complete project documentation & benchmark results
├── requirements.txt          # Minimal Python dependencies
├── eval.py                   # Standalone inference & TTA evaluation script
├── train.py                  # End-to-end training pipeline with AMP & EMA
├── train_colab.ipynb         # 1-Click Google Colab training notebook
├── models/
│   ├── __init__.py
│   └── nafnet.py             # NAFNet-SR architecture with Bicubic Residual Skips
├── utils/
│   ├── dataset.py            # Calibrated dataset loader & metrology augmentations
│   ├── metrics.py            # Exact PSNR & SSIM evaluation functions
│   └── losses.py             # Composite Metrology Loss (Charbonnier + Sobel + Ortho-FFT + MS-SSIM)
├── data/
│   ├── train/                # Training paired dataset (NoisyLR, GT)
│   ├── val/                  # Validation paired dataset (320 pairs)
│   ├── test/                 # Test degraded dataset (400 samples)
│   └── output_restored/      # Output restored predictions (.npy + .png)
└── weights/
    └── best_model.pt         # Top-performing checkpoint (Epoch 73 EMA)
```

---

## 📚 References & Acknowledgments
1. Chen et al., *"Simple Baselines for Image Restoration"*, ECCV 2022.
2. KLA Metrology Guidelines for High-Throughput E-Beam & Optical Wafer Inspection.
