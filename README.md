# AI-Based Restoration of Degraded Images for Semiconductor Inspection (KLA Challenge PS01)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kmbeddedd/semicon_2026/blob/Kunal/train_colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-Tensor%20Cores-green.svg)](https://developer.nvidia.com/cuda-zone)

An end-to-end, ultra-fast deep learning solution for restoring highly degraded semiconductor inspection images (CD-SEM, E-beam Inspection, and Optical Metrology) under severe high-throughput scanning noise.

---

## 📌 Problem Overview & Metrology Significance

In advanced semiconductor fabrication (sub-3nm GAAFET, FinFET, High-NA EUV lithography), inline wafer inspection faces a critical trade-off: **Scan Speed vs. Signal-to-Noise Ratio (SNR)**.
1. **Signal-dependent and additive noise**: Fast electron-beam scanning reduces dwell time to maximize Wafers-Per-Hour (WPH), which can introduce shot, electronic, and speckle-like degradation. The included patch regression explores $\text{Var}(y | \mu) = a \mu^2 + b$ but is not a calibrated sensor model.
2. **Spatial Undersampling**: Downsampled raster acquisitions ($128\times128 \to 256\times256$) lose high-frequency silicon pattern boundaries, critical dimension (CD) contacts, and line perimeters.
3. **Ground-Truth Noise Estimate**: Wavelet-MAD estimates a validation noise scale of $\sigma \approx 0.0168$, equivalent to **$38.72\text{ dB}$**. This is a useful high-frequency noise-floor heuristic, not a formal upper bound on model-to-target PSNR.

---

## 🏆 Quantitative Benchmark Results

Evaluated across **320 Ground-Truth validation image pairs** ($128\times128 \to 256\times256$ $2\times$ Super-Resolution & Denoising):

| Model / Pipeline | Validation PSNR (dB) ↑ | Validation SSIM ↑ | Validation LPIPS ↓ | Gain-Normalized Estimate ↑ | GPU Inference Latency | Throughput | Clean-input PSNR\* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Preprocessing-matched Bicubic** | `22.79` | `0.5330` | `0.4400` | `0.0%` (reference) | Not benchmarked | Not benchmarked | `30.80` |
| **NAFNet-SR, FP16 + JIT, batch 8** | **`28.71`** | **`0.7831`** | **`0.2436`** | **`37.2%`** | **`13.35 ms/image`** | **`74.9 images/s`** | **`29.13`** |

> \* Clean-input audit uses the first 50 validation GT images, area-downsampled and restored. The current checkpoint is **1.68 dB below bicubic** on this audit; clean-input gating is not present in the shipped weights.

### 🎯 Key Performance Highlights
1. **Reproduced speed**: `13.35 ms/image` (`74.9 images/s`) on an RTX 2050 with FP16, TorchScript JIT and batch size 8. Timing covers synchronized model compute, not image I/O or metrics.
2. **Measured restoration gain**: `+5.92 dB` PSNR over the preprocessing-matched bicubic baseline.
3. **Perceptual improvement**: LPIPS falls from `0.4400` to `0.2436`, a **44.6% reduction**.
4. **Gain-normalized estimate**:
   $$\frac{28.71 - 22.79}{38.72 - 22.79} = \mathbf{37.2\%}$$
   The `38.72 dB` Wavelet-MAD value is an estimate rather than a guaranteed ceiling.

---

## 🔬 Empirical Forward-Model Characterization & Signal Analysis

The degradation data was empirically characterized across all 3,200 pairs using [characterize_data.py](characterize_data.py):

### 1. Ground-Truth Wavelet-MAD Noise Estimate
Noise standard deviation is estimated via the Median Absolute Deviation (MAD) of the finest-scale diagonal wavelet subband ($HH$) using Daubechies wavelets ($\text{db2}$):
$$\sigma = \frac{\text{median}(|HH|)}{0.6745}, \quad \text{PSNR}_{\text{ceiling}} = 10 \log_{10}\left(\frac{1.0}{\sigma^2}\right)$$

* **Full Dataset (3,200 GT Images)**: Mean $\sigma = 0.016649 \pm 0.026946$, equivalent mean PSNR **`39.31 dB`** (Median `39.93 dB`, Range `[10.21, 56.73] dB).
* **Validation Subset (320 GT Images)**: Mean $\sigma = 0.016798$, equivalent PSNR **`38.72 dB`**.
* Wavelet high-frequency energy can include real semiconductor structure, so these values must not be interpreted as strict performance bounds.

### 2. Forward-Model Downsampling & Optical Blur Sweep
Candidate downsampling operators were compared on 200 paired samples:

* **Downsampling Operator Comparison**:
  * **2×2 Area Averaging**: $\text{MSE} = 0.006438$
  * **Bicubic Downsampling**: $\text{MSE} = \mathbf{0.006392}$ (**lowest tested mean residual**)
  * **Nearest Neighbor**: $\text{MSE} = 0.008428$ (+30.9% error)
  * **Strided Subsampling**: $\text{MSE} = 0.008428$ (+30.9% error)
  * **Paired area-vs-bicubic test**: $t=6.901$, $p=6.74\times10^{-11}$

* **Optical Blur Sweep (Gaussian $\sigma$ vs Residual MSE)**:
  $$\text{Blur } \sigma = 0.0: \text{MSE} = \mathbf{0.006438} \quad (\text{Exact Minimum Residual})$$
  $$\text{Blur } \sigma = 0.5: \text{MSE} = 0.006616, \quad \text{Blur } \sigma = 1.0: \text{MSE} = 0.007357, \quad \text{Blur } \sigma = 1.5: \text{MSE} = 0.008162$$

> 📌 **Finding**: The lowest residual in the tested Gaussian grid occurs at $\sigma=0.0$. This supports selecting no added blur in the working forward model, but does not prove the physical optical blur is exactly zero.

### 3. Multiplicative-Additive Noise Parameters & Arcsinh VST
An exploratory local-patch regression fits $\text{Var}(y | \mu) = a \mu^2 + b$:
* **Multiplicative Parameter ($a$)**: **$3.346 \times 10^{-2}$**
* **Additive Parameter ($b$)**: **$1.781 \times 10^{-2}$**
* **Variance-Stabilizing Transform (Arcsinh VST)**:
  $$f(y) = \frac{1}{\sqrt{a}} \operatorname{arcsinh}\left(y \sqrt{\frac{a}{b}}\right), \quad y = \sqrt{\frac{b}{a}} \sinh(f \sqrt{a})$$
* **FP32 Numerical Safety**: Implemented in FP32 with input bounds ($[-15, 15]$) to eliminate $\sinh$ overflow and MS-SSIM underflow under mixed-precision AMP. Reversion error: **$2.384 \times 10^{-7}$**.
* The regression can conflate image texture with sensor noise. VST is an analysis utility and is not used by the shipped training or inference path.

---

## 🌐 Seeded Synthetic Noise Robustness Benchmark

The checkpoint was tested with seeded additive Gaussian noise applied to area-downsampled images from the project's own validation distribution. This measures synthetic noise robustness, not cross-dataset generalization or a calibrated physical scanner model:

| Noise Regime | Noise Std ($\sigma$) | Test condition | Restored PSNR (dB) | Restored SSIM |
| :--- | :---: | :--- | :---: | :---: |
| **Low noise** | $\sigma = 0.01$ | Seeded synthetic Gaussian noise | **`28.96 dB`** | **`0.7925`** |
| **Standard noise** | $\sigma = 0.03$ | Seeded synthetic Gaussian noise | **`28.13 dB`** | **`0.7698`** |
| **High noise** | $\sigma = 0.05$ | Seeded synthetic Gaussian noise | **`27.30 dB`** | **`0.7451`** |
| **Extreme noise** | $\sigma = 0.08$ | Seeded synthetic Gaussian noise | **`25.53 dB`** | **`0.6586`** |

---

## 📊 Design Decisions and Evidence Status

The current model uses a bicubic residual skip, zero-initialized SR head, EMA weights, and a Charbonnier/Sobel/FFT/two-scale SSIM loss. These choices are visible and reproducible in code. Historical leave-one-out numbers are not presented as verified ablations because the repository does not contain the corresponding checkpoints, logs, or experiment configurations. The optional NoiseGate and VST utilities are experimental and are **not** represented in `weights/best_model.pt`.

---

## 🔬 Core Architecture & Pipeline

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
                         [PixelShuffle 2x Head] (Zero-Init)
                                      |
                                      v
                                  [Sum (+)] <-------+
                                      |
                     [Experimental NoiseGate: not trained]
                                      |
                    +-------------------------------------------------+
                    | Restored Metrology Image (Clean, Bx1x2Hx2W)     |
                    +-------------------------------------------------+
```

### Composite Metrology Loss Function
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{Charbonnier}} + 0.05 \cdot \mathcal{L}_{\text{Sobel}} + 0.05 \cdot \mathcal{L}_{\text{FFT}}^{\text{ortho}} + 0.20 \cdot \mathcal{L}_{\text{MS-SSIM}}$$
* **Charbonnier Loss**: Robust pixel outlier recovery.
* **Calibrated Sobel Edge Loss**: Sub-10nm feature perimeter preservation.
* **Ortho-Normalized 2D FFT Loss**: Frequency domain suppression of periodic speckle noise.
* **Multi-Scale SSIM (FP32)**: Structural symmetry enforcement across nano- and macro-scale wafer patterns.

The composite loss is forced to FP32 under AMP. During training the model output is left unclamped to preserve gradients; inference output is clamped to $[0,1]$.

---

## ⚡ Environment Setup

### 1. Clone the Repository
```bash
git clone -b Kunal https://github.com/kmbeddedd/semicon_2026.git
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

# Install requirements (includes PyTorch, PyWavelets, Scipy, LPIPS, OpenCV)
pip install -r requirements.txt

# Optional: reproduce the direct dependency versions used for the verified benchmark
pip install -r requirements-verified.txt
```

---

## 🔍 Run Metrology Signal Characterization

Audit candidate degradation models, compute the Wavelet-MAD estimate, run a paired kernel test, and measure clean-input damage:

```bash
# Run forward-model characterization across dataset
python characterize_data.py --gt_dir data/train/GT --lr_dir data/train/NoisyLR --weights weights/best_model.pt --max_pairs 200
```

---

## 🎯 How to Run Inference & Evaluation

The evaluation script `eval.py` is standalone and accepts any directory of input images or `.npy` files:

```bash
# Ultra-Fast Batched Production Inference (13.35 ms / frame, 74.9 FPS)
python eval.py --input_dir data/test/NoisyLR --output_dir data/output_restored --weights weights/best_model.pt --scale 2 --batch_size 8 --no_tta

# Single-Image Streaming Inference (15.17 ms / frame, 65.9 FPS)
python eval.py --input_dir data/test/NoisyLR --output_dir data/output_restored --weights weights/best_model.pt --scale 2 --batch_size 1 --no_tta

# Benchmark against Ground Truth with LPIPS, measured bicubic baseline, Wavelet-MAD estimate, and clean audit
python eval.py --input_dir data/val/NoisyLR --target_dir data/val/GT --output_dir data/val_restored --weights weights/best_model.pt --scale 2 --batch_size 8 --no_tta --check_clean_damage

# Optional 8-Fold Test-Time Augmentation (TTA) Ensemble Mode
python eval.py --input_dir data/val/NoisyLR --target_dir data/val/GT --output_dir data/val_restored --weights weights/best_model.pt --scale 2 --check_clean_damage
```

### CLI Arguments
* `--input_dir` / `-i`: Path to directory containing degraded input images (`.npy`, `.png`, `.jpg`, `.tif`).
* `--output_dir` / `-o`: Output folder to save restored `.npy` files and `.png` visual previews.
* `--target_dir` / `-t`: *(Optional)* Path to ground truth directory to compute PSNR, SSIM, LPIPS, a preprocessing-matched bicubic baseline, and the Wavelet-MAD estimate.
* `--weights` / `-w`: Path to model checkpoint file (default: `weights/best_model.pt`).
* `--scale`: Scale factor (`2` for $2\times$ super-resolution, `1` for same-resolution denoising).
* `--batch_size` / `-b`: Batch size for GPU parallel inference (default: `8`, use `1` for streaming).
* `--no_fp16`: Disable FP16 Tensor Core acceleration.
* `--no_jit`: Disable TorchScript JIT kernel fusion.
* `--no_tta`: Disable 8-fold test-time augmentation for ultra-low latency (74.9 FPS).
* `--check_clean_damage`: Audit model preservation fidelity on clean downsampled GT patterns.

---

## 🧪 Automated Tests & Synthetic Robustness Audit

```bash
# Run unit tests for signal utilities, data validation, model contracts,
# checkpoint loading, scheduling, TTA, and FP32 AMP loss behavior
python -m unittest discover tests

# Run the seeded same-distribution synthetic Gaussian-noise audit
python utils/generalization.py --weights weights/best_model.pt --num_samples 50 --seed 42
```

---

## 🏋️ Reproduce Training

### Option A: 1-Click Cloud Training on Google Colab (Single T4 GPU)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kmbeddedd/semicon_2026/blob/Kunal/train_colab.ipynb)

### Option B: Distributed Cloud Training on Kaggle (Dual Tesla T4 x2 GPUs)
Use the included [`train_kaggle.ipynb`](train_kaggle.ipynb) notebook on Kaggle with Accelerator set to **GPU T4 x2**:
* Uses **PyTorch `DataParallel`** to split mini-batches (`batch_size=64`, 32 per GPU).
* Uses Automatic Mixed Precision (`AMP FP16`); training time depends on accelerator, storage, and validation settings.

### Option C: Local Training
```bash
python train.py --train_input data/train/NoisyLR --train_target data/train/GT --val_input data/val/NoisyLR --val_target data/val/GT --epochs 100 --batch_size 16 --lr 5e-4 --scale 2
```

Training is seeded by default (`--seed 42`). Use `--deterministic` for deterministic kernels, `--no_cache` on memory-constrained systems, and `--resume weights/latest_model.pt` to restore model, EMA, optimizer, scheduler, and AMP scaler state. Checkpoint loading is strict and rejects incompatible architectures.

---

## 📂 Repository Structure

```
semicon_2026/
├── README.md                 # Reproduced benchmarks, usage, and evidence limitations
├── requirements.txt          # Supported direct dependency ranges
├── requirements-verified.txt # Direct versions used for the reproduced benchmark
├── characterize_data.py      # Candidate forward-model, paired-statistics & noise-estimate tool
├── eval.py                   # Inference, TTA, LPIPS, bicubic baseline & clean-input evaluation
├── train.py                  # End-to-end training pipeline with AMP, EMA & calibrated loss
├── train_colab.ipynb         # 1-Click Google Colab training notebook
├── models/
│   ├── __init__.py
│   └── nafnet.py             # NAFNet-SR with residual skip and experimental optional NoiseGate
├── utils/
│   ├── dataset.py            # Calibrated dataset loader & metrology augmentations
│   ├── signal_analysis.py    # Wavelet-MAD, Arcsinh VST, blur sweep & kernel tests
│   ├── metrics.py            # PSNR, SSIM, Wavelet noise estimate & normalized gain
│   ├── losses.py             # Composite Metrology Loss (FP32 FFT, SSIM, Charbonnier, Sobel)
│   └── generalization.py     # Seeded same-distribution synthetic noise benchmark
├── tests/
│   └── test_signal_processing.py # Signal, data, model, scheduler, checkpoint, TTA & AMP tests
├── data/
│   ├── train/                # Training paired dataset (NoisyLR, GT - 2880 pairs)
│   ├── val/                  # Validation paired dataset (NoisyLR, GT - 320 pairs)
│   ├── test/                 # Test degraded dataset (400 samples)
│   └── output_restored/      # Output restored predictions (.npy + .png)
└── weights/
    └── best_model.pt         # Top-performing checkpoint (Epoch 73 EMA)
```

### Reproducibility and storage notes

* The shipped checkpoint reproduces the table above, but a full retraining run has not yet been executed after the training-engine hardening changes.
* Dataset ZIPs use Git LFS, but the repository contains both full and partitioned archives (about 1.79 GiB when checked out). Removing that duplication or moving it to release/DVC storage requires a separate artifact migration.
* Exact historical ablation results require experiment-specific checkpoints and logs; they are intentionally not claimed by the current reproducible benchmark.

---

## 📚 References & Acknowledgments
1. Donoho & Johnstone, *"Ideal spatial adaptation by wavelet shrinkage"*, Biometrika 1994.
2. Chen et al., *"Simple Baselines for Image Restoration"*, ECCV 2022.
3. Zhang et al., *"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"*, CVPR 2018.
4. KLA Metrology Guidelines for High-Throughput E-Beam & Optical Wafer Inspection.
