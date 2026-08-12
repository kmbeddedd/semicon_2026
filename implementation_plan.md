# Implementation Plan: AI-Based Restoration of Degraded Images for Semiconductor Inspection (KLA Challenge)

## Overview & Domain Context

In advanced semiconductor nodes (N3/N2 FinFETs, GAAFETs, High-NA EUV lithography), high-throughput inline wafer inspection (CD-SEM, E-beam Inspection, DUV/EUV optical metrology) faces a fundamental trade-off: **Scan Speed vs. Signal-to-Noise Ratio (SNR)**. 
- Fast e-beam scans reduce electron dwell time to achieve high Wafers-Per-Hour (WPH), but low electron counts produce heavy **Poisson-Gaussian shot noise (speckle)** and pixel intensity saturation beyond physical bounds.
- Optical point-spread function (PSF) limitations and spatial pixel binning degrade high-frequency structural resolution, obscuring sub-10nm line-edge roughness (LER), contact hole shrinking, and bridge/break defects.

This plan details an end-to-end, competition-winning solution combining semiconductor metrology domain insights with a lightweight, high-speed **NAFNet (Nonlinear Activation Free Network)** restoration architecture optimized for ONNX inference (< 15ms/frame on GPU).

---

## Technical Strategy & Architecture Choice

### 1. Model Architecture: NAFNet-SR (Nonlinear Activation Free Super-Resolution & Denoising)
Instead of heavy Transformer/Diffusion models (which fail the hackathon's real-time latency requirement), we employ a **NAFNet** variant tailored for joint single-channel grayscale Denoising + Super-Resolution ($2\times$ upscaling and same-scale restoration).

```
                      +-------------------------------------------------+
                      | Input Degraded Image (Noisy, Low-Res: Bx1xHxW)  |
                      +-------------------------------------------------+
                                               |
                                    [3x3 Conv Stem Layer]
                                               |
                     +---------------------------------------------------+
                     |  Encoder Stage (3x NAF Blocks + Strided Downsample) |
                     +---------------------------------------------------+
                                               |
                     +---------------------------------------------------+
                     |    Bottleneck Stage (4x NAF Blocks + FFT Layer)   |
                     +---------------------------------------------------+
                                               |
                     +---------------------------------------------------+
                     |  Decoder Stage (3x NAF Blocks + Skip Connections) |
                     +---------------------------------------------------+
                                               |
                                     [PixelShuffle Upsampler]
                                               |
                      +-------------------------------------------------+
                      | Output Restored Image (Clean, High-Res: Bx1x2Hx2W)|
                      +-------------------------------------------------+
```

#### Why NAFNet for Semiconductor Metrology?
* **Zero Expensive Non-Linearities**: Replaces GELU/Softmax with element-wise Gated Mechanisms ($x_1 \odot x_2$) and Simple Channel Attention (SCA), drastically accelerating FLOPs/W on GPU/TensorRT.
* **Preserves Spatial Frequencies**: High-frequency silicon features (dielectric line edges, contact via perimeters) are maintained without ringing or artificial hallucination.
* **Unified Scale & Denoising**: Handles simultaneous $2\times$ upscaling ($128\to256$, $256\to512$) and joint speckle/Gaussian noise removal in a single forward pass.

---

## Preprocessing & Metrology-Aware Loss Function

### 1. Robust Dynamic Range Handling (Speckle Suppression)
Speckle noise pushes raw pixel values beyond standard dynamic range limits.
* **Pre-scaling**: Apply robust percentile intensity normalization $[P_{0.01}, P_{99.99}]$ per image to clip extreme out-of-range speckle spikes while mapping the valid signal to $[0, 1]$.

### 2. Multi-Objective Metrology Loss ($L_{\text{total}}$)
Standard L1/MSE losses cause blurred edges on nanoscale features. We design a composite loss:

$$L_{\text{total}} = \lambda_1 L_{\text{Charbonnier}} + \lambda_2 L_{\text{Edge}} + \lambda_3 L_{\text{FFT}} + \lambda_4 L_{\text{MS-SSIM}}$$

1. **Charbonnier Loss ($L_{\text{Charbonnier}}$)**: Robust $L_1$ variant $\sqrt{\|I_{\text{pred}} - I_{\text{gt}}\|^2 + \epsilon^2}$ ($\epsilon=10^{-3}$) resilient against heavy noise spikes.
2. **Sobel Gradient Edge Loss ($L_{\text{Edge}}$)**: Measures $X$/$Y$ spatial gradient discrepancies to enforce crisp feature edges (critical for Line-Edge Roughness and overlay metrology).
3. **2D Fourier Frequency Loss ($L_{\text{FFT}}$)**: L1 distance in 2D FFT spectral magnitude domain $|\mathcal{F}(I_{\text{pred}}) - \mathcal{F}(I_{\text{gt}})|$, specifically eliminating periodic grain and high-frequency noise.
4. **Multi-Scale SSIM ($L_{\text{MS-SSIM}}$)**: Preserves structural symmetry of repetitive memory arrays (DRAM/SRAM grids).

---

## Out-of-Distribution (OOD) Generalization & Data Augmentation

To ensure top performance on unseen wafer layers (e.g. transitioning from STI pattern training to Metal-1 / Via inspection test sets):
1. **Synthetic Noise Injection**: Randomly overlay Poisson-Gaussian noise, synthetic multiplicative speckle, and Gaussian PSF blur on training samples during augmentation.
2. **Spatial Transformations**: Random $90^\circ, 180^\circ, 270^\circ$ rotations, horizontal/vertical flips, and elastic spatial deformations.
3. **MixUp & CutMix for Images**: Linear combination of image pairs to enforce smooth decision boundaries across different fab patterns.

---

## Deliverables & Submission Structure

### Component 1: Slide Deck Mapping (Slides 1–9 PDF)
* **Slide 1: Team Details**: Team Name, Members, Affiliation, Contact Details.
* **Slide 2: Problem Statement & Fab Significance**: Explaining WPH throughput vs. E-beam SNR trade-off, why single-pixel defect precision matters in yield engineering.
* **Slide 3: Idea Description**: NAFNet architecture overview, low-latency design rationale, multi-degradation unified handling.
* **Slide 4: Proposed Solution & Pipeline**: Detailed block diagram (Stem, Encoder, Bottleneck, Decoder, PixelShuffle Head), composite loss function formulation.
* **Slide 5: Innovation & Uniqueness**: Metrology-tailored FFT+Sobel Gradient Loss, percentile speckle clipping, ONNX FP16 acceleration.
* **Slide 6: Empirical Results & Visual Evidence**: Quantitative metrics table (PSNR, SSIM, LPIPS) and side-by-side visual comparisons (Degraded $\to$ Restored $\to$ Ground Truth).
* **Slide 7: Technology & Feasibility**: Tech stack (PyTorch, ONNX Runtime), hardware profile (Tesla T4 / RTX 4090 benchmark), inference time (< 15ms per image), model parameters (~1.2M).
* **Slide 8: Code Repo & Demo Video**: Public GitHub URL & Walkthrough Video link.
* **Slide 9: Key References**: NAFNet (ECCV 2022), Semiconductor Metrology & SEM noise papers, PyTorch/ONNX references.

---

## Component 2: GitHub Repository Architecture

```
d:\Education\Project\SemiCon\
├── README.md                 # Clear setup, training, and standalone evaluation commands
├── requirements.txt          # Python dependencies (torch, torchvision, opencv-python, onnxruntime, etc.)
├── eval.py                   # Standalone CLI test script (input_dir -> output_dir)
├── train.py                  # End-to-end reproducible training script
├── models/
│   ├── __init__.py
│   └── nafnet.py             # Optimized NAFNet architecture implementation
├── utils/
│   ├── dataset.py            # Paired dataset loader with normalization & augmentations
│   ├── metrics.py            # PSNR, SSIM, LPIPS calculation utilities
│   └── losses.py             # Composite Metrology Loss (Charbonnier + Sobel + FFT + MS-SSIM)
└── weights/
    └── best_model.onnx       # Quantized/FP16 ONNX model weights for sub-15ms inference
```

---

## Verification & Execution Plan

### Automated Verification
1. **Sanity Check Unit Test**: Run `eval.py` on synthetic dummy images to verify output dimensions, dynamic range, and execution zero-error status.
2. **Speed Benchmark**: Benchmark batch inference time across CPU and GPU using `onnxruntime`.
3. **Metric Validation**: Compute PSNR (>32 dB benchmark target) and SSIM (>0.92 target) on validation splits.

### Manual Verification
1. Visual inspection of edge sharpness and speckle artifact removal across representative sample pairs.
2. Confirm slide deck and GitHub repo structure align 100% with submission guidelines.
