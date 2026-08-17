# Historical Implementation Plan: Model Accuracy Improvement

> This document records an earlier design plan. The current behavior, reproduced metrics, and evidence limitations are documented in `README.md`; this file is not the active specification.

Upgrade the entire Semiconductor Metrology Restoration pipeline to eliminate preprocessing distortions, restore loss gradient balance, integrate global bicubic residual learning, add Exponential Moving Average (EMA), enable 8-fold Test-Time Augmentation (TTA), and retrain the model.

## User Review Required

> [!IMPORTANT]
> **GPU Training on Local Machine**: Detected **NVIDIA GeForce RTX 2050 (4GB VRAM)** with CUDA & Tensor Core acceleration.
> Training will be configured with Automatic Mixed Precision (AMP FP16), `batch_size=16`, and full $128\times128$ resolution to ensure fast execution (~15-20 seconds/epoch) without Out-Of-Memory (OOM) risks.

---

## Key Proposed Changes

### 1. Preprocessing & Augmentation Pipeline
#### [MODIFY] [`utils/dataset.py`](utils/dataset.py)
- **Fix `robust_percentile_normalize`**: Replace destructive division by noisy spike percentiles ($P_{99.99}$) with physical $[0.0, 1.0]$ range clamping to eliminate the ~18.5% photometric dimming artifact.
- **Enhanced Augmentations**: Add random horizontal/vertical flips, $90^\circ$ rotations, CutBlur / patch blending, and synthetic Poisson-Gaussian noise jittering.
- **Default Resolution**: Default to full $128\times128$ input resolution without aggressive $64\times64$ cropping to preserve line-space pitch context.

---

### 2. Model Architecture
#### [MODIFY] [`models/nafnet.py`](models/nafnet.py)
- **Global Bicubic Residual Skip Connection**: In `NAFNetSR.forward()`, add `F.interpolate(inp, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)` to the network output before clamping. This allows the network to learn pure high-frequency residual correction ($I_{\text{clean}} - I_{\text{bicubic}}$) instead of synthesizing the base structure from scratch.

---

### 3. Loss Functions & Gradient Balancing
#### [MODIFY] [`utils/losses.py`](utils/losses.py)
- **Ortho-Normalized FFT Loss**: Change `norm='backward'` to `norm='ortho'` in `torch.fft.rfft2`, bringing FFT loss scale from $\sim 7.20$ down to $\sim 0.028$, restoring gradient equilibrium with Charbonnier and SSIM.
- **Multi-Scale Structural Similarity (MS-SSIM)**: Enhance SSIM loss with multi-scale pooling to enforce boundary fidelity across nanoscale line edges and die-level macro structures.
- **Calibrated Metrology Weights**: Set `w_charb=1.0, w_edge=0.15, w_fft=0.05, w_ssim=0.2`.

---

### 4. Training Engine & Optimization
#### [MODIFY] [`train.py`](train.py)
- **Model EMA (Exponential Moving Average)**: Maintain an EMA model shadow (`decay=0.999`) during training to smooth weight updates and maximize validation PSNR.
- **Gradient Clipping**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to protect FP16 mixed precision from unstable gradient bursts.
- **Cosine Annealing with Warmup**: Implement learning rate warmup over initial epochs followed by smooth Cosine Annealing decay.
- **Checkpointing**: Save both regular epoch checkpoints and top-performing EMA checkpoints (`best_model.pt`).

---

### 5. Inference & Evaluation Pipeline
#### [MODIFY] [`eval.py`](eval.py)
- **8-Fold Test-Time Augmentation (TTA)**: Implement dihedral ensemble inference (4 rotations $\times$ 2 flips) to squeeze an extra $+0.3$ to $+0.6$ dB PSNR gain during evaluation on test data.
- **Batch Processing & Timing**: Support efficient batching and automatic NPY/PNG saving.

#### [MODIFY] [`train_colab.ipynb`](train_colab.ipynb)
- Update training invocation arguments to match newly calibrated parameters.

---

## Verification Plan

### Automated Tests
1. **Unit Verification**:
   - Test data loading shapes, normalization values, and augmentation integrity.
   - Verify `NAFNetSR` forward pass with bicubic skip connection.
   - Verify `MetrologyLoss` component balance (Charbonnier, Edge, FFT-ortho, SSIM).
   - Test TTA inference function on dummy tensors.
2. **Model Retraining**:
   - Run `train.py` on NVIDIA GeForce RTX 2050 GPU for 100+ epochs.
   - Track epoch-by-epoch loss, validation PSNR, and validation SSIM.
3. **Evaluation Benchmark**:
   - Run `eval.py` with `--use_tta` on the validation/test split.
   - Measure final PSNR/SSIM improvements over the baseline.
