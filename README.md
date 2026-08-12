# Semiconductor Image Restoration — NAFNet-SR

AI-based  restoration of degraded semiconductor inspection images using a lightweight NAFNet-based super-resolution and denoising model.

This repository contains the implementation, experiments, training configuration, validation setup, and results obtained while training the model on the provided semiconductor inspection dataset.

## 1. Project Overview

The task is to restore degraded low-resolution semiconductor inspection images.

The current dataset contains paired:

* **NoisyLR:** `128 × 128` degraded images
* **GT:** `256 × 256` ground-truth images

The model performs:

```text
128×128 NoisyLR
       ↓
  NAFNet-SR
       ↓
256×256 Restored Image
```

The model simultaneously performs denoising and 2× super-resolution.

## 2. Model

The implementation is based on a lightweight NAFNet-style architecture.

Main components:

* Encoder-decoder architecture
* NAFBlocks
* SimpleGate
* Simple Channel Attention
* Skip connections
* PixelShuffle 2× upsampling
* Single-channel grayscale input/output

Current model configuration:

```text
Input channels : 1
Output channels: 1
Width          : 32
Scale factor   : 2
Encoder blocks : [2, 2, 2]
Decoder blocks : [2, 2, 2]
Middle blocks  : 3
```

## 3. Loss Function

The current training objective is a composite loss:

```text
Total Loss =
    1.0 × Charbonnier Loss
  + 0.5 × Sobel Edge Loss
  + 0.1 × FFT Loss
```

### Charbonnier Loss

Provides robust pixel-level reconstruction.

### Sobel Edge Loss

Encourages preservation of image boundaries and high-frequency structures.

### FFT Loss

Compares frequency-domain magnitude information to help suppress periodic/high-frequency noise.

## 4. Dataset

The original dataset is not included in this repository because of its size.

The provided training archive contains:

```text
3,200 GT images
3,200 NoisyLR images
```

Each pair uses the same filename.

Example:

```text
train/
├── GT/
│   └── 000298.npy
└── NoisyLR/
    └── 000298.npy
```

The `.npy` data is stored as `float32`.

Typical dimensions:

```text
NoisyLR: 128 × 128
GT:     256 × 256
```

### Dataset acquisition

Obtain the dataset from the original project repository:

https://github.com/KJ-CORE/semicon_2026

The dataset files are managed through Git LFS in the original repository.

## 5. Validation Split

The original dataset did not contain a validation directory.

A deterministic 90/10 split was created:

```text
Training pairs   : 2,880
Validation pairs : 320
```

The original dataset under `train/` was preserved.

The validation split was created using hard links, so the files were not physically duplicated.

Directory structure:

```text
data/
├── train/
│   ├── NoisyLR/
│   └── GT/
└── val/
    ├── NoisyLR/
    └── GT/
```

## 6. Environment

Development environment:

```text
OS          : Windows + WSL2
GPU         : NVIDIA GeForce RTX 4060 Laptop GPU
GPU VRAM    : 8 GB
Python      : 3.12.3
```

CUDA access was verified from WSL using PyTorch.

Example verification:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected:

```text
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

## 7. Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the CUDA-enabled PyTorch environment separately:

```bash
pip install --upgrade pip
pip install torch torchvision
```

Install the remaining dependencies:

```bash
pip install numpy opencv-python Pillow onnxruntime scikit-image tqdm
```

## 8. Training

The training script accepts dataset paths through command-line arguments.

Current training command:

```bash
python train.py \
  --train_input data/train/NoisyLR \
  --train_target data/train/GT \
  --val_input data/val/NoisyLR \
  --val_target data/val/GT \
  --epochs 50 \
  --batch_size 8 \
  --scale 2
```

## 9. GPU Safety / VRAM Testing

Before full training, GPU compatibility was verified using controlled tests.

For batch size 1:

```text
Peak VRAM ≈ 0.315 GB
```

For batch size 8:

```text
Peak VRAM ≈ 2.437 GB
```

The RTX 4060 Laptop GPU has approximately 8 GB VRAM, leaving substantial headroom at batch size 8.

GPU temperature was also monitored during the experiment.

## 10. Baseline Training Results

The baseline model was trained for 50 epochs.

Results:

| Metric               |       Result |
| -------------------- | -----------: |
| Best validation PSNR | **25.43 dB** |
| Best PSNR epoch      |       **36** |
| Final PSNR           |     25.37 dB |
| Final SSIM           |       0.7474 |
| Final training loss  |       0.5840 |

The validation PSNR plateaued around epoch 36.

This suggests that simply continuing the same training configuration is unlikely to provide a large improvement.

## 11. Bicubic Baseline

A simple bicubic 2× interpolation baseline was evaluated on the same 320 validation samples.

Results:

| Method     |         PSNR |      SSIM |
| ---------- | -----------: | --------: |
| Bicubic 2× |     18.42 dB |    0.5487 |
| NAFNet-SR  | **25.43 dB** | **~0.75** |

The trained model therefore provides a substantial improvement over conventional bicubic interpolation.

## 12. Inference Performance

The trained model was evaluated on all 320 validation images.

Measured result:

```text
Average inference time: 10.75 ms/frame
```

Hardware:

```text
NVIDIA RTX 4060 Laptop GPU
```

This satisfies the project's stated target of:

```text
< 15 ms/frame
```

## 13. Current Checkpoint

The best model is stored as:

```text
weights/best_model.pt
```

The checkpoint corresponding to the best validation PSNR was obtained at epoch 36.

Large datasets and unnecessary training artifacts should not be committed to Git.

## 14. Current Findings

Visual inspection shows that the model:

* significantly reduces noise
* reconstructs major structures
* performs genuine super-resolution
* produces sharper results than simple interpolation

However, some fine structures remain smoother than the ground truth.

The current validation metrics are also substantially below the project's proposed target of approximately:

```text
PSNR > 32 dB
SSIM > 0.92
```

Therefore, further investigation is required.

## 15. Current Investigation

The first baseline used per-image percentile normalization:

```text
P0.01 → P99.99 → [0,1]
```
Analysis across the 320-image validation set showed a systematic intensity mismatch:

```text
Normalized NoisyLR mean : 0.313
GT mean                 : 0.413

Normalized NoisyLR std  : 0.155
GT std                  : 0.187
```

A controlled experiment replaced percentile normalization with simple clipping:

```text
NoisyLR → clip to [0,1]
```
```
The model architecture, loss, optimizer, batch size, dataset split, and training schedule were kept unchanged.

### Normalization Experiment
| Configuration | Best PSNR | Best SSIM | Best Epoch |
|---|---:|---:|---:|
| Percentile normalization | 25.43 dB | ~0.75 | 36 |
| Simple clipping | **27.93 dB** | **0.7700** | 47 |

The clipping approach improved validation PSNR by approximately **2.50 dB**.

This indicates that preserving the original NoisyLR intensity relationship with the ground truth is beneficial for this dataset.

The clipping experiment is now the current training baseline for further optimization.
```

## 16. Repository Structure

```text
semicon_2026/
├── README.md
├── requirements.txt
├── train.py
├── eval.py
├── implementation_plan.md
│
├── models/
│   ├── __init__.py
│   └── nafnet.py
│
├── utils/
│   ├── dataset.py
│   ├── losses.py
│   └── metrics.py
│
├── weights/
│   └── best_model.pt
│
└── data/
    ├── train/
    │   ├── NoisyLR/
    │   └── GT/
    └── val/
        ├── NoisyLR/
        └── GT/
```

The `data/` directory should not be committed to GitHub.

## 17. Experiment Roadmap

Planned experiments:

1. Test alternative input normalization.
2. Compare validation metrics against the current baseline.
3. Inspect visual reconstruction quality.
4. Evaluate different loss configurations.
5. Investigate learning-rate scheduling.
6. Compare model capacity.
7. Evaluate inference performance after improvements.
8. Select the best model based on both quantitative and visual quality.

The current 50-epoch NAFNet-SR experiment serves as the baseline for these experiments.
