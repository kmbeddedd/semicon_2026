import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    return float(psnr_metric(target, pred, data_range=data_range))

def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute Structural Similarity Index (SSIM)."""
    return float(ssim_metric(target, pred, data_range=data_range))

def evaluate_metrics(pred_tensor: torch.Tensor, target_tensor: torch.Tensor):
    """
    Compute average PSNR and SSIM across a batch of single-channel grayscale image tensors [B, 1, H, W] in [0, 1].
    """
    pred_np = pred_tensor.detach().cpu().clamp(0, 1).numpy()
    target_np = target_tensor.detach().cpu().clamp(0, 1).numpy()

    psnrs, ssims = [], []
    batch_size = pred_np.shape[0]

    for i in range(batch_size):
        p = pred_np[i, 0]
        t = target_np[i, 0]
        psnrs.append(compute_psnr(p, t))
        ssims.append(compute_ssim(p, t))

    return float(np.mean(psnrs)), float(np.mean(ssims))
