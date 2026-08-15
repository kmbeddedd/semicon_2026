import torch
import numpy as np
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    return float(psnr_metric(target, pred, data_range=data_range))

def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute Structural Similarity Index (SSIM)."""
    return float(ssim_metric(target, pred, data_range=data_range))

def wavelet_noise_sigma(img: np.ndarray, wavelet: str = 'db2') -> float:
    """
    Robust per-image noise std via MAD of finest-scale diagonal wavelet coefficients (HH band).
    """
    if img.ndim == 3:
        img = img.squeeze()
    coeffs = pywt.dwt2(img.astype(np.float32), wavelet)
    _, (_, _, HH) = coeffs
    sigma = np.median(np.abs(HH)) / 0.6745
    return float(max(sigma, 1e-7))

def psnr_ceiling(sigma: float, peak: float = 1.0) -> float:
    """
    Theoretical maximum achievable PSNR (dB) bounded by Ground Truth noise floor:
    PSNR_ceiling = 10 * log10(peak^2 / sigma^2)
    """
    mse_floor = max(sigma ** 2, 1e-14)
    return float(10.0 * np.log10((peak ** 2) / mse_floor))

def relative_ceiling_efficiency(psnr: float, ceiling: float) -> float:
    """Computes percentage of theoretical noise floor ceiling achieved."""
    if ceiling <= 0:
        return 0.0
    return float((psnr / ceiling) * 100.0)

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
