import numpy as np
import pywt
import cv2
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Union, Optional

def wavelet_noise_sigma(img: np.ndarray, wavelet: str = 'db2') -> float:
    """
    Robust per-image noise standard deviation estimation via Median Absolute Deviation (MAD)
    of the finest-scale diagonal wavelet coefficients (HH subband).
    
    Standard Donoho & Johnstone estimator: sigma = median(|HH|) / 0.6745
    """
    if img.ndim == 3:
        img = img.squeeze()
    img_f = img.astype(np.float32)
    
    # 2D Discrete Wavelet Transform
    coeffs = pywt.dwt2(img_f, wavelet)
    _, (_, _, HH) = coeffs
    
    median_abs = np.median(np.abs(HH))
    sigma = float(median_abs / 0.6745)
    return max(sigma, 1e-7)

def psnr_ceiling(sigma: float, peak: float = 1.0) -> float:
    """
    Theoretical maximum achievable PSNR (dB) bounded by the ground-truth noise floor:
    PSNR_ceiling = 10 * log10(peak^2 / sigma^2) = 20 * log10(peak / sigma)
    """
    mse_floor = max(sigma ** 2, 1e-14)
    return float(10.0 * np.log10((peak ** 2) / mse_floor))

def compute_dataset_noise_ceiling(gt_images: List[np.ndarray]) -> Dict[str, float]:
    """
    Computes noise floor sigma and theoretical PSNR ceiling across a collection of GT images.
    """
    sigmas = [wavelet_noise_sigma(img) for img in gt_images]
    ceilings = [psnr_ceiling(s) for s in sigmas]
    
    return {
        "mean_sigma": float(np.mean(sigmas)),
        "std_sigma": float(np.std(sigmas)),
        "mean_ceiling_db": float(np.mean(ceilings)),
        "median_ceiling_db": float(np.median(ceilings)),
        "min_ceiling_db": float(np.min(ceilings)),
        "max_ceiling_db": float(np.max(ceilings)),
        "count": len(gt_images)
    }

def estimate_noise_parameters_ab(img: np.ndarray, patch_size: int = 8) -> Tuple[float, float]:
    """
    Estimates multiplicative (a) and additive (b) noise parameters from local patches:
    Var(y | mu) = a * mu^2 + b
    
    Uses least-squares linear regression on patch mean^2 vs patch variance.
    """
    if img.ndim == 3:
        img = img.squeeze()
    h, w = img.shape
    means, vars_ = [], []
    
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            means.append(float(patch.mean()))
            vars_.append(float(patch.var()))
            
    means = np.array(means, dtype=np.float64)
    vars_ = np.array(vars_, dtype=np.float64)
    
    # Var = a * mean^2 + b -> regression on [mean^2, 1]
    A = np.vstack([means ** 2, np.ones_like(means)]).T
    try:
        sol, _, _, _ = np.linalg.lstsq(A, vars_, rcond=None)
        a, b = float(sol[0]), float(sol[1])
    except Exception:
        a, b = 1e-4, 1e-4
        
    return max(a, 1e-6), max(b, 1e-6)

def vst_forward_np(y: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Variance-Stabilizing Transform (Arcsinh VST) for multiplicative-additive noise:
    f(y) = (1 / sqrt(a)) * arcsinh(y * sqrt(a / b))
    
    Transforms signal-dependent variance to approximately constant unit variance across brightness levels.
    Enforces FP32 computation.
    """
    y_f32 = y.astype(np.float32)
    scale_in = np.sqrt(a / b).astype(np.float32)
    inv_scale_out = (1.0 / np.sqrt(a)).astype(np.float32)
    return inv_scale_out * np.arcsinh(y_f32 * scale_in)

def vst_inverse_np(f: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Inverse Variance-Stabilizing Transform:
    y = sqrt(b / a) * sinh(f * sqrt(a))
    
    Includes safety clamping on argument to prevent numerical overflow in FP32.
    """
    f_f32 = f.astype(np.float32)
    scale_arg = np.sqrt(a).astype(np.float32)
    scale_out = np.sqrt(b / a).astype(np.float32)
    
    # Clamp argument to [-15.0, 15.0] to prevent sinh overflow (sinh(15) ~ 1.6e6)
    arg = np.clip(f_f32 * scale_arg, -15.0, 15.0)
    y = scale_out * np.sinh(arg)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def vst_forward_torch(y: torch.Tensor, a: float = 1e-3, b: float = 1e-3) -> torch.Tensor:
    """
    PyTorch FP32 implementation of Arcsinh VST.
    """
    y_fp32 = y.float()
    scale_in = np.sqrt(a / b)
    inv_scale_out = 1.0 / np.sqrt(a)
    return inv_scale_out * torch.asinh(y_fp32 * scale_in)

def vst_inverse_torch(f: torch.Tensor, a: float = 1e-3, b: float = 1e-3) -> torch.Tensor:
    """
    PyTorch FP32 implementation of Inverse Arcsinh VST with overflow safeguards.
    """
    f_fp32 = f.float()
    scale_arg = np.sqrt(a)
    scale_out = np.sqrt(b / a)
    arg = torch.clamp(f_fp32 * scale_arg, -15.0, 15.0)
    return torch.clamp(scale_out * torch.sinh(arg), 0.0, 1.0)

def sweep_blur_hypothesis(gt_lr_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                          sigmas: Optional[np.ndarray] = None) -> Dict[float, float]:
    """
    Sweeps blur standard deviation sigma on GT images, downsamples via 2x2 area averaging,
    and computes the residual MSE against actual LR inspection images.
    
    If minimum MSE occurs at sigma = 0.0, optical blur operator is absent.
    """
    if sigmas is None:
        sigmas = np.arange(0.0, 2.1, 0.1)
        
    results = {}
    for sigma in sigmas:
        errs = []
        for gt_img, lr_img in gt_lr_pairs:
            gt_f = gt_img.astype(np.float32)
            lr_f = lr_img.astype(np.float32)
            
            blurred = gaussian_filter(gt_f, sigma) if sigma > 0 else gt_f
            h_lr, w_lr = lr_f.shape[:2]
            down = cv2.resize(blurred, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
            
            errs.append(float(np.mean((down - lr_f) ** 2)))
        results[float(round(sigma, 2))] = float(np.mean(errs))
        
    return results

def test_downsample_kernels(gt_lr_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                            candidates: Tuple[str, ...] = ('area', 'nearest', 'bicubic', 'strided')) -> Dict[str, float]:
    """
    Evaluates candidate spatial downsampling operators to identify which best models
    the empirical downsampling transformation from GT to LR.
    """
    interp_map = {
        'area': cv2.INTER_AREA,
        'nearest': cv2.INTER_NEAREST,
        'bicubic': cv2.INTER_CUBIC,
        'linear': cv2.INTER_LINEAR
    }
    
    results = {}
    for name in candidates:
        errs = []
        for gt_img, lr_img in gt_lr_pairs:
            gt_f = gt_img.astype(np.float32)
            lr_f = lr_img.astype(np.float32)
            h_lr, w_lr = lr_f.shape[:2]
            
            if name == 'strided':
                pred = gt_f[::2, ::2]
                if pred.shape != (h_lr, w_lr):
                    pred = pred[:h_lr, :w_lr]
            else:
                pred = cv2.resize(gt_f, (w_lr, h_lr), interpolation=interp_map[name])
                
            errs.append(float(np.mean((pred - lr_f) ** 2)))
        results[name] = float(np.mean(errs))
        
    return results

class NoiseGate(nn.Module):
    """
    Learned Dynamic Noise Gate to prevent clean input damage.
    Given feature representations and input noise level descriptors (e.g. sigma, mu),
    learns an adaptive gating weight in [0, 1] to softly bypass or apply denoising.
    """
    def __init__(self, in_features: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Initialize bias towards 1.0 (denoising active by default)
        nn.init.constant_(self.mlp[2].bias, 2.0)

    def forward(self, noisy_feat: torch.Tensor, base_feat: torch.Tensor, noise_stats: torch.Tensor) -> torch.Tensor:
        """
        noisy_feat: Restored / denoised feature tensor [B, C, H, W]
        base_feat: Identity / bicubic upsampled base feature tensor [B, C, H, W]
        noise_stats: Tensor of noise descriptors per image [B, in_features]
        """
        gate = self.mlp(noise_stats).view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        return gate * noisy_feat + (1.0 - gate) * base_feat
