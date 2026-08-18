import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant smooth near zero for robust outlier & noise handling)."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred.float() - target.float()
        return torch.mean(torch.sqrt(diff ** 2 + self.eps2))

class SobelEdgeLoss(nn.Module):
    """
    Sobel edge loss for preserving sub-10nm feature boundaries and Line-Edge Roughness.
    Operates with epsilon protection to prevent zero-gradient singularities.
    """
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred, target):
        pred_f32 = pred.float()
        target_f32 = target.float()
        c = pred.size(1)
        kx = self.kernel_x.repeat(c, 1, 1, 1).to(pred.device, torch.float32)
        ky = self.kernel_y.repeat(c, 1, 1, 1).to(pred.device, torch.float32)

        pred_gx = F.conv2d(pred_f32, kx, padding=1, groups=c)
        pred_gy = F.conv2d(pred_f32, ky, padding=1, groups=c)
        target_gx = F.conv2d(target_f32, kx, padding=1, groups=c)
        target_gy = F.conv2d(target_f32, ky, padding=1, groups=c)

        pred_mag = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-6)
        target_mag = torch.sqrt(target_gx**2 + target_gy**2 + 1e-6)

        return F.l1_loss(pred_mag, target_mag)

class FFTLoss(nn.Module):
    """
    2D Fast Fourier Transform Spectral Loss with Ortho-Normalization.
    Enforces FP32 computation to avoid FP16 underflow/overflow artifacts under AMP.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Force FP32 for numerical stability in complex FFT operations
        pred_f32 = pred.float()
        target_f32 = target.float()

        pred_fft = torch.fft.rfft2(pred_f32, norm='ortho')
        target_fft = torch.fft.rfft2(target_f32, norm='ortho')
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return F.l1_loss(pred_mag, target_mag)

class SSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity Loss with FP32 precision guarantees.
    Prevents variance underflow in FP16 mixed precision training.
    """
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def _ssim_single_scale(self, pred, target, ws):
        # Compute in FP32
        p_f32 = pred.float()
        t_f32 = target.float()

        C1, C2 = 0.01**2, 0.03**2
        pad = ws // 2
        mu_p = F.avg_pool2d(p_f32, ws, stride=1, padding=pad)
        mu_t = F.avg_pool2d(t_f32, ws, stride=1, padding=pad)
        sigma_pp = F.avg_pool2d(p_f32 * p_f32, ws, stride=1, padding=pad) - mu_p * mu_p
        sigma_tt = F.avg_pool2d(t_f32 * t_f32, ws, stride=1, padding=pad) - mu_t * mu_t
        sigma_pt = F.avg_pool2d(p_f32 * t_f32, ws, stride=1, padding=pad) - mu_p * mu_t

        # Epsilon clamp for stability
        sigma_pp = torch.clamp(sigma_pp, min=0.0)
        sigma_tt = torch.clamp(sigma_tt, min=0.0)

        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (sigma_pp + sigma_tt + C2))
        return 1.0 - ssim.mean()

    def forward(self, pred, target):
        loss_scale1 = self._ssim_single_scale(pred, target, self.window_size)
        p_down = F.avg_pool2d(pred, 2)
        t_down = F.avg_pool2d(target, 2)
        loss_scale2 = self._ssim_single_scale(p_down, t_down, max(3, self.window_size // 2 * 2 + 1))
        return 0.6 * loss_scale1 + 0.4 * loss_scale2


class BetaGaussianNLLLoss(nn.Module):
    """Variance-stabilized heteroscedastic Gaussian NLL.

    The detached variance weighting follows the beta-NLL idea described in the
    supplied research note and limits the incentive to inflate uncertainty.
    """
    def __init__(self, beta=0.5, eps=1e-6):
        super().__init__()
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.beta = beta
        self.eps = eps

    def forward(self, mean, target, raw_variance):
        mean_f32 = mean.float()
        target_f32 = target.float()
        variance = F.softplus(raw_variance.float()) + self.eps
        nll = 0.5 * (torch.log(variance) + (target_f32 - mean_f32).square() / variance)
        if self.beta > 0:
            nll = nll * variance.detach().pow(self.beta)
        return nll.mean()

class MetrologyLoss(nn.Module):
    """
    Composite Metrology Loss tailored for Semiconductor Inspection Image Restoration.
    Combines Charbonnier (pixel fidelity), calibrated Sobel (boundary roughness),
    Ortho-2D FFT (frequency speckle), and MS-SSIM (structural patterns).
    """
    def __init__(
        self,
        w_mse=0.0,
        w_charb=1.0,
        w_edge=0.05,
        w_fft=0.05,
        w_ssim=0.2,
        w_nll=0.0,
        nll_beta=0.5,
    ):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.edge = SobelEdgeLoss()
        self.fft = FFTLoss()
        self.ssim = SSIMLoss()
        self.nll = BetaGaussianNLLLoss(beta=nll_beta)

        self.set_weights(
            mse=w_mse,
            charb=w_charb,
            edge=w_edge,
            fft=w_fft,
            ssim=w_ssim,
            nll=w_nll,
        )

    def set_weights(self, *, mse, charb, edge, fft, ssim, nll):
        """Update loss weights without rebuilding the criterion or optimizer."""
        weights = {
            "mse": mse,
            "charb": charb,
            "edge": edge,
            "fft": fft,
            "ssim": ssim,
            "nll": nll,
        }
        if any(value < 0 for value in weights.values()):
            raise ValueError("Loss weights must be non-negative")
        if not any(value > 0 for value in weights.values()):
            raise ValueError("At least one loss weight must be positive")
        self.w_mse = float(mse)
        self.w_charb = float(charb)
        self.w_edge = float(edge)
        self.w_fft = float(fft)
        self.w_ssim = float(ssim)
        self.w_nll = float(nll)

    @property
    def weights(self):
        return {
            "mse": self.w_mse,
            "charb": self.w_charb,
            "edge": self.w_edge,
            "fft": self.w_fft,
            "ssim": self.w_ssim,
            "nll": self.w_nll,
        }

    def forward(self, pred, target, raw_variance=None):
        # Loss evaluation stays in FP32 even when the model forward pass uses AMP.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_f32 = pred.float()
            target_f32 = target.float()
            zero = pred_f32.new_zeros(())
            l_mse = F.mse_loss(pred_f32, target_f32) if self.w_mse > 0 else zero
            l_charb = self.charbonnier(pred_f32, target_f32) if self.w_charb > 0 else zero
            l_edge = self.edge(pred_f32, target_f32) if self.w_edge > 0 else zero
            l_fft = self.fft(pred_f32, target_f32) if self.w_fft > 0 else zero
            l_ssim = self.ssim(pred_f32, target_f32) if self.w_ssim > 0 else zero
            if self.w_nll > 0:
                if raw_variance is None:
                    raise ValueError("raw_variance is required when w_nll > 0")
                l_nll = self.nll(pred_f32, target_f32, raw_variance.float())
            else:
                l_nll = zero

            total = (
                self.w_mse * l_mse
                + self.w_charb * l_charb
                + self.w_edge * l_edge
                + self.w_fft * l_fft
                + self.w_ssim * l_ssim
                + self.w_nll * l_nll
            )
        return total, {
            "mse": float(l_mse.item()),
            "charb": float(l_charb.item()),
            "edge": float(l_edge.item()),
            "fft": float(l_fft.item()),
            "ssim": float(l_ssim.item()),
            "nll": float(l_nll.item()),
        }
