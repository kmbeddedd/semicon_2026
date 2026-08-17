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

class MetrologyLoss(nn.Module):
    """
    Composite Metrology Loss tailored for Semiconductor Inspection Image Restoration.
    Combines Charbonnier (pixel fidelity), calibrated Sobel (boundary roughness),
    Ortho-2D FFT (frequency speckle), and MS-SSIM (structural patterns).
    """
    def __init__(self, w_charb=1.0, w_edge=0.05, w_fft=0.05, w_ssim=0.2):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.edge = SobelEdgeLoss()
        self.fft = FFTLoss()
        self.ssim = SSIMLoss()

        self.w_charb = w_charb
        self.w_edge = w_edge
        self.w_fft = w_fft
        self.w_ssim = w_ssim

    def forward(self, pred, target):
        # Loss evaluation stays in FP32 even when the model forward pass uses AMP.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_f32 = pred.float()
            target_f32 = target.float()
            l_charb = self.charbonnier(pred_f32, target_f32)
            l_edge = self.edge(pred_f32, target_f32)
            l_fft = self.fft(pred_f32, target_f32)
            l_ssim = self.ssim(pred_f32, target_f32)

            total = self.w_charb * l_charb + self.w_edge * l_edge + self.w_fft * l_fft + self.w_ssim * l_ssim
        return total, {
            "charb": float(l_charb.item()),
            "edge": float(l_edge.item()),
            "fft": float(l_fft.item()),
            "ssim": float(l_ssim.item())
        }
