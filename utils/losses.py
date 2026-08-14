import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant smooth near zero for robust outlier & noise handling)."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

class SobelEdgeLoss(nn.Module):
    """Sobel edge loss for preserving sub-10nm feature boundaries and Line-Edge Roughness."""
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred, target):
        c = pred.size(1)
        kx = self.kernel_x.repeat(c, 1, 1, 1)
        ky = self.kernel_y.repeat(c, 1, 1, 1)

        pred_gx = F.conv2d(pred, kx, padding=1, groups=c)
        pred_gy = F.conv2d(pred, ky, padding=1, groups=c)
        target_gx = F.conv2d(target, kx, padding=1, groups=c)
        target_gy = F.conv2d(target, ky, padding=1, groups=c)

        pred_mag = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-6)
        target_mag = torch.sqrt(target_gx**2 + target_gy**2 + 1e-6)

        return F.l1_loss(pred_mag, target_mag)

class FFTLoss(nn.Module):
    """
    2D Fast Fourier Transform Spectral Loss with Ortho-Normalization.
    Eliminates periodic speckle noise in frequency domain while maintaining balanced gradient scaling.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # norm='ortho' scales by 1/sqrt(H*W) to prevent spectral magnitude explosion
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return F.l1_loss(pred_mag, target_mag)

class SSIMLoss(nn.Module):
    """Multi-Scale Structural Similarity Loss."""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def _ssim_single_scale(self, pred, target, ws):
        C1, C2 = 0.01**2, 0.03**2
        pad = ws // 2
        mu_p = F.avg_pool2d(pred, ws, stride=1, padding=pad)
        mu_t = F.avg_pool2d(target, ws, stride=1, padding=pad)
        sigma_pp = F.avg_pool2d(pred * pred, ws, stride=1, padding=pad) - mu_p * mu_p
        sigma_tt = F.avg_pool2d(target * target, ws, stride=1, padding=pad) - mu_t * mu_t
        sigma_pt = F.avg_pool2d(pred * target, ws, stride=1, padding=pad) - mu_p * mu_t
        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (sigma_pp + sigma_tt + C2))
        return 1.0 - ssim.mean()

    def forward(self, pred, target):
        loss_scale1 = self._ssim_single_scale(pred, target, self.window_size)
        # Downsample 2x for macro-structure similarity
        p_down = F.avg_pool2d(pred, 2)
        t_down = F.avg_pool2d(target, 2)
        loss_scale2 = self._ssim_single_scale(p_down, t_down, max(3, self.window_size // 2 * 2 + 1))
        return 0.6 * loss_scale1 + 0.4 * loss_scale2

class MetrologyLoss(nn.Module):
    """
    Composite Metrology Loss tailored for Semiconductor Inspection Image Restoration.
    Combines Charbonnier (robust pixel), Sobel (edge/boundary), Ortho-2D FFT (speckle frequency), and MS-SSIM (structural).
    """
    def __init__(self, w_charb=1.0, w_edge=0.15, w_fft=0.05, w_ssim=0.2):
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
        l_charb = self.charbonnier(pred, target)
        l_edge = self.edge(pred, target)
        l_fft = self.fft(pred, target)
        l_ssim = self.ssim(pred, target)

        total = self.w_charb * l_charb + self.w_edge * l_edge + self.w_fft * l_fft + self.w_ssim * l_ssim
        return total, {
            "charb": l_charb.item(),
            "edge": l_edge.item(),
            "fft": l_fft.item(),
            "ssim": l_ssim.item()
        }
