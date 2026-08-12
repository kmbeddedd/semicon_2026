import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant smooth near zero for robust noise handling)."""
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
        # Repeat kernel across input channels if needed
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
    """2D Fast Fourier Transform Loss to eliminate periodic speckle noise in spatial domain."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='backward')
        target_fft = torch.fft.rfft2(target, norm='backward')
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return F.l1_loss(pred_mag, target_mag)

class MetrologyLoss(nn.Module):
    """
    Composite Metrology Loss tailored for Semiconductor Inspection Image Restoration.
    Combines Charbonnier (robust pixel), Sobel (edge/boundary), and 2D FFT (speckle frequency).
    """
    def __init__(self, w_charb=1.0, w_edge=0.5, w_fft=0.1):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.edge = SobelEdgeLoss()
        self.fft = FFTLoss()

        self.w_charb = w_charb
        self.w_edge = w_edge
        self.w_fft = w_fft

    def forward(self, pred, target):
        l_charb = self.charbonnier(pred, target)
        l_edge = self.edge(pred, target)
        l_fft = self.fft(pred, target)
        
        total = self.w_charb * l_charb + self.w_edge * l_edge + self.w_fft * l_fft
        return total, {"charb": l_charb.item(), "edge": l_edge.item(), "fft": l_fft.item()}
