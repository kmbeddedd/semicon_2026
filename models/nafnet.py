import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """Channel-first 2D Layer Normalization."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

class SimpleGate(nn.Module):
    """Element-wise Gated Mechanism (replaces GELU/ReLU activations)."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimpleChannelAttention(nn.Module):
    """Lightweight Channel Attention without Softmax/Sigmoid bottlenecks."""
    def __init__(self, channels):
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return x * self.sca(x)

class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block (NAFBlock).
    Combines Depthwise Conv, Gated Linear Units, and Simple Channel Attention.
    """
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channels = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, stride=1, groups=dw_channels, bias=True)
        self.sg = SimpleGate()
        self.sca = SimpleChannelAttention(dw_channels // 2)
        self.conv3 = nn.Conv2d(dw_channels // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Feed Forward Network (FFN)
        ffn_channels = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channels // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # Spatial Block
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        # FFN Block
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma

class NAFNetSR(nn.Module):
    """
    NAFNet for Semiconductor Inspection Image Restoration & Super-Resolution.
    Handles joint Denoising (Speckle + Gaussian) and 2x Upscaling with Global Bicubic Residual Skip.
    """
    def __init__(self, in_channels=1, out_channels=1, width=64, enc_blocks=[2, 2, 2], dec_blocks=[2, 2, 2], scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        curr_width = width
        for num in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(curr_width) for _ in range(num)]))
            self.downs.append(nn.Conv2d(curr_width, curr_width * 2, kernel_size=2, stride=2))
            curr_width *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(curr_width) for _ in range(3)])

        for num in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(curr_width, curr_width * 2, kernel_size=1),
                nn.PixelShuffle(2)
            ))
            curr_width = curr_width // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(curr_width) for _ in range(num)]))

        # Final Head with Scale Upsampling (PixelShuffle) if scale_factor > 1
        if scale_factor > 1:
            self.head = nn.Sequential(
                nn.Conv2d(width, width * (scale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(width, out_channels, kernel_size=3, padding=1)
            )
        else:
            self.head = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

    def forward(self, inp):
        x = self.intro(inp)
        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        out = self.head(x)

        # Global residual connection: learn residual on top of bicubic upsampled base
        if self.scale_factor > 1:
            base = F.interpolate(inp, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
            out = out + base
        elif out.shape == inp.shape:
            out = out + inp

        return torch.clamp(out, 0.0, 1.0)
