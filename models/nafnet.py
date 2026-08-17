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
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

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
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN Block
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class DualDomainMixer(nn.Module):
    """Gated local/spectral feature mixer with an identity-safe output projection.

    This adapts the supplied research note's FFT/convolution dual-path idea from
    1D telemetry to 2D image features. The final projection is zero-initialized,
    so enabling the block preserves a transferred base model exactly at step 0.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = LayerNorm2d(channels)
        self.local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.spectral_mix = nn.Conv2d(2 * channels, 2 * channels, kernel_size=1, groups=channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        z = self.norm(x)
        local_features = self.local(z)

        # FFT kernels run in FP32. Explicit functional convolution keeps this
        # valid after eval.py converts the surrounding model to FP16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z_f32 = z.float()
            spectrum = torch.fft.rfft2(z_f32, norm="ortho")
            batch, channels, height, freq_width = spectrum.shape
            spectral_pairs = torch.stack((spectrum.real, spectrum.imag), dim=2)
            spectral_pairs = spectral_pairs.reshape(batch, 2 * channels, height, freq_width)
            mixed_pairs = F.conv2d(
                spectral_pairs,
                self.spectral_mix.weight.float(),
                self.spectral_mix.bias.float() if self.spectral_mix.bias is not None else None,
                groups=self.channels,
            )
            mixed_pairs = mixed_pairs.reshape(batch, channels, 2, height, freq_width)
            mixed_spectrum = torch.complex(mixed_pairs[:, :, 0], mixed_pairs[:, :, 1])
            spectral_features = torch.fft.irfft2(mixed_spectrum, s=z_f32.shape[-2:], norm="ortho")

        spectral_features = spectral_features.to(local_features.dtype)
        frequency_gate = self.gate(z)
        mixed_features = frequency_gate * spectral_features + (1.0 - frequency_gate) * local_features
        return x + self.project(mixed_features)

class NoiseGate(nn.Module):
    """
    Learned Dynamic Noise Gate to prevent clean input damage.
    Conditioned on estimated noise statistics (e.g. noise std sigma, patch variance, or (a, b) parameters),
    outputs an adaptive gate weight in [0, 1] to softly blend between the base bicubic input and restored output.
    """
    def __init__(self, in_features=2, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Default bias to 2.0 (gate ~ 0.88 towards denoising)
        nn.init.constant_(self.mlp[2].bias, 2.0)

    def forward(self, noisy_feat, base_feat, noise_stats):
        gate = self.mlp(noise_stats).view(-1, 1, 1, 1)
        return gate * noisy_feat + (1.0 - gate) * base_feat

class NAFNetSR(nn.Module):
    """
    NAFNet for Semiconductor Inspection Image Restoration & Super-Resolution.
    Handles joint Denoising (Speckle + Gaussian) and 2x Upscaling with Global Bicubic Residual Skip.
    Supports experimental optional NoiseGate conditioning and a zero-initialized residual head.
    The bundled checkpoint was trained without NoiseGate conditioning.
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        width=64,
        enc_blocks=(2, 2, 2),
        dec_blocks=(2, 2, 2),
        scale_factor=2,
        use_noise_gate=False,
        use_spectral_mixer=False,
        predict_uncertainty=False,
    ):
        super().__init__()
        if len(enc_blocks) != len(dec_blocks):
            raise ValueError("enc_blocks and dec_blocks must contain the same number of stages")
        if scale_factor < 1 or int(scale_factor) != scale_factor:
            raise ValueError("scale_factor must be a positive integer")
        scale_factor = int(scale_factor)
        self.scale_factor = scale_factor
        self.use_noise_gate = use_noise_gate
        self.use_spectral_mixer = use_spectral_mixer
        self.predict_uncertainty = predict_uncertainty
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
        self.spectral_mixer = DualDomainMixer(curr_width) if use_spectral_mixer else None

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
            # Zero-initialize the final convolution so the initial state is exact bicubic identity
            nn.init.zeros_(self.head[-1].weight)
            if self.head[-1].bias is not None:
                nn.init.zeros_(self.head[-1].bias)
        else:
            self.head = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

        if use_noise_gate:
            self.noise_gate = NoiseGate(in_features=2, hidden_dim=16)
        else:
            self.noise_gate = None

        if predict_uncertainty:
            if scale_factor > 1:
                self.uncertainty_head = nn.Sequential(
                    nn.Conv2d(width, width * (scale_factor ** 2), kernel_size=3, padding=1),
                    nn.PixelShuffle(scale_factor),
                    nn.Conv2d(width, out_channels, kernel_size=3, padding=1),
                )
            else:
                self.uncertainty_head = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)
            nn.init.zeros_(self.uncertainty_head[-1].weight if isinstance(self.uncertainty_head, nn.Sequential) else self.uncertainty_head.weight)
            uncertainty_bias = self.uncertainty_head[-1].bias if isinstance(self.uncertainty_head, nn.Sequential) else self.uncertainty_head.bias
            nn.init.constant_(uncertainty_bias, -6.0)
        else:
            self.uncertainty_head = None

    def forward(self, inp, noise_stats=None, return_uncertainty=False):
        # Extract primary intensity channel for bicubic base (cleanly JIT trace compatible)
        raw_inp = inp[:, :1, :, :]
        x = self.intro(inp)
        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)
        if self.spectral_mixer is not None:
            x = self.spectral_mixer(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        res = self.head(x)

        # Global residual connection: learn residual on top of bicubic upsampled base
        if self.scale_factor > 1:
            base = F.interpolate(raw_inp, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
            restored = base + res
        elif res.shape == raw_inp.shape:
            base = raw_inp
            restored = base + res
        else:
            base = raw_inp
            restored = res

        # Optional learned dynamic noise gating
        if self.noise_gate is not None and noise_stats is None:
            raise ValueError("noise_stats must be provided when use_noise_gate=True")
        if self.noise_gate is not None:
            out = self.noise_gate(restored, base, noise_stats)
        else:
            out = restored

        # Preserve gradients for out-of-range training predictions; inference is
        # clamped to the physical image range.
        out = out if self.training else torch.clamp(out, 0.0, 1.0)

        if return_uncertainty:
            if self.uncertainty_head is None:
                raise ValueError("return_uncertainty=True requires predict_uncertainty=True")
            raw_variance = self.uncertainty_head(x)
            return out, raw_variance
        return out


def resolve_nafnet_config(checkpoint=None, scale_factor=2):
    """Resolve a validated model config from legacy or self-describing checkpoints."""
    default_config = {
        "in_channels": 1,
        "out_channels": 1,
        "width": 64,
        "enc_blocks": (2, 2, 2),
        "dec_blocks": (2, 2, 2),
        "scale_factor": scale_factor,
        "use_noise_gate": False,
        "use_spectral_mixer": False,
        "predict_uncertainty": False,
    }
    checkpoint_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    unknown_keys = set(checkpoint_config) - set(default_config)
    if unknown_keys:
        raise ValueError(f"Checkpoint contains unsupported model config keys: {sorted(unknown_keys)}")
    return {**default_config, **checkpoint_config}
