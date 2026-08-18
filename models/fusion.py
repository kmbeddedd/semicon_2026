"""Identity-safe local/global fusion for semiconductor restoration.

The global branch is constructed from the official MambaIRv2 repository instead
of carrying a modified copy in this project. This keeps provenance and licensing
clear and makes upgrades explicit through the checkpoint configuration.
"""

import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nafnet import NAFNetSR, resolve_nafnet_config


MAMBAIRV2_LIGHT_X2_CONFIG = {
    "img_size": 64,
    "patch_size": 1,
    "in_chans": 3,
    "embed_dim": 48,
    "d_state": 8,
    "depths": (5, 5, 5, 5),
    "num_heads": (4, 4, 4, 4),
    "window_size": 16,
    "inner_rank": 32,
    "num_tokens": 64,
    "convffn_kernel_size": 5,
    "mlp_ratio": 1.0,
    "upscale": 2,
    "img_range": 1.0,
    "upsampler": "pixelshuffledirect",
    "resi_connection": "1conv",
}

MAMBAIRV2_BASE_X2_CONFIG = {
    "img_size": 64,
    "patch_size": 1,
    "in_chans": 3,
    "embed_dim": 174,
    "d_state": 16,
    "depths": (6, 6, 6, 6, 6, 6),
    "num_heads": (6, 6, 6, 6, 6, 6),
    "window_size": 16,
    "inner_rank": 64,
    "num_tokens": 128,
    "convffn_kernel_size": 5,
    "mlp_ratio": 2.0,
    "upscale": 2,
    "img_range": 1.0,
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
}

MAMBAIRV2_X2_CONFIGS = {
    "light": MAMBAIRV2_LIGHT_X2_CONFIG,
    "base": MAMBAIRV2_BASE_X2_CONFIG,
}


def _strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def extract_model_state(checkpoint):
    """Extract common raw, BasicSR, and project checkpoint state formats."""
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("params_ema", "params", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return _strip_module_prefix(value)
    return _strip_module_prefix(checkpoint)


def build_official_mambairv2(
    repo_path: str,
    variant: str = "base",
    config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Build an official MambaIRv2 x2 variant from the authors' repository."""
    if variant not in MAMBAIRV2_X2_CONFIGS:
        raise ValueError(f"MambaIRv2 variant must be one of {sorted(MAMBAIRV2_X2_CONFIGS)}, got '{variant}'")
    repo_path = os.path.abspath(os.path.expanduser(repo_path)) if repo_path else ""
    architecture_module = "mambairv2light_arch" if variant == "light" else "mambairv2_arch"
    architecture_file = os.path.join(repo_path, "basicsr", "archs", f"{architecture_module}.py")
    if not repo_path or not os.path.isfile(architecture_file):
        raise FileNotFoundError(
            "MambaIRv2 checkout not found. Clone https://github.com/csguoh/MambaIR "
            "and pass --mambair_repo /path/to/MambaIR."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    try:
        if variant == "light":
            from basicsr.archs.mambairv2light_arch import MambaIRv2Light
            model_class = MambaIRv2Light
        else:
            from basicsr.archs.mambairv2_arch import MambaIRv2
            model_class = MambaIRv2
    except Exception as exc:
        raise RuntimeError(
            f"Could not import official MambaIRv2-{variant}. Install the MambaIR "
            "requirements, including a causal_conv1d/mamba_ssm build compatible "
            "with the active PyTorch and CUDA versions."
        ) from exc

    resolved_config = {**MAMBAIRV2_X2_CONFIGS[variant], **(config or {})}
    return model_class(**resolved_config)


def build_official_mambairv2_light(
    repo_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Backward-compatible helper for existing Light experiment callers."""
    return build_official_mambairv2(repo_path, variant="light", config=config)


class GrayscaleMambaIRv2(nn.Module):
    """Adapt the official three-channel SR model to metrology grayscale data."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, inp):
        rgb_input = inp.repeat(1, 3, 1, 1)
        rgb_output = self.backbone(rgb_input)
        return rgb_output.mean(dim=1, keepdim=True)


class GlobalLocalFusionSR(nn.Module):
    """Spatially fuse NAFNet detail with a global restoration branch.

    The final gate projection is all-zero at construction, so the output is
    bit-identical to ``local_model`` at initialization. The local uncertainty
    map is supplied to the gate as a difficulty cue when available.
    """

    def __init__(
        self,
        local_model: nn.Module,
        global_model: nn.Module,
        fusion_hidden: int = 24,
        use_uncertainty: bool = True,
        freeze_local: bool = False,
        freeze_global: bool = True,
    ):
        super().__init__()
        if fusion_hidden < 4:
            raise ValueError("fusion_hidden must be at least 4")
        self.local_model = local_model
        self.global_model = global_model
        self.use_uncertainty = bool(use_uncertainty)
        self.freeze_local = bool(freeze_local)
        self.freeze_global = bool(freeze_global)
        gate_channels = 4 if self.use_uncertainty else 3
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(gate_channels, fusion_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fusion_hidden, fusion_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fusion_hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.zeros_(self.fusion_gate[-1].bias)
        self.set_branch_trainability(freeze_local=freeze_local, freeze_global=freeze_global)

    def set_branch_trainability(self, *, freeze_local: bool, freeze_global: bool):
        self.freeze_local = bool(freeze_local)
        self.freeze_global = bool(freeze_global)
        for parameter in self.local_model.parameters():
            parameter.requires_grad = not self.freeze_local
        for parameter in self.global_model.parameters():
            parameter.requires_grad = not self.freeze_global

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen pretrained branches must also keep deterministic inference
        # behavior (DropPath/dropout disabled) while the fusion gate trains.
        if self.freeze_local:
            self.local_model.eval()
        if self.freeze_global:
            self.global_model.eval()
        return self

    @staticmethod
    def _run_branch(module, inp, frozen, **kwargs):
        if frozen:
            with torch.no_grad():
                return module(inp, **kwargs) if kwargs else module(inp)
        return module(inp, **kwargs) if kwargs else module(inp)

    def forward(self, inp, noise_stats=None, return_uncertainty=False):
        del noise_stats
        if self.use_uncertainty:
            local_output, raw_variance = self._run_branch(
                self.local_model,
                inp,
                self.freeze_local,
                return_uncertainty=True,
            )
        else:
            local_output = self._run_branch(self.local_model, inp, self.freeze_local)
            raw_variance = None

        global_output = self._run_branch(self.global_model, inp, self.freeze_global)
        if global_output.shape[-2:] != local_output.shape[-2:]:
            global_output = F.interpolate(
                global_output,
                size=local_output.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )

        gate_inputs = [local_output, global_output, torch.abs(global_output - local_output)]
        if raw_variance is not None:
            # Bounded, monotonic variance cue; raw values are otherwise uncalibrated.
            gate_inputs.append(torch.sigmoid(raw_variance))
        gate = torch.tanh(self.fusion_gate(torch.cat(gate_inputs, dim=1)))
        fused = local_output + gate * (global_output - local_output)
        fused = fused if self.training else fused.clamp(0.0, 1.0)
        if return_uncertainty:
            if raw_variance is None:
                raise ValueError("return_uncertainty=True requires a local uncertainty head")
            return fused, raw_variance
        return fused


def build_model_from_checkpoint(checkpoint, scale_factor=2, mambair_repo=""):
    """Build either the legacy NAFNet or a self-described fusion checkpoint."""
    model_type = checkpoint.get("model_type", "nafnet") if isinstance(checkpoint, dict) else "nafnet"
    if model_type == "nafnet":
        config = resolve_nafnet_config(checkpoint, scale_factor=scale_factor)
        return NAFNetSR(**config), config, model_type
    if model_type != "global_local_fusion":
        raise ValueError(f"Unsupported checkpoint model_type: {model_type}")

    config = checkpoint.get("model_config", {})
    required = {"local_config", "global_backend", "global_config", "fusion_hidden"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Fusion checkpoint is missing config keys: {sorted(missing)}")
    supported_backends = {"mambairv2_light": "light", "mambairv2_base": "base"}
    if config["global_backend"] not in supported_backends:
        raise ValueError(f"Unsupported global backend: {config['global_backend']}")

    local_config = resolve_nafnet_config(
        {"model_config": config["local_config"]},
        scale_factor=scale_factor,
    )
    global_backbone = build_official_mambairv2(
        mambair_repo,
        variant=supported_backends[config["global_backend"]],
        config=config["global_config"],
    )
    model = GlobalLocalFusionSR(
        NAFNetSR(**local_config),
        GrayscaleMambaIRv2(global_backbone),
        fusion_hidden=int(config["fusion_hidden"]),
        use_uncertainty=bool(config.get("use_uncertainty", local_config["predict_uncertainty"])),
        freeze_local=bool(config.get("freeze_local", False)),
        freeze_global=bool(config.get("freeze_global", True)),
    )
    return model, config, model_type
