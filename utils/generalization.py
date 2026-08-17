import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import glob
import numpy as np
import cv2
import torch
from models.nafnet import NAFNetSR
from utils.metrics import compute_psnr, compute_ssim


def _load_model_state_strict(model, state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def run_synthetic_noise_benchmarks(weights_path="weights/best_model.pt", num_samples=50, seed=42):
    """Measure seeded synthetic Gaussian-noise robustness on known validation content.

    This is not a cross-dataset generalization test: the clean source images come
    from the project's own train/validation distribution.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NAFNetSR(in_channels=1, out_channels=1, width=64, scale_factor=2).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    _load_model_state_strict(model, sd)
    model.eval()

    rng = np.random.default_rng(seed)

    gt_files = sorted(glob.glob("data/val/GT/*.*"))[:num_samples]
    if not gt_files:
        gt_files = sorted(glob.glob("data/train/GT/*.*"))[:num_samples]
    if not gt_files:
        raise FileNotFoundError("No validation or training GT images were found.")

    results = {}
    noise_regimes = [
        ("Low Noise (sigma=0.01)", 0.01),
        ("Standard Inspection (sigma=0.03)", 0.03),
        ("High Noise (sigma=0.05)", 0.05),
        ("Extreme Noise (sigma=0.08)", 0.08),
    ]

    print("=" * 80)
    print("      SEEDED SYNTHETIC GAUSSIAN-NOISE ROBUSTNESS AUDIT")
    print("=" * 80)

    for regime_name, sigma in noise_regimes:
        psnrs, ssims = [], []
        with torch.no_grad():
            for g_path in gt_files:
                gt = np.load(g_path).squeeze().astype(np.float32) if g_path.endswith('.npy') else (cv2.imread(g_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
                gt = np.clip(gt, 0.0, 1.0)
                h, w = gt.shape
                
                # Working 2x area downsampling + seeded synthetic Gaussian noise.
                down = cv2.resize(gt, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                noisy_lr = np.clip(down + rng.normal(0, sigma, down.shape).astype(np.float32), 0.0, 1.0)
                
                inp_t = torch.from_numpy(noisy_lr).unsqueeze(0).unsqueeze(0).float().to(device)
                pred_t = model(inp_t)
                pred_np = pred_t.squeeze().cpu().numpy().clip(0.0, 1.0)
                
                psnrs.append(compute_psnr(pred_np, gt))
                ssims.append(compute_ssim(pred_np, gt))
                
        mean_p, mean_s = np.mean(psnrs), np.mean(ssims)
        results[regime_name] = (mean_p, mean_s)
        print(f"  * {regime_name:<35}: Restored PSNR = {mean_p:.2f} dB, SSIM = {mean_s:.4f}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Seeded synthetic noise robustness benchmark")
    parser.add_argument("--weights", default="weights/best_model.pt")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_synthetic_noise_benchmarks(cli_args.weights, cli_args.num_samples, cli_args.seed)
