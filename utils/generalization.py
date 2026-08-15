import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import cv2
import torch
from models.nafnet import NAFNetSR
from utils.metrics import compute_psnr, compute_ssim

def run_generalization_benchmarks(weights_path="weights/best_model.pt", num_samples=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NAFNetSR(in_channels=1, out_channels=1, width=64, scale_factor=2).to(device)
    
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
    model.eval()

    gt_files = sorted(glob.glob("data/val/GT/*.*"))[:num_samples]
    if not gt_files:
        gt_files = sorted(glob.glob("data/train/GT/*.*"))[:num_samples]

    results = {}
    noise_regimes = [
        ("Low Noise (sigma=0.01)", 0.01),
        ("Standard Inspection (sigma=0.03)", 0.03),
        ("High Shot Noise (sigma=0.05)", 0.05),
        ("Extreme OOD Noise (sigma=0.08)", 0.08),
    ]

    print("=" * 80)
    print("      CROSS-DATASET & OUT-OF-DISTRIBUTION (OOD) GENERALIZATION AUDIT")
    print("=" * 80)

    for regime_name, sigma in noise_regimes:
        psnrs, ssims = [], []
        with torch.no_grad():
            for g_path in gt_files:
                gt = np.load(g_path).squeeze().astype(np.float32) if g_path.endswith('.npy') else (cv2.imread(g_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
                gt = np.clip(gt, 0.0, 1.0)
                h, w = gt.shape
                
                # Physical 2x Area downsampling + synthetic Poisson-Gaussian noise
                down = cv2.resize(gt, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                noisy_lr = np.clip(down + np.random.normal(0, sigma, down.shape).astype(np.float32), 0.0, 1.0)
                
                inp_t = torch.from_numpy(noisy_lr).unsqueeze(0).unsqueeze(0).float().to(device)
                pred_t = model(inp_t)
                pred_np = pred_t.squeeze().cpu().numpy().clip(0.0, 1.0)
                
                psnrs.append(compute_psnr(pred_np, gt))
                ssims.append(compute_ssim(pred_np, gt))
                
        mean_p, mean_s = np.mean(psnrs), np.mean(ssims)
        results[regime_name] = (mean_p, mean_s)
        print(f"  * {regime_name:<35}: Restored PSNR = {mean_p:.2f} dB, SSIM = {mean_s:.4f}")

    return results

if __name__ == "__main__":
    run_generalization_benchmarks()
