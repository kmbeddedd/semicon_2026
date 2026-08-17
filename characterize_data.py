import os
import argparse
import glob
import time
import numpy as np
import cv2
import torch
from scipy.stats import ttest_rel
from utils.signal_analysis import (
    wavelet_noise_sigma,
    psnr_ceiling,
    compute_dataset_noise_ceiling,
    estimate_noise_parameters_ab,
    sweep_blur_hypothesis,
    test_downsample_kernels,
    vst_forward_np,
    vst_inverse_np,
    relative_ceiling_efficiency
)
from utils.metrics import compute_psnr, compute_ssim
from models.nafnet import NAFNetSR, resolve_nafnet_config


def load_model_state_strict(model: torch.nn.Module, state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Semiconductor Inspection Dataset Signal Characterization & Degradation Audit")
    parser.add_argument("--gt_dir", type=str, default="data/train/GT", help="Path to GT images directory")
    parser.add_argument("--lr_dir", type=str, default="data/train/NoisyLR", help="Path to NoisyLR images directory")
    parser.add_argument("--val_gt_dir", type=str, default="data/val/GT", help="Path to Val GT images directory")
    parser.add_argument("--val_lr_dir", type=str, default="data/val/NoisyLR", help="Path to Val NoisyLR images directory")
    parser.add_argument("--weights", type=str, default="weights/best_model.pt", help="Path to model weights to test clean-input damage")
    parser.add_argument("--max_pairs", type=int, default=500, help="Max pairs to use for slow sweeps (0 for all)")
    return parser.parse_args()

def load_img(path: str) -> np.ndarray:
    if path.endswith('.npy'):
        img = np.load(path).astype(np.float32)
        if img.ndim == 3:
            img = img.squeeze()
        return np.clip(img, 0.0, 1.0)
    else:
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return (raw.astype(np.float32) / 255.0).clip(0.0, 1.0)

def main():
    args = parse_args()
    print("=" * 80)
    print("      SEMICONDUCTOR METROLOGY & SIGNAL DEGRADATION CHARACTERIZATION")
    print("=" * 80)
    
    # 1. Collect all GT and LR files
    all_gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.*")))
    val_gt_files = []
    if os.path.exists(args.val_gt_dir):
        val_gt_files = sorted(glob.glob(os.path.join(args.val_gt_dir, "*.*")))
        all_gt_files += val_gt_files

    if not all_gt_files:
        raise FileNotFoundError("No ground-truth images were found in the configured train/validation directories.")
        
    print(f"\n[1] WAVELET-MAD GROUND-TRUTH NOISE ESTIMATE")
    print(f"    Loaded {len(all_gt_files)} Ground-Truth metrology images.")
    
    gt_sigmas, gt_ceilings = [], []
    for p in all_gt_files:
        img = load_img(p)
        s = wavelet_noise_sigma(img)
        gt_sigmas.append(s)
        gt_ceilings.append(psnr_ceiling(s))
        
    mean_gt_s = np.mean(gt_sigmas)
    mean_gt_ceil = np.mean(gt_ceilings)
    median_gt_ceil = np.median(gt_ceilings)
    min_gt_ceil = np.min(gt_ceilings)
    max_gt_ceil = np.max(gt_ceilings)
    
    print(f"    - Wavelet Noise Std (Sigma):  {mean_gt_s:.6f} +/- {np.std(gt_sigmas):.6f}")
    print(f"    - Wavelet-MAD PSNR Estimate:  {mean_gt_ceil:.2f} dB (Median: {median_gt_ceil:.2f} dB, Range: [{min_gt_ceil:.2f}, {max_gt_ceil:.2f}] dB)")
    print("    - Interpretation: this is a high-frequency noise-floor heuristic, not a formal model-to-GT upper bound.")

    # Measure a preprocessing-matched bicubic reference on validation pairs.
    baseline_psnrs = []
    val_ceilings = []
    for g_path in val_gt_files:
        l_path = os.path.join(args.val_lr_dir, os.path.basename(g_path))
        if not os.path.exists(l_path):
            continue
        gt_img = load_img(g_path)
        lr_img = load_img(l_path)
        h, w = gt_img.shape
        bicubic = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC).clip(0.0, 1.0)
        baseline_psnrs.append(compute_psnr(bicubic, gt_img))
        val_ceilings.append(psnr_ceiling(wavelet_noise_sigma(gt_img)))

    weights_file = args.weights if os.path.exists(args.weights) else "best_model.pt"
    checkpoint_metric = None
    if os.path.exists(weights_file):
        checkpoint_metadata = torch.load(weights_file, map_location="cpu")
        if isinstance(checkpoint_metadata, dict) and "val_psnr" in checkpoint_metadata:
            checkpoint_metric = float(checkpoint_metadata["val_psnr"])

    if baseline_psnrs:
        mean_baseline = float(np.mean(baseline_psnrs))
        print(f"    - Measured Val Bicubic PSNR:  {mean_baseline:.2f} dB ({len(baseline_psnrs)} matched pairs)")
        if checkpoint_metric is not None:
            val_ceiling = float(np.mean(val_ceilings))
            efficiency = relative_ceiling_efficiency(checkpoint_metric, val_ceiling, mean_baseline)
            print(f"    - Checkpoint Gain Estimate:   {efficiency:.1f}% using checkpoint PSNR={checkpoint_metric:.2f} dB and Val estimate={val_ceiling:.2f} dB")

    # 2. Pairwise Degradation Analysis (Blur & Downsampling Tests)
    print(f"\n[2] EMPIRICAL FORWARD-MODEL & DEGRADATION HYPOTHESIS TESTING")
    gt_lr_pairs = []
    
    # Check train & val pairs
    search_dirs = [(args.gt_dir, args.lr_dir)]
    if os.path.exists(args.val_gt_dir) and os.path.exists(args.val_lr_dir):
        search_dirs.append((args.val_gt_dir, args.val_lr_dir))
        
    for g_dir, l_dir in search_dirs:
        for g_path in sorted(glob.glob(os.path.join(g_dir, "*.*"))):
            fname = os.path.basename(g_path)
            l_path = os.path.join(l_dir, fname)
            if os.path.exists(l_path):
                gt_lr_pairs.append((g_path, l_path))
                
    num_to_test = len(gt_lr_pairs) if args.max_pairs == 0 else min(len(gt_lr_pairs), args.max_pairs)
    print(f"    Testing degradation operators on {num_to_test} empirical (GT, NoisyLR) image pairs...")

    if num_to_test == 0:
        raise FileNotFoundError("No matching GT/NoisyLR pairs were found for degradation analysis.")
    
    sample_pairs = []
    for g_path, l_path in gt_lr_pairs[:num_to_test]:
        sample_pairs.append((load_img(g_path), load_img(l_path)))
        
    # 2A. Downsample kernel candidate test
    kernel_results = test_downsample_kernels(sample_pairs, candidates=('area', 'nearest', 'bicubic', 'strided'))
    best_kernel = min(kernel_results, key=kernel_results.get)

    area_errors, bicubic_errors = [], []
    for gt_img, lr_img in sample_pairs:
        h_lr, w_lr = lr_img.shape
        area_pred = cv2.resize(gt_img, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
        bicubic_pred = cv2.resize(gt_img, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
        area_errors.append(float(np.mean((area_pred - lr_img) ** 2)))
        bicubic_errors.append(float(np.mean((bicubic_pred - lr_img) ** 2)))
    paired_test = ttest_rel(area_errors, bicubic_errors)

    print(f"\n    [A] Downsampling Operator Comparison (Residual MSE vs NoisyLR):")
    for kernel_name, mse in kernel_results.items():
        marker = " <--- LOWEST MEAN RESIDUAL" if kernel_name == best_kernel else ""
        print(f"        * {kernel_name.capitalize():<11}: MSE = {mse:.6f}{marker}")
    print(f"        * Paired area-vs-bicubic test: t={paired_test.statistic:.3f}, p={paired_test.pvalue:.3e}")
    print("        * This comparison identifies the best empirical fit among candidates; it does not by itself prove the sensor's physical operator.")

    # 2B. Blur operator sweep test (against 2x2 Area Downsampling forward model)
    blur_results = sweep_blur_hypothesis(sample_pairs, sigmas=np.arange(0.0, 1.6, 0.1))
    print(f"\n    [B] Optical Blur Sweep (Gaussian Sigma vs Residual MSE with 2x2 Area Downsampling):")
    best_blur_sigma = min(blur_results, key=blur_results.get)
    for sigma, mse in sorted(blur_results.items()):
        marker = " <--- MINIMUM MEAN RESIDUAL" if sigma == best_blur_sigma else ""
        print(f"        * Blur Sigma {sigma:.1f}: MSE = {mse:.6f}{marker}")

    print(f"    => FINDING: The tested Gaussian blur candidate with the lowest mean residual is sigma={best_blur_sigma:.1f} (MSE={blur_results[best_blur_sigma]:.6f}).")
    print("       Treat this as a model-selection result over the tested grid, not proof that the physical optical blur is exactly zero.")

    # 3. Variance-Stabilizing Transform & Noise Parameter Modeling
    print(f"\n[3] EXPLORATORY LOCAL-PATCH VARIANCE MODEL (Var = a*mu^2 + b)")
    a_list, b_list = [], []
    for gt_img, lr_img in sample_pairs[:100]:
        a, b = estimate_noise_parameters_ab(lr_img, patch_size=8)
        a_list.append(a)
        b_list.append(b)
    print(f"    - Mean Multiplicative Parameter (a): {np.mean(a_list):.6e}")
    print(f"    - Mean Additive Parameter (b):       {np.mean(b_list):.6e}")
    print("    - Caveat: local image texture can be conflated with sensor noise in this regression.")
    
    # Test VST round-trip consistency
    sample_img = sample_pairs[0][1]
    a_est, b_est = a_list[0], b_list[0]
    vst_fwd = vst_forward_np(sample_img, a_est, b_est)
    vst_rec = vst_inverse_np(vst_fwd, a_est, b_est)
    vst_err = np.max(np.abs(sample_img - vst_rec))
    print(f"    - Arcsinh VST Round-Trip Max Reversion Error: {vst_err:.6e} (FP32 validated)")

    # 4. Model Clean-Input Degradation Audit
    if os.path.exists(weights_file):
        print(f"\n[4] CLEAN-INPUT DEGRADATION AUDIT (Does unconditional denoising damage clean inputs?)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights_file, map_location=device)
        model = NAFNetSR(**resolve_nafnet_config(ckpt, scale_factor=2)).to(device)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        load_model_state_strict(model, sd)
        model.eval()
        
        # Test on clean GT images (downsampled 2x or direct)
        clean_psnrs = []
        clean_ssims = []
        with torch.no_grad():
            for gt_img, _ in sample_pairs[:50]:
                # Prepare 128x128 clean downsampled input
                h, w = gt_img.shape
                clean_lr = cv2.resize(gt_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                inp_t = torch.from_numpy(clean_lr).unsqueeze(0).unsqueeze(0).float().to(device)
                pred_t = model(inp_t)
                pred_np = pred_t.squeeze().cpu().numpy().clip(0.0, 1.0)
                
                clean_psnrs.append(compute_psnr(pred_np, gt_img))
                clean_ssims.append(compute_ssim(pred_np, gt_img))
                
        clean_bicubic_psnrs = []
        for gt_img, _ in sample_pairs[:50]:
            h, w = gt_img.shape
            clean_lr = cv2.resize(gt_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            clean_bicubic = cv2.resize(clean_lr, (w, h), interpolation=cv2.INTER_CUBIC).clip(0.0, 1.0)
            clean_bicubic_psnrs.append(compute_psnr(clean_bicubic, gt_img))

        clean_model_mean = float(np.mean(clean_psnrs))
        clean_baseline_mean = float(np.mean(clean_bicubic_psnrs))
        print(f"    - Model Restored PSNR on Clean Inputs:   {clean_model_mean:.2f} dB (SSIM: {np.mean(clean_ssims):.4f})")
        print(f"    - Bicubic PSNR on Clean Inputs:          {clean_baseline_mean:.2f} dB")
        print(f"    - Model Delta vs Bicubic:                {clean_model_mean - clean_baseline_mean:+.2f} dB")
    else:
        print(f"\n[4] Checkpoint '{weights_file}' not found; skipping model audit.")

    print("\n" + "=" * 80)
    print("      METROLOGY SIGNAL CHARACTERIZATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
