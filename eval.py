import os
import argparse
import time
from glob import glob
import cv2
import numpy as np
import torch
from models.nafnet import NAFNetSR
from utils.dataset import robust_percentile_normalize
from utils.metrics import compute_psnr, compute_ssim, wavelet_noise_sigma, psnr_ceiling, relative_ceiling_efficiency

def parse_args():
    parser = argparse.ArgumentParser(description="High-Precision Evaluation & Inference Pipeline for Semiconductor Restoration")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Path to input degraded images directory")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Path to output directory for restored images")
    parser.add_argument("--target_dir", "-t", type=str, default=None, help="Optional Ground Truth directory to compute benchmark PSNR/SSIM and ceiling")
    parser.add_argument("--weights", "-w", type=str, default="weights/best_model.pt", help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (1 for denoising, 2 for 2x super-resolution)")
    parser.add_argument("--no_tta", action="store_true", help="Disable 8-Fold Test-Time Augmentation (TTA)")
    parser.add_argument("--check_clean_damage", action="store_true", help="Audit model performance on clean downsampled GT inputs")
    return parser.parse_args()

def predict_tta(model: torch.nn.Module, inp_tensor: torch.Tensor) -> torch.Tensor:
    """
    8-Fold Test-Time Augmentation (Dihedral Group D4 ensemble: 4 rotations x 2 flips).
    Averages predictions across all geometric symmetries for maximum PSNR/SSIM boost.
    """
    preds = []
    for k in range(4):
        # 1. Rotated input
        rot_inp = torch.rot90(inp_tensor, k, dims=[2, 3])
        pred_rot = model(rot_inp)
        # Invert rotation
        preds.append(torch.rot90(pred_rot, -k, dims=[2, 3]))

        # 2. Flipped & Rotated input
        flip_inp = torch.flip(rot_inp, dims=[3])
        pred_flip = model(flip_inp)
        # Invert flip & rotation
        inv_flip = torch.flip(pred_flip, dims=[3])
        preds.append(torch.rot90(inv_flip, -k, dims=[2, 3]))

    return torch.mean(torch.stack(preds, dim=0), dim=0)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Metrology Eval] Running inference on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Model
    model = NAFNetSR(in_channels=1, out_channels=1, width=64, scale_factor=args.scale)

    weights_path = args.weights
    if not os.path.exists(weights_path) and os.path.exists("best_model.pt"):
        weights_path = "best_model.pt"

    if os.path.exists(weights_path):
        print(f"[Metrology Eval] Loading checkpoint: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[Metrology Eval] WARNING: Weights file '{weights_path}' not found! Running with initialized model.")

    model.to(device)
    model.eval()

    use_tta = not args.no_tta
    print(f"[Metrology Eval] 8-Fold Test-Time Augmentation (TTA): {use_tta}")

    # Find test images
    valid_exts = ('*.npy', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
    img_paths = []
    for ext in valid_exts:
        img_paths.extend(glob(os.path.join(args.input_dir, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"[Metrology Eval] No images found in '{args.input_dir}'.")
        return

    print(f"[Metrology Eval] Processing {len(img_paths)} images...")
    total_time = 0.0
    psnr_scores, ssim_scores = [], []
    gt_ceilings, gt_sigmas = [], []

    with torch.no_grad():
        for path in img_paths:
            fname = os.path.basename(path)
            if path.endswith('.npy'):
                raw_img = np.load(path).astype(np.float32)
                if raw_img.ndim == 3:
                    raw_img = raw_img.squeeze()
            else:
                img_bgr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img_bgr is None:
                    continue
                raw_img = img_bgr.astype(np.float32) / 255.0

            # Calibrated range normalization
            norm_img = robust_percentile_normalize(raw_img)
            inp_t = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float().to(device)

            start_t = time.perf_counter()
            if use_tta:
                out_t = predict_tta(model, inp_t)
            else:
                out_t = model(inp_t)
            proc_t = (time.perf_counter() - start_t) * 1000.0
            total_time += proc_t

            out_np = out_t.squeeze().cpu().numpy()
            out_clamped = np.clip(out_np, 0.0, 1.0).astype(np.float32)

            # Compute metrics if Ground Truth exists
            if args.target_dir and os.path.exists(args.target_dir):
                gt_path = os.path.join(args.target_dir, fname)
                if os.path.exists(gt_path):
                    gt_raw = np.load(gt_path).astype(np.float32) if gt_path.endswith('.npy') else (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
                    if gt_raw.ndim == 3:
                        gt_raw = gt_raw.squeeze()
                    gt_clamped = np.clip(gt_raw, 0.0, 1.0)
                    
                    # Compute signal restoration metrics
                    psnr_scores.append(compute_psnr(out_clamped, gt_clamped))
                    ssim_scores.append(compute_ssim(out_clamped, gt_clamped))

                    # Compute physical GT noise floor ceiling (Wavelet-MAD)
                    s_gt = wavelet_noise_sigma(gt_clamped)
                    gt_sigmas.append(s_gt)
                    gt_ceilings.append(psnr_ceiling(s_gt))

            # Save restored outputs
            if fname.endswith('.npy'):
                save_npy_path = os.path.join(args.output_dir, fname)
                np.save(save_npy_path, out_clamped)
                png_name = fname.replace('.npy', '_restored.png')
                cv2.imwrite(os.path.join(args.output_dir, png_name), (out_clamped * 255.0).astype(np.uint8))
            else:
                save_path = os.path.join(args.output_dir, fname)
                cv2.imwrite(save_path, (out_clamped * 255.0).astype(np.uint8))

    avg_time = total_time / max(len(img_paths), 1)
    print(f"\n[Metrology Eval] Completed! Processed {len(img_paths)} images.")
    print(f"[Metrology Eval] Average Inference Latency: {avg_time:.2f} ms / frame.")
    print(f"[Metrology Eval] Restored outputs saved to: '{args.output_dir}'")

    if psnr_scores:
        mean_psnr = np.mean(psnr_scores)
        mean_ssim = np.mean(ssim_scores)
        mean_ceiling = np.mean(gt_ceilings)
        eff = relative_ceiling_efficiency(mean_psnr, mean_ceiling)
        margin = mean_ceiling - mean_psnr

        print(f"\n[Metrology Eval] Benchmark Metrics across {len(psnr_scores)} Ground Truth pairs:")
        print(f"  [+] Average Restored PSNR:    {mean_psnr:.2f} dB")
        print(f"  [+] Average Restored SSIM:    {mean_ssim:.4f}")
        print(f"  [+] Physical GT Noise Ceiling:{mean_ceiling:.2f} dB (Wavelet-MAD sigma: {np.mean(gt_sigmas):.6f})")
        print(f"  [+] Ceiling Efficiency:       {eff:.1f}% of theoretical maximum (-{margin:.2f} dB from noise ceiling)")

    if args.check_clean_damage and args.target_dir and os.path.exists(args.target_dir):
        print(f"\n[Metrology Eval] Running Clean-Input Degradation Audit...")
        clean_psnrs, clean_ssims = [], []
        with torch.no_grad():
            for path in img_paths[:50]:
                fname = os.path.basename(path)
                gt_path = os.path.join(args.target_dir, fname)
                if os.path.exists(gt_path):
                    gt_raw = np.load(gt_path).astype(np.float32) if gt_path.endswith('.npy') else (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
                    if gt_raw.ndim == 3:
                        gt_raw = gt_raw.squeeze()
                    gt_clamped = np.clip(gt_raw, 0.0, 1.0)
                    h, w = gt_clamped.shape
                    clean_down = cv2.resize(gt_clamped, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                    inp_clean_t = torch.from_numpy(clean_down).unsqueeze(0).unsqueeze(0).float().to(device)
                    out_clean_t = model(inp_clean_t)
                    out_clean_np = out_clean_t.squeeze().cpu().numpy().clip(0.0, 1.0)

                    clean_psnrs.append(compute_psnr(out_clean_np, gt_clamped))
                    clean_ssims.append(compute_ssim(out_clean_np, gt_clamped))

        if clean_psnrs:
            print(f"  [+] Clean Input Retention PSNR: {np.mean(clean_psnrs):.2f} dB (SSIM: {np.mean(clean_ssims):.4f})")

if __name__ == "__main__":
    main()
