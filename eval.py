import os
import argparse
import time
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.nafnet import NAFNetSR
from utils.dataset import robust_percentile_normalize
from utils.metrics import compute_psnr, compute_ssim, wavelet_noise_sigma, psnr_ceiling, relative_ceiling_efficiency

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

def parse_args():
    parser = argparse.ArgumentParser(description="Ultra-Fast High-Precision Evaluation & Inference Pipeline for Semiconductor Restoration")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Path to input degraded images directory")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Path to output directory for restored images")
    parser.add_argument("--target_dir", "-t", type=str, default=None, help="Optional Ground Truth directory to compute benchmark PSNR/SSIM/LPIPS")
    parser.add_argument("--weights", "-w", type=str, default="weights/best_model.pt", help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (1 for denoising, 2 for 2x super-resolution)")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size for parallel GPU inference (1 for streaming)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers for async disk prefetching")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 Tensor Core acceleration")
    parser.add_argument("--no_jit", action="store_true", help="Disable TorchScript JIT kernel fusion")
    parser.add_argument("--no_tta", action="store_true", help="Disable 8-Fold Test-Time Augmentation (TTA)")
    parser.add_argument("--check_clean_damage", action="store_true", help="Audit model performance on clean downsampled GT inputs")
    return parser.parse_args()

class EvalDataset(Dataset):
    """Fast Async Prefetching Dataset for evaluation images."""
    def __init__(self, img_paths, target_dir=None):
        self.img_paths = img_paths
        self.target_dir = target_dir

    def __len__(self):
        return len(self.img_paths)

    def _load_img(self, path):
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

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        fname = os.path.basename(path)
        raw_inp = self._load_img(path)
        norm_inp = robust_percentile_normalize(raw_inp)

        gt_img = None
        if self.target_dir and os.path.exists(self.target_dir):
            gt_path = os.path.join(self.target_dir, fname)
            if os.path.exists(gt_path):
                gt_img = self._load_img(gt_path)

        inp_tensor = torch.from_numpy(norm_inp).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).float() if gt_img is not None else torch.zeros_like(inp_tensor)
        has_gt = gt_img is not None

        return inp_tensor, gt_tensor, has_gt, fname

def predict_tta_batched(model: torch.nn.Module, inp_tensor: torch.Tensor) -> torch.Tensor:
    """
    Batched 8-Fold Test-Time Augmentation (Dihedral Group D4 ensemble: 4 rotations x 2 flips).
    """
    preds = []
    for k in range(4):
        rot_inp = torch.rot90(inp_tensor, k, dims=[2, 3])
        pred_rot = model(rot_inp)
        preds.append(torch.rot90(pred_rot, -k, dims=[2, 3]))

        flip_inp = torch.flip(rot_inp, dims=[3])
        pred_flip = model(flip_inp)
        inv_flip = torch.flip(pred_flip, dims=[3])
        preds.append(torch.rot90(inv_flip, -k, dims=[2, 3]))

    return torch.mean(torch.stack(preds, dim=0), dim=0)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Metrology Eval] Running on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    use_fp16 = (device.type == "cuda") and (not args.no_fp16)
    use_jit = (not args.no_jit)
    use_tta = not args.no_tta

    # Load Base PyTorch Model
    base_model = NAFNetSR(in_channels=1, out_channels=1, width=64, scale_factor=args.scale).to(device)

    weights_path = args.weights
    if not os.path.exists(weights_path) and os.path.exists("best_model.pt"):
        weights_path = "best_model.pt"

    if os.path.exists(weights_path):
        print(f"[Metrology Eval] Loading checkpoint: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        base_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[Metrology Eval] WARNING: Weights file '{weights_path}' not found! Running with initialized model.")

    base_model.eval()

    # Channels-Last & FP16 Acceleration
    if use_fp16:
        base_model = base_model.half()
    if device.type == "cuda":
        base_model = base_model.to(memory_format=torch.channels_last)

    # TorchScript JIT Tracing & Kernel Fusion
    model = base_model
    if use_jit and (not use_tta):
        try:
            sample_dtype = torch.float16 if use_fp16 else torch.float32
            dummy_input = torch.rand(args.batch_size, 1, 128, 128, device=device, dtype=sample_dtype)
            if device.type == "cuda":
                dummy_input = dummy_input.to(memory_format=torch.channels_last)
            
            with torch.no_grad():
                traced_model = torch.jit.trace(base_model, dummy_input)
                # Warmup
                for _ in range(5):
                    _ = traced_model(dummy_input)
            model = traced_model
            model.eval()
            print(f"[Metrology Eval] TorchScript JIT Kernel Fusion Enabled (Fused C++ runtime).")
        except Exception as e:
            print(f"[Metrology Eval] JIT trace fallback to eager model: {e}")
            model = base_model

    print(f"[Metrology Eval] Acceleration Settings -> FP16 Tensor Cores: {use_fp16} | JIT Fusion: {use_jit and (not use_tta)} | Batch Size: {args.batch_size} | 8-Fold TTA: {use_tta}")

    # LPIPS Perceptual Evaluator Setup
    lpips_fn = None
    if HAS_LPIPS and args.target_dir and os.path.exists(args.target_dir):
        try:
            lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
        except Exception:
            lpips_fn = None

    # Find test images
    valid_exts = ('*.npy', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
    img_paths = []
    for ext in valid_exts:
        img_paths.extend(glob(os.path.join(args.input_dir, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"[Metrology Eval] No images found in '{args.input_dir}'.")
        return

    print(f"[Metrology Eval] Processing {len(img_paths)} images with Async DataLoader...")

    dataset = EvalDataset(img_paths, target_dir=args.target_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if os.name != 'nt' else 0,  # 0 on Windows avoids spawn overhead for quick eval
        pin_memory=(device.type == "cuda")
    )

    psnr_scores, ssim_scores, lpips_scores = [], [], []
    gt_ceilings, gt_sigmas = [], []
    total_compute_time = 0.0
    total_images_processed = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for inp_batch, gt_batch, has_gt_mask, fnames in loader:
            b_size = inp_batch.shape[0]
            inp_dev = inp_batch.to(device, non_blocking=True)
            if use_fp16:
                inp_dev = inp_dev.half()
            if device.type == "cuda":
                inp_dev = inp_dev.to(memory_format=torch.channels_last)

            # Measure GPU compute latency
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_t = time.perf_counter()

            if use_tta:
                out_dev = predict_tta_batched(model, inp_dev)
            else:
                out_dev = model(inp_dev)

            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_time = (time.perf_counter() - start_t) * 1000.0
            total_compute_time += batch_time
            total_images_processed += b_size

            out_np_batch = out_dev.float().clamp(0.0, 1.0).cpu().numpy()

            for i in range(b_size):
                fname = fnames[i]
                out_clamped = out_np_batch[i, 0]

                # Quantitative benchmark against GT
                if has_gt_mask[i].item():
                    gt_np = gt_batch[i, 0].numpy()
                    psnr_scores.append(compute_psnr(out_clamped, gt_np))
                    ssim_scores.append(compute_ssim(out_clamped, gt_np))

                    s_gt = wavelet_noise_sigma(gt_np)
                    gt_sigmas.append(s_gt)
                    gt_ceilings.append(psnr_ceiling(s_gt))

                    if lpips_fn is not None:
                        t_p = torch.from_numpy(out_clamped).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0
                        t_g = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0
                        lpips_scores.append(lpips_fn(t_p, t_g).item())

                # Save restored files
                if fname.endswith('.npy'):
                    save_npy = os.path.join(args.output_dir, fname)
                    np.save(save_npy, out_clamped)
                    png_name = fname.replace('.npy', '_restored.png')
                    cv2.imwrite(os.path.join(args.output_dir, png_name), (out_clamped * 255.0).astype(np.uint8))
                else:
                    save_path = os.path.join(args.output_dir, fname)
                    cv2.imwrite(save_path, (out_clamped * 255.0).astype(np.uint8))

    avg_ms = total_compute_time / max(total_images_processed, 1)
    fps = 1000.0 / max(avg_ms, 1e-3)

    print(f"\n[Metrology Eval] Completed! Processed {total_images_processed} images.")
    print(f"[Metrology Eval] GPU Inference Latency: {avg_ms:.2f} ms / frame ({fps:.1f} FPS / throughput).")
    print(f"[Metrology Eval] Restored outputs saved to: '{args.output_dir}'")

    if psnr_scores:
        mean_psnr = np.mean(psnr_scores)
        mean_ssim = np.mean(ssim_scores)
        mean_ceiling = np.mean(gt_ceilings)
        eff = relative_ceiling_efficiency(mean_psnr, mean_ceiling, bicubic_baseline=20.14)
        margin = mean_ceiling - mean_psnr

        print(f"\n[Metrology Eval] Benchmark Metrics across {len(psnr_scores)} Ground Truth pairs:")
        print(f"  [+] Average Restored PSNR:    {mean_psnr:.2f} dB")
        print(f"  [+] Average Restored SSIM:    {mean_ssim:.4f}")
        if lpips_scores:
            print(f"  [+] Average Restored LPIPS:   {np.mean(lpips_scores):.4f} (AlexNet, lower is better)")
        print(f"  [+] Physical GT Noise Ceiling:{mean_ceiling:.2f} dB (Wavelet-MAD sigma: {np.mean(gt_sigmas):.6f})")
        print(f"  [+] Gain-Normalized Ceiling:  {eff:.1f}% of theoretical potential (formula: (PSNR - 20.14) / (Ceiling - 20.14))")
        print(f"  [+] Absolute Noise Margin:    -{margin:.2f} dB from theoretical GT ceiling")

    # Clean Input Preservation Audit
    if args.check_clean_damage and args.target_dir and os.path.exists(args.target_dir):
        print(f"\n[Metrology Eval] Running Clean-Input Degradation Audit...")
        clean_psnrs, clean_ssims = [], []
        with torch.no_grad():
            for path in img_paths[:50]:
                fname = os.path.basename(path)
                gt_path = os.path.join(args.target_dir, fname)
                if os.path.exists(gt_path):
                    gt_img = dataset._load_img(gt_path)
                    h, w = gt_img.shape
                    clean_down = cv2.resize(gt_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                    inp_c = torch.from_numpy(clean_down).unsqueeze(0).unsqueeze(0).to(device)
                    if use_fp16:
                        inp_c = inp_c.half()
                    if device.type == "cuda":
                        inp_c = inp_c.to(memory_format=torch.channels_last)

                    out_c = model(inp_c).float().squeeze().cpu().numpy().clip(0.0, 1.0)
                    clean_psnrs.append(compute_psnr(out_c, gt_img))
                    clean_ssims.append(compute_ssim(out_c, gt_img))

        if clean_psnrs:
            print(f"  [+] Clean Input Retention PSNR: {np.mean(clean_psnrs):.2f} dB (SSIM: {np.mean(clean_ssims):.4f})")

if __name__ == "__main__":
    main()
