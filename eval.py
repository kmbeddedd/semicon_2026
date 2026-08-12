import os
import argparse
import time
from glob import glob
import cv2
import numpy as np
import torch
from models.nafnet import NAFNetSR
from utils.dataset import robust_percentile_normalize

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Evaluation Script for Semiconductor Inspection Image Restoration")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Path to input test images directory")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Path to output directory for restored images")
    parser.add_argument("--weights", "-w", type=str, default="weights/best_model.pt", help="Path to trained model weights (.pt or .onnx)")
    parser.add_argument("--scale", type=int, default=2, help="Upscaling scale factor (1 or 2)")
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Metrology Eval] Using device: {device}")

    # Load Model
    model = NAFNetSR(in_channels=1, out_channels=1, width=64, scale_factor=args.scale)

    if os.path.exists(args.weights):
        print(f"[Metrology Eval] Loading weights from: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[Metrology Eval] WARNING: Weights file '{args.weights}' not found! Running evaluation with initialized model.")

    model.to(device)
    model.eval()

    # Find test images
    valid_exts = ('*.npy', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
    img_paths = []
    for ext in valid_exts:
        img_paths.extend(glob(os.path.join(args.input_dir, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"[Metrology Eval] No image/npy files found in '{args.input_dir}'. Exit.")
        return

    print(f"[Metrology Eval] Found {len(img_paths)} test files. Running inference...")
    total_time = 0.0

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
                    print(f"Skipping unreadable file: {path}")
                    continue
                raw_img = img_bgr.astype(np.float32) / 255.0

            # Percentile normalization to handle speckle overflow
            norm_img = robust_percentile_normalize(raw_img)

            # Tensor conversion [1, 1, H, W]
            inp_t = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float().to(device)

            start_t = time.perf_counter()
            out_t = model(inp_t)
            proc_t = (time.perf_counter() - start_t) * 1000.0  # in ms
            total_time += proc_t

            # Output array in float32 [0.0, 1.0]
            out_np = out_t.squeeze().cpu().numpy()
            out_clamped = np.clip(out_np, 0.0, 1.0).astype(np.float32)

            # Save as .npy if input was .npy or as standard image
            if fname.endswith('.npy'):
                save_npy_path = os.path.join(args.output_dir, fname)
                np.save(save_npy_path, out_clamped)
                # Also save visual PNG preview for reviewer inspectability
                png_name = fname.replace('.npy', '_restored.png')
                cv2.imwrite(os.path.join(args.output_dir, png_name), (out_clamped * 255.0).astype(np.uint8))
            else:
                out_uint8 = (out_clamped * 255.0).astype(np.uint8)
                save_path = os.path.join(args.output_dir, fname)
                cv2.imwrite(save_path, out_uint8)

    avg_time = total_time / max(len(img_paths), 1)
    print(f"[Metrology Eval] Completed! Processed {len(img_paths)} images.")
    print(f"[Metrology Eval] Average Inference Time: {avg_time:.2f} ms / frame.")
    print(f"[Metrology Eval] Restored outputs saved to: '{args.output_dir}'")

if __name__ == "__main__":
    main()
