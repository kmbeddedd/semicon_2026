import os
import argparse
import copy
import torch
from torch.utils.data import DataLoader
from models.nafnet import NAFNetSR
from utils.dataset import PairedSemiconDataset
from utils.losses import MetrologyLoss
from utils.metrics import evaluate_metrics, relative_ceiling_efficiency

class ModelEMA:
    """Exponential Moving Average (EMA) of model parameters for smoother, higher-accuracy weights."""
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
            for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
                ema_b.data.copy_(model_b.data)

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Metrology Training Pipeline for Semiconductor Image Restoration")
    parser.add_argument("--train_input", type=str, default="data/train/NoisyLR", help="Directory with degraded training images/npy files")
    parser.add_argument("--train_target", type=str, default="data/train/GT", help="Directory with target clean images/npy files")
    parser.add_argument("--val_input", type=str, default="data/val/NoisyLR", help="Directory with degraded validation images/npy files")
    parser.add_argument("--val_target", type=str, default="data/val/GT", help="Directory with target validation images/npy files")
    parser.add_argument("--save_dir", type=str, default="weights", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of linear warmup epochs")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (1 for same-res denoising, 2 for SR)")
    parser.add_argument("--patch_size", type=int, default=0, help="Patch size (0 for full 128x128 image)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="Exponential moving average decay factor")
    parser.add_argument("--resume", type=str, default="", help="Optional path to checkpoint (.pt) to resume training from")
    return parser.parse_args()

def auto_detect_dataset_paths(args):
    """
    Automatically detects and resolves training and validation paths on Kaggle, Colab, or local disks.
    If zip archives are found in /kaggle/input/ or current directory, extracts them automatically.
    """
    import zipfile, glob, shutil, random

    if os.path.exists(args.train_input) and os.path.exists(args.train_target):
        return args

    print("[Dataset Auto-Detect] Checking /kaggle/input, /content, and local directories...")

    # Check for zip files in /kaggle/input/ or root
    search_dirs = ["/kaggle/input", "/kaggle/working", "/content", "data", "."]
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for z in glob.glob(os.path.join(s_dir, "**/*.zip"), recursive=True):
                if any(k in z.lower() for k in ["train", "semicon", "dataset", "noisylr"]):
                    print(f"[Dataset Auto-Detect] Found zip archive: '{z}'. Extracting to 'data/'...")
                    os.makedirs("data", exist_ok=True)
                    try:
                        with zipfile.ZipFile(z, 'r') as zip_ref:
                            zip_ref.extractall("data")
                        print("[Dataset Auto-Detect] Extraction complete!")
                        break
                    except Exception as e:
                        print(f"[Dataset Auto-Detect] Zip extraction note: {e}")

    # Search for NoisyLR and GT directories anywhere
    candidate_lr, candidate_gt = [], []
    for s_dir in ["data", "/kaggle/input", "/content", "."]:
        if os.path.exists(s_dir):
            for root, dirs, _ in os.walk(s_dir):
                for d in dirs:
                    d_lower = d.lower()
                    full_p = os.path.join(root, d)
                    if "noisylr" in d_lower or "noisy_lr" in d_lower:
                        candidate_lr.append(full_p)
                    elif d == "GT" or d_lower == "gt" or "clean" in d_lower:
                        candidate_gt.append(full_p)

    for lr_p in candidate_lr:
        parent = os.path.dirname(lr_p)
        for gt_p in candidate_gt:
            if os.path.dirname(gt_p) == parent or ("train" in lr_p.lower() and "train" in gt_p.lower()):
                args.train_input = lr_p
                args.train_target = gt_p
                print(f"[Dataset Auto-Detect] Found Train Input: '{args.train_input}' | Target: '{args.train_target}'")
                break
        if os.path.exists(args.train_input) and os.path.exists(args.train_target):
            break

    # Auto-create 10% validation split if not present
    if os.path.exists(args.train_input) and os.path.exists(args.train_target):
        if not (os.path.exists(args.val_input) and os.path.exists(args.val_target)):
            val_lr_dir = os.path.join("data", "val", "NoisyLR")
            val_gt_dir = os.path.join("data", "val", "GT")
            if not os.path.exists(val_lr_dir):
                os.makedirs(val_lr_dir, exist_ok=True)
                os.makedirs(val_gt_dir, exist_ok=True)
                exts = ('*.npy', '*.png', '*.jpg', '*.tif')
                files = []
                for e in exts:
                    files.extend(glob.glob(os.path.join(args.train_input, e)))
                if files:
                    random.seed(42)
                    val_k = max(1, int(len(files) * 0.1))
                    val_files = random.sample(files, k=val_k)
                    for vf in val_files:
                        fn = os.path.basename(vf)
                        tgt_f = os.path.join(args.train_target, fn)
                        shutil.copy(vf, os.path.join(val_lr_dir, fn))
                        if os.path.exists(tgt_f):
                            shutil.copy(tgt_f, os.path.join(val_gt_dir, fn))
                    print(f"[Dataset Auto-Detect] Created 10% Val Split: {len(val_files)} samples in '{val_lr_dir}'")
            args.val_input = val_lr_dir
            args.val_target = val_gt_dir

    return args

def main():
    args = parse_args()
    args = auto_detect_dataset_paths(args)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Metrology Training] Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("[Metrology Training] Enabled cuDNN Benchmark & Tensor Core optimization.")

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[Metrology Training] Automatic Mixed Precision (AMP FP16): {use_amp}")

    # Dataset & DataLoader
    if not os.path.exists(args.train_input) or not os.path.exists(args.train_target):
        print(f"[Metrology Training] Dataset directory not found: '{args.train_input}'. Please verify paths.")
        return

    train_ds = PairedSemiconDataset(args.train_input, args.train_target, is_train=True, patch_size=args.patch_size, scale_factor=args.scale)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0)
    )

    val_ds = PairedSemiconDataset(args.val_input, args.val_target, is_train=False, scale_factor=args.scale) if os.path.exists(args.val_input) else None
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")) if val_ds else None

    # Model Architecture with Bicubic Skip
    raw_model = NAFNetSR(in_channels=1, out_channels=1, width=64, scale_factor=args.scale).to(device)
    ema = ModelEMA(raw_model, decay=args.ema_decay)

    if torch.cuda.device_count() > 1:
        print(f"[Metrology Training] Multi-GPU Detected: Distributing across {torch.cuda.device_count()} GPUs (T4 x2) with DataParallel!")
        model = torch.nn.DataParallel(raw_model)
    else:
        model = raw_model

    # Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))

    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f"[Metrology Training] Resuming from checkpoint: '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(sd, strict=False)
        if "ema_state_dict" in checkpoint:
            ema.ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
        else:
            ema.ema_model.load_state_dict(sd, strict=False)
        if "val_psnr" in checkpoint:
            best_psnr = float(checkpoint["val_psnr"])
            print(f"[Metrology Training] Loaded previous best Val PSNR: {best_psnr:.2f} dB")
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"[Metrology Training] Resuming from epoch {start_epoch}")

    # Warmup + Cosine Annealing Scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        progress = float(epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Loss Function (Calibrated Composite Metrology Loss with empirical blur-free weighting)
    criterion = MetrologyLoss(w_charb=1.0, w_edge=0.05, w_fft=0.05, w_ssim=0.2).to(device)

    best_psnr = best_psnr if 'best_psnr' in locals() else 0.0
    print(f"[Metrology Training] Training {len(train_ds)} samples for {args.epochs} epochs (Batch Size: {args.batch_size})...")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_parts = {"charb": 0.0, "edge": 0.0, "fft": 0.0, "ssim": 0.0}

        for inp, tgt, _ in train_loader:
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            optimizer.zero_grad()
            with (torch.amp.autocast('cuda', enabled=use_amp) if hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=use_amp)):
                pred = model(inp)
                loss, parts = criterion(pred, tgt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            raw_m = model.module if hasattr(model, "module") else model
            ema.update(raw_m)

            running_loss += loss.item()
            for k in running_parts:
                running_parts[k] += parts.get(k, 0.0)

        scheduler.step()
        num_batches = len(train_loader)
        avg_loss = running_loss / num_batches
        avg_parts = {k: v / num_batches for k, v in running_parts.items()}

        # Validation phase (evaluate on both Model and EMA model)
        val_psnr, val_ssim = 0.0, 0.0
        val_psnr_ema, val_ssim_ema = 0.0, 0.0

        if val_loader:
            model.eval()
            ema.ema_model.eval()
            val_psnrs, val_ssims = [], []
            val_psnrs_ema, val_ssims_ema = [], []

            with torch.no_grad():
                for inp, tgt, _ in val_loader:
                    inp = inp.to(device, non_blocking=True)
                    tgt = tgt.to(device, non_blocking=True)

                    with (torch.amp.autocast('cuda', enabled=use_amp) if hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=use_amp)):
                        pred = model(inp)
                        pred_ema = ema.ema_model(inp)

                    p, s = evaluate_metrics(pred, tgt)
                    p_ema, s_ema = evaluate_metrics(pred_ema, tgt)
                    val_psnrs.append(p)
                    val_ssims.append(s)
                    val_psnrs_ema.append(p_ema)
                    val_ssims_ema.append(s_ema)

            val_psnr = sum(val_psnrs) / len(val_psnrs)
            val_ssim = sum(val_ssims) / len(val_ssims)
            val_psnr_ema = sum(val_psnrs_ema) / len(val_psnrs_ema)
            val_ssim_ema = sum(val_ssims_ema) / len(val_ssims_ema)

        cur_lr = scheduler.get_last_lr()[0]
        eff_ema = relative_ceiling_efficiency(val_psnr_ema, 38.72, 20.14)
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] (LR: {cur_lr:.6f}) - Loss: {avg_loss:.4f} [C:{avg_parts['charb']:.3f}|E:{avg_parts['edge']:.3f}|F:{avg_parts['fft']:.3f}|S:{avg_parts['ssim']:.3f}] | Val PSNR: {val_psnr:.2f}dB (EMA: {val_psnr_ema:.2f}dB, {eff_ema:.1f}% ceiling) | Val SSIM: {val_ssim:.4f} (EMA: {val_ssim_ema:.4f})")

        raw_m = model.module if hasattr(model, "module") else model
        model_sd = raw_m.state_dict()

        # Save Latest Checkpoint
        save_path = os.path.join(args.save_dir, "latest_model.pt")
        torch.save({
            "epoch": epoch,
            "state_dict": model_sd,
            "ema_state_dict": ema.ema_model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, save_path)

        # Select higher of model or EMA for best checkpoint
        best_val = max(val_psnr, val_psnr_ema)
        best_weights = ema.ema_model.state_dict() if val_psnr_ema >= val_psnr else model_sd

        if best_val > best_psnr:
            best_psnr = best_val
            torch.save({
                "epoch": epoch,
                "state_dict": best_weights,
                "val_psnr": best_psnr,
                "val_ssim": max(val_ssim, val_ssim_ema)
            }, os.path.join(args.save_dir, "best_model.pt"))
            # Also save to root best_model.pt for easy access
            torch.save({
                "epoch": epoch,
                "state_dict": best_weights,
                "val_psnr": best_psnr,
                "val_ssim": max(val_ssim, val_ssim_ema)
            }, "best_model.pt")
            print(f"  [+] Saved new best model checkpoint! Val PSNR: {best_psnr:.2f} dB")

    print(f"\n[Metrology Training] Training Complete! Best Validation PSNR: {best_psnr:.2f} dB")

if __name__ == "__main__":
    main()
