import os
import argparse
import copy
import torch
from torch.utils.data import DataLoader
from models.nafnet import NAFNetSR
from utils.dataset import PairedSemiconDataset
from utils.losses import MetrologyLoss
from utils.metrics import evaluate_metrics

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
    return parser.parse_args()

def main():
    args = parse_args()
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
    model = NAFNetSR(in_channels=1, out_channels=1, width=64, scale_factor=args.scale).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)

    # Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))

    # Warmup + Cosine Annealing Scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        progress = float(epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Loss Function (Calibrated Composite Metrology Loss with empirical blur-free weighting)
    criterion = MetrologyLoss(w_charb=1.0, w_edge=0.05, w_fft=0.05, w_ssim=0.2).to(device)

    best_psnr = 0.0
    print(f"[Metrology Training] Training {len(train_ds)} samples for {args.epochs} epochs (Batch Size: {args.batch_size})...")

    for epoch in range(1, args.epochs + 1):
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

            ema.update(model)

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
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] (LR: {cur_lr:.6f}) - Loss: {avg_loss:.4f} [C:{avg_parts['charb']:.3f}|E:{avg_parts['edge']:.3f}|F:{avg_parts['fft']:.3f}|S:{avg_parts['ssim']:.3f}] | Val PSNR: {val_psnr:.2f}dB (EMA: {val_psnr_ema:.2f}dB) | Val SSIM: {val_ssim:.4f} (EMA: {val_ssim_ema:.4f})")

        # Save Latest Checkpoint
        save_path = os.path.join(args.save_dir, "latest_model.pt")
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "ema_state_dict": ema.ema_model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, save_path)

        # Select higher of model or EMA for best checkpoint
        best_val = max(val_psnr, val_psnr_ema)
        best_weights = ema.ema_model.state_dict() if val_psnr_ema >= val_psnr else model.state_dict()

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
