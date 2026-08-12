import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.nafnet import NAFNetSR
from utils.dataset import PairedSemiconDataset
from utils.losses import MetrologyLoss
from utils.metrics import evaluate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Training Script for Semiconductor Image Restoration")
    parser.add_argument("--train_input", type=str, default="data/train/NoisyLR", help="Directory with degraded training images/npy files")
    parser.add_argument("--train_target", type=str, default="data/train/GT", help="Directory with target clean images/npy files")
    parser.add_argument("--val_input", type=str, default="data/val/NoisyLR", help="Directory with degraded validation images/npy files")
    parser.add_argument("--val_target", type=str, default="data/val/GT", help="Directory with target validation images/npy files")
    parser.add_argument("--save_dir", type=str, default="weights", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (1 for same-res denoising, 2 for SR)")
    parser.add_argument("--patch_size", type=int, default=64, help="Random crop patch size (0 for full image)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Metrology Training] Training on device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("[Metrology Training] Enabled cuDNN Benchmark & Tensor Core optimization.")

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[Metrology Training] Automatic Mixed Precision (AMP FP16): {use_amp}")

    # Dataset & DataLoader
    if not os.path.exists(args.train_input) or not os.path.exists(args.train_target):
        print(f"[Metrology Training] Dataset directory not found: '{args.train_input}'. Please populate your dataset before training.")
        return

    train_ds = PairedSemiconDataset(args.train_input, args.train_target, is_train=True, patch_size=args.patch_size, scale_factor=args.scale)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))

    val_ds = PairedSemiconDataset(args.val_input, args.val_target, is_train=False, scale_factor=args.scale) if os.path.exists(args.val_input) else None
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_ds else None

    # Model, Optimizer, Loss
    model = NAFNetSR(in_channels=1, out_channels=1, width=64, scale_factor=args.scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = MetrologyLoss(w_charb=1.0, w_edge=0.3, w_fft=0.3, w_ssim=0.2).to(device)

    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for inp, tgt, _ in train_loader:
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(inp)
                loss, _ = criterion(pred, tgt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # Validation phase
        val_psnr, val_ssim = 0.0, 0.0
        if val_loader:
            model.eval()
            val_psnrs, val_ssims = [], []
            with torch.no_grad():
                for inp, tgt, _ in val_loader:
                    inp = inp.to(device, non_blocking=True)
                    tgt = tgt.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred = model(inp)
                    p, s = evaluate_metrics(pred, tgt)
                    val_psnrs.append(p)
                    val_ssims.append(s)
            val_psnr = sum(val_psnrs) / len(val_psnrs)
            val_ssim = sum(val_ssims) / len(val_ssims)

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] - Loss: {avg_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")

        # Save Checkpoint
        save_path = os.path.join(args.save_dir, "latest_model.pt")
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, save_path)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, os.path.join(args.save_dir, "best_model.pt"))
            print(f"  -> Saved new best model checkpoint with PSNR {best_psnr:.2f} dB!")

    print("[Metrology Training] Training complete!")

if __name__ == "__main__":
    main()
