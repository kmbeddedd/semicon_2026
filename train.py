import os
import argparse
import copy
import math
import random
import glob
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.nafnet import NAFNetSR
from utils.dataset import PairedSemiconDataset
from utils.losses import MetrologyLoss
from utils.metrics import evaluate_metrics, relative_ceiling_efficiency


def seed_everything(seed: int, deterministic: bool = False):
    """Seed training randomness and optionally request deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int):
    """Give each DataLoader worker a deterministic Python/NumPy seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def warmup_cosine_factor(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Learning-rate multiplier for linear warmup followed by cosine decay."""
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def load_model_state_strict(model: torch.nn.Module, state_dict):
    """Load raw or DataParallel state dictionaries without silently dropping keys."""
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def load_model_state_for_extension(model: torch.nn.Module, state_dict, allowed_missing_prefixes):
    """Transfer base weights while allowing only explicitly new module keys."""
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    incompatible = model.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unsafe transfer checkpoint mismatch; missing={invalid_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return incompatible.missing_keys


def atomic_torch_save(payload, path):
    """Write a checkpoint beside its destination and atomically replace it."""
    temp_path = f"{path}.tmp"
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def ensure_validation_split(args):
    """Create a deterministic, non-overlapping paired validation split when needed."""
    valid_exts = ('*.npy', '*.NPY', '*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.tif', '*.TIF')

    if os.path.abspath(args.train_input) == os.path.abspath(args.val_input):
        raise ValueError("Training and validation input directories must be different")
    if os.path.abspath(args.train_target) == os.path.abspath(args.val_target):
        raise ValueError("Training and validation target directories must be different")

    existing_val_inputs = []
    if os.path.exists(args.val_input) and os.path.exists(args.val_target):
        for ext in valid_exts:
            existing_val_inputs.extend(glob.glob(os.path.join(args.val_input, ext)))
        existing_val_inputs = [
            path for path in set(existing_val_inputs)
            if os.path.exists(os.path.join(args.val_target, os.path.basename(path)))
        ]
        if existing_val_inputs:
            return args

    os.makedirs(args.val_input, exist_ok=True)
    os.makedirs(args.val_target, exist_ok=True)
    train_files = []
    for ext in valid_exts:
        train_files.extend(glob.glob(os.path.join(args.train_input, ext)))
    train_files = sorted({
        path for path in train_files
        if os.path.exists(os.path.join(args.train_target, os.path.basename(path)))
    })
    if not train_files:
        raise FileNotFoundError("Cannot create validation split because no paired training files were found.")

    split_rng = random.Random(args.seed)
    val_k = min(len(train_files), max(1, int(len(train_files) * 0.1)))
    val_files = split_rng.sample(train_files, k=val_k)
    for input_path in val_files:
        filename = os.path.basename(input_path)
        target_path = os.path.join(args.train_target, filename)
        shutil.move(input_path, os.path.join(args.val_input, filename))
        shutil.move(target_path, os.path.join(args.val_target, filename))
    print(f"[Dataset Auto-Detect] Created non-overlapping 10% validation split: {len(val_files)} pairs")
    return args

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
    parser.add_argument("--no_cache", action="store_true", help="Disable in-memory dataset caching to reduce RAM use")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="Exponential moving average decay factor")
    parser.add_argument("--resume", type=str, default="", help="Optional path to checkpoint (.pt) to resume training from")
    parser.add_argument("--init_weights", type=str, default="", help="Initialize an extended architecture from compatible base weights")
    parser.add_argument(
        "--spectral_mixer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the local/FFT bottleneck mixer (default: enabled)",
    )
    parser.add_argument(
        "--uncertainty_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the heteroscedastic uncertainty head with beta-NLL (default: enabled)",
    )
    parser.add_argument("--w_nll", type=float, default=0.02, help="Beta-NLL auxiliary loss weight when --uncertainty_head is enabled")
    parser.add_argument("--nll_beta", type=float, default=0.5, help="Detached variance weighting exponent for beta-NLL")
    parser.add_argument("--extension_lr_multiplier", type=float, default=1.0, help="LR multiplier for spectral/uncertainty modules during transfer training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training")
    parser.add_argument("--deterministic", action="store_true", help="Prefer deterministic kernels over maximum throughput")
    return parser.parse_args()

def auto_detect_dataset_paths(args):
    """
    Automatically detects and resolves training and validation paths on Kaggle, Colab, or local disks.
    If zip archives are found in /kaggle/input/ or current directory, extracts them automatically.
    """
    import zipfile

    if os.path.exists(args.train_input) and os.path.exists(args.train_target):
        return ensure_validation_split(args)

    print("[Dataset Auto-Detect] Checking /kaggle/input, /content, and local directories...")

    # Check for zip files in dataset/ or /kaggle/input/ or root
    search_dirs = ["dataset", "/kaggle/input", "/kaggle/working", "/content", "data", "."]
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for z in sorted(glob.glob(os.path.join(s_dir, "**/*.zip"), recursive=True)):
                if any(k in z.lower() for k in ["train", "semicon", "dataset", "noisylr", "gt", "val", "test"]):
                    print(f"[Dataset Auto-Detect] Extracting archive: '{z}' -> 'data/'...")
                    os.makedirs("data", exist_ok=True)
                    try:
                        with zipfile.ZipFile(z, 'r') as zip_ref:
                            zip_ref.extractall("data")
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

    if not (os.path.exists(args.train_input) and os.path.exists(args.train_target)):
        return args
    return ensure_validation_split(args)

def main():
    args = parse_args()
    if args.resume and args.init_weights:
        raise ValueError("Use either --resume or --init_weights, not both")
    if args.w_nll < 0:
        raise ValueError("--w_nll must be non-negative")
    if args.extension_lr_multiplier <= 0:
        raise ValueError("--extension_lr_multiplier must be positive")
    seed_everything(args.seed, deterministic=args.deterministic)
    args = auto_detect_dataset_paths(args)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Metrology Training] Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if device.type == "cuda" and not args.deterministic:
        torch.backends.cudnn.benchmark = True
        print("[Metrology Training] Enabled cuDNN Benchmark & Tensor Core optimization.")

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[Metrology Training] Automatic Mixed Precision (AMP FP16): {use_amp}")

    # Dataset & DataLoader
    if not os.path.exists(args.train_input) or not os.path.exists(args.train_target):
        print(f"[Metrology Training] ERROR: Dataset directory not found: '{args.train_input}'. Please verify paths.")
        return

    cache_in_memory = not args.no_cache
    train_ds = PairedSemiconDataset(args.train_input, args.train_target, is_train=True, patch_size=args.patch_size, scale_factor=args.scale, cache_in_memory=cache_in_memory)
    if len(train_ds) == 0:
        print(f"[Metrology Training] ERROR: No valid training image pairs found in '{args.train_input}' and '{args.train_target}'.")
        return

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker,
        generator=train_generator
    )

    val_ds = PairedSemiconDataset(args.val_input, args.val_target, is_train=False, scale_factor=args.scale, cache_in_memory=cache_in_memory) if (os.path.exists(args.val_input) and os.path.exists(args.val_target)) else None
    if val_ds and len(val_ds) == 0:
        val_ds = None
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda")) if val_ds else None

    # Model Architecture with Bicubic Skip
    model_config = {
        "in_channels": 1,
        "out_channels": 1,
        "width": 64,
        "enc_blocks": (2, 2, 2),
        "dec_blocks": (2, 2, 2),
        "scale_factor": args.scale,
        "use_noise_gate": False,
        "use_spectral_mixer": args.spectral_mixer,
        "predict_uncertainty": args.uncertainty_head,
    }
    raw_model = NAFNetSR(**model_config).to(device)
    initialization_best_psnr = 0.0

    if args.init_weights:
        if not os.path.exists(args.init_weights):
            raise FileNotFoundError(f"Initialization checkpoint not found: {args.init_weights}")
        initialization_checkpoint = torch.load(args.init_weights, map_location=device)
        initialization_state = initialization_checkpoint.get("state_dict", initialization_checkpoint)
        if isinstance(initialization_checkpoint, dict):
            initialization_best_psnr = float(initialization_checkpoint.get("val_psnr", 0.0))
        allowed_prefixes = []
        if args.spectral_mixer:
            allowed_prefixes.append("spectral_mixer.")
        if args.uncertainty_head:
            allowed_prefixes.append("uncertainty_head.")
        missing_keys = load_model_state_for_extension(raw_model, initialization_state, tuple(allowed_prefixes))
        print(f"[Metrology Training] Initialized base weights from '{args.init_weights}' ({len(missing_keys)} new tensors).")

    ema = ModelEMA(raw_model, decay=args.ema_decay)

    if torch.cuda.device_count() > 1:
        print(f"[Metrology Training] Multi-GPU Detected: Distributing across {torch.cuda.device_count()} GPUs (T4 x2) with DataParallel!")
        model = torch.nn.DataParallel(raw_model)
    else:
        model = raw_model

    # Optimizer & LR Scheduler
    extension_params = []
    if raw_model.spectral_mixer is not None:
        extension_params.extend(raw_model.spectral_mixer.parameters())
    if raw_model.uncertainty_head is not None:
        extension_params.extend(raw_model.uncertainty_head.parameters())
    extension_param_ids = {id(param) for param in extension_params}
    base_params = [param for param in raw_model.parameters() if id(param) not in extension_param_ids]
    optimizer_groups = [{"params": base_params, "lr": args.lr}]
    if extension_params:
        optimizer_groups.append({"params": extension_params, "lr": args.lr * args.extension_lr_multiplier})
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))

    # Warmup + Cosine Annealing Scheduler
    def lr_lambda(epoch):
        return warmup_cosine_factor(epoch, args.warmup_epochs, args.epochs)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_epoch = 1
    best_psnr = initialization_best_psnr
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"[Metrology Training] Resuming from checkpoint: '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_model_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
        if checkpoint_model_config and checkpoint_model_config != model_config:
            raise ValueError(f"Resume checkpoint model config {checkpoint_model_config} does not match requested config {model_config}")
        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        load_model_state_strict(raw_model, sd)
        load_model_state_strict(ema.ema_model, checkpoint.get("ema_state_dict", sd))

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if "best_psnr" in checkpoint or "val_psnr" in checkpoint:
            best_psnr = float(checkpoint.get("best_psnr", checkpoint.get("val_psnr", 0.0)))
            print(f"[Metrology Training] Loaded previous best Val PSNR: {best_psnr:.2f} dB")
        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1
            print(f"[Metrology Training] Resuming from epoch {start_epoch}")

        # Align the next epoch's LR with the requested schedule. Recomputing it
        # also allows a completed checkpoint to be extended with a larger --epochs.
        if start_epoch > 1:
            scheduler.last_epoch = start_epoch - 1
            resume_factor = lr_lambda(start_epoch - 1)
            resume_lrs = [base_lr * resume_factor for base_lr in scheduler.base_lrs]
            for param_group, resumed_lr in zip(optimizer.param_groups, resume_lrs):
                param_group["lr"] = resumed_lr
            scheduler._last_lr = resume_lrs
            print(f"[Metrology Training] Resumed learning rate: {resume_lrs[0]:.6f}")

    # Loss Function (Calibrated Composite Metrology Loss with empirical blur-free weighting)
    criterion = MetrologyLoss(
        w_charb=1.0,
        w_edge=0.05,
        w_fft=0.05,
        w_ssim=0.2,
        w_nll=args.w_nll if args.uncertainty_head else 0.0,
        nll_beta=args.nll_beta,
    ).to(device)

    print(f"[Metrology Training] Training {len(train_ds)} samples for {args.epochs} epochs (Batch Size: {args.batch_size})...")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_lr = optimizer.param_groups[0]["lr"]
        running_loss = 0.0
        running_parts = {"charb": 0.0, "edge": 0.0, "fft": 0.0, "ssim": 0.0, "nll": 0.0}

        for inp, tgt, _ in train_loader:
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            optimizer.zero_grad()
            with (torch.amp.autocast('cuda', enabled=use_amp) if hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=use_amp)):
                if args.uncertainty_head:
                    pred, raw_variance = model(inp, return_uncertainty=True)
                else:
                    pred, raw_variance = model(inp), None
                loss, parts = criterion(pred, tgt, raw_variance=raw_variance)

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

        # Validation phase using the same per-image metrics as eval.py.
        val_psnr, val_ssim = 0.0, 0.0
        val_psnr_ema, val_ssim_ema = 0.0, 0.0
        val_bicubic_psnr = 0.0

        if val_loader:
            model.eval()
            ema.ema_model.eval()
            metric_sums = {"psnr": 0.0, "ssim": 0.0, "psnr_ema": 0.0, "ssim_ema": 0.0, "bicubic_psnr": 0.0}
            val_sample_count = 0

            with torch.no_grad():
                for inp, tgt, _ in val_loader:
                    inp = inp.to(device, non_blocking=True)
                    tgt = tgt.to(device, non_blocking=True)

                    with (torch.amp.autocast('cuda', enabled=use_amp) if hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=use_amp)):
                        pred = model(inp)
                        pred_ema = ema.ema_model(inp)

                    bicubic = F.interpolate(inp.float(), scale_factor=args.scale, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
                    p, s = evaluate_metrics(pred.float(), tgt)
                    p_ema, s_ema = evaluate_metrics(pred_ema.float(), tgt)
                    p_bicubic, _ = evaluate_metrics(bicubic, tgt)
                    batch_count = inp.shape[0]
                    metric_sums["psnr"] += p * batch_count
                    metric_sums["ssim"] += s * batch_count
                    metric_sums["psnr_ema"] += p_ema * batch_count
                    metric_sums["ssim_ema"] += s_ema * batch_count
                    metric_sums["bicubic_psnr"] += p_bicubic * batch_count
                    val_sample_count += batch_count

            val_psnr = metric_sums["psnr"] / val_sample_count
            val_ssim = metric_sums["ssim"] / val_sample_count
            val_psnr_ema = metric_sums["psnr_ema"] / val_sample_count
            val_ssim_ema = metric_sums["ssim_ema"] / val_sample_count
            val_bicubic_psnr = metric_sums["bicubic_psnr"] / val_sample_count

        eff_ema = relative_ceiling_efficiency(val_psnr_ema, 38.72, val_bicubic_psnr) if val_loader else 0.0
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] (LR: {epoch_lr:.6f}) - Loss: {avg_loss:.4f} [C:{avg_parts['charb']:.3f}|E:{avg_parts['edge']:.3f}|F:{avg_parts['fft']:.3f}|S:{avg_parts['ssim']:.3f}|N:{avg_parts['nll']:.3f}] | Val PSNR: {val_psnr:.2f}dB (EMA: {val_psnr_ema:.2f}dB, Bicubic: {val_bicubic_psnr:.2f}dB, {eff_ema:.1f}% estimated gain) | Val SSIM: {val_ssim:.4f} (EMA: {val_ssim_ema:.4f})")

        raw_m = model.module if hasattr(model, "module") else model
        model_sd = raw_m.state_dict()

        # Select higher of model or EMA for best checkpoint.
        best_val = max(val_psnr, val_psnr_ema)
        selected_ema = val_psnr_ema >= val_psnr
        best_weights = ema.ema_model.state_dict() if selected_ema else model_sd
        best_ssim = val_ssim_ema if selected_ema else val_ssim

        # Save Latest Checkpoint
        save_path = os.path.join(args.save_dir, "latest_model.pt")
        atomic_torch_save({
            "epoch": epoch,
            "state_dict": model_sd,
            "ema_state_dict": ema.ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": max(best_psnr, best_val),
            "val_psnr": max(val_psnr, val_psnr_ema),
            "val_ssim": val_ssim_ema if val_psnr_ema >= val_psnr else val_ssim,
            "config": vars(args),
            "model_config": model_config
        }, save_path)

        if best_val > best_psnr:
            best_psnr = best_val
            atomic_torch_save({
                "epoch": epoch,
                "state_dict": best_weights,
                "val_psnr": best_psnr,
                "val_ssim": best_ssim,
                "config": vars(args),
                "model_config": model_config,
                "selected_weights": "ema" if selected_ema else "model"
            }, os.path.join(args.save_dir, "best_model.pt"))
            print(f"  [+] Saved new best model checkpoint! Val PSNR: {best_psnr:.2f} dB")

    print(f"\n[Metrology Training] Training Complete! Best Validation PSNR: {best_psnr:.2f} dB")

if __name__ == "__main__":
    main()
