"""Train the experimental NAFNet + official full/Light MambaIRv2 fusion model."""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from models.fusion import (
    MAMBAIRV2_X2_CONFIGS,
    GlobalLocalFusionSR,
    GrayscaleMambaIRv2,
    build_model_from_checkpoint,
    build_official_mambairv2,
    extract_model_state,
)
from models.nafnet import NAFNetSR, resolve_nafnet_config
from train import (
    ModelEMA,
    atomic_torch_save,
    auto_detect_dataset_paths,
    autotune_cuda_batch_size,
    load_model_state_strict,
    psnr_polish_weights,
    seed_everything,
    seed_worker,
    warmup_cosine_factor,
)
from utils.dataset import PairedSemiconDataset
from utils.losses import MetrologyLoss
from utils.metrics import evaluate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train identity-safe NAFNet/MambaIRv2 fusion")
    parser.add_argument("--train_input", default="data/train/NoisyLR")
    parser.add_argument("--train_target", default="data/train/GT")
    parser.add_argument("--val_input", default="data/val/NoisyLR")
    parser.add_argument("--val_target", default="data/val/GT")
    parser.add_argument("--local_weights", default="weights/best_model.pt")
    parser.add_argument("--global_weights", default="", help="Official MambaIRv2 x2 checkpoint matching --mambair_variant")
    parser.add_argument("--mambair_repo", required=True, help="Checkout of https://github.com/csguoh/MambaIR")
    parser.add_argument(
        "--mambair_variant",
        choices=tuple(MAMBAIRV2_X2_CONFIGS),
        default="base",
        help="Official x2 architecture to use; base is the full classic-SR model",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--save_dir", default="weights/fusion_experiment")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--auto_batch_size", action="store_true")
    parser.add_argument("--target_vram_fraction", type=float, default=0.82)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=0)
    parser.add_argument("--fusion_hidden", type=int, default=24)
    parser.add_argument("--local_lr", type=float, default=2e-6)
    parser.add_argument("--fusion_lr", type=float, default=2e-5)
    parser.add_argument("--global_lr", type=float, default=1e-6)
    parser.add_argument("--freeze_local", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze_global", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument(
        "--augmentation",
        choices=("none", "d4", "classic", "add", "add_plus"),
        default="add_plus",
    )
    parser.add_argument("--add_probability", type=float, default=0.5)
    parser.add_argument("--saliency_dir", default="")
    parser.add_argument("--w_mse", type=float, default=0.8)
    parser.add_argument("--w_charb", type=float, default=0.2)
    parser.add_argument("--w_edge", type=float, default=0.0)
    parser.add_argument("--w_fft", type=float, default=0.0)
    parser.add_argument("--w_ssim", type=float, default=0.0)
    parser.add_argument("--w_nll", type=float, default=0.0)
    parser.add_argument("--nll_beta", type=float, default=0.5)
    parser.add_argument("--psnr_polish_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _validate_args(args):
    if args.scale != 2:
        raise ValueError("The provided MambaIRv2 presets are x2 models")
    if args.resume and (args.local_weights or args.global_weights):
        # local_weights has a useful default, so only reject an explicit global initializer.
        if args.global_weights:
            raise ValueError("Use --resume without --global_weights")
    if not args.resume and not args.global_weights:
        raise ValueError("--global_weights is required for a new fusion experiment")
    if args.psnr_polish_epochs < 0 or args.psnr_polish_epochs > args.epochs:
        raise ValueError("--psnr_polish_epochs must be between 0 and --epochs")
    if not 0.0 <= args.add_probability <= 1.0:
        raise ValueError("--add_probability must be in [0, 1]")
    if not 0.5 <= args.target_vram_fraction <= 0.95:
        raise ValueError("--target_vram_fraction must be between 0.5 and 0.95")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if args.max_batch_size < 2:
        raise ValueError("--max_batch_size must be at least 2")
    for name in ("local_lr", "fusion_lr", "global_lr"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")


def _new_model(args, device):
    if not os.path.isfile(args.local_weights):
        raise FileNotFoundError(f"Local checkpoint not found: {args.local_weights}")
    if not os.path.isfile(args.global_weights):
        raise FileNotFoundError(f"Global checkpoint not found: {args.global_weights}")

    local_checkpoint = torch.load(args.local_weights, map_location=device)
    local_config = resolve_nafnet_config(local_checkpoint, scale_factor=args.scale)
    if int(local_config["scale_factor"]) != args.scale:
        raise ValueError("Local and requested scale factors differ")
    local_model = NAFNetSR(**local_config)
    load_model_state_strict(local_model, extract_model_state(local_checkpoint))

    global_config = dict(MAMBAIRV2_X2_CONFIGS[args.mambair_variant])
    global_backbone = build_official_mambairv2(
        args.mambair_repo,
        variant=args.mambair_variant,
        config=global_config,
    )
    global_checkpoint = torch.load(args.global_weights, map_location="cpu")
    global_backbone.load_state_dict(extract_model_state(global_checkpoint), strict=True)

    model = GlobalLocalFusionSR(
        local_model,
        GrayscaleMambaIRv2(global_backbone),
        fusion_hidden=args.fusion_hidden,
        use_uncertainty=bool(local_config["predict_uncertainty"]),
        freeze_local=args.freeze_local,
        freeze_global=args.freeze_global,
    ).to(device)
    config = {
        "local_config": local_config,
        "global_backend": f"mambairv2_{args.mambair_variant}",
        "global_config": global_config,
        "fusion_hidden": args.fusion_hidden,
        "use_uncertainty": bool(local_config["predict_uncertainty"]),
        "freeze_local": args.freeze_local,
        "freeze_global": args.freeze_global,
    }
    return model, config, float(local_checkpoint.get("val_psnr", 0.0))


def _optimizer(model, args):
    groups = []
    fusion_parameters = [p for p in model.fusion_gate.parameters() if p.requires_grad]
    local_parameters = [p for p in model.local_model.parameters() if p.requires_grad]
    global_parameters = [p for p in model.global_model.parameters() if p.requires_grad]
    if fusion_parameters:
        groups.append({"params": fusion_parameters, "lr": args.fusion_lr, "name": "fusion"})
    if local_parameters:
        groups.append({"params": local_parameters, "lr": args.local_lr, "name": "local"})
    if global_parameters:
        groups.append({"params": global_parameters, "lr": args.global_lr, "name": "global"})
    if not groups:
        raise ValueError("At least the fusion gate must remain trainable")
    return torch.optim.AdamW(groups, weight_decay=1e-4, betas=(0.9, 0.99))


def _validate(model, loader, device, use_amp, scale):
    if loader is None:
        return 0.0, 0.0
    model.eval()
    sums = {"psnr": 0.0, "ssim": 0.0}
    count = 0
    with torch.no_grad():
        for inp, target, _ in loader:
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                prediction = model(inp)
            psnr, ssim = evaluate_metrics(prediction.float(), target)
            batch = inp.shape[0]
            sums["psnr"] += psnr * batch
            sums["ssim"] += ssim * batch
            count += batch
    return sums["psnr"] / count, sums["ssim"] / count


def main():
    args = parse_args()
    _validate_args(args)
    seed_everything(args.seed, deterministic=args.deterministic)
    args = auto_detect_dataset_paths(args)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda" and not args.deterministic:
        torch.backends.cudnn.benchmark = True
    print(f"[Fusion Training] device={device} | AMP={use_amp}")

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        raw_model, model_config, model_type = build_model_from_checkpoint(
            resume_checkpoint,
            scale_factor=args.scale,
            mambair_repo=args.mambair_repo,
        )
        if model_type != "global_local_fusion":
            raise ValueError("--resume must point to a fusion checkpoint")
        raw_model = raw_model.to(device)
        load_model_state_strict(raw_model, resume_checkpoint["state_dict"])
        best_psnr = float(resume_checkpoint.get("best_psnr", resume_checkpoint.get("val_psnr", 0.0)))
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
    else:
        raw_model, model_config, best_psnr = _new_model(args, device)
        start_epoch = 1

    cache = not args.no_cache
    train_dataset = PairedSemiconDataset(
        args.train_input,
        args.train_target,
        is_train=True,
        patch_size=args.patch_size,
        scale_factor=args.scale,
        cache_in_memory=cache,
        augmentation_mode=args.augmentation,
        add_probability=args.add_probability,
        saliency_dir=args.saliency_dir,
    )
    val_dataset = PairedSemiconDataset(
        args.val_input,
        args.val_target,
        scale_factor=args.scale,
        cache_in_memory=cache,
    ) if os.path.isdir(args.val_input) and os.path.isdir(args.val_target) else None
    if not train_dataset:
        raise FileNotFoundError("No paired training samples were found")

    base_loss_weights = {
        "mse": args.w_mse,
        "charb": args.w_charb,
        "edge": args.w_edge,
        "fft": args.w_fft,
        "ssim": args.w_ssim,
        "nll": args.w_nll,
    }
    criterion = MetrologyLoss(
        w_mse=args.w_mse,
        w_charb=args.w_charb,
        w_edge=args.w_edge,
        w_fft=args.w_fft,
        w_ssim=args.w_ssim,
        w_nll=args.w_nll,
        nll_beta=args.nll_beta,
    ).to(device)
    optimizer = _optimizer(raw_model, args)
    lr_lambda = lambda epoch: warmup_cosine_factor(epoch, args.warmup_epochs, args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(raw_model, decay=args.ema_decay)

    if resume_checkpoint:
        load_model_state_strict(ema.ema_model, resume_checkpoint.get("ema_state_dict", resume_checkpoint["state_dict"]))
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        if "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        if start_epoch > 1:
            # Recompute the next LR when a completed stage is extended with a
            # larger --epochs value (e.g. ADD+ stage -> D4/MSE polish stage).
            scheduler.last_epoch = start_epoch - 1
            resume_factor = lr_lambda(start_epoch - 1)
            resumed_lrs = [base_lr * resume_factor for base_lr in scheduler.base_lrs]
            for group, resumed_lr in zip(optimizer.param_groups, resumed_lrs):
                group["lr"] = resumed_lr
            scheduler._last_lr = resumed_lrs

    if args.auto_batch_size:
        if device.type != "cuda" or torch.cuda.device_count() != 1:
            print("[Fusion Training] Auto-batch needs one CUDA GPU; retaining configured batch.")
        else:
            sample_input, sample_target, _ = train_dataset[0]
            args.batch_size, _ = autotune_cuda_batch_size(
                raw_model,
                criterion,
                device,
                tuple(sample_input.shape),
                tuple(sample_target.shape),
                use_amp,
                return_uncertainty=args.w_nll > 0,
                target_fraction=args.target_vram_fraction,
                max_batch_size=args.max_batch_size,
            )
            seed_everything(args.seed, deterministic=args.deterministic)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    ) if val_dataset and len(val_dataset) else None
    model = torch.nn.DataParallel(raw_model) if torch.cuda.device_count() > 1 else raw_model
    print(
        f"[Fusion Training] samples={len(train_dataset)} | batch={args.batch_size} | "
        f"global={model_config['global_backend']} | augmentation={args.augmentation} | "
        f"accepted threshold={best_psnr:.5f} dB"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        criterion.set_weights(**psnr_polish_weights(epoch, args.epochs, args.psnr_polish_epochs, base_loss_weights))
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        running_loss = 0.0
        for inp, target, _ in train_loader:
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if args.w_nll > 0:
                    prediction, raw_variance = model(inp, return_uncertainty=True)
                else:
                    prediction, raw_variance = model(inp), None
                loss, _ = criterion(prediction, target, raw_variance)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(raw_model)
            running_loss += float(loss.detach())

        scheduler.step()
        raw_psnr, raw_ssim = _validate(raw_model, val_loader, device, use_amp, args.scale)
        ema_psnr, ema_ssim = _validate(ema.ema_model, val_loader, device, use_amp, args.scale)
        selected_ema = ema_psnr >= raw_psnr
        selected_psnr = ema_psnr if selected_ema else raw_psnr
        selected_ssim = ema_ssim if selected_ema else raw_ssim
        vram_text = ""
        if device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
            total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
            vram_text = f" | VRAM={allocated:.2f} GiB allocated/{reserved:.2f} reserved/{total:.2f} total"
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] loss={running_loss / len(train_loader):.6f} | "
            f"PSNR raw={raw_psnr:.5f}, EMA={ema_psnr:.5f} | SSIM={selected_ssim:.5f}{vram_text}"
        )

        latest = {
            "model_type": "global_local_fusion",
            "model_config": model_config,
            "epoch": epoch,
            "state_dict": raw_model.state_dict(),
            "ema_state_dict": ema.ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": max(best_psnr, selected_psnr),
            "val_psnr": selected_psnr,
            "val_ssim": selected_ssim,
            "config": vars(args),
        }
        atomic_torch_save(latest, os.path.join(args.save_dir, "latest_model.pt"))
        if selected_psnr > best_psnr:
            best_psnr = selected_psnr
            best = {
                "model_type": "global_local_fusion",
                "model_config": model_config,
                "epoch": epoch,
                "state_dict": ema.ema_model.state_dict() if selected_ema else raw_model.state_dict(),
                "val_psnr": selected_psnr,
                "val_ssim": selected_ssim,
                "selected_weights": "ema" if selected_ema else "model",
                "config": vars(args),
            }
            atomic_torch_save(best, os.path.join(args.save_dir, "best_model.pt"))
            print(f"  [+] New fusion best: {best_psnr:.5f} dB")

    print(f"[Fusion Training] Complete. Best validation PSNR: {best_psnr:.5f} dB")


if __name__ == "__main__":
    main()
