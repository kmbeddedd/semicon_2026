import os
from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def robust_percentile_normalize(img: np.ndarray) -> np.ndarray:
    """
    Robust intensity normalization for semiconductor inspection images.
    Preserves exact physical [0, 1] dynamic range while clipping extreme shot noise outliers.
    """
    return np.clip(img, 0.0, 1.0).astype(np.float32)

class PairedSemiconDataset(Dataset):
    """
    Paired dataset for semiconductor inspection image restoration.
    Expects input_dir and target_dir containing matching filenames (.npy, .png, .jpg, .tif).
    Applies selectable metrology-aware augmentations during training. ``add`` and
    ``add_plus`` implement the paired mixing equations from ADD (CVPR 2025).
    They can consume precomputed CAM masks, while a deterministic high-frequency
    proxy keeps the experiment usable when no masks have been generated yet.
    Includes in-memory RAM caching for zero-latency multi-GPU streaming without disk I/O bottlenecks.
    """
    def __init__(
        self,
        input_dir: str,
        target_dir: str = None,
        is_train: bool = False,
        patch_size: int = 0,
        scale_factor: int = 2,
        cache_in_memory: bool = True,
        augmentation_mode: str = "classic",
        add_probability: float = 0.5,
        saliency_dir: str = "",
    ):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.is_train = is_train
        self.patch_size = patch_size  # 0 = full image (128x128), >0 = random crop
        self.scale_factor = scale_factor
        self.cache_in_memory = cache_in_memory
        valid_augmentation_modes = {"none", "d4", "classic", "add", "add_plus"}
        if augmentation_mode not in valid_augmentation_modes:
            raise ValueError(
                f"augmentation_mode must be one of {sorted(valid_augmentation_modes)}, "
                f"got '{augmentation_mode}'"
            )
        if not 0.0 <= add_probability <= 1.0:
            raise ValueError("add_probability must be in [0, 1]")
        self.augmentation_mode = augmentation_mode
        self.add_probability = float(add_probability)
        self.saliency_dir = saliency_dir

        valid_exts = ('*.npy', '*.NPY', '*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.tif', '*.TIF', '*.tiff', '*.TIFF', '*.bmp', '*.BMP')
        all_inputs = []
        for ext in valid_exts:
            all_inputs.extend(glob(os.path.join(input_dir, ext)))
        all_inputs = sorted(list(set(all_inputs)))

        self.input_paths = []
        self.target_paths = []

        if target_dir and os.path.exists(target_dir):
            for inp_p in all_inputs:
                fname = os.path.basename(inp_p)
                tgt_p = os.path.join(target_dir, fname)
                if os.path.exists(tgt_p):
                    self.input_paths.append(inp_p)
                    self.target_paths.append(tgt_p)
        else:
            self.input_paths = all_inputs
            self.target_paths = None

        # Pre-cache images in RAM to saturate multi-GPU training
        self.cached_inputs = []
        self.cached_targets = []
        if self.cache_in_memory and len(self.input_paths) > 0:
            for i in range(len(self.input_paths)):
                raw_inp = self._load_image(self.input_paths[i])
                self.cached_inputs.append(robust_percentile_normalize(raw_inp))
                if self.target_paths and i < len(self.target_paths):
                    raw_tgt = self._load_image(self.target_paths[i])
                    self.cached_targets.append(np.clip(raw_tgt, 0.0, 1.0).astype(np.float32))

    def __len__(self):
        return len(self.input_paths)

    def _load_image(self, path: str) -> np.ndarray:
        if path.lower().endswith('.npy'):
            img = np.load(path).astype(np.float32)
            if img.ndim == 3:
                img = img.squeeze()
        else:
            img_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                raise FileNotFoundError(f"Failed to read image: {path}")
            img = img_raw.astype(np.float32) / 255.0

        if img.ndim != 2:
            raise ValueError(f"Expected a single-channel 2D image at '{path}', got shape {img.shape}")
        if not np.isfinite(img).all():
            raise ValueError(f"Image contains NaN or infinite values: '{path}'")
        return img

    def _load_pair(self, idx: int):
        """Load a pair without augmentation; used for both samples in ADD mixing."""
        inp_path = self.input_paths[idx]
        if self.cache_in_memory and len(self.cached_inputs) > idx:
            inp_img = self.cached_inputs[idx].copy()
            tgt_img = self.cached_targets[idx].copy() if len(self.cached_targets) > idx else inp_img.copy()
            return inp_img, tgt_img

        inp_img = robust_percentile_normalize(self._load_image(inp_path))
        if self.target_paths and idx < len(self.target_paths):
            tgt_img = np.clip(self._load_image(self.target_paths[idx]), 0.0, 1.0).astype(np.float32)
        else:
            tgt_img = inp_img.copy()
        return inp_img, tgt_img

    def _proxy_saliency_mask(self, inp: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """Return a coarse, detail-rich LR mask when an external CAM is absent.

        ADD reports that coarse partitions work better than fine masks. We score
        a 2x2 grid using target gradients plus the paired degradation residual,
        then retain the most informative cell. This is deliberately labelled a
        proxy and is not claimed to reproduce the paper's calibrated IG map.
        """
        h, w = inp.shape
        target_lr = cv2.resize(tgt, (w, h), interpolation=cv2.INTER_AREA)
        grad_x = cv2.Sobel(target_lr, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(target_lr, cv2.CV_32F, 0, 1, ksize=3)
        saliency = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        saliency += np.abs(target_lr - inp)

        row_edges = np.linspace(0, h, 3, dtype=np.int32)
        col_edges = np.linspace(0, w, 3, dtype=np.int32)
        cells = []
        for row in range(2):
            for col in range(2):
                y0, y1 = row_edges[row], row_edges[row + 1]
                x0, x1 = col_edges[col], col_edges[col + 1]
                cells.append((float(saliency[y0:y1, x0:x1].mean()), y0, y1, x0, x1))
        _, y0, y1, x0, x1 = max(cells, key=lambda item: item[0])
        mask = np.zeros((h, w), dtype=np.float32)
        mask[y0:y1, x0:x1] = 1.0
        return mask

    def _saliency_mask(self, idx: int, inp: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        if self.saliency_dir:
            filename = os.path.basename(self.input_paths[idx])
            stem, _ = os.path.splitext(filename)
            candidates = (
                os.path.join(self.saliency_dir, f"{stem}.npy"),
                os.path.join(self.saliency_dir, filename),
            )
            for path in candidates:
                if os.path.exists(path):
                    mask = self._load_image(path)
                    mask = cv2.resize(mask, (inp.shape[1], inp.shape[0]), interpolation=cv2.INTER_NEAREST)
                    if float(mask.max() - mask.min()) <= 1e-8:
                        return self._proxy_saliency_mask(inp, tgt)
                    threshold = float(np.quantile(mask, 0.75))
                    return (mask >= threshold).astype(np.float32)
        return self._proxy_saliency_mask(inp, tgt)

    def _apply_add(self, idx: int, inp: np.ndarray, tgt: np.ndarray):
        """Apply ADD/ADD+ to a geometrically corresponding LR/HR pair."""
        if len(self.input_paths) < 2 or np.random.rand() >= self.add_probability:
            return inp, tgt

        partner_idx = np.random.randint(0, len(self.input_paths) - 1)
        if partner_idx >= idx:
            partner_idx += 1
        partner_inp, partner_tgt = self._load_pair(partner_idx)
        if partner_inp.shape != inp.shape or partner_tgt.shape != tgt.shape:
            return inp, tgt

        mask_lr = self._saliency_mask(idx, inp, tgt)
        mask_hr = cv2.resize(mask_lr, (tgt.shape[1], tgt.shape[0]), interpolation=cv2.INTER_NEAREST)

        strategies = ["mix"] if self.augmentation_mode == "add" else ["mix", "paste", "intensity"]
        strategy = strategies[np.random.randint(0, len(strategies))]
        if strategy == "mix":
            lam = float(np.random.beta(1.2, 1.2))
            mixed_lr = lam * inp + (1.0 - lam) * partner_inp
            mixed_hr = lam * tgt + (1.0 - lam) * partner_tgt
            inp = mask_lr * mixed_lr + (1.0 - mask_lr) * partner_inp
            tgt = mask_hr * mixed_hr + (1.0 - mask_hr) * partner_tgt
        elif strategy == "paste":
            inp = mask_lr * inp + (1.0 - mask_lr) * partner_inp
            tgt = mask_hr * tgt + (1.0 - mask_hr) * partner_tgt
        else:
            # Paired CutBlur-style intensity diversity at the salient location.
            clean_lr = cv2.resize(tgt, (inp.shape[1], inp.shape[0]), interpolation=cv2.INTER_AREA)
            inp = mask_lr * clean_lr + (1.0 - mask_lr) * inp

        return np.clip(inp, 0.0, 1.0), np.clip(tgt, 0.0, 1.0)

    def _apply_augmentations(self, idx: int, inp: np.ndarray, tgt: np.ndarray):
        if self.augmentation_mode in {"add", "add_plus"}:
            inp, tgt = self._apply_add(idx, inp, tgt)

        # 1. Random 90-degree rotations & Dihedral Flips
        rot_k = np.random.randint(0, 4)
        if rot_k > 0:
            inp = np.rot90(inp, rot_k)
            tgt = np.rot90(tgt, rot_k)

        if np.random.rand() > 0.5:
            inp = np.fliplr(inp)
            tgt = np.fliplr(tgt)

        if np.random.rand() > 0.5:
            inp = np.flipud(inp)
            tgt = np.flipud(tgt)

        # 2. Legacy CutBlur is retained only in the original "classic" recipe.
        if self.augmentation_mode == "classic" and np.random.rand() < 0.2:
            s = self.scale_factor
            h, w = inp.shape
            # Random box in input space
            bh, bw = np.random.randint(h // 4, h // 2), np.random.randint(w // 4, w // 2)
            ry, rx = np.random.randint(0, h - bh), np.random.randint(0, w - bw)
            
            # Downsampled clean target patch placed into input
            tgt_down = cv2.resize(tgt[ry*s:(ry+bh)*s, rx*s:(rx+bw)*s], (bw, bh), interpolation=cv2.INTER_AREA)
            inp[ry:ry+bh, rx:rx+bw] = tgt_down

        # 3. Legacy Gaussian jitter is intentionally absent from PSNR-focused modes.
        if self.augmentation_mode == "classic" and np.random.rand() < 0.3:
            noise_std = np.random.uniform(0.01, 0.04)
            inp = inp + np.random.normal(0.0, noise_std, inp.shape).astype(np.float32)
            inp = np.clip(inp, 0.0, 1.0)

        return inp.copy(), tgt.copy()

    def __getitem__(self, idx: int):
        inp_path = self.input_paths[idx]
        inp_img, tgt_img = self._load_pair(idx)

        if self.is_train and self.target_paths and self.augmentation_mode != "none":
            inp_img, tgt_img = self._apply_augmentations(idx, inp_img, tgt_img)

        expected_target_shape = (inp_img.shape[0] * self.scale_factor, inp_img.shape[1] * self.scale_factor)
        if self.target_paths and tgt_img.shape != expected_target_shape:
            raise ValueError(
                f"Pair shape mismatch for '{os.path.basename(inp_path)}': input {inp_img.shape}, "
                f"target {tgt_img.shape}, expected target {expected_target_shape} for scale={self.scale_factor}"
            )

        # Optional random patch crop during training (if patch_size > 0 and < image size)
        if self.is_train and self.patch_size > 0:
            h, w = inp_img.shape
            if self.patch_size < h and self.patch_size < w:
                ps = self.patch_size
                rh = np.random.randint(0, h - ps + 1)
                rw = np.random.randint(0, w - ps + 1)
                inp_img = inp_img[rh:rh+ps, rw:rw+ps]
                s = self.scale_factor
                tgt_img = tgt_img[rh*s:(rh+ps)*s, rw*s:(rw+ps)*s]

        # Convert to Tensor [1, H, W]
        inp_tensor = torch.from_numpy(inp_img).unsqueeze(0).float()
        tgt_tensor = torch.from_numpy(tgt_img).unsqueeze(0).float()

        return inp_tensor, tgt_tensor, os.path.basename(inp_path)
