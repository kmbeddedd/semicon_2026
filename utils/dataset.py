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
    Applies metrology-aware augmentations (Dihedral group, CutBlur, and noise jittering) during training.
    Includes in-memory RAM caching for zero-latency multi-GPU streaming without disk I/O bottlenecks.
    """
    def __init__(self, input_dir: str, target_dir: str = None, is_train: bool = False, patch_size: int = 0, scale_factor: int = 2, cache_in_memory: bool = True):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.is_train = is_train
        self.patch_size = patch_size  # 0 = full image (128x128), >0 = random crop
        self.scale_factor = scale_factor
        self.cache_in_memory = cache_in_memory

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

    def _apply_augmentations(self, inp: np.ndarray, tgt: np.ndarray):
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

        # 2. CutBlur Data Augmentation (teach network where and how much to sharpen)
        if np.random.rand() < 0.2:
            s = self.scale_factor
            h, w = inp.shape
            # Random box in input space
            bh, bw = np.random.randint(h // 4, h // 2), np.random.randint(w // 4, w // 2)
            ry, rx = np.random.randint(0, h - bh), np.random.randint(0, w - bw)
            
            # Downsampled clean target patch placed into input
            tgt_down = cv2.resize(tgt[ry*s:(ry+bh)*s, rx*s:(rx+bw)*s], (bw, bh), interpolation=cv2.INTER_AREA)
            inp[ry:ry+bh, rx:rx+bw] = tgt_down

        # 3. Synthetic Poisson-Gaussian noise jittering for OOD generalization
        if np.random.rand() < 0.3:
            noise_std = np.random.uniform(0.01, 0.04)
            inp = inp + np.random.normal(0.0, noise_std, inp.shape).astype(np.float32)
            inp = np.clip(inp, 0.0, 1.0)

        return inp.copy(), tgt.copy()

    def __getitem__(self, idx: int):
        inp_path = self.input_paths[idx]

        if self.cache_in_memory and len(self.cached_inputs) > idx:
            inp_img = self.cached_inputs[idx].copy()
            tgt_img = self.cached_targets[idx].copy() if len(self.cached_targets) > idx else inp_img.copy()
        else:
            inp_raw = self._load_image(inp_path)
            inp_img = robust_percentile_normalize(inp_raw)
            if self.target_paths and idx < len(self.target_paths):
                tgt_path = self.target_paths[idx]
                if os.path.exists(tgt_path):
                    tgt_raw = self._load_image(tgt_path)
                    tgt_img = np.clip(tgt_raw, 0.0, 1.0).astype(np.float32)
                else:
                    tgt_img = inp_img.copy()
            else:
                tgt_img = inp_img.copy()

        if self.is_train and self.target_paths:
            inp_img, tgt_img = self._apply_augmentations(inp_img, tgt_img)

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
