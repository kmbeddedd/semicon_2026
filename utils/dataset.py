import os
from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def robust_percentile_normalize(img: np.ndarray, p_low: float = 0.01, p_high: float = 99.99) -> np.ndarray:
    """
    Robust intensity normalization for semiconductor inspection images.
    Clips extreme out-of-range speckle noise spikes before mapping signal to [0, 1].
    """
    low, high = np.percentile(img, (p_low, p_high))
    if high - low < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    img_clipped = np.clip(img, low, high)
    normalized = (img_clipped - low) / (high - low)
    return normalized.astype(np.float32)

class PairedSemiconDataset(Dataset):
    """
    Paired dataset for semiconductor inspection image restoration.
    Expects input_dir and target_dir containing matching filenames (.npy, .png, .jpg, .tif).
    Optionally applies dynamic augmentations for training.
    """
    def __init__(self, input_dir: str, target_dir: str = None, is_train: bool = False):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.is_train = is_train

        valid_exts = ('*.npy', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
        self.input_paths = []
        for ext in valid_exts:
            self.input_paths.extend(glob(os.path.join(input_dir, ext)))
        self.input_paths = sorted(self.input_paths)

        if target_dir and os.path.exists(target_dir):
            self.target_paths = [
                os.path.join(target_dir, os.path.basename(p)) for p in self.input_paths
            ]
        else:
            self.target_paths = None

    def __len__(self):
        return len(self.input_paths)

    def _load_image(self, path: str) -> np.ndarray:
        if path.endswith('.npy'):
            img = np.load(path).astype(np.float32)
            if img.ndim == 3:
                img = img.squeeze()
            return img
        else:
            img_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                raise FileNotFoundError(f"Failed to read image: {path}")
            return img_raw.astype(np.float32) / 255.0

    def _apply_augmentations(self, inp: np.ndarray, tgt: np.ndarray):
        # Random 90 degree rotations and flips
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

        # Synthetic speckle noise injection for robust OOD generalization
        if np.random.rand() > 0.7:
            speckle_noise = np.random.normal(1.0, 0.05, inp.shape)
            inp = inp * speckle_noise
            inp = np.clip(inp, 0.0, 1.0)

        return inp.copy(), tgt.copy()

    def __getitem__(self, idx: int):
        inp_path = self.input_paths[idx]
        inp_raw = self._load_image(inp_path)

        # Apply robust dynamic range percentile normalization
        inp_img = robust_percentile_normalize(inp_raw)

        if self.target_paths:
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

        # Convert to Tensor [1, H, W]
        inp_tensor = torch.from_numpy(inp_img).unsqueeze(0).float()
        tgt_tensor = torch.from_numpy(tgt_img).unsqueeze(0).float()

        return inp_tensor, tgt_tensor, os.path.basename(inp_path)
