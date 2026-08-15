import unittest
import numpy as np
import torch
from utils.signal_analysis import (
    wavelet_noise_sigma,
    psnr_ceiling,
    estimate_noise_parameters_ab,
    vst_forward_np,
    vst_inverse_np,
    vst_forward_torch,
    vst_inverse_torch,
    NoiseGate
)
from utils.losses import MetrologyLoss, FFTLoss, SSIMLoss, SobelEdgeLoss, CharbonnierLoss
from models.nafnet import NAFNetSR

class TestSignalProcessing(unittest.TestCase):

    def test_wavelet_noise_sigma(self):
        # Generate known noise on a flat image
        np.random.seed(42)
        target_sigma = 0.05
        noise = np.random.normal(0, target_sigma, (128, 128)).astype(np.float32)
        est_sigma = wavelet_noise_sigma(noise)
        self.assertAlmostEqual(est_sigma, target_sigma, delta=0.01)

        # Test ceiling calculation
        ceil = psnr_ceiling(0.01)
        self.assertAlmostEqual(ceil, 40.0, places=1)

    def test_vst_roundtrip_numpy(self):
        np.random.seed(42)
        img = np.random.uniform(0.1, 0.9, (128, 128)).astype(np.float32)
        a, b = 0.02, 0.01
        
        vst_fwd = vst_forward_np(img, a, b)
        vst_rev = vst_inverse_np(vst_fwd, a, b)
        
        max_diff = np.max(np.abs(img - vst_rev))
        self.assertLess(max_diff, 1e-4)

    def test_vst_roundtrip_torch(self):
        img_t = torch.rand(2, 1, 64, 64, dtype=torch.float32) * 0.8 + 0.1
        a, b = 0.03, 0.015
        
        fwd_t = vst_forward_torch(img_t, a, b)
        rev_t = vst_inverse_torch(fwd_t, a, b)
        
        max_diff = torch.max(torch.abs(img_t - rev_t)).item()
        self.assertLess(max_diff, 1e-4)

    def test_noise_gate(self):
        gate = NoiseGate(in_features=2, hidden_dim=16)
        noisy = torch.ones(2, 1, 32, 32)
        base = torch.zeros(2, 1, 32, 32)
        stats = torch.tensor([[0.05, 0.5], [0.01, 0.5]], dtype=torch.float32)
        
        out = gate(noisy, base, stats)
        self.assertEqual(out.shape, (2, 1, 32, 32))
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_nafnet_zero_init_and_forward(self):
        model = NAFNetSR(in_channels=1, out_channels=1, width=32, enc_blocks=[1, 1], dec_blocks=[1, 1], scale_factor=2)
        inp = torch.rand(1, 1, 32, 32)
        out = model(inp)
        self.assertEqual(out.shape, (1, 1, 64, 64))

    def test_losses_fp32_safety(self):
        criterion = MetrologyLoss(w_charb=1.0, w_edge=0.05, w_fft=0.05, w_ssim=0.2)
        pred = torch.rand(2, 1, 64, 64, requires_grad=True)
        target = torch.rand(2, 1, 64, 64)
        
        loss, parts = criterion(pred, target)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.isnan(pred.grad).any())

if __name__ == '__main__':
    unittest.main()
