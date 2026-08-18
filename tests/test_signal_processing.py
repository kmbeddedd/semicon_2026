import unittest
import os
import tempfile
from types import SimpleNamespace
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
from utils.losses import MetrologyLoss, FFTLoss, SSIMLoss, SobelEdgeLoss, CharbonnierLoss, BetaGaussianNLLLoss
from models.nafnet import NAFNetSR
from models.fusion import GlobalLocalFusionSR
from utils.dataset import PairedSemiconDataset
from train import warmup_cosine_factor, psnr_polish_weights, load_model_state_strict, load_model_state_for_extension, atomic_torch_save, ensure_validation_split
from eval import predict_tta_batched

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

    def test_nafnet_configuration_and_noise_gate_contract(self):
        with self.assertRaises(ValueError):
            NAFNetSR(width=8, enc_blocks=(1, 1), dec_blocks=(1,), scale_factor=2)

        model = NAFNetSR(width=8, enc_blocks=(1,), dec_blocks=(1,), scale_factor=2, use_noise_gate=True).eval()
        inp = torch.rand(2, 1, 8, 8)
        with self.assertRaises(ValueError):
            model(inp)
        out = model(inp, noise_stats=torch.tensor([[0.05, 0.5], [0.01, 0.5]]))
        self.assertEqual(out.shape, (2, 1, 16, 16))

    def test_training_output_preserves_out_of_range_gradients(self):
        model = NAFNetSR(width=8, enc_blocks=(1,), dec_blocks=(1,), scale_factor=2)
        inp = torch.ones(1, 1, 8, 8) * 2.0
        train_out = model.train()(inp)
        eval_out = model.eval()(inp)
        self.assertGreater(train_out.max().item(), 1.0)
        self.assertLessEqual(eval_out.max().item(), 1.0)

    def test_spectral_extension_preserves_transferred_model_at_initialization(self):
        base = NAFNetSR(width=8, enc_blocks=(1,), dec_blocks=(1,), scale_factor=2).eval()
        extended = NAFNetSR(
            width=8,
            enc_blocks=(1,),
            dec_blocks=(1,),
            scale_factor=2,
            use_spectral_mixer=True,
        ).eval()
        missing = load_model_state_for_extension(extended, base.state_dict(), ("spectral_mixer.",))
        self.assertTrue(missing)

        inp = torch.rand(1, 1, 16, 16)
        with torch.no_grad():
            base_out = base(inp)
            extended_out = extended(inp)
        self.assertTrue(torch.equal(base_out, extended_out))

    def test_global_local_fusion_is_identity_safe_and_trainable(self):
        class DummyGlobal(torch.nn.Module):
            def forward(self, value):
                return torch.zeros(
                    value.shape[0],
                    1,
                    value.shape[2] * 2,
                    value.shape[3] * 2,
                    device=value.device,
                    dtype=value.dtype,
                )

        local = NAFNetSR(
            width=8,
            enc_blocks=(1,),
            dec_blocks=(1,),
            scale_factor=2,
            predict_uncertainty=True,
        )
        fusion = GlobalLocalFusionSR(
            local,
            DummyGlobal(),
            fusion_hidden=8,
            use_uncertainty=True,
            freeze_global=True,
        )
        inp = torch.rand(1, 1, 8, 8)
        fusion.eval()
        with torch.no_grad():
            expected = local.eval()(inp)
            actual = fusion(inp)
        self.assertTrue(torch.equal(expected, actual))

        fusion.train()
        fusion(inp).mean().backward()
        self.assertIsNotNone(fusion.fusion_gate[-1].weight.grad)
        self.assertGreater(fusion.fusion_gate[-1].weight.grad.abs().sum().item(), 0.0)

    def test_uncertainty_head_and_beta_nll(self):
        model = NAFNetSR(
            width=8,
            enc_blocks=(1,),
            dec_blocks=(1,),
            scale_factor=2,
            predict_uncertainty=True,
        ).train()
        inp = torch.rand(2, 1, 16, 16)
        pred, raw_variance = model(inp, return_uncertainty=True)
        self.assertEqual(pred.shape, raw_variance.shape)

        target = torch.rand_like(pred)
        criterion = MetrologyLoss(w_nll=0.05, nll_beta=0.5)
        loss, parts = criterion(pred, target, raw_variance)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("nll", parts)
        loss.backward()
        self.assertIsNotNone(model.uncertainty_head[-1].weight.grad)

        with self.assertRaises(ValueError):
            BetaGaussianNLLLoss(beta=1.5)

    def test_scheduler_transition_and_strict_checkpoint_loading(self):
        self.assertAlmostEqual(warmup_cosine_factor(0, 5, 100), 0.2)
        self.assertAlmostEqual(warmup_cosine_factor(4, 5, 100), 1.0)
        self.assertAlmostEqual(warmup_cosine_factor(5, 5, 100), 1.0)
        self.assertAlmostEqual(warmup_cosine_factor(100, 5, 100), 0.0)

        source = NAFNetSR(width=8, enc_blocks=(1,), dec_blocks=(1,), scale_factor=2)
        target = NAFNetSR(width=8, enc_blocks=(1,), dec_blocks=(1,), scale_factor=2)
        prefixed = {f"module.{key}": value for key, value in source.state_dict().items()}
        load_model_state_strict(target, prefixed)
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(source_param, target_param))

        incomplete = dict(source.state_dict())
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaises(RuntimeError):
            load_model_state_strict(target, incomplete)

    def test_dataset_pair_validation_and_clipping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = os.path.join(tmp_dir, "input")
            target_dir = os.path.join(tmp_dir, "target")
            os.makedirs(input_dir)
            os.makedirs(target_dir)
            np.save(os.path.join(input_dir, "sample.npy"), np.linspace(-0.5, 1.5, 64, dtype=np.float32).reshape(8, 8))
            np.save(os.path.join(target_dir, "sample.npy"), np.ones((16, 16), dtype=np.float32))

            dataset = PairedSemiconDataset(input_dir, target_dir, scale_factor=2, cache_in_memory=False)
            inp, target, name = dataset[0]
            self.assertEqual(name, "sample.npy")
            self.assertEqual(tuple(inp.shape), (1, 8, 8))
            self.assertEqual(tuple(target.shape), (1, 16, 16))
            self.assertGreaterEqual(inp.min().item(), 0.0)
            self.assertLessEqual(inp.max().item(), 1.0)

            np.save(os.path.join(target_dir, "sample.npy"), np.ones((15, 16), dtype=np.float32))
            with self.assertRaises(ValueError):
                dataset[0]

    def test_add_plus_preserves_paired_geometry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = os.path.join(tmp_dir, "input")
            target_dir = os.path.join(tmp_dir, "target")
            os.makedirs(input_dir)
            os.makedirs(target_dir)
            for index, value in enumerate((0.0, 1.0)):
                np.save(os.path.join(input_dir, f"{index}.npy"), np.full((8, 8), value, np.float32))
                np.save(os.path.join(target_dir, f"{index}.npy"), np.full((16, 16), value, np.float32))
            dataset = PairedSemiconDataset(
                input_dir,
                target_dir,
                is_train=True,
                scale_factor=2,
                cache_in_memory=False,
                augmentation_mode="add_plus",
                add_probability=1.0,
            )
            np.random.seed(7)
            inp, target, _ = dataset[0]
            downsampled_target = torch.nn.functional.avg_pool2d(target.unsqueeze(0), 2).squeeze(0)
            self.assertTrue(torch.allclose(inp, downsampled_target, atol=1e-6))

    def test_validation_split_is_paired_and_non_overlapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_input = os.path.join(tmp_dir, "train_input")
            train_target = os.path.join(tmp_dir, "train_target")
            val_input = os.path.join(tmp_dir, "val_input")
            val_target = os.path.join(tmp_dir, "val_target")
            os.makedirs(train_input)
            os.makedirs(train_target)
            for index in range(10):
                filename = f"{index:03d}.npy"
                np.save(os.path.join(train_input, filename), np.zeros((8, 8), dtype=np.float32))
                np.save(os.path.join(train_target, filename), np.zeros((16, 16), dtype=np.float32))

            args = SimpleNamespace(
                train_input=train_input,
                train_target=train_target,
                val_input=val_input,
                val_target=val_target,
                seed=42,
            )
            ensure_validation_split(args)
            train_names = set(os.listdir(train_input))
            val_names = set(os.listdir(val_input))
            self.assertEqual(len(train_names), 9)
            self.assertEqual(len(val_names), 1)
            self.assertFalse(train_names & val_names)
            self.assertEqual(val_names, set(os.listdir(val_target)))

    def test_atomic_checkpoint_save(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")
            atomic_torch_save({"epoch": 3, "value": torch.tensor([1.0])}, checkpoint_path)
            self.assertTrue(os.path.exists(checkpoint_path))
            self.assertFalse(os.path.exists(f"{checkpoint_path}.tmp"))
            loaded = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(loaded["epoch"], 3)

    def test_tta_identity(self):
        inp = torch.rand(2, 1, 16, 16)
        out = predict_tta_batched(torch.nn.Identity(), inp)
        self.assertTrue(torch.allclose(out, inp, atol=1e-7))

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
        self.assertEqual(loss.dtype, torch.float32)

    def test_psnr_polish_reaches_exact_mse(self):
        base = {"mse": 0.0, "charb": 1.0, "edge": 0.05, "fft": 0.05, "ssim": 0.2, "nll": 0.02}
        before = psnr_polish_weights(5, total_epochs=10, polish_epochs=3, base_weights=base)
        final = psnr_polish_weights(10, total_epochs=10, polish_epochs=3, base_weights=base)
        self.assertEqual(before, base)
        self.assertEqual(final, {"mse": 1.0, "charb": 0.0, "edge": 0.0, "fft": 0.0, "ssim": 0.0, "nll": 0.0})

        criterion = MetrologyLoss(w_mse=1.0, w_charb=0.0, w_edge=0.0, w_fft=0.0, w_ssim=0.0)
        pred = torch.rand(1, 1, 8, 8)
        target = torch.rand_like(pred)
        loss, parts = criterion(pred, target)
        self.assertAlmostEqual(loss.item(), torch.nn.functional.mse_loss(pred, target).item(), places=7)
        self.assertIn("mse", parts)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the AMP regression test")
    def test_losses_remain_fp32_under_amp(self):
        criterion = MetrologyLoss().cuda()
        pred = torch.rand(2, 1, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        target = torch.rand(2, 1, 32, 32, device="cuda", dtype=torch.float16)
        with torch.amp.autocast("cuda"):
            loss, _ = criterion(pred, target)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

if __name__ == '__main__':
    unittest.main()
