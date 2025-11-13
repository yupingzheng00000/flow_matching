# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for LUT training pipeline."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

# Add examples/image to path for imports
_test_dir = Path(__file__).resolve().parent
_examples_image = _test_dir.parent / "examples" / "image"
if str(_examples_image) not in sys.path:
    sys.path.insert(0, str(_examples_image))

from train_arg_parser import get_args_parser  # noqa: E402
from training.load_and_save import save_model  # noqa: E402
from flow_matching.path import MetricInducedGibbsProbPath  # noqa: E402


class TestLUTArgumentParsing(unittest.TestCase):
    """Test CLI argument parsing for LUT options."""

    def test_lut_default_disabled(self):
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--output_dir", "/tmp",
        ])
        self.assertFalse(getattr(args, "mi_learnable_lut", False))

    def test_lut_enable_flag(self):
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--output_dir", "/tmp",
            "--mi_learnable_lut",
        ])
        self.assertTrue(args.mi_learnable_lut)

    def test_lut_num_channels(self):
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--output_dir", "/tmp",
            "--mi_learnable_lut",
            "--mi_lut_num_channels", "1",
        ])
        self.assertEqual(args.mi_lut_num_channels, 1)

    def test_lut_share_channels(self):
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--output_dir", "/tmp",
            "--mi_learnable_lut",
            "--mi_lut_share_channels",
        ])
        self.assertTrue(getattr(args, "mi_lut_share_channels", False))

    def test_lut_freeze_flag(self):
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--mi_learnable_lut",
            "--mi_freeze_lut",
        ])
        self.assertTrue(args.mi_freeze_lut)

    def test_lut_lr_scale(self):
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--mi_learnable_lut",
            "--mi_lut_lr_scale", "0.05",
        ])
        self.assertAlmostEqual(args.mi_lut_lr_scale, 0.05)

    def test_lut_weight_decay(self):
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--mi_learnable_lut",
            "--mi_lut_weight_decay", "1e-3",
        ])
        self.assertAlmostEqual(args.mi_lut_weight_decay, 1e-3)


class TestLUTOptimizerIntegration(unittest.TestCase):
    """Test LUT parameter groups in optimizer."""

    def test_lut_parameters_in_optimizer(self):
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=3,
        )
        lut_params = list(path.lut_parameters())
        self.assertGreater(len(lut_params), 0)

        model_params = [torch.nn.Parameter(torch.randn(10, 10))]
        optimizer = torch.optim.AdamW([
            {"params": model_params, "lr": 1e-3, "name": "model"},
            {"params": lut_params, "lr": 1e-4, "name": "lut"},
        ])
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[1]["name"], "lut")

    def test_lut_gradient_step(self):
        path = MetricInducedGibbsProbPath(
            vocab_size=16,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=1,
        )
        optimizer = torch.optim.SGD(path.lut_parameters(), lr=0.1)
        initial = path.learnable_lut.weight.clone()
        for _ in range(3):
            optimizer.zero_grad()
            table = path._build_lut_distance_table(device=torch.device("cpu"), dtype=torch.float32)
            loss = table.sum()
            loss.backward()
            optimizer.step()
        self.assertFalse(torch.allclose(path.learnable_lut.weight, initial, atol=1e-6))


class TestLUTCheckpointSaving(unittest.TestCase):
    """Test LUT state_dict saving/loading."""

    def test_lut_state_dict(self):
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=3,
        )
        lut_state = path.learnable_lut.state_dict()
        self.assertIn("weight", lut_state)

    def test_lut_load_state_dict(self):
        path1 = MetricInducedGibbsProbPath(
            vocab_size=128,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=2,
        )
        with torch.no_grad():
            path1.learnable_lut.weight.fill_(0.5)
        state = path1.learnable_lut.state_dict()
        path2 = MetricInducedGibbsProbPath(
            vocab_size=128,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=2,
        )
        path2.learnable_lut.load_state_dict(state)
        self.assertTrue(torch.allclose(path1.learnable_lut.weight, path2.learnable_lut.weight))

    def test_checkpoint_contains_path_state(self):
        path = MetricInducedGibbsProbPath(
            vocab_size=64,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=1,
        )
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=1, factor=1.0)

        class _DummyScaler:
            def state_dict(self):
                return {}

            def load_state_dict(self, _state):
                pass

        scaler = _DummyScaler()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_dir=tmp, resume="")
            save_model(
                args=args,
                epoch=0,
                model=model,
                model_without_ddp=model,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                loss_scaler=scaler,
                path=path,
                extra_modules={},
            )
            ckpt_path = Path(tmp) / "checkpoint-0.pth"
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.assertIn("path", checkpoint)
            self.assertTrue(any(key.startswith("learnable_lut.") for key in checkpoint["path"].keys()))


class TestLUTMutualExclusion(unittest.TestCase):
    """Test mutual exclusion between LUT and learnable metric."""

    def test_cannot_enable_both_lut_and_metric(self):
        with self.assertRaisesRegex(ValueError, "Cannot enable both"):
            MetricInducedGibbsProbPath(
                vocab_size=256,
                metric="lp",
                learnable_lut=True,
                learnable_metric_dim=16,
            )

    def test_lut_only(self):
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            learnable_lut=True,
        )
        self.assertIsNotNone(path.learnable_lut)
        self.assertIsNone(path.learnable_metric)

    def test_metric_only(self):
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            learnable_metric_dim=16,
        )
        self.assertIsNone(path.learnable_lut)
        self.assertIsNotNone(path.learnable_metric)

    def test_neither_enabled(self):
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
        )
        self.assertIsNone(path.learnable_lut)
        self.assertIsNone(path.learnable_metric)


if __name__ == "__main__":
    unittest.main()
