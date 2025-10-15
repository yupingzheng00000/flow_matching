# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for LUT training pipeline."""

import unittest
import sys
import tempfile
import torch
from pathlib import Path

# Add examples/image to path for imports
_test_dir = Path(__file__).resolve().parent
_examples_image = _test_dir.parent / "examples" / "image"
if str(_examples_image) not in sys.path:
    sys.path.insert(0, str(_examples_image))

from train_arg_parser import get_args_parser
from flow_matching.path import MetricInducedGibbsProbPath


class TestLUTArgumentParsing(unittest.TestCase):
    """Test CLI argument parsing for LUT options."""
    
    def test_lut_default_disabled(self):
        """Test LUT is disabled by default."""
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--output_dir", "/tmp",
        ])
        
        self.assertFalse(getattr(args, "mi_learnable_lut", False))
    
    def test_lut_enable_flag(self):
        """Test --mi_learnable_lut flag."""
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--output_dir", "/tmp",
            "--mi_learnable_lut",
        ])
        
        self.assertTrue(args.mi_learnable_lut)
    
    def test_lut_num_channels(self):
        """Test --mi_lut_num_channels argument."""
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--output_dir", "/tmp",
            "--mi_learnable_lut",
            "--mi_lut_num_channels", "1",
        ])
        
        self.assertEqual(args.mi_lut_num_channels, 1)
    
    def test_lut_share_channels(self):
        """Test --mi_lut_share_channels flag."""
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--output_dir", "/tmp",
            "--mi_learnable_lut",
            "--mi_lut_share_channels",
        ])
        
        self.assertTrue(getattr(args, "mi_lut_share_channels", False))
    
    def test_lut_freeze_flag(self):
        """Test --mi_freeze_lut flag."""
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--output_dir", "/tmp",
            "--mi_learnable_lut",
            "--mi_freeze_lut",
        ])
        
        self.assertTrue(args.mi_freeze_lut)
    
    def test_lut_lr_scale(self):
        """Test --mi_lut_lr_scale argument."""
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--output_dir", "/tmp",
            "--mi_learnable_lut",
            "--mi_lut_lr_scale", "0.05",
        ])
        
        self.assertAlmostEqual(args.mi_lut_lr_scale, 0.05)
    
    def test_lut_weight_decay(self):
        """Test --mi_lut_weight_decay argument."""
        parser = get_args_parser()
        args = parser.parse_args([
            "--data_path", "/tmp",
            "--output_dir", "/tmp",
            "--mi_learnable_lut",
            "--mi_lut_weight_decay", "1e-3",
        ])
        
        self.assertAlmostEqual(args.mi_lut_weight_decay, 1e-3)


class TestLUTOptimizerIntegration(unittest.TestCase):
    """Test LUT parameter groups in optimizer."""
    
    def test_lut_parameters_in_optimizer(self):
        """Test LUT parameters are added to optimizer."""
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=3,
        )
        
        # Collect LUT parameters
        lut_params = list(path.lut_parameters())
        self.assertEqual(len(lut_params), 1)
        
        # Create optimizer with separate groups
        model_params = [torch.nn.Parameter(torch.randn(10, 10))]
        
        optimizer = torch.optim.AdamW([
            {"params": model_params, "lr": 1e-4, "name": "model"},
            {"params": lut_params, "lr": 1e-5, "name": "lut"},
        ])
        
        # Check optimizer has 2 param groups
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["name"], "model")
        self.assertEqual(optimizer.param_groups[1]["name"], "lut")
        self.assertEqual(optimizer.param_groups[1]["lr"], 1e-5)
    
    def test_frozen_lut_excluded_from_optimizer(self):
        """Test frozen LUT parameters are not in optimizer."""
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=3,
        )
        
        # Freeze LUT
        for param in path.lut_parameters():
            param.requires_grad = False
        
        # Filter parameters by requires_grad
        lut_params_trainable = [p for p in path.lut_parameters() if p.requires_grad]
        self.assertEqual(len(lut_params_trainable), 0)
        
        # Optimizer should skip frozen params
        model_params = [torch.nn.Parameter(torch.randn(10, 10))]
        
        optimizer = torch.optim.AdamW([
            {"params": model_params, "lr": 1e-4},
        ])
        
        # Only model params in optimizer
        self.assertEqual(len(optimizer.param_groups), 1)
    
    def test_lut_gradient_step(self):
        """Test LUT parameters can be optimized."""
        path = MetricInducedGibbsProbPath(
            vocab_size=16,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=1,
        )
        
        lut_params = list(path.lut_parameters())
        optimizer = torch.optim.SGD(lut_params, lr=0.1)
        
        initial_weight = path.learnable_lut.weight.clone()
        
        # Perform optimization step
        for _ in range(3):
            optimizer.zero_grad()
            table = path._build_lut_distance_table(
                device=torch.device("cpu"),
                dtype=torch.float32
            )
            loss = table.sum()
            loss.backward()
            optimizer.step()
        
        # Weights should have changed
        self.assertFalse(torch.allclose(path.learnable_lut.weight, initial_weight, atol=1e-6))


class TestLUTCheckpointSaving(unittest.TestCase):
    """Test LUT state_dict saving/loading."""
    
    def test_lut_state_dict(self):
        """Test LUT parameters appear in state_dict."""
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=3,
        )
        
        # LUT module should have state_dict
        lut_state = path.learnable_lut.state_dict()
        self.assertIn("weight", lut_state)
        self.assertEqual(lut_state["weight"].shape, (3, 256))
    
    def test_lut_load_state_dict(self):
        """Test loading LUT from state_dict."""
        path1 = MetricInducedGibbsProbPath(
            vocab_size=128,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=2,
        )
        
        # Modify LUT weights
        with torch.no_grad():
            path1.learnable_lut.weight.fill_(0.5)
        
        # Save state
        state = path1.learnable_lut.state_dict()
        
        # Create new path and load
        path2 = MetricInducedGibbsProbPath(
            vocab_size=128,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=2,
        )
        
        path2.learnable_lut.load_state_dict(state)
        
        # Weights should match
        self.assertTrue(torch.allclose(
            path1.learnable_lut.weight,
            path2.learnable_lut.weight
        ))


class TestLUTMutualExclusion(unittest.TestCase):
    """Test mutual exclusion between LUT and learnable metric."""
    
    def test_cannot_enable_both_lut_and_metric(self):
        """Test error when both LUT and metric are enabled."""
        with self.assertRaisesRegex(ValueError, "Cannot enable both"):
            MetricInducedGibbsProbPath(
                vocab_size=256,
                metric="lp",
                learnable_lut=True,
                learnable_metric_dim=16,
            )
    
    def test_lut_only(self):
        """Test LUT can be enabled alone."""
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            learnable_lut=True,
        )
        
        self.assertIsNotNone(path.learnable_lut)
        self.assertIsNone(path.learnable_metric)
    
    def test_metric_only(self):
        """Test metric can be enabled alone."""
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            learnable_metric_dim=16,
        )
        
        self.assertIsNone(path.learnable_lut)
        self.assertIsNotNone(path.learnable_metric)
    
    def test_neither_enabled(self):
        """Test neither LUT nor metric enabled."""
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
        )
        
        self.assertIsNone(path.learnable_lut)
        self.assertIsNone(path.learnable_metric)


if __name__ == "__main__":
    unittest.main()
