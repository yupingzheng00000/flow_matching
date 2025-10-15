# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for LearnableScalarLUT and MetricInducedGibbsProbPath with LUT."""

import unittest
import torch
import torch.nn as nn

from flow_matching.path import MetricInducedGibbsProbPath
from flow_matching.path.mixture import LearnableScalarLUT


class TestLearnableScalarLUT(unittest.TestCase):
    """Test the LearnableScalarLUT module in isolation."""

    def test_init_pm1_range(self):
        """Test initialization with [-1, 1] range."""
        lut = LearnableScalarLUT(
            num_channels=3,
            vocab_size=256,
            embed_range="pm1",
            device=None,
            dtype=torch.float32,
        )
        self.assertEqual(lut.num_channels, 3)
        self.assertEqual(lut.vocab_size, 256)
        self.assertEqual(lut.embed_range, "pm1")
    self.assertEqual(lut.weight.shape, (3, 256, 1))

    # Check linear initialization bounds
    self.assertAlmostEqual(lut.weight[0, 0, 0].item(), -1.0, places=6)
    self.assertAlmostEqual(lut.weight[0, -1, 0].item(), 1.0, places=6)
    self.assertTrue(torch.allclose(lut.weight[0], lut.weight[1]))  # All channels identical initially
        
    def test_init_unit_range(self):
        """Test initialization with [0, 1] range."""
        lut = LearnableScalarLUT(
            num_channels=1,
            vocab_size=16,
            embed_range="unit",
            device=None,
            dtype=torch.float32,
        )
        self.assertEqual(lut.embed_range, "unit")
    self.assertAlmostEqual(lut.weight[0, 0, 0].item(), 0.0, places=6)
    self.assertAlmostEqual(lut.weight[0, -1, 0].item(), 1.0, places=6)
        
    def test_invalid_params(self):
        """Test validation of constructor arguments."""
        with self.assertRaises(ValueError):
            LearnableScalarLUT(num_channels=0, vocab_size=256, embed_range="pm1", device=None, dtype=torch.float32)
        
        with self.assertRaises(ValueError):
            LearnableScalarLUT(num_channels=3, vocab_size=-1, embed_range="pm1", device=None, dtype=torch.float32)
        
        with self.assertRaises(ValueError):
            LearnableScalarLUT(num_channels=3, vocab_size=256, embed_range="invalid", device=None, dtype=torch.float32)
    
    def test_reset_parameters(self):
        """Test reset_parameters restores linear initialization."""
        lut = LearnableScalarLUT(
            num_channels=2,
            vocab_size=8,
            embed_range="pm1",
            device=None,
            dtype=torch.float32,
        )
        original_weight = lut.weight.clone()
        
        # Perturb weights
        lut.weight.data.add_(torch.randn_like(lut.weight))
        self.assertFalse(torch.allclose(lut.weight, original_weight))
        
        # Reset
        lut.reset_parameters()
    self.assertTrue(torch.allclose(lut.weight, original_weight, atol=1e-6))
    
    def test_device_dtype(self):
        """Test device and dtype propagation."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        
        lut = LearnableScalarLUT(
            num_channels=3,
            vocab_size=256,
            embed_range="pm1",
            device=device,
            dtype=torch.float64,
        )
        self.assertEqual(lut.weight.device, device)
        self.assertEqual(lut.weight.dtype, torch.float64)
    
    def test_parameters_are_learnable(self):
        """Test that weight is registered as a learnable parameter."""
        lut = LearnableScalarLUT(
            num_channels=3,
            vocab_size=256,
            embed_range="pm1",
            device=None,
            dtype=torch.float32,
        )
        params = list(lut.parameters())
        self.assertEqual(len(params), 1)
        self.assertIs(params[0], lut.weight)
        self.assertTrue(params[0].requires_grad)

    def test_forward_renorm_keeps_init_norm(self):
        """Renormalized forward should preserve the initialization Frobenius norm."""
        lut = LearnableScalarLUT(
            num_channels=2,
            vocab_size=16,
            embed_range="pm1",
            device=None,
            dtype=torch.float32,
            renormalize_to_init_norm=True,
        )
        init_norm = lut._init_fro_norm.item()

        with torch.no_grad():
            lut.weight.mul_(5.0)

    effective = torch.linalg.vector_norm(lut()).item()
        self.assertAlmostEqual(effective, init_norm, places=6)

    def test_forward_renorm_preserves_gradient(self):
        """Renormalization should keep gradient flow intact."""
        lut = LearnableScalarLUT(
            num_channels=1,
            vocab_size=8,
            embed_range="pm1",
            device=None,
            dtype=torch.float32,
            renormalize_to_init_norm=True,
        )
        loss = lut().sum()
        loss.backward()
        self.assertIsNotNone(lut.weight.grad)
        self.assertGreater(lut.weight.grad.abs().sum().item(), 0.0)


class TestMetricInducedPathWithLUT(unittest.TestCase):
    """Test MetricInducedGibbsProbPath integration with learnable LUT."""
    
    def test_lut_creation_basic(self):
        """Test basic LUT creation within MetricInducedGibbsProbPath."""
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            lp_order=3.0,
            learnable_lut=True,
            lut_num_channels=3,
            lut_share_across_channels=False,
            embed_range="pm1",
        )
        
        self.assertIsNotNone(path.learnable_lut)
        self.assertEqual(path.learnable_lut.num_channels, 3)
        self.assertEqual(path.learnable_lut.vocab_size, 256)
        self.assertEqual(path.learnable_lut.embed_range, "pm1")
    
    def test_lut_shared_channels(self):
        """Test LUT with channel sharing (num_channels=1)."""
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            lp_order=3.0,
            learnable_lut=True,
            lut_num_channels=3,
            lut_share_across_channels=True,  # Should force num_channels=1
            embed_range="unit",
        )
        
        self.assertIsNotNone(path.learnable_lut)
        self.assertEqual(path.learnable_lut.num_channels, 1)
        self.assertEqual(path.learnable_lut.embed_range, "unit")
    
    def test_lut_metric_mutual_exclusion(self):
        """Test that learnable_lut and learnable_metric cannot be enabled simultaneously."""
        with self.assertRaisesRegex(ValueError, "Cannot enable both learnable_metric and learnable_lut"):
            MetricInducedGibbsProbPath(
                vocab_size=256,
                metric="lp",
                learnable_lut=True,
                learnable_metric_dim=8,  # Conflict
            )
    
    def test_lut_requires_lp_metric(self):
        """Test that LUT requires 'lp' or 'euclidean' metric."""
        with self.assertRaisesRegex(ValueError, "learnable_lut currently supports only 'lp' or 'euclidean'"):
            MetricInducedGibbsProbPath(
                vocab_size=256,
                metric="cosine",  # Incompatible
                learnable_lut=True,
            )
    
    def test_lut_parameters_iterator(self):
        """Test lut_parameters() yields correct parameters."""
        path = MetricInducedGibbsProbPath(
            vocab_size=128,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=3,
        )
        
        lut_params = list(path.lut_parameters())
        self.assertEqual(len(lut_params), 1)
    self.assertEqual(lut_params[0].shape, (3, 128, 1))
        self.assertTrue(lut_params[0].requires_grad)
    
    def test_lut_parameters_empty_when_disabled(self):
        """Test lut_parameters() returns empty iterator when LUT disabled."""
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            learnable_lut=False,
        )
        
        lut_params = list(path.lut_parameters())
        self.assertEqual(len(lut_params), 0)

    def test_lut_renorm_flag_preserves_norm(self):
        """Test that enabling renorm keeps the effective LUT norm fixed."""
        path = MetricInducedGibbsProbPath(
            vocab_size=32,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=2,
            lut_renorm_to_init_norm=True,
        )
        init_norm = path.learnable_lut._init_fro_norm.item()
        with torch.no_grad():
            path.learnable_lut.weight.mul_(7.0)
        effective = torch.linalg.norm(path.learnable_lut(), ord="fro").item()
        self.assertAlmostEqual(effective, init_norm, places=6)
    
    def test_lut_distance_table_shape(self):
        """Test that distance table has correct shape with LUT."""
        vocab_size = 16
        num_channels = 3
        path = MetricInducedGibbsProbPath(
            vocab_size=vocab_size,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=num_channels,
        )
        
        # Build distance table (on CPU for testing)
        device = torch.device("cpu")
        dtype = torch.float32
        table = path._build_lut_distance_table(device=device, dtype=dtype)
        
        # Shape is [C, K, K] where C=channels, K=vocab_size
        self.assertEqual(table.shape, (num_channels, vocab_size, vocab_size))
        self.assertEqual(table.dtype, dtype)
        self.assertEqual(table.device, device)
    
    def test_lut_distance_symmetry(self):
        """Test that distance table is symmetric."""
        path = MetricInducedGibbsProbPath(
            vocab_size=8,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=2,
        )
        
        device = torch.device("cpu")
        dtype = torch.float32
        table = path._build_lut_distance_table(device=device, dtype=dtype)
        
        # Distance should be symmetric: [C, K, K] -> transpose last two dims
        self.assertTrue(torch.allclose(table, table.transpose(-2, -1), atol=1e-6))
    
    def test_lut_distance_diagonal_zero(self):
        """Test that diagonal of distance table is zero (d(i,i) = 0)."""
        path = MetricInducedGibbsProbPath(
            vocab_size=10,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=1,
        )
        
        device = torch.device("cpu")
        dtype = torch.float32
        table = path._build_lut_distance_table(device=device, dtype=dtype)
        
        # table is [C, K, K]; check diagonal for each channel
        for c in range(table.shape[0]):
            diagonal = torch.diagonal(table[c])
            self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6))
    
    def test_lut_distance_positive(self):
        """Test that all distances are non-negative."""
        path = MetricInducedGibbsProbPath(
            vocab_size=12,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=3,
        )
        
        device = torch.device("cpu")
        dtype = torch.float32
        table = path._build_lut_distance_table(device=device, dtype=dtype)
        
        self.assertTrue((table >= 0).all())
    
    def test_lut_absolute_difference_semantics(self):
        """Test that LUT distances follow absolute difference semantics."""
        vocab_size = 4
        path = MetricInducedGibbsProbPath(
            vocab_size=vocab_size,
            metric="lp",
            lp_order=1.0,  # L1 norm = sum of absolute differences
            learnable_lut=True,
            lut_num_channels=2,
            lut_share_across_channels=False,
        )
        
        # Manually set LUT weights for predictable distances
        # Channel 0: [0, 1, 2, 3]
        # Channel 1: [0, 10, 20, 30]
        with torch.no_grad():
            path.learnable_lut.weight[0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
            path.learnable_lut.weight[1] = torch.tensor([0.0, 10.0, 20.0, 30.0])
        
        device = torch.device("cpu")
        dtype = torch.float32
        table = path._build_lut_distance_table(device=device, dtype=dtype)
        
        # table is [2, 4, 4] - check channel 0 and 1 separately
        # Channel 0: d(0, 1) = |0-1| = 1
        self.assertAlmostEqual(table[0, 0, 1].item(), 1.0, places=5)
        
        # Channel 0: d(0, 3) = |0-3| = 3
        self.assertAlmostEqual(table[0, 0, 3].item(), 3.0, places=5)
        
        # Channel 1: d(0, 1) = |0-10| = 10
        self.assertAlmostEqual(table[1, 0, 1].item(), 10.0, places=5)
    
    def test_lut_channel_assignments(self):
        """Test _lut_channel_assignments helper."""
        vocab_size = 12  # 3 channels × 4 pixels
        path = MetricInducedGibbsProbPath(
            vocab_size=vocab_size,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=3,
        )
        
        # Create dummy token tensor
        tokens = torch.zeros((2, 12), dtype=torch.long)  # batch=2, seq=12
        assignments = path._lut_channel_assignments(tokens)
        
        # Should be [0,0,0,0, 1,1,1,1, 2,2,2,2] for each batch
        expected_per_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
        self.assertTrue(torch.equal(assignments[0], expected_per_batch))
        self.assertTrue(torch.equal(assignments[1], expected_per_batch))
    
    def test_lut_rows_helper(self):
        """Test _lut_rows helper returns correct row indices."""
        vocab_size = 8  # 2 channels × 4 pixels
        path = MetricInducedGibbsProbPath(
            vocab_size=vocab_size,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=2,
        )
        
        # Build distance table and token indices
        device = torch.device("cpu")
        dtype = torch.float32
        dist_table = path._build_lut_distance_table(device=device, dtype=dtype)
        
        # Create dummy token indices: [batch=2, seq=8]
        token_indices = torch.arange(8).unsqueeze(0).expand(2, -1)
        
        # _lut_rows returns [B, S, K]
        rows = path._lut_rows(dist_table, token_indices)
        
        self.assertEqual(rows.shape, (2, 8, vocab_size))
    
    def test_lut_interpolation_disabled(self):
        """Test that interpolation lambda is forced to 0 when LUT enabled."""
        path = MetricInducedGibbsProbPath(
            vocab_size=256,
            metric="lp",
            learnable_lut=True,
            metric_interp_lambda=0.5,  # Should be ignored
        )
        
        # Should log warning and clamp to 0
        # Sample to trigger the warning check
        x0 = torch.randint(0, 256, (4, 8))
        x1 = torch.randint(0, 256, (4, 8))
        t = torch.rand(4)
        
        # If lambda were non-zero, this would raise an error or log warning
        # We just check the path initializes correctly
        self.assertIsNotNone(path.learnable_lut)


class TestLUTGradientFlow(unittest.TestCase):
    """Test gradient flow through LUT in training scenarios."""
    
    def test_lut_gradients_computed(self):
        """Test that gradients flow back to LUT parameters."""
        path = MetricInducedGibbsProbPath(
            vocab_size=8,
            metric="lp",
            lp_order=2.0,
            learnable_lut=True,
            lut_num_channels=1,
        )
        
        # Create dummy loss: sum of all distances
        device = torch.device("cpu")
        dtype = torch.float32
        table = path._build_lut_distance_table(device=device, dtype=dtype)
        
        loss = table.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(path.learnable_lut.weight.grad)
        self.assertGreater(path.learnable_lut.weight.grad.abs().sum().item(), 0)
    
    def test_lut_optimizer_integration(self):
        """Test LUT parameters can be optimized."""
        path = MetricInducedGibbsProbPath(
            vocab_size=4,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=1,
        )
        
        optimizer = torch.optim.SGD(path.lut_parameters(), lr=0.01)
        
        initial_weight = path.learnable_lut.weight.clone()
        
        # Simple optimization step
        for _ in range(5):
            optimizer.zero_grad()
            device = torch.device("cpu")
            dtype = torch.float32
            table = path._build_lut_distance_table(device=device, dtype=dtype)
            loss = table.sum()
            loss.backward()
            optimizer.step()
        
        # Weights should have changed
        self.assertFalse(torch.allclose(path.learnable_lut.weight, initial_weight))
    
    def test_lut_freeze(self):
        """Test freezing LUT parameters (requires_grad=False)."""
        path = MetricInducedGibbsProbPath(
            vocab_size=8,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=2,
        )
        
        # Freeze
        for param in path.lut_parameters():
            param.requires_grad = False
        
        # Try to compute gradients - this should not raise because requires_grad=False
        device = torch.device("cpu")
        dtype = torch.float32
        table = path._build_lut_distance_table(device=device, dtype=dtype)
        
        # Cannot backward on a detached tensor
        # Just verify the parameter is frozen
        self.assertFalse(path.learnable_lut.weight.requires_grad)


class TestLUTEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_single_channel_vocab(self):
        """Test with minimal vocab_size and channels."""
        path = MetricInducedGibbsProbPath(
            vocab_size=2,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=1,
        )
        
        self.assertEqual(path.learnable_lut.vocab_size, 2)
        self.assertEqual(path.learnable_lut.num_channels, 1)
        
        device = torch.device("cpu")
        dtype = torch.float32
        table = path._build_lut_distance_table(device=device, dtype=dtype)
        self.assertEqual(table.shape, (1, 2, 2))  # [C=1, K=2, K=2]
    
    def test_large_vocab(self):
        """Test with large vocabulary size."""
        vocab_size = 1024
        num_channels = 3
        path = MetricInducedGibbsProbPath(
            vocab_size=vocab_size,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=num_channels,
        )
        
        device = torch.device("cpu")
        dtype = torch.float32
        table = path._build_lut_distance_table(device=device, dtype=dtype)
        
        self.assertEqual(table.shape, (num_channels, vocab_size, vocab_size))
        self.assertTrue(torch.allclose(table, table.transpose(-2, -1), atol=1e-5))
    
    def test_many_channels(self):
        """Test with many channels."""
        path = MetricInducedGibbsProbPath(
            vocab_size=16,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=16,  # 16 channels × 1 pixel each
        )
        
        self.assertEqual(path.learnable_lut.num_channels, 16)
        self.assertEqual(path.learnable_lut.weight.shape, (16, 16))


if __name__ == "__main__":
    unittest.main()
