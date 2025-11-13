# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the unified LearnableLUT module."""

import unittest

import torch

from flow_matching.path.mixture import LearnableLUT, MetricInducedGibbsProbPath


class TestLearnableLUTFreeMode(unittest.TestCase):
    def _make_lut(self, **overrides) -> LearnableLUT:
        defaults = dict(
            num_channels=3,
            vocab_size=256,
            emb_dim=1,
            embed_range="pm1",
            param_mode="none",
            device=None,
            dtype=torch.float32,
        )
        defaults.update(overrides)
        return LearnableLUT(**defaults)

    def test_init_shapes_pm1(self) -> None:
        torch.manual_seed(0)
        lut = self._make_lut()
        self.assertEqual(lut.num_channels, 3)
        self.assertEqual(lut.vocab_size, 256)
        self.assertEqual(lut.emb_dim, 1)
        self.assertEqual(lut.embed_range, "pm1")
        self.assertTrue(isinstance(lut.weight, torch.Tensor))
        self.assertEqual(lut.weight.shape, (3, 256, 1))
        self.assertTrue(lut.weight.requires_grad)
        baseline = lut.baseline_weight()
        self.assertEqual(baseline.shape, lut.weight.shape)

    def test_initialize_from_weight(self) -> None:
        lut = self._make_lut(num_channels=2, vocab_size=8)
        target = torch.linspace(-1.0, 1.0, steps=8).view(1, 8, 1).repeat(2, 1, 1)
        lut.initialize_from_weight(target)
        self.assertTrue(torch.allclose(lut.weight, target))

    def test_forward_renorm_keeps_norm(self) -> None:
        torch.manual_seed(0)
        lut = self._make_lut(num_channels=2, vocab_size=16, renormalize_to_init_norm=True)
        baseline_norm = torch.linalg.vector_norm(lut.baseline_weight(), dim=(1, 2))
        with torch.no_grad():
            lut.weight.mul_(7.0)
        effective = torch.linalg.vector_norm(lut(), dim=(1, 2))
        self.assertTrue(torch.allclose(effective, baseline_norm, atol=1e-5))

    def test_reset_parameters_maintains_grad(self) -> None:
        lut = self._make_lut()
        with torch.no_grad():
            lut.weight.add_(torch.randn_like(lut.weight))
        lut.reset_parameters()
        self.assertTrue(lut.weight.requires_grad)
        self.assertEqual(lut.weight.shape, (3, 256, 1))

    def test_bounded_residual_scale(self) -> None:
        lut = self._make_lut(
            num_channels=1,
            vocab_size=4,
            bounded_residual_scale=True,
            scale_baseline=2.0,
            scale_epsilon=0.5,
        )
        self.assertIsNotNone(lut.scale_c)
        out = lut()
        self.assertEqual(out.shape, lut.weight.shape)


class TestLearnableLUTParametric(unittest.TestCase):
    def test_small_noise_qr_param_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            LearnableLUT(
                num_channels=1,
                vocab_size=16,
                emb_dim=2,
                embed_range="pm1",
                param_mode="line2d",
                init_method="small_noise_qr",
                device=None,
                dtype=torch.float32,
            )

    def test_line2d_output_is_monotone(self) -> None:
        lut = LearnableLUT(
            num_channels=2,
            vocab_size=32,
            emb_dim=3,
            embed_range="pm1",
            param_mode="line2d",
            device=None,
            dtype=torch.float32,
        )
        weight = lut().cpu()
        diffs = weight[:, 1:, 0] - weight[:, :-1, 0]
        self.assertTrue(torch.all(diffs > 0))

    def test_arc2d_output_shape(self) -> None:
        lut = LearnableLUT(
            num_channels=1,
            vocab_size=32,
            emb_dim=3,
            embed_range="pm1",
            param_mode="arc2d",
            arc_radius=1.3,
            device=None,
            dtype=torch.float32,
        )
        weight = lut()
        self.assertEqual(weight.shape, (1, 32, 3))
        norms = torch.linalg.vector_norm(weight[0], dim=-1)
        self.assertTrue(torch.allclose(norms, torch.full_like(norms, 1.3), atol=1e-2))


class TestMetricInducedPathIntegration(unittest.TestCase):
    def test_lut_parameters_grad_flow(self) -> None:
        path = MetricInducedGibbsProbPath(
            vocab_size=8,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=1,
        )
        params = list(path.lut_parameters())
        self.assertGreater(len(params), 0)
        loss = path._build_lut_distance_table(device=torch.device("cpu"), dtype=torch.float32).sum()
        loss.backward()
        grads = [p.grad for p in params]
        self.assertTrue(any(g is not None and torch.any(torch.ne(g, 0)) for g in grads))

    def test_distance_table_shape(self) -> None:
        path = MetricInducedGibbsProbPath(
            vocab_size=16,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=2,
        )
        table = path._build_lut_distance_table(device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(table.shape, (2, 16, 16))

    def test_baseline_weight_helper(self) -> None:
        lut = LearnableLUT(
            num_channels=2,
            vocab_size=10,
            emb_dim=2,
            embed_range="pm1",
            param_mode="none",
            device=None,
            dtype=torch.float32,
        )
        baseline = lut.baseline_weight()
        self.assertEqual(baseline.shape, (2, 10, 2))
        self.assertTrue(torch.all((baseline[..., 0] >= -1.0) & (baseline[..., 0] <= 1.0)))


class TestLUTEdgeCases(unittest.TestCase):
    def test_many_channels(self) -> None:
        path = MetricInducedGibbsProbPath(
            vocab_size=16,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=16,
        )
        weight = path.learnable_lut()
        self.assertEqual(weight.shape, (16, 16, path.learnable_lut.emb_dim))

    def test_small_vocab(self) -> None:
        path = MetricInducedGibbsProbPath(
            vocab_size=2,
            metric="lp",
            learnable_lut=True,
            lut_num_channels=1,
        )
        table = path._build_lut_distance_table(device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(table.shape, (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
