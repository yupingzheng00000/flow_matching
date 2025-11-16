# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for LUT geometry diagnostics used during training."""

import types
import unittest
import sys
from pathlib import Path
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
examples_image_root = _PROJECT_ROOT / "examples" / "image"
if str(examples_image_root) not in sys.path:
    sys.path.insert(0, str(examples_image_root))

# Provide a minimal stub for models.ema so train_loop imports succeed.
if "models" not in sys.modules:
    models_module = types.ModuleType("models")
    ema_module = types.ModuleType("models.ema")

    class _StubEMA:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
            pass

        def update(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
            pass

    ema_module.EMA = _StubEMA
    models_module.ema = ema_module
    sys.modules["models"] = models_module
    sys.modules["models.ema"] = ema_module

from examples.image.training.train_loop import _compute_lut_regularizer_and_metrics
from flow_matching.path.mixture import LearnableScalarLUT


class _DummyPath:
    def __init__(self, lut: torch.nn.Module) -> None:
        self.learnable_lut = lut


class TestLUTRegularizerMetrics(unittest.TestCase):
    """Tests ensuring LUT diagnostics operate on forward outputs and scalars."""

    def test_metrics_use_forward_scaled_weights(self) -> None:
        """Bounded residual scaling should influence the logged metrics."""

        lut = LearnableScalarLUT(
            num_channels=1,
            vocab_size=4,
            emb_dim=1,
            embed_range="pm1",
            device=None,
            dtype=torch.float32,
            bounded_residual_scale=True,
            scale_baseline=2.0,
            scale_epsilon=1.0,
            init_method="linear",
            init_noise_scale=0.0,
        )

        base_values = torch.tensor([[0.0, 1.0, -0.5, 0.0]], dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            lut.weight.copy_(base_values)
            lut.scale_c.fill_(4.0)

        path = _DummyPath(lut)
        device = torch.device("cpu")
        penalty, metrics = _compute_lut_regularizer_and_metrics(
            path=path,
            device=device,
            reg_align=1.0,
            reg_step=0.0,
            reg_curv=0.0,
            compute_metrics=True,
        )

        # Baseline (unscaled) negative slope magnitude.
        base_scalar = base_values.squeeze(-1)
        base_delta = base_scalar[:, 1:] - base_scalar[:, :-1]
        base_align = torch.clamp(-base_delta, min=0.0).mean().item()

        scale_factor = float(
            lut.scale_baseline * (1.0 + lut.scale_epsilon * torch.tanh(lut.scale_c).item())
        )
        expected_align = base_align * scale_factor

        self.assertAlmostEqual(metrics["lut_align"], expected_align, places=6)
        self.assertNotAlmostEqual(metrics["lut_align"], base_align)
        self.assertAlmostEqual(penalty.item(), expected_align, places=6)


if __name__ == "__main__":
    unittest.main()
