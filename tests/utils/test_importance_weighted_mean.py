import sys
import types

import torch

stub_models = types.ModuleType("models")
stub_models_ema = types.ModuleType("models.ema")
stub_training = types.ModuleType("training")
stub_training_grad_scaler = types.ModuleType("training.grad_scaler")
stub_training_distributed = types.ModuleType("training.distributed_mode")


class _StubEMA:  # pragma: no cover - simple placeholder
    def __init__(self, *args, **kwargs):
        raise RuntimeError("EMA should not be constructed during tests")


stub_models_ema.EMA = _StubEMA
stub_models.ema = stub_models_ema
sys.modules.setdefault("models", stub_models)
sys.modules.setdefault("models.ema", stub_models_ema)


class _StubScaler:  # pragma: no cover - placeholder
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Scaler should not be constructed during tests")


stub_training_grad_scaler.NativeScalerWithGradNormCount = _StubScaler
stub_training.grad_scaler = stub_training_grad_scaler
stub_training.distributed_mode = stub_training_distributed
stub_training_distributed.get_world_size = lambda: 1
stub_training_distributed.get_rank = lambda: 0
stub_training_distributed.is_main_process = lambda: True
stub_training_distributed.is_dist_avail_and_initialized = lambda: False
sys.modules.setdefault("training", stub_training)
sys.modules.setdefault("training.grad_scaler", stub_training_grad_scaler)
sys.modules.setdefault("training.distributed_mode", stub_training_distributed)

from examples.image.training.train_loop import _importance_weighted_mean


def test_importance_weighted_mean_without_weights():
    values = torch.tensor([1.0, 2.0, 3.0])
    result = _importance_weighted_mean(values, None)
    assert torch.isclose(result, values.mean())


def test_importance_weighted_mean_broadcasts_weights():
    values = torch.ones(2, 3)
    weights = torch.tensor([2.0, 0.5])
    expected = (values * weights.unsqueeze(-1)).mean()
    result = _importance_weighted_mean(values, weights)
    assert torch.isclose(result, expected)
