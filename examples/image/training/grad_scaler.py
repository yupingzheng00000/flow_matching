# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import torch

from torch import Tensor
from typing import Any, Callable, Optional


def _create_grad_scaler(enabled: bool):
    """Create a GradScaler using torch.amp if available, else fall back to torch.cuda.amp."""
    try:
        from torch import amp as _amp  # type: ignore
        return _amp.GradScaler("cuda", enabled=enabled)  # type: ignore[arg-type]
    except Exception:
        from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
        return _GradScaler(enabled=enabled)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> Tensor:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if getattr(p, "grad", None) is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return Tensor(0.0)
    device = parameters[0].grad.device  # type: ignore[union-attr]
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)  # type: ignore[union-attr]
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]  # type: ignore[union-attr]
            ),
            norm_type,
        )
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled: bool = True):
        # When training in bf16 (autocast only), GradScaler should be disabled
        self._enabled = enabled
        self._scaler = _create_grad_scaler(enabled)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
        pre_step_fn: Optional[Callable[[], None]] = None,
    ):
        if self._enabled:
            self._scaler.scale(loss).backward(create_graph=create_graph)
        else:
            loss.backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                if self._enabled:
                    self._scaler.unscale_(optimizer)  # unscale grads in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                if self._enabled:
                    self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            if pre_step_fn is not None:
                pre_step_fn()
            optimizer.step() if not self._enabled else self._scaler.step(optimizer)
            if self._enabled:
                self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict() if self._enabled else {}

    def load_state_dict(self, state_dict):
        if self._enabled:
            self._scaler.load_state_dict(state_dict)

#python train.py --dataset=cifar10 --discrete_flow_matching --cfg_scale=0.0 --metric_induced --test_run