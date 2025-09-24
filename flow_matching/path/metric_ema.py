# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class LearnableMetricEMA(nn.Module):
    """EMA teacher for learnable metric modules used in metric-induced paths."""

    def __init__(self, metric_module: nn.Module, decay: float = 0.999) -> None:
        super().__init__()
        if not isinstance(metric_module, nn.Module):
            raise TypeError("metric_module must be an nn.Module")
        if decay <= 0.0 or decay > 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = float(decay)

        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))

        self.teacher = copy.deepcopy(metric_module)
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def synchronize_from(self, metric_module: nn.Module) -> None:
        self.teacher.load_state_dict(metric_module.state_dict())
        self.num_updates.zero_()

    @torch.no_grad()
    def update(self, metric_module: nn.Module) -> None:
        self.num_updates += 1
        num_updates = int(self.num_updates.item())
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        one_minus_decay = 1.0 - decay

        teacher_state = dict(self.teacher.named_parameters())
        module_state = dict(metric_module.named_parameters())
        for name, param in module_state.items():
            if name not in teacher_state:
                continue
            teacher_param = teacher_state[name]
            teacher_param.add_(one_minus_decay * (param.detach() - teacher_param))

        teacher_buffers = dict(self.teacher.named_buffers())
        module_buffers = dict(metric_module.named_buffers())
        for name, buffer in module_buffers.items():
            if name not in teacher_buffers:
                continue
            teacher_buffers[name].add_(one_minus_decay * (buffer.detach() - teacher_buffers[name]))
