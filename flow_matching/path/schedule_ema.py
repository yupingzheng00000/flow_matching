# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .beta_schedules import BetaSchedule


class BetaScheduleEMA(nn.Module):
    """Exponential moving average teacher for a learnable beta schedule."""

    def __init__(self, schedule: BetaSchedule, decay: float = 0.999) -> None:
        super().__init__()
        if not isinstance(schedule, nn.Module):
            raise TypeError("schedule must be an nn.Module implementing BetaSchedule")
        if decay <= 0.0 or decay > 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = float(decay)

        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))

        # Clone the schedule parameters/buffers into the teacher module.
        self.teacher = copy.deepcopy(schedule)
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        # Buffers are non-trainable by definition; ensure teacher stays in eval mode.
        self.teacher.eval()

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.beta_and_derivative(t)

    def beta_and_derivative(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate the EMA teacher schedule."""

        return self.teacher.beta_and_derivative(t)

    @torch.no_grad()
    def synchronize_from(self, schedule: BetaSchedule) -> None:
        """Hard reset the teacher parameters to match the provided schedule."""

        self.teacher.load_state_dict(schedule.state_dict())
        self.num_updates.zero_()

    @torch.no_grad()
    def update(self, schedule: BetaSchedule) -> None:
        """Update the teacher parameters with an EMA step."""

        self.num_updates += 1
        num_updates = int(self.num_updates.item())
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        one_minus_decay = 1.0 - decay

        teacher_state = dict(self.teacher.named_parameters())
        schedule_state = dict(schedule.named_parameters())
        for name, param in schedule_state.items():
            if name not in teacher_state:
                continue
            teacher_param = teacher_state[name]
            teacher_param.add_(one_minus_decay * (param.detach() - teacher_param))

        # Buffers (e.g., running stats) should match the current schedule exactly.
        teacher_buffers = dict(self.teacher.named_buffers())
        schedule_buffers = dict(schedule.named_buffers())
        for name, buffer in schedule_buffers.items():
            if name not in teacher_buffers:
                continue
            teacher_buffers[name].add_(one_minus_decay * (buffer.detach() - teacher_buffers[name]))

