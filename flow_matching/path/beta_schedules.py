# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BetaSchedule(nn.Module):
    """Abstract interface for time-dependent schedules used by discrete paths."""

    def beta_and_derivative(self, t: Tensor) -> Tuple[Tensor, Tensor]:  # pragma: no cover - interface
        raise NotImplementedError


def _normalize_bin_sizes(
    unnormalized: Tensor, min_size: float, total_size: float
) -> Tensor:
    num_bins = unnormalized.shape[-1]
    softmax = F.softmax(unnormalized, dim=-1)
    return softmax * (total_size - min_size * num_bins) + min_size


def _rational_quadratic_spline(
    inputs: Tensor,
    widths: Tensor,
    heights: Tensor,
    derivatives: Tensor,
    left: float,
    right: float,
    bottom: float,
    top: float,
) -> Tensor:
    """One-dimensional monotone rational quadratic spline with linear tails."""

    inputs_flat = inputs.reshape(-1)
    outputs_flat = torch.empty_like(inputs_flat)

    cumwidths = torch.cumsum(widths, dim=-1)
    cumheights = torch.cumsum(heights, dim=-1)

    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)

    cumwidths = cumwidths + left
    cumheights = cumheights + bottom

    # Tails
    left_mask = inputs_flat <= left
    right_mask = inputs_flat >= right
    center_mask = (~left_mask) & (~right_mask)

    if left_mask.any():
        slope = derivatives[..., 0]
        outputs_flat[left_mask] = bottom + (inputs_flat[left_mask] - left) * slope

    if right_mask.any():
        slope = derivatives[..., -1]
        outputs_flat[right_mask] = top + (inputs_flat[right_mask] - right) * slope

    if center_mask.any():
        inside_x = inputs_flat[center_mask]
        bin_idx = torch.searchsorted(cumwidths, inside_x, right=True) - 1
        bin_idx = bin_idx.clamp(min=0, max=widths.shape[-1] - 1)

        input_cumwidth = cumwidths.index_select(dim=-1, index=bin_idx)
        input_cumheight = cumheights.index_select(dim=-1, index=bin_idx)

        w = widths.index_select(dim=-1, index=bin_idx)
        h = heights.index_select(dim=-1, index=bin_idx)
        s = (inside_x - input_cumwidth) / w

        delta = h / w
        d0 = derivatives.index_select(dim=-1, index=bin_idx)
        d1 = derivatives.index_select(dim=-1, index=bin_idx + 1)

        numerator = h * (delta * s**2 + d0 * s * (1 - s))
        denominator = delta + (d0 + d1 - 2 * delta) * s * (1 - s)
        outputs_flat[center_mask] = input_cumheight + numerator / denominator

    return outputs_flat.view_as(inputs)


@dataclass
class MonotoneRQConfig:
    num_bins: int = 8
    tail_bound: float = 6.0
    beta_min: float = 0.0
    beta_max: float = 20.0
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3
    t_eps: float = 1e-4
    logit_eps: float = 1e-6


class MonotoneRQBetaSchedule(BetaSchedule):
    """Logit-time monotone rational quadratic spline schedule for β(t)."""

    def __init__(self, config: MonotoneRQConfig | None = None):
        super().__init__()
        self.config = config or MonotoneRQConfig()
        self.num_bins = self.config.num_bins
        assert self.num_bins >= 2, "num_bins must be >= 2"
        assert self.config.beta_max > self.config.beta_min, "beta_max must exceed beta_min"

        self.unnormalized_widths = nn.Parameter(torch.zeros(self.num_bins))
        self.unnormalized_heights = nn.Parameter(torch.zeros(self.num_bins))
        self.unnormalized_derivatives = nn.Parameter(torch.zeros(self.num_bins + 1))

    def _spline(self, s: Tensor) -> Tensor:
        cfg = self.config
        widths = _normalize_bin_sizes(
            self.unnormalized_widths,
            cfg.min_bin_width,
            cfg.tail_bound * 2.0,
        )
        heights = _normalize_bin_sizes(
            self.unnormalized_heights,
            cfg.min_bin_height,
            cfg.tail_bound * 2.0,
        )
        derivatives = F.softplus(self.unnormalized_derivatives) + cfg.min_derivative
        return _rational_quadratic_spline(
            inputs=s,
            widths=widths,
            heights=heights,
            derivatives=derivatives,
            left=-cfg.tail_bound,
            right=cfg.tail_bound,
            bottom=-cfg.tail_bound,
            top=cfg.tail_bound,
        )

    def _beta_from_t(self, t: Tensor) -> Tensor:
        cfg = self.config
        s = torch.logit(t, eps=cfg.logit_eps)
        spline_val = self._spline(s)
        sigma = torch.sigmoid(spline_val)
        return cfg.beta_min + (cfg.beta_max - cfg.beta_min) * sigma

    def beta_and_derivative(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        cfg = self.config
        t_clamped = t.clamp(min=cfg.t_eps, max=1.0 - cfg.t_eps)
        grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            t_req = t_clamped.detach().requires_grad_(True)
            beta = self._beta_from_t(t_req)
            ones = torch.ones_like(beta)
            d_beta = torch.autograd.grad(
                beta,
                t_req,
                grad_outputs=ones,
                create_graph=grad_enabled,
            )[0]
        if not grad_enabled:
            beta = beta.detach()
            d_beta = d_beta.detach()
        beta = beta.view_as(t_clamped)
        d_beta = d_beta.view_as(t_clamped)
        return beta, d_beta

