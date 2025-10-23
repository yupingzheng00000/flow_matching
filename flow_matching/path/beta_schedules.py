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


def _inv_softplus(x: Tensor) -> Tensor:
    """Stable inverse of ``softplus`` for positive targets."""

    return torch.log(torch.expm1(x))


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


@dataclass
class ExpMonotoneRQSConfig:
    num_bins: int = 16
    tail_bound: float = 6.0
    init_c: float = 1.0
    init_a: float = 5.0
    t_eps: float = 1e-4
    logit_eps: float = 1e-6


class _ExpRQS1D(nn.Module):
    """Monotone rational–quadratic spline on logit time with linear tails."""

    def __init__(self, num_bins: int, tail_bound: float):
        super().__init__()
        if num_bins < 1:
            raise ValueError("num_bins must be >= 1")
        if tail_bound <= 0:
            raise ValueError("tail_bound must be > 0")

        self.num_bins = int(num_bins)
        self.tail_bound = float(tail_bound)

        self.theta_w = nn.Parameter(torch.zeros(self.num_bins))
        self.theta_h = nn.Parameter(torch.zeros(self.num_bins))
        self.theta_d = nn.Parameter(torch.zeros(max(self.num_bins - 1, 0)))

        if self.num_bins > 1:
            with torch.no_grad():
                target = torch.tensor(1.0)
                self.theta_d.copy_(_inv_softplus(target).expand_as(self.theta_d))

    def _knot_tensors(
        self, *, dtype: torch.dtype, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        B = self.tail_bound
        widths = F.softmax(self.theta_w, dim=0) * (2.0 * B)
        heights = F.softmax(self.theta_h, dim=0) * (2.0 * B)
        eps = torch.finfo(widths.dtype).eps
        internal = F.softplus(self.theta_d) + eps

        cumsum_widths = torch.cumsum(widths, dim=0)
        cumsum_heights = torch.cumsum(heights, dim=0)

        left = torch.tensor([-B], device=device, dtype=dtype)
        xk = torch.cat((left, (left + cumsum_widths).to(device=device, dtype=dtype)))
        yk = torch.cat((left, (left + cumsum_heights).to(device=device, dtype=dtype)))

        delta_left = torch.ones(1, device=device, dtype=dtype)
        delta_right = torch.ones(1, device=device, dtype=dtype)
        if self.num_bins > 1:
            delta_mid = internal.to(device=device, dtype=dtype)
            delta = torch.cat((delta_left, delta_mid, delta_right))
        else:
            delta = torch.cat((delta_left, delta_right))
        return xk, yk, delta

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        device = inputs.device
        dtype = inputs.dtype
        B = self.tail_bound
        eps_val = torch.finfo(dtype).eps

        xk, yk, delta = self._knot_tensors(dtype=dtype, device=device)

        flat_inputs = inputs.reshape(-1)
        outputs = flat_inputs.clone()
        derivatives = torch.ones_like(flat_inputs)

        left_mask = flat_inputs <= -B
        right_mask = flat_inputs >= B
        center_mask = (~left_mask) & (~right_mask)

        if center_mask.any():
            s = flat_inputs[center_mask]
            bin_idx = torch.bucketize(s, xk[1:])

            x0 = xk[bin_idx]
            x1 = xk[bin_idx + 1]
            y0 = yk[bin_idx]
            y1 = yk[bin_idx + 1]
            d0 = delta[bin_idx]
            d1 = delta[bin_idx + 1]

            w = x1 - x0
            h = y1 - y0
            slope = h / w
            xi = (s - x0) / w

            eps_val = torch.finfo(dtype).eps
            t1 = slope * xi * xi + d0 * xi * (1.0 - xi)
            t2 = slope + (d1 + d0 - 2.0 * slope) * xi * (1.0 - xi)
            outputs_mid = y0 + h * (t1 / (t2 + eps_val))

            # Analytical derivative of the monotone RQS (matches finite differences)
            delta_val = slope
            coeff = d0 + d1 - 2.0 * delta_val
            B_val = delta_val + coeff * xi * (1.0 - xi)
            A_val = delta_val * xi * xi + d0 * xi * (1.0 - xi)
            dA_dxi = 2.0 * delta_val * xi + d0 * (1.0 - 2.0 * xi)
            dB_dxi = coeff * (1.0 - 2.0 * xi)
            derivatives_mid = delta_val * (dA_dxi * B_val - A_val * dB_dxi) / (
                (B_val + eps_val) ** 2
            )

            outputs[center_mask] = outputs_mid
            derivatives[center_mask] = derivatives_mid

        if left_mask.any():
            slope = delta[0]
            outputs[left_mask] = -B + (flat_inputs[left_mask] + B) * slope
            derivatives[left_mask] = slope

        if right_mask.any():
            slope = delta[-1]
            outputs[right_mask] = B + (flat_inputs[right_mask] - B) * slope
            derivatives[right_mask] = slope

        return outputs.view_as(inputs), derivatives.view_as(inputs)

    @torch.no_grad()
    def inverse(self, outputs: Tensor) -> Tuple[Tensor, Tensor]:
        device = outputs.device
        dtype = outputs.dtype
        B = self.tail_bound
        eps_val = torch.finfo(dtype).eps

        xk, yk, delta = self._knot_tensors(dtype=dtype, device=device)

        flat_outputs = outputs.reshape(-1)
        s_flat = flat_outputs.clone()

        left_mask = flat_outputs <= -B
        right_mask = flat_outputs >= B
        center_mask = (~left_mask) & (~right_mask)

        if center_mask.any():
            y = flat_outputs[center_mask]
            bin_idx = torch.bucketize(y, yk[1:])
            bin_idx = bin_idx.clamp(min=0, max=self.num_bins - 1)

            x0 = xk[bin_idx]
            x1 = xk[bin_idx + 1]
            y0 = yk[bin_idx]
            y1 = yk[bin_idx + 1]
            d0 = delta[bin_idx]
            d1 = delta[bin_idx + 1]

            w = x1 - x0
            h = y1 - y0
            slope = h / w
            z = (y - y0) / h

            coeff = d0 + d1 - 2.0 * slope
            a = slope - d0 + z * coeff
            b = d0 - z * coeff
            c = -z * slope

            discriminant = torch.clamp(b * b - 4.0 * a * c, min=0.0)
            sqrt_disc = torch.sqrt(discriminant)

            denom = 2.0 * a
            # Handle near-linear bins by falling back to linear solution
            linear_mask = torch.isclose(
                denom, torch.zeros_like(denom), atol=1e-12, rtol=0.0
            )

            eps_denom = torch.full_like(denom, eps_val)
            denom_safe = denom + torch.where(denom >= 0, eps_denom, -eps_denom)
            xi_quad1 = (-b - sqrt_disc) / denom_safe
            xi_quad2 = (-b + sqrt_disc) / denom_safe
            xi_quad = torch.where(
                (xi_quad1 >= 0.0) & (xi_quad1 <= 1.0), xi_quad1, xi_quad2
            )
            eps_b = torch.full_like(b, eps_val)
            b_safe = b + torch.where(b >= 0, eps_b, -eps_b)
            xi_linear = (-c) / b_safe
            xi = torch.where(linear_mask, xi_linear, xi_quad)
            xi = xi.clamp(0.0, 1.0)

            s_center = x0 + w * xi
            s_flat[center_mask] = s_center

        s = s_flat.view_as(outputs)
        _, derivatives = self.forward(s)
        return s, derivatives


class ExpMonotoneRQSSchedule(BetaSchedule):
    """Logit-time exp-monotone spline with an exact KO warm start."""

    def __init__(self, config: ExpMonotoneRQSConfig | None = None):
        super().__init__()
        self.config = config or ExpMonotoneRQSConfig()
        if self.config.init_c <= 0:
            raise ValueError("init_c must be > 0")
        if self.config.init_a <= 0:
            raise ValueError("init_a must be > 0")

        self.rqs = _ExpRQS1D(self.config.num_bins, self.config.tail_bound)

        base_dtype = torch.get_default_dtype()
        init_c = torch.tensor(float(self.config.init_c), dtype=base_dtype)
        init_a = torch.tensor(float(self.config.init_a), dtype=base_dtype)
        self.y0 = nn.Parameter(torch.log(init_c))
        self._a_unconstrained = nn.Parameter(_inv_softplus(init_a))

    @property
    def a(self) -> Tensor:
        eps = torch.finfo(self._a_unconstrained.dtype).eps
        return F.softplus(self._a_unconstrained) + eps

    @torch.no_grad()
    def set_warm_start(self, c: float, a: float) -> None:
        if c <= 0:
            raise ValueError("c must be > 0")
        if a <= 0:
            raise ValueError("a must be > 0")
        dtype = self.y0.dtype
        device = self.y0.device
        self.y0.copy_(torch.log(torch.tensor(float(c), dtype=dtype, device=device)))
        self._a_unconstrained.copy_(
            _inv_softplus(torch.tensor(float(a), dtype=dtype, device=device))
        )

    def _ell_and_derivative(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        cfg = self.config
        t_clamped = t.clamp(min=cfg.t_eps, max=1.0 - cfg.t_eps)
        s = torch.logit(t_clamped, eps=cfg.logit_eps)
        r_s, dr_ds = self.rqs(s)
        a = self.a
        ell = self.y0 + a * r_s
        denom = (t_clamped * (1.0 - t_clamped)).clamp_min(1e-8)
        dell_dt = (a * dr_ds) / denom
        return ell.view_as(t_clamped), dell_dt.view_as(t_clamped)

    def ell_from_t(self, t: Tensor) -> Tensor:
        ell, _ = self._ell_and_derivative(t)
        return ell

    def dtdell(self, t: Tensor) -> Tensor:
        _, dell_dt = self._ell_and_derivative(t)
        return dell_dt.reciprocal()

    @torch.no_grad()
    def t_from_ell(self, ell: Tensor) -> Tuple[Tensor, Tensor]:
        dtype = self.y0.dtype
        device = self.y0.device
        ell = ell.to(device=device, dtype=dtype)
        a = self.a
        y = (ell - self.y0) / a
        s, dr_ds = self.rqs.inverse(y)
        t = torch.sigmoid(s)
        t = t.clamp_(self.config.t_eps, 1.0 - self.config.t_eps)
        denom = (a * dr_ds).clamp_min(1e-8)
        dt_dell = (t * (1.0 - t)) / denom
        return t, dt_dell

    @torch.no_grad()
    def sample_t_uniform_logbeta(
        self, batch_shape, lmin: float, lmax: float
    ) -> Tuple[Tensor, Tensor]:
        """Sample ``log beta`` uniformly, invert to ``t``, and return ``(t, dt/dℓ)``.

        Args:
            batch_shape: Shape of the samples to draw (same semantics as ``torch.empty``).
            lmin: Lower bound of the uniform distribution over ``log beta``.
            lmax: Upper bound of the uniform distribution over ``log beta``.

        Returns:
            A tuple ``(t, weight)`` where ``t`` are the recovered time values and
            ``weight`` is the Jacobian ``dt/dℓ`` that preserves expectations when
            importance-sampling with uniform ``log beta`` draws.
        """

        if lmax < lmin:
            raise ValueError("lmax must be >= lmin")

        dtype = self.y0.dtype
        device = self.y0.device
        ell = torch.empty(batch_shape, dtype=dtype, device=device).uniform_(lmin, lmax)
        interval = max(lmax - lmin, float(torch.finfo(dtype).eps))
        t, dt_dell = self.t_from_ell(ell)
        weight = interval * dt_dell
        return t, weight

    def logbeta_regularization(
        self,
        *,
        t_lo: float | None = None,
        t_hi: float | None = None,
        power: float = 0.0,
    ) -> dict[str, Tensor]:
        """Compute smoothness penalties using ANALYTICAL derivatives from RQS.
        
        Key features:
        1. Uses exact dℓ/ds = a * dr/ds from RQS.forward() (not finite differences)
        2. NORMALIZED penalties using Coefficient of Variation (CV = std/mean)
        3. Second derivative: finite difference of exact first derivatives
        
        Normalization strategy (CV-based):
        - delta: CV² of first derivatives → penalizes relative variability
        - delta2: CV² of second derivatives → penalizes relative curvature changes
        - endpoint: normalized by interior scale → detects boundary anomalies
        
        Why CV is scale-invariant:
        - CV = std/mean is dimensionless (scale-free)
        - β(t) ×10 → std ×10, mean ×10 → CV unchanged
        - Penalties ∈ [0, ∞) with typical values ~ 0.01-1.0
        - Automatically adapts to any β magnitude
        """
        dtype = self.y0.dtype
        device = self.y0.device
        zero = torch.zeros((), dtype=dtype, device=device)
        eps = 1e-8  # Numerical stability
        cv_min_abs_mean = 1e-3  # Huber-like saturation: clamp |mean| to prevent CV² explosion

        xk, yk, _ = self.rqs._knot_tensors(dtype=dtype, device=device)
        if xk.numel() < 2:
            return {"delta": zero, "delta2": zero, "endpoint": zero}

        cfg = self.config
        default_lo = float(cfg.t_eps)
        default_hi = float(1.0 - cfg.t_eps)
        lo_val = max(default_lo, float(t_lo) if t_lo is not None else default_lo)
        hi_val = min(default_hi, float(t_hi) if t_hi is not None else default_hi)
        if hi_val <= lo_val:
            hi_val = lo_val + 1e-6
        t_lo_tensor = torch.tensor(lo_val, dtype=dtype, device=device)
        t_hi_tensor = torch.tensor(hi_val, dtype=dtype, device=device)

        # Map knot positions to t-space for masking
        t_knots = torch.sigmoid(xk)
        t_knots = t_knots.clamp(cfg.t_eps, 1.0 - cfg.t_eps)
        
        # ========== ANALYTICAL DERIVATIVE COMPUTATION ==========
        # Evaluate RQS at knot positions to get exact dr/ds
        r_knots, dr_ds_knots = self.rqs(xk)  # Analytical derivatives!
        a = self.a
        ell_knots = self.y0 + a * r_knots
        dell_ds_knots = a * dr_ds_knots  # Exact dℓ/ds = a * dr/ds
        # ========================================================

        # ========== FIRST-ORDER PENALTY (NORMALIZED) ==========
        # Use midpoints between knots for interval masking
        centers = 0.5 * (t_knots[1:] + t_knots[:-1])
        interval_mask = (centers >= t_lo_tensor) & (centers <= t_hi_tensor)
        weights = torch.ones_like(centers)
        if power != 0.0:
            weights = torch.pow((centers * (1.0 - centers)).clamp_min(1e-6), power)

        # Average first derivatives at interval endpoints
        first_slopes = 0.5 * (dell_ds_knots[1:] + dell_ds_knots[:-1])
        
        if interval_mask.any():
            weighted = weights[interval_mask]
            slopes = first_slopes[interval_mask]
            
            # NORMALIZED: Coefficient of Variation (CV) penalty
            # CV = std/mean measures relative variability (scale-invariant)
            # Penalizing CV² encourages smoothness regardless of magnitude
            # Huber-like saturation: clamp |mean| to prevent explosion when mean ≈ 0
            if slopes.numel() > 1:
                slopes_mean_raw = slopes.mean().detach()
                slopes_mean = torch.clamp(torch.abs(slopes_mean_raw), min=cv_min_abs_mean)  # Saturate denominator
                slopes_std = torch.std(slopes, unbiased=False).detach() + eps
                cv = slopes_std / slopes_mean  # Coefficient of variation with bounded denominator
                first_penalty = cv.square()  # Penalize relative variability
            else:
                # Single sample: no variability to penalize
                first_penalty = zero
        else:
            first_penalty = zero

        # ========== SECOND-ORDER PENALTY (NORMALIZED) ==========
        # Finite difference of EXACT first derivatives
        if dell_ds_knots.numel() >= 2:
            delta_s = xk[1:] - xk[:-1]
            delta_s = delta_s.clamp_min(1e-6)
            
            # Finite difference on analytical first derivatives
            delta_dell_ds = dell_ds_knots[1:] - dell_ds_knots[:-1]
            second_est = delta_dell_ds / delta_s  # Approximate d²ℓ/ds²
            
            # Mask to interior knots
            interior_t = t_knots[1:-1]
            interior_mask = (interior_t >= t_lo_tensor) & (interior_t <= t_hi_tensor)
            interior_weights = torch.ones_like(interior_t)
            if power != 0.0:
                interior_weights = torch.pow(
                    (interior_t * (1.0 - interior_t)).clamp_min(1e-6), power
                )
            
            # Trim second_est to match interior knots (exclude boundaries)
            if second_est.numel() > interior_t.numel():
                start_idx = 0
                end_idx = interior_t.numel()
                second_est_interior = second_est[start_idx:end_idx]
            else:
                second_est_interior = second_est
                
            if interior_mask.any() and second_est_interior.numel() == interior_mask.numel():
                w2 = interior_weights[interior_mask]
                s2 = second_est_interior[interior_mask]
                
                # NORMALIZED: Coefficient of Variation (CV) penalty
                # CV = std/mean measures relative variability (scale-invariant)
                # Huber-like saturation: clamp |mean| to prevent explosion when mean ≈ 0
                if s2.numel() > 1:
                    s2_mean_raw = s2.mean().detach()
                    s2_mean = torch.clamp(torch.abs(s2_mean_raw), min=cv_min_abs_mean)  # Saturate denominator
                    s2_std = torch.std(s2, unbiased=False).detach() + eps
                    cv2 = s2_std / s2_mean  # Coefficient of variation with bounded denominator
                    second_penalty = cv2.square()  # Penalize relative variability
                else:
                    # Single sample: no variability to penalize
                    second_penalty = zero
            else:
                second_penalty = zero
        else:
            second_penalty = zero

        # ========== ENDPOINT PENALTY (NORMALIZED) ==========
        # Penalize steep slopes at boundaries, normalized by interior scale
        if dell_ds_knots.numel() >= 3:
            # Use interior slopes (exclude endpoints) for scale reference
            interior_slopes = dell_ds_knots[1:-1]
            interior_scale = torch.abs(interior_slopes.mean()).detach() + eps
            
            # Normalized endpoint slopes (relative to interior)
            endpoint_left = dell_ds_knots[0] / interior_scale
            endpoint_right = dell_ds_knots[-1] / interior_scale
            
            # Penalize if endpoints deviate from interior scale
            endpoint_penalty = 0.5 * (endpoint_left.square() + endpoint_right.square())
        elif dell_ds_knots.numel() >= 1:
            # Too few knots: use absolute value normalization
            all_scale = torch.abs(dell_ds_knots.mean()).detach() + eps
            endpoint_left = dell_ds_knots[0] / all_scale
            endpoint_right = dell_ds_knots[-1] / all_scale
            endpoint_penalty = 0.5 * (endpoint_left.square() + endpoint_right.square())
        else:
            endpoint_penalty = zero

        return {
            "delta": first_penalty,
            "delta2": second_penalty,
            "endpoint": endpoint_penalty,
        }

    def beta_and_derivative(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        cfg = self.config
        t_clamped = t.clamp(min=cfg.t_eps, max=1.0 - cfg.t_eps)
        s = torch.logit(t_clamped, eps=cfg.logit_eps)
        r_s, dr_ds = self.rqs(s)
        a = self.a
        f = self.y0 + a * r_s
        beta = torch.exp(f)
        denom = t_clamped * (1.0 - t_clamped)
        dot_beta = beta * (a * dr_ds) / denom
        return beta.view_as(t), dot_beta.view_as(t)

