# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from collections import deque
from typing import Iterable, Optional

import torch
import torch.nn as nn
import numpy as np
from flow_matching.path import (
    CondOTProbPath,
    MixtureDiscreteProbPath,
    MetricInducedGibbsProbPath,
    ProbPath,
    BetaScheduleEMA,
    LearnableMetricEMA,
)
from flow_matching.path.beta_schedules import ExpMonotoneRQSSchedule
from flow_matching.path.scheduler import PolynomialConvexScheduler
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel

from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount
from training import distributed_mode

logger = logging.getLogger(__name__)

# NOTE: KO metric-induced path trains on 256-way tokens (no mask token).
# The original mixture path branch used a MASK_TOKEN=256 scheme.
MASK_TOKEN = 256
PRINT_FREQUENCY = 10


# Compatibility autocast wrapper: prefer torch.amp.autocast('cuda', ...) if available,
# otherwise fall back to torch.cuda.amp.autocast(...).
def _autocast(dtype: Optional[torch.dtype] = None):  # type: ignore[name-defined]
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        if dtype is None:
            return amp_mod.autocast("cuda")  # type: ignore[attr-defined]
        else:
            return amp_mod.autocast("cuda", dtype=dtype)  # type: ignore[attr-defined]
    # Fallback for older PyTorch
    if dtype is None:
        return torch.cuda.amp.autocast()
    else:
        return torch.cuda.amp.autocast(dtype=dtype)


def _collect_weight_norm_modules(module: nn.Module) -> list[nn.Module]:
    cached = getattr(module, "_cached_weight_norm_targets", None)
    if cached is not None:
        return cached

    targets: list[nn.Module] = []
    seen: set[int] = set()

    attr = getattr(module, "weight_norm_targets", None)
    if isinstance(attr, (list, tuple)):
        for candidate in attr:
            if isinstance(candidate, nn.Module) and id(candidate) not in seen:
                targets.append(candidate)
                seen.add(id(candidate))

    for submodule in module.modules():
        if hasattr(submodule, "force_weight_renorm") and id(submodule) not in seen:
            targets.append(submodule)
            seen.add(id(submodule))

    setattr(module, "_cached_weight_norm_targets", targets)
    setattr(module, "weight_norm_targets", targets)
    return targets


def _importance_weighted_mean(
    values: torch.Tensor, weights: Optional[torch.Tensor]
) -> torch.Tensor:
    """Return the mean of ``values`` with optional importance weights."""

    if weights is None:
        return values.mean()

    weight_tensor = weights.to(device=values.device, dtype=values.dtype)
    while weight_tensor.dim() < values.dim():
        weight_tensor = weight_tensor.unsqueeze(-1)
    return (values * weight_tensor).mean()


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def compute_geodesic_energy_embedding(
    logits: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    metric_module: nn.Module,
) -> torch.Tensor:
    """
    Compute geodesic energy regularization based on velocity in embedding space.
    
    Energy = ||v_t||²_M where v_t = (embed(x_1_pred) - embed(x_t)) / (1 - t)
    
    This penalizes high-velocity trajectories in the learned metric space,
    encouraging paths that follow geodesics and reducing entropy of intermediate
    distributions p(x_t | x_1, t).
    
    Args:
        logits: Model output logits [B, H, W, V] where V is vocab size (256)
        x_t: Current state [B, H, W] (discrete tokens) or [B, H, W, V] (soft)
        t: Time values [B] in [0, 1]
        metric_module: Learnable metric with .codes attribute [V, d]
    
    Returns:
        Scalar energy value (mean over batch and spatial dimensions)
    
    References:
        - Energy-guided geometric flow matching: 10-20% entropy reduction
        - Geodesic Gaussian regularization: smoother paths, lower variance
        - Entropic Gromov-Wasserstein: 5-15% entropy drops with λ tuning
    """
    # Get metric embedding codes [vocab_size, embedding_dim]
    codes = metric_module.codes  # [256, d]
    
    # Predicted distribution over x_1
    x_1_pred = torch.nn.functional.softmax(logits, dim=-1)  # [B, H, W, 256]
    
    # Convert x_t to soft distribution if needed
    if x_t.dtype == torch.long or x_t.dim() == 3:
        # Hard tokens [B, H, W] → one-hot [B, H, W, 256]
        vocab_size = codes.shape[0]
        x_t_soft = torch.nn.functional.one_hot(
            x_t.long(), num_classes=vocab_size
        ).float()
    else:
        # Already soft [B, H, W, 256]
        x_t_soft = x_t
    
    # Compute embeddings: weighted average of codes
    # x @ codes: [B, H, W, 256] @ [256, d] → [B, H, W, d]
    x_t_embed = x_t_soft @ codes  # [B, H, W, d]
    x_1_embed = x_1_pred @ codes  # [B, H, W, d]
    
    # Velocity in embedding space: change per unit time remaining
    # v_t = (x_1 - x_t) / (1 - t)  where t ∈ [0, 1]
    delta_t = 1.0 - t.view(-1, 1, 1, 1) + 1e-8  # [B, 1, 1, 1]
    v_t = (x_1_embed - x_t_embed) / delta_t  # [B, H, W, d]
    
    # Energy: L2 norm squared (already in metric space, no need for M weighting)
    # ||v_t||² = sum_d v_t[d]²
    energy = (v_t ** 2).sum(dim=-1).mean()  # scalar
    
    return energy


class AdaptiveKLController(nn.Module):
    """Adaptive penalty controller mirroring PPO/TRPO style KL annealing."""

    def __init__(
        self,
        *,
        target: float,
        init_weight: float,
        adapt_rate: float = 2.0,
        tolerance: float = 1.5,
        min_weight: float = 1e-4,
        max_weight: float = 1e4,
    ) -> None:
        super().__init__()
        if adapt_rate <= 1.0:
            raise ValueError("adapt_rate must be > 1")
        if tolerance <= 1.0:
            raise ValueError("tolerance must be > 1")
        if min_weight <= 0.0:
            raise ValueError("min_weight must be > 0")
        if max_weight <= 0.0 or max_weight < min_weight:
            raise ValueError("max_weight must be >= min_weight > 0")

        self.target = float(target)
        self.adapt_rate = float(adapt_rate)
        self.tolerance = float(tolerance)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.register_buffer("weight", torch.tensor(float(init_weight)))

    def current_weight(self) -> float:
        return float(self.weight.item())

    def compute_penalty(self, kl_value: torch.Tensor) -> torch.Tensor:
        scale = self.weight.to(device=kl_value.device, dtype=kl_value.dtype)
        return kl_value * scale

    def update(self, measured_kl: Optional[float]) -> None:
        if measured_kl is None:
            return
        kl_val = float(measured_kl)
        if not math.isfinite(kl_val):
            return

        weight = self.current_weight()
        upper = self.target * self.tolerance
        lower = self.target / self.tolerance if self.target > 0 else 0.0

        if self.target > 0 and kl_val > upper:
            weight = min(self.max_weight, weight * self.adapt_rate)
        elif self.target > 0 and kl_val < lower:
            weight = max(self.min_weight, weight / self.adapt_rate)

        self.weight.copy_(torch.tensor(weight, device=self.weight.device))

def _load_kl_window_state(
    args: argparse.Namespace, window_size: int
) -> tuple[deque, float]:
    """Restore the rolling KL window from ``args`` and respect the new size."""

    stored = getattr(args, "_schedule_kl_window", None)
    if stored is not None:
        kl_window = deque(stored, maxlen=window_size)
    else:
        kl_window = deque(maxlen=window_size)

    stored_sum = getattr(args, "_schedule_kl_window_sum", None)
    if len(kl_window) == 0:
        window_sum = 0.0
    elif stored_sum is not None:
        window_sum = float(stored_sum)
        actual_sum = float(sum(kl_window))
        if not math.isfinite(window_sum) or abs(window_sum - actual_sum) > 1e-9:
            window_sum = actual_sum
    else:
        # Recompute the sum in case the max length changed between epochs.
        window_sum = float(sum(kl_window))

    return kl_window, window_sum


def _save_kl_window_state(
    args: argparse.Namespace, kl_window: deque, window_sum: float
) -> None:
    """Persist the KL window so the next epoch continues the average."""

    setattr(args, "_schedule_kl_window", list(kl_window))
    setattr(args, "_schedule_kl_window_sum", float(window_sum))


def _anneal_scalar(
    step: int, total_steps: int, start: float, end: float, schedule: str
) -> float:
    """Return an annealed scalar following the provided schedule."""
    if total_steps <= 0:
        return end
    
    if step <= 0:
        return start
    if step >= total_steps:
        return end
    
    progress = step / total_steps
    
    if schedule == "linear":
        return start + (end - start) * progress
    elif schedule == "cosine":
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return end + (start - end) * cos_factor
    elif schedule == "exp":
        return start * math.exp(progress * math.log(end / start))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def adjust_metric_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    """Adjust learning rate for metric parameters based on decay schedule.
    
    This allows the metric to stabilize in later training while UNet continues
    adapting to the learned geometry. Inspired by staged learning strategies.
    
    Args:
        optimizer: The optimizer containing metric parameter groups
        epoch: Current training epoch
        args: Training arguments containing decay configuration
    """
    if not getattr(args, "mi_learnable_metric", False):
        return
    
    decay_start = int(getattr(args, "mi_metric_lr_decay_start", 3500))
    if epoch < decay_start:
        return
    
    decay_mode = getattr(args, "mi_metric_lr_decay_mode", "cosine")
    if decay_mode == "none":
        return
    
    # Base learning rate for metric (before decay)
    base_lr = args.lr * float(getattr(args, "mi_metric_lr_scale", 0.1))
    end_ratio = float(getattr(args, "mi_metric_lr_decay_end_ratio", 0.1))
    
    # Compute decay factor based on mode
    if decay_mode == "cosine":
        # Cosine annealing from 1.0 to end_ratio
        progress = (epoch - decay_start) / (args.epochs - decay_start)
        progress = min(progress, 1.0)
        
        decay_factor = end_ratio + (1.0 - end_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
        
    elif decay_mode == "exp":
        # Exponential decay
        decay_steps = (epoch - decay_start) / 100.0
        # Decay to end_ratio over ~10 steps (1000 epochs)
        decay_factor = max(end_ratio, end_ratio ** (decay_steps / 10.0))
        
    elif decay_mode == "step":
        # Step-wise decay at fixed milestones
        if epoch >= 4000:
            decay_factor = 0.25
        elif epoch >= 3500:
            decay_factor = 0.5
        else:
            decay_factor = 1.0
    else:
        decay_factor = 1.0
    
    # Apply to metric parameter groups only
    adjusted_any = False
    for param_group in optimizer.param_groups:
        # Identify metric parameters by group name
        group_name = param_group.get("name", "")
        # Also check if this group has fewer parameters (metric typically has 1-2 params)
        # and weight_decay is set (metric uses weight_decay, UNet typically doesn't)
        is_metric_group = (
            "learnable_metric" in group_name 
            or "metric" in group_name
            or (param_group.get("weight_decay", 0) > 0 and len(param_group["params"]) <= 10)
        )
        if is_metric_group:
            new_lr = base_lr * decay_factor
            param_group["lr"] = new_lr
            adjusted_any = True
    
    # Log only once per epoch (rank 0 only)
    if adjusted_any and distributed_mode.is_main_process():
        logger.info(
            f"Epoch {epoch}: Metric LR adjusted to {base_lr * decay_factor:.2e} "
            f"(mode={decay_mode}, factor={decay_factor:.4f})"
        )



def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
    path: Optional[ProbPath] = None,
    schedule_ema: Optional[BetaScheduleEMA] = None,
    metric_ema: Optional[LearnableMetricEMA] = None,
    kl_controller: Optional[AdaptiveKLController] = None,
):
    gc.collect()
    model.train(True)
    weight_norm_modules = _collect_weight_norm_modules(model)

    def _renorm_weight_norm_modules() -> None:
        for module in weight_norm_modules:
            if hasattr(module, "force_weight_renorm"):
                module.force_weight_renorm()

    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    # Select path according to args. For KO CIFAR-10 metric-induced path, set
    #   args.ko_metric_induced = True
    # For original discrete mixture path training, keep args.discrete_flow_matching = True
    if getattr(args, "ko_metric_induced", False):
        assert isinstance(path, MetricInducedGibbsProbPath), (
            "Metric-induced training expects a pre-instantiated MetricInducedGibbsProbPath."
        )
    elif args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
    else:
        path = CondOTProbPath()

    # Snapshot LUT state at epoch start for delta computations at epoch end.
    # Stored on args so state persists across function calls. Non-fatal if any error.
    if hasattr(path, "learnable_lut") and path.learnable_lut is not None:
        try:
            with torch.no_grad():
                start_w = path.learnable_lut().detach().cpu().clone()
            setattr(args, "_epoch_lut_start", start_w)
            if hasattr(path.learnable_lut, "scale_c") and path.learnable_lut.scale_c is not None:
                setattr(args, "_epoch_lut_scale_c_start", path.learnable_lut.scale_c.detach().cpu().clone())
        except Exception:
            pass

    beta_schedule_module: Optional[nn.Module] = None
    if isinstance(path, MetricInducedGibbsProbPath) and isinstance(
        path.beta_schedule, nn.Module
    ):
        beta_schedule_module = path.beta_schedule

    # Freeze schedule: params frozen but EMA continues (soft landing mechanism)
    freeze_schedule = getattr(args, "mi_freeze_beta_schedule", False)
    if freeze_schedule and schedule_ema is not None and epoch == args.start_epoch:
        # Only log once at the first epoch, and only on rank 0
        if distributed_mode.is_main_process():
            logger.info(
                "Freeze mode: schedule params frozen (no gradient), EMA continues (smooth training target). "
                "EMA will gradually converge to frozen values."
            )
    
    teacher_schedule_available = (
        schedule_ema is not None and beta_schedule_module is not None
    )
    teacher_metric_available = (
        metric_ema is not None
        and isinstance(path, MetricInducedGibbsProbPath)
        and path.learnable_metric is not None
    )
    use_path_ema = teacher_schedule_available or teacher_metric_available
    use_path_trust_region = use_path_ema and kl_controller is not None
    
    if kl_controller is not None and not use_path_ema:
        logger.warning(
            "KL controller was provided without an EMA teacher; disabling the controller."
        )
        kl_controller = None
        use_path_trust_region = False

    kl_metric = kl_penalty_metric = None
    kl_updates_total = 0
    kl_sum_for_step = 0.0
    kl_micro_steps = 0
    kl_avg_for_logging: Optional[float] = None
    kl_avg_window = int(getattr(args, "mi_beta_kl_avg_window", 1) or 1)
    if kl_avg_window <= 0:
        logger.warning(
            "mi_beta_kl_avg_window must be positive; received %s. Falling back to 1.",
            kl_avg_window,
        )
        kl_avg_window = 1
    if use_path_trust_region:
        kl_window, kl_window_sum = _load_kl_window_state(args, kl_avg_window)
    else:
        kl_window = None
        kl_window_sum = 0.0
    if use_path_trust_region:
        kl_metric = MeanMetric().to(device, non_blocking=True)
        kl_penalty_metric = MeanMetric().to(device, non_blocking=True)

    # Try to get dataloader length for global step computation
    try:
        _dl_len = len(data_loader)  # type: ignore[arg-type]
    except Exception:
        _dl_len = None

    gumbel_tau_steps = int(getattr(args, "mi_gumbel_tau_anneal_steps", 0) or 0)
    gumbel_tau_start = float(
        getattr(args, "mi_gumbel_tau_start", getattr(args, "mi_gumbel_tau", 1.0))
    )
    gumbel_tau_end = float(
        getattr(args, "mi_gumbel_tau_end", getattr(args, "mi_gumbel_tau", gumbel_tau_start))
    )
    gumbel_tau_schedule = getattr(args, "mi_gumbel_tau_schedule", "quadratic")
    gumbel_schedule_active = (
        gumbel_tau_steps > 0
        and isinstance(path, MetricInducedGibbsProbPath)
        and path.use_gumbel
    )
    current_gumbel_tau = getattr(path, "gumbel_tau", None)
    updates_per_epoch = None
    if _dl_len is not None:
        updates_per_epoch = (_dl_len + accum_iter - 1) // accum_iter
    gumbel_update_step = int(getattr(args, "_gumbel_update_step", 0))
    # CRITICAL: If explicitly set to 0 (e.g., when resuming but wanting fresh annealing),
    # don't override it with epoch_offset. Only use epoch_offset if counter is uninitialized.
    was_explicitly_reset = hasattr(args, "_annealing_counters_reset") and getattr(args, "_annealing_counters_reset", False)
    if gumbel_schedule_active and updates_per_epoch is not None and not was_explicitly_reset:
        epoch_offset = epoch * updates_per_epoch
        if gumbel_update_step < epoch_offset:
            gumbel_update_step = epoch_offset

    metric_interp_steps = int(getattr(args, "mi_metric_interp_anneal_steps", 0) or 0)
    metric_interp_start = float(getattr(args, "mi_metric_interp_start", 0.0))
    metric_interp_end = float(getattr(args, "mi_metric_interp_end", metric_interp_start))
    metric_interp_schedule = getattr(args, "mi_metric_interp_schedule", "cosine")
    metric_interp_active = (
        metric_interp_steps > 0
        and isinstance(path, MetricInducedGibbsProbPath)
        and path.has_learnable_metric
    )
    current_metric_interp = (
        path.get_metric_interpolation_lambda() if metric_interp_active else None
    )
    metric_interp_update_step = int(getattr(args, "_metric_interp_update_step", 0))
    # CRITICAL: Same logic as gumbel_update_step - respect explicit reset to 0
    if metric_interp_active and updates_per_epoch is not None and not was_explicitly_reset:
        epoch_offset_interp = epoch * updates_per_epoch
        if metric_interp_update_step < epoch_offset_interp:
            metric_interp_update_step = epoch_offset_interp

    logbeta_delta_weight = float(getattr(args, "mi_logbeta_reg_delta_weight", 0.0))
    logbeta_delta2_weight = float(getattr(args, "mi_logbeta_reg_delta2_weight", 0.0))
    logbeta_endpoint_weight = float(getattr(args, "mi_logbeta_endpoint_weight", 0.0))
    logbeta_reg_power = float(getattr(args, "mi_logbeta_reg_power", 0.0))
    logbeta_reg_steps = int(getattr(args, "mi_logbeta_reg_anneal_steps", 0) or 0)
    logbeta_reg_update_step = int(getattr(args, "_mi_logbeta_reg_step", 0))

    def _logbeta_annealed_weight(base: float) -> float:
        if base <= 0.0:
            return 0.0
        if logbeta_reg_steps <= 0:
            return base
        progress = min(
            max(logbeta_reg_update_step / float(logbeta_reg_steps), 0.0), 1.0
        )
        return base * (1.0 - progress)

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"label": labels}

        schedule_kl_value = None
        schedule_kl_penalty = None

        if getattr(args, "ko_metric_induced", False):
            # KO: 256-way classification; no mask token
            samples = (samples * 255.0).to(torch.long)
            logbeta_weights: Optional[torch.Tensor] = None
            mis_uniform_frac: Optional[float] = None
            schedule = getattr(path, "beta_schedule", None)
            schedule_is_exp = isinstance(schedule, ExpMonotoneRQSSchedule)
            lmin_cfg = getattr(args, "mi_logbeta_min", None)
            lmax_cfg = getattr(args, "mi_logbeta_max", None)
            band_lo_cfg = getattr(args, "mi_logbeta_band_t_lo", None)
            band_hi_cfg = getattr(args, "mi_logbeta_band_t_hi", None)
            trunc_t_cfg = getattr(args, "mi_logbeta_trunc_t", None)

            lmin_val = float(lmin_cfg) if lmin_cfg is not None else None
            lmax_val = float(lmax_cfg) if lmax_cfg is not None else None
            band_lo: Optional[float] = None
            band_hi: Optional[float] = None

            if schedule_is_exp:
                cfg_t_eps = float(schedule.config.t_eps)
                default_lo = cfg_t_eps
                default_hi = 1.0 - cfg_t_eps
                band_lo = max(
                    default_lo,
                    float(band_lo_cfg) if band_lo_cfg is not None else default_lo,
                )
                band_hi = min(
                    default_hi,
                    float(band_hi_cfg) if band_hi_cfg is not None else default_hi,
                )
                if band_hi <= band_lo:
                    if not getattr(args, "_mi_logbeta_band_warned", False):
                        logger.warning(
                            "Adjusted log-β t-band to maintain ordering (received %.4f, %.4f)",
                            float(band_lo),
                            float(band_hi),
                        )
                        setattr(args, "_mi_logbeta_band_warned", True)
                    band_hi = min(default_hi, band_lo + 1e-6)

            uniform_lo = 0.0
            uniform_hi = 1.0
            if band_lo is not None and band_hi is not None:
                uniform_lo = band_lo
                uniform_hi = band_hi

            trunc_t: Optional[float] = None
            if trunc_t_cfg is not None:
                trunc_t = float(trunc_t_cfg)
                min_trunc = max(
                    0.0,
                    float(schedule.config.t_eps) if schedule_is_exp else float(getattr(args, "mi_t_eps", 0.0)),
                )
                if trunc_t <= min_trunc:
                    if not getattr(args, "_mi_logbeta_trunc_warned", False):
                        logger.warning(
                            "mi_logbeta_trunc_t=%.4f must exceed %.4f; ignoring truncation.",
                            trunc_t,
                            min_trunc,
                        )
                        setattr(args, "_mi_logbeta_trunc_warned", True)
                    trunc_t = None
                elif trunc_t >= 0.5:
                    if not getattr(args, "_mi_logbeta_trunc_clip_warned", False):
                        logger.warning(
                            "mi_logbeta_trunc_t=%.4f is too close to 0.5; clipping to 0.499.",
                            trunc_t,
                        )
                        setattr(args, "_mi_logbeta_trunc_clip_warned", True)
                    trunc_t = 0.499
            if trunc_t is not None:
                uniform_lo = max(uniform_lo, trunc_t)
                uniform_hi = min(uniform_hi, 1.0 - trunc_t)

            if uniform_hi <= uniform_lo:
                if not getattr(args, "_mi_logbeta_uniform_warned", False):
                    logger.warning(
                        "Falling back to full [0,1] uniform support because bounds collapsed (%.4f, %.4f).",
                        uniform_lo,
                        uniform_hi,
                    )
                    setattr(args, "_mi_logbeta_uniform_warned", True)
                uniform_lo = 0.0
                uniform_hi = 1.0

            if (uniform_lo > 0.0 or uniform_hi < 1.0) and not getattr(
                args, "_mi_logbeta_uniform_announced", False
            ):
                logger.info(
                    "MIS uniform support set to [%.4f, %.4f]",
                    uniform_lo,
                    uniform_hi,
                )
                setattr(args, "_mi_logbeta_uniform_announced", True)

            if schedule_is_exp and band_lo is not None and band_hi is not None:
                effective_band_lo = max(band_lo, uniform_lo)
                effective_band_hi = min(band_hi, uniform_hi)
                if effective_band_hi <= effective_band_lo:
                    if not getattr(args, "_mi_logbeta_effective_band_warned", False):
                        logger.warning(
                            "Uniform truncation [%.4f, %.4f] conflicts with log-β band [%.4f, %.4f];"
                            " keeping schedule band for log-β proposals.",
                            uniform_lo,
                            uniform_hi,
                            band_lo,
                            band_hi,
                        )
                        setattr(args, "_mi_logbeta_effective_band_warned", True)
                    effective_band_lo = band_lo
                    effective_band_hi = band_hi
                else:
                    band_lo = effective_band_lo
                    band_hi = effective_band_hi

                with torch.no_grad():
                    t_bounds = torch.tensor(
                        [band_lo, band_hi],
                        dtype=schedule.y0.dtype,
                        device=schedule.y0.device,
                    )
                    ell_bounds = schedule.ell_from_t(t_bounds)
                derived_lmin = float(torch.min(ell_bounds).item())
                derived_lmax = float(torch.max(ell_bounds).item())
                if lmin_val is None:
                    lmin_val = derived_lmin
                else:
                    lmin_val = max(lmin_val, derived_lmin)
                if lmax_val is None:
                    lmax_val = derived_lmax
                else:
                    lmax_val = min(lmax_val, derived_lmax)

            use_logbeta_sampling = (
                schedule_is_exp
                and hasattr(schedule, "sample_t_uniform_logbeta")
                and lmin_val is not None
                and lmax_val is not None
            )
            if use_logbeta_sampling:
                assert lmin_val is not None and lmax_val is not None
                if lmax_val <= lmin_val:
                    if not getattr(args, "_mi_logbeta_interval_warned", False):
                        logger.warning(
                            "Ignoring log-β sampling interval with l_min >= l_max (%.4f, %.4f)",
                            lmin_val,
                            lmax_val,
                        )
                        setattr(args, "_mi_logbeta_interval_warned", True)
                    use_logbeta_sampling = False

            mix_alpha_raw = float(getattr(args, "mi_logbeta_mis_alpha", 0.0))
            mix_alpha = float(min(max(mix_alpha_raw, 0.0), 1.0))
            use_is = bool(getattr(args, "mi_logbeta_use_is", False))

            batch_size = samples.shape[0]
            uniform_span = max(uniform_hi - uniform_lo, 1e-6)
            t_uniform = uniform_lo + torch.rand(batch_size, device=device) * uniform_span

            if use_logbeta_sampling:
                assert lmin_val is not None and lmax_val is not None
                t_logbeta, _ = schedule.sample_t_uniform_logbeta(
                    batch_shape=(batch_size,),
                    lmin=float(lmin_val),
                    lmax=float(lmax_val),
                )
                t_logbeta = t_logbeta.to(device=device)
                selector = torch.rand(batch_size, device=device) < mix_alpha
                t = torch.where(selector, t_uniform, t_logbeta)

                if not getattr(args, "_mi_logbeta_sampling_announced", False):
                    logger.info(
                        "Using uniform log-β sampling with interval [%.4f, %.4f]",
                        float(lmin_val),
                        float(lmax_val),
                    )
                    setattr(args, "_mi_logbeta_sampling_announced", True)
                if 0.0 < mix_alpha < 1.0 and not getattr(args, "_mi_logbeta_mis_announced", False):
                    logger.info(
                        "Using MIS with α=%.3f (uniform-t) and %.3f (log-β proposal)",
                        mix_alpha,
                        1.0 - mix_alpha,
                    )
                    setattr(args, "_mi_logbeta_mis_announced", True)
                if not use_is and not getattr(args, "_mi_logbeta_no_is_announced", False):
                    logger.info(
                        "Importance sampling DISABLED: using direct log-β sampling without reweighting"
                    )
                    setattr(args, "_mi_logbeta_no_is_announced", True)

                # Only compute IS weights when explicitly enabled
                if use_is:
                    interval = max(float(lmax_val) - float(lmin_val), 1e-6)
                    t_for_schedule = t.to(device=t_logbeta.device, dtype=t_logbeta.dtype)
                    beta_vals, beta_deriv = schedule.beta_and_derivative(t_for_schedule)
                    beta_vals = beta_vals.clamp_min(1e-12)
                    q_logbeta = (beta_deriv / beta_vals).clamp_min(1e-12) / interval
                    q_logbeta = q_logbeta.to(device=device, dtype=torch.float32)
                    one_over_span = torch.tensor(
                        1.0 / uniform_span,
                        device=device,
                        dtype=q_logbeta.dtype,
                    )
                    q_uniform = torch.zeros_like(q_logbeta)
                    within_uniform = (t >= uniform_lo) & (t <= uniform_hi)
                    q_uniform = torch.where(within_uniform, one_over_span, q_uniform)
                    q_mix = mix_alpha * q_uniform + (1.0 - mix_alpha) * q_logbeta
                    logbeta_weights = (1.0 / q_mix.clamp_min(1e-12)).to(device=device)
                    mis_uniform_frac = float(selector.float().mean().detach().cpu().item())
                else:
                    # IS disabled: no reweighting
                    logbeta_weights = None
                    mis_uniform_frac = float(selector.float().mean().detach().cpu().item()) if mix_alpha > 0.0 else None
            else:
                t = t_uniform

            logbeta_reg_penalty: Optional[torch.Tensor] = None
            logbeta_reg_terms: Optional[dict[str, torch.Tensor]] = None
            if schedule_is_exp and isinstance(schedule, ExpMonotoneRQSSchedule):
                reg_terms = schedule.logbeta_regularization(
                    t_lo=band_lo,
                    t_hi=band_hi,
                    power=logbeta_reg_power,
                )
                logbeta_reg_terms = reg_terms
                penalty_components: Optional[torch.Tensor] = None
                delta_weight_curr = _logbeta_annealed_weight(logbeta_delta_weight)
                delta2_weight_curr = _logbeta_annealed_weight(logbeta_delta2_weight)
                endpoint_weight_curr = _logbeta_annealed_weight(logbeta_endpoint_weight)
                if delta_weight_curr > 0.0:
                    penalty = reg_terms["delta"] * delta_weight_curr
                    penalty_components = penalty if penalty_components is None else penalty_components + penalty
                if delta2_weight_curr > 0.0:
                    penalty = reg_terms["delta2"] * delta2_weight_curr
                    penalty_components = penalty if penalty_components is None else penalty_components + penalty
                if endpoint_weight_curr > 0.0:
                    penalty = reg_terms["endpoint"] * endpoint_weight_curr
                    penalty_components = penalty if penalty_components is None else penalty_components + penalty
                if penalty_components is not None:
                    logbeta_reg_penalty = penalty_components

            # Provide dummy x_0 for signature compatibility (not used by metric-induced path)
            x_0 = torch.zeros_like(samples)
            path_sample = path.sample(t=t, x_0=x_0, x_1=samples)
            x_t_model = path_sample.x_t_soft if path_sample.x_t_soft is not None else path_sample.x_t

            # Model should output logits with last dim = 256
            if getattr(args, "bf16", False):
                # Use compatibility autocast wrapper
                with _autocast(dtype=torch.bfloat16):
                    logits = model(x_t_model, t=t, extra=conditioning)
            else:
                logits = model(x_t_model, t=t, extra=conditioning)
            vocab_size = logits.shape[-1]
            logits_flat = logits.float().reshape(-1, vocab_size)
            targets_flat = samples.reshape(-1)
            token_loss = torch.nn.functional.cross_entropy(
                logits_flat, targets_flat, reduction="none"
            )
            with torch.no_grad():
                x1_flat = samples.view(samples.shape[0], -1)
                path_probs = path.get_prob_distribution_from_tokens(x1_flat, t)
                path_probs = path_probs.clamp_min(1e-12)
                per_site_entropy = -(path_probs * path_probs.log()).sum(dim=-1)
                target_entropy = per_site_entropy.mean(dim=1)
            per_sample_loss = token_loss.view(samples.shape[0], -1).mean(dim=1)
            uw_loss = per_sample_loss.mean().item()
            loss = _importance_weighted_mean(per_sample_loss, logbeta_weights)

            if logbeta_reg_penalty is not None:
                loss = loss + logbeta_reg_penalty

            # Bounded residual scale penalty (L2 on scale parameter c)
            scale_penalty: Optional[torch.Tensor] = None
            scale_penalty_weight = getattr(args, "mi_lut_scale_penalty_weight", 0.01)
            if (
                scale_penalty_weight > 0.0
                and hasattr(path, "learnable_lut")
                and path.learnable_lut is not None
                and hasattr(path.learnable_lut, "scale_c")
                and path.learnable_lut.scale_c is not None
            ):
                # L2 penalty on scale parameter c to keep it near 0
                scale_penalty = scale_penalty_weight * (path.learnable_lut.scale_c ** 2).sum()
                loss = loss + scale_penalty

            if use_path_trust_region:
                x1_flat = samples.view(samples.shape[0], -1)
                with torch.no_grad():
                    teacher_beta = None
                    if teacher_schedule_available and schedule_ema is not None:
                        teacher_beta, _ = schedule_ema.beta_and_derivative(t)
                    teacher_metric_module = (
                        metric_ema.teacher if teacher_metric_available else None
                    )
                    teacher_probs = path.get_prob_distribution_from_tokens(
                        x1_flat,
                        t,
                        beta_values=teacher_beta,
                        metric_module=teacher_metric_module,
                    )
                student_probs = path.get_prob_distribution_from_tokens(x1_flat, t)
                eps = 1e-8
                teacher_probs = teacher_probs.clamp_min(eps)
                student_probs = student_probs.clamp_min(eps)
                kl_tensor = (
                    teacher_probs
                    * (torch.log(teacher_probs) - torch.log(student_probs))
                ).sum(dim=-1)
                schedule_kl_value = _importance_weighted_mean(
                    kl_tensor, logbeta_weights
                )
                schedule_kl_penalty = kl_controller.compute_penalty(schedule_kl_value)
                loss = loss + schedule_kl_penalty
                kl_metric.update(schedule_kl_value.detach())
                kl_penalty_metric.update(schedule_kl_penalty.detach())
                kl_updates_total += 1
                kl_sum_for_step += float(schedule_kl_value.detach())
                kl_micro_steps += 1

            # Geodesic energy regularization (optional)
            geodesic_energy_val: Optional[torch.Tensor] = None
            geodesic_penalty: Optional[torch.Tensor] = None
            
            geodesic_weight = getattr(args, "mi_geodesic_energy_weight", 0.0)
            if geodesic_weight > 0.0:
                if isinstance(path, MetricInducedGibbsProbPath):
                    if path.learnable_metric is not None:
                        # Only apply when learned metric is active
                        current_lambda = path.get_metric_interpolation_lambda()
                        if current_lambda > 0.0:
                            # Compute geodesic energy
                            geo_energy = compute_geodesic_energy_embedding(
                                logits=logits,
                                x_t=path_sample.x_t,  # Use discrete tokens
                                t=t,
                                metric_module=path.learnable_metric,
                            )
                            
                            # Weight by metric interpolation (stronger as lambda increases)
                            # and by user-specified regularization strength
                            geo_penalty = geodesic_weight * current_lambda * geo_energy
                            
                            # Add to total loss
                            loss = loss + geo_penalty
                            
                            # Store for logging (detach to avoid gradients)
                            geodesic_energy_val = geo_energy.detach()
                            
                            # Announce once per training
                            if not getattr(args, "_geodesic_energy_announced", False):
                                logger.info(
                                    f"[Geodesic Energy] Enabled with λ={geodesic_weight:.4f}, "
                                    f"current metric interpolation={current_lambda:.3f}"
                                )
                                setattr(args, "_geodesic_energy_announced", True)

            wandb_logger = None
            if getattr(args, "wandb", False) and distributed_mode.is_main_process():
                try:
                    import swanlab as wandb  # type: ignore
                    wandb_logger = wandb
                except ImportError:
                    wandb_logger = None

            if wandb_logger is not None:
                weighted_entropy = _importance_weighted_mean(target_entropy, logbeta_weights)
                wandb_log_data = {
                    "diag/target_entropy_mean": float(weighted_entropy.detach().cpu().item()),
                }
                if logbeta_weights is not None:
                    w = logbeta_weights.detach()
                    ess_num = (w.sum() ** 2) / (w.square().sum() + 1e-12)
                    wandb_log_data["diag/ess_frac"] = float((ess_num / (w.numel() + 1e-12)).item())
                    wandb_log_data["diag/weight_mean"] = float(w.mean().item())
                    wandb_log_data["diag/weight_std"] = float(w.std().item())
                    wandb_log_data["diag/weight_max"] = float(w.max().item())
                    wandb_log_data["diag/weight_min"] = float(w.min().item())
                else:
                    # IS disabled: ESS = 1.0 (all samples have equal weight)
                    wandb_log_data["diag/ess_frac"] = 1.0
                if mis_uniform_frac is not None:
                    wandb_log_data["diag/mis_uniform_frac"] = mis_uniform_frac
                if logbeta_reg_penalty is not None:
                    wandb_log_data["loss/logbeta_reg_penalty"] = float(
                        logbeta_reg_penalty.detach().cpu().item()
                    )
                    if logbeta_reg_terms is not None:
                        wandb_log_data.update(
                            {
                                "loss/logbeta_reg_delta": float(
                                    logbeta_reg_terms["delta"].detach().cpu().item()
                                ),
                                "loss/logbeta_reg_delta2": float(
                                    logbeta_reg_terms["delta2"].detach().cpu().item()
                                ),
                                "loss/logbeta_reg_endpoint": float(
                                    logbeta_reg_terms["endpoint"].detach().cpu().item()
                                ),
                            }
                        )
                
                # Log geodesic energy regularization
                if geodesic_energy_val is not None:
                    wandb_log_data["train/geodesic_energy"] = float(
                        geodesic_energy_val.cpu().item()
                    )
                if geodesic_penalty is not None:
                    wandb_log_data["train/geodesic_penalty"] = float(
                        geodesic_penalty.detach().cpu().item()
                    )

                # Log bounded residual scale parameters and penalty
                if (
                    hasattr(path, "learnable_lut")
                    and path.learnable_lut is not None
                    and hasattr(path.learnable_lut, "scale_c")
                    and path.learnable_lut.scale_c is not None
                ):
                    scale_c = path.learnable_lut.scale_c.detach().cpu()
                    # Log each channel's scale_c value
                    for ch_idx in range(scale_c.numel()):
                        wandb_log_data[f"lut/scale_c_ch{ch_idx}"] = float(scale_c[ch_idx].item())
                    # Log scale_c statistics
                    wandb_log_data["lut/scale_c_mean"] = float(scale_c.mean().item())
                    wandb_log_data["lut/scale_c_std"] = float(scale_c.std().item())
                    wandb_log_data["lut/scale_c_min"] = float(scale_c.min().item())
                    wandb_log_data["lut/scale_c_max"] = float(scale_c.max().item())
                
                # Log scale penalty if it exists
                if scale_penalty is not None:
                    wandb_log_data["loss/scale_penalty"] = float(
                        scale_penalty.detach().cpu().item()
                    )

                # Log entropy vs t chart (reduced frequency to save space)
                if data_iter_step % (PRINT_FREQUENCY * 20) == 0:  # Reduced from 10x to 20x
                    t_cpu = t.detach().float().cpu()
                    entropy_cpu = target_entropy.detach().float().cpu()
                    
                    # Log simple statistics (always)
                    wandb_log_data["diag/entropy_mean"] = float(entropy_cpu.mean().item())
                    wandb_log_data["diag/entropy_std"] = float(entropy_cpu.std().item())
                    wandb_log_data["diag/t_mean"] = float(t_cpu.mean().item())
                wandb_logger.log(wandb_log_data)
        elif args.discrete_flow_matching:
            samples = (samples * 255.0).to(torch.long)
            t = torch.rand(samples.shape[0]).to(device)

            # sample probability path (mixture)
            x_0 = (
                torch.zeros(samples.shape, dtype=torch.long, device=device) + MASK_TOKEN
            )
            path_sample = path.sample(t=t, x_0=x_0, x_1=samples)

            # discrete flow matching loss (257-way: 256 tokens + MASK)
            if getattr(args, "bf16", False):
                # Use compatibility autocast wrapper
                with _autocast(dtype=torch.bfloat16):
                    logits = model(path_sample.x_t, t=t, extra=conditioning)
            else:
                logits = model(path_sample.x_t, t=t, extra=conditioning)
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape([-1, 257]), samples.reshape([-1])
            ).mean()
        else:
            # Scaling to [-1, 1] from [0, 1]
            samples = samples * 2.0 - 1.0
            noise = torch.randn_like(samples).to(device)
            if args.skewed_timesteps:
                t = skewed_timestep_sample(samples.shape[0], device=device)
            else:
                t = torch.rand(samples.shape[0]).to(device)
            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t  # type: ignore[attr-defined]

            # Use compatibility autocast wrapper (default dtype)
            with _autocast():
                loss = torch.pow(model(x_t, t, extra=conditioning) - u_t, 2).mean()

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        grad_norm = loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
            pre_step_fn=_renorm_weight_norm_modules if weight_norm_modules else None,
        )
        grad_step_skipped = False
        if apply_update:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if grad_norm is None:
                grad_step_skipped = True
            elif isinstance(grad_norm, torch.Tensor):
                grad_step_skipped = not torch.isfinite(grad_norm.detach()).all().item()
            else:
                grad_step_skipped = not math.isfinite(float(grad_norm))
            
            # Compute gradient norms before clipping for logging
            metric_grad_norm_value = None
            if hasattr(path, "learnable_metric") and path.learnable_metric is not None:
                # Compute metric gradient norm BEFORE clipping
                metric_params = list(path.learnable_metric.parameters())
                if metric_params:
                    metric_grad_norm_value = torch.nn.utils.clip_grad_norm_(
                        metric_params, max_norm=float('inf')
                    )
                    # Now actually clip to max_norm=1.0
                    torch.nn.utils.clip_grad_norm_(metric_params, max_norm=1.0)
            
            # LUT gradient diagnostics
            lut_grad_norm_value = None
            lut_param_delta = None
            if hasattr(path, "learnable_lut") and path.learnable_lut is not None:
                lut_params = list(path.learnable_lut.parameters())
                if lut_params and lut_params[0].grad is not None:
                    # Compute gradient norm
                    lut_grad_norm_value = torch.nn.utils.clip_grad_norm_(
                        lut_params, max_norm=float('inf')
                    )
                    # Store pre-update LUT for delta computation
                    if not hasattr(args, '_prev_lut_weight'):
                        args._prev_lut_weight = lut_params[0].data.detach().clone()
                    else:
                        # Compute parameter change
                        lut_param_delta = (lut_params[0].data - args._prev_lut_weight).norm().item()
                        args._prev_lut_weight = lut_params[0].data.detach().clone()
            
            # Post-update projection REMOVED for reparameterization mode
            # In reparameterization mode (default), Frobenius constraint is enforced by construction
            # via codes property: codes = scale * (codes_raw / ||codes_raw||_F)
            # This avoids fighting the optimizer and eliminates gradient explosion
            #
            # Legacy post-update projection (only for old checkpoints with use_reparameterization=False):
            # if apply_update and not grad_step_skipped:
            #     if hasattr(path, "learnable_metric") and path.learnable_metric is not None:
            #         if not getattr(path.learnable_metric, 'use_reparameterization', True):
            #             with torch.no_grad():
            #                 lower = path.learnable_metric.cholesky_factor()
            #                 Z = path.learnable_metric.codes @ lower.T
            #                 fro_norm = torch.linalg.norm(Z, ord='fro')
            #                 if fro_norm > 1e-8:
            #                     path.learnable_metric.codes.mul_(1.0 / fro_norm)
            
            optimizer.zero_grad(set_to_none=True)
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        if (
            apply_update
            and not grad_step_skipped
            and teacher_schedule_available
            and beta_schedule_module is not None
        ):
            # Only update schedule EMA if schedule is not frozen
            freeze_schedule = getattr(args, "mi_freeze_beta_schedule", False)
            if not freeze_schedule:
                schedule_ema.update(beta_schedule_module)
            # If frozen, skip EMA update (student == teacher, no need to track)
        if (
            apply_update
            and not grad_step_skipped
            and teacher_metric_available
            and isinstance(path, MetricInducedGibbsProbPath)
            and path.learnable_metric is not None
        ):
            metric_ema.update(path.learnable_metric)
        if apply_update and not grad_step_skipped:
            logbeta_reg_update_step += 1
        if apply_update:
            if use_path_trust_region:
                if grad_step_skipped:
                    kl_sum_for_step = 0.0
                    kl_micro_steps = 0
                    kl_avg_for_logging = None
                    if kl_window is not None:
                        kl_window.clear()
                    kl_window_sum = 0.0
                else:
                    mean_kl_step = (
                        kl_sum_for_step / kl_micro_steps if kl_micro_steps > 0 else None
                    )
                    if mean_kl_step is not None and kl_window is not None:
                        if len(kl_window) == kl_window.maxlen:
                            removed = kl_window.popleft()
                            kl_window_sum -= removed
                        kl_window.append(mean_kl_step)
                        kl_window_sum += mean_kl_step
                        if len(kl_window) > 0:
                            averaged_kl = kl_window_sum / len(kl_window)
                            if math.isfinite(averaged_kl):
                                kl_controller.update(averaged_kl)
                                kl_avg_for_logging = averaged_kl
                            else:
                                kl_avg_for_logging = None
                        else:
                            kl_avg_for_logging = None
                    else:
                        kl_avg_for_logging = None
                    kl_sum_for_step = 0.0
                    kl_micro_steps = 0
            else:
                kl_sum_for_step = 0.0
                kl_micro_steps = 0

        if apply_update and gumbel_schedule_active:
            step_index = min(gumbel_update_step + 1, gumbel_tau_steps)
            new_tau = _anneal_scalar(
                step_index,
                gumbel_tau_steps,
                gumbel_tau_start,
                gumbel_tau_end,
                gumbel_tau_schedule,
            )
            path.gumbel_tau = float(new_tau)
            current_gumbel_tau = float(new_tau)
            gumbel_update_step += 1

        if apply_update and metric_interp_active:
            step_index_lambda = min(metric_interp_update_step + 1, metric_interp_steps)
            new_lambda = _anneal_scalar(
                step_index_lambda,
                metric_interp_steps,
                metric_interp_start,
                metric_interp_end,
                metric_interp_schedule,
            )
            path.set_metric_interpolation_lambda(float(new_lambda))
            current_metric_interp = float(new_lambda)
            metric_interp_update_step += 1

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            # Console log
            dl_len_str = str(_dl_len) if _dl_len is not None else "?"
            log_msg = (
                f"Epoch {epoch} [{data_iter_step}/{dl_len_str}]: loss = {batch_loss.compute()}, lr = {lr}"
            )
            if use_path_trust_region and schedule_kl_value is not None:
                log_msg += (
                    f", kl = {float(schedule_kl_value.detach().cpu()):.4g},"
                    f" kl_w = {kl_controller.current_weight():.4g}"
                )
                if kl_avg_for_logging is not None:
                    log_msg += f", kl_avg = {float(kl_avg_for_logging):.4g}"
            if gumbel_schedule_active and current_gumbel_tau is not None:
                log_msg += f", tau = {float(current_gumbel_tau):.4g}"
            if metric_interp_active and current_metric_interp is not None:
                log_msg += f", lambda = {float(current_metric_interp):.4g}"
            
            # Add LUT diagnostics to log message
            if lut_grad_norm_value is not None:
                log_msg += f", lut_grad = {float(lut_grad_norm_value):.4g}"
            if lut_param_delta is not None:
                log_msg += f", lut_delta = {float(lut_param_delta):.6f}"
            
            logger.info(log_msg)

            # Optional Weights & Biases step-level logging (main process only)
            if getattr(args, "wandb", False) and distributed_mode.is_main_process():
                try:
                    import swanlab as wandb  # type: ignore
                    if _dl_len is not None:
                        global_step = epoch * _dl_len + data_iter_step
                    else:
                        global_step = None
                    
                    # Compute metric diagnostics for logging
                    metric_diagnostics = {}
                    if hasattr(path, "learnable_metric") and path.learnable_metric is not None:
                        with torch.no_grad():
                            Z = path.learnable_metric.transformed_codes(
                                device=device, dtype=torch.float32
                            )
                            fro_norm = torch.linalg.norm(Z, ord='fro')
                            dist_table = path.learnable_metric.pairwise_distance_table(
                                device=device, dtype=torch.float32
                            )
                            metric_diagnostics = {
                                "metric/frobenius_norm": float(fro_norm.cpu()),
                                "metric/distance_mean": float(dist_table.mean().cpu()),
                                "metric/distance_std": float(dist_table.std().cpu()),
                                "metric/distance_max": float(dist_table.max().cpu()),
                                "metric/distance_min": float(dist_table.min().cpu()),
                            }
                            # Log learned scale (in reparameterization mode)
                            if hasattr(path.learnable_metric, 'log_scale'):
                                metric_diagnostics["metric/learned_scale"] = float(
                                    torch.exp(path.learnable_metric.log_scale).cpu()
                                )
                            if metric_grad_norm_value is not None:
                                metric_diagnostics["metric/grad_norm"] = float(
                                    metric_grad_norm_value.cpu() if isinstance(metric_grad_norm_value, torch.Tensor) 
                                    else metric_grad_norm_value
                                )
                    
                    # Add LUT diagnostics
                    lut_diagnostics = {}
                    if hasattr(path, "learnable_lut") and path.learnable_lut is not None:
                        with torch.no_grad():
                            # Use forward() to get renormalized weights if enabled
                            lut_weight = path.learnable_lut()
                            lut_diagnostics["lut/weight_mean"] = float(lut_weight.mean().cpu())
                            lut_diagnostics["lut/weight_std"] = float(lut_weight.std().cpu())
                            lut_diagnostics["lut/weight_min"] = float(lut_weight.min().cpu())
                            lut_diagnostics["lut/weight_max"] = float(lut_weight.max().cpu())
                            # Per-channel std
                            for ch_idx in range(lut_weight.shape[0]):
                                lut_diagnostics[f"lut/ch{ch_idx}_std"] = float(lut_weight[ch_idx].std().cpu())
                        if lut_grad_norm_value is not None:
                            lut_diagnostics["lut/grad_norm"] = float(
                                lut_grad_norm_value.cpu() if isinstance(lut_grad_norm_value, torch.Tensor) 
                                else lut_grad_norm_value
                            )
                        if lut_param_delta is not None:
                            lut_diagnostics["lut/param_delta"] = float(lut_param_delta)
                    
                    wandb.log(  # type: ignore[attr-defined]
                        {
                            "train/step_loss": float(batch_loss.compute().detach().cpu()),
                            "train/inst_loss": float(loss_value),
                            "train/uw_loss": float(uw_loss) if 'uw_loss' in locals() else float(loss_value),
                            "train/lr": float(lr),
                            "epoch": int(epoch),
                            **metric_diagnostics,
                            **lut_diagnostics,
                            **({
                                "train/model_grad_norm": float(
                                    grad_norm.cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm
                                )
                            } if grad_norm is not None and not grad_step_skipped else {}),
                            **(
                                {
                                    "train/schedule_kl": float(
                                        schedule_kl_value.detach().cpu()
                                    ),
                                    "train/schedule_kl_penalty": float(
                                        schedule_kl_penalty.detach().cpu()
                                    ),
                                    "train/schedule_kl_weight": float(
                                        kl_controller.current_weight()
                                    ),
                                    **(
                                        {
                                            "train/schedule_kl_avg": float(
                                                kl_avg_for_logging
                                            )
                                        }
                                        if kl_avg_for_logging is not None
                                        else {}
                                    ),
                                }
                                if use_path_trust_region and schedule_kl_value is not None
                                else {}
                            ),
                            **(
                                {
                                    "train/gumbel_tau": float(current_gumbel_tau)
                                }
                                if gumbel_schedule_active and current_gumbel_tau is not None
                                else {}
                            ),
                            **(
                                {
                                    "train/metric_interp_lambda": float(
                                        current_metric_interp
                                    )
                                }
                                if metric_interp_active
                                and current_metric_interp is not None
                                else {}
                            ),
                            **(
                                {
                                    "train/logbeta_reg_penalty": float(
                                        logbeta_reg_penalty.detach().cpu()
                                    ),
                                    **(
                                        {
                                            "train/logbeta_reg_delta": float(
                                                logbeta_reg_terms["delta"].detach().cpu()
                                            ),
                                            "train/logbeta_reg_delta2": float(
                                                logbeta_reg_terms["delta2"].detach().cpu()
                                            ),
                                            "train/logbeta_reg_endpoint": float(
                                                logbeta_reg_terms["endpoint"].detach().cpu()
                                            ),
                                        }
                                        if logbeta_reg_terms is not None
                                        else {}
                                    ),
                                }
                                if logbeta_reg_penalty is not None
                                else {}
                            ),
                        },
                        step=global_step,
                    )
                except Exception:
                    pass

    setattr(args, "_mi_logbeta_reg_step", logbeta_reg_update_step)
    lr_schedule.step()
    stats = {"loss": float(epoch_loss.compute().detach().cpu())}
    if use_path_trust_region and kl_updates_total > 0 and kl_metric and kl_penalty_metric:
        stats.update(
            {
                "schedule_kl": float(kl_metric.compute().detach().cpu()),
                "schedule_kl_penalty": float(kl_penalty_metric.compute().detach().cpu()),
                "schedule_kl_weight": float(kl_controller.current_weight()),
            }
        )
        if kl_avg_for_logging is not None:
            stats["schedule_kl_avg"] = float(kl_avg_for_logging)
    if gumbel_schedule_active:
        setattr(args, "_gumbel_update_step", gumbel_update_step)
        if current_gumbel_tau is not None:
            stats["gumbel_tau"] = float(current_gumbel_tau)
    if metric_interp_active:
        setattr(args, "_metric_interp_update_step", metric_interp_update_step)
        if current_metric_interp is not None:
            stats["metric_interp_lambda"] = float(current_metric_interp)
    if use_path_trust_region and kl_window is not None:
        _save_kl_window_state(args, kl_window, kl_window_sum)
    else:
        for attr in ("_schedule_kl_window", "_schedule_kl_window_sum"):
            if hasattr(args, attr):
                delattr(args, attr)
    
    # LUT diagnostics at end of epoch (every 25 epochs)
    # Per-epoch numeric LUT diagnostics (always compute on main process).
    if hasattr(path, "learnable_lut") and path.learnable_lut is not None:
        try:
            if distributed_mode.is_main_process():
                with torch.no_grad():
                    lut_w = path.learnable_lut()  # forward() may renormalize
                    lut_w_cpu = lut_w.detach().cpu()
                    # Basic statistics
                    stats.update(
                        {
                            "lut/weight_mean": float(lut_w_cpu.mean().item()),
                            "lut/weight_std": float(lut_w_cpu.std().item()),
                            "lut/weight_min": float(lut_w_cpu.min().item()),
                            "lut/weight_max": float(lut_w_cpu.max().item()),
                        }
                    )

                    # Per-channel norms and (if available) renorm targets
                    try:
                        per_chan_norm = torch.linalg.vector_norm(lut_w_cpu, dim=(1, 2))
                        for i, val in enumerate(per_chan_norm.tolist()):
                            stats[f"lut/ch{i}_norm"] = float(val)
                        # base norm buffer exists on module (no-noise baseline)
                        base_buf = getattr(path.learnable_lut, "_base_fro_norm_per_channel", None)
                        if base_buf is not None:
                            base_cpu = base_buf.detach().cpu()
                            renorm_factors = (base_cpu / (per_chan_norm + getattr(path.learnable_lut, "renorm_eps", 1e-12)))
                            for i, val in enumerate(renorm_factors.tolist()):
                                stats[f"lut/ch{i}_renorm_factor"] = float(val)
                    except Exception:
                        pass

                    # Bounded residual scale diagnostics
                    if hasattr(path.learnable_lut, "scale_c") and path.learnable_lut.scale_c is not None:
                        sc = path.learnable_lut.scale_c.detach().cpu()
                        s0 = float(getattr(path.learnable_lut, "scale_baseline", 1.0))
                        eps = float(getattr(path.learnable_lut, "scale_epsilon", 0.25))
                        s_vals = s0 * (1.0 + eps * torch.tanh(sc))
                        stats.update(
                            {
                                "lut/scale_c_mean": float(sc.mean().item()),
                                "lut/scale_c_std": float(sc.std().item()),
                                "lut/scale_s_mean": float(s_vals.mean().item()),
                                "lut/scale_s_min": float(s_vals.min().item()),
                                "lut/scale_s_max": float(s_vals.max().item()),
                            }
                        )

                    # Orthogonality check for multi-dim embeddings (cheap: only max off-diag)
                    try:
                        if lut_w_cpu.ndim == 3 and lut_w_cpu.shape[2] > 1:
                            C, V, D = lut_w_cpu.shape
                            for ch in range(C):
                                W = lut_w_cpu[ch].numpy()  # [V, D]
                                G = W.T @ W
                                off_diag = np.abs(G - np.diag(np.diag(G)))
                                stats[f"lut/ch{ch}_orth_offdiag_max"] = float(off_diag.max())
                    except Exception:
                        # numpy may not be imported; skip if any error
                        pass

                    # Parameter delta since epoch start (if snapshot exists)
                    prev = getattr(args, "_epoch_lut_start", None)
                    if prev is not None:
                        try:
                            delta = (lut_w_cpu - prev).norm().item()
                            stats["lut/epoch_param_delta"] = float(delta)
                        except Exception:
                            pass

                # Also call the richer visualization every 25 epochs (unchanged)
                if epoch % 25 == 0:
                    _log_lut_diagnostics(path, epoch, logger, args)
        except Exception:
            logger.exception("Error while collecting per-epoch LUT diagnostics")
    
    return stats


def _log_lut_diagnostics(path: MetricInducedGibbsProbPath, epoch: int, logger: logging.Logger, args: argparse.Namespace) -> None:
    """Visualize LUT diagnostics with matplotlib plots.
    
    Creates a comprehensive visualization showing:
    1. LUT curves for all channels
    2. Channel correlations and metrics
    3. Health indicators
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping LUT visualization")
        return
    
    with torch.no_grad():
        # Use forward() to get renormalized weights (if enabled)
        lut_weights_raw = path.learnable_lut().cpu()  # [C, V, D]
        shape = lut_weights_raw.shape
        
        # Handle both [C, V] (legacy 1D) and [C, V, D] (new multi-dim) formats
        if len(shape) == 2:
            # Legacy 1D format: [C, V]
            C, V = shape
            D = 1
            lut_weights = lut_weights_raw.numpy()
            is_multidim = False
        elif len(shape) == 3:
            # New multi-dim format: [C, V, D]
            C, V, D = shape
            if D == 1:
                # Squeeze out singleton dimension for backward compatibility
                lut_weights = lut_weights_raw.squeeze(-1).numpy()  # [C, V]
                is_multidim = False
            else:
                # Compute L2 norm per token as scalar proxy for visualization
                lut_weights = torch.norm(lut_weights_raw, p=2, dim=-1).numpy()  # [C, V]
                is_multidim = True
        else:
            logger.error(f"Unexpected LUT weight shape: {shape}")
            return
        
        channel_names = ['R', 'G', 'B'] if C == 3 else [f'Ch{i}' for i in range(C)]
        colors = ['red', 'green', 'blue'] if C == 3 else [f'C{i}' for i in range(C)]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main LUT curves
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        x = np.arange(V)
        for c, (name, color) in enumerate(zip(channel_names, colors)):
            ax1.plot(x, lut_weights[c], label=name, color=color, alpha=0.8, linewidth=1.5)
        ax1.set_xlabel('Token Value', fontsize=12)
        ylabel = 'L2 Norm of Embedding' if is_multidim else 'Embedding Value'
        ax1.set_ylabel(ylabel, fontsize=12)
        title_suffix = f' (D={D})' if is_multidim else ''
        ax1.set_title(f'LUT Curves{title_suffix} - Epoch {epoch}', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Channel L2 Norms
        ax2 = fig.add_subplot(gs[0, 2])
        norms = [np.linalg.norm(lut_weights[c]) for c in range(C)]
        ax2.bar(channel_names, norms, color=colors, alpha=0.7)
        ax2.set_ylabel('L2 Norm', fontsize=11)
        ax2.set_title('Channel Norms', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(norms):
            ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Spearman ρ (rank correlation)
        ax3 = fig.add_subplot(gs[1, 2])
        init_order = np.arange(V, dtype=np.float32)
        rhos = []
        for c in range(C):
            emb = lut_weights[c]
            rank_emb = np.argsort(np.argsort(emb)).astype(np.float32)
            # Pearson correlation of ranks = Spearman
            rho = np.corrcoef(rank_emb, init_order)[0, 1]
            rhos.append(rho)
        ax3.bar(channel_names, rhos, color=colors, alpha=0.7)
        ax3.set_ylabel('Spearman ρ', fontsize=11)
        ax3.set_title('Rank Preservation', fontsize=12, fontweight='bold')
        ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_ylim([min(rhos) - 0.01, 1.01])
        ax3.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(rhos):
            ax3.text(i, v - 0.005, f'{v:.4f}', ha='center', va='top', fontsize=9)
        
        # 4. Inversions (monotonicity)
        ax4 = fig.add_subplot(gs[2, 0])
        inversions = []
        for c in range(C):
            diffs = lut_weights[c][1:] - lut_weights[c][:-1]
            inv_count = np.sum(diffs < 0)
            inversions.append(inv_count)
        ax4.bar(channel_names, inversions, color=colors, alpha=0.7)
        ax4.set_ylabel('Inversion Count', fontsize=11)
        ax4.set_title('Monotonicity Check', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(inversions):
            ax4.text(i, v + 0.5, f'{int(v)}', ha='center', va='bottom', fontsize=9)
        
        # 5. Channel correlations (heatmap)
        ax5 = fig.add_subplot(gs[2, 1])
        if C > 1:
            corr_matrix = np.corrcoef(lut_weights)
            im = ax5.imshow(corr_matrix, cmap='RdYlGn_r', vmin=0.95, vmax=1.0, aspect='auto')
            ax5.set_xticks(range(C))
            ax5.set_yticks(range(C))
            ax5.set_xticklabels(channel_names)
            ax5.set_yticklabels(channel_names)
            ax5.set_title('Channel Correlation', fontsize=12, fontweight='bold')
            # Add correlation values
            for i in range(C):
                for j in range(C):
                    text = ax5.text(j, i, f'{corr_matrix[i, j]:.4f}',
                                   ha='center', va='center', color='black', fontsize=9)
            plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
        else:
            ax5.text(0.5, 0.5, 'Single Channel', ha='center', va='center', fontsize=12)
            ax5.set_xticks([])
            ax5.set_yticks([])
        
        # 6. Health summary
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        health_text = f"Epoch {epoch}\n\n"
        health_text += "Health Status:\n"
        all_good = True
        for c, name in enumerate(channel_names):
            emb = lut_weights[c]
            diffs = emb[1:] - emb[:-1]
            inv_count = np.sum(diffs < 0)
            inv_ratio = inv_count / (V - 1)
            std_val = np.std(emb)
            
            issues = []
            if inv_ratio > 0.06:
                issues.append(f"inv>{int(inv_ratio*100)}%")
                all_good = False
            if std_val > 2.0:
                issues.append(f"std>{std_val:.1f}")
                all_good = False
            if std_val < 0.2:
                issues.append(f"std<{std_val:.2f}")
                all_good = False
            if emb[0] > emb[-1]:
                issues.append("flipped")
                all_good = False
            
            if issues:
                health_text += f"  {name}: ⚠ {', '.join(issues)}\n"
            else:
                health_text += f"  {name}: ✓\n"
        
        if all_good:
            health_text += "\n[OK] All checks passed"
        
        ax6.text(0.1, 0.9, health_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Save and log
        plt.suptitle(f'LUT Diagnostics - Epoch {epoch}', fontsize=16, fontweight='bold', y=0.98)
        
        # Save to file
        save_dir = getattr(args, 'output_dir', './output_dir')
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'lut_diagnostics_epoch_{epoch:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"LUT diagnostics visualization saved to {save_path}")
        
        # Log to wandb if available
        if getattr(args, 'wandb', False):
            try:
                import wandb
                # Only log if wandb is actually initialized
                if wandb.run is not None:
                    wandb.log({"lut/diagnostics": wandb.Image(save_path)}, step=epoch)
                    logger.info(f"✓ LUT diagnostics uploaded to wandb (run: {wandb.run.name})")
                else:
                    logger.info("wandb not initialized, image saved locally only")
            except ImportError:
                logger.info("wandb not installed, image saved locally only")
            except Exception as e:
                logger.info(f"wandb upload skipped ({type(e).__name__}), image saved locally")
        
        plt.close(fig)
