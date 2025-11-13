# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import contextlib
import gc
import logging
import math
import os
from collections import deque, defaultdict
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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


def _resolve_wandb_module(args: argparse.Namespace) -> Optional[object]:
    """
    Try importing swanlab (preferred) and fall back to the official wandb client.
    Cache the result on args to reuse the module across logging paths.
    """
    if not getattr(args, "wandb", False):
        return None
    cached = getattr(args, "_cached_wandb_module", None)
    attempted = getattr(args, "_cached_wandb_attempted", False)
    if attempted:
        return cached
    setattr(args, "_cached_wandb_attempted", True)
    if not distributed_mode.is_main_process():
        setattr(args, "_cached_wandb_module", None)
        return None

    module: Optional[object] = None
    try:
        import swanlab as wandb  # type: ignore

        module = wandb
    except ImportError:
        try:
            import wandb  # type: ignore

            module = wandb
        except ImportError:
            module = None
            if not getattr(args, "_wandb_import_warned", False):
                logger.warning(
                    "Weights & Biases logging disabled: unable to import swanlab or wandb."
                )
                setattr(args, "_wandb_import_warned", True)
    setattr(args, "_cached_wandb_module", module)
    return module


# NOTE: KO metric-induced path trains on 256-way tokens (no mask token).
# The original mixture path branch used a MASK_TOKEN=256 scheme.
MASK_TOKEN = 256
PRINT_FREQUENCY = 10


def _ks_uniform_metric(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    sorted_vals, _ = torch.sort(values)
    n = sorted_vals.numel()
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    if (max_val - min_val).abs() < 1e-9:
        return 0.0
    standardized = (sorted_vals - min_val) / (max_val - min_val + 1e-12)
    uniform_cdf = torch.linspace(0.0, 1.0, n, device=sorted_vals.device, dtype=sorted_vals.dtype)
    d = torch.max(torch.abs(standardized - uniform_cdf))
    return float(d.detach().cpu())


def _compute_lut_regularizer_and_metrics(
    path,
    device: torch.device,
    reg_align: float,
    reg_step: float,
    reg_curv: float,
    compute_metrics: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute regularization penalty and diagnostic metrics for learnable LUTs.

    For emb_dim=1 (scalar embeddings):
        - Uses original scalar-based constraints (monotonicity, smoothness, curvature)

    For emb_dim>1 (vector embeddings):
        - Direction continuity: penalizes step vectors that reverse direction (zig-zag)
        - Step length smoothness: penalizes abrupt changes in step magnitudes
        - Norm monotonicity: ensures pixel values map to increasing distances from origin

    Args:
        path: ProbPath containing learnable_lut attribute
        device: Device for computation
        reg_align: Weight for monotonicity/direction continuity constraint
        reg_step: Weight for step smoothness constraint
        reg_curv: Weight for curvature/norm monotonicity constraint
        compute_metrics: Whether to compute diagnostic metrics

    Returns:
        (penalty, metrics): Total regularization loss and diagnostic dict
    """
    lut = getattr(path, "learnable_lut", None)
    if lut is None:
        return torch.tensor(0.0, device=device), {}

    # Extract LUT weight tensor [C, V, D] where:
    # C = num_channels, V = vocab_size, D = emb_dim
    weight: Optional[torch.Tensor] = None
    if callable(lut):
        weight = lut()
    if weight is None and hasattr(lut, "weight"):
        weight = lut.weight
    if weight is None:
        return torch.tensor(0.0, device=device), {}

    weight = weight.to(device=device)
    if weight.size(1) < 2:
        return torch.tensor(0.0, device=device), {}

    dtype = weight.dtype
    emb_dim = weight.size(-1)

    # ========================================================================
    # Branch 1: emb_dim == 1 (Scalar embeddings)
    # Preserve original behavior for backward compatibility
    # ========================================================================
    if emb_dim == 1:
        # Scalar values: [C, V]
        if hasattr(lut, "scalar_trajectories"):
            values = lut.scalar_trajectories(weight)
        else:
            values = weight.squeeze(-1)
        values = values.to(device=device)

        # Monotonicity: penalize negative slopes
        first_diff = values[:, 1:] - values[:, :-1]  # [C, V-1]
        align_term = torch.clamp(-first_diff, min=0.0).mean()

        # Step smoothness: penalize direction reversals in (1, Δ) tangent space
        if first_diff.size(1) >= 2:
            tangent = torch.stack([
                torch.ones_like(first_diff),
                first_diff,
            ], dim=-1)  # [C, V-1, 2]
            tangent_norm = F.normalize(tangent, dim=-1, eps=1e-12)
            tangent_prev = tangent_norm[:, :-1, :]
            tangent_next = tangent_norm[:, 1:, :]
            cos_steps = (tangent_prev * tangent_next).sum(dim=-1)
            step_term = torch.clamp(-cos_steps, min=0.0).mean()
        else:
            cos_steps = torch.empty(0, device=device, dtype=dtype)
            step_term = weight.new_tensor(0.0)

        # Curvature: second-order finite difference
        if values.size(1) >= 3:
            curvature_mag = torch.abs(
                values[:, 2:] - 2 * values[:, 1:-1] + values[:, :-2]
            )
            curvature_term = curvature_mag.mean()
        else:
            curvature_mag = torch.empty(0, device=device, dtype=dtype)
            curvature_term = weight.new_tensor(0.0)

        penalty = (
            reg_align * align_term
            + reg_step * step_term
            + reg_curv * curvature_term
        )

        # Metrics for scalar case
        metrics: Dict[str, float] = {}
        if compute_metrics:
            if cos_steps.numel() > 0:
                clamped_cos = torch.clamp(cos_steps, -1.0 + 1e-6, 1.0 - 1e-6)
                angles = torch.acos(clamped_cos)
                angles_flat = angles.reshape(-1)
                angle_p50 = float(
                    torch.quantile(angles_flat, 0.5).detach().cpu() * (180.0 / math.pi)
                )
                angle_p90 = float(
                    torch.quantile(angles_flat, 0.9).detach().cpu() * (180.0 / math.pi)
                )
                cos_flat = cos_steps.detach().reshape(-1).float()
                direction_mean = float(cos_flat.mean().cpu())
                direction_std = float(cos_flat.std(unbiased=False).cpu())
                direction_p10 = float(torch.quantile(cos_flat, 0.1).cpu())
                direction_p90 = float(torch.quantile(cos_flat, 0.9).cpu())
                flip_rate = float((cos_flat < 0).float().mean().cpu())
            else:
                angle_p50 = 0.0
                angle_p90 = 0.0
                direction_mean = 0.0
                direction_std = 0.0
                direction_p10 = 0.0
                direction_p90 = 0.0
                flip_rate = 0.0

            ks = _ks_uniform_metric(values.detach().reshape(-1))
            metrics = {
                "lut_align": float(align_term.detach().cpu()),
                "lut_step": float(step_term.detach().cpu()),
                "lut_curvature": float(curvature_term.detach().cpu()),
                "lut_flip_rate": flip_rate,
                "lut_angle_p50_deg": angle_p50,
                "lut_angle_p90_deg": angle_p90,
                "lut_ks_uniform": ks,
                "lut_step_cos_mean": direction_mean,
                "lut_step_cos_std": direction_std,
                "lut_step_cos_p10": direction_p10,
                "lut_step_cos_p90": direction_p90,
            }

        return penalty, metrics

    # ========================================================================
    # Branch 2: emb_dim > 1 (Vector embeddings)
    # Use euclidean vector geometry constraints
    # ========================================================================

    # Compute step vectors: [C, V-1, D]
    steps = weight[:, 1:, :] - weight[:, :-1, :]

    # --- Constraint 1: Direction continuity (prevent zig-zag) ---
    # Use mild scale normalization: normalize by average step length to avoid
    # the "tiny step + reversal = no penalty" loophole, while still preserving
    # magnitude information (unlike full cosine normalization).
    if steps.size(1) >= 2:
        step_prev = steps[:, :-1, :]  # [C, V-2, D]
        step_next = steps[:, 1:, :]   # [C, V-2, D]

        # Raw dot product (unnormalized)
        dot_product = (step_prev * step_next).sum(dim=-1)  # [C, V-2]

        # Mild normalization: scale by average step magnitude
        # This balances between:
        # - Unnormalized (large steps dominate, tiny steps ignored)
        # - Fully normalized cosine (completely ignores magnitude)
        avg_len = 0.5 * (step_prev.norm(dim=-1) + step_next.norm(dim=-1)) + 1e-8
        scaled_dot = dot_product / avg_len  # [C, V-2]

        # Penalize negative scaled dot (direction reversal)
        direction_penalty = F.relu(-scaled_dot).mean()
    else:
        scaled_dot = torch.empty(0, device=device, dtype=dtype)
        direction_penalty = weight.new_tensor(0.0)

    # --- Constraint 2: Step length smoothness ---
    # Penalize abrupt changes in step magnitudes
    step_lengths = torch.norm(steps, dim=-1)  # [C, V-1]
    if step_lengths.size(1) >= 2:
        length_diff = step_lengths[:, 1:] - step_lengths[:, :-1]  # [C, V-2]
        length_smoothness_penalty = length_diff.abs().mean()
    else:
        length_smoothness_penalty = weight.new_tensor(0.0)

    # --- Constraint 3: Norm monotonicity (pixel order preservation) ---
    # Only apply for euclidean/Lp metrics where radius has geometric meaning.
    # For cosine metric, radius is arbitrary (only direction matters).
    metric_name = getattr(path, "metric", "euclidean")
    if isinstance(metric_name, str) and metric_name.lower() in ["euclidean", "lp"]:
        # Ensure embeddings move away from origin as pixel value increases
        norms = torch.norm(weight, dim=-1)  # [C, V]
        norm_diff = norms[:, 1:] - norms[:, :-1]  # [C, V-1]
        norm_monotone_penalty = F.relu(-norm_diff).mean()
    else:
        # For cosine or other metrics, skip norm constraint
        norms = torch.empty(0, device=device, dtype=dtype)
        norm_monotone_penalty = weight.new_tensor(0.0)

    # Total penalty: map hyperparameters to new constraints
    # reg_align -> direction continuity (most important for preventing zig-zag)
    # reg_step -> step length smoothness
    # reg_curv -> norm monotonicity (preserves pixel value ordering)
    penalty = (
        reg_align * direction_penalty
        + reg_step * length_smoothness_penalty
        + reg_curv * norm_monotone_penalty
    )

    # Diagnostic metrics
    metrics: Dict[str, float] = {}
    if compute_metrics:
        # Direction statistics
        if scaled_dot.numel() > 0:
            # Normalize to get cosine for interpretable angles
            step_prev_norm = F.normalize(steps[:, :-1, :], dim=-1, eps=1e-12)
            step_next_norm = F.normalize(steps[:, 1:, :], dim=-1, eps=1e-12)
            cos_angle = (step_prev_norm * step_next_norm).sum(dim=-1)  # [C, V-2]
            cos_flat = cos_angle.detach().reshape(-1).float()

            # Compute angle statistics
            clamped_cos = cos_flat.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            angles_rad = torch.acos(clamped_cos)
            angles_deg = angles_rad * (180.0 / math.pi)

            flip_rate = float((cos_flat < 0).float().mean().cpu())
            cos_mean = float(cos_flat.mean().cpu())
            cos_std = float(cos_flat.std(unbiased=False).cpu())
            cos_p10 = float(torch.quantile(cos_flat, 0.1).cpu())
            cos_p90 = float(torch.quantile(cos_flat, 0.9).cpu())
            angle_p50 = float(torch.quantile(angles_deg, 0.5).cpu())
            angle_p90 = float(torch.quantile(angles_deg, 0.9).cpu())

            # Scaled dot statistics (exposes "tiny step + reversal" issues)
            scaled_dot_flat = scaled_dot.detach().reshape(-1).float()
            scaled_dot_p10 = float(torch.quantile(scaled_dot_flat, 0.1).cpu())
            scaled_dot_mean = float(scaled_dot_flat.mean().cpu())
        else:
            flip_rate = 0.0
            cos_mean = 0.0
            cos_std = 0.0
            cos_p10 = 0.0
            cos_p90 = 0.0
            angle_p50 = 0.0
            angle_p90 = 0.0
            scaled_dot_p10 = 0.0
            scaled_dot_mean = 0.0

        # Step length statistics
        step_len_mean = float(step_lengths.mean().detach().cpu())
        step_len_std = float(step_lengths.std(unbiased=False).detach().cpu())
        step_len_min = float(step_lengths.min().detach().cpu())
        step_len_max = float(step_lengths.max().detach().cpu())

        # Step length stretch ratio (exposes uneven step distribution)
        # Ratio relative to mean: values >> 1 indicate "long jump" outliers
        if step_len_mean > 1e-8:
            stretch_ratio = step_lengths / (step_lengths.mean(dim=1, keepdim=True) + 1e-8)
            stretch_p90 = float(torch.quantile(stretch_ratio.detach().reshape(-1), 0.9).cpu())
        else:
            stretch_p90 = 0.0

        # Norm statistics (only if norm constraint is active)
        if norms.numel() > 0:
            norm_mean = float(norms.mean().detach().cpu())
            norm_std = float(norms.std(unbiased=False).detach().cpu())
            norm_min = float(norms.min().detach().cpu())
            norm_max = float(norms.max().detach().cpu())
        else:
            norm_mean = 0.0
            norm_std = 0.0
            norm_min = 0.0
            norm_max = 0.0

        metrics = {
            # Penalty components
            "lut_direction_penalty": float(direction_penalty.detach().cpu()),
            "lut_length_smooth_penalty": float(length_smoothness_penalty.detach().cpu()),
            "lut_norm_monotone_penalty": float(norm_monotone_penalty.detach().cpu()),

            # Direction/angle statistics
            "lut_step_flip_rate": flip_rate,
            "lut_step_cos_mean": cos_mean,
            "lut_step_cos_std": cos_std,
            "lut_step_cos_p10": cos_p10,
            "lut_step_cos_p90": cos_p90,
            "lut_step_angle_p50_deg": angle_p50,
            "lut_step_angle_p90_deg": angle_p90,

            # Scaled dot statistics (mild normalization exposes tiny-step issues)
            "lut_scaled_dot_p10": scaled_dot_p10,
            "lut_scaled_dot_mean": scaled_dot_mean,

            # Step length statistics
            "lut_step_length_mean": step_len_mean,
            "lut_step_length_std": step_len_std,
            "lut_step_length_min": step_len_min,
            "lut_step_length_max": step_len_max,
            "lut_step_stretch_p90": stretch_p90,  # Exposes uneven step distribution

            # Norm statistics
            "lut_norm_mean": norm_mean,
            "lut_norm_std": norm_std,
            "lut_norm_min": norm_min,
            "lut_norm_max": norm_max,
        }

    return penalty, metrics


def _compute_lut_reconstruction_loss(
    path: ProbPath,
    targets_flat: torch.Tensor,
    device: torch.device,
    *,
    sample_frac: float = 0.25,
    alpha: Optional[float] = None,
    t: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    zero = torch.zeros((), device=device, dtype=torch.float32)

    if not isinstance(path, MetricInducedGibbsProbPath):
        return zero, {}

    if path.learnable_lut is None:
        return zero, {}

    # ensure device & dtype
    targets_flat = targets_flat.to(device=device, non_blocking=True).long()
    N = targets_flat.shape[0]
    if N == 0:
        return zero, {}

    sample_frac = float(sample_frac)
    if not math.isfinite(sample_frac):
        sample_frac = 0.25
    sample_frac = min(max(sample_frac, 0.0), 1.0)
    if N == 0 or sample_frac == 0.0:
        zero = torch.tensor(0.0, device=device)
        alpha_val = float(alpha) if alpha is not None else 0.0
        return zero, {
            "train/lut_recon_loss": 0.0,
            "train/lut_recon_acc": 0.0,
            "train/lut_recon_alpha": alpha_val,
        }

    N_sub = max(1, min(N, int(N * sample_frac)))
    indices = torch.randperm(N, device=device)[:N_sub]
    targets_sub = targets_flat[indices]

    lut_weight = path.learnable_lut()
    E = lut_weight
    if E.dim() == 1:
        E = E.unsqueeze(0).unsqueeze(-1)
    elif E.dim() == 2:
        E = E.unsqueeze(0)
    if E.dim() != 3:
        logger.warning("Unexpected LUT weight shape %s; skipping recon loss.", tuple(E.shape))
        zero = torch.tensor(0.0, device=device)
        return zero, {}

    # Use first channel embeddings for reconstruction (channels share structure)
    E0 = E[0]
    z = E0[targets_sub]
    dist = path._pairwise_dist(z, E0)

    if alpha is None:
        alpha_val = 5.0
        beta_schedule = getattr(path, "beta_schedule", None)
        if beta_schedule is not None and t is not None:
            with torch.no_grad():
                beta_t = beta_schedule(t)
                alpha_val = float(beta_t.median().clamp_(1e-2, 50.0).item())
    else:
        alpha_val = float(alpha)

    logits_rec = (-alpha_val * dist).to(dtype=torch.float32)
    loss_rec = F.cross_entropy(logits_rec, targets_sub, reduction="mean")

    with torch.no_grad():
        acc = (logits_rec.argmax(-1) == targets_sub).float().mean()

    metrics = {
        "train/lut_recon_loss": float(loss_rec.item()),
        "train/lut_recon_acc": float(acc.item()),
        "train/lut_recon_alpha": float(alpha_val),
    }
    return loss_rec, metrics



def _extract_embedding_matrix_for_diagnostics(
    path: ProbPath,
) -> tuple[Optional[torch.Tensor], str]:
    """Return a 2-D embedding matrix [N, D] plus its source label for diagnostics."""

    metric_module = getattr(path, "learnable_metric", None)
    if metric_module is not None:
        try:
            matrix = metric_module.transformed_codes()
        except Exception:
            matrix = getattr(metric_module, "codes", None)
        if matrix is not None:
            return matrix.detach().to(device="cpu", dtype=torch.float32), "metric"

    lut_module = getattr(path, "learnable_lut", None)
    if lut_module is not None:
        weight: Optional[torch.Tensor] = None
        try:
            with torch.no_grad():
                if callable(lut_module):
                    weight = lut_module()
                elif hasattr(lut_module, "weight"):
                    weight = lut_module.weight
        except Exception:
            weight = None
        if weight is not None:
            weight = weight.detach().to(device="cpu", dtype=torch.float32)
            flat = weight.reshape(-1, weight.shape[-1])
            return flat, "lut"

    return None, "none"


def _compute_embedding_collapse_metrics(
    matrix: torch.Tensor,
    *,
    abs_eps: float = 1e-6,
    rel_eps: float = 1e-3,
    smallest_k: int = 4,
) -> Dict[str, float]:
    """Compute variance- and singular-value-based collapse diagnostics."""

    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return {}

    with torch.no_grad():
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        var_dim = centered.var(dim=0, unbiased=False)
        if var_dim.numel() == 0:
            return {}

        stats: Dict[str, float] = {}
        var_sorted, _ = torch.sort(var_dim)
        var_median = float(torch.median(var_dim).item())
        stats["lut/var_min"] = float(var_sorted[0].item())
        stats["lut/var_p05"] = float(torch.quantile(var_dim, 0.05).item())
        stats["lut/var_median"] = var_median

        collapse_abs = (var_dim < abs_eps).float().mean().item()
        rel_threshold = max(var_median * rel_eps, abs_eps)
        collapse_rel = (var_dim < rel_threshold).float().mean().item()
        stats["lut/collapse_frac_abs"] = collapse_abs
        stats["lut/collapse_frac_rel"] = collapse_rel

        top_k = int(min(max(smallest_k, 1), var_sorted.numel()))
        for idx in range(top_k):
            stats[f"lut/var_smallest_{idx + 1}"] = float(var_sorted[idx].item())

        # Singular values from Gram matrix (size D x D)
        gram = torch.matmul(centered.transpose(0, 1), centered)
        eigvals = torch.linalg.eigvalsh(gram)
        eigvals = torch.clamp(eigvals, min=0.0)
        sigma = torch.sqrt(eigvals)
        sigma, _ = torch.sort(sigma, descending=True)
        if sigma.numel() > 0:
            stats["lut/sigma_max"] = float(sigma[0].item())
            stats["lut/sigma_min"] = float(sigma[-1].item())
            cond = float("inf")
            if sigma[-1] > 0:
                cond = float((sigma[0] / sigma[-1]).item())
            stats["lut/cond"] = cond
            sigma_sq = sigma.square()
            total_power = float(sigma_sq.sum().item())
            if total_power > 0:
                stable_rank = total_power / float(sigma_sq.max().item() + 1e-12)
                probs = (sigma_sq / sigma_sq.sum()).clamp_min(1e-12)
                ent = float((-probs * probs.log()).sum().item())
                stats["lut/stable_rank"] = stable_rank
                stats["lut/effective_rank"] = math.exp(ent)
            top_sig = int(min(top_k, sigma.numel()))
            for idx in range(top_sig):
                stats[f"lut/sigma_{idx + 1}"] = float(sigma[idx].item())

    return stats


@torch.no_grad()
def eval_cross_entropy_vs_t(
    model: torch.nn.Module,
    path: ProbPath,
    data_loader: Iterable,
    device: torch.device,
    *,
    num_bins: int = 10,
    batches_per_eval: int = 1,
    t_eps: float = 1e-4,
    use_bf16: bool = False,
    ko_mode: bool = False,
    diag_batch_size: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Evaluate CE across fixed t bins for collapse diagnostics."""

    was_training = model.training
    model.eval()

    bins = torch.arange(num_bins, device=device, dtype=torch.float32)
    centers = (bins + 0.5) / max(num_bins, 1)
    t_centers = t_eps + centers * (1.0 - 2.0 * t_eps)

    ce_sum = torch.zeros(num_bins, device=device)
    ce_count = torch.zeros(num_bins, device=device)

    if use_bf16 and device.type == "cuda":
        dtype_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)  # type: ignore[attr-defined]
    else:
        dtype_ctx = contextlib.nullcontext()

    data_iter = iter(data_loader)
    processed = 0

    for _ in range(max(1, batches_per_eval)):
        try:
            samples, labels = next(data_iter)
        except StopIteration:
            break

        processed += 1
        # Optionally downsample batch for faster diagnostic
        if diag_batch_size is not None and samples.shape[0] > diag_batch_size:
            samples = samples[:diag_batch_size]
            if labels is not None:
                labels = labels[:diag_batch_size]
        samples = samples.to(device, non_blocking=True)
        if ko_mode:
            samples = (samples * 255.0).to(torch.long)
        labels = labels.to(device, non_blocking=True) if labels is not None else None

        batch_size = samples.shape[0]
        x_0 = torch.zeros_like(samples)

        for idx, t_value in enumerate(t_centers):
            t = torch.full((batch_size,), float(t_value.item()), device=device)
            path_sample = path.sample(t=t, x_0=x_0, x_1=samples)
            x_t_model = (
                path_sample.x_t_soft
                if getattr(path_sample, "x_t_soft", None) is not None
                else path_sample.x_t
            )
            conditioning = {"label": labels} if labels is not None else {}

            with dtype_ctx:
                logits = model(x_t_model, t=t, extra=conditioning)

            vocab_size = logits.shape[-1]
            token_loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, vocab_size),
                samples.reshape(-1),
                reduction="none",
            )
            ce_per_sample = token_loss.view(batch_size, -1).mean(dim=1)
            ce_sum[idx] += ce_per_sample.sum()
            ce_count[idx] += ce_per_sample.numel()

    if was_training:
        model.train()

    if processed == 0 or ce_count.sum() == 0:
        return None, None

    avg_ce = ce_sum / (ce_count + 1e-8)
    return t_centers.detach().cpu(), avg_ce.detach().cpu()


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
    entropy_metric = MeanMetric().to(device, non_blocking=True)
    lut_recon_loss_metric = MeanMetric().to(device, non_blocking=True)
    lut_recon_acc_metric = MeanMetric().to(device, non_blocking=True)
    lut_recon_metrics_recorded = False
    lut_recon_alpha_latest: Optional[float] = None
    cosine_scale_ratio_metric = MeanMetric().to(device, non_blocking=True)
    geodesic_penalty_metric = MeanMetric().to(device, non_blocking=True)
    geodesic_penalty_updated = False
    geometry_penalty_metric = MeanMetric().to(device, non_blocking=True)
    geometry_penalty_updated = False
    lut_reg_align = float(getattr(args, "mi_lut_reg_align", 0.0))
    lut_reg_step = float(getattr(args, "mi_lut_reg_step", 0.0))
    lut_reg_curvature = float(getattr(args, "mi_lut_reg_curvature", 0.0))
    lut_geometry_log = bool(getattr(args, "mi_lut_geometry_log", False))
    geometry_reg_enabled = any(w > 0.0 for w in (lut_reg_align, lut_reg_step, lut_reg_curvature))
    geometry_enabled = getattr(args, "mi_learnable_lut", False) and (geometry_reg_enabled or lut_geometry_log)
    geometry_accum = defaultdict(float)
    geometry_count = 0
    last_geometry_metrics: dict[str, float] = {}

    cosine_effective_neighbor_metric = MeanMetric().to(device, non_blocking=True)
    cosine_metrics_updated = False
    entropy_samples: list[torch.Tensor] = []

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
        if distributed_mode.is_main_process():
            logger.warning(
                "KL controller was provided without an EMA teacher; disabling the controller."
            )
        kl_controller = None
        use_path_trust_region = False

    cosine_monitor_enabled = (
        isinstance(path, MetricInducedGibbsProbPath)
        and getattr(path, "metric_name", "") == "cosine"
        and getattr(path, "learnable_lut", None) is not None
    )
    cosine_base_neighbor = float(getattr(args, "_mi_lut_cosine_base_neighbor", 0.0) or 0.0)
    cosine_raw_neighbor = float(getattr(args, "_mi_lut_cosine_neighbor_median", 0.0) or 0.0)
    if not (cosine_monitor_enabled and cosine_base_neighbor > 0.0 and cosine_raw_neighbor > 0.0):
        cosine_monitor_enabled = False

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

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        geom_metrics_step: dict[str, float] = {}
        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"label": labels}

        schedule_kl_value = None
        schedule_kl_penalty = None
        step_entropy_mean: Optional[float] = None
        step_entropy_median: Optional[float] = None
        step_entropy_p90: Optional[float] = None
        step_scale_ratio: Optional[float] = None
        step_effective_neighbor: Optional[float] = None

        if getattr(args, "ko_metric_induced", False):
            # KO: 256-way classification; no mask token
            samples = (samples * 255.0).to(torch.long)

            batch_size = samples.shape[0]
            schedule = getattr(path, "beta_schedule", None)
            schedule_is_exp = isinstance(schedule, ExpMonotoneRQSSchedule)
            gamma = max(float(getattr(args, "t_bias_gamma", 1.5)), 1e-6)
            u = torch.rand(batch_size, device=device)
            t_raw = u.pow(gamma)
            if schedule_is_exp:
                schedule_t_eps = float(schedule.config.t_eps)
            else:
                schedule_t_eps = float(getattr(args, "mi_t_eps", 1e-4))
            schedule_t_eps = min(max(schedule_t_eps, 0.0), 0.499)
            t = schedule_t_eps + (1.0 - 2.0 * schedule_t_eps) * t_raw
            t = t.clamp(schedule_t_eps, 1.0 - schedule_t_eps)
            t_mean_value = float(t.mean().detach().item())
            if t.numel() > 1:
                t_std_value = float(t.std(unbiased=False).detach().item())
            else:
                t_std_value = 0.0

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
            ce_token = torch.nn.functional.cross_entropy(
                logits_flat, targets_flat, reduction="none"
            )
            ce_per_sample = ce_token.view(samples.shape[0], -1).mean(dim=1)
            ce_unweighted = ce_per_sample.mean()
            ce_weighted = ce_unweighted
            t_weight_mode = str(getattr(args, "t_weight_mode", "none")).lower()
            lambda_t = float(getattr(args, "t_weight_lambda", 1.0))
            reported_lambda = lambda_t if t_weight_mode == "linear_t" else 0.0
            t_weight_mean = 1.0
            t_weight_std = 0.0
            if t_weight_mode == "linear_t" and ce_per_sample.numel() > 0:
                weights = 1.0 + lambda_t * (1.0 - t)
                if getattr(args, "t_weight_normalize", True):
                    weights = weights / weights.mean().clamp_min(1e-8)
                ce_weighted = (weights * ce_per_sample).mean()
                t_weight_mean = float(weights.mean().detach().item())
                if weights.numel() > 1:
                    t_weight_std = float(weights.std(unbiased=False).detach().item())
            loss = ce_weighted

            # LUT embedding reconstruction loss (prevent collapse)
            lut_recon_weight = float(getattr(args, "lut_recon_weight", 0.0))
            lut_recon_active = (
                lut_recon_weight > 0.0 and isinstance(path, MetricInducedGibbsProbPath)
            )
            if lut_recon_active:
                lut_recon_alpha = getattr(args, "lut_recon_alpha", None)
                lut_recon_sample_frac = float(getattr(args, "lut_recon_sample_frac", 0.25))

                loss_rec, rec_metrics = _compute_lut_reconstruction_loss(
                    path=path,
                    targets_flat=targets_flat,
                    device=device,
                    sample_frac=lut_recon_sample_frac,
                    alpha=lut_recon_alpha,
                    t=t,
                )
                loss = loss + lut_recon_weight * loss_rec
                lut_recon_loss_metric.update(loss_rec.detach())
                lut_recon_acc_metric.update(
                    torch.tensor(rec_metrics.get("train/lut_recon_acc", 0.0), device=device)
                )
                lut_recon_alpha_latest = rec_metrics.get("train/lut_recon_alpha", None)
                lut_recon_metrics_recorded = True

            ce_unweighted_scalar = float(ce_unweighted.detach().item())
            ce_weighted_scalar = float(ce_weighted.detach().item())
            with torch.no_grad():
                x1_flat = samples.view(samples.shape[0], -1)
                path_probs = path.get_prob_distribution_from_tokens(x1_flat, t)
                path_probs = path_probs.clamp_min(1e-12)
                per_site_entropy = -(path_probs * path_probs.log()).sum(dim=-1)
                target_entropy = per_site_entropy.mean(dim=1)
                entropy_metric.update(target_entropy.mean())
                entropy_detached = target_entropy.detach()
                if entropy_detached.numel() > 0:
                    entropy_cpu = entropy_detached.to(device="cpu")
                    entropy_samples.append(entropy_cpu)
                    step_entropy_mean = float(entropy_cpu.mean().item())
                    step_entropy_median = float(torch.quantile(entropy_cpu, 0.5).item())
                    step_entropy_p90 = float(torch.quantile(entropy_cpu, 0.9).item())
            geometry_penalty_value: Optional[torch.Tensor] = None
            if geometry_enabled:
                penalty, geom_metrics = _compute_lut_regularizer_and_metrics(
                    path=path,
                    device=device,
                    reg_align=lut_reg_align,
                    reg_step=lut_reg_step,
                    reg_curv=lut_reg_curvature,
                    compute_metrics=lut_geometry_log or geometry_reg_enabled,
                )
                if geometry_reg_enabled:
                    geometry_penalty_value = penalty.detach()
                    loss = loss + penalty
                    geometry_penalty_metric.update(geometry_penalty_value)
                    geometry_penalty_updated = True
                if geom_metrics:
                    geom_metrics_step = geom_metrics
                    geometry_count += 1
                    last_geometry_metrics = geom_metrics
                    for k, v in geom_metrics.items():
                        geometry_accum[k] += v


            beta_t_values, _ = path.beta(t)
            if cosine_monitor_enabled:
                beta_mean_value = float(beta_t_values.mean().detach().item())
                lut_scale = float(getattr(path, "_lut_cosine_scale", 1.0) or 1.0)
                effective_neighbor = beta_mean_value * lut_scale * cosine_raw_neighbor
                ratio = (
                    effective_neighbor / cosine_base_neighbor
                    if cosine_base_neighbor > 0.0
                    else float("nan")
                )
                if math.isfinite(ratio):
                    cosine_scale_ratio_metric.update(
                        torch.tensor(ratio, device=device, dtype=torch.float32)
                    )
                    cosine_metrics_updated = True
                if math.isfinite(effective_neighbor):
                    cosine_effective_neighbor_metric.update(
                        torch.tensor(effective_neighbor, device=device, dtype=torch.float32)
                    )
                    cosine_metrics_updated = True
                step_scale_ratio = ratio if math.isfinite(ratio) else None
                step_effective_neighbor = (
                    effective_neighbor if math.isfinite(effective_neighbor) else None
                )

            # KL trust region to baseline geometry for LUT (teacher = baseline distances)
            lut_kl_weight = float(getattr(args, "mi_lut_kl_weight", 0.0) or 0.0)
            if (
                lut_kl_weight > 0.0
                and isinstance(path, MetricInducedGibbsProbPath)
                and getattr(path, "learnable_lut", None) is not None
            ):
                # Optionally restrict to a t-band
                t_lo = getattr(args, "mi_lut_kl_t_lo", None)
                t_hi = getattr(args, "mi_lut_kl_t_hi", None)
                apply_mask = None
                if t_lo is not None and t_hi is not None:
                    t_lo_f = float(t_lo)
                    t_hi_f = float(t_hi)
                    apply_mask = (t >= t_lo_f) & (t <= t_hi_f)

                # Build baseline table once per step and gather rows for x1
                base_table = path._get_base_distance_table(
                    device=path.embedding.weight.device,
                    dtype=path.embedding.weight.dtype,
                )
                # Gather baseline distances rows for tokens
                B = x1_flat.shape[0]
                S = x1_flat.shape[1]
                K = path.vocab_size
                base_rows = base_table.index_select(0, x1_flat.view(-1)).view(B, S, K)
                # Compute teacher probs from baseline with the same beta(t)
                beta_t = beta_t_values.view(B, 1, 1).to(device=base_rows.device, dtype=base_rows.dtype)
                logits_base = -beta_t * base_rows
                logits_base = logits_base - logits_base.max(dim=-1, keepdim=True).values
                teacher_probs = torch.softmax(logits_base, dim=-1).to(device=path_probs.device, dtype=path_probs.dtype)

                # Student probs are current path_probs (already clamped)
                student_probs = path_probs.clamp_min(1e-12)
                teacher_probs = teacher_probs.clamp_min(1e-12).detach()  # no grad through teacher
                kl_tensor = (teacher_probs * (torch.log(teacher_probs) - torch.log(student_probs))).sum(dim=-1)
                if apply_mask is not None:
                    # Zero out KL where mask is false
                    kl_tensor = torch.where(apply_mask.view(-1, 1), kl_tensor, torch.zeros_like(kl_tensor))
                lut_kl_value = kl_tensor.mean()
                loss = loss + lut_kl_weight * lut_kl_value

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
                schedule_kl_value = kl_tensor.mean()
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
                            geodesic_penalty = geo_penalty.detach()
                            geodesic_penalty_metric.update(geodesic_penalty)
                            geodesic_penalty_updated = True

                            # Announce once per training
                            if not getattr(args, "_geodesic_energy_announced", False):
                                logger.info(
                                    f"[Geodesic Energy] Enabled with λ={geodesic_weight:.4f}, "
                                    f"current metric interpolation={current_lambda:.3f}"
                                )
                                setattr(args, "_geodesic_energy_announced", True)

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
        metric_grad_norm_value = None
        lut_grad_norm_value = None
        lut_param_delta = None

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
            tau_progress = None
            if gumbel_schedule_active and gumbel_tau_steps > 0:
                tau_progress = min(gumbel_update_step, gumbel_tau_steps) / float(gumbel_tau_steps)
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
                if tau_progress is not None:
                    log_msg += f" (prog={tau_progress:.2%})"
            if metric_interp_active and current_metric_interp is not None:
                log_msg += f", lambda = {float(current_metric_interp):.4g}"
            if step_entropy_mean is not None:
                log_msg += f", H={step_entropy_mean:.3f}"
                if step_entropy_median is not None:
                    log_msg += f" (p50={step_entropy_median:.3f})"
            if step_scale_ratio is not None:
                log_msg += f", cos_ratio={step_scale_ratio:.3f}"
                if step_effective_neighbor is not None:
                    log_msg += f" (eff={step_effective_neighbor:.3e})"
            
            # Add LUT diagnostics to log message
            if lut_grad_norm_value is not None:
                log_msg += f", lut_grad = {float(lut_grad_norm_value):.4g}"
            if lut_param_delta is not None:
                log_msg += f", lut_delta = {float(lut_param_delta):.6f}"
            
            logger.info(log_msg)

            # Optional Weights & Biases step-level logging (main process only)
            wandb_logger = _resolve_wandb_module(args)
            if wandb_logger is not None:
                try:
                    global_step = epoch * _dl_len + data_iter_step if _dl_len is not None else None

                    metric_diagnostics: Dict[str, float] = {}
                    if hasattr(path, "learnable_metric") and path.learnable_metric is not None:
                        with torch.no_grad():
                            Z = path.learnable_metric.transformed_codes(
                                device=device, dtype=torch.float32
                            )
                            fro_norm = torch.linalg.norm(Z, ord="fro")
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
                            if hasattr(path.learnable_metric, "log_scale"):
                                metric_diagnostics["metric/learned_scale"] = float(
                                    torch.exp(path.learnable_metric.log_scale).cpu()
                                )
                            if metric_grad_norm_value is not None:
                                metric_diagnostics["metric/grad_norm"] = float(
                                    metric_grad_norm_value.cpu()
                                    if isinstance(metric_grad_norm_value, torch.Tensor)
                                    else metric_grad_norm_value
                                )

                    lut_diagnostics: Dict[str, float] = {}
                    if hasattr(path, "learnable_lut") and path.learnable_lut is not None:
                        with torch.no_grad():
                            lut_weight = path.learnable_lut()
                            lut_flat = lut_weight.view(lut_weight.shape[0], -1)
                            channel_fro = torch.linalg.vector_norm(lut_flat, ord=2, dim=1)
                            lut_diagnostics["lut/fro_norm_mean"] = float(channel_fro.mean().cpu())
                        if hasattr(path.learnable_lut, "scale_c") and path.learnable_lut.scale_c is not None:
                            scale_c_cpu = path.learnable_lut.scale_c.detach().cpu()
                            lut_diagnostics["lut/scale_c_mean"] = float(scale_c_cpu.mean().item())
                        if lut_grad_norm_value is not None:
                            lut_diagnostics["lut/grad_norm"] = float(
                                lut_grad_norm_value.detach().cpu().item()
                                if isinstance(lut_grad_norm_value, torch.Tensor)
                                else float(lut_grad_norm_value)
                            )

                    collapse_metrics: Dict[str, float] = {}
                    log_structural_metrics = (
                        isinstance(path, MetricInducedGibbsProbPath)
                        and epoch % 5 == 0
                        and data_iter_step == 0
                    )
                    if log_structural_metrics:
                        embedding_matrix, embedding_source = _extract_embedding_matrix_for_diagnostics(path)
                        if embedding_matrix is not None:
                            collapse_metrics = _compute_embedding_collapse_metrics(embedding_matrix)
                            logger.info(
                                "Collapse diag source=%s shape=%s stable_rank=%.4f effective_rank=%.4f",
                                embedding_source,
                                tuple(embedding_matrix.shape),
                                collapse_metrics.get("lut/stable_rank", float("nan")),
                                collapse_metrics.get("lut/effective_rank", float("nan")),
                            )
                            # Extra check: compare raw vs effective LUT representations
                            try:
                                lut_mod = getattr(path, "learnable_lut", None)
                                if lut_mod is not None:
                                    with torch.no_grad():
                                        eff = lut_mod() if callable(lut_mod) else getattr(lut_mod, "weight", None)
                                        raw = getattr(lut_mod, "weight", None)
                                        if isinstance(eff, torch.Tensor) and eff.ndim >= 2:
                                            eff_flat = eff.detach().to(device="cpu", dtype=torch.float32).reshape(-1, eff.shape[-1])
                                            eff_m = _compute_embedding_collapse_metrics(eff_flat)
                                            logger.info(
                                                "LUT effective: shape=%s stable_rank=%.4f effective_rank=%.4f",
                                                tuple(eff_flat.shape),
                                                eff_m.get("lut/stable_rank", float("nan")),
                                                eff_m.get("lut/effective_rank", float("nan")),
                                            )
                                        if isinstance(raw, torch.Tensor) and raw.ndim >= 2:
                                            raw_flat = raw.detach().to(device="cpu", dtype=torch.float32).reshape(-1, raw.shape[-1])
                                            raw_m = _compute_embedding_collapse_metrics(raw_flat)
                                            logger.info(
                                                "LUT raw param: shape=%s stable_rank=%.4f effective_rank=%.4f",
                                                tuple(raw_flat.shape),
                                                raw_m.get("lut/stable_rank", float("nan")),
                                                raw_m.get("lut/effective_rank", float("nan")),
                                            )
                            except Exception:
                                logger.exception("Extra LUT collapse checks failed")
                        else:
                            logger.info(
                                "Collapse diag skipped: source=%s (no embedding matrix available)",
                                embedding_source,
                            )

                    wandb_payload = {
                        "train/step_loss": float(batch_loss.compute().detach().cpu()),
                        "train/inst_loss": float(loss_value),
                        "train/lr": float(lr),
                        "epoch": int(epoch),
                    }
                    wandb_payload.update(
                        {
                            "train/t_bias_gamma": float(gamma),
                            "train/t_mean": float(t_mean_value),
                            "train/t_std": float(t_std_value),
                            "train/t_weight_lambda": float(reported_lambda),
                            "train/t_weight_mean": float(t_weight_mean),
                            "train/t_weight_std": float(t_weight_std),
                            "train/ce_unweighted": float(ce_unweighted_scalar),
                            "train/ce_weighted": float(ce_weighted_scalar),
                        }
                    )
                    if step_scale_ratio is not None and step_effective_neighbor is not None:
                        wandb_payload.update(
                            {
                                "train/cosine_scale_ratio": float(step_scale_ratio),
                                "train/cosine_effective_neighbor": float(step_effective_neighbor),
                            }
                        )
                    if grad_norm is not None and not grad_step_skipped:
                        wandb_payload["train/model_grad_norm"] = float(
                            grad_norm.cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm
                        )
                    if use_path_trust_region and schedule_kl_value is not None:
                        wandb_payload["train/schedule_kl"] = float(schedule_kl_value.detach().cpu())
                        if schedule_kl_penalty is not None:
                            wandb_payload["train/schedule_kl_penalty"] = float(
                                schedule_kl_penalty.detach().cpu()
                            )
                        wandb_payload["train/schedule_kl_weight"] = float(kl_controller.current_weight())
                        if kl_avg_for_logging is not None:
                            wandb_payload["train/schedule_kl_avg"] = float(kl_avg_for_logging)
                    if gumbel_schedule_active and current_gumbel_tau is not None:
                        wandb_payload["train/gumbel_tau"] = float(current_gumbel_tau)
                    if tau_progress is not None:
                        wandb_payload["train/gumbel_tau_progress"] = float(tau_progress)
                    if step_entropy_mean is not None:
                        wandb_payload["train/entropy_mean"] = float(step_entropy_mean)
                    log_entropy_percentiles = (
                        epoch % 5 == 0 and data_iter_step == 0
                    )
                    if log_entropy_percentiles and step_entropy_median is not None:
                        wandb_payload["train/entropy_median"] = float(step_entropy_median)
                    if log_entropy_percentiles and step_entropy_p90 is not None:
                        wandb_payload["train/entropy_p90"] = float(step_entropy_p90)
                    if log_entropy_percentiles and target_entropy.numel() > 0:
                        wandb_payload["train/entropy_std"] = float(
                            target_entropy.detach().to(device="cpu").std().item()
                        )
                    if metric_interp_active and current_metric_interp is not None:
                        wandb_payload["train/metric_interp_lambda"] = float(current_metric_interp)
                    if geometry_penalty_value is not None:
                        wandb_payload["loss/lut_geometry_penalty"] = float(
                            geometry_penalty_value.cpu().item()
                        )
                    if geom_metrics_step:
                        for key, value in geom_metrics_step.items():
                            wandb_payload[key.replace("lut_", "lut_")] = value
                    if geodesic_energy_val is not None:
                        wandb_payload["train/geodesic_energy"] = float(geodesic_energy_val.cpu().item())
                    if geodesic_penalty is not None:
                        wandb_payload["train/geodesic_penalty"] = float(geodesic_penalty.cpu().item())
                    if scale_penalty is not None:
                        wandb_payload["loss/scale_penalty"] = float(
                            scale_penalty.detach().cpu().item()
                        )
                    wandb_payload.update(metric_diagnostics)
                    wandb_payload.update(lut_diagnostics)
                    wandb_payload.update(collapse_metrics)

                    wandb_logger.log(wandb_payload, step=global_step)
                except Exception as exc:
                    logger.warning(f"WandB logging failed at step {data_iter_step}: {exc}")

    lr_schedule.step()
    stats = {"loss": float(epoch_loss.compute().detach().cpu())}
    if geometry_penalty_updated:
        try:
            stats["loss/lut_geometry_penalty"] = float(
                geometry_penalty_metric.compute().detach().cpu().item()
            )
        except Exception:
            pass
    if geodesic_penalty_updated:
        try:
            stats["train/geodesic_penalty"] = float(
                geodesic_penalty_metric.compute().detach().cpu().item()
            )
        except Exception:
            pass
    if lut_recon_metrics_recorded:
        try:
            stats["train/lut_recon_loss"] = float(
                lut_recon_loss_metric.compute().detach().cpu().item()
            )
            stats["train/lut_recon_acc"] = float(
                lut_recon_acc_metric.compute().detach().cpu().item()
            )
            if lut_recon_alpha_latest is not None:
                stats["train/lut_recon_alpha"] = float(lut_recon_alpha_latest)
        except Exception:
            logger.exception("Failed to aggregate LUT reconstruction metrics")
    if geometry_count > 0:
        try:
            avg_metrics = {
                key: value / float(geometry_count) for key, value in geometry_accum.items()
            }
            for key, value in avg_metrics.items():
                stats[f"{key.replace('lut_', 'lut/')}_mean"] = float(value)
            stats["lut/geometry_updates"] = float(geometry_count)
            if last_geometry_metrics:
                for key, value in last_geometry_metrics.items():
                    stats[f"{key.replace('lut_', 'lut/')}_last"] = float(value)
        except Exception:
            logger.exception("Failed to aggregate LUT geometry metrics")

    # Optional CE-vs-t diagnostic: align start with eval_start_epoch to avoid heavy probes right after resume
    epoch_one = int(epoch) + 1
    eval_start = int(getattr(args, "eval_start_epoch", 0))
    want_diag = (
        isinstance(path, MetricInducedGibbsProbPath)
        and getattr(args, "ko_metric_induced", False)
        and epoch_one % 1 == 0
    )
    # All ranks enter two barriers so non-main ranks don't run ahead to DDP collectives
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if want_diag:
        schedule = getattr(path, "beta_schedule", None)
        ce_t_eps = float(getattr(args, "mi_t_eps", 1e-4))
        if isinstance(schedule, ExpMonotoneRQSSchedule):
            ce_t_eps = float(schedule.config.t_eps)
        t_vals, ce_vals = eval_cross_entropy_vs_t(
            model=model,
            path=path,
            data_loader=data_loader,
            device=device,
            num_bins=50,
            batches_per_eval=1,
            t_eps=ce_t_eps,
            use_bf16=bool(getattr(args, "bf16", False)),
            ko_mode=bool(getattr(args, "ko_metric_induced", False)),
            diag_batch_size=64,
        )
        if distributed_mode.is_main_process() and t_vals is not None and ce_vals is not None:
            wandb_logger = _resolve_wandb_module(args)
            if wandb_logger is not None:
                ce_payload = {
                    "train/ce_t_mean": float(ce_vals.mean().item()),
                    "train/ce_t_min": float(ce_vals.min().item()),
                    "train/ce_t_max": float(ce_vals.max().item()),
                }
                try:
                    module_name = getattr(wandb_logger, "__name__", "").lower()
                    if hasattr(wandb_logger, "Table") and hasattr(wandb_logger, "plot"):
                        table = wandb_logger.Table(
                            data=[[float(t), float(ce)] for t, ce in zip(t_vals, ce_vals)],
                            columns=["t", "cross_entropy"],
                        )
                        ce_payload["train/ce_vs_t"] = wandb_logger.plot.line(
                            table, "t", "cross_entropy", title="Cross Entropy vs Time"
                        )
                    elif module_name == "swanlab":
                        try:
                            from pyecharts.charts import Line  # type: ignore
                            from pyecharts import options as opts  # type: ignore

                            x_labels = [f"{float(t):.3f}" for t in t_vals]
                            y_values = [float(ce) for ce in ce_vals]
                            line = (
                                Line()
                                .add_xaxis(x_labels)
                                .add_yaxis(
                                    "cross_entropy",
                                    y_values,
                                    is_smooth=True,
                                    symbol_size=4,
                                    label_opts=opts.LabelOpts(is_show=False),
                                )
                                .set_global_opts(
                                    title_opts=opts.TitleOpts(title="Cross Entropy vs Time"),
                                    xaxis_opts=opts.AxisOpts(name="t"),
                                    yaxis_opts=opts.AxisOpts(
                                        name="cross_entropy",
                                        axislabel_opts=opts.LabelOpts(formatter="{value:.2f}"),
                                    ),
                                    datazoom_opts=[
                                        opts.DataZoomOpts(type_="inside"),
                                        opts.DataZoomOpts(type_="slider"),
                                    ],
                                    tooltip_opts=opts.TooltipOpts(trigger="axis"),
                                    grid_opts=opts.GridOpts(left="10%", right="8%", top="12%", bottom="18%"),
                                )
                            )
                            ce_payload["train/ce_vs_t_chart"] = line
                        except ImportError:
                            logger.warning(
                                "pyecharts not available; skipping SwanLab CE-vs-t chart."
                            )
                except Exception:
                    logger.exception("Failed to build CE-vs-t visualization payload")
                wandb_logger.log(ce_payload, step=epoch)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if entropy_samples:
        try:
            entropy_epoch_tensor = torch.cat(entropy_samples)
            stats.update(
                {
                    "entropy/mean": float(entropy_epoch_tensor.mean().item()),
                    "entropy/std": float(entropy_epoch_tensor.std(unbiased=False).item()),
                    "entropy/p25": float(torch.quantile(entropy_epoch_tensor, 0.25).item()),
                    "entropy/p50": float(torch.quantile(entropy_epoch_tensor, 0.5).item()),
                    "entropy/p90": float(torch.quantile(entropy_epoch_tensor, 0.9).item()),
                    "entropy/batch_mean": float(entropy_metric.compute().detach().cpu()),
                }
            )
        except Exception:
            pass
    if cosine_monitor_enabled and cosine_metrics_updated:
        try:
            stats.update(
                {
                    "cosine/scale_ratio": float(
                        cosine_scale_ratio_metric.compute().detach().cpu()
                    ),
                    "cosine/effective_neighbor": float(
                        cosine_effective_neighbor_metric.compute().detach().cpu()
                    ),
                }
            )
        except Exception:
            pass
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
        if gumbel_tau_steps > 0:
            stats["gumbel_tau_progress"] = float(
                min(gumbel_update_step, gumbel_tau_steps) / float(gumbel_tau_steps)
            )
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


def _log_lut_diagnostics(
    path: MetricInducedGibbsProbPath,
    epoch: int,
    logger: logging.Logger,
    args: argparse.Namespace,
) -> None:
    """Visualize LUT diagnostics as heatmaps plus a compact text panel.

    - Supports [C,V] or [C,V,D] LUT weights. For D>1, reduces to per-token
      scalars via L2 norm across the embedding dimension for plotting.
    - Only the main process saves/logs artifacts to avoid duplication.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
    except ImportError:
        logger.warning("matplotlib not available, skipping LUT visualization")
        return

    if getattr(path, "learnable_lut", None) is None:
        logger.info("LUT diagnostics skipped: path has no learnable_lut")
        return

    with torch.no_grad():
        lut_weights_raw = path.learnable_lut()
        if lut_weights_raw is None:
            logger.info("learnable_lut() returned None; skipping diagnostics")
            return

        lut_weights_raw = lut_weights_raw.detach().cpu()
        shape = tuple(lut_weights_raw.shape)

        if len(shape) == 2:
            C, V = shape
            D = 1
            lut_vals = lut_weights_raw.numpy()
            is_vector = False
        elif len(shape) == 3:
            C, V, D = shape
            if D == 1:
                lut_vals = lut_weights_raw.squeeze(-1).numpy()
                is_vector = False
            else:
                lut_vals = torch.linalg.norm(lut_weights_raw, ord=2, dim=-1).numpy()
                is_vector = True
        else:
            logger.error(f"Unexpected LUT weight shape: {shape}")
            return

        channel_names = ["R", "G", "B"] if C == 3 else [f"Ch{i}" for i in range(C)]

        vals = lut_vals.reshape(-1)
        vmin = float(vals.min()) if vals.size > 0 else 0.0
        vmax = float(vals.max()) if vals.size > 0 else 0.0

        if V > 1:
            diffs = lut_vals[:, 1:] - lut_vals[:, :-1]
            neg_frac = float((diffs < 0).mean())
        else:
            diffs = None
            neg_frac = 0.0

        norms = [float(np.linalg.norm(lut_vals[c])) for c in range(C)]

        try:
            ks_uniform = float(_ks_uniform_metric(torch.from_numpy(vals)))
        except Exception:
            ks_uniform = 0.0

    # Figure layout: 2x2 grid
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)

    # 1) LUT heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(lut_vals, aspect="auto")
    ax1.set_title(
        f"LUT values{' (L2 norm)' if is_vector else ''} - Epoch {epoch}",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylabel("Channel", fontsize=11)
    ax1.set_xlabel("Token", fontsize=11)
    ax1.set_yticks(range(C))
    ax1.set_yticklabels(channel_names)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2) First-difference heatmap (monotonicity)
    ax2 = fig.add_subplot(gs[0, 1])
    if diffs is not None:
        im2 = ax2.imshow(diffs, aspect="auto", cmap="RdBu_r")
        ax2.set_title("LUT finite differences (Δ along token)", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Channel", fontsize=11)
        ax2.set_xlabel("Token index (Δ)", fontsize=11)
        ax2.set_yticks(range(C))
        ax2.set_yticklabels(channel_names)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    else:
        ax2.set_title("LUT finite differences (V=1, skipped)", fontsize=12)
        ax2.axis("off")

    # 3) Per-channel norm bar
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(range(C), norms)
    ax3.set_xticks(range(C))
    ax3.set_xticklabels(channel_names)
    ax3.set_ylabel("L2 norm", fontsize=11)
    ax3.set_title("Channel norms", fontsize=12, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.3)

    # 4) Text panel with health summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    lines = []
    lines.append(f"Shape: C={C}, V={V}, D={D}")
    lines.append(f"Value range: [{vmin:.4f}, {vmax:.4f}]")
    lines.append(f"Neg. Δ fraction (monotonic violations): {neg_frac:.4f}")
    lines.append(f"KS distance to uniform (all values): {ks_uniform:.4f}")
    lines.append("")
    lines.append("Channel norms:")
    for name, n in zip(channel_names, norms):
        lines.append(f"  {name}: {n:.3f}")
    ax4.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax4.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle("LUT Diagnostics", fontsize=14, fontweight="bold")

    # Save & log only from the main process to avoid duplication
    if distributed_mode.is_main_process():
        out_root = getattr(args, "log_dir", None) or getattr(args, "output_dir", ".")
        out_dir = Path(out_root) / "lut_diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"lut_epoch_{epoch:04d}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        logger.info(f"LUT diagnostics saved to {fname}")

        wandb = _resolve_wandb_module(args)
        if wandb is not None and getattr(wandb, "run", None) is not None:
            try:
                wandb.log(
                    {
                        "lut_diag_image": wandb.Image(str(fname)),
                        "lut/neg_delta_frac": neg_frac,
                        "lut/ks_uniform_snapshot": ks_uniform,
                    },
                    step=epoch,
                )
            except Exception:
                pass

    # Close the figure in all processes
    plt.close(fig)

