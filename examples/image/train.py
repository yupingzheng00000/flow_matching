# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Path bootstrap
_this_dir = _Path(__file__).resolve().parent
# Go up three levels: .../flow_matching/examples/image -> .../flow_matching
_pkg_root = _this_dir.parents[1]
if str(_pkg_root) not in _sys.path:
    _sys.path.insert(0, str(_pkg_root))

"""

import datetime
import json
import logging
import gc
import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser

from flow_matching.path import (
    BetaSchedule,
    BetaScheduleEMA,
    ExpMonotoneRQSConfig,
    ExpMonotoneRQSSchedule,
    LearnableMetricEMA,
    MetricInducedGibbsProbPath,
    MonotoneRQBetaSchedule,
    MonotoneRQConfig,
)
from training import distributed_mode
from training.data_transform import get_train_transform
from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import AdaptiveKLController, train_one_epoch

logger = logging.getLogger(__name__)


def _warm_start_cosine_lut(path: MetricInducedGibbsProbPath) -> None:
    """Initialize cosine-metric LUT on a great-circle warm start."""
    lut = getattr(path, "learnable_lut", None)
    if lut is None:
        return
    C, V, D = lut.weight.shape
    if D < 2:
        logger.warning(
            "Cosine LUT warm start requires emb_dim >= 2; current emb_dim=%d. Skipping warm start.",
            D,
        )
        return
    with torch.no_grad():
        device = lut.weight.device
        dtype = lut.weight.dtype
        m = torch.linspace(-1.0, 1.0, V, device=device, dtype=dtype)  # baseline linear scale
        theta = (m + 1.0) * (math.pi / 2.0)  # map to [0, π]
        base = torch.zeros(C, V, D, device=device, dtype=dtype)
        base[:, :, 0] = torch.cos(theta).unsqueeze(0).expand(C, -1)
        base[:, :, 1] = torch.sin(theta).unsqueeze(0).expand(C, -1)
        if D > 2:
            base[:, :, 2:] = 1e-4 * torch.randn(C, V, D - 2, device=device, dtype=dtype)
        if hasattr(lut, "initialize_from_weight"):
            lut.initialize_from_weight(base)
        else:
            lut.weight.copy_(base)
    logger.info("Applied cosine LUT warm start on great-circle initialization")


def _calibrate_cosine_scale(
    path: MetricInducedGibbsProbPath,
    *,
    mode: str = "neighbor",
    t_mid_values: Sequence[float] = (0.5,),
    num_samples: int = 4096,
) -> Dict[str, Union[bool, float, str, list]]:
    """
    Calibrate cosine LUT scale by matching baseline and cosine distances.

    Args:
        path: Metric-induced path with cosine metric and learnable LUT.
        mode: "neighbor" (adjacent-token distances) or "median" (full-table medians).
        t_mid_values: Reference t values in (0,1); first value sets the applied scale,
                      the full list is logged for diagnostics.
        num_samples: Sampling budget used when mode=="median".
    """
    summary: Dict[str, Union[bool, float, str, list]] = {
        "applied": False,
        "mode": mode,
        "t_mid_values": list(t_mid_values),
    }
    if getattr(path, "metric_name", "") != "cosine":
        return summary
    lut = getattr(path, "learnable_lut", None)
    if lut is None:
        return summary

    device = lut.weight.device
    dtype = lut.weight.dtype
    vocab = path.vocab_size
    if vocab < 2:
        logger.warning("Cosine calibration skipped: vocab size < 2")
        return summary

    t_mids: list[float] = [
        float(t) for t in t_mid_values if isinstance(t, (int, float)) and 0.0 < float(t) < 1.0
    ]
    if not t_mids:
        t_mids = [0.5]
    summary["t_mid_values"] = t_mids

    with torch.no_grad():
        base_table = path._get_base_distance_table(device=device, dtype=dtype).to(device=device, dtype=dtype)
        cos_table_scaled = path._build_lut_distance_table(device=device, dtype=dtype).to(device=device, dtype=dtype)

    current_scale = float(getattr(path, "_lut_cosine_scale", 1.0) or 1.0)
    scale_denom = max(current_scale, 1e-12)
    cos_table_raw = cos_table_scaled / scale_denom

    idx_adj = torch.arange(vocab - 1, device=device, dtype=torch.long)
    neighbor_base = torch.median(base_table[idx_adj, idx_adj + 1]).item() if idx_adj.numel() > 0 else float("nan")
    neighbor_cos = torch.median(cos_table_raw[:, idx_adj, idx_adj + 1]).item() if idx_adj.numel() > 0 else float("nan")
    summary["neighbor_base"] = float(neighbor_base)
    summary["neighbor_cos"] = float(neighbor_cos)

    target_base = neighbor_base
    cosine_stat = neighbor_cos

    if mode == "median":
        if num_samples is None or num_samples <= 0:
            num_samples = 4096
        sample_count = int(min(vocab, num_samples))
        if sample_count < 1:
            logger.warning("Cosine calibration skipped: num_samples < 1")
            return summary

        if sample_count >= vocab:
            sample_idx = torch.arange(vocab, device=device, dtype=torch.long)
        else:
            step = max(1, vocab // sample_count)
            sample_idx = torch.arange(0, vocab, step, device=device, dtype=torch.long)[:sample_count]
            if sample_idx.numel() > 0 and sample_idx[-1].item() != vocab - 1:
                sample_idx[-1] = vocab - 1

        if sample_idx.numel() == 0:
            logger.warning("Cosine calibration skipped: no indices sampled")
            return summary

        base_rows = base_table.index_select(0, sample_idx)
        cos_rows = cos_table_raw[:, sample_idx, :]
        target_base = torch.median(base_rows).item()
        cosine_stat = torch.median(cos_rows).item()
        summary["median_base"] = float(target_base)
        summary["median_cos"] = float(cosine_stat)
    else:
        summary["median_base"] = float(target_base)
        summary["median_cos"] = float(cosine_stat)

    if not math.isfinite(cosine_stat) or cosine_stat <= 1e-12:
        logger.warning("Cosine calibration skipped: cosine_stat=%.3e", cosine_stat)
        return summary
    if not math.isfinite(target_base) or target_base <= 0.0:
        logger.warning("Cosine calibration skipped: target_base=%.3e", target_base)
        return summary

    grid_entries: list[Dict[str, float]] = []
    for t_mid in t_mids:
        t_tensor = torch.tensor([t_mid], device=device, dtype=dtype)
        beta_t, _ = path.beta(t_tensor)
        beta_value = float(beta_t.item())
        denom = max(beta_value * cosine_stat, 1e-12)
        scale_value = target_base / denom
        grid_entries.append(
            {
                "t_mid": float(t_mid),
                "beta": beta_value,
                "scale": scale_value,
                "effective_neighbor": beta_value * scale_value * cosine_stat,
                "ratio": (beta_value * scale_value * cosine_stat) / target_base,
            }
        )

    if not grid_entries:
        return summary

    summary["grid"] = grid_entries
    applied = grid_entries[0]
    path._lut_cosine_scale = float(applied["scale"])
    summary["applied"] = True
    summary["applied_scale"] = float(applied["scale"])
    summary["applied_t_mid"] = float(applied["t_mid"])
    summary["applied_beta"] = float(applied["beta"])
    summary["applied_effective_neighbor"] = float(applied["effective_neighbor"])
    summary["applied_ratio"] = float(applied["ratio"])

    logger.info(
        "Calibrated cosine LUT scale (mode=%s): base_neighbor=%.4e, raw_neighbor=%.4e, "
        "applied_t_mid=%.3f, beta=%.4f, new_scale=%.4f, effective_neighbor=%.4e, ratio=%.4f",
        mode,
        target_base,
        cosine_stat,
        applied["t_mid"],
        applied["beta"],
        applied["scale"],
        applied["effective_neighbor"],
        applied["ratio"],
    )
    if len(grid_entries) > 1:
        logger.info("t_mid grid diagnostics:")
        for entry in grid_entries:
            logger.info(
                "  t_mid=%.3f -> beta=%.4f, scale=%.4f, effective_neighbor=%.4e, ratio=%.4f",
                entry["t_mid"],
                entry["beta"],
                entry["scale"],
                entry["effective_neighbor"],
                entry["ratio"],
            )

    return summary


def _record_cosine_calibration_summary(args, summary: Optional[Dict[str, Union[bool, float, str, list]]]) -> None:
    if summary is None:
        return
    setattr(args, "_mi_lut_cosine_calibration", summary)
    base_neighbor = summary.get("neighbor_base")
    cos_neighbor = summary.get("neighbor_cos")
    if isinstance(base_neighbor, float) and math.isfinite(base_neighbor):
        setattr(args, "_mi_lut_cosine_base_neighbor", float(base_neighbor))
    if isinstance(cos_neighbor, float) and math.isfinite(cos_neighbor):
        setattr(args, "_mi_lut_cosine_neighbor_median", float(cos_neighbor))
    t_values = summary.get("t_mid_values")
    if isinstance(t_values, list):
        setattr(args, "_mi_lut_cosine_t_mid_values", [float(t) for t in t_values if isinstance(t, (int, float))])


def _compute_entropy_from_rows(dist_rows: torch.Tensor, beta: float) -> float:
    """Return mean entropy over rows given pairwise distances and beta scalar."""
    logits = (-beta) * dist_rows
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(logits, dim=-1)
    probs = probs.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return float(entropy.mean().item())


def _cosine_entropy_match_scale(
    path: MetricInducedGibbsProbPath,
    *,
    t_values: Sequence[float],
    weights: Sequence[float],
    s_lo: float = 1.0,
    s_hi: float = 200.0,
    tol: float = 0.01,
    max_iter: int = 12,
) -> Dict[str, Union[float, int, list]]:
    """Match cosine scale so weighted conditional entropy aligns with 1D baseline."""
    assert getattr(path, "metric_name", "") == "cosine"
    lut = getattr(path, "learnable_lut", None)
    if lut is None:
        raise RuntimeError("Entropy matching requires a learnable LUT.")

    # Validate t grid
    t_list = [float(t) for t in t_values if 0.0 < float(t) < 1.0]
    if not t_list:
        raise ValueError("At least one t value in (0,1) is required for entropy matching.")

    # Normalize weights
    weight_list = [float(w) for w in weights]
    if len(weight_list) not in {1, len(t_list)}:
        raise ValueError(
            "Number of weights must be 1 or match the number of t values "
            f"(got {len(weight_list)} weights for {len(t_list)} t values)."
        )
    if len(weight_list) == 1:
        weight_list = weight_list * len(t_list)
    weight_tensor = torch.tensor(weight_list, dtype=torch.float64)
    if (weight_tensor < 0).any():
        raise ValueError("Entropy matching weights must be non-negative.")
    if weight_tensor.sum().item() == 0.0:
        raise ValueError("Entropy matching weights must sum to a positive value.")
    weight_tensor = weight_tensor / weight_tensor.sum()

    vocab = int(path.vocab_size)
    device_cpu = torch.device("cpu")
    base_values = torch.linspace(-1.0, 1.0, vocab, device=device_cpu, dtype=torch.float64)
    base_diff = (base_values.unsqueeze(0) - base_values.unsqueeze(1)).abs()

    beta_list: list[float] = []
    base_entropies: list[float] = []
    for t_val in t_list:
        t_tensor = torch.tensor([t_val], dtype=torch.float64, device=device_cpu)
        beta_val = float(path.beta(t_tensor)[0].item())
        beta_list.append(beta_val)
        base_entropies.append(_compute_entropy_from_rows(base_diff, beta_val))

    weighted_base = float(torch.tensor(base_entropies, dtype=torch.float64) @ weight_tensor)

    lut_device = lut.weight.device
    lut_dtype = lut.weight.dtype
    initial_scale = float(getattr(path, "_lut_cosine_scale", 1.0))

    def eval_cosine_entropy(scale_value: float) -> tuple[float, list[float]]:
        previous_scale = float(getattr(path, "_lut_cosine_scale", 1.0))
        try:
            path._lut_cosine_scale = float(scale_value)
            with torch.no_grad():
                dist_table = path._build_lut_distance_table(device=lut_device, dtype=lut_dtype)
            dist_rows = dist_table.to(device=device_cpu, dtype=torch.float64)
            if dist_rows.dim() == 3:
                dist_rows = dist_rows.mean(dim=0)
            cos_entropies = [
                _compute_entropy_from_rows(dist_rows, beta_val) for beta_val in beta_list
            ]
            weighted_cos = float(torch.tensor(cos_entropies, dtype=torch.float64) @ weight_tensor)
            return weighted_cos, cos_entropies
        finally:
            path._lut_cosine_scale = previous_scale

    weighted_lo, _ = eval_cosine_entropy(s_lo)
    weighted_hi, _ = eval_cosine_entropy(s_hi)
    if not (weighted_lo >= weighted_base >= weighted_hi):
        # If the bracket does not contain the target, clamp to the closest endpoint.
        if abs(weighted_lo - weighted_base) < abs(weighted_hi - weighted_base):
            chosen_scale = s_lo
            weighted_cos, cos_entropies = weighted_lo, eval_cosine_entropy(s_lo)[1]
            iterations = 1
        else:
            chosen_scale = s_hi
            weighted_cos, cos_entropies = weighted_hi, eval_cosine_entropy(s_hi)[1]
            iterations = 1
        path._lut_cosine_scale = float(chosen_scale)
        path.clear_lut_cache()
        rel_error = abs(weighted_cos - weighted_base) / max(weighted_base, 1e-12)
        return {
            "scale": float(chosen_scale),
            "iterations": int(iterations),
            "t_values": t_list,
            "weights": weight_list,
            "base_entropies": base_entropies,
            "cosine_entropies": cos_entropies,
            "weighted_base": weighted_base,
            "weighted_cosine": weighted_cos,
            "relative_error": float(rel_error),
            "s_lo": float(s_lo),
            "s_hi": float(s_hi),
            "bracket_saturated": True,
        }

    lo, hi = float(s_lo), float(s_hi)
    weighted_cos = weighted_lo
    cos_entropies = eval_cosine_entropy(lo)[1]
    iterations = 0
    while iterations < max_iter:
        mid = 0.5 * (lo + hi)
        weighted_cos, cos_entropies = eval_cosine_entropy(mid)
        rel_error = abs(weighted_cos - weighted_base) / max(weighted_base, 1e-12)
        if rel_error <= tol:
            lo = hi = mid
            break
        if weighted_cos < weighted_base:
            hi = mid
        else:
            lo = mid
        iterations += 1

    chosen_scale = 0.5 * (lo + hi)
    # Re-evaluate at chosen scale for final logging
    weighted_cos, cos_entropies = eval_cosine_entropy(chosen_scale)
    path._lut_cosine_scale = float(chosen_scale)
    path.clear_lut_cache()
    rel_error = abs(weighted_cos - weighted_base) / max(weighted_base, 1e-12)
    return {
        "scale": float(chosen_scale),
        "iterations": int(iterations),
        "t_values": t_list,
        "weights": weight_list,
        "base_entropies": base_entropies,
        "cosine_entropies": cos_entropies,
        "weighted_base": weighted_base,
        "weighted_cosine": weighted_cos,
        "relative_error": float(rel_error),
        "s_lo": float(s_lo),
        "s_hi": float(s_hi),
        "bracket_saturated": False,
    }

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    # Per-rank file logging
    try:
        rank = distributed_mode.get_rank()
    except Exception:
        rank = 0
    log_dir = Path(args.output_dir)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"console_rank{rank}.log", mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(fh)

    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_train_transform()
    if args.dataset == "imagenet":
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=False,
            transform=transform_train,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    logger.info(dataset_train)

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))

    eval_batch_size = int(getattr(args, "eval_batch_size", None) or args.batch_size)
    if args.distributed:
        sampler_eval = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_eval = torch.utils.data.SequentialSampler(dataset_train)
    data_loader_eval = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_eval,
        batch_size=eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    beta_schedule: Optional[BetaSchedule] = None
    beta_schedule_ema: Optional[BetaScheduleEMA] = None
    metric_ema: Optional[LearnableMetricEMA] = None
    kl_controller: Optional[AdaptiveKLController] = None
    metric_path: Optional[MetricInducedGibbsProbPath] = None
    
    # Load metric from checkpoint if specified (before creating path)
    loaded_metric_codes: Optional[torch.Tensor] = None
    loaded_metric_source: str = "unknown"
    
    if getattr(args, "mi_init_metric_from_checkpoint", ""):
        checkpoint_path = args.mi_init_metric_from_checkpoint
        logger.info(f"[Freeze Metric] Loading metric from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # PRIORITY 1: Try EMA teacher version first (this is what eval uses!)
        # This ensures consistency with reported FID scores
        if "path" in checkpoint and "learnable_metric_ema" in checkpoint["path"]:
            ema_state = checkpoint["path"]["learnable_metric_ema"]
            if "teacher" in ema_state and "codes" in ema_state["teacher"]:
                loaded_metric_codes = ema_state["teacher"]["codes"]
                loaded_metric_source = "EMA teacher"
                logger.info(f"  ✓ Loaded EMA teacher metric codes: {loaded_metric_codes.shape}")
                logger.info("  → This is the version used during evaluation (consistent with reported FID)")
        
        # PRIORITY 2: Fall back to raw student metric if no EMA
        if loaded_metric_codes is None and "path" in checkpoint and "learnable_metric" in checkpoint["path"]:
            metric_state = checkpoint["path"]["learnable_metric"]
            loaded_metric_source = "raw student"
            logger.info("  ⚠ Warning: This may differ from eval metric (EMA teacher)")
            if "codes" in metric_state:
                loaded_metric_codes = metric_state["codes"]
                logger.info(f"  ✓ Loaded raw student metric codes: {loaded_metric_codes.shape}")
        
        # PRIORITY 3: Try model state dict
        if loaded_metric_codes is None and "model" in checkpoint:
            model_state = checkpoint["model"]
            metric_keys = [k for k in model_state.keys() if "metric" in k.lower() and "codes" in k]
            if metric_keys:
                loaded_metric_codes = model_state[metric_keys[0]]
                loaded_metric_source = f"model.{metric_keys[0]}"
                logger.info(f"  ✓ Loaded metric codes from {metric_keys[0]}: {loaded_metric_codes.shape}")
        
        # If still not found, raise error
        if loaded_metric_codes is None:
            available_keys = list(checkpoint.keys())
            if "path" in checkpoint:
                path_keys = list(checkpoint["path"].keys())
                raise KeyError(
                    f"Could not find metric codes in checkpoint.\n"
                    f"Available top-level keys: {available_keys}\n"
                    f"Available path keys: {path_keys}"
                )
            else:
                raise KeyError(f"Checkpoint structure not recognized. Available keys: {available_keys}")
        
        if args.mi_freeze_metric:
            logger.info("[Freeze Metric] Metric will be FROZEN (requires_grad=False)")
            logger.info(f"[Freeze Metric] Loaded from: {loaded_metric_source}")
            logger.info("[Freeze Metric] Only UNet parameters will be trained")
    
    if getattr(args, "ko_metric_induced", False):
        embed_range = "pm1" if args.mi_embed_range == "pm1" else "unit"
        if getattr(args, "mi_learnable_beta", False):
            schedule_type = getattr(args, "mi_beta_schedule", "bounded_rqs")
            if schedule_type == "exp_rqs":
                spline_config = ExpMonotoneRQSConfig(
                    num_bins=int(getattr(args, "mi_spline_bins", 8)),
                    tail_bound=float(getattr(args, "mi_spline_tail_bound", 6.0)),
                    init_c=float(getattr(args, "mi_c", 1.0)),
                    init_a=float(getattr(args, "mi_a", 5.0)),
                    t_eps=float(getattr(args, "mi_t_eps", 1e-4)),
                    logit_eps=float(getattr(args, "mi_logit_eps", 1e-6)),
                )
                beta_schedule = ExpMonotoneRQSSchedule(config=spline_config)
            elif schedule_type == "bounded_rqs":
                spline_config = MonotoneRQConfig(
                    num_bins=int(getattr(args, "mi_spline_bins", 8)),
                    tail_bound=float(getattr(args, "mi_spline_tail_bound", 6.0)),
                    beta_min=float(getattr(args, "mi_beta_min", 0.0)),
                    beta_max=float(getattr(args, "mi_beta_max", 20.0)),
                    t_eps=float(getattr(args, "mi_t_eps", 1e-4)),
                    logit_eps=float(getattr(args, "mi_logit_eps", 1e-6)),
                )
                beta_schedule = MonotoneRQBetaSchedule(config=spline_config)
            else:
                raise ValueError(f"Unsupported β schedule type: {schedule_type}")
            beta_schedule.to(device=device)

        gumbel_tau_default = float(getattr(args, "mi_gumbel_tau", 1.0))
        tau_start = getattr(args, "mi_gumbel_tau_start", None)
        tau_end = getattr(args, "mi_gumbel_tau_end", None)
        if tau_start is None:
            tau_start = gumbel_tau_default
        if tau_end is None:
            tau_end = gumbel_tau_default
        args.mi_gumbel_tau_start = float(tau_start)
        args.mi_gumbel_tau_end = float(tau_end)

        if getattr(args, "mi_learnable_metric", False) and getattr(args, "mi_learnable_lut", False):
            raise ValueError("Cannot enable both --mi_learnable_metric and --mi_learnable_lut at the same time.")

        metric_kwargs = {}
        if getattr(args, "mi_metric", "lp") == "cosine" and getattr(args, "mi_lut_param_mode", "none") not in ("none", None):
            logger.warning("mi_lut_param_mode has no effect when metric='cosine'; using standard LUT")
            setattr(args, "mi_lut_param_mode", "none")
        if getattr(args, "mi_learnable_metric", False):
            metric_kwargs.update(
                learnable_metric_dim=int(getattr(args, "mi_metric_dim", 0)),
                learnable_metric_diag_eps=float(getattr(args, "mi_metric_diag_eps", 1e-4)),
                metric_interp_lambda=float(getattr(args, "mi_metric_interp_start", 0.0)),
            )
        if getattr(args, "mi_learnable_lut", False):
            metric_kwargs.update(
                learnable_lut=True,
                lut_num_channels=int(getattr(args, "mi_lut_num_channels", 3)),
                lut_emb_dim=int(getattr(args, "mi_lut_emb_dim", 1)),
                lut_share_across_channels=bool(getattr(args, "mi_lut_share_channels", False)),
                lut_renorm_to_init_norm=bool(getattr(args, "mi_lut_renorm_init_norm", False)),
                lut_bounded_residual_scale=bool(getattr(args, "mi_lut_bounded_residual_scale", False)),
                lut_scale_baseline=float(getattr(args, "mi_lut_scale_baseline", 1.0)),
                lut_scale_epsilon=float(getattr(args, "mi_lut_scale_epsilon", 0.25)),
                use_normalized_distance=bool(getattr(args, "mi_use_normalized_distance", False)),
                lut_cosine_scale=float(getattr(args, "mi_lut_cosine_scale", 1.0)),
                lut_param_mode=getattr(args, "mi_lut_param_mode", "none"),
                lut_monotone_mode=str(getattr(args, "mi_lut_monotone_mode", "softplus")),
                lut_arc_radius=float(getattr(args, "mi_lut_arc_radius", 1.0)),
            )
        args.mi_metric_interp_start = float(getattr(args, "mi_metric_interp_start", 0.0))
        args.mi_metric_interp_end = float(getattr(args, "mi_metric_interp_end", 1.0))

        metric_path = MetricInducedGibbsProbPath(
            embedding_path_or_weight=None,
            vocab_size=256,
            emb_dim=1,
            metric=args.mi_metric,
            lp_order=args.mi_lp,
            embed_range=embed_range,
            a=args.mi_a,
            c=args.mi_c,
            device=device,
            dtype=torch.float32,
            beta_schedule=beta_schedule,
            use_gumbel=getattr(args, "mi_use_gumbel", False),
            gumbel_tau=float(args.mi_gumbel_tau_start),
            gumbel_hard=True,
            **metric_kwargs,
        )

        if (
            getattr(args, "mi_learnable_lut", False)
            and getattr(args, "mi_metric", "lp") == "cosine"
            and (not getattr(args, "resume", "") or getattr(args, "mi_lut_force_warm_start", False))
        ):
            _warm_start_cosine_lut(metric_path)
            if getattr(args, "mi_lut_cosine_calibrate", False):
                logger.info("Calibrating cosine LUT scale against baseline distances (warm start).")
                summary = _calibrate_cosine_scale(
                    metric_path,
                    mode=str(getattr(args, "mi_lut_cosine_calibrate_mode", "neighbor")),
                    t_mid_values=getattr(args, "mi_lut_cosine_t_mid", (0.5,)),
                )
                _record_cosine_calibration_summary(args, summary)
        
        # Apply loaded metric codes if available
        if loaded_metric_codes is not None:
            if metric_path.learnable_metric is not None:
                logger.info("[Freeze Metric] Applying loaded metric codes to path...")
                # Load the codes
                with torch.no_grad():
                    metric_path.learnable_metric.codes.copy_(loaded_metric_codes.to(device))
                logger.info(f"  ✓ Metric codes loaded: {metric_path.learnable_metric.codes.shape}")
                
                # Freeze if requested
                if args.mi_freeze_metric:
                    metric_path.learnable_metric.codes.requires_grad = False
                    logger.info("  ✓ Metric codes FROZEN (requires_grad=False)")
                    logger.info("  → Only UNet will be trained, metric geometry is fixed")
                    
                    # Force lambda to 1.0 (use learned metric fully)
                    metric_path.set_metric_interpolation_lambda(1.0)
                    logger.info("  → Metric interpolation lambda forced to 1.0")
            else:
                logger.warning("[Freeze Metric] Loaded metric codes but path has no learnable_metric!")
        
        if getattr(args, "mi_learnable_metric", False):
            metric_path.set_metric_interpolation_lambda(args.mi_metric_interp_start)
        kl_target = float(getattr(args, "mi_beta_kl_target", 0.0))
        kl_init_weight = float(getattr(args, "mi_beta_kl_init_weight", 0.0))

        # Create EMA even when schedule is frozen (EMA provides smooth training target)
        # Freeze only prevents gradient updates to raw parameters
        freeze_schedule = getattr(args, "mi_freeze_beta_schedule", False)
        
        if (
            getattr(args, "mi_learnable_beta", False)
            and getattr(args, "mi_beta_use_ema", False)
            and isinstance(metric_path.beta_schedule, nn.Module)
        ):
            beta_schedule_ema = BetaScheduleEMA(
                metric_path.beta_schedule,
                decay=float(getattr(args, "mi_beta_ema_decay", 0.999)),
            )
            beta_schedule_ema.to(device=device)
            beta_schedule_ema.synchronize_from(metric_path.beta_schedule)
            if freeze_schedule:
                logger.info("Created β schedule EMA (schedule params frozen, EMA provides smooth training target)")
            else:
                logger.info("Created β schedule EMA (will track student schedule)")

        if (
            getattr(args, "mi_learnable_metric", False)
            and getattr(args, "mi_metric_use_ema", False)
            and metric_path.learnable_metric is not None
        ):
            metric_ema = LearnableMetricEMA(
                metric_path.learnable_metric,
                decay=float(getattr(args, "mi_metric_ema_decay", 0.999)),
            )
            metric_ema.to(device=device)
            metric_ema.synchronize_from(metric_path.learnable_metric)

        # KL controller monitors EMA vs raw schedule divergence
        # But in freeze mode, raw has no gradient → KL penalty has no effect → disable it
        if (
            kl_target > 0.0
            and kl_init_weight > 0.0
            and (beta_schedule_ema is not None or metric_ema is not None)
            and not freeze_schedule  # Disable KL in freeze mode
        ):
            kl_controller = AdaptiveKLController(
                target=kl_target,
                init_weight=kl_init_weight,
                adapt_rate=float(getattr(args, "mi_beta_kl_adapt_rate", 2.0)),
                tolerance=float(getattr(args, "mi_beta_kl_tolerance", 1.5)),
                min_weight=float(getattr(args, "mi_beta_kl_min_weight", 1e-4)),
                max_weight=float(getattr(args, "mi_beta_kl_max_weight", 1e4)),
            )
            kl_controller.to(device=device)
            logger.info("Enabled schedule KL trust region")
        elif freeze_schedule and kl_target > 0.0:
            # CHANGED: Enable KL controller even when schedule is frozen
            # Reason: We now have learnable metric that can benefit from KL penalty
            # The penalty will affect the metric parameters (not schedule)
            kl_controller = AdaptiveKLController(
                target=kl_target,
                init_weight=kl_init_weight,
                adapt_rate=float(getattr(args, "mi_beta_kl_adapt_rate", 2.0)),
                tolerance=float(getattr(args, "mi_beta_kl_tolerance", 1.5)),
                min_weight=float(getattr(args, "mi_beta_kl_min_weight", 1e-4)),
                max_weight=float(getattr(args, "mi_beta_kl_max_weight", 1e4)),
            )
            kl_controller.to(device=device)
            logger.info(
                "Freeze mode: KL controller ENABLED (will regularize learnable metric, "
                "even though schedule is frozen)"
            )

    # define the model
    logger.info("Initializing Model")
    model = instantiate_model(
        architechture=args.dataset,
        is_discrete=args.discrete_flow_matching,
        ko=getattr(args, "ko_metric_induced", False),
        use_ema=args.use_ema,
        use_cosine_attention=getattr(args, "cosine_attention", False),
        use_head_weight_norm=getattr(args, "head_weight_norm", False),
    )

    model.to(device)

    model_without_ddp = model
    logger.info(str(model_without_ddp))

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module
        setattr(model, "weight_norm_targets", getattr(model_without_ddp, "weight_norm_targets", []))

    optimizer_param_groups = [{"params": list(model_without_ddp.parameters())}]
    extra_modules = {}
    if metric_path is not None:
        freeze_schedule = getattr(args, "mi_freeze_beta_schedule", False)
        
        # Only add schedule parameters to optimizer if not frozen
        schedule_params = list(metric_path.schedule_parameters())
        if schedule_params and not freeze_schedule:
            schedule_lr_scale = float(getattr(args, "mi_beta_lr_scale", 1.0))
            optimizer_param_groups.append(
                {
                    "params": schedule_params,
                    "lr": args.lr * schedule_lr_scale,
                    "name": "beta_schedule",
                }
            )
            logger.info(f"Added {len(schedule_params)} schedule parameters to optimizer (lr_scale={schedule_lr_scale})")
        elif schedule_params and freeze_schedule:
            # Freeze schedule parameters
            for param in schedule_params:
                param.requires_grad = False
            logger.info(f"FROZEN {len(schedule_params)} schedule parameters (excluded from optimizer)")
        
        metric_params = list(metric_path.metric_parameters())
        if metric_params:
            freeze_metric = getattr(args, "mi_freeze_metric", False)
            if not freeze_metric:
                metric_lr_scale = float(getattr(args, "mi_metric_lr_scale", 0.1))
                metric_weight_decay = float(getattr(args, "mi_metric_weight_decay", 1e-4))
                optimizer_param_groups.append(
                    {
                        "params": metric_params,
                        "lr": args.lr * metric_lr_scale,
                        "weight_decay": metric_weight_decay,
                        "name": "learnable_metric",
                    }
                )
                logger.info(
                    f"Added {len(metric_params)} metric parameters to optimizer (lr_scale={metric_lr_scale}, weight_decay={metric_weight_decay})"
                )
            else:
                for param in metric_params:
                    param.requires_grad = False
                logger.info(f"EXCLUDED {len(metric_params)} metric parameters from optimizer (frozen)")
                logger.info("  → Only model (UNet) parameters will be optimized")

        lut_params = list(metric_path.lut_parameters())
        if lut_params:
            freeze_lut = getattr(args, "mi_freeze_lut", False)
            if not freeze_lut:
                lut_lr_scale = float(getattr(args, "mi_lut_lr_scale", 0.1))
                lut_weight_decay = float(getattr(args, "mi_lut_weight_decay", 1e-4))
                optimizer_param_groups.append(
                    {
                        "params": lut_params,
                        "lr": args.lr * lut_lr_scale,
                        "weight_decay": lut_weight_decay,
                        "name": "learnable_lut",
                    }
                )
                logger.info(
                    f"Added {len(lut_params)} LUT parameters to optimizer (lr_scale={lut_lr_scale}, weight_decay={lut_weight_decay})"
                )
            else:
                for param in lut_params:
                    param.requires_grad = False
                logger.info(f"EXCLUDED {len(lut_params)} LUT parameters from optimizer (frozen)")
                logger.info("  → Only model (UNet) parameters will be optimized")
        if isinstance(metric_path.beta_schedule, nn.Module):
            extra_modules["metric_beta_schedule"] = metric_path.beta_schedule
        if metric_path.learnable_metric is not None:
            extra_modules["metric_learnable_metric"] = metric_path.learnable_metric
        if beta_schedule_ema is not None:
            extra_modules["metric_beta_schedule_ema"] = beta_schedule_ema
        if metric_ema is not None:
            extra_modules["metric_learnable_metric_ema"] = metric_ema
        if kl_controller is not None:
            extra_modules["metric_beta_kl_controller"] = kl_controller
    optimizer = torch.optim.AdamW(
        optimizer_param_groups, lr=args.lr, betas=args.optimizer_betas
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    if getattr(args, "bf16", False):
        logger.info("bf16 autocast enabled: disabling GradScaler and using bfloat16 forwards.")
    loss_scaler = NativeScaler(enabled=not getattr(args, "bf16", False))

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
        path=metric_path,
        extra_modules=extra_modules,
    )
    if (
        getattr(args, "mi_lut_force_warm_start", False)
        and getattr(args, "mi_learnable_lut", False)
        and getattr(args, "mi_metric", "lp") == "cosine"
        and metric_path is not None
    ):
        logger.info("Forcing cosine LUT warm start after loading checkpoint.")
        _warm_start_cosine_lut(metric_path)
    if (
        getattr(args, "mi_metric", "lp") == "cosine"
        and getattr(args, "mi_lut_cosine_calibrate", False)
        and metric_path is not None
    ):
        logger.info("Calibrating cosine LUT scale against baseline distances (post-load).")
        summary = _calibrate_cosine_scale(
            metric_path,
            mode=str(getattr(args, "mi_lut_cosine_calibrate_mode", "neighbor")),
            t_mid_values=getattr(args, "mi_lut_cosine_t_mid", (0.5,)),
        )
        _record_cosine_calibration_summary(args, summary)
    entropy_summary: Optional[Dict[str, Union[float, int, list]]] = None
    if (
        getattr(args, "mi_metric", "lp") == "cosine"
        and getattr(args, "mi_lut_cosine_entropy_match", False)
        and metric_path is not None
        and getattr(metric_path, "learnable_lut", None) is not None
    ):
        if distributed_mode.is_main_process():
            try:
                entropy_summary = _cosine_entropy_match_scale(
                    metric_path,
                    t_values=getattr(args, "mi_lut_cosine_entropy_t_mid", (0.3, 0.5, 0.7)),
                    weights=getattr(args, "mi_lut_cosine_entropy_weights", (0.2, 0.6, 0.2)),
                    s_lo=1.0,
                    s_hi=200.0,
                    tol=float(getattr(args, "mi_lut_cosine_entropy_tol", 0.01)),
                    max_iter=int(getattr(args, "mi_lut_cosine_entropy_max_iter", 12)),
                )
                logger.info(
                    "Entropy-matched cosine scale: %.4f (iters=%d, rel_err=%.3e)",
                    entropy_summary["scale"],
                    entropy_summary["iterations"],
                    entropy_summary["relative_error"],
                )
            except Exception:
                logger.exception("Cosine entropy matching failed; keeping current scale.")
                entropy_summary = None
        final_scale = float(getattr(metric_path, "_lut_cosine_scale", 1.0))
        if distributed_mode.is_dist_avail_and_initialized():
            import torch.distributed as dist

            scale_tensor = torch.tensor([final_scale], device=device, dtype=torch.float32)
            dist.broadcast(scale_tensor, src=0)
            final_scale = float(scale_tensor.item())
        metric_path._lut_cosine_scale = float(final_scale)
        metric_path.clear_lut_cache()
        if entropy_summary is not None:
            setattr(args, "_mi_lut_cosine_entropy_summary", entropy_summary)
    if (
        beta_schedule_ema is not None
        and metric_path is not None
        and isinstance(metric_path.beta_schedule, BetaSchedule)
        and beta_schedule_ema.num_updates.item() == 0
    ):
        beta_schedule_ema.synchronize_from(metric_path.beta_schedule)
    if (
        metric_ema is not None
        and metric_path is not None
        and metric_path.learnable_metric is not None
        and metric_ema.num_updates.item() == 0
    ):
        metric_ema.synchronize_from(metric_path.learnable_metric)

    # Optional Weights & Biases
    wandb_run = None
    if getattr(args, "wandb", False):
        try:
            import swanlab as wandb  # type: ignore
            if distributed_mode.is_main_process():
                wandb_kwargs = dict(
                    project=getattr(args, "wandb_project", "flow_matching"),
                    name=getattr(args, "wandb_run_name", None),
                    entity=getattr(args, "wandb_entity", None),
                    dir=args.output_dir,
                    config=vars(args),
                )
                if getattr(args, "wandb_offline", False):
                    os.environ.setdefault("WANDB_MODE", "offline")
                wandb_run = wandb.init(**{k: v for k, v in wandb_kwargs.items() if v is not None}, mode="offline" if getattr(args, "wandb_offline", False) else "online")
        except Exception as e:
            logger.warning(f"wandb not enabled ({e})")

    # Compute display-only epoch mapping
    display_offset: Optional[int] = None
    if getattr(args, "force_display_start_epoch", None) is not None:
        try:
            display_offset = int(args.force_display_start_epoch) - int(args.start_epoch)
        except Exception:
            display_offset = None
    elif getattr(args, "epoch_display_offset", None) is not None:
        try:
            display_offset = int(args.epoch_display_offset)
        except Exception:
            display_offset = None

    if display_offset is not None:
        logger.info(
            f"Epoch display mapping enabled: display_epoch = epoch + ({display_offset}). "
            f"Raw training range: [{args.start_epoch}, {args.epochs})"
        )
    else:
        logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        display_epoch = epoch + (display_offset or 0)
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # Adjust metric learning rate (before training if enabled)
        if not args.eval_only:
            from training.train_loop import adjust_metric_learning_rate
            adjust_metric_learning_rate(optimizer, epoch, args)
        
        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
                path=metric_path,
                schedule_ema=beta_schedule_ema,
                metric_ema=metric_ema,
                kl_controller=kl_controller,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "display_epoch": display_epoch,
            }
        else:
            log_stats = {
                "epoch": epoch,
                "display_epoch": display_epoch,
            }

        eval_start = int(getattr(args, "eval_start_epoch", 0))
        should_eval = False
        if args.eval_frequency > 0:
            if (epoch + 1) >= eval_start:
                relative = (epoch + 1) - eval_start
                if relative % args.eval_frequency == 0:
                    should_eval = True
        if args.eval_only or args.test_run:
            should_eval = True
        if args.output_dir and should_eval:
            if not args.eval_only:
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    path=metric_path,
                    extra_modules=extra_modules,
                )
                # Optionally create display-epoch symlink for convenience in plotting
                try:
                    if getattr(args, "save_display_epoch_symlinks", False) and display_offset is not None:
                        ckpt_dir = Path(args.output_dir)
                        src = ckpt_dir / f"checkpoint-{epoch}.pth"
                        dst = ckpt_dir / f"checkpoint-{display_epoch}.pth"
                        if src.exists():
                            # Avoid overwriting real files; only make symlink if dst absent
                            if not dst.exists():
                                try:
                                    # On systems without symlink perms, fall back to hardlink or copy
                                    dst.symlink_to(src.name)
                                except Exception:
                                    try:
                                        os.link(src, dst)  # type: ignore[attr-defined]
                                    except Exception:
                                        import shutil
                                        shutil.copy2(src, dst)
                            logger.info(f"Display-epoch alias created: {dst.name} -> {src.name}")
                except Exception as _e:
                    logger.warning(f"Could not create display-epoch alias: {_e}")
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)
                eval_sampler = getattr(data_loader_eval, "sampler", None)
                if isinstance(eval_sampler, torch.utils.data.DistributedSampler):
                    eval_sampler.set_epoch(0)
            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                fid_samples = args.fid_samples // num_tasks
            try:
                eval_stats = eval_model(
                    model,
                    data_loader_eval,
                    device,
                    epoch=epoch,
                    fid_samples=fid_samples,
                    args=args,
                    metric_path=metric_path,
                    metric_ema=metric_ema,
                    schedule_ema=beta_schedule_ema,  # Pass EMA for eval
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            eval_prefixed = {f"eval_{k}": v for k, v in eval_stats.items()}
            if display_offset is not None:
                eval_prefixed["eval_display_epoch"] = display_epoch
            log_stats.update(eval_prefixed)

        # Log to wandb (only on main process)
        if wandb_run is not None and distributed_mode.is_main_process():
            try:
                import swanlab as wandb  # type: ignore

                wandb_payload: Dict[str, Union[int, float]] = {}
                for key, value in log_stats.items():
                    if isinstance(value, (int, float)):
                        wandb_payload[key] = float(value)
                    elif isinstance(value, np.generic):  # type: ignore[arg-type]
                        wandb_payload[key] = float(value.item())

                if wandb_payload:
                    wandb_step = display_epoch if display_offset is not None else epoch
                    wandb.log(wandb_payload, step=wandb_step)
            except Exception as e:
                logger.warning(f"wandb.log failed: {e}")

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")

    if 'wandb' in globals() and wandb_run is not None and distributed_mode.is_main_process():
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    # Backward-compat: map consolidated flag to legacy alias expected elsewhere
    if not hasattr(args, "ko_metric_induced"):
        setattr(args, "ko_metric_induced", getattr(args, "metric_induced", False))
    
    # CRITICAL: Reset annealing step counters when resuming from baseline
    # This ensures Gumbel tau and metric interpolation start fresh
    # even when resuming from a checkpoint at epoch > 0
    if args.resume:
        setattr(args, "_gumbel_update_step", 0)
        setattr(args, "_metric_interp_update_step", 0)
        setattr(args, "_annealing_counters_reset", True)  # Flag to prevent override
        print(
            "=" * 80 + "\n"
            "RESET ANNEALING COUNTERS TO 0\n"
            "  - Gumbel tau will anneal from start\n"
            "  - Metric interpolation will anneal from start\n"
            "  - Resuming from checkpoint but with fresh annealing schedules\n"
            + "=" * 80
        )
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
