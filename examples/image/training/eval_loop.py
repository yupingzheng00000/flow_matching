"""
Path bootstrap
Ensures the top-level 'flow_matching' package is importable when running
this module from within 'examples/image' (e.g., via torchrun).
"""
import sys as _sys
from pathlib import Path as _Path

_this_dir = _Path(__file__).resolve().parent
# Go up three levels: .../flow_matching/examples/image/training -> .../flow_matching
_pkg_root = _this_dir.parents[2]
if str(_pkg_root) not in _sys.path:
    _sys.path.insert(0, str(_pkg_root))

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import csv
import gc
import logging
import math
import os
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Union, cast

import PIL.Image

import torch
import torch.distributed as dist

def _autocast_cuda():
    """Return an autocast context manager for CUDA with torch.amp if available, else torch.cuda.amp."""
    try:
        from torch import amp as _amp  # type: ignore
        return _amp.autocast("cuda")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return torch.cuda.amp.autocast()
from flow_matching.path import MixtureDiscreteProbPath, MetricInducedGibbsProbPath
from flow_matching.path.metric_ema import LearnableMetricEMA
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver, KODiscreteGibbsEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image, make_grid
from training import distributed_mode
from training.edm_time_discretization import get_time_discretization
from training.train_loop import MASK_TOKEN

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def _save_sampling_gif(
    trajectories: torch.Tensor,
    output_root: Path,
    epoch: int,
    step: int,
    *,
    is_discrete: bool,
    max_batch: int,
    stride: int,
    fps: int,
    log_to_wandb: bool,
    wandb_step: int,
) -> Optional[Path]:
    """Persist a GIF visualizing sampling trajectories as a tiled grid."""

    if trajectories.ndim < 4:
        logger.debug(
            "Skipping GIF export because trajectory tensor has unexpected rank %d",
            trajectories.ndim,
        )
        return None

    max_batch = max(1, int(max_batch))
    stride = max(1, int(stride))
    fps = max(1, int(fps))

    frames = trajectories.detach().to(device="cpu", dtype=torch.float32)[::stride]
    if frames.shape[0] == 0:
        logger.debug("Skipping GIF export because no frames remain after striding")
        return None

    frames = frames[:, :max_batch]
    if frames.shape[1] == 0:
        logger.debug(
            "Skipping GIF export because max_batch=%d removed all samples", max_batch
        )
        return None

    if frames.ndim == 4:
        frames = frames.unsqueeze(2)

    if is_discrete:
        frames = frames / 255.0
    else:
        frames = torch.clamp(frames, -1.0, 1.0) * 0.5 + 0.5
    frames = torch.clamp(frames, 0.0, 1.0)

    num_samples = frames.shape[1]
    nrow = int(math.sqrt(num_samples))
    if nrow * nrow < num_samples:
        nrow += 1
    nrow = max(1, nrow)

    frame_images = []
    for frame in frames:
        grid = make_grid(frame, nrow=nrow, padding=2)
        if grid.shape[0] == 1:
            grid = grid.repeat(3, 1, 1)
        elif grid.shape[0] == 2:
            grid = torch.cat((grid, grid[:1]), dim=0)
        elif grid.shape[0] > 3:
            grid = grid[:3]
        grid = torch.clamp(grid, 0.0, 1.0)
        grid_np = (
            (grid * 255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        frame_images.append(PIL.Image.fromarray(grid_np))

    if not frame_images:
        logger.debug("Skipping GIF export because no frame images were generated")
        return None

    gif_dir = output_root / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / f"epoch_{epoch:04d}_step_{step:04d}.gif"
    duration_ms = max(1, int(1000 / fps))
    frame_images[0].save(
        gif_path,
        save_all=True,
        append_images=frame_images[1:],
        duration=duration_ms,
        loop=0,
    )

    if log_to_wandb:
        try:
            try:
                import swanlab as wandb  # type: ignore
            except Exception:  # pragma: no cover - swanlab not installed
                import wandb  # type: ignore

            if hasattr(wandb, "Video"):
                wandb.log(  # type: ignore[attr-defined]
                    {
                        "eval/sample_gif": wandb.Video(  # type: ignore[attr-defined]
                            str(gif_path), fps=fps, format="gif"
                        )
                    },
                    step=wandb_step,
                )
        except Exception as wandb_exc:  # pragma: no cover - wandb unavailable
            logger.debug("Unable to log evaluation GIF to wandb: %s", wandb_exc)

    logger.info("Saved evaluation GIF to %s", gif_path)
    return gif_path


def _pad_samples_to_square_grid(samples: torch.Tensor) -> torch.Tensor:
    """Pad a batch of images by repeating early samples to fill a square grid."""

    if samples.ndim != 4:
        return samples
    batch = samples.shape[0]
    if batch == 0:
        return samples

    grid = int(math.ceil(math.sqrt(batch)))
    target = grid * grid
    if target <= batch:
        return samples

    pad = target - batch
    if pad <= 0:
        return samples

    repeat = samples[:pad]
    if repeat.numel() == 0:
        return samples

    return torch.cat((samples, repeat), dim=0)


def _select_metric_eval_indices(vocab_size: int, subset: int) -> torch.Tensor:
    subset = int(subset)
    if subset <= 0 or subset >= vocab_size:
        return torch.arange(vocab_size, dtype=torch.long)
    return torch.arange(subset, dtype=torch.long)


def _upper_triangle_values(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    n = matrix.shape[0]
    if n <= 1:
        return matrix.new_empty(0)
    idx = torch.triu_indices(n, n, offset=1, device=matrix.device)
    return matrix[idx[0], idx[1]]


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float64, device=values.device)
    unique_vals, inverse, counts = torch.unique(
        values, sorted=True, return_inverse=True, return_counts=True
    )
    if torch.any(counts > 1):
        cumsum = torch.cumsum(counts, dim=0)
        start = torch.cat((counts.new_zeros(1), cumsum[:-1]), dim=0)
        avg = (start + cumsum - 1).to(torch.float64) / 2.0
        ranks += avg[inverse] - ranks
    return ranks


def _spearman_corrcoef(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    if x.numel() != y.numel() or x.numel() < 2:
        return None
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    x_std = x_rank.std(unbiased=False)
    y_std = y_rank.std(unbiased=False)
    denom = x_std * y_std
    denom_val = float(denom.item()) if denom.numel() == 1 else float(denom)
    if denom_val <= 0.0 or not math.isfinite(denom_val):
        return None
    cov = torch.mean(x_rank * y_rank)
    cov_val = float(cov.item()) if cov.numel() == 1 else float(cov)
    if not math.isfinite(cov_val):
        return None
    return cov_val / denom_val


def _knn_indices(dist: torch.Tensor, k: int) -> Optional[torch.Tensor]:
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist must be square")
    n = dist.shape[0]
    if n <= 1 or k <= 0:
        return None
    k = min(k, n - 1)
    dist_clone = dist.clone()
    eye = torch.eye(n, dtype=torch.bool, device=dist_clone.device)
    dist_clone[eye] = float("inf")
    _, indices = torch.topk(dist_clone, k=k, dim=1, largest=False)
    return indices


def _knn_overlap(base: torch.Tensor, other: torch.Tensor, k: int) -> Optional[float]:
    base_idx = _knn_indices(base, k)
    other_idx = _knn_indices(other, k)
    if base_idx is None or other_idx is None:
        return None
    n = base_idx.shape[0]
    device = base_idx.device
    mask_base = torch.zeros(n, base.shape[0], dtype=torch.bool, device=device)
    rows = torch.arange(n, device=device).unsqueeze(1).expand_as(base_idx)
    mask_base[rows, base_idx] = True
    mask_other = torch.zeros_like(mask_base)
    mask_other[rows, other_idx] = True
    intersection = torch.logical_and(mask_base, mask_other).sum(dim=1)
    union = torch.logical_or(mask_base, mask_other).sum(dim=1).clamp_min(1)
    overlap = (intersection.to(torch.float64) / union.to(torch.float64)).mean()
    return float(overlap.item())


def _export_metric_heatmap(
    base: torch.Tensor,
    other: torch.Tensor,
    *,
    output_dir: Path,
    epoch: int,
    label: str,
    log_to_wandb: bool,
    wandb_step: int,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib optional
        logger.warning(
            "Skipping metric heatmap for %s because matplotlib is unavailable: %s",
            label,
            exc,
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    base_np = base.detach().cpu().numpy()
    other_np = other.detach().cpu().numpy()
    delta_np = other_np - base_np

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(
        axes,
        (base_np, other_np, delta_np),
        ("baseline", label, f"{label} - baseline"),
    ):
        im = ax.imshow(data, cmap="magma")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Metric geometry ({label}) epoch {epoch}")
    fig.tight_layout()

    file_path = output_dir / f"epoch_{epoch:04d}_{label}.png"
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)

    if log_to_wandb:
        try:
            try:
                import swanlab as wandb  # type: ignore
            except Exception:  # pragma: no cover - swanlab not installed
                import wandb  # type: ignore

            if hasattr(wandb, "Image"):
                wandb.log(  # type: ignore[attr-defined]
                    {f"eval/metric_heatmap_{label}": wandb.Image(str(file_path))},  # type: ignore[attr-defined]
                    step=wandb_step,
                )
        except Exception as wandb_exc:  # pragma: no cover - wandb unavailable
            logger.debug(
                "Unable to log metric heatmap %s to wandb: %s", label, wandb_exc
            )

    logger.info("Saved metric heatmap (%s) to %s", label, file_path)
    return file_path


def _evaluate_metric_geometry(
    path: MetricInducedGibbsProbPath,
    *,
    args: Namespace,
    epoch: int,
    metric_ema: Optional[LearnableMetricEMA],
    output_root: Optional[Path],
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    vocab_size = int(path.vocab_size)
    subset_cfg = int(getattr(args, "mi_metric_eval_subset", 0))
    subset_idx = _select_metric_eval_indices(vocab_size, subset_cfg)
    subset_count = int(subset_idx.numel())

    stats["metric_geom_vocab"] = float(vocab_size)
    stats["metric_geom_tokens"] = float(subset_count)
    lam = float(path.get_metric_interpolation_lambda())
    stats["metric_geom_lambda"] = lam

    eval_device = torch.device("cpu")
    eval_dtype = torch.float32

    base_full = path._get_base_distance_table(device=eval_device, dtype=eval_dtype)
    base_subset = (
        base_full.index_select(0, subset_idx)
        .index_select(1, subset_idx)
        .to(dtype=torch.float64)
    )

    tables: Dict[str, torch.Tensor] = {}
    student_full = path._learned_distance_table(
        device=eval_device, dtype=eval_dtype
    )
    if student_full is not None:
        tables["student"] = (
            student_full.index_select(0, subset_idx)
            .index_select(1, subset_idx)
            .to(dtype=torch.float64)
        )

    teacher_module = getattr(metric_ema, "teacher", None) if metric_ema else None
    if isinstance(teacher_module, torch.nn.Module):
        teacher_full = path._learned_distance_table(
            device=eval_device,
            dtype=eval_dtype,
            metric_module=teacher_module,
        )
        if teacher_full is not None:
            tables["teacher"] = (
                teacher_full.index_select(0, subset_idx)
                .index_select(1, subset_idx)
                .to(dtype=torch.float64)
            )

    if lam > 0.0:
        if "student" in tables:
            if lam >= 1.0:
                tables["blended"] = tables["student"]
            else:
                tables["blended"] = (
                    (1.0 - lam) * base_subset + lam * tables["student"]
                )
        else:
            tables["blended"] = base_subset

    base_pairs_all = _upper_triangle_values(base_subset)
    pair_total = int(base_pairs_all.numel())
    stats["metric_geom_pairs_total"] = float(pair_total)

    pair_sample_cfg = max(0, int(getattr(args, "mi_metric_eval_pair_samples", 0)))
    sample_indices = None
    if pair_total >= 2 and 0 < pair_sample_cfg < pair_total:
        generator = torch.Generator(device=base_pairs_all.device)
        generator.manual_seed(int(getattr(args, "mi_metric_eval_seed", 0)))
        perm = torch.randperm(pair_total, generator=generator)
        sample_indices = perm[:pair_sample_cfg]
        stats["metric_geom_pairs_used"] = float(sample_indices.numel())
    else:
        stats["metric_geom_pairs_used"] = float(pair_total)

    if pair_total < 2:
        logger.warning(
            "Not enough off-diagonal pairs (%d) to compute Spearman correlation.",
            pair_total,
        )
    else:
        base_pairs = (
            base_pairs_all
            if sample_indices is None
            else base_pairs_all.index_select(0, sample_indices)
        )
        for name, table in tables.items():
            other_pairs = _upper_triangle_values(table)
            if sample_indices is not None:
                other_pairs = other_pairs.index_select(0, sample_indices)
            rho = _spearman_corrcoef(base_pairs, other_pairs)
            if rho is not None:
                stats[f"metric_geom_spearman_{name}"] = rho

    k_cfg = max(0, int(getattr(args, "mi_metric_eval_knn_k", 5)))
    if subset_count > 1 and k_cfg > 0:
        k_eff = min(k_cfg, subset_count - 1)
        stats["metric_geom_knn_k"] = float(k_eff)
        for key in ("student", "teacher", "blended"):
            table = tables.get(key)
            if table is None:
                continue
            overlap = _knn_overlap(base_subset, table, k_eff)
            if overlap is not None:
                stats[f"metric_geom_knn_overlap_{key}@{k_eff}"] = overlap
    else:
        stats["metric_geom_knn_k"] = 0.0

    if getattr(args, "mi_metric_eval_heatmap", False):
        if output_root is None:
            logger.warning(
                "Skipping metric geometry heatmaps because --output_dir is not set."
            )
        elif subset_count < 2:
            logger.warning(
                "Skipping metric geometry heatmaps because the evaluated subset has < 2 tokens."
            )
        else:
            heatmap_limit = int(getattr(args, "mi_metric_eval_heatmap_subset", 0))
            heatmap_count = (
                subset_count if heatmap_limit <= 0 else min(heatmap_limit, subset_count)
            )
            heatmap_dir = output_root / "metric_geometry"
            base_heatmap = base_subset[:heatmap_count, :heatmap_count].to(torch.float32)
            for key in ("student", "teacher"):
                table = tables.get(key)
                if table is None:
                    continue
                _export_metric_heatmap(
                    base_heatmap,
                    table[:heatmap_count, :heatmap_count].to(torch.float32),
                    output_dir=heatmap_dir,
                    epoch=epoch,
                    label=key,
                    log_to_wandb=bool(getattr(args, "wandb", False)),
                    wandb_step=epoch,
                )

    metric_keys = [
        key
        for key in stats.keys()
        if key.startswith("metric_geom_spearman")
        or key.startswith("metric_geom_knn_overlap")
    ]
    if metric_keys:
        summary = ", ".join(
            f"{key}={stats[key]:.4f}" for key in sorted(metric_keys)
        )
        logger.info(
            "Metric geometry diagnostics (tokens=%d, λ=%.4f): %s",
            subset_count,
            lam,
            summary,
        )

    return stats

class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module, return_logits: bool = False):
        super().__init__(model)
        self.nfe_counter = 0
        # If True and model is discrete, return raw logits instead of softmax probabilities
        self.return_logits = return_logits

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor
    ):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        is_discrete = isinstance(module, DiscreteUNetModel) or (
            isinstance(module, EMA) and isinstance(module.model, DiscreteUNetModel)
        )
        assert (
            cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with _autocast_cuda(), torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with _autocast_cuda(), torch.no_grad():
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        if is_discrete:
            out = result.to(dtype=torch.float32)
            return out if self.return_logits else torch.softmax(out, dim=-1)
        else:
            return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
    metric_path: Optional[MetricInducedGibbsProbPath] = None,
    metric_ema: Optional[LearnableMetricEMA] = None,
):
    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    # For KO solver we need logits; instantiate a logits-returning view lazily
    cfg_scaled_logits_model = None
    cfg_scaled_model.train(False)

    if args.discrete_flow_matching:
        # Branch between mixture path (Meta) and metric-induced path (KO-style)
        if getattr(args, "metric_induced", False):
            disc_solver = None  # KO solver set up lazily below
        else:
            scheduler = PolynomialConvexScheduler(n=3.0)
            path = MixtureDiscreteProbPath(scheduler=scheduler)
            p = torch.zeros(size=[257], dtype=torch.float32, device=device)
            p[256] = 1.0
            disc_solver = MixtureDiscreteEulerSolver(
                model=cfg_scaled_model,
                path=path,
                vocabulary_size=257,
                source_distribution_p=p,
            )
        cont_solver = None
        cont_ode_opts = None
    else:
        disc_solver = None
        cont_solver = ODESolver(velocity_model=cfg_scaled_model)
        cont_ode_opts = args.ode_options

    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )

    num_synthetic = 0
    num_real = 0
    snapshots_saved = False
    gif_logged = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    # Try to get the length for logging; fall back gracefully if unknown
    try:
        _data_loader_len_for_log = len(data_loader)  # type: ignore[arg-type]
    except Exception:
        _data_loader_len_for_log = None

    # Lazily constructed KO solver and path (once K is known)
    ko_solver = None
    ko_path = metric_path
    schedule_snapshot_logged = False
    geometry_stats: Dict[str, float] = {}
    geometry_logged = False

    def maybe_run_metric_geometry(path_obj: Optional[MetricInducedGibbsProbPath]) -> None:
        nonlocal geometry_logged, geometry_stats
        if geometry_logged:
            return
        if not getattr(args, "mi_metric_eval_geometry", False):
            geometry_logged = True
            return
        if path_obj is None:
            return
        if not distributed_mode.is_main_process():
            geometry_logged = True
            return

        output_root = (
            Path(getattr(args, "output_dir"))
            if getattr(args, "output_dir", None)
            else None
        )
        try:
            geometry_stats = _evaluate_metric_geometry(
                path_obj,
                args=args,
                epoch=epoch,
                metric_ema=metric_ema,
                output_root=output_root,
            )
        except Exception as geom_exc:  # pragma: no cover - diagnostic failures
            logger.warning("Metric geometry evaluation failed: %s", geom_exc)
        geometry_logged = True

    def maybe_log_schedule_snapshot(path_obj: MetricInducedGibbsProbPath) -> None:
        nonlocal schedule_snapshot_logged
        if schedule_snapshot_logged:
            return
        if not getattr(args, "mi_beta_log_schedule", False):
            schedule_snapshot_logged = True
            return
        if not distributed_mode.is_main_process():
            schedule_snapshot_logged = True
            return
        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            logger.warning("Skipping β(t) snapshot logging because --output_dir is not set.")
            schedule_snapshot_logged = True
            return
        num_points = int(getattr(args, "mi_beta_log_points", 256))
        if num_points <= 1:
            logger.warning(
                "Skipping β(t) snapshot logging because --mi_beta_log_points must be > 1."
            )
            schedule_snapshot_logged = True
            return

        schedule_dir = Path(output_dir) / "beta_schedule_logs"
        schedule_dir.mkdir(parents=True, exist_ok=True)
        device = path_obj.embedding.weight.device
        dtype = path_obj.embedding.weight.dtype
        eps = max(float(path_obj.eps_t), 1e-8)
        t = torch.linspace(eps, 1.0 - eps, steps=num_points, device=device, dtype=torch.float32)
        if t.dtype != dtype:
            t = t.to(dtype=dtype)

        with torch.no_grad():
            beta_curr, dot_curr = path_obj.beta(t)

        t_cpu = t.detach().cpu().double()
        beta_curr_cpu = beta_curr.detach().cpu().double()
        dot_curr_cpu = dot_curr.detach().cpu().double()

        denom = t_cpu * (1.0 - t_cpu)
        beta_baseline = path_obj.c * torch.pow(t_cpu / (1.0 - t_cpu), path_obj.a)
        dot_baseline = beta_baseline * path_obj.a / denom

        data = torch.stack(
            (
                t_cpu,
                beta_curr_cpu,
                beta_baseline,
                dot_curr_cpu,
                dot_baseline,
            ),
            dim=-1,
        )

        data_rows = data.tolist()

        output_path = schedule_dir / f"epoch_{epoch:04d}.csv"
        header = [
            "t",
            "beta_current",
            "beta_baseline",
            "dot_beta_current",
            "dot_beta_baseline",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)

        if getattr(args, "wandb", False):
            try:
                try:
                    import swanlab as wandb  # type: ignore
                except Exception:  # pragma: no cover - swanlab not installed
                    import wandb  # type: ignore

                wandb_payload = {}
                try:
                    schedule_table = wandb.Table(columns=header, rows=data_rows)  # type: ignore[attr-defined]
                    wandb_payload["eval/beta_schedule_table"] = schedule_table
                except Exception as table_exc:  # pragma: no cover - wandb table unavailable
                    logger.debug("Unable to build wandb table for β(t): %s", table_exc)

                try:
                    beta_plot = wandb.plot.line_series(  # type: ignore[attr-defined]
                        xs=t_cpu.tolist(),
                        ys=[
                            beta_curr_cpu.tolist(),
                            beta_baseline.tolist(),
                        ],
                        keys=["beta_current", "beta_baseline"],
                        title=f"β(t) epoch {epoch}",
                        xname="t",
                    )
                    wandb_payload["eval/beta_schedule"] = beta_plot
                except Exception as plot_exc:  # pragma: no cover - wandb plot unavailable
                    logger.debug("Unable to build wandb β(t) plot: %s", plot_exc)

                try:
                    dot_beta_plot = wandb.plot.line_series(  # type: ignore[attr-defined]
                        xs=t_cpu.tolist(),
                        ys=[
                            dot_curr_cpu.tolist(),
                            dot_baseline.tolist(),
                        ],
                        keys=["dot_beta_current", "dot_beta_baseline"],
                        title=f"β̇(t) epoch {epoch}",
                        xname="t",
                    )
                    wandb_payload["eval/dot_beta_schedule"] = dot_beta_plot
                except Exception as dot_plot_exc:  # pragma: no cover - wandb plot unavailable
                    logger.debug("Unable to build wandb β̇(t) plot: %s", dot_plot_exc)

                if wandb_payload:
                    try:
                        wandb.log(wandb_payload, step=epoch)  # type: ignore[attr-defined]
                    except Exception as log_exc:  # pragma: no cover - wandb logging failure
                        logger.warning("Failed to log β(t) snapshot to wandb: %s", log_exc)
            except Exception as wandb_exc:  # pragma: no cover - wandb import failure
                logger.debug("wandb not available for β(t) snapshot logging: %s", wandb_exc)

        schedule_snapshot_logged = True
        logger.info(
            "Saved β(t) snapshot for epoch %d with %d samples to %s",
            epoch,
            num_points,
            output_path,
        )

    if ko_path is not None:
        maybe_log_schedule_snapshot(ko_path)
        maybe_run_metric_geometry(ko_path)

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        remaining_real = max(fid_samples - num_real, 0)
        if remaining_real > 0 and samples.shape[0] > 0:
            real_batch = samples[:remaining_real]
            fid_metric.update(real_batch, real=True)
            num_real += real_batch.shape[0]

        remaining_fake = max(fid_samples - num_synthetic, 0)
        if remaining_fake > 0 and samples.shape[0] > 0:
            conditioning_samples = samples[:remaining_fake]
            conditioning_labels = labels[:remaining_fake]
        else:
            conditioning_samples = samples[:0]
            conditioning_labels = labels[:0]

        if conditioning_samples.shape[0] > 0 and num_synthetic < fid_samples:
            # Reset NFE counter on the wrapper that will actually be used
            # For mixture/continuous branches we use cfg_scaled_model; for metric-induced we use cfg_scaled_logits_model
            # Note: metric-induced branch performs a dummy forward to infer K; we reset AFTER that to avoid +1 in the count
            cfg_scaled_model.reset_nfe_counter()
            record_gif = (
                bool(getattr(args, "save_eval_gif", False))
                and bool(getattr(args, "output_dir", None))
                and not gif_logged
                and distributed_mode.is_main_process()
            )
            gif_trajectories: Optional[torch.Tensor] = None

            if args.discrete_flow_matching:
                # Discrete sampling
                if args.sym_func:
                    # Ensure a pure-Python function returning float for div_free / symmetrize coefficients
                    def sym_schedule(tau: float) -> float:
                        return 12.0 * (tau ** 2.0) * ((1.0 - tau) ** 0.25)

                    sym: Union[float, Callable[[float], float]] = sym_schedule
                else:
                    sym: Union[float, Callable[[float], float]] = float(args.sym)

                if getattr(args, "metric_induced", False):
                    # Metric-induced Gibbs path using dedicated KO solver
                    # Lazily build logits-wrapper and KO solver with correct vocab size K
                    if cfg_scaled_logits_model is None:
                        cfg_scaled_logits_model = CFGScaledModel(model=model, return_logits=True)
                    if ko_solver is None or ko_path is None:
                        # infer K by one forward pass at t=0
                        x_dummy = torch.zeros(
                            conditioning_samples.shape, dtype=torch.long, device=device
                        )
                        # IMPORTANT: do not apply CFG scaling with discrete logits
                        logits_dummy = cfg_scaled_logits_model(
                            x=x_dummy,
                            t=torch.tensor(0.0, device=device),
                            cfg_scale=0.0,
                            label=conditioning_labels,
                        )
                        K = int(logits_dummy.shape[-1])
                        # Build path
                        if ko_path is not None and ko_path.vocab_size != K:
                            raise ValueError(
                                f"Provided metric-induced path expects vocab size {ko_path.vocab_size},"
                                f" but model produced logits with last dim {K}."
                            )
                        if ko_path is None:
                            mi_metric = getattr(args, "mi_metric", "lp")
                            mi_lp = float(getattr(args, "mi_lp", 3.0))
                            mi_a = float(getattr(args, "mi_a", 5.0))
                            mi_c = float(getattr(args, "mi_c", 1.0))
                            mi_embed_range = getattr(args, "mi_embed_range", "pm1")
                            embed_range = "pm1" if mi_embed_range == "pm1" else "unit"
                            ko_path = MetricInducedGibbsProbPath(
                                embedding_path_or_weight=None,
                                vocab_size=K,
                                emb_dim=1,
                                metric=mi_metric,
                                lp_order=mi_lp,
                                embed_range=embed_range,
                                a=mi_a,
                                c=mi_c,
                                device=device,
                                dtype=torch.float32,
                            )
                        maybe_log_schedule_snapshot(ko_path)
                        maybe_run_metric_geometry(ko_path)
                        ko_solver = KODiscreteGibbsEulerSolver(
                            model=cfg_scaled_logits_model,
                            path=ko_path,
                            vocabulary_size=K,
                        )
                    # Reset NFE counter on the logits wrapper before stepping to avoid counting the dummy forward
                    if cfg_scaled_logits_model is not None:
                        cfg_scaled_logits_model.reset_nfe_counter()
                    # Start tokens: uniform over [0, K) since β(0)=0 ⇒ p0 is uniform
                    K_init = ko_solver.vocabulary_size
                    x_0 = torch.randint(
                        0,
                        K_init,
                        conditioning_samples.shape,
                        device=device,
                        dtype=torch.long,
                    )
                    dtype_cat = torch.float32 if args.sampling_dtype == "float32" else torch.float64
                    sample_result = ko_solver.sample(
                        x_init=x_0,
                        step_size=1.0 / args.discrete_fm_steps,
                        dtype_categorical=dtype_cat,
                        label=conditioning_labels,
                        # IMPORTANT: disable CFG scaling when using discrete logits
                        cfg_scale=0.0,
                        symmetrize=sym,
                        return_intermediates=record_gif,
                    )
                    if record_gif:
                        gif_trajectories = sample_result
                        synthetic_samples = sample_result[-1]
                    else:
                        synthetic_samples = sample_result
                else:
                    x_0 = (
                        torch.zeros(
                            conditioning_samples.shape, dtype=torch.long, device=device
                        )
                        + MASK_TOKEN
                    )
                    dtype = torch.float32 if args.sampling_dtype == "float32" else torch.float64

                    # Guard against missing solver (should never be None in this branch)
                    assert disc_solver is not None, "Discrete solver not initialized"
                    sample_result = disc_solver.sample(
                        x_init=x_0,
                        step_size=1.0 / args.discrete_fm_steps,
                        verbose=False,
                        div_free=sym,
                        dtype_categorical=dtype,
                        label=conditioning_labels,
                        # Disable CFG scaling for discrete models (logits)
                        cfg_scale=0.0,
                        return_intermediates=record_gif,
                    )
                    if record_gif:
                        gif_trajectories = sample_result
                        synthetic_samples = sample_result[-1]
                    else:
                        synthetic_samples = sample_result
            else:
                # Continuous sampling
                x_0 = torch.randn(
                    conditioning_samples.shape, dtype=torch.float32, device=device
                )

                # Safe defaults for ODE options
                nfe_default = 50
                atol_default = 1e-5
                rtol_default = 1e-5
                step_default = None
                if cont_ode_opts is not None:
                    ode_nfe = int(cont_ode_opts.get("nfe", nfe_default))
                    ode_atol = float(cont_ode_opts.get("atol", atol_default))
                    ode_rtol = float(cont_ode_opts.get("rtol", rtol_default))
                    ode_step = cont_ode_opts.get("step_size", step_default)
                else:
                    ode_nfe = nfe_default
                    ode_atol = atol_default
                    ode_rtol = rtol_default
                    ode_step = step_default

                if args.edm_schedule:
                    time_grid = get_time_discretization(nfes=ode_nfe)
                else:
                    time_grid = torch.tensor([0.0, 1.0], device=device)

                # Guard against missing solver
                assert cont_solver is not None, "Continuous solver not initialized"
                sample_result = cont_solver.sample(
                    time_grid=time_grid,
                    x_init=x_0,
                    method=args.ode_method,
                    return_intermediates=record_gif,
                    atol=ode_atol,
                    rtol=ode_rtol,
                    step_size=ode_step,
                    label=conditioning_labels,
                    cfg_scale=args.cfg_scale,
                )

                # Scaling to [0, 1] from [-1, 1]
                if isinstance(sample_result, (list, tuple)):
                    synthetic_samples = sample_result[-1]
                else:
                    synthetic_samples = sample_result
                synthetic_samples = cast(torch.Tensor, synthetic_samples)
                if record_gif and isinstance(sample_result, torch.Tensor):
                    gif_trajectories = sample_result
                synthetic_samples = torch.clamp(
                    synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
                )
                synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
            # Report NFE from the active wrapper (metric-induced uses logits wrapper)
            _nfe_model = (
                cfg_scaled_logits_model if getattr(args, "metric_induced", False) and 'cfg_scaled_logits_model' in locals() and cfg_scaled_logits_model is not None else cfg_scaled_model
            )
            batch_generated = synthetic_samples.shape[0]
            logger.info(
                f"{batch_generated} samples generated in {_nfe_model.get_nfe()} evaluations."
            )
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            fid_metric.update(synthetic_samples, real=False)
            num_synthetic = min(num_synthetic + synthetic_samples.shape[0], fid_samples)
            if not snapshots_saved and args.output_dir:
                snapshot_batch = synthetic_samples
                if snapshot_batch.ndim == 4:
                    snapshot_batch = _pad_samples_to_square_grid(snapshot_batch)
                save_image(
                    snapshot_batch,
                    fp=Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}.png",
                )
                snapshots_saved = True

            if args.save_fid_samples and args.output_dir:
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.output_dir) / "fid_samples"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

            if (
                gif_trajectories is not None
                and getattr(args, "output_dir", None)
                and not gif_logged
                and distributed_mode.is_main_process()
            ):
                try:
                    _save_sampling_gif(
                        gif_trajectories,
                        Path(args.output_dir),
                        epoch,
                        data_iter_step,
                        is_discrete=args.discrete_flow_matching,
                        max_batch=getattr(args, "eval_gif_max_batch", 8),
                        stride=getattr(args, "eval_gif_stride", 16),
                        fps=getattr(args, "eval_gif_fps", 8),
                        log_to_wandb=getattr(args, "wandb", False),
                        wandb_step=epoch,
                    )
                except Exception as gif_exc:  # pragma: no cover - PIL/image errors
                    logger.warning("Failed to save evaluation GIF: %s", gif_exc)
                finally:
                    gif_logged = True

        if not args.compute_fid:
            return {}

        if data_iter_step % PRINT_FREQUENCY == 0:
            # Sync fid metric to ensure that the processes dont deviate much.
            gc.collect()
            running_fid = fid_metric.compute()
            if distributed_mode.is_dist_avail_and_initialized():
                counts = torch.tensor(
                    [num_real, num_synthetic],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                total_real = float(counts[0].item())
                total_fake = float(counts[1].item())
                target_total = float(
                    getattr(
                        args,
                        "fid_samples",
                        fid_samples * distributed_mode.get_world_size(),
                    )
                )
            else:
                total_real = float(num_real)
                total_fake = float(num_synthetic)
                target_total = float(fid_samples)
            target_total = max(target_total, 1.0)
            if _data_loader_len_for_log is not None:
                _len_str = str(_data_loader_len_for_log)
            else:
                _len_str = "?"
            logger.info(
                "Evaluating ["
                f"{data_iter_step}/{_len_str}] samples generated [{total_fake:.0f}/{target_total}] "
                f"reals [{total_real:.0f}/{target_total}] running fid {running_fid}"
            )

        if args.test_run:
            break

        if num_real >= fid_samples and num_synthetic >= fid_samples:
            break

    fid_value = float(fid_metric.compute().detach().cpu())
    stats = {"fid": fid_value}
    stats.update(geometry_stats)
    return stats
