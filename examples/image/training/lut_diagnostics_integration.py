# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""Integration helpers for LUT diagnostics in training loop.

Provides utilities to:
  - Create fixed probe batch for consistent metrics
  - Log LUT diagnostics每 epoch
  - Save LUT snapshots for plotting
  - Handle wandb/swanlab logging
"""

import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from flow_matching.path import MetricInducedGibbsProbPath
from flow_matching.path.lut_diagnostics import LUTDiagnostics

logger = logging.getLogger(__name__)


def _scalarize_lut_embeddings(weight: Tensor, *, metric: str) -> Tensor:
    """Convert LUT embeddings [C,V,D] into scalar representation for diagnostics."""
    if weight is None:
        return weight
    if weight.ndim == 2:
        return weight
    if weight.ndim == 3 and weight.shape[-1] == 1:
        return weight.squeeze(-1)

    if weight.ndim != 3:
        raise ValueError(f"Unexpected LUT weight shape {tuple(weight.shape)}")

    if metric == "cosine":
        normed = F.normalize(weight, p=2, dim=-1, eps=1e-12)
        if normed.shape[-1] >= 2:
            theta = torch.atan2(normed[:, :, 1], normed[:, :, 0])
            theta = torch.remainder(theta, 2.0 * math.pi)
            return theta
        return normed.squeeze(-1)

    # Default: use L2 norm for scalar summary
    return torch.linalg.vector_norm(weight, ord=2, dim=-1)


def create_probe_batch(
    dataset,
    num_samples: int = 128,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Tensor:
    """Create a fixed probe batch for consistent diagnostics.
    
    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples
        device: Device to place batch on
        seed: Random seed for reproducibility
    
    Returns:
        [num_samples, C, H, W] tensor
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    
    indices = torch.randperm(len(dataset), generator=rng)[:num_samples]
    
    samples = []
    for idx in indices:
        img, _ = dataset[idx]
        samples.append(img)
    
    batch = torch.stack(samples, dim=0)  # [N, C, H, W]
    
    # Convert to int64 tokens if needed
    if batch.dtype == torch.float32:
        # Assume [0, 1] range
        batch = (batch * 255).long().clamp(0, 255)
    
    return batch.to(device)


def log_lut_diagnostics_epoch(
    path: MetricInducedGibbsProbPath,
    probe_batch: Tensor,
    diagnostics: LUTDiagnostics,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    prev_lut: Optional[Tensor] = None,
    wandb_run=None,
    channel_names: Optional[list] = None,
) -> Dict[str, Tensor]:
    """Compute and log LUT diagnostics for one epoch.
    
    Args:
        path: Metric-induced Gibbs path with learnable LUT
        probe_batch: Fixed probe batch [B, C, H, W]
        diagnostics: LUTDiagnostics instance
        epoch: Current epoch number
        optimizer: Optimizer (to extract LR and gradients)
        prev_lut: Previous epoch's LUT weights (for dynamics)
        wandb_run: Optional wandb/swanlab run object
        channel_names: Optional channel names (e.g., ['R', 'G', 'B'])
    
    Returns:
        Dictionary of all computed metrics
    """
    if path.learnable_lut is None:
        logger.warning("No learnable LUT found in path; skipping diagnostics")
        return {}
    
    metric_name = getattr(path, "metric_name", "lp")

    # Get current LUT weights
    lut_weights_raw = path.learnable_lut.weight.detach()
    lut_weights = _scalarize_lut_embeddings(lut_weights_raw, metric=metric_name)
    
    # Extract gradient and LR (if optimizer available)
    grad = None
    lr = None
    if optimizer is not None:
        # Find LUT parameter group
        for param_group in optimizer.param_groups:
            if param_group.get("name") == "learnable_lut":
                lr = param_group["lr"]
                # Get gradient from parameter
                if path.learnable_lut.weight.grad is not None:
                    grad = _scalarize_lut_embeddings(
                        path.learnable_lut.weight.grad.detach(), metric=metric_name
                    )
                break
    if grad is None and path.learnable_lut.weight.grad is not None:
        grad = _scalarize_lut_embeddings(
            path.learnable_lut.weight.grad.detach(), metric=metric_name
        )
    
    # Beta function for path quality metrics
    def beta_fn(t: Tensor) -> Tensor:
        return path.beta_schedule(t)
    
    # Beta values at probe times (for distance scaling)
    beta_values = torch.tensor([
        beta_fn(torch.tensor(t, device=lut_weights.device)).item()
        for t in diagnostics.probe_times
    ], device=lut_weights.device)
    
    # Compute all metrics
    metrics = diagnostics.compute_all_metrics(
        embeddings=lut_weights,
        probe_batch=probe_batch,
        beta_fn=beta_fn,
        embeddings_prev=_scalarize_lut_embeddings(prev_lut, metric=metric_name) if prev_lut is not None else None,
        grad=grad,
        lr=lr,
        beta_values=beta_values,
    )
    
    # Log summary to console
    diagnostics.log_metrics_summary(metrics, epoch)
    
    # Log to wandb/swanlab
    if wandb_run is not None:
        # Convert metrics to scalars for logging
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, Tensor):
                if value.numel() == 1:
                    flat_metrics[f"lut/{key}"] = value.item()
                else:
                    # Log per-channel values
                    for i, v in enumerate(value):
                        ch_name = channel_names[i] if channel_names and i < len(channel_names) else f"ch{i}"
                        flat_metrics[f"lut/{key}_{ch_name}"] = v.item()
            elif isinstance(value, (int, float, bool)):
                flat_metrics[f"lut/{key}"] = value
        
        wandb_run.log(flat_metrics, step=epoch)
    
    return metrics


def save_lut_snapshot(
    path: MetricInducedGibbsProbPath,
    output_dir: Path,
    epoch: int,
    save_init: bool = False,
) -> None:
    """Save LUT weights for later plotting.
    
    Args:
        path: Metric-induced path
        output_dir: Directory to save snapshots
        epoch: Current epoch
        save_init: If True, also save initial/baseline LUT
    """
    if path.learnable_lut is None:
        return
    
    snapshot_dir = output_dir / "lut_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Save current LUT
    lut_weights = path.learnable_lut.weight.detach().cpu()
    torch.save({
        "epoch": epoch,
        "lut_weights": lut_weights,
        "num_channels": path.learnable_lut.num_channels,
        "vocab_size": path.learnable_lut.vocab_size,
        "embed_range": path.learnable_lut.embed_range,
    }, snapshot_dir / f"lut_epoch_{epoch:04d}.pt")
    
    # Save init/baseline (once)
    if save_init:
        init_weights = path.learnable_lut._linear_init().cpu()
        torch.save({
            "epoch": -1,
            "lut_weights": init_weights,
            "num_channels": path.learnable_lut.num_channels,
            "vocab_size": path.learnable_lut.vocab_size,
            "embed_range": path.learnable_lut.embed_range,
        }, snapshot_dir / "lut_init.pt")


def setup_lut_diagnostics(
    path: MetricInducedGibbsProbPath,
    dataset,
    device: torch.device,
    probe_samples: int = 128,
    probe_times: Optional[list] = None,
) -> tuple[Optional[LUTDiagnostics], Optional[Tensor]]:
    """Setup LUT diagnostics and probe batch.
    
    Args:
        path: Metric-induced path
        dataset: Training dataset
        device: Device to use
        probe_samples: Number of probe samples
        probe_times: Times to probe (default: [0.2, 0.5, 0.8])
    
    Returns:
        (diagnostics, probe_batch) tuple, or (None, None) if no LUT
    """
    if path.learnable_lut is None:
        logger.info("No learnable LUT in path; skipping diagnostics setup")
        return None, None
    
    logger.info(f"Setting up LUT diagnostics with {probe_samples} probe samples")
    
    # Create diagnostics tracker
    diagnostics = LUTDiagnostics(
        num_channels=path.learnable_lut.num_channels,
        vocab_size=path.learnable_lut.vocab_size,
        probe_times=probe_times,
    )
    
    # Register baseline (linear init)
    baseline_lut = path.learnable_lut._linear_init().to(device)
    diagnostics.register_baseline(baseline_lut)
    
    # Create fixed probe batch
    probe_batch = create_probe_batch(
        dataset=dataset,
        num_samples=probe_samples,
        device=device,
    )
    
    logger.info(
        f"LUT diagnostics ready: {diagnostics.num_channels} channels, "
        f"{diagnostics.vocab_size} vocab, probe times={diagnostics.probe_times}"
    )
    
    return diagnostics, probe_batch
