#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Utility to inspect the local geometry of a learnable LUT snapshot.
#
# Usage (snapshot):
#   python lut_geometry_inspect.py --lut path/to/lut_epoch_xxxx.pt \
#       [--normalized] [--print-projection] [--export-projection path.npy]
# Usage (checkpoint):
#   python lut_geometry_inspect.py --checkpoint path/to/checkpoint-5499.pth \
#       --lut-key metric_learnable_lut [--normalized] [...]
#
# The LUT snapshot can be produced via training.lut_diagnostics_integration.save_lut_snapshot.
# When pointing to a training checkpoint, the LUT is fetched from checkpoint["extra_modules"][lut_key].
# The script reports:
#   - cos_mean / cos_median / flip_rate: directional continuity of adjacent token differences.
#   - stretch_median / stretch_mean: ratio of adjacent distances vs baseline linear LUT.
#   - curvature_mean / curvature_std: second-order finite difference magnitude ("zig-zag").
#   - Optional: saves/prints a 1-D signed projection curve for visual inspection.

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


def _ensure_3d(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 2:
        weight = weight.unsqueeze(-1)
    if weight.ndim != 3:
        raise ValueError(f"Expected LUT weight with 2 or 3 dims; got shape {tuple(weight.shape)}")
    return weight


def _linear_baseline(
    num_channels: int,
    vocab_size: int,
    embed_range: str,
    device,
    dtype,
    embed_dim: int = 1,
) -> torch.Tensor:
    weight = torch.zeros(num_channels, vocab_size, embed_dim, device=device, dtype=dtype)
    if embed_range == "pm1":
        base = torch.linspace(-1.0, 1.0, steps=vocab_size, device=device, dtype=dtype)
    elif embed_range == "unit":
        base = torch.linspace(0.0, 1.0, steps=vocab_size, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported embed_range '{embed_range}'.")
    base = base.unsqueeze(0).expand(num_channels, -1)  # [C,V]
    weight[:] = base.unsqueeze(-1)  # broadcast across embed_dim
    return weight


def load_lut_snapshot(path: Path) -> Tuple[torch.Tensor, Dict[str, int]]:
    data = torch.load(path, map_location="cpu")
    if "lut_weights" not in data:
        raise KeyError(f"{path} does not contain 'lut_weights'")
    weight = data["lut_weights"]
    meta = {
        "num_channels": int(data.get("num_channels", weight.shape[0])),
        "vocab_size": int(data.get("vocab_size", weight.shape[1])),
        "embed_range": data.get("embed_range", "pm1"),
    }
    return weight, meta


def load_lut_from_checkpoint(
    path: Path,
    lut_key: str,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    checkpoint = torch.load(path, map_location="cpu")
    extra = checkpoint.get("extra_modules")
    if not isinstance(extra, dict):
        raise KeyError(f"{path} has no 'extra_modules' dict; cannot locate '{lut_key}'")
    if lut_key not in extra:
        available = ", ".join(sorted(extra.keys()))
        raise KeyError(f"'{lut_key}' not found in checkpoint extra_modules. Available keys: {available}")
    state_dict = extra[lut_key]
    if not isinstance(state_dict, dict):
        raise TypeError(f"extra_modules['{lut_key}'] is not a state_dict (got {type(state_dict).__name__})")
    if "weight" not in state_dict:
        raise KeyError(f"State dict for '{lut_key}' is missing 'weight'")
    weight = state_dict["weight"]
    embed_range = None
    checkpoint_args = checkpoint.get("args")
    if hasattr(checkpoint_args, "mi_embed_range"):
        embed_range = getattr(checkpoint_args, "mi_embed_range")
    if embed_range is None:
        embed_range = "pm1"
    meta = {
        "num_channels": weight.shape[0],
        "vocab_size": weight.shape[1],
        "embed_range": embed_range,
    }
    return weight, meta


def compute_geometry_metrics(
    lut_weight: torch.Tensor,
    baseline_weight: torch.Tensor,
    normalized_distance: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        lut_weight: [C,V,D]
        baseline_weight: [C,V,D]
    Returns:
        Dictionary of per-channel metrics (torch tensors on CPU).
    """
    lut_weight = _ensure_3d(lut_weight).to(dtype=torch.float64)
    baseline_weight = _ensure_3d(baseline_weight).to(dtype=torch.float64, device=lut_weight.device)

    if baseline_weight.shape[-1] == 1 and lut_weight.shape[-1] > 1:
        baseline_weight = baseline_weight.expand(-1, -1, lut_weight.shape[-1])

    if lut_weight.shape != baseline_weight.shape:
        raise ValueError(f"Shape mismatch: learned {tuple(lut_weight.shape)} vs baseline {tuple(baseline_weight.shape)}")

    C, V, D = lut_weight.shape
    if V < 3:
        raise ValueError("Need vocab_size >= 3 to compute curvature/angles")

    diff = lut_weight[:, 1:, :] - lut_weight[:, :-1, :]
    diff_base = baseline_weight[:, 1:, :] - baseline_weight[:, :-1, :]

    if normalized_distance:
        scale = torch.sqrt(torch.tensor(float(D), device=lut_weight.device, dtype=lut_weight.dtype))
        diff = diff / scale
        diff_base = diff_base / scale

    # directional continuity
    a = diff[:, :-1, :]
    b = diff[:, 1:, :]
    denom = a.norm(dim=-1) * b.norm(dim=-1) + 1e-12
    cos_theta = (a * b).sum(dim=-1) / denom

    # stretch ratio
    dist = diff.norm(dim=-1)  # [C,V-1]
    dist_base = diff_base.norm(dim=-1) + 1e-12
    stretch = dist / dist_base

    # curvature magnitude
    curv = lut_weight[:, 2:, :] - 2 * lut_weight[:, 1:-1, :] + lut_weight[:, :-2, :]
    curv_mag = curv.norm(dim=-1)

    metrics = {
        "cos_mean": cos_theta.mean(dim=1).cpu(),
        "cos_median": cos_theta.median(dim=1).values.cpu(),
        "flip_rate": (cos_theta < 0.0).float().mean(dim=1).cpu(),
        "cos_lt_half_rate": (cos_theta < 0.5).float().mean(dim=1).cpu(),
        "stretch_median": stretch.median(dim=1).values.cpu(),
        "stretch_mean": stretch.mean(dim=1).cpu(),
        "stretch_p90": torch.quantile(stretch.cpu(), 0.9, dim=1),
        "curvature_mean": curv_mag.mean(dim=1).cpu(),
        "curvature_std": curv_mag.std(dim=1).cpu(),
    }
    return metrics


def signed_projection_curve(weight: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    Returns a [V] curve summarizing embeddings with sign information.
    mode: 'mean' (average across dims) or 'uniform' (dot with 1/sqrt(D) vector).
    """
    weight = _ensure_3d(weight).to(dtype=torch.float64)
    if mode == "mean":
        curve = weight.mean(dim=-1)
    elif mode == "uniform":
        D = weight.shape[-1]
        u = torch.ones(D, device=weight.device, dtype=weight.dtype) / torch.sqrt(torch.tensor(float(D)))
        curve = torch.matmul(weight, u)
    else:
        raise ValueError(f"Unknown projection mode '{mode}'")
    return curve.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect learnable LUT geometry.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lut", type=Path, help="Path to LUT snapshot (.pt) saved via save_lut_snapshot.")
    group.add_argument("--checkpoint", type=Path, help="Path to training checkpoint containing extra_modules.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline LUT snapshot (.pt).")
    parser.add_argument("--lut-key", type=str, default="metric_learnable_lut", help="Key inside checkpoint['extra_modules'] for the learnable LUT.")
    parser.add_argument("--normalized", action="store_true", help="Use normalized distances (divide by sqrt(D)).")
    parser.add_argument("--projection-mode", choices=["none", "mean", "uniform"], default="none", help="Compute signed projection curve.")
    parser.add_argument("--export-projection", type=Path, default=None, help="Where to save projection curve (npz with per-channel arrays).")
    args = parser.parse_args()

    if args.lut is not None:
        lut_weight, meta = load_lut_snapshot(args.lut)
        lut_source = args.lut
    else:
        lut_weight, meta = load_lut_from_checkpoint(args.checkpoint, lut_key=args.lut_key)
        lut_source = args.checkpoint
    lut_weight = _ensure_3d(lut_weight)

    if args.baseline is not None:
        base_weight, _ = load_lut_snapshot(args.baseline)
        base_weight = _ensure_3d(base_weight)
    else:
        base_weight = _linear_baseline(
            num_channels=meta["num_channels"],
            vocab_size=meta["vocab_size"],
            embed_range=meta.get("embed_range", "pm1"),
            device=lut_weight.device,
            dtype=lut_weight.dtype,
            embed_dim=lut_weight.shape[-1],
        )

    metrics = compute_geometry_metrics(
        lut_weight,
        base_weight,
        normalized_distance=args.normalized,
    )

    print(f"LUT source: {lut_source}")
    print(f"Shape: C={meta['num_channels']}, V={meta['vocab_size']}, D={lut_weight.shape[-1]}")
    print(f"Embed range baseline: {meta.get('embed_range', 'pm1')}")
    print("=== Directional continuity ===")
    for key in ["cos_mean", "cos_median", "flip_rate", "cos_lt_half_rate"]:
        print(f"{key}: {metrics[key].tolist()}")

    print("=== Neighbour stretch ratio (learned vs baseline) ===")
    for key in ["stretch_median", "stretch_mean", "stretch_p90"]:
        print(f"{key}: {metrics[key].tolist()}")

    print("=== Curvature (finite-difference magnitude) ===")
    for key in ["curvature_mean", "curvature_std"]:
        print(f"{key}: {metrics[key].tolist()}")

    if args.projection_mode != "none":
        proj = signed_projection_curve(lut_weight, mode=args.projection_mode)
        print(f"Projection curve mode={args.projection_mode}, shape={tuple(proj.shape)} (per channel stacked).")
        if args.export_projection:
            args.export_projection.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"projection": proj}, args.export_projection)
            print(f"Saved projection curve to {args.export_projection}")
        else:
            print("First 10 values per channel:")
            for c in range(proj.shape[0]):
                vals = proj[c][:10].tolist()
                print(f"  ch{c}: {vals}")


if __name__ == "__main__":
    main()
