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
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
import seaborn as sns

def _load_token_tensor(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
    elif suffix in {".npy"}:
        data = np.load(path)
    elif suffix in {".npz"}:
        npz = np.load(path)
        data = next(iter(npz.values()))
    else:
        raise ValueError(f"Unsupported token file format: {path}")
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if not torch.is_tensor(data):
        raise TypeError(f"Token file {path} did not yield a tensor or ndarray.")
    return data.long()

from flow_matching.path.mixture import LearnableLUT


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


def _great_circle_baseline(
    num_channels: int,
    vocab_size: int,
    embed_dim: int,
    device,
    dtype,
) -> torch.Tensor:
    if embed_dim < 2:
        raise ValueError("Great-circle baseline requires embedding dimension >= 2.")
    weight = torch.zeros(num_channels, vocab_size, embed_dim, device=device, dtype=dtype)
    theta = torch.linspace(0.0, math.pi, steps=vocab_size, device=device, dtype=dtype)
    cos_theta = torch.cos(theta).unsqueeze(0).expand(num_channels, -1)
    sin_theta = torch.sin(theta).unsqueeze(0).expand(num_channels, -1)
    weight[:, :, 0] = cos_theta
    weight[:, :, 1] = sin_theta
    if embed_dim > 2:
        weight[:, :, 2:] = 0.0
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


def _infer_embed_range(args_obj) -> str:
    candidate = None
    if args_obj is not None:
        candidate = getattr(args_obj, "mi_embed_range", getattr(args_obj, "embed_range", None))
    if candidate == "01":
        return "unit"
    return "pm1"


def _instantiate_lut_from_state(state_dict: Dict[str, torch.Tensor], embed_range: str) -> Tuple[torch.Tensor, Dict[str, int]]:
    converted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("param."):
            converted["_geometry." + key[len("param."):]] = value
        else:
            converted[key] = value
    state_dict = converted

    if "weight" in state_dict:
        num_channels, vocab_size, emb_dim = state_dict["weight"].shape
        param_mode = "none"
    elif "_geometry.basis_raw" in state_dict:
        basis = state_dict["_geometry.basis_raw"]
        num_channels, emb_dim = basis.shape[0], basis.shape[1]
        delta = state_dict["_geometry.delta_raw"]
        vocab_size = delta.shape[1] + 1
        param_mode = "arc2d"
    elif "_geometry.direction_raw" in state_dict:
        direction = state_dict["_geometry.direction_raw"]
        num_channels, emb_dim = direction.shape[0], direction.shape[1]
        delta = state_dict["_geometry.delta_raw"]
        vocab_size = delta.shape[1] + 1
        param_mode = "line2d"
    else:
        raise ValueError("Unable to infer LUT configuration from state dictionary.")

    lut = LearnableLUT(
        num_channels=num_channels,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        embed_range=embed_range,
        param_mode=param_mode,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    lut.load_state_dict(state_dict, strict=False)
    lut.eval()
    weight = lut().detach()
    meta = {
        "num_channels": lut.num_channels,
        "vocab_size": lut.vocab_size,
        "embed_range": embed_range,
    }
    return weight, meta


def load_lut_from_checkpoint(path: Path, lut_key: str) -> Tuple[torch.Tensor, Dict[str, int]]:
    checkpoint = torch.load(path, map_location="cpu")
    embed_range = _infer_embed_range(checkpoint.get("args"))

    path_state = checkpoint.get("path", {})
    if isinstance(path_state, dict) and path_state:
        # Common case: flattened keys like "learnable_lut.weight"
        lut_state = {
            key[len("learnable_lut.") :]: value
            for key, value in path_state.items()
            if key.startswith("learnable_lut.")
        }
        if lut_state:
            weight, meta = _instantiate_lut_from_state(lut_state, embed_range)
            return weight, meta
        # Fallback: nested state dict stored directly under "learnable_lut"
        nested = path_state.get("learnable_lut")
        if isinstance(nested, dict):
            weight, meta = _instantiate_lut_from_state(nested, embed_range)
            return weight, meta

    extra = checkpoint.get("extra_modules")
    if isinstance(extra, dict) and lut_key in extra:
        state_dict = extra[lut_key]
        if isinstance(state_dict, dict):
            weight, meta = _instantiate_lut_from_state(state_dict, embed_range)
            return weight, meta

    lut_weight = None
    if "ema_model" in checkpoint:
        for key, value in checkpoint["ema_model"].items():
            if "learnable_lut.weight" in key:
                lut_weight = value
                break
    if lut_weight is None:
        raise KeyError(f"Unable to locate learnable LUT in checkpoint '{path}'")
    if lut_weight.ndim == 2:
        lut_weight = lut_weight.unsqueeze(-1)
    meta = {
        "num_channels": lut_weight.shape[0],
        "vocab_size": lut_weight.shape[1],
        "embed_range": embed_range,
    }
    return lut_weight, meta


def compute_geometry_metrics(
    lut_weight: torch.Tensor,
    baseline_weight: torch.Tensor,
    normalized_distance: bool = False,
    sphere_mode: bool = False,
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

    if sphere_mode:
        lut_weight = F.normalize(lut_weight, p=2, dim=-1, eps=1e-12)
        baseline_weight = F.normalize(baseline_weight, p=2, dim=-1, eps=1e-12)
        if torch.isnan(lut_weight).any() or torch.isnan(baseline_weight).any():
            raise ValueError("Encountered NaNs during unit normalization; check LUT/baseline weights.")

    diff = lut_weight[:, 1:, :] - lut_weight[:, :-1, :]
    diff_base = baseline_weight[:, 1:, :] - baseline_weight[:, :-1, :]

    if normalized_distance and not sphere_mode:
        scale = torch.sqrt(torch.tensor(float(D), device=lut_weight.device, dtype=lut_weight.dtype))
        diff = diff / scale
        diff_base = diff_base / scale

    # directional continuity
    a = diff[:, :-1, :]
    b = diff[:, 1:, :]
    denom = a.norm(dim=-1) * b.norm(dim=-1) + 1e-12
    cos_theta = (a * b).sum(dim=-1) / denom

    # stretch ratio
    if sphere_mode:
        dot = torch.sum(lut_weight[:, 1:, :] * lut_weight[:, :-1, :], dim=-1).clamp(-1.0, 1.0)
        dist = torch.sqrt(torch.clamp(2.0 - 2.0 * dot, min=0.0))  # chord length on unit sphere
        dot_base = torch.sum(baseline_weight[:, 1:, :] * baseline_weight[:, :-1, :], dim=-1).clamp(-1.0, 1.0)
        dist_base = torch.sqrt(torch.clamp(2.0 - 2.0 * dot_base, min=0.0)) + 1e-12
    else:
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


def _pairwise_lp_distance(weight: torch.Tensor, lp_order: float) -> torch.Tensor:
    weight = _ensure_3d(weight)
    C, V, D = weight.shape
    dist = torch.zeros(C, V, V, device=weight.device, dtype=weight.dtype)
    for c in range(C):
        dist[c] = torch.cdist(weight[c].unsqueeze(0), weight[c].unsqueeze(0), p=lp_order)[0]
    return dist


def compute_lut_similarity(
    weight: torch.Tensor,
    lp_order: float = 2.0,
    kernel: str = "gaussian",
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    weight = _ensure_3d(weight).to(dtype=torch.float32)
    dist = _pairwise_lp_distance(weight, lp_order=lp_order)
    sims = torch.empty_like(dist)
    eye_mask = torch.eye(dist.size(-1), dtype=torch.bool, device=dist.device)
    for c in range(dist.shape[0]):
        dc = dist[c]
        if kernel == "gaussian":
            off_diag = dc[~eye_mask]
            if off_diag.numel() == 0:
                sigma = torch.tensor(1.0, device=dc.device, dtype=dc.dtype)
            else:
                sigma = off_diag.median().clamp_min(1e-6) * sigma_scale
            sims[c] = torch.exp(-(dc.pow(2)) / (2 * sigma.pow(2)))
        elif kernel == "recip":
            sims[c] = 1.0 / (1.0 + dc)
        else:
            raise ValueError(f"Unknown similarity kernel '{kernel}'")
    return sims


def attention_like_similarity(
    weight: torch.Tensor,
    kind: str = "dot",
    temperature: Optional[float] = None,
    apply_softmax: bool = True,
) -> torch.Tensor:
    """
    Attention-style similarity matrix.
    kind:
      - 'dot': scaled dot-product (divide by sqrt(D))
      - 'cosine': cosine similarity
    temperature:
      - Optional temperature; when None uses 1.0 (after dot scaling)
    apply_softmax:
      - If True, apply softmax along the last dimension (attention map)
    Returns tensor of shape [C, V, V].
    """
    weight = _ensure_3d(weight).to(dtype=torch.float32)
    C, V, D = weight.shape
    if kind == "dot":
        scale = 1.0 / math.sqrt(max(D, 1))
        sim = torch.matmul(weight, weight.transpose(-1, -2)) * scale
    elif kind == "cosine":
        w = F.normalize(weight, p=2, dim=-1)
        sim = torch.matmul(w, w.transpose(-1, -2))
    else:
        raise ValueError(f"Unknown attention similarity kind '{kind}'")

    if apply_softmax:
        temp = float(temperature) if temperature is not None else 1.0
        sim = torch.softmax(sim / temp, dim=-1)
    return sim


def analyze_frequency(embeddings: torch.Tensor, channel_idx: int = 0):
    """
    Analyze frequency characteristics of a given channel in LUT embeddings.

    Args:
        embeddings: Tensor of shape [V, D] (or anything that can be viewed as such).
        channel_idx: Which embedding dimension (channel) to analyze.

    Returns:
        dominant_freq: Dominant frequency in cycles/token.
        power: Power spectrum array corresponding to positive frequencies.
    """
    emb_np = embeddings.detach().cpu().numpy()
    if emb_np.ndim != 2:
        raise ValueError(f"Expected embeddings of shape [V, D]; got shape {emb_np.shape}")
    if not (0 <= channel_idx < emb_np.shape[1]):
        raise ValueError(f"channel_idx {channel_idx} out of bounds for dim {emb_np.shape[1]}")

    signal = emb_np[:, channel_idx]
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1.0)

    xf = xf[: N // 2]
    power = 2.0 / N * np.abs(yf[: N // 2])

    if power.shape[0] > 1:
        dominant_freq_idx = np.argmax(power[1:]) + 1
        dominant_freq = xf[dominant_freq_idx]
        if dominant_freq != 0:
            print(f"Dominant frequency: {dominant_freq:.4f} cycles/token")
            print(f"Period: {1 / dominant_freq:.2f} tokens")
        else:
            print("Dominant frequency is DC component (0 cycles/token).")
    else:
        dominant_freq = 0.0
        print("Power spectrum too small to determine dominant frequency.")

    plt.figure(figsize=(10, 4))
    plt.plot(xf, power)
    plt.xlabel("Frequency (cycles/token)")
    plt.ylabel("Power")
    plt.title(f"Frequency Spectrum - Channel {channel_idx}")
    plt.grid(True)
    plt.show()

    return dominant_freq, power


def analyze_frequency_detailed(
    embeddings: torch.Tensor,
    channel_idx: int = 0,
    top_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    More detailed frequency analysis: prints top-K frequencies and plots spectrum.

    Args:
        embeddings: Tensor shaped [V, D].
        channel_idx: Which embedding dimension to inspect.
        top_k: Number of dominant frequencies to report (excluding DC when possible).

    Returns:
        (frequencies, power) arrays for positive frequencies.
    """
    emb_np = embeddings.detach().cpu().numpy()
    if emb_np.ndim != 2:
        raise ValueError(f"Expected embeddings of shape [V, D]; got shape {emb_np.shape}")
    if not (0 <= channel_idx < emb_np.shape[1]):
        raise ValueError(f"channel_idx {channel_idx} out of bounds for dim {emb_np.shape[1]}")

    signal = emb_np[:, channel_idx].astype(np.float64)
    signal = signal - signal.mean()
    N = signal.shape[0]

    yf = fft(signal)
    xf = fftfreq(N, 1.0)

    xf = xf[: N // 2]
    power = 2.0 / N * np.abs(yf[: N // 2])

    if power.size == 0:
        print("Power spectrum empty.")
        return xf, power

    peak_indices = np.argsort(power)[-top_k:][::-1]

    print(f"\nChannel {channel_idx} - Top {top_k} frequencies:")
    reported = 0
    for idx in peak_indices:
        freq = xf[idx]
        if freq <= 0:
            continue
        period = 1.0 / freq if freq != 0 else float("inf")
        print(
            f"  {reported + 1}. Freq: {freq:.4f} cycles/token, "
            f"Period: {period:.2f} tokens, Power: {power[idx]:.4f}"
        )
        reported += 1
        if reported >= top_k:
            break
    if reported == 0:
        print("  Dominant frequencies are at DC / non-positive frequencies.")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(xf, power)
    plt.xlabel("Frequency (cycles/token)")
    plt.ylabel("Power")
    plt.title("Full Spectrum")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if power.shape[0] > 10:
        plt.plot(xf[10:], power[10:])
    else:
        plt.plot(xf, power)
    plt.xlabel("Frequency (cycles/token)")
    plt.ylabel("Power")
    plt.title("High Frequency Detail")
    plt.grid(True)
    plt.show()

    return xf, power


def check_monotonicity(embeddings: torch.Tensor) -> float:
    """
    Check |i-j| < |i-k| => d(e_i, e_j) < d(e_i, e_k).
    Returns fraction of violations (lower is better).
    """
    emb = embeddings.detach()
    V = emb.shape[0]
    violations = 0
    total = 0
    for i in range(V):
        ref = emb[i]
        if i + 2 >= V:
            break
        dists_tail = torch.linalg.norm(ref - emb[i + 1 :], dim=-1)
        for offset_j in range(len(dists_tail) - 1):
            d_ij = dists_tail[offset_j]
            later = dists_tail[offset_j + 1 :]
            total += later.numel()
            violations += (d_ij > later).sum().item()
    if total == 0:
        return 0.0
    return violations / total


def compute_distance_correlation(embeddings: torch.Tensor) -> Tuple[float, float, float]:
    """
    Compute correlation between embedding distances and pixel distances.
    Returns (pearson_corr, slope, r_squared).
    """
    emb = embeddings.detach()
    V = emb.shape[0]
    dist_matrix = torch.cdist(emb, emb, p=2).cpu().numpy()
    idx_upper = np.triu_indices(V, k=1)
    embed_dists = dist_matrix[idx_upper]
    i_idx, j_idx = idx_upper
    pixel_dists = np.abs(i_idx - j_idx)
    if embed_dists.size == 0:
        return 0.0, 0.0, 0.0
    corr = np.corrcoef(pixel_dists, embed_dists)[0, 1]
    from scipy.stats import linregress

    slope, intercept, r_value, p_value, std_err = linregress(pixel_dists, embed_dists)
    return float(corr), float(slope), float(r_value**2)


def visualize_distance_matrix(
    embeddings: torch.Tensor,
    step: str | int,
    save_path: Optional[Path] = None,
) -> None:
    emb = embeddings.detach()
    device = emb.device
    V = emb.shape[0]
    dist_matrix = torch.cdist(emb, emb, p=2).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(dist_matrix, ax=axes[0], cmap="viridis")
    axes[0].set_title(f"Distance Matrix (Step {step})")
    axes[0].set_xlabel("Token index")
    axes[0].set_ylabel("Token index")

    pixel_dists = np.abs(np.arange(V)[:, None] - np.arange(V)[None, :])
    axes[1].scatter(pixel_dists.flatten(), dist_matrix.flatten(), alpha=0.1, s=2)
    axes[1].plot([0, V - 1], [0, dist_matrix.max()], "r--", label="Ideal")
    axes[1].set_xlabel("Pixel distance")
    axes[1].set_ylabel("Embedding distance")
    axes[1].set_title("Distance Correlation")
    axes[1].legend()

    ref_tokens = [0, V // 4, V // 2, 3 * V // 4, V - 1]
    for ref in ref_tokens:
        axes[2].plot(dist_matrix[ref], label=f"Token {ref}")
    axes[2].set_xlabel("Token index")
    axes[2].set_ylabel("Distance")
    axes[2].set_title("Distance Profiles")
    axes[2].legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def check_isotropy(embeddings: torch.Tensor) -> Tuple[float, float]:
    """
    Check embedding norm variance and angular isotropy.
    Returns (variance_of_norms, mean_abs_cov_error).
    """
    emb = embeddings.detach()
    norms = emb.norm(dim=-1)
    norm_var = norms.var().item()
    normed = emb / norms.clamp_min(1e-12).unsqueeze(-1)
    cov = normed.T @ normed / emb.shape[0]
    D = emb.shape[1]
    identity = torch.eye(D, device=cov.device, dtype=cov.dtype)
    isotropy = (cov - identity).abs().mean().item()
    return norm_var, isotropy


def compute_effective_rank(embeddings: torch.Tensor) -> float:
    """
    Compute normalized effective rank (in [0,1]) of embeddings.
    """
    emb = embeddings.detach()
    cov = emb.T @ emb / emb.shape[0]
    eigenvalues = torch.linalg.eigvalsh(cov).clamp_min(0)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    p = eigenvalues / total
    entropy = -(p * (p + 1e-12).log()).sum()
    eff_rank = torch.exp(entropy)
    return (eff_rank / eigenvalues.numel()).item()


def check_token_usage(
    dataloader,
    embeddings: torch.Tensor,
    top_k: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Estimate token usage from a dataloader and evaluate metric quality on most-used tokens.

    Args:
        dataloader: Iterable yielding token tensors (or tuples whose first element are tokens).
        embeddings: Tensor [V, D] of token embeddings.
        top_k: Number of most frequent tokens to inspect.
        device: Optional device for accumulation (defaults to embeddings.device).

    Returns:
        Dictionary with token counts, top tokens, and metric diagnostics.
    """
    emb = embeddings.detach()
    vocab_size = emb.shape[0]
    accum_device = device or emb.device
    counts = torch.zeros(vocab_size, device=accum_device, dtype=torch.float64)

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            tokens = batch[0]
        else:
            tokens = batch
        tokens = tokens.to(accum_device)
        flat = tokens.view(-1)
        unique, freq = flat.unique(return_counts=True)
        counts[unique] += freq.to(torch.float64)

    top_k = min(top_k, vocab_size)
    top_tokens = torch.topk(counts, k=top_k).indices.to(torch.long)
    sub_emb = emb[top_tokens]

    metrics = {
        "token_counts": counts.cpu(),
        "top_tokens": top_tokens.cpu(),
        "top_counts": counts[top_tokens].cpu(),
        "monotonicity_violation": check_monotonicity(sub_emb),
        "distance_metrics": compute_distance_correlation(sub_emb),
        "isotropy": check_isotropy(sub_emb),
        "effective_rank": compute_effective_rank(sub_emb),
    }
    return metrics


def plot_similarity_matrix(
    sim: torch.Tensor,
    channel: int = 0,
    title: str = "LUT similarity",
    cmap: str = "viridis",
    save_path: Path | None = None,
) -> None:
    if channel < 0 or channel >= sim.shape[0]:
        raise ValueError(f"Channel {channel} is out of range (0 <= ch < {sim.shape[0]})")
    matrix = sim[channel].detach().cpu().numpy()
    plt.figure(figsize=(6, 5))
    im = plt.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.title(f"{title} (channel={channel})")
    plt.xlabel("token index")
    plt.ylabel("token index")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect learnable LUT geometry.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lut", type=Path, help="Path to LUT snapshot (.pt) saved via save_lut_snapshot.")
    group.add_argument("--checkpoint", type=Path, help="Path to training checkpoint containing extra_modules.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline LUT snapshot (.pt).")
    parser.add_argument("--lut-key", type=str, default="metric_learnable_lut", help="Key inside checkpoint['extra_modules'] for the learnable LUT.")
    parser.add_argument("--normalized", action="store_true", help="Use normalized distances (divide by sqrt(D)).")
    parser.add_argument(
        "--sphere",
        action="store_true",
        help=("Compute metrics on unit-normalized embeddings using chord lengths (cosine geometry). "
              "When no baseline is provided, a great-circle baseline is used."),
    )
    parser.add_argument("--projection-mode", choices=["none", "mean", "uniform"], default="none", help="Compute signed projection curve.")
    parser.add_argument("--export-projection", type=Path, default=None, help="Where to save projection curve (npz with per-channel arrays).")
    parser.add_argument("--plot-similarity", action="store_true", help="Plot an Lp-distance-based similarity matrix heatmap.")
    parser.add_argument("--similarity-channel", type=int, default=0, help="Channel index for similarity plot.")
    parser.add_argument("--similarity-kernel", choices=["gaussian", "recip"], default="gaussian", help="Similarity kernel to apply to Lp distances.")
    parser.add_argument("--similarity-lp-order", type=float, default=2.0, help="Lp order used when computing pairwise distances.")
    parser.add_argument("--similarity-sigma-scale", type=float, default=1.0, help="Scale factor applied to Gaussian kernel sigma (median distance * scale).")
    parser.add_argument("--similarity-save", type=Path, default=None, help="Optional path to save the similarity heatmap image.")
    parser.add_argument("--plot-attention-sim", action="store_true", help="Plot an attention-style similarity matrix (dot/cosine).")
    parser.add_argument("--attention-kind", choices=["dot", "cosine"], default="dot", help="Similarity kind for attention-style matrix.")
    parser.add_argument("--attention-temperature", type=float, default=None, help="Temperature applied before softmax (default 1.0).")
    parser.add_argument("--attention-no-softmax", action="store_true", help="Disable softmax when computing attention-style similarity.")
    parser.add_argument("--attention-channel", type=int, default=0, help="Channel index for attention similarity plot.")
    parser.add_argument("--attention-save", type=Path, default=None, help="Optional path to save attention similarity heatmap.")
    parser.add_argument("--analyze-frequency", action="store_true", help="Analyze frequency characteristics of a LUT channel.")
    parser.add_argument("--frequency-lut-channel", type=int, default=0, help="LUT channel index (C dimension) for frequency analysis.")
    parser.add_argument("--frequency-embed-dim", type=int, default=0, help="Embedding dimension (within channel) for frequency analysis.")
    parser.add_argument("--analyze-frequency-detailed", action="store_true", help="Perform detailed frequency analysis (top-k peaks).")
    parser.add_argument("--frequency-topk", type=int, default=5, help="Number of dominant frequencies to report in detailed analysis.")
    parser.add_argument("--check-monotonicity", action="store_true", help="Check distance monotonicity constraint for a LUT channel.")
    parser.add_argument("--distance-correlation", action="store_true", help="Compute correlation between pixel distance and embedding distance.")
    parser.add_argument("--visualize-distance-matrix", action="store_true", help="Visualize distance matrix and related plots.")
    parser.add_argument("--distance-matrix-save", type=Path, default=None, help="Optional path to save distance matrix visualization.")
    parser.add_argument("--check-isotropy", action="store_true", help="Evaluate norm variance and angular isotropy of embeddings.")
    parser.add_argument("--effective-rank", action="store_true", help="Compute normalized effective rank of embedding covariance.")
    parser.add_argument("--check-token-usage", action="store_true", help="Analyze token usage statistics given a token file.")
    parser.add_argument("--token-usage-file", type=Path, default=None, help="Path to .pt/.pth/.npy/.npz file containing token indices.")
    parser.add_argument("--token-usage-topk", type=int, default=64, help="Top-K tokens to inspect when checking token usage.")
    args = parser.parse_args()

    if args.sphere and args.normalized:
        print("Note: --normalized is ignored when --sphere is enabled (unit vectors already used).")

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
        if args.sphere and base_weight.shape[-1] < 2:
            raise ValueError("Sphere mode requires a baseline with embedding dimension >= 2.")
    else:
        if args.sphere:
            if lut_weight.shape[-1] < 2:
                raise ValueError("Sphere mode requires embedding dimension >= 2.")
            base_weight = _great_circle_baseline(
                num_channels=meta["num_channels"],
                vocab_size=meta["vocab_size"],
                embed_dim=lut_weight.shape[-1],
                device=lut_weight.device,
                dtype=lut_weight.dtype,
            )
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
        sphere_mode=args.sphere,
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

    if args.plot_similarity:
        sim = compute_lut_similarity(
            lut_weight,
            lp_order=args.similarity_lp_order,
            kernel=args.similarity_kernel,
            sigma_scale=args.similarity_sigma_scale,
        )
        plot_similarity_matrix(
            sim,
            channel=args.similarity_channel,
            title=f"Lp({args.similarity_lp_order}) similarity",
            save_path=args.similarity_save,
        )
    if args.plot_attention_sim:
        attn_sim = attention_like_similarity(
            lut_weight,
            kind=args.attention_kind,
            temperature=args.attention_temperature,
            apply_softmax=not args.attention_no_softmax,
        )
        plot_similarity_matrix(
            attn_sim,
            channel=args.attention_channel,
            title=f"Attention-like ({args.attention_kind}) similarity",
            save_path=args.attention_save,
        )
    if args.analyze_frequency:
        if not (0 <= args.frequency_lut_channel < lut_weight.shape[0]):
            raise ValueError(
                f"frequency-lut-channel {args.frequency_lut_channel} out of range (0 <= ch < {lut_weight.shape[0]})"
            )
        embeddings = lut_weight[args.frequency_lut_channel]
        analyze_frequency(embeddings, channel_idx=args.frequency_embed_dim)
    if args.analyze_frequency_detailed:
        if not (0 <= args.frequency_lut_channel < lut_weight.shape[0]):
            raise ValueError(
                f"frequency-lut-channel {args.frequency_lut_channel} out of range (0 <= ch < {lut_weight.shape[0]})"
            )
        embeddings = lut_weight[args.frequency_lut_channel]
        analyze_frequency_detailed(
            embeddings,
            channel_idx=args.frequency_embed_dim,
            top_k=max(1, args.frequency_topk),
        )
    if (
        args.check_monotonicity
        or args.distance_correlation
        or args.visualize_distance_matrix
        or args.check_isotropy
        or args.effective_rank
        or args.check_token_usage
    ):
        if not (0 <= args.frequency_lut_channel < lut_weight.shape[0]):
            raise ValueError(
                f"frequency-lut-channel {args.frequency_lut_channel} out of range (0 <= ch < {lut_weight.shape[0]})"
            )
        embeddings = lut_weight[args.frequency_lut_channel]
        print(f"\nAnalysis channel (LUT channel={args.frequency_lut_channel})")
        if args.check_monotonicity:
            violation_rate = check_monotonicity(embeddings)
            print(f"Distance monotonicity violation rate: {violation_rate * 100:.2f}%")
        if args.distance_correlation:
            corr, slope, r2 = compute_distance_correlation(embeddings)
            print(f"Distance correlation (pearson): {corr:.4f}")
            print(f"Slope: {slope:.4f}, R^2: {r2:.4f}")
        if args.visualize_distance_matrix:
            step_label = args.checkpoint if args.checkpoint is not None else args.lut
            visualize_distance_matrix(
                embeddings,
                step=step_label,
                save_path=args.distance_matrix_save,
            )
        if args.check_isotropy:
            norm_var, isotropy = check_isotropy(embeddings)
            print(f"Norm variance: {norm_var:.6f}")
            print(f"Mean absolute covariance error: {isotropy:.6f}")
        if args.effective_rank:
            eff_rank = compute_effective_rank(embeddings)
            print(f"Normalized effective rank: {eff_rank:.6f}")
        if args.check_token_usage:
            if args.token_usage_file is None:
                raise ValueError("--token-usage-file must be provided when using --check-token-usage")
            token_tensor = _load_token_tensor(args.token_usage_file)
            dataloader = [token_tensor.reshape(-1)]
            usage_metrics = check_token_usage(
                dataloader,
                embeddings,
                top_k=args.token_usage_topk,
                device=embeddings.device,
            )
            print("Token usage summary:")
            print(f"  Top tokens: {usage_metrics['top_tokens'].tolist()}")
            print(f"  Counts: {usage_metrics['top_counts'].tolist()}")
            print(
                f"  Subset monotonicity violation: {usage_metrics['monotonicity_violation'] * 100:.2f}%"
            )
            corr, slope, r2 = usage_metrics["distance_metrics"]
            print(
                f"  Subset distance corr: {corr:.4f}, slope={slope:.4f}, R^2={r2:.4f}"
            )
            norm_var, isotropy = usage_metrics["isotropy"]
            print(f"  Subset norm variance: {norm_var:.6f}, covariance error: {isotropy:.6f}")
            print(f"  Subset effective rank: {usage_metrics['effective_rank']:.6f}")


if __name__ == "__main__":
    main()
