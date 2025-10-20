"""Shared utilities for analyzing and plotting LUT quantile diagnostics.

This module centralizes quantile baselines, scalarization helpers, and
visualizations so both training-time diagnostics and the offline
`analyze_lut_quantile_hypothesis.py` script can use the same logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# Scalarization helpers
# ---------------------------------------------------------------------------

def scalarize_embedding(e_raw: torch.Tensor, mode: str = "uniform") -> torch.Tensor:
    """Convert [V, D] embedding into a scalar curve [V] using the requested mode."""
    if e_raw.ndim != 2:
        raise ValueError(f"Expected [V, D] tensor, got {tuple(e_raw.shape)}")
    V, D = e_raw.shape
    if D == 1:
        return e_raw.squeeze(-1)

    mode_norm = mode.lower()
    if mode_norm == "norm":
        return torch.linalg.vector_norm(e_raw, ord=2, dim=-1)
    if mode_norm == "norm_normalized":
        return torch.linalg.vector_norm(e_raw, ord=2, dim=-1) / math.sqrt(float(D))
    if mode_norm == "mean":
        return e_raw.mean(dim=-1)
    if mode_norm == "uniform":
        u = torch.ones(D, device=e_raw.device, dtype=e_raw.dtype) / math.sqrt(float(D))
        return e_raw @ u
    if mode_norm == "pc1":
        X = e_raw.detach().double().cpu().numpy()
        X_center = X - X.mean(axis=0, keepdims=True)
        if np.allclose(X_center, 0.0):
            return torch.zeros(V, device=e_raw.device, dtype=e_raw.dtype)
        U, S, _ = np.linalg.svd(X_center, full_matrices=False)
        comp = U[:, 0] * S[0]
        return torch.from_numpy(comp).to(device=e_raw.device, dtype=e_raw.dtype)
    raise ValueError(f"Unknown scalarization mode '{mode}'")


# ---------------------------------------------------------------------------
# Quantile baselines
# ---------------------------------------------------------------------------

def compute_empirical_cdf(
    histogram: torch.Tensor,
    smoothing_sigma: float = 0.5,
) -> torch.Tensor:
    """Compute smoothed empirical CDF from histogram [C, V]."""
    if histogram.ndim != 2:
        raise ValueError(f"Expected histogram [C, V], got {tuple(histogram.shape)}")
    hist = histogram.detach().cpu().float()
    if smoothing_sigma > 0:
        hist_np = hist.numpy()
        hist_smooth = np.array(
            [
                gaussian_filter1d(hist_np[c], sigma=smoothing_sigma, mode="nearest")
                for c in range(hist_np.shape[0])
            ]
        )
        hist = torch.from_numpy(hist_smooth)
    hist = torch.clamp(hist, min=1e-8)
    pmf = hist / hist.sum(dim=1, keepdim=True)
    cdf = torch.cumsum(pmf, dim=1)
    return torch.clamp(cdf, 0.0, 1.0)


def baseline_uniform(
    cdf: torch.Tensor,
    target_range: Sequence[float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Quantile map to a uniform distribution over target_range."""
    low, high = float(target_range[0]), float(target_range[1])
    return cdf * (high - low) + low


def baseline_probit(
    cdf: torch.Tensor,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> torch.Tensor:
    """Quantile map to a normal distribution via probit."""
    cdf_np = torch.clamp(cdf, 1e-6, 1 - 1e-6).numpy()
    quantiles = stats.norm.ppf(cdf_np, loc=target_mean, scale=target_std)
    return torch.from_numpy(quantiles).to(cdf)


def baseline_isotonic(embedding: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Fit isotonic regression (monotone) to the learned embedding."""
    if embedding.ndim != 2:
        raise ValueError(f"embedding must be [C, V]; got {tuple(embedding.shape)}")
    C, V = embedding.shape
    X = tokens.cpu().numpy()
    iso = IsotonicRegression(out_of_bounds="clip")
    result = []
    for c in range(C):
        y = embedding[c].detach().cpu().numpy()
        iso.fit(X, y)
        result.append(iso.predict(X))
    return torch.from_numpy(np.stack(result, axis=0)).to(embedding)


def _match_range(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Rescale target to match min/max of reference (avoid zero division)."""
    t_min, t_max = target.min(), target.max()
    r_min, r_max = reference.min(), reference.max()
    if (t_max - t_min) < 1e-12:
        return reference * 0 + (t_min + t_max) * 0.5
    scale = (r_max - r_min) / (t_max - t_min + 1e-12)
    return (target - t_min) * scale + r_min


# ---------------------------------------------------------------------------
# Metrics & analysis bundle
# ---------------------------------------------------------------------------

def compute_baseline_metrics(
    learned: torch.Tensor,
    baseline: torch.Tensor,
    name: str,
) -> Dict[str, float]:
    """Compute regression-style metrics comparing learned vs baseline curves."""
    x = baseline.detach().cpu().numpy()
    y = learned.detach().cpu().numpy()
    residual = y - x
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    rmse = math.sqrt(ss_res / len(y))
    mae = float(np.abs(residual).mean())
    spearman, _ = stats.spearmanr(x, y)
    kendall, _ = stats.kendalltau(x, y)
    return {
        "name": name,
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "SpearmanR": float(spearman),
        "KendallTau": float(kendall),
    }


@dataclass
class BaselineResults:
    channel_index: int
    scalar_mode: str
    tokens: torch.Tensor
    embedding: torch.Tensor
    baselines: Dict[str, torch.Tensor]
    metrics: Dict[str, Dict[str, float]]
    cdf: Optional[torch.Tensor] = None
    histogram: Optional[torch.Tensor] = None


def analyze_baselines(
    lut_weight: torch.Tensor,
    data_histogram: Optional[torch.Tensor],
    channel_idx: int = 0,
    *,
    scalar_mode: str = "uniform",
    smoothing_sigma: float = 1.0,
) -> BaselineResults:
    """Prepare scalar embedding and baseline curves for one channel."""
    lut_weight = lut_weight.detach().cpu()
    C, V, D = lut_weight.shape
    if channel_idx < 0 or channel_idx >= C:
        raise IndexError(f"channel_idx {channel_idx} out of range (0..{C-1})")

    tokens = torch.arange(V, dtype=torch.float32)
    E_raw = lut_weight[channel_idx]
    E = scalarize_embedding(E_raw, mode=scalar_mode)

    if data_histogram is None:
        hist = torch.ones(C, V, dtype=torch.float32)
    else:
        hist = data_histogram.detach().cpu().float()
        if hist.shape != (C, V):
            raise ValueError(
                f"Histogram shape {tuple(hist.shape)} does not match ({C}, {V})"
            )

    cdf = compute_empirical_cdf(hist, smoothing_sigma=smoothing_sigma)[channel_idx]

    T1 = baseline_uniform(cdf.unsqueeze(0), target_range=(-1.0, 1.0)).squeeze(0)
    T1 = _match_range(T1, E)

    T2 = baseline_probit(cdf.unsqueeze(0)).squeeze(0)
    T2 = _match_range(T2, E)

    T3 = baseline_isotonic(E.unsqueeze(0), tokens).squeeze(0)

    metrics = {
        "T1_uniform": compute_baseline_metrics(E, T1, "T1: Uniform"),
        "T2_probit": compute_baseline_metrics(E, T2, "T2: Probit"),
        "T3_isotonic": compute_baseline_metrics(E, T3, "T3: Isotonic"),
    }

    return BaselineResults(
        channel_index=channel_idx,
        scalar_mode=scalar_mode,
        tokens=tokens,
        embedding=E,
        baselines={"T1": T1, "T2": T2, "T3": T3},
        metrics=metrics,
        cdf=cdf,
        histogram=hist[channel_idx],
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_baseline_comparison(
    results: BaselineResults,
    output_path: str,
    channel_label: str,
    *,
    epoch: Optional[int] = None,
    figure_title: Optional[str] = None,
) -> None:
    """Generate a detailed multi-panel diagnostic figure."""
    tokens = results.tokens.numpy()
    E = results.embedding.numpy()
    T1 = results.baselines["T1"].numpy()
    T2 = results.baselines["T2"].numpy()
    T3 = results.baselines["T3"].numpy()
    residuals = {
        "T1": E - T1,
        "T2": E - T2,
        "T3": E - T3,
    }

    dE = np.diff(E)
    dT1 = np.diff(T1)
    dT2 = np.diff(T2)
    dT3 = np.diff(T3)
    x_mid = tokens[:-1] + 0.5
    curvature = np.diff(dE)
    x_curv = tokens[1:-1] + 1.0

    # Normalize colors for scatter by token index
    norm = mcolors.Normalize(vmin=tokens.min(), vmax=tokens.max())
    cmap = plt.get_cmap("viridis")
    scatter_colors = cmap(norm(tokens))

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # --- row 0: curves, residuals, histogram/CDF --------------------------------
    ax = axes[0, 0]
    ax.plot(tokens, E, color="black", linewidth=2.0, label="E (Learned)")
    ax.plot(tokens, T1, color="#d62728", linestyle="--", linewidth=1.5, label="T1: Uniform")
    ax.plot(tokens, T2, color="#2ca02c", linestyle="--", linewidth=1.5, label="T2: Probit")
    ax.plot(tokens, T3, color="#1f77b4", linestyle="--", linewidth=1.5, label="T3: Isotonic")
    ax.set_xlabel("Token Value")
    ax.set_ylabel("Embedding Value")
    ax.set_title(f"Channel {channel_label}: Learned vs Baselines")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    ax = axes[0, 1]
    ax.plot(tokens, residuals["T1"], color="#d62728", linewidth=1.2, label="E - T1")
    ax.plot(tokens, residuals["T2"], color="#2ca02c", linewidth=1.2, label="E - T2")
    ax.plot(tokens, residuals["T3"], color="#1f77b4", linewidth=1.2, label="E - T3")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Token Value")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    ax = axes[0, 2]
    if results.histogram is not None:
        hist = results.histogram.numpy()
        hist = hist / max(hist.sum(), 1e-12)
        ax.bar(tokens, hist, width=1.0, color="lightgray", edgecolor="black", alpha=0.6, label="Histogram")
        ax.set_ylabel("Token PMF")
    if results.cdf is not None:
        ax_cdf = ax.twinx()
        ax_cdf.plot(tokens, results.cdf.numpy(), color="navy", linewidth=1.8, label="CDF")
        ax_cdf.set_ylabel("CDF")
        ax_cdf.set_ylim(0.0, 1.0)
        ax_cdf.grid(False)
        ax_cdf.tick_params(axis="y", colors="navy")
    ax.set_xlabel("Token Value")
    ax.set_title("Data Histogram & CDF")
    ax.grid(True, alpha=0.3)

    # --- row 1: derivatives ------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(x_mid, dE, color="black", linewidth=1.5, label="ΔE")
    ax.plot(x_mid, dT1, color="#d62728", linestyle="--", linewidth=1.0, label="ΔT1")
    ax.plot(x_mid, dT2, color="#2ca02c", linestyle="--", linewidth=1.0, label="ΔT2")
    ax.plot(x_mid, dT3, color="#1f77b4", linestyle="--", linewidth=1.0, label="ΔT3")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Token Value")
    ax.set_ylabel("First Difference")
    ax.set_title("Local Slope")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1, 1]
    ax.plot(x_curv, curvature, color="black", linewidth=1.2)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Token Value")
    ax.set_ylabel("Second Difference")
    ax.set_title("Curvature of E")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(x_mid, dE - dT1, color="#d62728", linewidth=1.0, label="ΔE - ΔT1")
    ax.plot(x_mid, dE - dT2, color="#2ca02c", linewidth=1.0, label="ΔE - ΔT2")
    ax.plot(x_mid, dE - dT3, color="#1f77b4", linewidth=1.0, label="ΔE - ΔT3")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Token Value")
    ax.set_ylabel("Slope Residual")
    ax.set_title("Slope Residuals")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # --- row 2: scatter + metrics -----------------------------------------------
    ax = axes[2, 0]
    ax.scatter(T1, E, c=scatter_colors, s=12, alpha=0.7)
    diag_min, diag_max = min(T1.min(), E.min()), max(T1.max(), E.max())
    ax.plot([diag_min, diag_max], [diag_min, diag_max], color="black", linestyle="--", linewidth=1.0)
    m1 = results.metrics["T1_uniform"]
    ax.set_xlabel("T1: Uniform")
    ax.set_ylabel("E (Learned)")
    ax.set_title(f"E vs T1 (R²={m1['R2']:.3f}, RMSE={m1['RMSE']:.3f})")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.scatter(T2, E, c=scatter_colors, s=12, alpha=0.7)
    diag_min, diag_max = min(T2.min(), E.min()), max(T2.max(), E.max())
    ax.plot([diag_min, diag_max], [diag_min, diag_max], color="black", linestyle="--", linewidth=1.0)
    m2 = results.metrics["T2_probit"]
    ax.set_xlabel("T2: Probit")
    ax.set_ylabel("E (Learned)")
    ax.set_title(f"E vs T2 (R²={m2['R2']:.3f}, RMSE={m2['RMSE']:.3f})")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.scatter(T3, E, c=scatter_colors, s=12, alpha=0.7)
    diag_min, diag_max = min(T3.min(), E.min()), max(T3.max(), E.max())
    ax.plot([diag_min, diag_max], [diag_min, diag_max], color="black", linestyle="--", linewidth=1.0)
    m3 = results.metrics["T3_isotonic"]
    ax.set_xlabel("T3: Isotonic")
    ax.set_ylabel("E (Learned)")
    ax.set_title(f"E vs T3 (R²={m3['R2']:.3f}, RMSE={m3['RMSE']:.3f})")
    ax.grid(True, alpha=0.3)

    summary_lines = [
        f"Scalar mode: {results.scalar_mode}",
        f"E range: [{E.min():.4f}, {E.max():.4f}]",
        f"E std: {E.std():.4f}",
    ]
    summary_lines += [
        f"{name}: R²={metrics['R2']:.3f}, Spearman={metrics['SpearmanR']:.3f}, Kendall={metrics['KendallTau']:.3f}"
        for name, metrics in results.metrics.items()
    ]
    if epoch is not None:
        summary_lines.insert(0, f"Epoch: {epoch}")

    fig.suptitle(
        figure_title
        or f"LUT Quantile Diagnostics — Channel {channel_label} (mode={results.scalar_mode})",
        fontsize=16,
        fontweight="bold",
    )
    axes[1, 2].text(
        0.02,
        0.95,
        "\n".join(summary_lines),
        transform=axes[1, 2].transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

