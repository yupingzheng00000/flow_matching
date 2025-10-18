"""
Comprehensive LUT Analysis: Quantile Hypothesis Testing

Implements two key diagnostic probes:

A. Quantile Equalization Hypothesis
   - Compare learned E_c(v) against three monotone baselines:
     T1: Equalize-to-uniform (2*F_c(v) - 1)
     T2: Equalize-to-normal (probit transform: Φ^(-1)(F_c(v)))
     T3: Isotonic regression (unconstrained monotone fit)
   - Report R², RMSE, MAE, Spearman/Kendall correlations

C. Pushforward Distribution Tests
   - Sample x_1 tokens, map through y = E_c(x_1)
   - Test if y is closer to Uniform[-1,1] or Normal than original x_1
   - Use KS/CvM/AD tests, QQ plots, Shapiro-Wilk

Handles both scalar (D=1) and multi-dimensional (D>1) embeddings.
For D>1, uses L2 norm or PC1 projection for visualization.

Usage:
    python analyze_lut_quantile_hypothesis.py --checkpoint path/to/checkpoint.pth \\
           --data_histogram path/to/cifar10_histogram.pt \\
           --output_dir ./lut_analysis_output
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent / "flow_matching"))


# ============================================================================
# Utility Functions
# ============================================================================

def load_lut_from_checkpoint(checkpoint_path: str) -> Tuple[torch.Tensor, dict]:
    """Load LUT weights from checkpoint.
    
    Returns:
        lut_weight: Tensor of shape [C, V, D]
        metadata: Dict with num_channels, vocab_size, emb_dim
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Try extra_modules first (newer format)
    lut_weight = None
    if "extra_modules" in ckpt and "metric_learnable_lut" in ckpt["extra_modules"]:
        lut_dict = ckpt["extra_modules"]["metric_learnable_lut"]
        if "weight" in lut_dict:
            lut_weight = lut_dict["weight"]
            print("  ✓ Found LUT in extra_modules['metric_learnable_lut']")
    
    # Fallback: search in ema_model (older format)
    if lut_weight is None and "ema_model" in ckpt:
        for key in ckpt["ema_model"].keys():
            if "learnable_lut.weight" in key:
                lut_weight = ckpt["ema_model"][key]
                print(f"  ✓ Found LUT in ema_model['{key}']")
                break
    
    if lut_weight is None:
        raise ValueError("No learnable_lut.weight found in checkpoint (checked extra_modules and ema_model)")
    
    # Ensure [C, V, D] shape
    
    # Infer dimensions
    if lut_weight.ndim == 2:
        lut_weight = lut_weight.unsqueeze(-1)  # [C, V, 1]
    
    C, V, D = lut_weight.shape
    metadata = {
        "num_channels": C,
        "vocab_size": V,
        "emb_dim": D,
        "checkpoint_path": checkpoint_path,
    }
    
    print(f"✓ Loaded LUT: shape={list(lut_weight.shape)}, channels={C}, vocab={V}, emb_dim={D}")
    return lut_weight, metadata


def _scalarize_embedding(e_raw: torch.Tensor, mode: str = "uniform") -> torch.Tensor:
    """Convert [V,D] embedding into a scalar curve [V].

    Modes:
      - 'norm': L2 norm ||E||
      - 'norm_normalized': ||E|| / sqrt(D)
      - 'mean': per-token dimension mean
      - 'uniform': dot(E, 1/sqrt(D)) (signed projection)
      - 'pc1': projection to first principal component (signed)
    """
    assert e_raw.ndim == 2, "Expected [V,D]"
    V, D = e_raw.shape
    if D == 1:
        return e_raw.squeeze(-1)
    mode = str(mode).lower()
    if mode == "norm":
        return torch.norm(e_raw, p=2, dim=-1)
    if mode == "norm_normalized":
        return torch.norm(e_raw, p=2, dim=-1) / torch.sqrt(torch.tensor(float(D), device=e_raw.device, dtype=e_raw.dtype))
    if mode == "mean":
        return e_raw.mean(dim=-1)
    if mode == "uniform":
        u = torch.ones(D, device=e_raw.device, dtype=e_raw.dtype) / torch.sqrt(torch.tensor(float(D), device=e_raw.device, dtype=e_raw.dtype))
        return e_raw @ u
    if mode == "pc1":
        with torch.no_grad():
            X = e_raw.double().cpu().numpy()
            pca = PCA(n_components=1)
            comp = pca.fit_transform(X)[:, 0]
            return torch.from_numpy(comp).to(dtype=e_raw.dtype, device=e_raw.device)
    raise ValueError(f"Unknown scalarization mode '{mode}'")


def estimate_data_histogram(
    data_path: Optional[str] = None,
    vocab_size: int = 256,
    num_channels: int = 3,
) -> torch.Tensor:
    """Estimate per-channel token histogram from data.
    
    Args:
        data_path: Path to saved histogram tensor, or None to use synthetic
        vocab_size: Number of tokens (256 for 8-bit images)
        num_channels: Number of channels (3 for RGB)
    
    Returns:
        histogram: Tensor of shape [C, V] with token counts (unnormalized)
    """
    if data_path and Path(data_path).exists():
        hist = torch.load(data_path, map_location="cpu")
        print(f"✓ Loaded data histogram from {data_path}, shape={list(hist.shape)}")
        return hist
    
    # Synthetic: assume approximately Gaussian-like distribution
    # (real CIFAR-10 is bimodal with peaks at edges, but this is a fallback)
    print("⚠ No data histogram provided, using synthetic Gaussian approximation")
    x = torch.linspace(0, vocab_size - 1, vocab_size)
    mu, sigma = vocab_size / 2, vocab_size / 4
    hist = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    hist = hist.unsqueeze(0).repeat(num_channels, 1)  # [C, V]
    hist = hist / hist.sum(dim=1, keepdim=True) * 10000  # Normalize to ~10k samples
    return hist


def compute_empirical_cdf(histogram: torch.Tensor, smoothing_sigma: float = 0.5) -> torch.Tensor:
    """Compute smoothed empirical CDF from histogram.
    
    Args:
        histogram: [C, V] token counts
        smoothing_sigma: Gaussian kernel std for smoothing (0 = no smoothing)
    
    Returns:
        cdf: [C, V] cumulative distribution function F_c(v) ∈ [0, 1]
    """
    C, V = histogram.shape
    
    # Smooth histogram to reduce noise
    if smoothing_sigma > 0:
        hist_np = histogram.numpy()
        smoothed = np.array([
            gaussian_filter1d(hist_np[c], sigma=smoothing_sigma, mode='nearest')
            for c in range(C)
        ])
        hist_smooth = torch.from_numpy(smoothed).float()
    else:
        hist_smooth = histogram.float()
    
    # Compute CDF
    hist_smooth = torch.clamp(hist_smooth, min=1e-8)  # Avoid zeros
    pmf = hist_smooth / hist_smooth.sum(dim=1, keepdim=True)  # Normalize to PMF
    cdf = torch.cumsum(pmf, dim=1)  # Cumulative sum
    cdf = torch.clamp(cdf, min=0.0, max=1.0)  # Numerical stability
    
    return cdf


# ============================================================================
# Probe A: Quantile Equalization Baselines
# ============================================================================

def baseline_uniform(cdf: torch.Tensor, target_range: Tuple[float, float] = (-1, 1)) -> torch.Tensor:
    """T1: Equalize to uniform distribution.
    
    Args:
        cdf: [C, V] empirical CDF F_c(v)
        target_range: (min, max) for output range
    
    Returns:
        T1: [C, V] baseline embeddings
    """
    a, b = target_range
    return a + (b - a) * cdf


def baseline_probit(cdf: torch.Tensor, target_mean: float = 0.0, target_std: float = 1.0) -> torch.Tensor:
    """T2: Equalize to normal distribution (probit transform).
    
    Args:
        cdf: [C, V] empirical CDF F_c(v)
        target_mean: Mean of target normal
        target_std: Std of target normal
    
    Returns:
        T2: [C, V] baseline embeddings
    """
    # Avoid probit(0) = -∞ and probit(1) = +∞
    cdf_clipped = torch.clamp(cdf, min=1e-6, max=1 - 1e-6)
    
    # Probit: Φ^(-1)(p)
    probit = torch.from_numpy(stats.norm.ppf(cdf_clipped.numpy())).float()
    
    # Rescale to target mean/std
    probit_scaled = target_mean + target_std * probit
    
    return probit_scaled


def baseline_isotonic(
    lut_values: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """T3: Isotonic regression (monotone unconstrained fit).
    
    Args:
        lut_values: [V] or [C, V] learned embeddings (target)
        tokens: [V] token indices (0, 1, ..., V-1)
    
    Returns:
        T3: [V] or [C, V] isotonic fit
    """
    if lut_values.ndim == 1:
        lut_values = lut_values.unsqueeze(0)  # [1, V]
    
    C, V = lut_values.shape
    T3 = torch.zeros_like(lut_values)
    
    for c in range(C):
        iso_reg = IsotonicRegression(increasing=True)
        y_fit = iso_reg.fit_transform(tokens.numpy(), lut_values[c].numpy())
        T3[c] = torch.from_numpy(y_fit).float()
    
    return T3.squeeze() if C == 1 else T3


def compute_baseline_metrics(
    E: torch.Tensor,
    T: torch.Tensor,
    name: str,
) -> Dict[str, float]:
    """Compute fit quality metrics between E and baseline T.
    
    Args:
        E: [V] learned embeddings
        T: [V] baseline embeddings
        name: Baseline name (for printing)
    
    Returns:
        metrics: Dict with R², RMSE, MAE, Spearman ρ, Kendall τ
    """
    E_np = E.numpy()
    T_np = T.numpy()
    
    # R² (coefficient of determination)
    ss_res = np.sum((E_np - T_np) ** 2)
    ss_tot = np.sum((E_np - E_np.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # RMSE and MAE
    rmse = np.sqrt(np.mean((E_np - T_np) ** 2))
    mae = np.mean(np.abs(E_np - T_np))
    
    # Spearman and Kendall (rank correlations)
    spearman_rho, _ = stats.spearmanr(E_np, T_np)
    kendall_tau, _ = stats.kendalltau(E_np, T_np)
    
    return {
        "name": name,
        "R²": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Spearman_ρ": spearman_rho,
        "Kendall_τ": kendall_tau,
    }


def analyze_baselines(
    lut_weight: torch.Tensor,
    data_histogram: torch.Tensor,
    channel_idx: int = 0,
    scalar_mode: str = "uniform",
) -> Dict[str, any]:
    """Run baseline comparison for one channel.
    
    Args:
        lut_weight: [C, V, D] LUT weights
        data_histogram: [C, V] token counts
        channel_idx: Which channel to analyze
    
    Returns:
        results: Dict with baselines, metrics, and plots
    """
    C, V, D = lut_weight.shape
    
    # Extract embeddings for this channel
    E_raw = lut_weight[channel_idx, :, :]  # [V, D]
    E = _scalarize_embedding(E_raw, mode=scalar_mode)
    if D > 1:
        print(f"  Using scalarization='{scalar_mode}' for {D}D embeddings (channel {channel_idx})")
    
    # Compute empirical CDF
    cdf = compute_empirical_cdf(data_histogram, smoothing_sigma=1.0)  # [C, V]
    F_c = cdf[channel_idx, :]  # [V]
    
    # Compute baselines
    tokens = torch.arange(V, dtype=torch.float32)
    
    # T1: Uniform equalization
    T1 = baseline_uniform(F_c.unsqueeze(0), target_range=(-1, 1)).squeeze(0)
    # Rescale to match E's range for fair comparison
    T1_rescaled = (T1 - T1.min()) / (T1.max() - T1.min()) * (E.max() - E.min()) + E.min()
    
    # T2: Probit (normal) equalization
    T2 = baseline_probit(F_c.unsqueeze(0)).squeeze(0)
    # Rescale to match E's range
    T2_rescaled = (T2 - T2.min()) / (T2.max() - T2.min()) * (E.max() - E.min()) + E.min()
    
    # T3: Isotonic regression
    T3 = baseline_isotonic(E.unsqueeze(0), tokens).squeeze(0)
    
    # Compute metrics
    metrics = {
        "T1_uniform": compute_baseline_metrics(E, T1_rescaled, "T1: Uniform"),
        "T2_probit": compute_baseline_metrics(E, T2_rescaled, "T2: Probit"),
        "T3_isotonic": compute_baseline_metrics(E, T3, "T3: Isotonic"),
    }
    
    return {
        "E": E,
        "T1": T1_rescaled,
        "T2": T2_rescaled,
        "T3": T3,
        "F_c": F_c,
        "metrics": metrics,
    }


# ============================================================================
# Probe C: Pushforward Distribution Tests
# ============================================================================

def sample_pushforward(
    lut_weight: torch.Tensor,
    data_histogram: torch.Tensor,
    channel_idx: int = 0,
    num_samples: int = 100000,
    scalar_mode: str = "uniform",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample tokens from data, map through LUT, return both.
    
    Args:
        lut_weight: [C, V, D] LUT weights
        data_histogram: [C, V] token counts
        channel_idx: Which channel to analyze
        num_samples: Number of samples to draw
    
    Returns:
        x_samples: [N] original token values (0..255)
        y_samples: [N] pushforward values E_c(x)
    """
    C, V, D = lut_weight.shape
    
    # Create sampling distribution from histogram
    hist = data_histogram[channel_idx, :].float()
    probs = hist / hist.sum()
    
    # Sample tokens according to data distribution
    x_samples = torch.multinomial(probs, num_samples, replacement=True)  # [N]
    
    # Map through LUT
    E_raw = lut_weight[channel_idx, :, :]  # [V, D]
    E = _scalarize_embedding(E_raw, mode=scalar_mode)
    
    y_samples = E[x_samples]  # [N]
    
    return x_samples.float(), y_samples


def test_uniformity(samples: torch.Tensor, target_range: Tuple[float, float] = (-1, 1)) -> Dict[str, float]:
    """Test if samples are uniform over target_range.
    
    Args:
        samples: [N] values
        target_range: (min, max) expected range
    
    Returns:
        results: Dict with KS/CvM/AD statistics and p-values
    """
    a, b = target_range
    
    # Standardize to [0, 1] for scipy's uniform test
    samples_01 = (samples.numpy() - a) / (b - a)
    samples_01 = np.clip(samples_01, 0, 1)
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.kstest(samples_01, 'uniform')
    
    # Cramér-von Mises test
    cvm_result = stats.cramervonmises(samples_01, 'uniform')
    cvm_stat, cvm_pval = cvm_result.statistic, cvm_result.pvalue
    
    # Anderson-Darling test (no p-value for uniform, only critical values)
    # Use KS distance as proxy
    uniform_cdf = np.linspace(0, 1, len(samples_01))
    ad_approx = np.max(np.abs(np.sort(samples_01) - uniform_cdf))
    
    return {
        "KS_stat": ks_stat,
        "KS_pval": ks_pval,
        "CvM_stat": cvm_stat,
        "CvM_pval": cvm_pval,
        "KS_distance": ad_approx,
    }


def test_normality(samples: torch.Tensor) -> Dict[str, float]:
    """Test if samples are normally distributed.
    
    Args:
        samples: [N] values
    
    Returns:
        results: Dict with Shapiro-Wilk, AD, KS statistics
    """
    samples_np = samples.numpy()
    
    # Standardize
    samples_std = (samples_np - samples_np.mean()) / (samples_np.std() + 1e-8)
    
    # Shapiro-Wilk test (up to 5000 samples for speed)
    if len(samples_std) > 5000:
        subsample = np.random.choice(samples_std, size=5000, replace=False)
    else:
        subsample = samples_std
    shapiro_stat, shapiro_pval = stats.shapiro(subsample)
    
    # KS test against normal
    ks_stat, ks_pval = stats.kstest(samples_std, 'norm')
    
    # Anderson-Darling test
    ad_result = stats.anderson(samples_std, dist='norm')
    ad_stat = ad_result.statistic
    
    # QQ plot metrics (linearity)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
    sample_quantiles = np.percentile(samples_std, np.linspace(1, 99, 100))
    qq_r2 = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
    
    return {
        "Shapiro_stat": shapiro_stat,
        "Shapiro_pval": shapiro_pval,
        "KS_stat": ks_stat,
        "KS_pval": ks_pval,
        "AD_stat": ad_stat,
        "QQ_R²": qq_r2,
    }


def analyze_pushforward(
    lut_weight: torch.Tensor,
    data_histogram: torch.Tensor,
    channel_idx: int = 0,
    num_samples: int = 100000,
    scalar_mode: str = "uniform",
) -> Dict[str, any]:
    """Analyze pushforward distribution y = E(x_1).
    
    Args:
        lut_weight: [C, V, D] LUT weights
        data_histogram: [C, V] token counts
        channel_idx: Which channel to analyze
        num_samples: Number of samples
    
    Returns:
        results: Dict with original/pushforward samples and test results
    """
    # Sample data and pushforward
    x_samples, y_samples = sample_pushforward(
        lut_weight, data_histogram, channel_idx, num_samples, scalar_mode=scalar_mode
    )
    
    # Test original distribution (x_1)
    x_uniform = test_uniformity(x_samples, target_range=(0, 255))
    x_normal = test_normality(x_samples)
    
    # Test pushforward distribution (y)
    y_min, y_max = y_samples.min().item(), y_samples.max().item()
    y_uniform = test_uniformity(y_samples, target_range=(y_min, y_max))
    y_normal = test_normality(y_samples)
    
    return {
        "x_samples": x_samples,
        "y_samples": y_samples,
        "original_uniform": x_uniform,
        "original_normal": x_normal,
        "pushforward_uniform": y_uniform,
        "pushforward_normal": y_normal,
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_baseline_comparison(results_A: Dict, output_path: str, channel_idx: int):
    """Plot learned E vs baselines T1/T2/T3."""
    E = results_A["E"].numpy()
    T1 = results_A["T1"].numpy()
    T2 = results_A["T2"].numpy()
    T3 = results_A["T3"].numpy()
    V = len(E)
    tokens = np.arange(V)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: All curves
    ax = axes[0, 0]
    ax.plot(tokens, E, 'k-', linewidth=2, label='E (Learned)', alpha=0.8)
    ax.plot(tokens, T1, 'r--', linewidth=1.5, label='T1: Uniform', alpha=0.7)
    ax.plot(tokens, T2, 'g--', linewidth=1.5, label='T2: Probit', alpha=0.7)
    ax.plot(tokens, T3, 'b--', linewidth=1.5, label='T3: Isotonic', alpha=0.7)
    ax.set_xlabel('Token Value')
    ax.set_ylabel('Embedding Value')
    ax.set_title(f'Channel {channel_idx}: Learned vs Baselines')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: Residuals
    ax = axes[0, 1]
    ax.plot(tokens, E - T1, 'r-', linewidth=1, label='E - T1', alpha=0.7)
    ax.plot(tokens, E - T2, 'g-', linewidth=1, label='E - T2', alpha=0.7)
    ax.plot(tokens, E - T3, 'b-', linewidth=1, label='E - T3', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Token Value')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals: E - T_i')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Scatter E vs T1
    ax = axes[1, 0]
    ax.scatter(T1, E, c=tokens, cmap='viridis', s=10, alpha=0.6)
    ax.plot([T1.min(), T1.max()], [T1.min(), T1.max()], 'k--', linewidth=1)
    metrics_T1 = results_A["metrics"]["T1_uniform"]
    ax.set_xlabel('T1: Uniform')
    ax.set_ylabel('E (Learned)')
    ax.set_title(f'E vs T1 (R²={metrics_T1["R²"]:.4f})')
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Scatter E vs T2
    ax = axes[1, 1]
    ax.scatter(T2, E, c=tokens, cmap='viridis', s=10, alpha=0.6)
    ax.plot([T2.min(), T2.max()], [T2.min(), T2.max()], 'k--', linewidth=1)
    metrics_T2 = results_A["metrics"]["T2_probit"]
    ax.set_xlabel('T2: Probit')
    ax.set_ylabel('E (Learned)')
    ax.set_title(f'E vs T2 (R²={metrics_T2["R²"]:.4f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved baseline comparison to {output_path}")


def plot_pushforward_analysis(results_C: Dict, output_path: str, channel_idx: int):
    """Plot original vs pushforward distributions."""
    x = results_C["x_samples"].numpy()
    y = results_C["y_samples"].numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Top row: Original distribution
    # Histogram
    ax = axes[0, 0]
    ax.hist(x, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Token Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Channel {channel_idx}: Original x_1 Distribution')
    ax.grid(True, alpha=0.3)
    
    # CDF
    ax = axes[0, 1]
    x_sorted = np.sort(x)
    cdf_x = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    ax.plot(x_sorted, cdf_x, 'b-', linewidth=1.5)
    ax.set_xlabel('Token Value')
    ax.set_ylabel('CDF')
    ax.set_title('Original CDF')
    ax.grid(True, alpha=0.3)
    
    # QQ plot (normal)
    ax = axes[0, 2]
    stats.probplot(x, dist="norm", plot=ax)
    ax.set_title(f'Original QQ (Normal)\nR²={results_C["original_normal"]["QQ_R²"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # Bottom row: Pushforward distribution
    # Histogram
    ax = axes[1, 0]
    ax.hist(y, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
    ax.set_xlabel('Pushforward Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Pushforward y=E(x_1) Distribution')
    ax.grid(True, alpha=0.3)
    
    # CDF vs Uniform
    ax = axes[1, 1]
    y_sorted = np.sort(y)
    cdf_y = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    y_normalized = (y_sorted - y_sorted.min()) / (y_sorted.max() - y_sorted.min())
    ax.plot(y_normalized, cdf_y, 'r-', linewidth=1.5, label='Pushforward')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Uniform')
    ks_dist = results_C["pushforward_uniform"]["KS_distance"]
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('CDF')
    ax.set_title(f'Pushforward CDF vs Uniform\nKS dist={ks_dist:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # QQ plot (normal)
    ax = axes[1, 2]
    stats.probplot(y, dist="norm", plot=ax)
    ax.set_title(f'Pushforward QQ (Normal)\nR²={results_C["pushforward_normal"]["QQ_R²"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved pushforward analysis to {output_path}")


def print_summary_report(results_A: Dict, results_C: Dict, channel_idx: int):
    """Print text summary of all metrics."""
    print("\n" + "="*80)
    print(f"ANALYSIS SUMMARY - Channel {channel_idx}")
    print("="*80)
    
    # Probe A: Baseline comparison
    print("\n[A] QUANTILE EQUALIZATION HYPOTHESIS")
    print("-" * 80)
    for baseline_name, metrics in results_A["metrics"].items():
        print(f"\n{metrics['name']}:")
        print(f"  R² = {metrics['R²']:.6f}")
        print(f"  RMSE = {metrics['RMSE']:.6f}")
        print(f"  MAE = {metrics['MAE']:.6f}")
        print(f"  Spearman ρ = {metrics['Spearman_ρ']:.6f}")
        print(f"  Kendall τ = {metrics['Kendall_τ']:.6f}")
    
    # Interpretation
    best_r2 = max(m["R²"] for m in results_A["metrics"].values())
    best_baseline = [k for k, m in results_A["metrics"].items() if m["R²"] == best_r2][0]
    print(f"\n✓ Best fit: {results_A['metrics'][best_baseline]['name']} (R²={best_r2:.6f})")
    
    if best_r2 >= 0.98:
        print("  → STRONG evidence for quantile-like behavior (R² ≥ 0.98)")
    elif best_r2 >= 0.95:
        print("  → MODERATE evidence for quantile-like behavior (R² ≥ 0.95)")
    else:
        print("  → WEAK evidence for quantile-like behavior (R² < 0.95)")
    
    # Probe C: Pushforward tests
    print("\n" + "-" * 80)
    print("[C] PUSHFORWARD DISTRIBUTION TESTS")
    print("-" * 80)
    
    print("\nOriginal Distribution (x_1):")
    print(f"  Uniformity: KS={results_C['original_uniform']['KS_stat']:.4f}, p={results_C['original_uniform']['KS_pval']:.4e}")
    print(f"  Normality: Shapiro W={results_C['original_normal']['Shapiro_stat']:.4f}, p={results_C['original_normal']['Shapiro_pval']:.4e}")
    
    print("\nPushforward Distribution (y=E(x_1)):")
    print(f"  Uniformity: KS={results_C['pushforward_uniform']['KS_stat']:.4f}, p={results_C['pushforward_uniform']['KS_pval']:.4e}")
    print(f"  Normality: Shapiro W={results_C['pushforward_normal']['Shapiro_stat']:.4f}, p={results_C['pushforward_normal']['Shapiro_pval']:.4e}")
    print(f"  QQ (Normal) R²={results_C['pushforward_normal']['QQ_R²']:.4f}")
    
    # Improvement metrics
    ks_improvement = (results_C['original_uniform']['KS_stat'] - results_C['pushforward_uniform']['KS_stat']) / results_C['original_uniform']['KS_stat']
    print(f"\n✓ KS distance improvement: {ks_improvement*100:.1f}%")
    
    if ks_improvement >= 0.5:
        print("  → STRONG pushforward effect (≥50% improvement)")
    elif ks_improvement >= 0.3:
        print("  → MODERATE pushforward effect (≥30% improvement)")
    else:
        print("  → WEAK pushforward effect (<30% improvement)")
    
    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    if best_r2 >= 0.98 and ks_improvement >= 0.5:
        print("✅ CONFIRMED: LUT behaves like quantile equalization")
        print("   The S-tail is a natural consequence of density reallocation.")
    elif best_r2 >= 0.95 or ks_improvement >= 0.3:
        print("⚠ LIKELY: LUT shows quantile-like behavior")
        print("   Some deviations exist, but overall trend is consistent.")
    else:
        print("❌ INCONCLUSIVE: LUT does not strongly match quantile hypothesis")
        print("   Consider other mechanisms (metric annealing, target geometry, etc.)")
    
    print("="*80 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LUT Quantile Hypothesis Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pth")
    parser.add_argument("--data_histogram", type=str, default=None, help="Path to data histogram tensor (optional)")
    parser.add_argument("--output_dir", type=str, default="./lut_quantile_analysis", help="Output directory")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to analyze (0=R, 1=G, 2=B)")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of samples for pushforward tests")
    parser.add_argument(
        "--scalar_mode",
        type=str,
        default="uniform",
        choices=["uniform", "mean", "norm", "norm_normalized", "pc1"],
        help="How to convert D-dim embeddings to scalar for analysis",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Load LUT and data
    print("\n[1/5] Loading checkpoint and data...")
    lut_weight, metadata = load_lut_from_checkpoint(args.checkpoint)
    data_histogram = estimate_data_histogram(
        args.data_histogram,
        vocab_size=metadata["vocab_size"],
        num_channels=metadata["num_channels"],
    )
    
    # Probe A: Baseline comparison
    print(f"\n[2/5] Running Probe A: Quantile Equalization Baselines (channel {args.channel}, scalar_mode={args.scalar_mode})...")
    results_A = analyze_baselines(lut_weight, data_histogram, channel_idx=args.channel, scalar_mode=args.scalar_mode)
    
    # Probe C: Pushforward tests
    print(f"\n[3/5] Running Probe C: Pushforward Distribution Tests (channel {args.channel}, scalar_mode={args.scalar_mode})...")
    results_C = analyze_pushforward(
        lut_weight,
        data_histogram,
        channel_idx=args.channel,
        num_samples=args.num_samples,
        scalar_mode=args.scalar_mode,
    )
    
    # Visualizations
    print("\n[4/5] Generating plots...")
    plot_baseline_comparison(results_A, output_dir / f"baseline_comparison_ch{args.channel}.png", args.channel)
    plot_pushforward_analysis(results_C, output_dir / f"pushforward_analysis_ch{args.channel}.png", args.channel)
    
    # Summary report
    print("\n[5/5] Generating summary report...")
    print_summary_report(results_A, results_C, args.channel)
    
    # Save numerical results
    results_file = output_dir / f"results_ch{args.channel}.txt"
    with open(results_file, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"LUT Quantile Analysis - Channel {args.channel}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"LUT shape: {list(lut_weight.shape)}\n")
        f.write(f"Embedding dimension: {metadata['emb_dim']}\n\n")
        
        f.write("[A] Baseline Metrics\n")
        f.write("-"*80 + "\n")
        for baseline_name, metrics in results_A["metrics"].items():
            f.write(f"\n{metrics['name']}:\n")
            for key, value in metrics.items():
                if key != "name":
                    f.write(f"  {key} = {value:.6f}\n")
        
        f.write("\n[C] Pushforward Tests\n")
        f.write("-"*80 + "\n")
        f.write("\nOriginal (x_1):\n")
        for key, value in results_C["original_uniform"].items():
            f.write(f"  {key} = {value:.6e}\n")
        f.write("\nPushforward (y):\n")
        for key, value in results_C["pushforward_uniform"].items():
            f.write(f"  {key} = {value:.6e}\n")
    
    print(f"✓ Saved results to {results_file}")
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
