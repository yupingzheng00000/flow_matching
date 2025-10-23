# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""Diagnostic metrics for learnable per-channel LUT in metric-induced path.

Implements the "lean-but-thorough" monitoring checklist:
  1) LUT geometry (range, scale, monotonicity, drift)
  2) Distance statistics (pairwise, β-scaled)
  3) Path quality (entropy, support)
  4) Sampling displacement (token space movement)
  5) Training dynamics (grad norms, update sizes)
  6) Cross-channel sanity (scale ratios, entropy gaps)
  7) Snapshots for plotting
  8) Alert thresholds

Design: minimal overhead, no regularization, just observation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from scipy.stats import spearmanr
import numpy as np

logger = logging.getLogger(__name__)


class LUTDiagnostics:
    """Diagnostic metrics for learnable scalar LUT.
    
    Tracks geometry, distances, path quality, and cross-channel behavior.
    Designed for 1-D per-channel LUTs with linear initialization.
    """

    def __init__(
        self,
        num_channels: int = 3,
        vocab_size: int = 256,
        probe_times: Optional[List[float]] = None,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        """Initialize diagnostics tracker.
        
        Args:
            num_channels: Number of channels (e.g., 3 for RGB)
            vocab_size: Vocabulary size (e.g., 256 for images)
            probe_times: Times to compute path metrics (default: [0.2, 0.5, 0.8])
            alert_thresholds: Custom alert thresholds (default: safe values)
        """
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.probe_times = probe_times or [0.2, 0.5, 0.8]
        
        # Alert thresholds (conservative defaults)
        self.alert_thresholds = alert_thresholds or {
            "entropy_collapse": 1.0,      # H_0.5 < 1.0
            "entropy_overflat": 5.3,      # H_0.5 > 5.3 (close to log(256))
            "scale_blowup": 3.0,          # std > 3x init
            "scale_collapse": 0.33,       # std < 1/3x init
            "inversion_ratio": 0.06,      # > 6% positions inverted
            "kl_drift": 0.5,              # KL > 0.5 at t=0.5
        }
        
        # Storage for baseline (linear init) reference
        self.baseline_embeddings: Optional[Tensor] = None
        self.init_embeddings: Optional[Tensor] = None
        self.init_std: Optional[Tensor] = None
        
        # Alert state tracking
        self.alert_counts: Dict[str, int] = {k: 0 for k in self.alert_thresholds}

    def register_baseline(self, embeddings: Tensor) -> None:
        """Register initial/baseline embeddings for comparison.
        
        Args:
            embeddings: [num_channels, vocab_size] tensor
        """
        self.baseline_embeddings = embeddings.detach().clone()
        self.init_embeddings = embeddings.detach().clone()
        self.init_std = embeddings.std(dim=1)  # [num_channels]
        logger.info(
            f"Registered baseline LUT: std_per_channel={self.init_std.cpu().numpy()}"
        )

    # =========================================================================
    # 1) LUT Geometry
    # =========================================================================

    def compute_lut_geometry(
        self, embeddings: Tensor, channel_names: Optional[List[str]] = None
    ) -> Dict[str, Tensor]:
        """Compute LUT geometry metrics per channel.
        
        Args:
            embeddings: [num_channels, vocab_size] current LUT weights
            channel_names: Optional names (e.g., ['R', 'G', 'B'])
        
        Returns:
            Dictionary of metrics (all tensors on same device as embeddings)
        """
        C, V = embeddings.shape
        if channel_names is None:
            channel_names = [f"ch{c}" for c in range(C)]
        
        metrics = {}
        
        # Range & endpoints
        metrics["min"] = embeddings.min(dim=1).values  # [C]
        metrics["max"] = embeddings.max(dim=1).values
        metrics["endpoint_0"] = embeddings[:, 0]       # [C]
        metrics["endpoint_255"] = embeddings[:, -1]
        
        # Scale
        metrics["std"] = embeddings.std(dim=1)         # [C]
        diffs = embeddings[:, 1:] - embeddings[:, :-1]  # [C, V-1]
        metrics["mean_abs_diff"] = diffs.abs().mean(dim=1)  # [C]
        
        # Monotonicity (inversion count)
        inversions = (diffs < 0).sum(dim=1).float()    # [C]
        metrics["inversion_count"] = inversions
        metrics["inversion_ratio"] = inversions / (V - 1)
        
        # Smoothness (total variation, curvature)
        metrics["total_variation"] = diffs.abs().sum(dim=1)  # [C]
        if V >= 3:
            second_diff = diffs[:, 1:] - diffs[:, :-1]   # [C, V-2]
            metrics["curvature"] = second_diff.abs().sum(dim=1)
        
        # Drift from init (if registered)
        if self.init_embeddings is not None:
            delta = embeddings - self.init_embeddings
            metrics["drift_from_init_l2"] = delta.norm(p=2, dim=1)  # [C]
        
        # Rank preservation (Spearman correlation)
        spearman_corrs = []
        for c in range(C):
            emb_c = embeddings[c].detach().cpu().numpy()
            vocab_idx = np.arange(V)
            rho, _ = spearmanr(vocab_idx, emb_c)
            spearman_corrs.append(rho)
        metrics["spearman_rho"] = torch.tensor(
            spearman_corrs, device=embeddings.device
        )
        
        # Saturation (duplicate count)
        duplicates = (diffs == 0).sum(dim=1).float()
        metrics["duplicate_count"] = duplicates
        
        # NaN/Inf checks
        metrics["has_nan"] = embeddings.isnan().any(dim=1)
        metrics["has_inf"] = embeddings.isinf().any(dim=1)
        
        return metrics

    # =========================================================================
    # 2) Distance Statistics
    # =========================================================================

    def compute_distance_stats(
        self,
        embeddings: Tensor,
        probe_batch: Tensor,
        beta_values: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute pairwise distance statistics for probe batch.
        
        Args:
            embeddings: [num_channels, vocab_size] LUT weights
            probe_batch: [B, C, H, W] probe images (int64, values in [0, vocab_size))
            beta_values: Optional [len(probe_times)] β(t) values to scale distances
        
        Returns:
            Dictionary with distance stats per channel and time
        """
        C, V = embeddings.shape
        B, C_img, H, W = probe_batch.shape
        assert C == C_img, f"Channel mismatch: {C} vs {C_img}"
        
        metrics = {}
        
        # Compute all pairwise distances: |E[v] - E[x_1]| for all v, x_1
        # [C, V, 1] - [C, 1, V] -> [C, V, V]
        emb_i = embeddings.unsqueeze(2)  # [C, V, 1]
        emb_j = embeddings.unsqueeze(1)  # [C, 1, V]
        all_dists = (emb_i - emb_j).abs()  # [C, V, V]
        
        # For each pixel in probe_batch, get distances to all other tokens
        probe_flat = probe_batch.view(B, C, -1)  # [B, C, HW]
        N_pixels = probe_flat.shape[-1]
        
        # Sample subset to avoid memory blow-up
        max_pixels = 1024
        if N_pixels > max_pixels:
            indices = torch.randperm(N_pixels, device=probe_flat.device)[:max_pixels]
            probe_flat = probe_flat[:, :, indices]
            N_pixels = max_pixels
        
        # Gather distances: for each (x_1), get distances to all vocab
        # probe_flat: [B, C, N] with values in [0, V)
        # all_dists: [C, V, V]
        # Want: [B, C, N, V]
        probe_expanded = probe_flat.unsqueeze(-1).expand(-1, -1, -1, V)  # [B, C, N, V]
        
        # Gather from all_dists using probe_flat as index
        # For channel c, pixel with value x_1, get all_dists[c, x_1, :]
        dists_per_pixel = []
        for c in range(C):
            # all_dists[c]: [V, V]
            # probe_flat[:, c, :]: [B, N]
            # Gather: [B, N, V]
            d_c = all_dists[c][probe_flat[:, c, :].long()]  # [B, N, V]
            dists_per_pixel.append(d_c)
        
        dists_per_pixel = torch.stack(dists_per_pixel, dim=1)  # [B, C, N, V]
        
        # Flatten to [B*C*N, V]
        dists_flat = dists_per_pixel.reshape(-1, V)  # [B*C*N, V]
        
        # Compute statistics per channel
        for c in range(C):
            start_idx = c * B * N_pixels
            end_idx = (c + 1) * B * N_pixels
            d_c = dists_flat[start_idx:end_idx, :]  # [B*N, V]
            
            metrics[f"ch{c}_dist_min"] = d_c.min()
            metrics[f"ch{c}_dist_mean"] = d_c.mean()
            metrics[f"ch{c}_dist_std"] = d_c.std()
            metrics[f"ch{c}_dist_max"] = d_c.max()
            
            # Quantiles (10%, 50%, 90%, 99%)
            quantiles = torch.quantile(d_c.flatten(), torch.tensor(
                [0.1, 0.5, 0.9, 0.99], device=d_c.device
            ))
            metrics[f"ch{c}_dist_q10"] = quantiles[0]
            metrics[f"ch{c}_dist_median"] = quantiles[1]
            metrics[f"ch{c}_dist_q90"] = quantiles[2]
            metrics[f"ch{c}_dist_q99"] = quantiles[3]
        
        # β-scaled distances (if provided)
        if beta_values is not None:
            for t_idx, beta_t in enumerate(beta_values):
                for c in range(C):
                    median_key = f"ch{c}_dist_median"
                    if median_key in metrics:
                        scaled = beta_t * metrics[median_key]
                        metrics[f"ch{c}_beta_scaled_median_t{t_idx}"] = scaled
        
        return metrics

    # =========================================================================
    # 3) Path Quality: Entropy & Support
    # =========================================================================

    def compute_path_quality(
        self,
        embeddings: Tensor,
        probe_batch: Tensor,
        beta_fn,  # Callable: t -> β(t)
    ) -> Dict[str, Tensor]:
        """Compute entropy and support size at probe times.
        
        Args:
            embeddings: [num_channels, vocab_size] LUT weights
            probe_batch: [B, C, H, W] probe images
            beta_fn: Function mapping t -> β(t) (scalar or [C] tensor)
        
        Returns:
            Dictionary with entropy/support metrics per channel and time
        """
        C, V = embeddings.shape
        B, C_img, H, W = probe_batch.shape
        assert C == C_img
        
        metrics = {}
        
        probe_flat = probe_batch.view(B, C, -1)  # [B, C, HW]
        N_pixels = probe_flat.shape[-1]
        
        # Sample subset
        max_pixels = 512
        if N_pixels > max_pixels:
            indices = torch.randperm(N_pixels, device=probe_flat.device)[:max_pixels]
            probe_flat = probe_flat[:, :, indices]
            N_pixels = max_pixels
        
        for t in self.probe_times:
            beta_t = beta_fn(torch.tensor(t, device=embeddings.device))
            
            # Compute Gibbs probabilities for each channel
            for c in range(C):
                # Get x_1 values for this channel: [B, N]
                x1_c = probe_flat[:, c, :].long()  # [B, N]
                
                # Embedding for x_1: [B, N]
                emb_x1 = embeddings[c, x1_c]  # [B, N]
                
                # All embeddings for channel c: [V]
                emb_all = embeddings[c, :]  # [V]
                
                # Distances: |E[v] - E[x_1]| for all v
                # emb_x1: [B, N] -> [B, N, 1]
                # emb_all: [V] -> [1, 1, V]
                # dists: [B, N, V]
                dists = (emb_x1.unsqueeze(-1) - emb_all.unsqueeze(0).unsqueeze(0)).abs()
                
                # Gibbs logits: -β * d
                if isinstance(beta_t, Tensor) and beta_t.numel() > 1:
                    beta_c = beta_t[c]
                else:
                    beta_c = beta_t
                
                logits = -beta_c * dists  # [B, N, V]
                probs = F.softmax(logits, dim=-1)  # [B, N, V]
                
                # Entropy: -sum(p log p)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [B, N]
                avg_entropy = entropy.mean()
                metrics[f"ch{c}_entropy_t{t:.1f}"] = avg_entropy
                
                # Effective support size (top-k for 90% mass)
                sorted_probs, _ = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)  # [B, N, V]
                support_sizes = (cumsum < 0.9).sum(dim=-1).float() + 1  # [B, N]
                avg_support = support_sizes.mean()
                metrics[f"ch{c}_support90_t{t:.1f}"] = avg_support
                
                # Top-1, top-3, top-5 mass
                metrics[f"ch{c}_top1_prob_t{t:.1f}"] = sorted_probs[:, :, 0].mean()
                metrics[f"ch{c}_top3_mass_t{t:.1f}"] = sorted_probs[:, :, :3].sum(dim=-1).mean()
                metrics[f"ch{c}_top5_mass_t{t:.1f}"] = sorted_probs[:, :, :5].sum(dim=-1).mean()
                
                # KL divergence vs baseline (if registered)
                if self.baseline_embeddings is not None:
                    emb_base = self.baseline_embeddings[c, :]
                    dists_base = (emb_x1.unsqueeze(-1) - emb_base.unsqueeze(0).unsqueeze(0)).abs()
                    logits_base = -beta_c * dists_base
                    probs_base = F.softmax(logits_base, dim=-1)
                    
                    # KL(p_lut || p_base)
                    kl = (probs * ((probs + 1e-10) / (probs_base + 1e-10)).log()).sum(dim=-1)
                    metrics[f"ch{c}_kl_vs_baseline_t{t:.1f}"] = kl.mean()
        
        return metrics

    # =========================================================================
    # 4) Sampling Displacement
    # =========================================================================

    def compute_sampling_displacement(
        self,
        embeddings: Tensor,
        probe_batch: Tensor,
        beta_fn,
        num_samples: int = 256,
    ) -> Dict[str, Tensor]:
        """Compute token displacement statistics from sampling.
        
        Args:
            embeddings: [num_channels, vocab_size] LUT weights
            probe_batch: [B, C, H, W] probe images
            beta_fn: Function mapping t -> β(t)
            num_samples: Number of pixels to sample per channel
        
        Returns:
            Dictionary with displacement metrics
        """
        C, V = embeddings.shape
        B, C_img, H, W = probe_batch.shape
        
        metrics = {}
        
        probe_flat = probe_batch.view(B, C, -1)  # [B, C, HW]
        N_pixels = probe_flat.shape[-1]
        
        # Sample pixels
        n_sample = min(num_samples, N_pixels)
        indices = torch.randperm(N_pixels, device=probe_flat.device)[:n_sample]
        sampled_pixels = probe_flat[:, :, indices]  # [B, C, n_sample]
        
        for t in self.probe_times:
            beta_t = beta_fn(torch.tensor(t, device=embeddings.device))
            
            for c in range(C):
                x1_c = sampled_pixels[:, c, :].long()  # [B, n_sample]
                
                # Compute Gibbs distribution and sample
                emb_x1 = embeddings[c, x1_c]  # [B, n_sample]
                emb_all = embeddings[c, :]  # [V]
                
                dists = (emb_x1.unsqueeze(-1) - emb_all.unsqueeze(0).unsqueeze(0)).abs()
                
                if isinstance(beta_t, Tensor) and beta_t.numel() > 1:
                    beta_c = beta_t[c]
                else:
                    beta_c = beta_t
                
                logits = -beta_c * dists  # [B, n_sample, V]
                probs = F.softmax(logits, dim=-1)
                
                # Sample x_t
                x_t = torch.multinomial(
                    probs.view(-1, V), num_samples=1
                ).view(B, n_sample)  # [B, n_sample]
                
                # Displacement: x_t - x_1
                displacement = (x_t - x1_c).float()  # [B, n_sample]
                
                metrics[f"ch{c}_mean_abs_disp_t{t:.1f}"] = displacement.abs().mean()
                metrics[f"ch{c}_mean_sq_disp_t{t:.1f}"] = (displacement ** 2).mean()
                
                # Signed symmetry: fraction moving up
                frac_up = (displacement > 0).float().mean()
                metrics[f"ch{c}_frac_move_up_t{t:.1f}"] = frac_up
        
        return metrics

    # =========================================================================
    # 5) Training Dynamics
    # =========================================================================

    def compute_training_dynamics(
        self,
        embeddings: Tensor,
        embeddings_prev: Optional[Tensor],
        grad: Optional[Tensor],
        lr: float,
    ) -> Dict[str, Tensor]:
        """Compute LUT training dynamics.
        
        Args:
            embeddings: [num_channels, vocab_size] current LUT
            embeddings_prev: Previous epoch's LUT (for delta computation)
            grad: [num_channels, vocab_size] gradient (if available)
            lr: Learning rate for LUT
        
        Returns:
            Dictionary with dynamics metrics
        """
        metrics = {}
        
        # Gradient norms (per channel)
        if grad is not None:
            grad_norms = grad.norm(p=2, dim=1)  # [C]
            for c in range(self.num_channels):
                metrics[f"ch{c}_grad_norm"] = grad_norms[c]
            metrics["grad_norm_avg"] = grad_norms.mean()
            metrics["grad_norm_max"] = grad_norms.max()
            
            # Step size estimate: lr * ||grad||
            step_sizes = lr * grad_norms
            metrics["step_size_avg"] = step_sizes.mean()
            metrics["step_size_max"] = step_sizes.max()
        
        # Actual parameter delta (if prev available)
        if embeddings_prev is not None:
            delta = embeddings - embeddings_prev  # [C, V]
            delta_norms = delta.norm(p=2, dim=1)  # [C]
            
            for c in range(self.num_channels):
                metrics[f"ch{c}_param_delta"] = delta_norms[c]
            metrics["param_delta_avg"] = delta_norms.mean()
            metrics["param_delta_max"] = delta_norms.max()
            
            # Max per-token update
            max_updates = delta.abs().max(dim=1).values  # [C]
            metrics["max_token_update_avg"] = max_updates.mean()
        
        return metrics

    # =========================================================================
    # 6) Cross-Channel Sanity
    # =========================================================================

    def compute_cross_channel_sanity(
        self,
        embeddings: Tensor,
        entropy_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Compute cross-channel consistency metrics.
        
        Args:
            embeddings: [num_channels, vocab_size] LUT weights
            entropy_dict: Optional dict from compute_path_quality (for gaps)
        
        Returns:
            Dictionary with cross-channel metrics
        """
        metrics = {}
        
        # Scale (std) per channel
        stds = embeddings.std(dim=1)  # [C]
        for c in range(self.num_channels):
            metrics[f"ch{c}_std"] = stds[c]
        
        # Scale ratios (if 3 channels, compute R:G:B ratios)
        if self.num_channels == 3:
            metrics["std_ratio_RG"] = stds[0] / (stds[1] + 1e-8)
            metrics["std_ratio_RB"] = stds[0] / (stds[2] + 1e-8)
            metrics["std_ratio_GB"] = stds[1] / (stds[2] + 1e-8)
        
        # Entropy gaps (if provided)
        if entropy_dict is not None and self.num_channels == 3:
            # Find entropy at t=0.5 (middle probe time)
            mid_t = self.probe_times[len(self.probe_times) // 2]
            h0 = entropy_dict.get(f"ch0_entropy_t{mid_t:.1f}")
            h1 = entropy_dict.get(f"ch1_entropy_t{mid_t:.1f}")
            h2 = entropy_dict.get(f"ch2_entropy_t{mid_t:.1f}")
            
            if h0 is not None and h1 is not None and h2 is not None:
                metrics["entropy_gap_RG"] = h0 - h1
                metrics["entropy_gap_RB"] = h0 - h2
                metrics["entropy_gap_GB"] = h1 - h2
        
        return metrics

    # =========================================================================
    # 8) Alert Thresholds
    # =========================================================================

    def check_alerts(
        self,
        geometry: Dict[str, Tensor],
        path_quality: Dict[str, Tensor],
    ) -> Dict[str, bool]:
        """Check if any alert thresholds are triggered.
        
        Args:
            geometry: Output from compute_lut_geometry
            path_quality: Output from compute_path_quality
        
        Returns:
            Dictionary of alert flags
        """
        alerts = {}
        
        # Entropy collapse/overflat (check at t=0.5)
        mid_t = self.probe_times[len(self.probe_times) // 2]
        for c in range(self.num_channels):
            entropy_key = f"ch{c}_entropy_t{mid_t:.1f}"
            if entropy_key in path_quality:
                H = path_quality[entropy_key].item()
                
                if H < self.alert_thresholds["entropy_collapse"]:
                    alerts[f"ch{c}_entropy_collapse"] = True
                    self.alert_counts["entropy_collapse"] += 1
                
                if H > self.alert_thresholds["entropy_overflat"]:
                    alerts[f"ch{c}_entropy_overflat"] = True
                    self.alert_counts["entropy_overflat"] += 1
        
        # Scale blow-up/collapse (if init_std registered)
        if self.init_std is not None and "std" in geometry:
            current_std = geometry["std"]
            for c in range(self.num_channels):
                ratio = current_std[c] / self.init_std[c]
                
                if ratio > self.alert_thresholds["scale_blowup"]:
                    alerts[f"ch{c}_scale_blowup"] = True
                    self.alert_counts["scale_blowup"] += 1
                
                if ratio < self.alert_thresholds["scale_collapse"]:
                    alerts[f"ch{c}_scale_collapse"] = True
                    self.alert_counts["scale_collapse"] += 1
        
        # Inversion ratio
        if "inversion_ratio" in geometry:
            inv_ratio = geometry["inversion_ratio"]
            for c in range(self.num_channels):
                if inv_ratio[c] > self.alert_thresholds["inversion_ratio"]:
                    alerts[f"ch{c}_inversion_spike"] = True
                    self.alert_counts["inversion_ratio"] += 1
        
        # KL drift (if baseline registered)
        if self.baseline_embeddings is not None:
            kl_key = f"ch0_kl_vs_baseline_t{mid_t:.1f}"
            if kl_key in path_quality:
                kl = path_quality[kl_key].item()
                if kl > self.alert_thresholds["kl_drift"]:
                    alerts["kl_drift"] = True
                    self.alert_counts["kl_drift"] += 1
        
        return alerts

    # =========================================================================
    # Convenience: Compute All Metrics
    # =========================================================================

    def compute_all_metrics(
        self,
        embeddings: Tensor,
        probe_batch: Tensor,
        beta_fn,
        embeddings_prev: Optional[Tensor] = None,
        grad: Optional[Tensor] = None,
        lr: Optional[float] = None,
        beta_values: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute all diagnostic metrics in one call.
        
        Args:
            embeddings: [num_channels, vocab_size] current LUT
            probe_batch: [B, C, H, W] probe images
            beta_fn: Callable mapping t -> β(t)
            embeddings_prev: Previous LUT (for dynamics)
            grad: Gradient tensor (for dynamics)
            lr: Learning rate (for dynamics)
            beta_values: Optional [len(probe_times)] β values for distance scaling
        
        Returns:
            Combined dictionary of all metrics
        """
        all_metrics = {}
        
        # 1) Geometry
        geometry = self.compute_lut_geometry(embeddings)
        all_metrics.update({f"geom/{k}": v for k, v in geometry.items()})
        
        # 2) Distance stats
        dist_stats = self.compute_distance_stats(embeddings, probe_batch, beta_values)
        all_metrics.update({f"dist/{k}": v for k, v in dist_stats.items()})
        
        # 3) Path quality
        path_quality = self.compute_path_quality(embeddings, probe_batch, beta_fn)
        all_metrics.update({f"path/{k}": v for k, v in path_quality.items()})
        
        # 4) Sampling displacement
        displacement = self.compute_sampling_displacement(embeddings, probe_batch, beta_fn)
        all_metrics.update({f"sample/{k}": v for k, v in displacement.items()})
        
        # 5) Training dynamics (if data available)
        if embeddings_prev is not None or grad is not None:
            lr_val = lr if lr is not None else 0.0
            dynamics = self.compute_training_dynamics(
                embeddings, embeddings_prev, grad, lr_val
            )
            all_metrics.update({f"dynamics/{k}": v for k, v in dynamics.items()})
        
        # 6) Cross-channel sanity
        cross_channel = self.compute_cross_channel_sanity(embeddings, path_quality)
        all_metrics.update({f"cross/{k}": v for k, v in cross_channel.items()})
        
        # 8) Alerts
        alerts = self.check_alerts(geometry, path_quality)
        all_metrics.update({f"alert/{k}": v for k, v in alerts.items()})
        
        return all_metrics

    def log_metrics_summary(
        self, metrics: Dict[str, Tensor], epoch: int, logger_obj=None
    ) -> None:
        """Log a human-readable summary of key metrics.
        
        Args:
            metrics: Dictionary from compute_all_metrics
            epoch: Current epoch number
            logger_obj: Logger to use (defaults to module logger)
        """
        log = logger_obj or logger
        
        # Header
        log.info(f"=== LUT Diagnostics Epoch {epoch} ===")
        
        # Geometry summary (per channel)
        log.info("Geometry:")
        for c in range(self.num_channels):
            if f"geom/std" in metrics:
                std = metrics[f"geom/std"][c].item()
                inv_count = metrics.get(f"geom/inversion_count", torch.zeros(self.num_channels))[c].item()
                spearman = metrics.get(f"geom/spearman_rho", torch.zeros(self.num_channels))[c].item()
                log.info(f"  ch{c}: std={std:.4f}, inversions={inv_count:.0f}, ρ={spearman:.3f}")
        
        # Path quality at t=0.5
        mid_t = self.probe_times[len(self.probe_times) // 2]
        log.info(f"Path Quality (t={mid_t}):")
        for c in range(self.num_channels):
            H_key = f"path/ch{c}_entropy_t{mid_t:.1f}"
            sup_key = f"path/ch{c}_support90_t{mid_t:.1f}"
            if H_key in metrics:
                H = metrics[H_key].item()
                sup = metrics.get(sup_key, torch.tensor(0.0)).item()
                log.info(f"  ch{c}: H={H:.3f}, support90={sup:.1f}")
        
        # Alerts
        alert_triggered = any("alert/" in k for k in metrics.keys())
        if alert_triggered:
            log.warning("⚠️  Alerts triggered:")
            for k, v in metrics.items():
                if k.startswith("alert/") and v:
                    log.warning(f"  {k}")
        else:
            log.info("✓ No alerts")
        
        log.info("=" * 50)
