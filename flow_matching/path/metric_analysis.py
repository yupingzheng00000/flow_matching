"""
Metric Analysis Utilities

Provides comprehensive analysis of learned Mahalanobis metrics during training.
Automatically generates visualizations (t-SNE/UMAP), statistics, and diagnostics.

Author: Based on Linus's philosophy of "glass box engineering"
Date: October 2025
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import warnings

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False
    warnings.warn("scikit-learn not installed, t-SNE visualization will be unavailable")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def analyze_metric(
    metric: "MahalanobisTokenMetric",
    output_dir: str,
    device: str = "cuda",
    use_umap: bool = True,
    use_tsne: bool = True,
    verbose: bool = True,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive analysis of a learned Mahalanobis metric.
    
    Args:
        metric: The MahalanobisTokenMetric instance to analyze
        output_dir: Directory to save analysis outputs
        device: Device for computation
        use_umap: Whether to generate UMAP visualization (if available)
        use_tsne: Whether to generate t-SNE visualization (if available)
        verbose: Whether to print analysis results
        save_plots: Whether to save visualization plots
        
    Returns:
        Dictionary containing all computed statistics and metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if verbose:
        print("\n" + "=" * 80)
        print("METRIC ANALYSIS")
        print("=" * 80)
    
    # Move metric to device and set to eval mode
    metric = metric.to(device)
    metric.eval()
    
    with torch.no_grad():
        # ========================================================================
        # 1. Basic Statistics
        # ========================================================================
        
        codes = metric.codes  # [vocab_size, metric_dim]
        lower = metric.cholesky_factor()  # [metric_dim, metric_dim]
        Z = codes @ lower.T  # [vocab_size, metric_dim]
        
        # Frobenius norms (raw and normalized)
        fro_norm_raw = torch.linalg.norm(Z, ord='fro').item()
        results['frobenius_norm_raw'] = fro_norm_raw
        
        # Normalized version (as used in distance computation)
        Z_normalized = Z / (fro_norm_raw + 1e-6)
        fro_norm_normalized = torch.linalg.norm(Z_normalized, ord='fro').item()
        results['frobenius_norm_normalized'] = fro_norm_normalized
        
        if verbose:
            print(f"\nFrobenius norm (raw):        {fro_norm_raw:.6f}")
            print(f"Frobenius norm (normalized): {fro_norm_normalized:.6f}  [constrained to 1.0]")
        
        # codes_raw norm (if available)
        if hasattr(metric, 'codes_raw'):
            codes_raw_norm = torch.linalg.norm(metric.codes_raw, ord='fro').item()
            results['codes_raw_norm'] = codes_raw_norm
            if verbose:
                print(f"codes_raw norm: {codes_raw_norm:.6f}")
        
        # Pairwise distances (using normalized Z, consistent with training)
        distances = torch.cdist(Z_normalized, Z_normalized, p=2.0)  # [vocab_size, vocab_size]
        
        # Distance statistics (exclude diagonal)
        vocab_size = Z.shape[0]
        mask = ~torch.eye(vocab_size, dtype=torch.bool, device=device)
        dist_values = distances[mask]
        
        results['distance_mean'] = dist_values.mean().item()
        results['distance_std'] = dist_values.std().item()
        results['distance_min'] = dist_values.min().item()
        results['distance_max'] = dist_values.max().item()
        results['distance_median'] = dist_values.median().item()
        
        if verbose:
            print(f"\nPairwise Distance Statistics:")
            print(f"  Mean:   {results['distance_mean']:.6f}")
            print(f"  Std:    {results['distance_std']:.6f}")
            print(f"  Min:    {results['distance_min']:.6f}")
            print(f"  Max:    {results['distance_max']:.6f}")
            print(f"  Median: {results['distance_median']:.6f}")
            
            # Coefficient of variation
            cv = results['distance_std'] / results['distance_mean']
            results['coefficient_of_variation'] = cv
            print(f"  CV:     {cv:.6f}")
            
            if results['distance_std'] > 0.03:
                print(f"\n✅ Meaningful structure learned (std={results['distance_std']:.4f})")
            else:
                print(f"\n⚠️  Weak structure (std={results['distance_std']:.4f})")
        
        # ========================================================================
        # 2. Nearest Neighbors Analysis
        # ========================================================================
        
        if verbose:
            print(f"\nNearest Neighbors (sample tokens):")
        
        # For grayscale tokens (0-255), check key anchors
        if vocab_size == 256:
            anchor_tokens = [(0, "Black"), (128, "Mid-Gray"), (255, "White")]
        else:
            # Generic anchors for other vocab sizes
            anchor_tokens = [
                (0, "Token-0"),
                (vocab_size // 2, f"Token-{vocab_size//2}"),
                (vocab_size - 1, f"Token-{vocab_size-1}")
            ]
        
        neighbors_info = {}
        for token_id, description in anchor_tokens:
            if token_id >= vocab_size:
                continue
                
            dists_from_anchor = distances[token_id]
            nearest_indices = torch.argsort(dists_from_anchor)[:6]
            nearest_dists = dists_from_anchor[nearest_indices]
            
            neighbors_info[token_id] = {
                'description': description,
                'neighbors': nearest_indices.cpu().tolist(),
                'distances': nearest_dists.cpu().tolist()
            }
            
            if verbose:
                print(f"  {description} ({token_id}): " + 
                      f"neighbors={nearest_indices[1:4].cpu().tolist()}")
        
        results['nearest_neighbors'] = neighbors_info
        
        # ========================================================================
        # 3. Visualizations (if requested)
        # ========================================================================
        
        if save_plots:
            Z_np = Z.cpu().numpy()
            
            # 3.1 Distance Heatmap
            if verbose:
                print(f"\nGenerating distance heatmap...")
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                distances.cpu().numpy(),
                cmap='viridis',
                square=True,
                xticklabels=False,
                yticklabels=False,
                cbar_kws={'label': 'Distance'}
            )
            plt.title(f'Pairwise Token Distances ({vocab_size}×{vocab_size})')
            heatmap_path = output_path / "distance_heatmap.png"
            plt.savefig(heatmap_path, dpi=120, bbox_inches='tight')
            plt.close()
            results['heatmap_path'] = str(heatmap_path)
            
            # 3.2 Distance Distribution
            if verbose:
                print(f"Generating distance distribution...")
            
            plt.figure(figsize=(10, 6))
            plt.hist(dist_values.cpu().numpy(), bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(results['distance_mean'], color='red', linestyle='--', 
                       linewidth=2, label=f"Mean: {results['distance_mean']:.3f}")
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
            plt.title('Distribution of Pairwise Distances')
            plt.legend()
            plt.grid(alpha=0.3)
            dist_hist_path = output_path / "distance_distribution.png"
            plt.savefig(dist_hist_path, dpi=120, bbox_inches='tight')
            plt.close()
            results['distribution_path'] = str(dist_hist_path)
            
            # 3.3 Dimensionality Reduction Visualizations
            viz_methods = []
            if use_umap and HAS_UMAP:
                viz_methods.append(('UMAP', _compute_umap))
            if use_tsne and HAS_TSNE:
                viz_methods.append(('t-SNE', _compute_tsne))
            
            if not viz_methods:
                if verbose:
                    print("\n⚠️  No dimensionality reduction available")
                    print("   Install: pip install umap-learn scikit-learn")
            
            for method_name, compute_fn in viz_methods:
                if verbose:
                    print(f"Running {method_name}...")
                
                try:
                    Z_2d = compute_fn(Z_np)
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
                    
                    # Plot 1: Simple scatter
                    axes[0].scatter(Z_2d[:, 0], Z_2d[:, 1], s=20, alpha=0.6, c='blue')
                    axes[0].set_title(f'{method_name}: Learned Metric Space → 2D')
                    axes[0].set_xlabel(f'{method_name} Dim 1')
                    axes[0].set_ylabel(f'{method_name} Dim 2')
                    axes[0].grid(alpha=0.3)
                    
                    # Plot 2: Colored by token value (for grayscale tokens)
                    if vocab_size == 256:
                        grayscale_colors = np.array([[i/255, i/255, i/255] for i in range(256)])
                        axes[1].scatter(Z_2d[:, 0], Z_2d[:, 1], s=20, alpha=0.8, 
                                       c=grayscale_colors)
                        axes[1].set_title(f'{method_name}: Colors=Token Values (0=Black→255=White)')
                    else:
                        # Use viridis colormap for non-grayscale
                        axes[1].scatter(Z_2d[:, 0], Z_2d[:, 1], s=20, alpha=0.8,
                                       c=np.arange(vocab_size), cmap='viridis')
                        axes[1].set_title(f'{method_name}: Colored by Token ID')
                    
                    axes[1].set_xlabel(f'{method_name} Dim 1')
                    axes[1].set_ylabel(f'{method_name} Dim 2')
                    axes[1].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    
                    viz_path = output_path / f"{method_name.lower().replace('-', '')}_visualization.png"
                    plt.savefig(viz_path, dpi=120, bbox_inches='tight')
                    plt.close()
                    
                    results[f'{method_name.lower()}_path'] = str(viz_path)
                    
                    if verbose:
                        print(f"  ✓ Saved: {viz_path.name}")
                
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Failed: {e}")
            
            # 3.4 Cholesky Factor Heatmap
            if verbose:
                print(f"Generating Cholesky factor heatmap...")
            
            L = lower.cpu().numpy()
            plt.figure(figsize=(8, 6))
            sns.heatmap(L, annot=False, cmap='coolwarm', center=0, square=True,
                       cbar_kws={'label': 'Value'})
            plt.title(f'Cholesky Factor L ({L.shape[0]}×{L.shape[1]})')
            cholesky_path = output_path / "cholesky_factor.png"
            plt.savefig(cholesky_path, dpi=120, bbox_inches='tight')
            plt.close()
            results['cholesky_path'] = str(cholesky_path)
            
            # Diagonal values
            diag_values = np.diag(L)
            results['cholesky_diag_min'] = float(diag_values.min())
            results['cholesky_diag_max'] = float(diag_values.max())
    
    if verbose:
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        if save_plots:
            print(f"\nOutputs saved to: {output_dir}")
    
    return results


def _compute_tsne(Z: np.ndarray) -> np.ndarray:
    """Compute t-SNE dimensionality reduction."""
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, Z.shape[0] // 4),
        learning_rate='auto',
        init='pca',
        random_state=42,
        max_iter=1000
    )
    return tsne.fit_transform(Z)


def _compute_umap(Z: np.ndarray) -> np.ndarray:
    """Compute UMAP dimensionality reduction."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, Z.shape[0] // 4),
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
        n_jobs=1
    )
    return reducer.fit_transform(Z)


def quick_metric_check(
    metric: "MahalanobisTokenMetric",
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Quick check of metric health without saving plots.
    Returns key statistics only.
    """
    metric = metric.to(device)
    metric.eval()
    
    with torch.no_grad():
        codes = metric.codes
        lower = metric.cholesky_factor()
        Z = codes @ lower.T
        
        distances = torch.cdist(Z, Z, p=2.0)
        vocab_size = Z.shape[0]
        mask = ~torch.eye(vocab_size, dtype=torch.bool, device=device)
        dist_values = distances[mask]
        
        fro_norm_raw = torch.linalg.norm(Z, ord='fro').item()
        
        return {
            'distance_mean': dist_values.mean().item(),
            'distance_std': dist_values.std().item(),
            'distance_min': dist_values.min().item(),
            'distance_max': dist_values.max().item(),
            'frobenius_norm_raw': fro_norm_raw,
        }
