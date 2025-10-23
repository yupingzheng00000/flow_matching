"""
MDS-based initialization for learnable Mahalanobis metric.

This module provides initialization methods that warm-start the learnable metric
from the baseline Lp metric using Multi-Dimensional Scaling (MDS).
"""

import torch
from torch import Tensor
from typing import Optional


def classical_mds(
    distance_matrix: Tensor,
    metric_dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Classical Multi-Dimensional Scaling (Torgerson 1952).
    
    Given a distance matrix D, find coordinates Z such that ||z_i - z_j||_2 ≈ D_ij.
    Uses eigendecomposition of the double-centered Gram matrix.
    
    Args:
        distance_matrix: [K, K] pairwise distance matrix
        metric_dim: Target embedding dimension
        device: Device for computation
        dtype: Data type for computation
    
    Returns:
        codes: [K, metric_dim] embedded coordinates
    
    Algorithm:
        1. Double-centering: B = -1/2 * H * D² * H  where H = I - 11ᵀ/K
        2. Eigendecomposition: B = V Λ Vᵀ
        3. Extract top-d components: Z = V[:,:d] * sqrt(Λ[:d,:d])
    """
    K = distance_matrix.shape[0]
    if device is None:
        device = distance_matrix.device
    if dtype is None:
        dtype = distance_matrix.dtype
    
    # Double-centering to get Gram matrix
    D_sq = distance_matrix ** 2
    row_means = D_sq.mean(dim=1, keepdim=True)
    col_means = D_sq.mean(dim=0, keepdim=True)
    grand_mean = D_sq.mean()
    
    B = -0.5 * (D_sq - row_means - col_means + grand_mean)
    
    # Eigendecomposition (ascending order by default)
    eigenvalues, eigenvectors = torch.linalg.eigh(B)
    
    # Take top-d eigenvalues/vectors (flip to descending order)
    eigenvalues = eigenvalues.flip(0)[:metric_dim]
    eigenvectors = eigenvectors.flip(1)[:, :metric_dim]
    
    # Clamp negative eigenvalues to zero (numerical artifacts or non-Euclidean)
    eigenvalues = eigenvalues.clamp(min=0.0)
    
    # Construct embedding: Z = V * sqrt(Λ)
    codes = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))
    
    return codes.to(device=device, dtype=dtype)


def classical_mds_full_spectrum(
    distance_matrix: Tensor,
    metric_dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Classical MDS using ALL eigenvalues (including negatives).
    
    This is equivalent to standard classical MDS but we compute it by:
    1. Using all eigenvalues (not clamping negatives)
    2. Taking top-d by absolute value
    3. Preserving sign when taking sqrt
    
    For non-Euclidean distances, this can sometimes give better stress
    by allowing "imaginary" dimensions.
    
    Args:
        distance_matrix: [K, K] pairwise distance matrix
        metric_dim: Target embedding dimension
        device: Device for computation
        dtype: Data type for computation
    
    Returns:
        codes: [K, metric_dim] embedded coordinates
    """
    K = distance_matrix.shape[0]
    if device is None:
        device = distance_matrix.device
    if dtype is None:
        dtype = distance_matrix.dtype
    
    # Double-centering
    D_sq = distance_matrix ** 2
    row_means = D_sq.mean(dim=1, keepdim=True)
    col_means = D_sq.mean(dim=0, keepdim=True)
    grand_mean = D_sq.mean()
    
    B = -0.5 * (D_sq - row_means - col_means + grand_mean)
    
    # Full eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(B)
    
    # Sort by absolute value (descending)
    abs_eigenvalues = eigenvalues.abs()
    sorted_indices = abs_eigenvalues.argsort(descending=True)[:metric_dim]
    
    selected_eigenvalues = eigenvalues[sorted_indices]
    selected_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Use sign-preserving sqrt: sqrt(|λ|) * sign(λ)
    sqrt_eigenvalues = torch.sqrt(selected_eigenvalues.abs()) * selected_eigenvalues.sign()
    
    # Construct embedding
    codes = selected_eigenvectors @ torch.diag(sqrt_eigenvalues)
    
    return codes.to(device=device, dtype=dtype)


def initialize_metric_from_lp(
    vocab_size: int,
    metric_dim: int,
    lp_order: float = 3.0,
    embed_range: str = "pm1",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Initialize metric codes using MDS on the baseline Lp distance.
    
    Args:
        vocab_size: Number of discrete tokens (e.g., 256 for CIFAR-10 pixels)
        metric_dim: Dimension of the learnable embedding
        lp_order: Lp norm order (e.g., 3.0 for L3)
        embed_range: "pm1" for [-1,1] or "unit" for [0,1]
        device: Target device
        dtype: Target dtype
    
    Returns:
        init_codes: [vocab_size, metric_dim] initialized codes
    
    Process:
        1. Build 1D baseline embedding (linspace)
        2. Compute Lp distance matrix
        3. Run MDS to get d-dimensional embedding
        4. Result: codes that preserve Lp geometry
    """
    # Step 1: Build baseline 1D embedding
    if embed_range == "pm1":
        baseline = torch.linspace(-1.0, 1.0, steps=vocab_size, device=device, dtype=dtype)
    else:  # "unit"
        baseline = torch.linspace(0.0, 1.0, steps=vocab_size, device=device, dtype=dtype)
    
    baseline = baseline.unsqueeze(1)  # [K, 1]
    
    # Step 2: Compute Lp distance matrix
    # For 1D: d(i,j) = |baseline[i] - baseline[j]|^p
    pairwise_diff = baseline - baseline.T  # [K, K]
    lp_distances = torch.abs(pairwise_diff) ** lp_order  # [K, K]
    
    # Step 3: MDS embedding
    init_codes = classical_mds(lp_distances, metric_dim)  # [K, metric_dim]
    
    return init_codes.to(device=device, dtype=dtype)


def verify_mds_quality(
    init_codes: Tensor,
    target_distances: Tensor,
    metric_dim: int,
) -> dict:
    """
    Verify the quality of MDS initialization using standard metrics.
    
    Args:
        init_codes: [K, d] MDS embedding
        target_distances: [K, K] target distance matrix
        metric_dim: Dimension used
    
    Returns:
        metrics: Dictionary with quality metrics
            - reconstruction_error: ||D_target - D_mds||_F / ||D_target||_F
            - spearman_correlation: Rank correlation between distances
            - num_negative_eigenvalues: Count of negative eigenvalues (non-Euclidean indicator)
            - gof_eigenvalue: Goodness-of-fit via eigenvalues (Σλ⁺ / Σ|λ|)
            - gof_stress1: Kruskal's stress-1 metric
            - explained_variance_positive: Variance explained by positive eigenvalues only
    """
    K = init_codes.shape[0]
    device = init_codes.device
    
    # Compute MDS distances
    mds_distances = torch.cdist(init_codes, init_codes, p=2.0)  # [K, K]
    
    # 1. Frobenius reconstruction error
    diff = target_distances - mds_distances
    reconstruction_error = torch.linalg.norm(diff, ord='fro') / torch.linalg.norm(target_distances, ord='fro')
    
    # 2. Spearman correlation (on flattened upper triangle)
    mask = torch.triu(torch.ones(K, K, dtype=torch.bool, device=device), diagonal=1)
    target_flat = target_distances[mask]
    mds_flat = mds_distances[mask]
    
    target_ranks = target_flat.argsort().argsort().float()
    mds_ranks = mds_flat.argsort().argsort().float()
    spearman_corr = torch.corrcoef(torch.stack([target_ranks, mds_ranks]))[0, 1]
    
    # 3. Double-centered Gram matrix and eigenvalues
    D_sq = target_distances ** 2
    row_means = D_sq.mean(dim=1, keepdim=True)
    col_means = D_sq.mean(dim=0, keepdim=True)
    grand_mean = D_sq.mean()
    B = -0.5 * (D_sq - row_means - col_means + grand_mean)
    
    # Full eigendecomposition (don't clamp yet - we need to count negatives)
    eigenvalues_raw = torch.linalg.eigvalsh(B)  # Ascending order
    eigenvalues_sorted = eigenvalues_raw.flip(0)  # Descending order
    
    # 4. Count negative eigenvalues (indicator of non-Euclidean geometry)
    num_negative = (eigenvalues_sorted < -1e-10).sum().item()
    
    # 5. GOF (Goodness-of-Fit) - Method 1: Eigenvalue-based
    # Uses absolute sum so negatives don't vanish
    # GOF = (Σ λ⁺ for top d) / (Σ |λ| for all)
    positive_eigenvalues = eigenvalues_sorted.clamp(min=0.0)
    abs_eigenvalues = eigenvalues_sorted.abs()
    
    top_d_positive = positive_eigenvalues[:metric_dim].sum()
    total_abs = abs_eigenvalues.sum()
    gof_eigenvalue = top_d_positive / (total_abs + 1e-10)
    
    # 6. GOF (Goodness-of-Fit) - Method 2: Kruskal's Stress-1
    # Stress-1 = sqrt(Σ(D_target - D_mds)² / Σ D_target²)
    # Lower is better; 0 = perfect fit
    numerator = ((target_distances - mds_distances) ** 2).sum()
    denominator = (target_distances ** 2).sum()
    stress1 = torch.sqrt(numerator / (denominator + 1e-10))
    
    # 7. Explained variance (using only positive eigenvalues)
    total_positive_var = positive_eigenvalues.sum()
    top_d_var = positive_eigenvalues[:metric_dim].sum()
    explained_variance_positive = top_d_var / (total_positive_var + 1e-10)
    
    return {
        "reconstruction_error": reconstruction_error.item(),
        "spearman_correlation": spearman_corr.item(),
        "num_negative_eigenvalues": num_negative,
        "gof_eigenvalue": gof_eigenvalue.item(),
        "gof_stress1": stress1.item(),
        "explained_variance_positive": explained_variance_positive.item(),
    }


if __name__ == "__main__":
    """Test MDS initialization with comparison between classical and full-spectrum methods."""
    print("="*70)
    print("MDS INITIALIZATION COMPARISON (K=256, d=8)")
    print("="*70)
    
    # Test parameters
    K, d = 256, 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Compute target L₃ distances
    baseline = torch.linspace(-1.0, 1.0, steps=K, device=device).unsqueeze(1)
    lp_distances = torch.abs(baseline - baseline.T) ** 3.0
    
    print(f"\nTarget: L₃ distance matrix from linearly spaced {K} tokens")
    print(f"Device: {device}")
    
    # Method 1: Classical MDS (positive eigenvalues only)
    print("\n" + "─"*70)
    print("METHOD 1: Classical MDS (positive eigenvalues only)")
    print("─"*70)
    
    init_codes_classical = classical_mds(lp_distances, d, device, torch.float32)
    
    print(f"Shape: {init_codes_classical.shape}")
    print(f"Mean:  {init_codes_classical.mean():.4f}")
    print(f"Std:   {init_codes_classical.std():.4f}")
    print(f"Rank:  {torch.linalg.matrix_rank(init_codes_classical).item()}")
    
    metrics_classical = verify_mds_quality(init_codes_classical, lp_distances, d)
    
    print(f"\nQuality Metrics:")
    print(f"  Reconstruction error:           {metrics_classical['reconstruction_error']:.4f}")
    print(f"  Spearman correlation:           {metrics_classical['spearman_correlation']:.4f}")
    print(f"  Num negative eigenvalues:       {metrics_classical['num_negative_eigenvalues']}")
    print(f"  GOF (eigenvalue-based):         {metrics_classical['gof_eigenvalue']:.4f}")
    print(f"  Kruskal's Stress-1:             {metrics_classical['gof_stress1']:.4f}")
    print(f"  Explained variance (positive):  {metrics_classical['explained_variance_positive']:.4f}")
    
    # Method 2: Full-spectrum MDS (including negative eigenvalues by |λ|)
    print("\n" + "─"*70)
    print("METHOD 2: Full-Spectrum MDS (top-d by |λ|, preserving sign)")
    print("─"*70)
    
    init_codes_full = classical_mds_full_spectrum(lp_distances, d, device, torch.float32)
    
    print(f"Shape: {init_codes_full.shape}")
    print(f"Mean:  {init_codes_full.mean():.4f}")
    print(f"Std:   {init_codes_full.std():.4f}")
    print(f"Rank:  {torch.linalg.matrix_rank(init_codes_full).item()}")
    
    metrics_full = verify_mds_quality(init_codes_full, lp_distances, d)
    
    print(f"\nQuality Metrics:")
    print(f"  Reconstruction error:           {metrics_full['reconstruction_error']:.4f}")
    print(f"  Spearman correlation:           {metrics_full['spearman_correlation']:.4f}")
    print(f"  Num negative eigenvalues:       {metrics_full['num_negative_eigenvalues']}")
    print(f"  GOF (eigenvalue-based):         {metrics_full['gof_eigenvalue']:.4f}")
    print(f"  Kruskal's Stress-1:             {metrics_full['gof_stress1']:.4f}")
    print(f"  Explained variance (positive):  {metrics_full['explained_variance_positive']:.4f}")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    stress_diff = metrics_classical['gof_stress1'] - metrics_full['gof_stress1']
    recon_diff = metrics_classical['reconstruction_error'] - metrics_full['reconstruction_error']
    gof_diff = metrics_full['gof_eigenvalue'] - metrics_classical['gof_eigenvalue']
    
    print(f"\nStress-1 difference (Classical - Full): {stress_diff:+.4f}")
    if abs(stress_diff) < 1e-6:
        print("  → Identical fit quality")
    elif stress_diff > 0:
        print(f"  → Full-spectrum is BETTER by {abs(stress_diff):.4f}")
    else:
        print(f"  → Classical is BETTER by {abs(stress_diff):.4f}")
    
    print(f"\nReconstruction error difference (Classical - Full): {recon_diff:+.4f}")
    if abs(recon_diff) < 1e-6:
        print("  → Identical reconstruction")
    elif recon_diff > 0:
        print(f"  → Full-spectrum is BETTER by {abs(recon_diff):.4f}")
    else:
        print(f"  → Classical is BETTER by {abs(recon_diff):.4f}")
    
    print(f"\nGOF difference (Full - Classical): {gof_diff:+.4f}")
    if abs(gof_diff) < 1e-6:
        print("  → Identical eigenvalue coverage")
    elif gof_diff > 0:
        print(f"  → Full-spectrum captures {abs(gof_diff)*100:.2f}% more variance")
    else:
        print(f"  → Classical captures {abs(gof_diff)*100:.2f}% more variance")
    
    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    if metrics_full['gof_stress1'] < metrics_classical['gof_stress1'] - 0.01:
        print("✅ Use FULL-SPECTRUM MDS (significantly better stress)")
        recommended_method = "classical_mds_full_spectrum"
    elif metrics_full['gof_stress1'] > metrics_classical['gof_stress1'] + 0.01:
        print("✅ Use CLASSICAL MDS (better stress with positive eigenvalues only)")
        recommended_method = "classical_mds"
    else:
        print("✅ Use CLASSICAL MDS (standard method, equivalent quality)")
        print("   (Full-spectrum gains negligible improvement for this case)")
        recommended_method = "classical_mds"
    
    print(f"\nRecommended function: {recommended_method}()")
    print(f"Recommended dimension: d={d}")
    print("\n" + "="*70)
