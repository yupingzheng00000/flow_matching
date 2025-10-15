#!/usr/bin/env python3
"""
Test: Verify LUT initialization with noise preserves monotonicity

Checks:
1. All channels are monotonically increasing (zero or very few inversions)
2. Channels are decorrelated (correlation < 1.0)
3. Noise magnitude is small (std ≈ sqrt(init_std^2 + noise_std^2))
"""
import sys
import torch
sys.path.insert(0, '/223040239/yuping/LLADA/flow_matching')

from flow_matching.path.mixture import LearnableScalarLUT


def test_noise_init_monotonicity():
    """Test that noise doesn't break monotonicity"""
    print("="*70)
    print("Test 1: Monotonicity with noise (σ=1e-3)")
    print("="*70)
    
    lut = LearnableScalarLUT(
        num_channels=3,
        vocab_size=256,
        embed_range="pm1",
        device=None,
        dtype=torch.float32
    )
    
    weight = lut.weight.data  # [3, 256]
    
    # Check monotonicity for each channel
    print("\n📈 Monotonicity Check:")
    total_inversions = 0
    for c, name in enumerate(['R', 'G', 'B']):
        emb = weight[c]
        diffs = emb[1:] - emb[:-1]
        inversions = (diffs < 0).sum().item()
        total_inversions += inversions
        
        status = "✓" if inversions == 0 else f"⚠️ {inversions}"
        print(f"  {name}: {status} inversions out of 255")
        
        if inversions > 0 and inversions <= 5:
            inv_locs = torch.where(diffs < 0)[0].tolist()
            print(f"      Locations: {inv_locs}")
    
    # Expected: ~0-2 inversions per channel with σ=1e-3
    assert total_inversions < 10, f"Too many inversions: {total_inversions}"
    print(f"\n  Total inversions: {total_inversions}/765 ({total_inversions/765:.2%})")
    print("  ✓ PASS: Monotonicity preserved (< 10 inversions)")
    

def test_noise_init_decorrelation():
    """Test that channels are decorrelated"""
    print("\n" + "="*70)
    print("Test 2: Channel Decorrelation")
    print("="*70)
    
    # Run multiple times to check variance
    correlations = []
    for trial in range(10):
        lut = LearnableScalarLUT(
            num_channels=3,
            vocab_size=256,
            embed_range="pm1",
            device=None,
            dtype=torch.float32
        )
        weight = lut.weight.data
        
        # Compute R-G, G-B, R-B correlations
        corr_rg = torch.corrcoef(torch.stack([weight[0], weight[1]]))[0, 1].item()
        corr_gb = torch.corrcoef(torch.stack([weight[1], weight[2]]))[0, 1].item()
        corr_rb = torch.corrcoef(torch.stack([weight[0], weight[2]]))[0, 1].item()
        
        correlations.append([corr_rg, corr_gb, corr_rb])
    
    correlations = torch.tensor(correlations)
    mean_corr = correlations.mean(dim=0)
    std_corr = correlations.std(dim=0)
    
    print("\n↔️  Correlation Statistics (10 trials):")
    print(f"  R-G: {mean_corr[0]:.6f} ± {std_corr[0]:.6f}")
    print(f"  G-B: {mean_corr[1]:.6f} ± {std_corr[1]:.6f}")
    print(f"  R-B: {mean_corr[2]:.6f} ± {std_corr[2]:.6f}")
    
    # Without noise: correlation ≈ 1.0
    # With σ=1e-3 noise: correlation should be < 1.0 (typically 0.998-0.9995)
    max_mean_corr = mean_corr.max().item()
    print(f"\n  Max mean correlation: {max_mean_corr:.6f}")
    
    assert max_mean_corr < 1.0, "Channels are perfectly correlated (no decorrelation)"
    assert max_mean_corr > 0.99, f"Correlation too low ({max_mean_corr:.6f} < 0.99), noise may be too large"
    
    print("  ✓ PASS: Channels decorrelated but not too much")


def test_noise_magnitude():
    """Test noise magnitude is as expected"""
    print("\n" + "="*70)
    print("Test 3: Noise Magnitude")
    print("="*70)
    
    # Theoretical std without noise (pm1 range, linear)
    # For linspace(-1, 1, 256): std ≈ 0.577
    # With added noise σ=1e-3: combined_std ≈ sqrt(0.577^2 + 0.001^2) ≈ 0.577
    
    lut = LearnableScalarLUT(
        num_channels=3,
        vocab_size=256,
        embed_range="pm1",
        device=None,
        dtype=torch.float32
    )
    
    weight = lut.weight.data
    
    print("\n📊 Standard Deviations:")
    for c, name in enumerate(['R', 'G', 'B']):
        std = weight[c].std().item()
        print(f"  {name}: {std:.6f}")
    
    mean_std = weight.std(dim=1).mean().item()
    print(f"\n  Mean std: {mean_std:.6f}")
    
    # Expected: ~0.577 (linspace std) + tiny increase from noise
    # Theoretical: sqrt(0.577^2 + 0.001^2) ≈ 0.5770009 ≈ 0.577
    # Empirical with random seed variance: allow up to 0.582
    assert 0.575 < mean_std < 0.582, f"Unexpected std: {mean_std:.6f}"
    print("  ✓ PASS: Std magnitude as expected (~0.577-0.581)")


def test_range_preserved():
    """Test that range is still approximately pm1"""
    print("\n" + "="*70)
    print("Test 4: Range Preservation")
    print("="*70)
    
    lut = LearnableScalarLUT(
        num_channels=3,
        vocab_size=256,
        embed_range="pm1",
        device=None,
        dtype=torch.float32
    )
    
    weight = lut.weight.data
    
    print("\n📏 Value Ranges:")
    for c, name in enumerate(['R', 'G', 'B']):
        min_val = weight[c].min().item()
        max_val = weight[c].max().item()
        print(f"  {name}: [{min_val:.6f}, {max_val:.6f}]")
    
    # Should be close to [-1, 1] with small noise
    global_min = weight.min().item()
    global_max = weight.max().item()
    
    print(f"\n  Global range: [{global_min:.6f}, {global_max:.6f}]")
    
    # With σ=1e-3, range should extend by ~±3σ at most (99.7%)
    assert -1.01 < global_min < -0.99, f"Min too far from -1.0: {global_min}"
    assert 0.99 < global_max < 1.01, f"Max too far from 1.0: {global_max}"
    
    print("  ✓ PASS: Range approximately [-1, 1]")


if __name__ == "__main__":
    print("\n" + "🔬 LUT Noise Initialization Tests")
    print("="*70)
    print("Verifying that σ=1e-3 noise breaks symmetry without breaking monotonicity\n")
    
    test_noise_init_monotonicity()
    test_noise_init_decorrelation()
    test_noise_magnitude()
    test_range_preserved()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)
    print("\nConclusion:")
    print("  • Noise σ=1e-3 preserves monotonicity (< 10 inversions total)")
    print("  • Channels are decorrelated (correlation < 1.0)")
    print("  • Noise magnitude is small and controlled")
    print("  • Value range remains approximately [-1, 1]")
    print("\nReady for training with symmetry-breaking initialization! 🚀")
