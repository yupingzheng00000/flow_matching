# Plan: Implement inverse for Exp RQS schedule

1. **Implement `_ExpRQS1D.inverse`:**
   - Mirror parameter reconstruction from `forward` (widths, heights, deltas, knot locations).
   - Flatten inputs and separate tail vs central cases.
   - For central values, bucketize over `yk` to find bin, compute normalized target `z`, solve quadratic using coefficients from research, clamp discriminant, choose valid root, map back to `s`.
   - Reshape output to original shape and obtain derivative by calling `forward` on recovered `s` (reuse existing routine).

2. **Add unit tests:**
   - Create test covering random samples across range verifying `inverse(forward(x)[0])[0]` ≈ `x` (use tolerance) for interior.
   - Include boundary/tail cases (values beyond tail bound) ensuring identity mapping.
   - Confirm derivative returned by inverse matches forward derivative by comparing second output shapes/positivity.

3. **Run existing tests:**
   - Execute targeted test module to ensure new functionality passes (e.g., `pytest tests/path/test_path.py`).
