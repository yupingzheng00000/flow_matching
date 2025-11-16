# Implementation Plan

## Step 1 – Update LUT regularizer input tensor
- Modify `examples/image/training/train_loop.py::_compute_lut_regularizer_and_metrics` to obtain the LUT tensor via `lut()` under `torch.no_grad()` so the diagnostics observe renormalization and bounded-residual scaling.
- Keep a fallback to `.weight` for modules without a callable forward, preserving device/dtype handling.

## Step 2 – Derive scalar trajectories per LUT mode
- Inside the same helper, derive per-channel scalar trajectories:
  - For free/"none" LUTs (`LearnableScalarLUT`), average embeddings over the last dimension to recover the scalar baseline.
  - For parametric LUTs, extend `_Line2DLUTParam` and `_Arc2DLUTParam` (and surface them through `LearnableParametricLUT`) with methods returning their underlying scalars (`tau` and `theta`).
- Introduce a shared helper that maps any learnable LUT module to its scalar trajectories without duplicating param logic.

## Step 3 – Rework penalties/metrics around scalar differences
- Recompute `align_term`, `step_term`, `curvature_term`, and all derived metrics using the scalar trajectories and their finite differences (first/second order) plus tangent orientation vectors `[1, delta]` for angle-style summaries.
- Ensure previously logged keys remain available while documenting the scalar-based semantics.

## Step 4 – Adjust linear initialization noise handling
- In `LearnableScalarLUT._linear_init`, scale Gaussian noise by `self.init_noise_scale` and short-circuit when the scale is effectively zero to maintain deterministic warm starts.
- Confirm `reset_parameters` still refreshes buffers correctly.

## Step 5 – Add regression tests
- Extend `tests/path/test_learnable_lut.py` with a case asserting that `init_noise_scale=0` yields a noise-free baseline and that nonzero scales inject noise.
- Add a new training-level unit test (e.g., `tests/training/test_lut_metrics.py`) covering:
  - Metrics computed on a stub LUT whose forward output differs from raw parameters (verifying we now use `lut()`).
  - Scalar-based penalties reacting to bounded-residual scaling (scale factor alters `lut_align` magnitude).

## Step 6 – Verify formatting/tests
- Run the relevant test suite (targeted unit tests) to confirm new behaviors.
