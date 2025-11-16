# Research Notes

## Relevant modules
- `examples/image/training/train_loop.py::_compute_lut_regularizer_and_metrics`
  - Currently pulls `lut.weight` directly and normalizes embeddings with `F.normalize` before computing align/step/curvature penalties and flip metric.
  - Metrics derived from normalized vectors: cosines, flip rate, KS projection using all-ones vector applied to **raw** weight (not forward output).
- `flow_matching/path/mixture.py::LearnableScalarLUT`
  - `forward` optionally renormalizes to stored Frobenius norms (`renormalize_to_init_norm`) and/or applies bounded residual scaling (`bounded_residual_scale`).
  - `_linear_init` always adds Gaussian noise scaled by fixed `1e-3`; ignores `init_noise_scale`.
  - Param modes (`none`, `line2d`, `arc2d`) produce embeddings via scalar trajectories (tiled baseline, tau along basis, theta on arc).

## Observations
- Diagnostics never call `lut()`; renormalization/bounded residual scaling applied in `forward` are invisible.
- After per-token normalization, scalar LUT baselines collapse to two orientations, so tiny jitter yields flip rate ~1 despite monotone scalar ordering.
- Need scalar trajectory extraction per mode: likely available in param classes or can project using mode-specific helpers.
- Linear init noise must respect `self.init_noise_scale` and allow zero noise.

## Open questions / considerations
- Determine best way to access mode-specific scalar trajectories without duplicating mode logic.
- Ensure metrics remain device/dtype safe when invoking `lut()` under `torch.no_grad()`.
- Need testing hook: either new unit test around diagnostics or integration in existing logging utilities.
