# Cosine LUT Updates (latest changes)

## Summary

- Added cosine-distance support for learnable LUTs in `MetricInducedGibbsProbPath`:
  - `metric` flag now accepts `cosine`.
  - Cosine distance is computed as `s * (1 - cos)` with per-LUT scaling `--mi_lut_cosine_scale`.
  - LUT vectors are normalized on-the-fly; cached tables reuse the same interface (shape `[C, V, V]`).
- Introduced warm-start for cosine LUTs:
  - `_warm_start_cosine_lut` maps tokens to a great-circle initialization (half-arc on the unit sphere).
  - Flag `--mi_lut_force_warm_start` allows reinitializing even when resuming from checkpoints (e.g., finetuning 1D baselines).
  - Warm start now applies the weight copy under `torch.no_grad()` to avoid autograd in-place pitfalls.
- Added cosine scale calibration:
  - `_calibrate_cosine_scale` matches the median distance of the cosine table to the baseline table.
  - Enabled via `--mi_lut_cosine_calibrate`, executed post warm-start and after loading checkpoints.
- Updated LUT diagnostics to project cosine embeddings (angle-based scalarization) before plotting/metrics, replacing the old L2-norm visualization.
- CLI updates:
  - `--mi_metric` choices expanded to include `"euclidean"` and `"cosine"`.
  - New flags `--mi_lut_cosine_scale`, `--mi_lut_cosine_calibrate`, `--mi_lut_force_warm_start`.
- bfloat16 workflow clarified:
  - When `--bf16` is set, GradScaler is disabled and bfloat16 autocast is used for UNet forward passes (train / eval).

## Files touched

- `flow_matching/flow_matching/path/mixture.py`
- `flow_matching/examples/image/train.py`
- `flow_matching/examples/image/train_arg_parser.py`
- `flow_matching/examples/image/training/train_loop.py`
