# Implementation Plan

1. **Expose KO difficulty-sampling knobs in the CLI.**
   - Add arguments in `examples/image/train_arg_parser.py` for the refresh interval, subsample fraction, maximum sampled positions, proposal mixture weight `α`, and probability ratios `(r_min, r_max)` plus fallback log-β bounds. Defaults: refresh every 128 micro-batches, subsample 5% up to 8192 sites, `α=0.5`, `r_min=0.5`, `r_max=0.05`, fallback band `[0.5, 2.2]`.
   - Ensure the parsed values are stored on `args` in `examples/image/train.py` so the training loop can consume them.

2. **Implement band estimation helpers.**
   - In `examples/image/training/train_loop.py`, add a small dataclass (e.g., `DifficultyBandState`) plus pure helper functions:
     * `_compute_gap_quantiles(distances)` → `(q10, q90)` using clamped two-nearest gaps.
     * `_map_gaps_to_logbeta_band(q10, q90, r_min, r_max, beta_min_clip, beta_max_clip)` returning `(ℓ_min, ℓ_max)` and guarding against degenerate statistics.
   - `DifficultyBandState.refresh(...)` should subsample the flattened target tokens, call `path.distances_from_tokens`, compute quantiles, update `(ℓ_min, ℓ_max)`, and log the refresh summary. Maintain the most recent quantiles and the last refresh step.

3. **Integrate MIS timestep sampling.**
   - Track (and persist on `args`) the difficulty state plus a monotonically increasing micro-step counter across epochs.
   - During the KO metric-induced branch:
     * Invoke `state.maybe_refresh` (using the per-batch counter) before sampling timesteps.
     * Draw per-sample component assignments Bernoulli(α); sample uniform `t` for the uniform component and transform log-β samples for the others via `ExpMonotoneRQSSchedule.sample_t_uniform_logbeta`.
     * Clamp `t` to `[t_eps, 1 - t_eps]`, compute `q_{logβ}(t)` via `beta_and_derivative`, and form detached importance weights `w = 1 / (α + (1-α) q_{logβ}(t))`.
     * Reuse the same weights in the KL penalty branch and ESS diagnostics.
   - Maintain the fallback band `[ℓ_min, ℓ_max]` until the first refresh succeeds; disable the mixture if no schedule or if the band is invalid.

4. **Unit tests.**
   - Add tests under `tests/examples/image/training/test_difficulty_band.py` to cover the quantile-to-band mapping (including clamping/swapping) and the band refresh path using a tiny `MetricInducedGibbsProbPath` with deterministic embeddings.
   - Verify that the refreshed band adheres to the specified clamps and that the state keeps fallback values when statistics are degenerate.
