# Research Notes

## Existing KO metric-induced training loop
- `examples/image/training/train_loop.py` handles the KO metric-induced path branch with tokenized samples.
- Timesteps `t` are currently sampled either uniformly in `[0, 1]` or via `ExpMonotoneRQSSchedule.sample_t_uniform_logbeta`, which returns `t` along with importance weights equal to `(lmax - lmin) * dt/dℓ`.
- Per-sample losses are aggregated through `_importance_weighted_mean`, which broadcasts optional weights across tensor dimensions before averaging.
- No existing mechanism adapts the log-β interval based on batch difficulty; `mi_logbeta_min`/`mi_logbeta_max` are fixed scalars stored on `args` in `examples/image/train.py`.

## Metric-induced path utilities
- `flow_matching/path/mixture.py` implements `MetricInducedGibbsProbPath` with Euclidean distance as the default metric (using `torch.cdist`).
- `distances_from_tokens` gathers per-token distances to the entire vocabulary by indexing into a cached [K, K] distance table, so obtaining per-site pairwise distances is straightforward once token indices are available.
- The learnable β schedule `ExpMonotoneRQSSchedule` (in `flow_matching/path/beta_schedules.py`) exposes `beta_and_derivative(t)` and `sample_t_uniform_logbeta`. The derivative satisfies `dt/dℓ = β / dβ/dt`, so `q_{logβ}(t) = (dβ/dt) / (β * (ℓ_max - ℓ_min))`.

## Relevant CLI wiring
- `examples/image/train_arg_parser.py` defines KO-specific flags; currently no arguments exist for adaptive log-β bands, MIS blending, or gap sampling controls.
- `examples/image/train.py` initializes `args.mi_logbeta_min`/`args.mi_logbeta_max` when an exponential spline schedule is used, defaulting to `log β` evaluated at `t = mi_t_eps` and `t = 1 - mi_t_eps` if the user did not supply values.

## Constraints from the spec
- Need periodic “refresh” that subsamples target tokens, computes the top-2 distance gap Δd per position (clamped ≥ 1e-12), and tracks 10th/90th percentiles.
- Map `(q10, q90)` to a log-β band using user-chosen probability ratios `r_min`, `r_max`, clamping β within `[1e-3, 1e3]`. Use fallback `[0.5, 2.2]` (natural log) before the first refresh.
- Perform mixture importance sampling with proposal `q(t) = α * U[0,1] + (1-α) * q_{logβ}(t)` and weight each sample by `w(t) = 1 / q(t)` (detached).
- Clamp timesteps to `[t_ε, 1 - t_ε]` for stability and detach importance weights from autograd before applying them.
