# Research Notes
- `ExpMonotoneRQSSchedule.beta_and_derivative` exponentiates `y0 + a * r(s)` with `s = logit(t)` to produce `beta` and its derivative `dot_beta`.
- `_ExpRQS1D.forward` already returns both the spline value `r(s)` and its derivative `dr/ds` for any logit input.
- `_ExpRQS1D.inverse` (added previously) reconstructs the logit input `s` from a spline output and reuses `forward` to obtain the derivative at the recovered point.
- No helper exists yet to sample log-β uniformly and map back to `t`; training callers still need to implement this manually.
- Tests in `tests/path/test_path.py` cover round-trip consistency between `_ExpRQS1D.forward` and `.inverse` but do not exercise any sampling helper on the schedule.
