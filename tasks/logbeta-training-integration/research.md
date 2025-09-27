# Research Notes

## Context
- `ExpMonotoneRQSSchedule.sample_t_uniform_logbeta` already exposes the helper to sample log-β uniformly and return both `t` and `dt/dℓ`.
- Metric-induced KO training in `examples/image/training/train_loop.py` currently samples `t = torch.rand(...)` in the metric-induced branch, so the helper is unused.
- Learnable β-schedule is instantiated in `examples/image/train.py` and stored on the metric-induced path when `--mi_learnable_beta` is enabled.
- CLI options for metric-induced schedule live in `examples/image/train_arg_parser.py`; no arguments currently expose log-β sampling limits.

## Integration requirements
- Replace the uniform-in-`t` draw with uniform-in-`log β` sampling when a learnable exponential spline is active.
- Importance weights must multiply the per-time loss to preserve the original objective.
- Need CLI flags (likely `--mi_logbeta_min`, `--mi_logbeta_max`) to configure the sampling interval.
- Training loop requires access to those args; defaults should be sensible and backward compatible.
