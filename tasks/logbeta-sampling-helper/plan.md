# Plan
1. **Implement sampling helper**: In `flow_matching/path/beta_schedules.py`, add a `sample_t_uniform_logbeta` method to `ExpMonotoneRQSSchedule` that draws uniform log-β samples, inverts them via `_ExpRQS1D.inverse`, maps logits back to `t`, clamps to `[t_eps, 1-t_eps]`, and returns both `t` and the importance weight `dt/dℓ = t(1-t)/(a * r'(s))` with a small clamp to avoid division by zero.
2. **Unit tests**: Extend `tests/path/test_path.py` to cover the new helper by sampling a batch of log-β values, verifying that forward/inverse agree with the original tensor, checking the returned weights against finite differences of β(t), and testing boundary behavior (e.g., lmin/lmax corresponding to spline tails).
3. **Documentation snippet**: Update the README or inline docstring if needed to mention the helper usage so downstream training can discover it.
4. **Run tests**: Execute the existing test suite (or focused subset) to ensure the new helper and tests pass.
