# Plan

1. **Expose CLI controls**
   - Add `--mi_logbeta_min` and `--mi_logbeta_max` to `examples/image/train_arg_parser.py` with defaults matching the analytic KO schedule evaluated at safe quantiles (e.g., log β at `t_eps` and `1 - t_eps`).
   - Document their usage in the help text and ensure values are stored on `args` for training/eval.

2. **Plumb arguments into training setup**
   - In `examples/image/train.py`, read the new args when constructing the metric-induced path or during loop prep so the training loop can access them (store on `args` if not already available).

3. **Update training loop sampling**
   - In `train_one_epoch`, replace the uniform `torch.rand` sampling with a call to `metric_path.beta_schedule.sample_t_uniform_logbeta` when the schedule is an `ExpMonotoneRQSSchedule` and log-β interval arguments are provided.
   - Multiply the per-sample loss by the returned importance weights before reduction.
   - Maintain existing behavior (uniform `t`) when the helper or schedule is unavailable to keep backward compatibility.

4. **Add safeguards and logging**
   - Validate `l_min < l_max` on startup and warn or fall back gracefully if not.
   - Optionally log once per epoch when the importance sampling branch is active (reuse existing logging infrastructure).

5. **Tests**
   - Extend `tests/path/test_path.py` or add a new test to cover the helper integration if feasible (e.g., check weights > 0), or add a focused unit test that mocks a schedule to confirm weight application in the training loop utility.
   - Run `pytest tests/path/test_path.py` to ensure existing schedule tests still pass.
