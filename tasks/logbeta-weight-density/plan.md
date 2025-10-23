1. Update `ExpMonotoneRQSSchedule.sample_t_uniform_logbeta` to multiply the Jacobian weight by the interval length `(lmax - lmin)` with an epsilon clamp for degenerate intervals.
2. Adjust the corresponding unit test in `tests/path/test_path.py` to expect the rescaled weight and, if needed, assert the interval factor explicitly.
3. Run `pytest tests/path/test_path.py` to confirm the schedule helper behaviour and ensure no unintended normalization changes in the training loop.
