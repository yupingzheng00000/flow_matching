# Plan

1. **Deduplicate KL controller setup in `train_one_epoch`.**
   - Remove the second initialization block that resets `kl_window`/`kl_metric` so the restored window from `_load_kl_window_state` is preserved.
   - Ensure downstream code still has valid defaults when trust region is disabled.

2. **Drop the unused `--temp` CLI flag.**
   - Delete the argument declaration from `examples/image/train_arg_parser.py`.
   - Search for any documentation mentions and prune them if found.

3. **Remove the dead `learnable_parameters` helper.**
   - Delete the method from `MetricInducedGibbsProbPath` in `flow_matching/path/mixture.py`.
   - Confirm no references remain.

4. **Run unit tests or the targeted test suite.**
   - Execute `pytest tests/path/test_path.py` to cover the modified path module.
   - If numerical tolerances in the cache refresh test prove too loose, tighten the
     assertion to check for a non-zero difference directly.
