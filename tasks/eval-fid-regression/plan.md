# Plan to restore eval stability

1. **Restore robust real-sample updates**
   - Always feed full dataloader batches to the FID metric as before.
   - Track `num_real` using a saturated counter for early-exit logic, but keep the underlying data unchanged.

2. **Target synthetic quota precisely**
   - Derive `remaining_fake` from the quota and construct conditioning batches by slicing the loader batch and labels.
   - Skip solver calls when no quota remains and ensure counters saturate after each update.

3. **Clarify distributed logging**
   - Aggregate per-rank sample counts with an `all_reduce` before printing progress so logs reflect global coverage.

4. **Fix snapshot padding artefacts**
   - Introduce a helper that pads sampled batches by repeating initial images to complete the final grid row before calling `save_image`.

5. **Regression checks**
   - Run the existing unit test suite (`pytest tests/path/test_path.py`) to verify no regressions.
