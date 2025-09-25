# Plan to align real FID counts with the requested quota

1. Update `examples/image/training/eval_loop.py` so that each iteration slices the real batch to the remaining quota before calling `fid_metric.update(..., real=True)` and increments `num_real` by the actual number of real images fed to the metric.
2. Preserve the ability to draw conditioning examples for synthesis by continuing to take them from the dataloader batch, but cap them by the outstanding synthetic quota as before.
3. Ensure the updated logic gracefully handles the case where the real quota is already satisfied (skip the metric update while still allowing early exit once fakes catch up).
4. Run the existing unit test suite (or the closest available subset) to confirm the change passes.
