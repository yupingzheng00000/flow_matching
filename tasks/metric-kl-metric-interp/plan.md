# Plan

1. Update the adaptive KL controller usage so the weight adapts as soon as at least one averaged KL observation is available. Compute the running average using the window sum/length and call `kl_controller.update` every optimizer step with valid KL statistics.
2. Surface the averaged KL for logging regardless of window fill so the dashboard reflects the measurements that drive the controller.
3. Simplify the metric interpolation by removing the MAD-based rescaling in `flow_matching/path/mixture.py`. Replace the `_scaled_learned_distance_table` helper with a direct distance-table fetch and blend using `(1 - λ) * base + λ * learned`.
4. Remove any now-unused helpers or cached scale fields associated with the MAD normalization.
5. Run the existing targeted unit test suite (`pytest tests/path/test_path.py`) to confirm the path helpers still function.
