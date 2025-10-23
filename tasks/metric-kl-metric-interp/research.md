# Research Notes

- `AdaptiveKLController` updates its weight only when the KL rolling window reaches its max length; with large `mi_beta_kl_avg_window` values this delays adaptation and can keep the weight stuck at its initial value even when the KL explodes.
- The update hook lives in `examples/image/training/train_loop.py` and currently requires `len(kl_window) == kl_window.maxlen` before adapting.
- Metric interpolation between the frozen baseline and the learnable Mahalanobis metric is implemented in `flow_matching/path/mixture.py`; the helper `_scaled_learned_distance_table` rescales the learned table to match the baseline MAD before blending.
- The user would like to drop the MAD renormalization and revert to a direct convex combination of the two distance tables.
