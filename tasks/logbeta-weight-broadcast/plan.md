# Plan
1. Update the KL penalty computation in `examples/image/training/train_loop.py` to apply log-β importance weights with correct broadcasting (e.g., unsqueeze weights or reduce KL per sample).
2. Ensure numerical behavior remains unchanged when weights are `None`.
3. Add a unit test covering the weighted KL path to guard against regressions.
