# Plan: Eval speed tweaks + GIF logging

1. **Trim real FID batches**
   - Track how many real samples have already been accumulated inside `eval_model`.
   - Slice the current batch (and labels) so at most `fid_samples` reals are pushed into `fid_metric` and skip updates once the quota is met.
   - Ensure synthetic sampling uses the trimmed tensors, and exit the loop once both real and synthetic quotas are satisfied.

2. **Introduce GIF logging controls**
   - Extend `examples/image/train_arg_parser.py` with flags such as `--save_eval_gif`, `--eval_gif_max_batch`, `--eval_gif_stride`, and `--eval_gif_fps` (choose sensible defaults).
   - Expose these settings via the parsed `args` object.

3. **Capture sampling trajectories**
   - In the eval loop, detect when GIF logging is requested and only for the first batch on the main process request solver trajectories via `return_intermediates=True`.
   - Handle all solver branches (mixture discrete, metric-induced, continuous) and fall back to final samples otherwise.

4. **Emit GIF artifacts**
   - Add a helper that downsamples trajectories (`stride`, `max_batch`), normalizes to `[0, 1]`, arranges them into a grid, and saves a GIF under `output_dir/gifs`.
   - Reuse PIL for GIF writing and optionally log to wandb when available.
   - Guard against datasets with 1 or 3 channels and release GPU memory promptly.

5. **Verification**
   - Run the existing targeted unit tests (e.g., `pytest tests/path/test_path.py`) to ensure the refactor doesn’t break prior behavior.
   - Manually lint the modified modules for obvious style regressions.

