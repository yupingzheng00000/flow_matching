# Research: Metric geometry evaluation hooks

## Existing metric distance infrastructure
- `flow_matching/path/mixture.py` implements `MetricInducedGibbsProbPath`.
  - `_get_base_distance_table(device, dtype)` returns the cached fixed geometry distance matrix for the embedding.
  - `_build_distance_table(device, dtype, metric_module=None, cache_result=True)` mixes the baseline table with either the in-place learnable metric or an override module (e.g. EMA teacher) scaled to match baseline MAD, and caches the blended matrix when `metric_module` is `None`.
  - `learnable_metric` (a `MahalanobisTokenMetric`) exposes `pairwise_distance_table(device, dtype)` that expands to the full learned distance matrix.

## Evaluation loop context
- `examples/image/training/eval_loop.py` runs sampling for FID and currently only logs snapshots/GIFs.
  - When metric-induced sampling is active, the loop instantiates the `MetricInducedGibbsProbPath` (either provided or constructed on-the-fly) and has access to the KO path object used for sampling.
  - Evaluation already branches on `args.metric_induced` / `args.ko_metric_induced`, so path-specific probes can be inserted after the solver is constructed.

## Potential reporting surface
- `eval_model` returns a dictionary merged into the eval stats. Existing keys include just `fid`.
- Wandb logging already happens higher up; eval loop could emit additional artifacts (CSVs, heatmaps, etc.) by writing into `args.output_dir` and optionally pushing to wandb.

## External utilities
- No direct helper for Spearman or k-NN overlap exists yet. PyTorch has `torchmetrics` but Spearman is available via `torchmetrics.functional.spearmanr` or SciPy (if available). Need to guard imports to avoid hard dependency.
- Heatmap generation can reuse matplotlib or seaborn if available; otherwise fallback to PIL/torch image export similar to existing snapshot logic.
