# Plan: Metric geometry evaluation features

1. **CLI extensions**
   - Add metric-geometry evaluation flags to `examples/image/train_arg_parser.py` so runs can opt-in and configure the probes.
   - Options will cover enabling the evaluation, choosing the token subset size, number of pair samples for Spearman, k for the overlap, optional RNG seed, and toggling heatmap export.

2. **Helper utilities in eval loop**
   - Implement helpers inside `examples/image/training/eval_loop.py` to
     - choose the evaluation subset of token indices,
     - extract baseline / learned / teacher / blended distance tables from the metric-induced path on CPU,
     - compute Spearman rank correlation with optional subsampling,
     - compute k-NN overlap (Jaccard) between neighbor sets, and
     - export heatmaps via matplotlib when available (falling back with a warning otherwise).

3. **Integrate geometry probes into evaluation**
   - When metric-induced sampling is active, invoke the helper once per eval (main process only) after the KO path is instantiated.
   - Generate statistics for the student, blended path, and optional EMA teacher, attach them to `eval_stats`, and save any requested heatmaps / wandb artifacts under the configured output directory.

4. **Documentation / logging polish**
   - Ensure log messages explain when computations are skipped (e.g. EMA absent, matplotlib missing, subset too small).
   - Keep evaluation return dict backward compatible when probes disabled.

5. **Testing / verification**
   - Run targeted unit tests (`pytest tests/path/test_path.py`) and, if necessary, create small synthetic sanity checks within the eval helper to validate numerical stability (e.g. shape assertions).
