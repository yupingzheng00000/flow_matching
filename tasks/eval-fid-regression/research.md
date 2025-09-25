# Eval FID regression investigation

## Context
- User noticed evaluation FID scores became worse after recent changes.
- Suspects running FID logging only counts single-GPU samples.
- Snapshot grid images now show black tiles in bottom-right corner.

## Findings
- Current eval loop trims both real and synthetic batches to respect the remaining `fid_samples` per rank.
- Real features fed into `FrechetInceptionDistance` are now limited to the same remainder, whereas the previous implementation always ingested the full dataloader batches (often many more real samples). Fewer real samples increase FID variance and can degrade the score.
- Logging reports per-rank counters (`num_real`, `num_synthetic`), so on multi-GPU runs the progress printout reflects a single worker’s contribution.
- Snapshot grids are saved with `torchvision.utils.save_image`, which lays out images in a fixed-width grid (default `nrow=8`). When the last batch is truncated the grid contains fewer images than expected, leaving zero-filled tiles rendered as black squares.

## Opportunities
- Keep the reduced synthetic workload but revert to feeding full real batches so FID statistics match historical behaviour.
- Clamp counters with `min(..., fid_samples)` and break once both quotas are satisfied to avoid iterating the whole loader.
- Build synthetic conditioning batches from the remaining quota only—this still saves solver work.
- Aggregate counters across distributed workers for logging to avoid confusion.
- Pad snapshot batches by repeating early samples so the saved grid is rectangular without black placeholders.
