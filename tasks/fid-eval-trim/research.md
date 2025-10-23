# FID eval trimming research

- The evaluation loop keeps a running `num_real` counter capped at the per-rank `fid_samples`, but it still feeds full dataloader batches into `fid_metric.update(..., real=True)`.
- When the dataloader batch exceeds the target real quota, the FID metric accumulates more than the desired number of real images even though the counters and early-exit logic assume only `fid_samples` examples were consumed.
- This mismatch explains the regression report: the loop can exit after a single oversized batch while the metric internally measured FID against the whole batch, so results differ from the intended "N real vs N fake" comparison.
