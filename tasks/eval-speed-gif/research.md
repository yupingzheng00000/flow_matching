# Research Notes: Eval speed + GIF logging

## Evaluation loop structure
- `examples/image/training/eval_loop.py` drives evaluation. It always updates the FID metric with the full batch of real samples (`fid_metric.update(samples, real=True)`) before any synthetic generation. The loop doesn't track how many real examples have been accumulated, so if `fid_samples` is smaller than the evaluation loader batch size it still processes entire batches of real images.
- Synthetic sampling stops when `num_synthetic` reaches `fid_samples`, but the dataloader keeps feeding batches even if enough real examples have already been collected. The code trims synthetic batches via slicing when the cumulative count would exceed `fid_samples`, but real samples are never truncated.

## Sampler APIs
- Discrete mixture solver (`flow_matching/solver/discrete_solver.py`) and KO Gibbs solver (`flow_matching/solver/ko_discrete_solver.py`) both accept `return_intermediates=True` and return a tensor of shape `(num_steps + 1, batch, ...)` capturing the intermediate states.
- Continuous ODE solver (`flow_matching/solver/ode_solver.py`) also supports `return_intermediates=True` and returns the entire trajectory sampled over the provided time grid.

## Existing logging utilities
- The eval loop already writes a PNG snapshot via `torchvision.utils.save_image` the first time synthetic samples are generated when `args.output_dir` is set.
- Metric-induced schedule logging uses disk writes and optionally wandb logging, so GIF export can follow a similar pattern (create directories under `args.output_dir` and optionally push to wandb if available).

## CLI surface
- `examples/image/train_arg_parser.py` currently exposes FID-related flags (`--fid_samples`, `--compute_fid`, `--save_fid_samples`) but nothing for trajectory logging, so adding explicit GIF controls is reasonable.
- No dedicated helper exists for trimming FID batches; implementing the logic in the eval loop or factoring out a small utility are both viable.

