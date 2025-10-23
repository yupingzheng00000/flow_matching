# Research Notes

## Train loop redundancy
- `examples/image/training/train_loop.py` initializes the KL controller state twice in succession within `train_one_epoch`.
- The first branch restores a saved deque via `_load_kl_window_state`, while the second immediately discards it and reinitializes an empty deque.

## Unused CLI argument
- `examples/image/train_arg_parser.py` still declares a `--temp` flag, but no caller reads `args.temp` across the repository (confirmed via `rg "args.temp"`).

## Dead helper on metric-induced path
- `MetricInducedGibbsProbPath.learnable_parameters()` in `flow_matching/path/mixture.py` simply concatenates schedule and metric parameter iterators, yet no code invokes it (verified via `rg "\.learnable_parameters"`).
- Optimizer setup explicitly consumes `schedule_parameters()` and `metric_parameters()` instead.
