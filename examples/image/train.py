# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Path bootstrap
Ensures the top-level 'flow_matching' package is importable when running
this script from within 'examples/image' (e.g., via torchrun).
"""
import os as _os
import sys as _sys
from pathlib import Path as _Path

_this_dir = _Path(__file__).resolve().parent
# Go up three levels: .../flow_matching/examples/image -> .../flow_matching
_pkg_root = _this_dir.parents[2]
if str(_pkg_root) not in _sys.path:
    _sys.path.insert(0, str(_pkg_root))

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser

from flow_matching.path import (
    BetaSchedule,
    ExpMonotoneRQSConfig,
    ExpMonotoneRQSSchedule,
    MetricInducedGibbsProbPath,
    MonotoneRQBetaSchedule,
    MonotoneRQConfig,
)
from training import distributed_mode
from training.data_transform import get_train_transform
from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch

logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    # Per-rank file logging
    try:
        rank = distributed_mode.get_rank()
    except Exception:
        rank = 0
    log_dir = Path(args.output_dir)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"console_rank{rank}.log", mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(fh)

    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_train_transform()
    if args.dataset == "imagenet":
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=False,
            transform=transform_train,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    logger.info(dataset_train)

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))

    beta_schedule: Optional[BetaSchedule] = None
    metric_path: Optional[MetricInducedGibbsProbPath] = None
    if getattr(args, "ko_metric_induced", False):
        embed_range = "pm1" if args.mi_embed_range == "pm1" else "unit"
        if getattr(args, "mi_learnable_beta", False):
            schedule_type = getattr(args, "mi_beta_schedule", "bounded_rqs")
            if schedule_type == "exp_rqs":
                spline_config = ExpMonotoneRQSConfig(
                    num_bins=int(getattr(args, "mi_spline_bins", 8)),
                    tail_bound=float(getattr(args, "mi_spline_tail_bound", 6.0)),
                    init_c=float(getattr(args, "mi_c", 1.0)),
                    init_a=float(getattr(args, "mi_a", 5.0)),
                    t_eps=float(getattr(args, "mi_t_eps", 1e-4)),
                    logit_eps=float(getattr(args, "mi_logit_eps", 1e-6)),
                )
                beta_schedule = ExpMonotoneRQSSchedule(config=spline_config)
            elif schedule_type == "bounded_rqs":
                spline_config = MonotoneRQConfig(
                    num_bins=int(getattr(args, "mi_spline_bins", 8)),
                    tail_bound=float(getattr(args, "mi_spline_tail_bound", 6.0)),
                    beta_min=float(getattr(args, "mi_beta_min", 0.0)),
                    beta_max=float(getattr(args, "mi_beta_max", 20.0)),
                    t_eps=float(getattr(args, "mi_t_eps", 1e-4)),
                    logit_eps=float(getattr(args, "mi_logit_eps", 1e-6)),
                )
                beta_schedule = MonotoneRQBetaSchedule(config=spline_config)
            else:
                raise ValueError(f"Unsupported β schedule type: {schedule_type}")
            beta_schedule.to(device=device)

        metric_path = MetricInducedGibbsProbPath(
            embedding_path_or_weight=None,
            vocab_size=256,
            emb_dim=1,
            metric=args.mi_metric,
            lp_order=args.mi_lp,
            embed_range=embed_range,
            a=args.mi_a,
            c=args.mi_c,
            device=device,
            dtype=torch.float32,
            beta_schedule=beta_schedule,
            use_gumbel=getattr(args, "mi_use_gumbel", False),
            gumbel_tau=float(getattr(args, "mi_gumbel_tau", 1.0)),
            gumbel_hard=True,
        )

    # define the model
    logger.info("Initializing Model")
    model = instantiate_model(
        architechture=args.dataset,
        is_discrete=args.discrete_flow_matching,
        ko=getattr(args, "ko_metric_induced", False),
        use_ema=args.use_ema,
    )

    model.to(device)

    model_without_ddp = model
    logger.info(str(model_without_ddp))

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    optimizer_params = list(model_without_ddp.parameters())
    extra_modules = {}
    if metric_path is not None:
        schedule_params = list(metric_path.learnable_parameters())
        if schedule_params:
            optimizer_params.extend(schedule_params)
            if isinstance(metric_path.beta_schedule, nn.Module):
                extra_modules["metric_beta_schedule"] = metric_path.beta_schedule
    optimizer = torch.optim.AdamW(
        optimizer_params, lr=args.lr, betas=args.optimizer_betas
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
        extra_modules=extra_modules,
    )

    # Optional Weights & Biases
    wandb_run = None
    if getattr(args, "wandb", False):
        try:
            import swanlab as wandb  # type: ignore
            if distributed_mode.is_main_process():
                wandb_kwargs = dict(
                    project=getattr(args, "wandb_project", "flow_matching"),
                    name=getattr(args, "wandb_run_name", None),
                    entity=getattr(args, "wandb_entity", None),
                    dir=args.output_dir,
                    config=vars(args),
                )
                if getattr(args, "wandb_offline", False):
                    os.environ.setdefault("WANDB_MODE", "offline")
                wandb_run = wandb.init(**{k: v for k, v in wandb_kwargs.items() if v is not None})
        except Exception as e:
            logger.warning(f"wandb not enabled ({e})")

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
                path=metric_path,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {
                "epoch": epoch,
            }

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    extra_modules=extra_modules,
                )
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)
            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                fid_samples = args.fid_samples // num_tasks
            eval_stats = eval_model(
                model,
                data_loader_train,
                device,
                epoch=epoch,
                fid_samples=fid_samples,
                args=args,
                metric_path=metric_path,
            )
            log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})

        # Log to wandb (only on main process)
        if wandb_run is not None and distributed_mode.is_main_process():
            try:
                import swanlab as wandb  # type: ignore
                wandb.log(log_stats, step=epoch)
            except Exception as e:
                logger.warning(f"wandb.log failed: {e}")

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")

    if 'wandb' in globals() and wandb_run is not None and distributed_mode.is_main_process():
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    # Backward-compat: map consolidated flag to legacy alias expected elsewhere
    if not hasattr(args, "ko_metric_induced"):
        setattr(args, "ko_metric_induced", getattr(args, "metric_induced", False))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

"""
CUDA_VISIBLE_DEVICES=0,1 nohup setsid torchrun --standalone 
--nproc_per_node=2 train.py   --dataset=cifar10   --discrete_flow_matching --metric_induced/
--batch_size=384   --lr=0.000125   --accum_iter=1   --epochs=3000  /
--class_drop_prob=1.0   --compute_fid   --sym_func   --cfg_scale=0.0   /
--wandb   --wandb_project flow_matching   --wandb_run_name cifar10_dfm_bs384   --bf16   > out.log 2>&1 &
"""