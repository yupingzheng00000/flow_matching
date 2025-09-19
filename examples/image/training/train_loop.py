# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable, Optional

import torch
from flow_matching.path import (
    CondOTProbPath,
    MixtureDiscreteProbPath,
    MetricInducedGibbsProbPath,
    ProbPath,
)
from flow_matching.path.scheduler import PolynomialConvexScheduler
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel

from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount
from training import distributed_mode

logger = logging.getLogger(__name__)

# NOTE: KO metric-induced path trains on 256-way tokens (no mask token).
# The original mixture path branch used a MASK_TOKEN=256 scheme.
MASK_TOKEN = 256
PRINT_FREQUENCY = 10


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
    path: Optional[ProbPath] = None,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    # Select path according to args. For KO CIFAR-10 metric-induced path, set
    #   args.ko_metric_induced = True
    # For original discrete mixture path training, keep args.discrete_flow_matching = True
    if getattr(args, "ko_metric_induced", False):
        assert isinstance(path, MetricInducedGibbsProbPath), (
            "Metric-induced training expects a pre-instantiated MetricInducedGibbsProbPath."
        )
    elif args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
    else:
        path = CondOTProbPath()

    # Try to get dataloader length for global step computation
    try:
        _dl_len = len(data_loader)  # type: ignore[arg-type]
    except Exception:
        _dl_len = None

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"label": labels}

        if getattr(args, "ko_metric_induced", False):
            # KO: 256-way classification; no mask token
            samples = (samples * 255.0).to(torch.long)
            t = torch.rand(samples.shape[0], device=device)

            # Provide dummy x_0 for signature compatibility (not used by metric-induced path)
            x_0 = torch.zeros_like(samples)
            path_sample = path.sample(t=t, x_0=x_0, x_1=samples)
            x_t_model = path_sample.x_t_soft if path_sample.x_t_soft is not None else path_sample.x_t

            # Model should output logits with last dim = 256
            if getattr(args, "bf16", False):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(x_t_model, t=t, extra=conditioning)
            else:
                logits = model(x_t_model, t=t, extra=conditioning)
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape([-1, 256]), samples.reshape([-1])
            ).mean()
        elif args.discrete_flow_matching:
            samples = (samples * 255.0).to(torch.long)
            t = torch.rand(samples.shape[0]).to(device)

            # sample probability path (mixture)
            x_0 = (
                torch.zeros(samples.shape, dtype=torch.long, device=device) + MASK_TOKEN
            )
            path_sample = path.sample(t=t, x_0=x_0, x_1=samples)

            # discrete flow matching loss (257-way: 256 tokens + MASK)
            if getattr(args, "bf16", False):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(path_sample.x_t, t=t, extra=conditioning)
            else:
                logits = model(path_sample.x_t, t=t, extra=conditioning)
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape([-1, 257]), samples.reshape([-1])
            ).mean()
        else:
            # Scaling to [-1, 1] from [0, 1]
            samples = samples * 2.0 - 1.0
            noise = torch.randn_like(samples).to(device)
            if args.skewed_timesteps:
                t = skewed_timestep_sample(samples.shape[0], device=device)
            else:
                t = torch.rand(samples.shape[0]).to(device)
            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t  # type: ignore[attr-defined]

            with torch.cuda.amp.autocast():
                loss = torch.pow(model(x_t, t, extra=conditioning) - u_t, 2).mean()

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            # Console log
            dl_len_str = str(_dl_len) if _dl_len is not None else "?"
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{dl_len_str}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

            # Optional Weights & Biases step-level logging (main process only)
            if getattr(args, "wandb", False) and distributed_mode.is_main_process():
                try:
                    import swanlab as wandb  # type: ignore
                    if _dl_len is not None:
                        global_step = epoch * _dl_len + data_iter_step
                    else:
                        global_step = None
                    wandb.log(  # type: ignore[attr-defined]
                        {
                            "train/step_loss": float(batch_loss.compute().detach().cpu()),
                            "train/inst_loss": float(loss_value),
                            "train/lr": float(lr),
                            "epoch": int(epoch),
                            "step": int(data_iter_step),
                        },
                        step=global_step,
                    )
                except Exception:
                    pass

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
