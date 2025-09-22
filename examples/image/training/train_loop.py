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
import torch.nn as nn
from flow_matching.path import (
    CondOTProbPath,
    MixtureDiscreteProbPath,
    MetricInducedGibbsProbPath,
    ProbPath,
    BetaScheduleEMA,
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


class AdaptiveKLController(nn.Module):
    """Adaptive penalty controller mirroring PPO/TRPO style KL annealing."""

    def __init__(
        self,
        *,
        target: float,
        init_weight: float,
        adapt_rate: float = 2.0,
        tolerance: float = 1.5,
        min_weight: float = 1e-4,
        max_weight: float = 1e4,
    ) -> None:
        super().__init__()
        if adapt_rate <= 1.0:
            raise ValueError("adapt_rate must be > 1")
        if tolerance <= 1.0:
            raise ValueError("tolerance must be > 1")
        if min_weight <= 0.0:
            raise ValueError("min_weight must be > 0")
        if max_weight <= 0.0 or max_weight < min_weight:
            raise ValueError("max_weight must be >= min_weight > 0")

        self.target = float(target)
        self.adapt_rate = float(adapt_rate)
        self.tolerance = float(tolerance)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.register_buffer("weight", torch.tensor(float(init_weight)))

    def current_weight(self) -> float:
        return float(self.weight.item())

    def compute_penalty(self, kl_value: torch.Tensor) -> torch.Tensor:
        scale = self.weight.to(device=kl_value.device, dtype=kl_value.dtype)
        return kl_value * scale

    def update(self, measured_kl: Optional[float]) -> None:
        if measured_kl is None:
            return
        kl_val = float(measured_kl)
        if not math.isfinite(kl_val):
            return

        weight = self.current_weight()
        upper = self.target * self.tolerance
        lower = self.target / self.tolerance if self.target > 0 else 0.0

        if self.target > 0 and kl_val > upper:
            weight = min(self.max_weight, weight * self.adapt_rate)
        elif self.target > 0 and kl_val < lower:
            weight = max(self.min_weight, weight / self.adapt_rate)

        self.weight.copy_(torch.tensor(weight, device=self.weight.device))


def _anneal_gumbel_tau(
    step: int, total_steps: int, start: float, end: float, schedule: str
) -> float:
    """Return the annealed Gumbel-Softmax temperature for the given step."""

    if total_steps <= 0:
        return end

    progress = min(max(step / float(total_steps), 0.0), 1.0)
    if schedule == "quadratic":
        weight = (1.0 - progress) ** 2
    elif schedule == "linear":
        weight = 1.0 - progress
    elif schedule == "cosine":
        weight = 0.5 * (math.cos(math.pi * progress) + 1.0)
    else:
        raise ValueError(f"Unsupported Gumbel annealing schedule: {schedule}")
    return end + (start - end) * weight


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
    schedule_ema: Optional[BetaScheduleEMA] = None,
    kl_controller: Optional[AdaptiveKLController] = None,
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

    beta_schedule_module: Optional[nn.Module] = None
    if isinstance(path, MetricInducedGibbsProbPath) and isinstance(
        path.beta_schedule, nn.Module
    ):
        beta_schedule_module = path.beta_schedule

    use_schedule_ema = schedule_ema is not None and beta_schedule_module is not None
    use_schedule_trust_region = use_schedule_ema and kl_controller is not None
    if kl_controller is not None and not use_schedule_ema:
        logger.warning(
            "Schedule KL controller was provided without an EMA teacher; disabling the controller."
        )
        kl_controller = None
        use_schedule_trust_region = False

    kl_metric = kl_penalty_metric = None
    kl_updates_total = 0
    kl_sum_for_step = 0.0
    kl_micro_steps = 0
    if use_schedule_trust_region:
        kl_metric = MeanMetric().to(device, non_blocking=True)
        kl_penalty_metric = MeanMetric().to(device, non_blocking=True)

    # Try to get dataloader length for global step computation
    try:
        _dl_len = len(data_loader)  # type: ignore[arg-type]
    except Exception:
        _dl_len = None

    gumbel_tau_steps = int(getattr(args, "mi_gumbel_tau_anneal_steps", 0) or 0)
    gumbel_tau_start = float(
        getattr(args, "mi_gumbel_tau_start", getattr(args, "mi_gumbel_tau", 1.0))
    )
    gumbel_tau_end = float(
        getattr(args, "mi_gumbel_tau_end", getattr(args, "mi_gumbel_tau", gumbel_tau_start))
    )
    gumbel_tau_schedule = getattr(args, "mi_gumbel_tau_schedule", "quadratic")
    gumbel_schedule_active = (
        gumbel_tau_steps > 0
        and isinstance(path, MetricInducedGibbsProbPath)
        and path.use_gumbel
    )
    current_gumbel_tau = getattr(path, "gumbel_tau", None)
    updates_per_epoch = None
    if gumbel_schedule_active and _dl_len is not None:
        updates_per_epoch = (_dl_len + accum_iter - 1) // accum_iter
    gumbel_update_step = int(getattr(args, "_gumbel_update_step", 0))
    if gumbel_schedule_active and updates_per_epoch is not None:
        epoch_offset = epoch * updates_per_epoch
        if gumbel_update_step < epoch_offset:
            gumbel_update_step = epoch_offset

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

        schedule_kl_value = None
        schedule_kl_penalty = None

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

            if use_schedule_trust_region:
                x1_flat = samples.view(samples.shape[0], -1)
                with torch.no_grad():
                    teacher_beta, _ = schedule_ema.beta_and_derivative(t)
                    teacher_probs = path.get_prob_distribution_from_tokens(
                        x1_flat, t, beta_values=teacher_beta
                    )
                student_probs = path.get_prob_distribution_from_tokens(x1_flat, t)
                eps = 1e-8
                teacher_probs = teacher_probs.clamp_min(eps)
                student_probs = student_probs.clamp_min(eps)
                kl_tensor = (
                    teacher_probs
                    * (torch.log(teacher_probs) - torch.log(student_probs))
                ).sum(dim=-1)
                schedule_kl_value = kl_tensor.mean()
                schedule_kl_penalty = kl_controller.compute_penalty(schedule_kl_value)
                loss = loss + schedule_kl_penalty
                kl_metric.update(schedule_kl_value.detach())
                kl_penalty_metric.update(schedule_kl_penalty.detach())
                kl_updates_total += 1
                kl_sum_for_step += float(schedule_kl_value.detach())
                kl_micro_steps += 1
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

        if apply_update and use_schedule_ema and beta_schedule_module is not None:
            schedule_ema.update(beta_schedule_module)
        if apply_update and use_schedule_trust_region:
            mean_kl_step = kl_sum_for_step / kl_micro_steps if kl_micro_steps > 0 else None
            kl_controller.update(mean_kl_step)
            kl_sum_for_step = 0.0
            kl_micro_steps = 0
        elif apply_update:
            kl_sum_for_step = 0.0
            kl_micro_steps = 0

        if apply_update and gumbel_schedule_active:
            step_index = min(gumbel_update_step + 1, gumbel_tau_steps)
            new_tau = _anneal_gumbel_tau(
                step_index,
                gumbel_tau_steps,
                gumbel_tau_start,
                gumbel_tau_end,
                gumbel_tau_schedule,
            )
            path.gumbel_tau = float(new_tau)
            current_gumbel_tau = float(new_tau)
            gumbel_update_step += 1

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            # Console log
            dl_len_str = str(_dl_len) if _dl_len is not None else "?"
            log_msg = (
                f"Epoch {epoch} [{data_iter_step}/{dl_len_str}]: loss = {batch_loss.compute()}, lr = {lr}"
            )
            if use_schedule_trust_region and schedule_kl_value is not None:
                log_msg += (
                    f", kl = {float(schedule_kl_value.detach().cpu()):.4g},"
                    f" kl_w = {kl_controller.current_weight():.4g}"
                )
            if gumbel_schedule_active and current_gumbel_tau is not None:
                log_msg += f", tau = {float(current_gumbel_tau):.4g}"
            logger.info(log_msg)

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
                            **(
                                {
                                    "train/schedule_kl": float(
                                        schedule_kl_value.detach().cpu()
                                    ),
                                    "train/schedule_kl_penalty": float(
                                        schedule_kl_penalty.detach().cpu()
                                    ),
                                    "train/schedule_kl_weight": float(
                                        kl_controller.current_weight()
                                    ),
                                }
                                if use_schedule_trust_region and schedule_kl_value is not None
                                else {}
                            ),
                            **(
                                {
                                    "train/gumbel_tau": float(current_gumbel_tau)
                                }
                                if gumbel_schedule_active and current_gumbel_tau is not None
                                else {}
                            ),
                        },
                        step=global_step,
                    )
                except Exception:
                    pass

    lr_schedule.step()
    stats = {"loss": float(epoch_loss.compute().detach().cpu())}
    if use_schedule_trust_region and kl_updates_total > 0 and kl_metric and kl_penalty_metric:
        stats.update(
            {
                "schedule_kl": float(kl_metric.compute().detach().cpu()),
                "schedule_kl_penalty": float(kl_penalty_metric.compute().detach().cpu()),
                "schedule_kl_weight": float(kl_controller.current_weight()),
            }
        )
    if gumbel_schedule_active:
        setattr(args, "_gumbel_update_step", gumbel_update_step)
    return stats
