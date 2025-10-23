# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""Model-level diagnostics for weight/activation stability analysis.

These utilities implement the diagnostics requested in the EDM2 analysis:
- Per-layer weight norms (RMS, Frobenius, spectral)
- Effective learning rate (update ratio) between checkpoints
- Activation RMS tracking via forward hooks on a fixed batch
- Attention statistics (Q/K RMS, logit std, attention entropy)

The diagnostics are designed to run offline during evaluation and emit CSV
summaries that can be post-processed for visualization.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import init as nn_init
from torch.nn.parallel import DistributedDataParallel

from models.unet import AttentionBlock, ResBlock

logger = logging.getLogger(__name__)


def _get_module(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _tensor_frobenius_norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.norm(tensor.float()).item())


def _tensor_rms(tensor: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(tensor.float().pow(2))).item())


def _fan_in_normalized_rms(tensor: torch.Tensor) -> float:
    rms = torch.sqrt(torch.mean(tensor.float().pow(2)))
    if tensor.ndim < 2:
        return float(rms.item())
    try:
        fan_in, _ = nn_init._calculate_fan_in_and_fan_out(tensor)
    except ValueError:
        return float(rms.item())
    fan_in = max(float(fan_in), 1.0)
    return float(rms * math.sqrt(fan_in))


def _spectral_norm_power_iteration(tensor: torch.Tensor, num_iters: int = 8) -> float:
    weight = tensor.float().reshape(tensor.shape[0], -1) if tensor.ndim >= 2 else tensor.float().reshape(1, -1)
    if weight.numel() == 0:
        return 0.0
    u = torch.randn(weight.size(0), device=weight.device)
    u = u / (u.norm() + 1e-12)
    for _ in range(max(num_iters, 1)):
        v = torch.matmul(weight.t(), u)
        v = v / (v.norm() + 1e-12)
        u = torch.matmul(weight, v)
        u = u / (u.norm() + 1e-12)
    sigma = torch.dot(u, torch.matmul(weight, v))
    return float(sigma.item())


@dataclass
class WeightMetrics:
    checkpoint: str
    step: Optional[int]
    layer: str
    rms_fanin: float
    frobenius: float
    spectral: float


@dataclass
class UpdateRatio:
    checkpoint_prev: str
    checkpoint_curr: str
    step_prev: Optional[int]
    step_curr: Optional[int]
    layer: str
    ratio: float


@dataclass
class ActivationRecord:
    checkpoint: str
    step: Optional[int]
    module: str
    num_channels: int
    rms_min: float
    rms_median: float
    rms_max: float


@dataclass
class AttentionRecord:
    checkpoint: str
    step: Optional[int]
    module: str
    q_rms: float
    k_rms: float
    logit_std: float
    entropy_mean: float
    entropy_min: float
    entropy_max: float


class _ActivationCollector:
    def __init__(self, root: nn.Module) -> None:
        self._root = root
        self._activation_records: List[ActivationRecord] = []
        self._attention_records: List[AttentionRecord] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._attn_handles: List[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "_ActivationCollector":
        self._register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        for handle in self._attn_handles:
            handle.remove()
        self._handles.clear()
        self._attn_handles.clear()

    def records(self) -> Tuple[List[ActivationRecord], List[AttentionRecord]]:
        return self._activation_records, self._attention_records

    def _register(self) -> None:
        for name, module in self._root.named_modules():
            if isinstance(module, (ResBlock, AttentionBlock)):
                handle = module.register_forward_hook(self._make_activation_hook(name))
                self._handles.append(handle)
            if isinstance(module, AttentionBlock):
                handle = module.attention.register_forward_hook(
                    self._make_attention_hook(name)
                )
                self._attn_handles.append(handle)

    def _make_activation_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                return
            data = tensor.detach().float()
            if data.ndim < 3:
                return
            # Collapse spatial dimensions
            dims = tuple(range(2, data.ndim))
            channel_rms = torch.sqrt(torch.mean(data.pow(2), dim=dims))
            if channel_rms.numel() == 0:
                return
            self._activation_records.append(
                ActivationRecord(
                    checkpoint="",
                    step=None,
                    module=name,
                    num_channels=int(channel_rms.numel()),
                    rms_min=float(channel_rms.min().item()),
                    rms_median=float(channel_rms.median().item()),
                    rms_max=float(channel_rms.max().item()),
                )
            )

        return hook

    def _make_attention_hook(self, parent_name: str):
        def hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...], _output) -> None:
            if not inputs:
                return
            qkv = inputs[0]
            if not isinstance(qkv, torch.Tensor):
                return
            bs, width, length = qkv.shape
            attention_block = self._locate_attention_block(parent_name)
            if attention_block is None:
                return
            num_heads = attention_block.num_heads
            if width % (3 * num_heads) != 0:
                return
            head_dim = width // (3 * num_heads)
            q, k, v = qkv.chunk(3, dim=1)
            q = q.view(bs, num_heads, head_dim, length)
            k = k.view(bs, num_heads, head_dim, length)
            q_rms = torch.sqrt(torch.mean(q.float().pow(2)))
            k_rms = torch.sqrt(torch.mean(k.float().pow(2)))

            scale = 1.0 / math.sqrt(math.sqrt(float(head_dim)))
            q_flat = (q.float() * scale).reshape(bs * num_heads, head_dim, length)
            k_flat = (k.float() * scale).reshape(bs * num_heads, head_dim, length)
            logits = torch.bmm(q_flat.transpose(1, 2), k_flat)
            logit_std = logits.std()

            attn = torch.softmax(logits.detach(), dim=-1)
            entropy = -(attn * torch.log(attn.clamp_min(1e-9))).sum(dim=-1)
            self._attention_records.append(
                AttentionRecord(
                    checkpoint="",
                    step=None,
                    module=f"{parent_name}.attention",
                    q_rms=float(q_rms.item()),
                    k_rms=float(k_rms.item()),
                    logit_std=float(logit_std.item()),
                    entropy_mean=float(entropy.mean().item()),
                    entropy_min=float(entropy.min().item()),
                    entropy_max=float(entropy.max().item()),
                )
            )

        return hook

    def _locate_attention_block(self, name: str) -> Optional[AttentionBlock]:
        module = dict(self._root.named_modules()).get(name)
        if isinstance(module, AttentionBlock):
            return module
        return None


class DiagnosticsRunner:
    """Execute diagnostics across a list of checkpoints and dump CSV summaries."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        args,
        data_loader,
    ) -> None:
        self.model = model
        self.device = device
        self.args = args
        self.data_loader = data_loader
        self.power_iters = int(getattr(args, "diag_power_iters", 8))
        self.batch_size = int(getattr(args, "diag_batch_size", getattr(args, "batch_size", 32)))
        diag_time = float(getattr(args, "diag_time", 0.5))
        self.diag_time = float(min(max(diag_time, 0.0), 1.0))
        self.checkpoint_paths = self._resolve_checkpoint_paths()
        self.output_dir = self._resolve_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.weight_records: List[WeightMetrics] = []
        self.update_records: List[UpdateRatio] = []
        self.activation_records: List[ActivationRecord] = []
        self.attention_records: List[AttentionRecord] = []

    def _resolve_checkpoint_paths(self) -> List[Path]:
        paths: List[str] = list(getattr(self.args, "diag_checkpoint", []) or [])
        if not paths:
            resume = getattr(self.args, "resume", None)
            if resume:
                paths.append(resume)
        resolved: List[Path] = []
        for p in paths:
            path = Path(p).expanduser()
            if path.exists():
                resolved.append(path)
            else:
                logger.warning("Diagnostics checkpoint %s does not exist", path)
        return resolved

    def _resolve_output_dir(self) -> Path:
        override = getattr(self.args, "diag_output_dir", None)
        if override:
            return Path(override).expanduser()
        base = getattr(self.args, "output_dir", None)
        if base:
            return Path(base) / "diagnostics"
        return Path.cwd() / "diagnostics"

    def run(self, metric_path: Optional[Any] = None) -> None:
        if not self.checkpoint_paths:
            logger.warning("Diagnostics enabled but no checkpoints were provided")
            return
        model_core = _get_module(self.model)
        orig_state = {k: v.detach().cpu().clone() for k, v in model_core.state_dict().items()}
        prev_state: Optional[Dict[str, torch.Tensor]] = None
        prev_name: Optional[str] = None
        prev_step: Optional[int] = None

        batch_data = self._prepare_batch()
        for ckpt_path in self.checkpoint_paths:
            step = self._load_checkpoint(model_core, ckpt_path, metric_path)
            logger.info("Running diagnostics for %s (step=%s)", ckpt_path, step)
            self._collect_weight_metrics(model_core, ckpt_path.name, step)
            if prev_state is not None and prev_name is not None:
                self._collect_update_ratios(model_core, ckpt_path.name, step, prev_name, prev_step, prev_state)
            self._collect_activation_metrics(model_core, metric_path, batch_data, ckpt_path.name, step)
            prev_state = {
                name: param.detach().cpu().clone()
                for name, param in model_core.named_parameters()
            }
            prev_name = ckpt_path.name
            prev_step = step

        model_core.load_state_dict(orig_state)
        self._write_csvs()

    def _prepare_batch(self):
        iterator = iter(self.data_loader)
        batch = next(iterator)
        if isinstance(batch, (tuple, list)):
            samples = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        else:
            samples = batch
            labels = None
        samples = samples[: self.batch_size].to(self.device, non_blocking=True)
        if labels is not None:
            labels = labels[: self.batch_size].to(self.device, non_blocking=True)
        return samples, labels

    def _load_checkpoint(
        self,
    model_core: nn.Module,
    ckpt_path: Path,
    metric_path: Optional[Any],
    ) -> Optional[int]:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        model_core.load_state_dict(state_dict)
        step = checkpoint.get("epoch")
        extra_modules = checkpoint.get("extra_modules", {})
        if metric_path is not None and isinstance(extra_modules, dict):
            beta_schedule = getattr(metric_path, "beta_schedule", None)
            if isinstance(beta_schedule, nn.Module):
                beta_state = extra_modules.get("metric_beta_schedule")
                if beta_state:
                    beta_schedule.load_state_dict(beta_state)
            learnable_metric = getattr(metric_path, "learnable_metric", None)
            if isinstance(learnable_metric, nn.Module):
                metric_state = extra_modules.get("metric_learnable_metric")
                if metric_state:
                    learnable_metric.load_state_dict(metric_state)
        return step

    def _collect_weight_metrics(
        self,
        model_core: nn.Module,
        checkpoint_name: str,
        step: Optional[int],
    ) -> None:
        for name, param in model_core.named_parameters():
            if param.ndim == 0:
                continue
            tensor = param.detach()
            self.weight_records.append(
                WeightMetrics(
                    checkpoint=checkpoint_name,
                    step=step,
                    layer=name,
                    rms_fanin=_fan_in_normalized_rms(tensor),
                    frobenius=_tensor_frobenius_norm(tensor),
                    spectral=_spectral_norm_power_iteration(tensor, self.power_iters),
                )
            )

    def _collect_update_ratios(
        self,
        model_core: nn.Module,
        checkpoint_name: str,
        step: Optional[int],
        prev_checkpoint: str,
        prev_step: Optional[int],
        prev_state: Dict[str, torch.Tensor],
    ) -> None:
        eps = 1e-12
        for name, param in model_core.named_parameters():
            if param.ndim == 0 or name not in prev_state:
                continue
            current = param.detach().float()
            previous = prev_state[name].to(current.device)
            delta = current - previous
            denom = current.norm().item()
            ratio = delta.norm().item() / (denom + eps)
            self.update_records.append(
                UpdateRatio(
                    checkpoint_prev=prev_checkpoint,
                    checkpoint_curr=checkpoint_name,
                    step_prev=prev_step,
                    step_curr=step,
                    layer=name,
                    ratio=ratio,
                )
            )

    def _collect_activation_metrics(
        self,
    model_core: nn.Module,
    metric_path: Optional[Any],
        batch_data,
        checkpoint_name: str,
        step: Optional[int],
    ) -> None:
        samples, labels = batch_data
        samples = samples.clone()
        labels = labels.clone()
        with torch.no_grad():
            if getattr(self.args, "metric_induced", False) and metric_path is not None:
                tokens = (samples * 255.0).to(torch.long)
                t = torch.full((tokens.shape[0],), self.diag_time, device=self.device)
                x0 = torch.zeros_like(tokens)
                path_sample = metric_path.sample(t=t, x_0=x0, x_1=tokens)
                x_t = path_sample.x_t_soft if path_sample.x_t_soft is not None else path_sample.x_t
            else:
                x_t = samples
                t = torch.full((samples.shape[0],), self.diag_time, device=self.device)
            extra = {"label": labels} if labels.numel() else {}
            model_core.eval()
            with _ActivationCollector(model_core) as collector:
                _ = model_core(x_t, t=t, extra=extra)
                activations, attentions = collector.records()
        for record in activations:
            self.activation_records.append(
                ActivationRecord(
                    checkpoint=checkpoint_name,
                    step=step,
                    module=record.module,
                    num_channels=record.num_channels,
                    rms_min=record.rms_min,
                    rms_median=record.rms_median,
                    rms_max=record.rms_max,
                )
            )
        for record in attentions:
            self.attention_records.append(
                AttentionRecord(
                    checkpoint=checkpoint_name,
                    step=step,
                    module=record.module,
                    q_rms=record.q_rms,
                    k_rms=record.k_rms,
                    logit_std=record.logit_std,
                    entropy_mean=record.entropy_mean,
                    entropy_min=record.entropy_min,
                    entropy_max=record.entropy_max,
                )
            )

    def _write_csvs(self) -> None:
        if self.weight_records:
            self._write_csv(
                "weights.csv",
                [
                    "checkpoint",
                    "step",
                    "layer",
                    "rms_fanin",
                    "frobenius",
                    "spectral",
                ],
                (
                    {
                        "checkpoint": rec.checkpoint,
                        "step": rec.step if rec.step is not None else "",
                        "layer": rec.layer,
                        "rms_fanin": rec.rms_fanin,
                        "frobenius": rec.frobenius,
                        "spectral": rec.spectral,
                    }
                    for rec in self.weight_records
                ),
            )
        if self.update_records:
            self._write_csv(
                "update_ratios.csv",
                [
                    "checkpoint_prev",
                    "step_prev",
                    "checkpoint_curr",
                    "step_curr",
                    "layer",
                    "ratio",
                ],
                (
                    {
                        "checkpoint_prev": rec.checkpoint_prev,
                        "step_prev": rec.step_prev if rec.step_prev is not None else "",
                        "checkpoint_curr": rec.checkpoint_curr,
                        "step_curr": rec.step_curr if rec.step_curr is not None else "",
                        "layer": rec.layer,
                        "ratio": rec.ratio,
                    }
                    for rec in self.update_records
                ),
            )
        if self.activation_records:
            self._write_csv(
                "activations.csv",
                [
                    "checkpoint",
                    "step",
                    "module",
                    "num_channels",
                    "rms_min",
                    "rms_median",
                    "rms_max",
                ],
                (
                    {
                        "checkpoint": rec.checkpoint,
                        "step": rec.step if rec.step is not None else "",
                        "module": rec.module,
                        "num_channels": rec.num_channels,
                        "rms_min": rec.rms_min,
                        "rms_median": rec.rms_median,
                        "rms_max": rec.rms_max,
                    }
                    for rec in self.activation_records
                ),
            )
        if self.attention_records:
            self._write_csv(
                "attention.csv",
                [
                    "checkpoint",
                    "step",
                    "module",
                    "q_rms",
                    "k_rms",
                    "logit_std",
                    "entropy_mean",
                    "entropy_min",
                    "entropy_max",
                ],
                (
                    {
                        "checkpoint": rec.checkpoint,
                        "step": rec.step if rec.step is not None else "",
                        "module": rec.module,
                        "q_rms": rec.q_rms,
                        "k_rms": rec.k_rms,
                        "logit_std": rec.logit_std,
                        "entropy_mean": rec.entropy_mean,
                        "entropy_min": rec.entropy_min,
                        "entropy_max": rec.entropy_max,
                    }
                    for rec in self.attention_records
                ),
            )

    def _write_csv(self, filename: str, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
        path = self.output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        logger.info("Diagnostics wrote %s", path)


__all__ = ["DiagnosticsRunner"]
