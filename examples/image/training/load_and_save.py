# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.nn import Module
from training.distributed_mode import is_main_process


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_model(
    args,
    epoch,
    model,
    model_without_ddp,
    optimizer,
    lr_schedule,
    loss_scaler,
    extra_modules: Optional[Dict[str, Module]] = None,
):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [
            output_dir / ("checkpoint-%s.pth" % epoch_name),
            output_dir / "checkpoint.pth",
        ]
        for checkpoint_path in checkpoint_paths:
            extra_state = {
                name: module.state_dict()
                for name, module in (extra_modules or {}).items()
            }
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_schedule": lr_schedule.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "args": args,
                "extra_modules": extra_state,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % epoch_name,
            client_state=client_state,
        )


def load_model(
    args,
    model_without_ddp,
    optimizer,
    loss_scaler,
    lr_schedule,
    extra_modules: Optional[Dict[str, Module]] = None,
):
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            # Prefer safe loading when supported; fall back for older PyTorch
            try:
                checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
            except TypeError:
                checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        if extra_modules and "extra_modules" in checkpoint:
            for name, module in extra_modules.items():
                state_dict = checkpoint["extra_modules"].get(name)
                if state_dict is not None:
                    # Handle legacy checkpoint format for MahalanobisTokenMetric
                    from flow_matching.path.mixture import MahalanobisTokenMetric
                    from flow_matching.path.metric_ema import LearnableMetricEMA
                    
                    if isinstance(module, MahalanobisTokenMetric):
                        # Check if this is a legacy format (has 'codes' but not 'codes_raw')
                        if "codes" in state_dict and "codes_raw" not in state_dict:
                            print(f"Converting legacy metric format for '{name}'...")
                            # Use the property setter to trigger automatic conversion
                            module.codes = state_dict["codes"]
                            if "_lower_params" in state_dict:
                                module._lower_params.data.copy_(state_dict["_lower_params"])
                            if "_extra_state" in state_dict:
                                module.set_extra_state(state_dict["_extra_state"])
                            print(f"  ✓ Converted: codes → codes_raw + log_scale")
                        else:
                            # New format, load normally
                            module.load_state_dict(state_dict)
                    elif isinstance(module, LearnableMetricEMA):
                        # Handle LearnableMetricEMA with legacy teacher metric
                        # Check if teacher has legacy format
                        if "teacher.codes" in state_dict and "teacher.codes_raw" not in state_dict:
                            print(f"Converting legacy EMA teacher metric format for '{name}'...")
                            # Extract teacher state
                            teacher_state = {
                                k.replace("teacher.", ""): v 
                                for k, v in state_dict.items() 
                                if k.startswith("teacher.")
                            }
                            # Convert teacher via property setter
                            if isinstance(module.teacher, MahalanobisTokenMetric):
                                module.teacher.codes = teacher_state["codes"]
                                if "_lower_params" in teacher_state:
                                    module.teacher._lower_params.data.copy_(teacher_state["_lower_params"])
                                if "_extra_state" in teacher_state:
                                    module.teacher.set_extra_state(teacher_state["_extra_state"])
                            # Load non-teacher parts
                            for k, v in state_dict.items():
                                if not k.startswith("teacher."):
                                    # Load buffers like num_updates
                                    if k in dict(module.named_buffers()):
                                        getattr(module, k).copy_(v)
                            print(f"  ✓ Converted EMA teacher: codes → codes_raw + log_scale")
                        else:
                            # New format, load normally
                            module.load_state_dict(state_dict)
                    else:
                        module.load_state_dict(state_dict)
        checkpoint_args = checkpoint.get("args")
        if checkpoint_args is not None:
            for attr in (
                "_gumbel_update_step",
                "_metric_interp_update_step",
                "_schedule_kl_window",
                "_schedule_kl_window_sum",
            ):
                if hasattr(checkpoint_args, attr):
                    setattr(args, attr, getattr(checkpoint_args, attr))
        if (
            "optimizer" in checkpoint
            and "epoch" in checkpoint
            and not (hasattr(args, "eval") and args.eval)
        ):
            # Handle freeze schedule mode: skip optimizer loading if param groups mismatch
            freeze_schedule = getattr(args, "mi_freeze_beta_schedule", False)
            
            if freeze_schedule:
                # In freeze mode, param groups differ: checkpoint has schedule params, current doesn't
                # Skip optimizer state loading, but keep lr_schedule and epoch
                print(f"Freeze mode: Skipping optimizer state loading (param group mismatch)")
                print(f"  Checkpoint was trained with schedule params, now excluded from optimizer")
                print(f"  UNet will start with fresh optimizer state (momentum reset)")
            else:
                # Normal mode: load optimizer state
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("Loaded optimizer state")
            
            # Always load lr_schedule and epoch
            lr_schedule.load_state_dict(checkpoint["lr_schedule"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print(f"Resumed from epoch {checkpoint['epoch']}, continuing to epoch {args.start_epoch}")
