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
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        if extra_modules and "extra_modules" in checkpoint:
            for name, module in extra_modules.items():
                state_dict = checkpoint["extra_modules"].get(name)
                if state_dict is not None:
                    module.load_state_dict(state_dict)
        if (
            "epoch" in checkpoint and not (hasattr(args, "eval") and args.eval)
        ):
            # Try to restore optimizer state. If the architecture changed (different
            # parameter groups), fall back to fresh optimizer instead of crashing.
            if "optimizer" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])  # type: ignore[arg-type]
                except ValueError as e:
                    print(
                        f"[resume] WARNING: optimizer state incompatible: {e}.\n"
                        "          Continuing with a fresh optimizer."
                    )
                except Exception as e:
                    print(
                        f"[resume] WARNING: failed to load optimizer state: {e}.\n"
                        "          Continuing with a fresh optimizer."
                    )

            # Restore LR scheduler if possible; otherwise align last_epoch.
            try:
                if "lr_schedule" in checkpoint:
                    lr_schedule.load_state_dict(checkpoint["lr_schedule"])  # type: ignore[arg-type]
                else:
                    # Best-effort alignment
                    lr_schedule.last_epoch = checkpoint["epoch"]
            except Exception as e:
                print(
                    f"[resume] WARNING: lr_schedule state incompatible: {e}.\n"
                    f"          Setting last_epoch to {checkpoint['epoch']}."
                )
                try:
                    lr_schedule.last_epoch = checkpoint["epoch"]
                except Exception:
                    pass

            # Restore AMP scaler if present; ignore incompatibilities.
            if "scaler" in checkpoint:
                try:
                    loss_scaler.load_state_dict(checkpoint["scaler"])  # type: ignore[arg-type]
                except Exception as e:
                    print(f"[resume] WARNING: failed to load scaler state: {e}")

            # Always continue from the next epoch when resuming weights.
            args.start_epoch = checkpoint["epoch"] + 1
            print("Resume: weights loaded; optimizer/scheduler restored if compatible.")
