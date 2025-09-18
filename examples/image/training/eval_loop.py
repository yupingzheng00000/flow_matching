"""
Path bootstrap
Ensures the top-level 'flow_matching' package is importable when running
this module from within 'examples/image' (e.g., via torchrun).
"""
import sys as _sys
from pathlib import Path as _Path

_this_dir = _Path(__file__).resolve().parent
# Go up three levels: .../flow_matching/examples/image/training -> .../flow_matching
_pkg_root = _this_dir.parents[2]
if str(_pkg_root) not in _sys.path:
    _sys.path.insert(0, str(_pkg_root))

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
import math
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable, cast

import PIL.Image

import torch

def _autocast_cuda():
    """Return an autocast context manager for CUDA with torch.amp if available, else torch.cuda.amp."""
    try:
        from torch import amp as _amp  # type: ignore
        return _amp.autocast("cuda")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return torch.cuda.amp.autocast()
from flow_matching.path import MixtureDiscreteProbPath, MetricInducedGibbsProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver, KODiscreteGibbsEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from training import distributed_mode
from training.edm_time_discretization import get_time_discretization
from training.train_loop import MASK_TOKEN

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module, return_logits: bool = False):
        super().__init__(model)
        self.nfe_counter = 0
        # If True and model is discrete, return raw logits instead of softmax probabilities
        self.return_logits = return_logits

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor
    ):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        is_discrete = isinstance(module, DiscreteUNetModel) or (
            isinstance(module, EMA) and isinstance(module.model, DiscreteUNetModel)
        )
        assert (
            cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with _autocast_cuda(), torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with _autocast_cuda(), torch.no_grad():
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        if is_discrete:
            out = result.to(dtype=torch.float32)
            return out if self.return_logits else torch.softmax(out, dim=-1)
        else:
            return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
):
    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    # For KO solver we need logits; instantiate a logits-returning view lazily
    cfg_scaled_logits_model = None
    cfg_scaled_model.train(False)

    if args.discrete_flow_matching:
        # Branch between mixture path (Meta) and metric-induced path (KO-style)
        if getattr(args, "metric_induced", False):
            disc_solver = None  # KO solver set up lazily below
        else:
            scheduler = PolynomialConvexScheduler(n=3.0)
            path = MixtureDiscreteProbPath(scheduler=scheduler)
            p = torch.zeros(size=[257], dtype=torch.float32, device=device)
            p[256] = 1.0
            disc_solver = MixtureDiscreteEulerSolver(
                model=cfg_scaled_model,
                path=path,
                vocabulary_size=257,
                source_distribution_p=p,
            )
        cont_solver = None
        cont_ode_opts = None
    else:
        disc_solver = None
        cont_solver = ODESolver(velocity_model=cfg_scaled_model)
        cont_ode_opts = args.ode_options

    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )

    num_synthetic = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    # Try to get the length for logging; fall back gracefully if unknown
    try:
        _data_loader_len_for_log = len(data_loader)  # type: ignore[arg-type]
    except Exception:
        _data_loader_len_for_log = None

    # Lazily constructed KO solver and path (once K is known)
    ko_solver = None
    ko_path = None

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        fid_metric.update(samples, real=True)

        if num_synthetic < fid_samples:
            # Reset NFE counter on the wrapper that will actually be used
            # For mixture/continuous branches we use cfg_scaled_model; for metric-induced we use cfg_scaled_logits_model
            # Note: metric-induced branch performs a dummy forward to infer K; we reset AFTER that to avoid +1 in the count
            cfg_scaled_model.reset_nfe_counter()
            if args.discrete_flow_matching:
                # Discrete sampling
                if getattr(args, "metric_induced", False):
                    # Metric-induced Gibbs path using dedicated KO solver
                    # Lazily build logits-wrapper and KO solver with correct vocab size K
                    if cfg_scaled_logits_model is None:
                        cfg_scaled_logits_model = CFGScaledModel(model=model, return_logits=True)
                    if ko_solver is None or ko_path is None:
                        # infer K by one forward pass at t=0
                        x_dummy = torch.zeros(samples.shape, dtype=torch.long, device=device)
                        # IMPORTANT: do not apply CFG scaling with discrete logits
                        logits_dummy = cfg_scaled_logits_model(
                            x=x_dummy,
                            t=torch.tensor(0.0, device=device),
                            cfg_scale=0.0,
                            label=labels,
                        )
                        K = int(logits_dummy.shape[-1])
                        # Build path
                        mi_metric = getattr(args, "mi_metric", "lp")
                        mi_lp = float(getattr(args, "mi_lp", 3.0))
                        mi_a = float(getattr(args, "mi_a", 5.0))
                        mi_c = float(getattr(args, "mi_c", 1.0))
                        mi_embed_range = getattr(args, "mi_embed_range", "pm1")
                        ko_path = MetricInducedGibbsProbPath(
                            embedding_path_or_weight=None,
                            vocab_size=K,
                            emb_dim=1,
                            metric=mi_metric,
                            lp_order=mi_lp,
                            embed_range=mi_embed_range,
                            a=mi_a,
                            c=mi_c,
                            device=device,
                            dtype=torch.float32,
                        )
                        ko_solver = KODiscreteGibbsEulerSolver(
                            model=cfg_scaled_logits_model,
                            path=ko_path,
                            vocabulary_size=K,
                        )
                    # Reset NFE counter on the logits wrapper before stepping to avoid counting the dummy forward
                    cfg_scaled_logits_model.reset_nfe_counter()
                    # Start tokens: uniform over [0, K) since β(0)=0 ⇒ p0 is uniform
                    K_init = ko_solver.vocabulary_size
                    x_0 = torch.randint(0, K_init, samples.shape, device=device, dtype=torch.long)
                    dtype_cat = torch.float32 if args.sampling_dtype == "float32" else torch.float64
                    synthetic_samples = ko_solver.sample(
                        x_init=x_0,
                        step_size=1.0 / args.discrete_fm_steps,
                        dtype_categorical=dtype_cat,
                        label=labels,
                        # IMPORTANT: disable CFG scaling when using discrete logits
                        cfg_scale=0.0,
                    )
                else:
                    x_0 = (
                        torch.zeros(samples.shape, dtype=torch.long, device=device)
                        + MASK_TOKEN
                    )
                    if args.sym_func:
                        # Ensure a pure-Python function returning float for div_free
                        def sym(tau: float) -> float:
                            return 12.0 * (tau ** 2.0) * ((1.0 - tau) ** 0.25)
                    else:
                        sym = args.sym
                    dtype = torch.float32 if args.sampling_dtype == "float32" else torch.float64

                    # Guard against missing solver (should never be None in this branch)
                    assert disc_solver is not None, "Discrete solver not initialized"
                    synthetic_samples = disc_solver.sample(
                        x_init=x_0,
                        step_size=1.0 / args.discrete_fm_steps,
                        verbose=False,
                        div_free=sym,
                        dtype_categorical=dtype,
                        label=labels,
                        # Disable CFG scaling for discrete models (logits)
                        cfg_scale=0.0,
                    )
            else:
                # Continuous sampling
                x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)

                # Safe defaults for ODE options
                nfe_default = 50
                atol_default = 1e-5
                rtol_default = 1e-5
                step_default = None
                if cont_ode_opts is not None:
                    ode_nfe = int(cont_ode_opts.get("nfe", nfe_default))
                    ode_atol = float(cont_ode_opts.get("atol", atol_default))
                    ode_rtol = float(cont_ode_opts.get("rtol", rtol_default))
                    ode_step = cont_ode_opts.get("step_size", step_default)
                else:
                    ode_nfe = nfe_default
                    ode_atol = atol_default
                    ode_rtol = rtol_default
                    ode_step = step_default

                if args.edm_schedule:
                    time_grid = get_time_discretization(nfes=ode_nfe)
                else:
                    time_grid = torch.tensor([0.0, 1.0], device=device)

                # Guard against missing solver
                assert cont_solver is not None, "Continuous solver not initialized"
                synthetic_samples = cont_solver.sample(
                    time_grid=time_grid,
                    x_init=x_0,
                    method=args.ode_method,
                    return_intermediates=False,
                    atol=ode_atol,
                    rtol=ode_rtol,
                    step_size=ode_step,
                    label=labels,
                    cfg_scale=args.cfg_scale,
                )

                # Scaling to [0, 1] from [-1, 1]
                if isinstance(synthetic_samples, (list, tuple)):
                    synthetic_samples = synthetic_samples[-1]
                synthetic_samples = cast(torch.Tensor, synthetic_samples)
                synthetic_samples = torch.clamp(
                    synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
                )
                synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
            # Report NFE from the active wrapper (metric-induced uses logits wrapper)
            _nfe_model = (
                cfg_scaled_logits_model if getattr(args, "metric_induced", False) and 'cfg_scaled_logits_model' in locals() and cfg_scaled_logits_model is not None else cfg_scaled_model
            )
            logger.info(
                f"{samples.shape[0]} samples generated in {_nfe_model.get_nfe()} evaluations."
            )
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            fid_metric.update(synthetic_samples, real=False)
            num_synthetic += synthetic_samples.shape[0]
            if not snapshots_saved and args.output_dir:
                save_image(
                    synthetic_samples,
                    fp=Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}.png",
                )
                snapshots_saved = True

            if args.save_fid_samples and args.output_dir:
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.output_dir) / "fid_samples"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if not args.compute_fid:
            return {}

        if data_iter_step % PRINT_FREQUENCY == 0:
            # Sync fid metric to ensure that the processes dont deviate much.
            gc.collect()
            running_fid = fid_metric.compute()
            if _data_loader_len_for_log is not None:
                _len_str = str(_data_loader_len_for_log)
            else:
                _len_str = "?"
            logger.info(
                f"Evaluating [{data_iter_step}/{_len_str}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}"
            )

        if args.test_run:
            break

    return {"fid": float(fid_metric.compute().detach().cpu())}
