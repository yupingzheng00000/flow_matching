# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import logging

from models.model_configs import MODEL_CONFIGS
from torchdiffeq._impl.odeint import SOLVERS

logger = logging.getLogger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser("Image dataset training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=921, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--optimizer_betas",
        nargs="+",
        type=float,
        default=[0.9, 0.999],
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--decay_lr",
        action="store_true",
        help="Adds a linear decay to the lr during training.",
    )
    parser.add_argument(
        "--class_drop_prob",
        type=float,
        default=0.2,
        help="Probability to drop conditioning during training",
    )
    parser.add_argument(
        "--skewed_timesteps",
        action="store_true",
        help="Use skewed timestep sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--edm_schedule",
        action="store_true",
        help="Use the alternative time discretization during sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="When evaluating, use the model Exponential Moving Average weights.",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        default=list(MODEL_CONFIGS.keys())[0],
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Dataset to use.",
    )
    parser.add_argument(
        "--data_path",
        default="./data/image_generation",
        type=str,
        help="imagenet root folder with train, val and test subfolders",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--ode_method",
        default="midpoint",
        choices=list(SOLVERS.keys()) + ["edm_heun"],
        help="ODE solver used to generate samples.",
    )
    parser.add_argument(
        "--ode_options",
        default='{"step_size": 0.01}',
        type=json.loads,
        help="ODE solver options. Eg. the midpoint solver requires step-size, dopri5 has no options to set.",
    )
    parser.add_argument(
        "--sym",
        default=0.0,
        type=float,
        help="Symmetric term for sampling the discrete flow.",
    )
    parser.add_argument(
        "--temp",
        default=1.0,
        type=float,
        help="Temperature for sampling the discrete flow.",
    )
    parser.add_argument(
        "--sym_func",
        action="store_true",
        help="Use a fixed function for the symmetric term in the discrete flow.",
    )
    parser.add_argument(
        "--sampling_dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Solver dtype for sampling the discrete flow.",
    )
    parser.add_argument(
        "--cfg_scale",
        default=0.2,
        type=float,
        help="Classifier-free guidance scale for generating samples.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 autocast during training/eval (disables GradScaler).",
    )
    parser.add_argument(
        "--fid_samples",
        default=1000,
        type=int,
        help="number of synthetic samples for FID evaluations",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="start epoch (used when resumed from checkpoint)",
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="No training, only run evaluation"
    )
    parser.add_argument(
        "--eval_frequency",
        default=50,
        type=int,
        help="Frequency (in number of epochs) for running FID evaluation. -1 to never run evaluation.",
    )
    parser.add_argument(
        "--compute_fid",
        action="store_true",
        help="Whether to compute FID in the evaluation loop. When disabled, the evaluation loop still runs and saves snapshots, but skips the FID computation.",
    )
    parser.add_argument(
        "--save_fid_samples",
        action="store_true",
        help="Save all samples generated for FID computation.",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Only run one batch of training and evaluation.",
    )
    parser.add_argument(
        "--discrete_flow_matching",
        action="store_true",
        help="Train discrete flow matching model.",
    )

    parser.add_argument(
        "--discrete_fm_steps",
        default=1024,
        type=int,
        help="Number of sampling steps for discrete FM.",
    )

    # Metric-induced (KO) discrete path options for evaluation/sampling
    parser.add_argument(
        "--metric_induced", "--ko_metric_induced",
        action="store_true",
        dest="metric_induced",
        help="Use metric-induced Gibbs path (KO Appendix E.3) for discrete sampling.",
    )
    parser.add_argument(
        "--mi_metric",
        default="lp",
        choices=["lp", "cosine"],
        type=str,
        help="Metric to use for the metric-induced path.",
    )
    parser.add_argument(
        "--mi_lp",
        default=3.0,
        type=float,
        help="Lp order when using --mi_metric=lp.",
    )
    parser.add_argument(
        "--mi_a",
        default=5.0,
        type=float,
        help="Exponent 'a' in beta(t) = c * (t/(1-t))^a.",
    )
    parser.add_argument(
        "--mi_c",
        default=1.0,
        type=float,
        help="Scale 'c' in beta(t) = c * (t/(1-t))^a.",
    )
    parser.add_argument(
        "--mi_embed_range",
        default="pm1",
        choices=["pm1", "01"],
        type=str,
        help="Embedding range for tokens: 'pm1' maps to [-1,1], '01' maps to [0,1]",
    )
    parser.add_argument(
        "--mi_learnable_beta",
        action="store_true",
        help="Enable learnable monotone RQ spline schedule for the metric-induced β(t).",
    )
    parser.add_argument(
        "--mi_beta_schedule",
        default="bounded_rqs",
        choices=["bounded_rqs", "exp_rqs"],
        type=str,
        help=(
            "Learnable β(t) schedule type. "
            "'bounded_rqs' matches the original sigmoid-bounded spline while "
            "'exp_rqs' uses an exponential spline with an exact warm start to c*(t/(1-t))^a."
        ),
    )
    parser.add_argument(
        "--mi_beta_min",
        default=0.0,
        type=float,
        help="Lower bound for the learnable β(t) schedule.",
    )
    parser.add_argument(
        "--mi_beta_max",
        default=20.0,
        type=float,
        help="Upper bound for the learnable β(t) schedule.",
    )
    parser.add_argument(
        "--mi_spline_bins",
        default=8,
        type=int,
        help="Number of knots (bins) for the monotone RQ spline β(t).",
    )
    parser.add_argument(
        "--mi_spline_tail_bound",
        default=6.0,
        type=float,
        help="Tail bound of the spline domain in logit time.",
    )
    parser.add_argument(
        "--mi_t_eps",
        default=1e-4,
        type=float,
        help="Clamp applied to t before evaluating the spline schedule.",
    )
    parser.add_argument(
        "--mi_logit_eps",
        default=1e-6,
        type=float,
        help="Stability epsilon used when applying the logit inside the spline schedule.",
    )
    parser.add_argument(
        "--mi_beta_use_ema",
        action="store_true",
        help="Track an exponential moving average teacher of the learnable β(t) schedule",
    )
    parser.add_argument(
        "--mi_beta_ema_decay",
        default=0.999,
        type=float,
        help="Decay for the EMA teacher of the β(t) schedule (closer to 1 slows the updates).",
    )
    parser.add_argument(
        "--mi_beta_kl_target",
        default=0.01,
        type=float,
        help="Target forward KL divergence between the EMA teacher and student β(t) policies.",
    )
    parser.add_argument(
        "--mi_beta_kl_init_weight",
        default=1.0,
        type=float,
        help="Initial multiplier for the schedule KL penalty (adaptive controller adjusts it).",
    )
    parser.add_argument(
        "--mi_beta_kl_adapt_rate",
        default=2.0,
        type=float,
        help="Multiplicative step applied to the KL weight when diverging from the target.",
    )
    parser.add_argument(
        "--mi_beta_kl_tolerance",
        default=1.5,
        type=float,
        help=(
            "Tolerance band around the target KL before the adaptive controller changes the"
            " penalty weight."
        ),
    )
    parser.add_argument(
        "--mi_beta_kl_min_weight",
        default=1e-4,
        type=float,
        help="Lower clamp for the adaptive schedule KL weight.",
    )
    parser.add_argument(
        "--mi_beta_kl_max_weight",
        default=1e4,
        type=float,
        help="Upper clamp for the adaptive schedule KL weight.",
    )
    parser.add_argument(
        "--mi_use_gumbel",
        action="store_true",
        help="Use straight-through Gumbel-Softmax sampling for metric-induced training inputs.",
    )
    parser.add_argument(
        "--mi_gumbel_tau",
        default=1.0,
        type=float,
        help="Temperature for the Gumbel-Softmax sampler when --mi_use_gumbel is set.",
    )

    # Optional Weights & Biases logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (main process only).",
    )
    parser.add_argument(
        "--wandb_project",
        default="flow_matching",
        type=str,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        default="",
        type=str,
        help="Weights & Biases run name (optional).",
    )
    parser.add_argument(
        "--wandb_entity",
        default="",
        type=str,
        help="Weights & Biases entity (optional).",
    )
    parser.add_argument(
        "--wandb_offline",
        action="store_true",
        help="Run Weights & Biases in offline mode.",
    )

    return parser


