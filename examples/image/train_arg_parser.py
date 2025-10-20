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
        help="Symmetric term coefficient for discrete sampling (mixture or metric-induced).",
    )
    parser.add_argument(
        "--sym_func",
        action="store_true",
        help="Use a fixed function for the symmetric term in the discrete sampler.",
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
        "--mi_init_metric_from_checkpoint",
        default="",
        type=str,
        help=(
            "Path to checkpoint (.pt/.pth) to initialize the learnable metric from. "
            "Only loads the metric codes, not the model or optimizer. "
            "Use with --mi_freeze_metric to freeze the loaded metric."
        ),
    )
    parser.add_argument(
        "--mi_freeze_metric",
        action="store_true",
        help=(
            "Freeze the learnable metric (set requires_grad=False). "
            "Typically used with --mi_init_metric_from_checkpoint to load a pre-trained metric "
            "and only train the UNet. This tests if UNet can adapt to a fixed metric geometry."
        ),
    )
    parser.add_argument(
        "--mi_eval_use_raw_metric",
        action="store_true",
        help=(
            "Use raw student metric (instead of EMA teacher) during evaluation. "
            "By default, evaluation uses EMA metric if available (smoother, more stable). "
            "Set this flag to use the raw student metric for evaluation (actual trained version)."
        ),
    )
    parser.add_argument(
        "--mi_eval_cache_lut",
        action="store_true",
        help=(
            "Precompute and cache learnable LUT distance tables during evaluation. "
            "This speeds up sampling when learnable LUT embeddings have dim > 1; "
            "training continues to recompute distances each step."
        ),
    )

    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="start epoch (used when resumed from checkpoint)",
    )
    # Display-only epoch mapping (does NOT affect training control flow)
    parser.add_argument(
        "--epoch_display_offset",
        default=None,
        type=int,
        help=(
            "Additive offset for display epoch: display_epoch = epoch + offset. "
            "This only affects logs/metrics/optional symlinks, not sampler, eval triggers, or resume."
        ),
    )
    parser.add_argument(
        "--force_display_start_epoch",
        default=None,
        type=int,
        help=(
            "Override to force the first displayed epoch after resume to this value. "
            "Effective offset will be computed as (force_display_start_epoch - start_epoch)."
        ),
    )
    parser.add_argument(
        "--save_display_epoch_symlinks",
        action="store_true",
        help=(
            "When saving checkpoints, also create a symlink named with the display epoch number "
            "(e.g., checkpoint-4499.pth -> checkpoint-5499.pth)."
        ),
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
    parser.add_argument(
        "--save_eval_gif",
        action="store_true",
        help="Save GIF animations of sampling trajectories during evaluation.",
    )
    parser.add_argument(
        "--eval_gif_max_batch",
        default=8,
        type=int,
        help="Maximum number of samples to tile in an evaluation GIF.",
    )
    parser.add_argument(
        "--eval_gif_stride",
        default=16,
        type=int,
        help="Stride applied when subsampling trajectory frames for GIF logging.",
    )
    parser.add_argument(
        "--eval_gif_fps",
        default=8,
        type=int,
        help="Playback speed (frames per second) for saved evaluation GIFs.",
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
        "--cosine_attention",
        action="store_true",
        help="L2-normalize Q and K before computing attention logits (cosine attention).",
    )
    parser.add_argument(
        "--no_cosine_attention",
        action="store_false",
        dest="cosine_attention",
        help="Disable cosine attention (default).",
    )
    parser.set_defaults(cosine_attention=False)

    parser.add_argument(
        "--head_weight_norm",
        action="store_true",
        help="Apply per-forward RMS normalization to the output head weights.",
    )
    parser.add_argument(
        "--no_head_weight_norm",
        action="store_false",
        dest="head_weight_norm",
        help="Disable output head weight normalization (default).",
    )
    parser.set_defaults(head_weight_norm=False)

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
        choices=["lp", "euclidean", "cosine"],
        type=str,
        help="Metric to use for the metric-induced path.",
    )
    parser.add_argument(
        "--mi_learnable_lut",
        action="store_true",
        help="Enable a learnable per-channel scalar LUT inside the metric-induced path.",
    )
    parser.add_argument(
        "--mi_lut_num_channels",
        default=3,
        type=int,
        help="Number of channels (e.g., 3 for RGB) when using a learnable LUT.",
    )
    parser.add_argument(
        "--mi_lut_emb_dim",
        default=1,
        type=int,
        help="Embedding dimension for learnable LUT (1=scalar, >1=vector embeddings).",
    )
    parser.add_argument(
        "--mi_lut_share_channels",
        action="store_true",
        help="Share a single LUT across all channels (instead of one per channel).",
    )
    parser.add_argument(
        "--mi_lut_renorm_init_norm",
        action="store_true",
        help=(
            "Renormalize LUT weights to the initialization Frobenius norm every forward pass. "
            "Preserves gradients and keeps the effective geometry aligned with the linear init."
        ),
    )
    parser.add_argument(
        "--mi_lut_bounded_residual_scale",
        action="store_true",
        help=(
            "Use bounded residual scale parameterization: s = s_0 * (1 + ε * tanh(c)). "
            "c is learnable per channel, L2 penalty keeps it near 0. "
            "Trust region approach prevents extreme scale changes."
        ),
    )
    parser.add_argument(
        "--mi_lut_scale_baseline",
        default=1.0,
        type=float,
        help="Baseline scale s_0 in bounded residual parameterization.",
    )
    parser.add_argument(
        "--mi_lut_scale_epsilon",
        default=0.25,
        type=float,
        help="Maximum deviation ε in bounded residual parameterization (s ∈ [s_0*(1-ε), s_0*(1+ε)]).",
    )
    parser.add_argument(
        "--mi_lut_scale_penalty_weight",
        default=0.01,
        type=float,
        help="L2 penalty weight on scale parameter c to keep it near 0.",
    )
    parser.add_argument(
        "--mi_use_normalized_distance",
        action="store_true",
        help=(
            "Use normalized distance: \tilde d = ||E[v]-E[x_1]||_2 / \sqrt{m} "
            "where m is the embedding dimension."
        ),
    )
    parser.add_argument(
        "--mi_lut_cosine_scale",
        default=1.0,
        type=float,
        help=(
            "Scale factor s applied to cosine LUT distance (dist = s * (1 - cos)). "
            "Only used when --mi_metric=cosine."
        ),
    )
    parser.add_argument(
        "--mi_lut_cosine_calibrate",
        action="store_true",
        help="Calibrate cosine LUT scale against baseline distances after load (optional).",
    )
    parser.add_argument(
        "--mi_lut_cosine_calibrate_mode",
        default="neighbor",
        choices=("neighbor", "median"),
        help=(
            "Cosine scale calibration mode. 'neighbor' matches adjacent-token distances "
            "(recommended); 'median' matches the full-table medians (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--mi_lut_cosine_t_mid",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7],
        help=(
            "One or more reference t values in (0,1) used when calibrating the cosine scale. "
            "The first value sets the applied scale; all values are logged for diagnostics. "
            "Default: 0.3 0.5 0.7."
        ),
    )
    parser.add_argument(
        "--mi_lut_cosine_entropy_match",
        action="store_true",
        help=(
            "Match cosine LUT scale to a weighted baseline conditional-entropy target using bisection. "
            "Runs after warm start / distance calibration."
        ),
    )
    parser.add_argument(
        "--mi_lut_cosine_entropy_t_mid",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7],
        help=(
            "Reference t values for entropy matching (default: 0.3 0.5 0.7). "
            "Weights are supplied via --mi_lut_cosine_entropy_weights."
        ),
    )
    parser.add_argument(
        "--mi_lut_cosine_entropy_weights",
        nargs="+",
        type=float,
        default=[0.2, 0.6, 0.2],
        help=(
            "Weights (non-negative) for each t_mid during entropy matching. "
            "If a single value is provided, it is broadcast to all t_mid entries."
        ),
    )
    parser.add_argument(
        "--mi_lut_cosine_entropy_tol",
        type=float,
        default=0.01,
        help="Relative tolerance |H_cos - H_base| / H_base for entropy matching (default: 0.01).",
    )
    parser.add_argument(
        "--mi_lut_cosine_entropy_max_iter",
        type=int,
        default=12,
        help="Maximum bisection iterations for entropy matching (default: 12).",
    )
    parser.add_argument(
        "--mi_lut_force_warm_start",
        action="store_true",
        help=(
            "Force cosine LUT warm start even when resuming from a checkpoint. "
            "Useful when finetuning a 1D baseline on higher-dimensional cosine geometry."
        ),
    )
    parser.add_argument(
        "--mi_freeze_lut",
        action="store_true",
        help="Freeze the learnable LUT parameters (requires_grad=False).",
    )
    parser.add_argument(
        "--mi_lut_lr_scale",
        default=0.1,
        type=float,
        help="Learning rate scale applied to learnable LUT parameters.",
    )
    parser.add_argument(
        "--mi_lut_weight_decay",
        default=1e-4,
        type=float,
        help="Weight decay applied to learnable LUT parameters.",
    )
    parser.add_argument(
        "--mi_lut_kl_weight",
        default=0.0,
        type=float,
        help=(
            "KL trust-region weight to keep learned LUT-induced p_t close to the baseline geometry. "
            "Computes KL(p_base || p_learned) at the sampled t and adds it to the loss."
        ),
    )
    parser.add_argument(
        "--mi_lut_kl_t_lo",
        default=None,
        type=float,
        help=(
            "Optional lower t-bound for LUT KL penalty (only apply when t in [lo, hi])."
        ),
    )
    parser.add_argument(
        "--mi_lut_kl_t_hi",
        default=None,
        type=float,
        help=(
            "Optional upper t-bound for LUT KL penalty (only apply when t in [lo, hi])."
        ),
    )
    parser.add_argument(
        "--mi_lut_init_method",
        default="linear",
        choices=["linear", "small_noise_qr"],
        type=str,
        help=(
            "Initialization method for learnable LUT weights. "
            "'linear': Standard linear spacing with small noise (default). "
            "'small_noise_qr': QR decomposition with controlled noise for high-dim embeddings."
        ),
    )
    parser.add_argument(
        "--mi_lut_init_noise_scale",
        default=0.01,
        type=float,
        help=(
            "Noise scale for small_noise_qr initialization. "
            "Controls perturbation strength in higher dimensions (σ parameter). "
            "Recommended: 0.01-0.05 for balanced warm start vs orthogonality."
        ),
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
        "--mi_beta_log_schedule",
        action="store_true",
        help=(
            "When set, dump β(t) snapshots during evaluation for comparison against the fixed schedule "
            "and log the curves to wandb if enabled."
        ),
    )
    parser.add_argument(
        "--mi_beta_log_points",
        default=256,
        type=int,
        help="Number of time samples to evaluate when exporting β(t) snapshots.",
    )
    parser.add_argument(
        "--mi_logbeta_min",
        default=None,
        type=float,
        help=(
            "Lower bound for uniform log-β sampling when using the exponential spline schedule. "
            "Defaults to log β evaluated at t=mi_t_eps if unspecified."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_max",
        default=None,
        type=float,
        help=(
            "Upper bound for uniform log-β sampling when using the exponential spline schedule. "
            "Defaults to log β evaluated at t=1-mi_t_eps if unspecified."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_reg_delta_weight",
        default=0.0,
        type=float,
        help=(
            "Weight for the first-order smoothness penalty on log β knots (∑(Δℓ/Δs)^2)."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_reg_delta2_weight",
        default=0.0,
        type=float,
        help=(
            "Weight for the second-order smoothness penalty on log β knots (∑(Δ^2ℓ/Δs^2)^2)."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_reg_power",
        default=0.0,
        type=float,
        help=(
            "Power-law exponent for reweighting log β knot penalties; positive values emphasize central knots."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_endpoint_weight",
        default=0.0,
        type=float,
        help=(
            "Weight for penalizing the spline endpoint slopes to prevent exploding dℓ/dt near t ∈ {0,1}."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_reg_anneal_steps",
        default=0,
        type=int,
        help=(
            "Number of optimizer steps to linearly anneal the log β smoothness penalties towards zero."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_trunc_t",
        default=None,
        type=float,
        help=(
            "Optional truncation applied to the sampling domain; restrict t to [mi_logbeta_trunc_t, 1-mi_logbeta_trunc_t] "
            "when drawing log-β proposals (must be greater than mi_t_eps)."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_band_t_lo",
        default=None,
        type=float,
        help=(
            "Lower cutoff in t-space for computing the active log-β sampling range."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_band_t_hi",
        default=None,
        type=float,
        help=(
            "Upper cutoff in t-space for computing the active log-β sampling range."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_mis_alpha",
        default=0.3,
        type=float,
        help=(
            "Mixture coefficient α for MIS between uniform-t and log-β proposals. "
            "Set to 0 to disable; recommended range is [0.1, 0.5]."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_eval_alpha",
        default=None,
        type=float,
        help=(
            "Mixture coefficient α for evaluation grid when using uniform_logbeta. "
            "If None, uses the training value from --mi_logbeta_mis_alpha. "
            "Set to 0 for pure log-β sampling, 0.5 for balanced 50-50 mix, "
            "1.0 for pure uniform-t sampling."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_sampling_strategy",
        default="uniform",
        choices=["uniform", "log_normal_broad", "log_normal_focused"],
        type=str,
        help=(
            "Sampling strategy in log-β space for uniform_logbeta grid. "
            "'uniform': uniform distribution in log-β. "
            "'log_normal_broad': Gaussian centered on full interval (μ=0, σ=2.5 for [-5,5]). "
            "'log_normal_focused': Gaussian concentrated on informative region (auto-computed)."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_lognormal_mu",
        default=None,
        type=float,
        help=(
            "Mean (μ) of log-normal distribution in log-β space. "
            "If None: auto-computed based on strategy. "
            "  - 'broad': μ = (ℓ_max + ℓ_min) / 2 (center of full interval). "
            "  - 'focused': μ = center of informative region in log-β. "
            "Manual override: set explicit value (e.g., 0.0 for β=1)."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_lognormal_sigma",
        default=None,
        type=float,
        help=(
            "Standard deviation (σ) of log-normal distribution in log-β space. "
            "If None: auto-computed based on strategy. "
            "  - 'broad': σ = (ℓ_max - ℓ_min) / 4 (4σ covers full interval). "
            "  - 'focused': σ = (informative_width) / 2 (1σ covers informative). "
            "Manual override: set explicit value (e.g., 2.5 for broad coverage)."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_informative_H_min_ratio",
        default=0.2,
        type=float,
        help=(
            "Lower bound of informative region as ratio of H_max. "
            "Informative region: H ∈ [ratio_min × H_max, ratio_max × H_max]. "
            "Used for auto-computing focused log-normal parameters."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_informative_H_max_ratio",
        default=0.8,
        type=float,
        help=(
            "Upper bound of informative region as ratio of H_max. "
            "Informative region: H ∈ [ratio_min × H_max, ratio_max × H_max]. "
            "Used for auto-computing focused log-normal parameters."
        ),
    )
    parser.add_argument(
        "--mi_logbeta_use_is",
        action="store_true",
        help=(
            "Enable importance sampling (IS) for log-β proposals. "
            "When disabled (default), samples are drawn from log-β distribution "
            "without reweighting (logbeta_weights = None). "
            "When enabled, applies IS weights: w(β) = 1/q(β)."
        ),
    )
    parser.add_argument(
        "--mi_infer_grid",
        default="uniform_t",
        choices=["uniform_t", "uniform_logbeta"],
        type=str,
        help=(
            "Time grid used by the metric-induced evaluator. "
            "'uniform_logbeta' builds a grid uniform in log β."
        ),
    )
    parser.add_argument(
        "--mi_infer_steps",
        default=None,
        type=int,
        help=(
            "Number of intervals for the inference grid. "
            "Defaults to --discrete_fm_steps when unspecified."
        ),
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
        "--mi_beta_kl_avg_window",
        default=1,
        type=int,
        help="Number of optimizer updates to average the schedule KL before adapting the penalty weight.",
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
        "--mi_geodesic_energy_weight",
        default=0.0,
        type=float,
        help=(
            "Weight (λ) for geodesic energy regularization in metric-induced path training. "
            "Penalizes ||v_t||²_M where v_t is the velocity in the learned metric embedding space. "
            "This encourages smoother, lower-energy trajectories that follow geodesics under the "
            "learned Mahalanobis metric, reducing entropy of intermediate distributions p(x_t|x_1,t). "
            "Recommended values: 0.01-0.1 (start conservative, e.g., 0.05). "
            "Expected effects: 10-20%% entropy reduction (target: 4.3 → 3.5-4.0), improved FID. "
            "Set to 0.0 to disable. The penalty is scaled by the current metric interpolation λ, "
            "so it only applies when the learned metric is active (λ > 0)."
        ),
    )
    parser.add_argument(
        "--diag_enable",
        action="store_true",
        help="Run stability diagnostics (weights, activations, attention) during evaluation.",
    )
    parser.add_argument(
        "--diag_checkpoint",
        action="append",
        default=[],
        help="Checkpoint path to include in diagnostics. Provide multiple times for a sequence.",
    )
    parser.add_argument(
        "--diag_batch_size",
        default=32,
        type=int,
        help="Batch size used for activation diagnostics hooks.",
    )
    parser.add_argument(
        "--diag_time",
        default=0.5,
        type=float,
        help="Time value in [0,1] used when probing activations for diagnostics.",
    )
    parser.add_argument(
        "--diag_power_iters",
        default=8,
        type=int,
        help="Number of power-iteration steps for spectral norm estimation.",
    )
    parser.add_argument(
        "--diag_output_dir",
        default=None,
        type=str,
        help="Optional override for the diagnostics CSV output directory.",
    )
    parser.add_argument(
        "--mi_beta_lr_scale",
        default=1.0,
        type=float,
        help="Relative learning rate scale applied to the β schedule parameter group.",
    )
    parser.add_argument(
        "--mi_freeze_beta_schedule",
        action="store_true",
        help=(
            "Freeze the β schedule parameters during training (no gradient updates). "
            "When enabled, schedule EMA and KL controller are disabled. "
            "Use this to train only the UNet with a fixed schedule."
        ),
    )
    parser.add_argument(
        "--mi_learnable_metric",
        action="store_true",
        help="Enable a learnable Mahalanobis metric for the metric-induced path.",
    )
    parser.add_argument(
        "--mi_metric_dim",
        default=8,
        type=int,
        help="Dimension of the learnable Mahalanobis token codes when enabled.",
    )
    parser.add_argument(
        "--mi_metric_diag_eps",
        default=1e-4,
        type=float,
        help="Stability epsilon added to the Mahalanobis transform diagonal.",
    )
    parser.add_argument(
        "--mi_metric_lr_scale",
        default=0.1,
        type=float,
        help="Relative learning rate scale applied to the learnable metric parameter group.",
    )
    parser.add_argument(
        "--mi_metric_weight_decay",
        default=0.0,
        type=float,
        help="Weight decay (L2 regularization) applied to learnable metric parameters.",
    )
    parser.add_argument(
        "--mi_metric_use_ema",
        action="store_true",
        help="Track an EMA teacher of the learnable metric for KL regularization.",
    )
    parser.add_argument(
        "--mi_metric_ema_decay",
        default=0.999,
        type=float,
        help="Decay applied to the learnable metric EMA updates.",
    )
    parser.add_argument(
        "--mi_metric_eval_geometry",
        action="store_true",
        help=(
            "When using a metric-induced path, compute geometry agreement diagnostics "
            "(Spearman correlation, k-NN overlap, optional heatmaps) during evaluation."
        ),
    )
    parser.add_argument(
        "--mi_metric_eval_subset",
        default=0,
        type=int,
        help=(
            "Optional number of tokens to include when probing metric geometry. "
            "Set to 0 to evaluate the full vocabulary."
        ),
    )
    parser.add_argument(
        "--mi_metric_eval_pair_samples",
        default=0,
        type=int,
        help=(
            "Optional number of off-diagonal pairs sampled when computing Spearman "
            "correlation. Set to 0 to use all pairs."
        ),
    )
    parser.add_argument(
        "--mi_metric_eval_knn_k",
        default=5,
        type=int,
        help="Neighborhood size k used for the k-NN overlap diagnostic.",
    )
    parser.add_argument(
        "--mi_metric_eval_seed",
        default=0,
        type=int,
        help="Seed for any random subsampling performed by the metric geometry probes.",
    )
    parser.add_argument(
        "--mi_metric_eval_heatmap",
        action="store_true",
        help=(
            "Export baseline/learned/difference heatmaps for the probed token subset "
            "during metric geometry evaluation (requires matplotlib)."
        ),
    )
    parser.add_argument(
        "--mi_metric_eval_heatmap_subset",
        default=64,
        type=int,
        help=(
            "Maximum number of tokens visualized in metric heatmaps. "
            "Set to 0 to reuse the evaluation subset size."
        ),
    )
    parser.add_argument(
        "--mi_metric_detailed_analysis",
        action="store_true",
        help=(
            "Run comprehensive metric analysis during evaluation including "
            "t-SNE/UMAP visualizations, distance distributions, and nearest neighbors. "
            "This provides deep insights into what the metric learned."
        ),
    )
    parser.add_argument(
        "--mi_metric_use_umap",
        action="store_true",
        default=False,
        help=(
            "Use UMAP for dimensionality reduction in detailed metric analysis. "
            "UMAP is faster and better preserves global structure than t-SNE. "
            "Enabled by default when --mi_metric_detailed_analysis is set."
        ),
    )
    parser.add_argument(
        "--mi_metric_use_tsne",
        action="store_true",
        default=False,
        help=(
            "Use t-SNE for dimensionality reduction in detailed metric analysis. "
            "t-SNE emphasizes local structure and clusters. "
            "Enabled by default when --mi_metric_detailed_analysis is set."
        ),
    )
    parser.add_argument(
        "--mi_metric_interp_start",
        default=0.0,
        type=float,
        help="Initial interpolation weight between the fixed Lp distance and the learned metric.",
    )
    parser.add_argument(
        "--mi_metric_lr_decay_start",
        default=3500,
        type=int,
        help=(
            "Epoch to start decaying the metric learning rate. "
            "This allows the metric to stabilize while UNet continues adapting. "
            "Default: 3500 (75%% of 4500 epoch training)."
        ),
    )
    parser.add_argument(
        "--mi_metric_lr_decay_mode",
        default="cosine",
        type=str,
        choices=["cosine", "exp", "step", "none"],
        help=(
            "Learning rate decay schedule for the metric parameters. "
            "cosine: smooth cosine annealing (recommended), "
            "exp: exponential decay, "
            "step: step-wise decay at fixed epochs, "
            "none: no decay. "
            "Default: cosine."
        ),
    )
    parser.add_argument(
        "--mi_metric_lr_decay_end_ratio",
        default=0.1,
        type=float,
        help=(
            "Final learning rate ratio for metric parameters at the end of training. "
            "With cosine/exp decay, the metric LR will decay from base_lr to base_lr*end_ratio. "
            "Default: 0.1 (decay to 10%%)."
        ),
    )
    parser.add_argument(
        "--mi_metric_interp_end",
        default=1.0,
        type=float,
        help="Final interpolation weight between the fixed Lp distance and the learned metric.",
    )
    parser.add_argument(
        "--mi_metric_interp_anneal_steps",
        default=0,
        type=int,
        help="Number of optimizer steps used to anneal the metric interpolation weight.",
    )
    parser.add_argument(
        "--mi_metric_interp_schedule",
        default="cosine",
        choices=["quadratic", "linear", "cosine"],
        help="Schedule shape used to anneal the metric interpolation weight.",
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
    parser.add_argument(
        "--mi_gumbel_tau_start",
        default=None,
        type=float,
        help=(
            "Optional warm-start temperature for Gumbel-Softmax annealing. "
            "Defaults to --mi_gumbel_tau when unspecified."
        ),
    )
    parser.add_argument(
        "--mi_gumbel_tau_end",
        default=None,
        type=float,
        help=(
            "Optional final temperature for Gumbel-Softmax annealing. "
            "Defaults to --mi_gumbel_tau when unspecified."
        ),
    )
    parser.add_argument(
        "--mi_gumbel_tau_anneal_steps",
        default=0,
        type=int,
        help=(
            "Number of optimizer steps used to anneal the Gumbel temperature. "
            "Set to zero to keep a fixed temperature."
        ),
    )
    parser.add_argument(
        "--mi_gumbel_tau_schedule",
        default="quadratic",
        choices=["quadratic", "linear", "cosine"],
        help=(
            "Shape of the annealing curve applied between the start and end "
            "Gumbel temperatures."
        ),
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


