# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Iterable, Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch import Tensor

logger = logging.getLogger(__name__)


def _inv_softplus_tensor(x: Tensor) -> Tensor:
    """Stable inverse softplus used for positive-diagonal initialization."""

    return torch.log(torch.expm1(x))


class LearnableScalarLUT(nn.Module):
    """Per-channel learnable embedding lookup table.

    This module stores C independent embedding lookup tables, each with vocab_size
    entries of dimension emb_dim. It is designed for the metric-induced probability
    path where the distance is defined by Lp norm between embeddings.
    
    Shape: weight [num_channels, vocab_size, emb_dim]
    - When emb_dim=1: behaves as scalar LUT (backward compatible)
    - When emb_dim>1: uses vector embeddings with Lp distance
    """

    def __init__(
        self,
        *,
        num_channels: int,
        vocab_size: int,
        emb_dim: int = 1,
        embed_range: str,
        device: Optional[torch.device],
        dtype: torch.dtype,
        lp_order: float = 1.0,
        renormalize_to_init_norm: bool = False,
        renorm_eps: float = 1e-12,
        bounded_residual_scale: bool = False,
        scale_baseline: float = 1.0,
        scale_epsilon: float = 0.25,
        init_method: str = "linear",
        init_noise_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if emb_dim <= 0:
            raise ValueError("emb_dim must be positive")
        if embed_range not in {"pm1", "unit"}:
            raise ValueError("embed_range must be 'pm1' or 'unit'")

        self.num_channels = int(num_channels)
        self.vocab_size = int(vocab_size)
        self.emb_dim = int(emb_dim)
        self.embed_range = embed_range
        self.lp_order = float(lp_order)

        # Set initialization parameters before calling _init_weight
        self.init_method = str(init_method)
        self.init_noise_scale = float(init_noise_scale)
        if self.init_method == "small_noise_qr" and self.emb_dim > self.vocab_size:
            raise ValueError(
                "small_noise_qr initialization requires emb_dim <= vocab_size "
                f"(got emb_dim={self.emb_dim}, vocab_size={self.vocab_size})"
            )

        weight = self._init_weight()
        if device is not None:
            weight = weight.to(device=device)
        weight = weight.to(dtype=dtype)
        self.weight = nn.Parameter(weight)
        
        # Compute base norm (without noise) PER CHANNEL for renormalization target
        # Shape: [num_channels] - each channel has its own target norm
        base_weight = self._linear_init_base()
        if device is not None:
            base_weight = base_weight.to(device=device)
        base_weight = base_weight.to(dtype=dtype)
        # Compute norm per channel: norm over dims (vocab_size, emb_dim)
        base_norm_per_channel = torch.linalg.vector_norm(base_weight, dim=(1, 2))  # [num_channels]
        self.register_buffer("_base_fro_norm_per_channel", base_norm_per_channel, persistent=False)
        
        self.renormalize_to_init_norm = bool(renormalize_to_init_norm)
        self.renorm_eps = float(renorm_eps)
        
        # Bounded residual scale parameterization
        self.bounded_residual_scale = bool(bounded_residual_scale)
        self.scale_baseline = float(scale_baseline)  # s_0
        self.scale_epsilon = float(scale_epsilon)    # ε
        
        if self.bounded_residual_scale:
            # Learnable scale parameter c per channel: [num_channels]
            # Initialized to 0 (tanh(0) = 0, so s = s_0 initially)
            self.scale_c = nn.Parameter(torch.zeros(num_channels, dtype=dtype))
        else:
            self.scale_c = None

    def _init_weight(self) -> Tensor:
        """Initialize LUT weights using the selected initialization method."""
        if self.init_method == "linear":
            return self._linear_init()
        elif self.init_method == "small_noise_qr":
            return self._small_noise_qr_init()
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

    def _linear_init(self) -> Tensor:
        """Initialize LUT with linearly spaced values along each dimension.
        
        Returns:
            Tensor of shape [num_channels, vocab_size, emb_dim]
        """
        # Initialize each dimension independently with linspace + small noise
        weight = torch.zeros(self.num_channels, self.vocab_size, self.emb_dim)
        
        for d in range(self.emb_dim):
            if self.embed_range == "pm1":
                base = torch.linspace(-1.0, 1.0, steps=self.vocab_size)
            else:
                base = torch.linspace(0.0, 1.0, steps=self.vocab_size)
            
            # Repeat base pattern for all channels
            base_repeated = base.repeat(self.num_channels, 1)  # [C, V]
            
            # Add small Gaussian noise to break channel symmetry
            # σ = 1e-3 is ~7.8x smaller than linear step size (≈0.00784 for V=256)
            # This preserves monotonicity (P(inversion) ≈ 0) while decorrelating RGB
            noise = torch.randn_like(base_repeated) * 1e-3
            
            weight[:, :, d] = base_repeated + noise
        
        return weight
    
    def _linear_init_base(self) -> Tensor:
        """Initialize base LUT (without noise) for renormalization target.
        
        Returns:
            Tensor of shape [num_channels, vocab_size, emb_dim]
        """
        weight = torch.zeros(self.num_channels, self.vocab_size, self.emb_dim)
        
        for d in range(self.emb_dim):
            if self.embed_range == "pm1":
                base = torch.linspace(-1.0, 1.0, steps=self.vocab_size)
            else:
                base = torch.linspace(0.0, 1.0, steps=self.vocab_size)
            
            # Repeat base pattern for all channels (no noise)
            base_repeated = base.repeat(self.num_channels, 1)  # [C, V]
            weight[:, :, d] = base_repeated
        
        return weight

    def _small_noise_qr_init(self) -> Tensor:
        """Initialize LUT with small noise QR decomposition for orthogonal warm start.
        
        This method provides an orthogonal initialization that balances:
        - Orthogonality: QR decomposition ensures dimension-wise independence
        - Warm start: Small noise prevents exact orthogonality for gradient flow
        - Stability: Maintains consistent norms across channels and dimensions
        
        Returns:
            Tensor of shape [num_channels, vocab_size, emb_dim]
        """
        # Step 1: Compute baseline unit vector (linear spacing normalized)
        if self.embed_range == "pm1":
            base_values = torch.linspace(-1.0, 1.0, steps=self.vocab_size)
        else:
            base_values = torch.linspace(0.0, 1.0, steps=self.vocab_size)
        
        # Normalize to unit vector for each dimension
        base_unit = base_values / torch.linalg.vector_norm(base_values, ord=2)
        
        # Step 2: Build matrix A with small noise for QR decomposition
        # A will be [vocab_size, emb_dim] - we want orthogonal columns
        A = torch.zeros(self.vocab_size, self.emb_dim)
        
        # First column: baseline unit vector
        A[:, 0] = base_unit
        
        # Remaining columns: add small noise to break symmetry
        if self.emb_dim > 1:
            # Generate small orthogonal noise
            noise_scale = self.init_noise_scale
            for d in range(1, self.emb_dim):
                # Start with small random perturbation of the baseline
                noise = torch.randn(self.vocab_size) * noise_scale
                A[:, d] = base_unit + noise
        
        # Step 3: Apply QR decomposition for orthogonality
        Q, R = torch.linalg.qr(A)
        
        # Step 4: Scale to target norm (same as linear init for consistency)
        # Target norm should match the linear initialization norm
        target_norm = torch.linalg.vector_norm(base_values, ord=2)
        current_norm = torch.linalg.vector_norm(Q, dim=0)  # norm per column
        scale_factors = target_norm / (current_norm + 1e-8)
        Q_scaled = Q * scale_factors.unsqueeze(0)  # broadcast to [vocab_size, emb_dim]
        
        # Step 5: Replicate across channels (with small channel-wise noise)
        weight = torch.zeros(self.num_channels, self.vocab_size, self.emb_dim)
        for c in range(self.num_channels):
            if c == 0:
                # First channel: use the QR result directly
                weight[c] = Q_scaled
            else:
                # Other channels: add small channel-specific noise
                channel_noise = torch.randn_like(Q_scaled) * (noise_scale * 0.1)
                weight[c] = Q_scaled + channel_noise
        
        return weight

    @torch.no_grad()
    def reset_parameters(self) -> None:
        init_weight = self._init_weight().to(device=self.weight.device, dtype=self.weight.dtype)
        self.weight.copy_(init_weight)
        
        # Recompute base norm per channel (without noise)
        base_weight = self._linear_init_base().to(device=self.weight.device, dtype=self.weight.dtype)
        base_norm_per_channel = torch.linalg.vector_norm(base_weight, dim=(1, 2))  # [num_channels]
        self._base_fro_norm_per_channel.copy_(base_norm_per_channel)
        
        # Reset scale parameter c to 0 if using bounded residual
        if self.bounded_residual_scale and self.scale_c is not None:
            self.scale_c.zero_()

    def forward(self) -> Tensor:
        """Return LUT weights, optionally renormalized to base norm PER CHANNEL.
        
        When renormalize_to_init_norm=True:
            - Renormalizes EACH channel independently to match its base (no-noise) norm
            - Each channel maintains its own geometry (single-channel norm ≈ 9.27)
            - Preserves gradients (scaling is differentiable)
            - Keeps noise injection benefits while maintaining consistent geometry
        
        When bounded_residual_scale=True:
            - Uses bounded residual parameterization: s = s_0 * (1 + ε * tanh(c))
            - c is learnable per channel, L2 penalty keeps it near 0
            - s_0 is baseline scale, ε controls maximum deviation
            - Trust region approach prevents extreme scale changes
        
        Returns:
            Tensor of shape [num_channels, vocab_size, emb_dim]
        """
        if self.bounded_residual_scale and self.scale_c is not None:
            # Bounded residual scale: s = s_0 * (1 + ε * tanh(c))
            # c is per-channel learnable parameter initialized to 0
            scale = self.scale_baseline * (1.0 + self.scale_epsilon * torch.tanh(self.scale_c))
            # Apply per-channel scaling: scale.view(C, 1, 1) * weight[C, V, D]
            return self.weight * scale.view(-1, 1, 1)
        
        if not self.renormalize_to_init_norm:
            return self.weight
        
        # Renormalize each channel independently to its base norm
        # target: [num_channels] - target norm for each channel
        # current: [num_channels] - current norm for each channel
        target = self._base_fro_norm_per_channel.to(device=self.weight.device, dtype=self.weight.dtype)
        current = torch.linalg.vector_norm(self.weight, dim=(1, 2))  # [num_channels]
        scale = target / (current + self.renorm_eps)  # [num_channels]
        
        # Apply per-channel scaling: scale.view(C, 1, 1) * weight[C, V, D]
        # Broadcasting: [C, 1, 1] * [C, V, D] -> [C, V, D]
        return self.weight * scale.view(-1, 1, 1)

    def extra_repr(self) -> str:
        return (
            f"num_channels={self.num_channels}, vocab_size={self.vocab_size}, "
            f"emb_dim={self.emb_dim}, embed_range='{self.embed_range}', "
            f"lp_order={self.lp_order}, bounded_residual_scale={self.bounded_residual_scale}, "
            f"init_method='{self.init_method}', init_noise_scale={self.init_noise_scale}"
        )


class MahalanobisTokenMetric(nn.Module):
    """Learnable PSD metric with unit-Frobenius retraction constraint.
    
    Uses reparameterization to enforce unit Frobenius norm ||Z||_F = 1:
    - codes_raw: learnable parameters [vocab_size, metric_dim]
    - codes (property): per-row normalized codes = normalize(codes_raw, dim=-1)
    - Z = codes @ L^T is normalized to ||Z||_F = 1 via stop-gradient retraction
    
    This removes scale ambiguity with β schedule (β controls all distance scaling).
    Similar to weight normalization: separates direction (learnable) from magnitude (controlled by β).
    """

    def __init__(
        self,
        vocab_size: int,
        metric_dim: int,
        *,
        init_codes: Optional[Tensor] = None,
        diag_eps: float = 1e-4,
        target_norm: float = 1.0,
        use_reparameterization: bool = False,  # Disabled by default for backward compatibility
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if metric_dim <= 0:
            raise ValueError("metric_dim must be positive")
        if diag_eps < 0.0:
            raise ValueError("diag_eps must be non-negative")

        self.vocab_size = int(vocab_size)
        self.metric_dim = int(metric_dim)
        self.diag_eps = float(diag_eps)
        self.target_norm = float(target_norm)
        self.use_reparameterization = bool(use_reparameterization)

        if self.use_reparameterization:
            # Reparameterization mode: unit-Frobenius constraint by construction
            # Only codes_raw is learnable; Frobenius norm enforced via retraction
            self.codes_raw = nn.Parameter(torch.zeros(self.vocab_size, self.metric_dim))
            # No log_scale: scale ambiguity removed (β schedule controls all scaling)
        else:
            # Legacy mode: direct parameter (requires post-update projection)
            # Directly register parameter to avoid property getter trigger
            self._parameters['codes'] = nn.Parameter(torch.zeros(self.vocab_size, self.metric_dim))
        
        self._lower_params = nn.Parameter(torch.zeros(self.metric_dim, self.metric_dim))

        self.reset_parameters(init_codes=init_codes)

    @torch.no_grad()
    def reset_parameters(self, init_codes: Optional[Tensor] = None) -> None:
        if init_codes is not None:
            if init_codes.shape != (self.vocab_size, self.metric_dim):
                raise ValueError(
                    "init_codes must have shape (vocab_size, metric_dim)"
                )
            if self.use_reparameterization:
                # Initialize codes_raw such that normalized version equals init_codes
                # Compute current norm of init_codes
                init_norm = torch.linalg.norm(init_codes, ord='fro')
                if init_norm > 1e-8:
                    # Simply normalize per-row (Frobenius constraint applied in distance computation)
                    self.codes_raw.copy_(F.normalize(init_codes, dim=-1))
                else:
                    # Fallback: small random initialization
                    std = 1.0 / math.sqrt(self.vocab_size * self.metric_dim)
                    self.codes_raw.normal_(0, std)
            else:
                # Legacy mode: set parameter directly
                param = self._parameters.get('codes')
                if param is not None:
                    param.copy_(init_codes)
        else:
            # Default initialization: unit-Frobenius constraint
            if self.use_reparameterization:
                # Initialize so that E[||Z||_F] = 1 after Frobenius normalization
                # E[||Z||²_F] = vocab_size × metric_dim × σ²
                # Want E[||Z||_F] = 1 => σ = 1 / sqrt(vocab_size × metric_dim)
                std = 1.0 / math.sqrt(self.vocab_size * self.metric_dim)
                self.codes_raw.normal_(0, std)
            else:
                # Legacy mode: zero initialization
                param = self._parameters.get('codes')
                if param is not None:
                    param.zero_()

        self._lower_params.zero_()
        inv_sp_one = _inv_softplus_tensor(torch.ones(self.metric_dim, dtype=self._lower_params.dtype))
        torch.diagonal(self._lower_params).copy_(inv_sp_one)
    
    @property
    def codes(self) -> Tensor:
        """Return codes (learnable per-token embeddings).
        
        In reparameterization mode:
            codes = codes_raw (already normalized per-row during updates)
            Frobenius normalization is applied later in distance computation.
        
        In legacy mode:
            codes = self.codes (direct parameter)
        
        NOTE: In reparameterization mode, codes_raw is kept normalized after
        each gradient update via projection. Direct modifications to codes
        (via .mul_, .add_, etc.) will break normalization and should be avoided.
        Use optimizer updates instead.
        """
        if self.use_reparameterization:
            # Return codes_raw directly (already per-row normalized)
            return self.codes_raw
        else:
            # Legacy mode: return parameter directly (fallback for old checkpoints)
            param = self._parameters.get('codes')
            if param is None:
                raise RuntimeError("Legacy codes parameter not found")
            return param
    
    @codes.setter
    def codes(self, value: Tensor) -> None:
        """Allow setting codes (mainly for checkpoint loading).
        
        In reparameterization mode:
            Normalize and assign to codes_raw (Frobenius norm enforced at distance computation)
        In legacy mode:
            Sets codes parameter directly
        """
        if value.shape != (self.vocab_size, self.metric_dim):
            raise ValueError(f"codes must have shape ({self.vocab_size}, {self.metric_dim})")
        
        if self.use_reparameterization:
            with torch.no_grad():
                # Simply normalize per-row and assign
                # Frobenius normalization happens in pairwise_distance_table
                self.codes_raw.copy_(F.normalize(value, dim=-1))
        else:
            # Legacy mode: set parameter directly (fallback for old checkpoints)
            param = self._parameters.get('codes')
            if param is not None:
                param.copy_(value)

    def cholesky_factor(self) -> Tensor:
        lower = torch.tril(self._lower_params)
        diag = torch.diagonal(lower)
        positive_diag = F.softplus(diag) + self.diag_eps
        lower = lower - torch.diag_embed(diag) + torch.diag_embed(positive_diag)
        return lower

    def transformed_codes(self, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Tensor:
        """Return Z = C @ L^T (before Frobenius normalization).
        
        Note: Frobenius normalization ||Z||_F = 1 is applied in pairwise_distance_table()
        with stop-gradient to enable manifold optimization.
        
        In reparameterization mode (if enabled):
            - Per-row normalization enforced on-the-fly (to handle manual modifications)
            - Frobenius normalization happens at distance computation
        
        In legacy mode (default):
            - Direct parameter access (backward compatibility)
        """
        codes = self.codes  # In reparam mode (if enabled), this returns codes_raw
        if device is not None or dtype is not None:
            codes = codes.to(device=device or codes.device, dtype=dtype or codes.dtype)
        
        # In reparameterization mode, enforce per-row normalization on-the-fly
        # This handles cases where codes may have been modified directly (e.g., in tests)
        # Only apply if reparameterization is actually enabled
        if self.use_reparameterization:
            codes = F.normalize(codes, dim=-1)
        
        lower = self.cholesky_factor().to(device=codes.device, dtype=codes.dtype)
        Z = codes @ lower.T  # [K, d]
        return Z

    def pairwise_distance_table(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Compute pairwise distance table with unit-Frobenius normalization.
        
        Applies Frobenius normalization: Z = Z_raw / ||Z_raw||_F
        Gradient flows through both the norm and the normalized matrix.
        """
        transformed = self.transformed_codes(device=device, dtype=dtype)
        
        # Unit-Frobenius normalization (gradient flows through norm)
        fro_norm = torch.linalg.norm(transformed, ord='fro')  # gradient enabled
        transformed = transformed / (fro_norm + 1e-6)  # normalized to ||Z||_F = 1
        
        return torch.cdist(transformed, transformed, p=2.0)

    def get_extra_state(self) -> dict:
        return {
            "diag_eps": self.diag_eps,
            "target_norm": self.target_norm,
            "use_reparameterization": self.use_reparameterization,
        }

    def set_extra_state(self, state: dict) -> None:  # pragma: no cover - API requirement
        self.diag_eps = float(state.get("diag_eps", self.diag_eps))
        self.target_norm = float(state.get("target_norm", getattr(self, "target_norm", 1.0)))
        self.use_reparameterization = bool(state.get("use_reparameterization", getattr(self, "use_reparameterization", True)))

from flow_matching.path.beta_schedules import BetaSchedule
from flow_matching.path.path import ProbPath

from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import ConvexScheduler
from flow_matching.utils import expand_tensor_like, unsqueeze_to_match


class MixtureDiscreteProbPath(ProbPath):
    r"""The ``MixtureDiscreteProbPath`` class defines a factorized discrete probability path.

    This path remains constant at the source data point :math:`X_0` until a random time, determined by the scheduler, when it flips to the target data point :math:`X_1`.
    The scheduler determines the flip probability using the parameter :math:`\sigma_t`, which is a function of time `t`. Specifically, :math:`\sigma_t` represents the probability of remaining at :math:`X_0`, while :math:`1 - \sigma_t` is the probability of flipping to :math:`X_1`:

    .. math::

        P(X_t = X_0) = \sigma_t \quad \text{and} \quad  P(X_t = X_1) = 1 - \sigma_t,

    where :math:`\sigma_t` is provided by the scheduler.

    Example:

    .. code-block:: python

        >>> x_0 = torch.zeros((1, 3, 3))
        >>> x_1 = torch.ones((1, 3, 3))

        >>> path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))
        >>> result = path.sample(x_0, x_1, t=torch.tensor([0.1])).x_t
        >>> result
        tensor([[[0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0]]])

        >>> result = path.sample(x_0, x_1, t=torch.tensor([0.5])).x_t
        >>> result
        tensor([[[1.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0]]])

        >>> result = path.sample(x_0, x_1, t=torch.tensor([1.0])).x_t
        >>> result
        tensor([[[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]]])

    Args:
        scheduler (ConvexScheduler): The scheduler that provides :math:`\sigma_t`.
    """

    def __init__(self, scheduler: ConvexScheduler):
        assert isinstance(
            scheduler, ConvexScheduler
        ), "Scheduler for ConvexProbPath must be a ConvexScheduler."

        self.scheduler = scheduler

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> DiscretePathSample:
        r"""Sample from the affine probability path:
            | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`(\alpha_t,\sigma_t)`.
            | return :math:`X_0, X_1, t`, and :math:`X_t \sim p_t`.
        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            DiscretePathSample: a conditional sample at :math:`X_t ~ p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        sigma_t = self.scheduler(t).sigma_t

        sigma_t = expand_tensor_like(input_tensor=sigma_t, expand_to=x_1)

        source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t
        x_t = torch.where(condition=source_indices, input=x_0, other=x_1)

        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)

    def posterior_to_velocity(
        self, posterior_logits: Tensor, x_t: Tensor, t: Tensor
    ) -> Tensor:
        r"""Convert the factorized posterior to velocity.

        | given :math:`p(X_1|X_t)`. In the factorized case: :math:`\prod_i p(X_1^i | X_t)`.
        | return :math:`u_t`.

        Args:
            posterior_logits (Tensor): logits of the x_1 posterior conditional on x_t, shape (..., vocab size).
            x_t (Tensor): path sample at time t, shape (...).
            t (Tensor): time in [0,1].

        Returns:
            Tensor: velocity.
        """
        posterior = torch.softmax(posterior_logits, dim=-1)
        vocabulary_size = posterior.shape[-1]
        x_t = F.one_hot(x_t, num_classes=vocabulary_size)
        t = unsqueeze_to_match(source=t, target=x_t)

        scheduler_output = self.scheduler(t)

        kappa_t = scheduler_output.alpha_t
        d_kappa_t = scheduler_output.d_alpha_t

        return (d_kappa_t / (1 - kappa_t)) * (posterior - x_t)
    
class MetricInducedGibbsProbPath(ProbPath):
    """
    Conditional (factorized) discrete path:
        p_t(x_i | x1_i) ∝ exp{-beta(t) * d(E[x_i], E[x1_i])}
    with a Euclidean (default) or cosine distance over a *fixed* embedding table E.

    Notes:
    - Use this to SAMPLE X_t during training. The generalized KL loss in Meta FM
      only uses the scheduler from the path; it expects model logits shaped (B, d, K)
      and integer tokens for (x_t, x_1).
    - Meta’s MixtureDiscreteEulerSolver is tied to the mixture path’s velocity
      (not this metric-induced path), so `posterior_to_velocity`
      is intentionally left unimplemented here.
    """

    def __init__(
        self,
        embedding_path_or_weight: Optional[Union[str, Tensor, nn.Embedding]] = None,
        vocab_size: int = 256,
        emb_dim: int = 1,
        metric: str = "euclidean",  # "euclidean" | "cosine" | "lp"
        lp_order: float = 3.0,       # used when metric == "lp" (KO: lp=3)
        embed_range: str = "unit",  # "unit" -> [0,1], "pm1" -> [-1,1] (KO)
        a: float = 5.0,              # β(t) = c * (t/(1-t))**a (KO: a=5)
        c: float = 1.0,              # (KO: c=1)
        eps_t: float = 1e-7,          # clamp t away from 0,1 for beta(t) stability
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        beta_schedule: Optional[BetaSchedule] = None,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        gumbel_hard: bool = True,
        *,
        learnable_metric_dim: int = 0,
        learnable_metric_diag_eps: float = 1e-4,
        metric_interp_lambda: float = 0.0,
        learnable_lut: bool = False,
        lut_num_channels: int = 3,
        lut_emb_dim: int = 1,
        lut_share_across_channels: bool = False,
        lut_renorm_to_init_norm: bool = False,
        lut_bounded_residual_scale: bool = False,
        lut_scale_baseline: float = 1.0,
        lut_scale_epsilon: float = 0.25,
        lut_init_method: str = "linear",
        lut_init_noise_scale: float = 0.01,
        use_normalized_distance: bool = False,
        lut_cosine_scale: float = 1.0,
    ):
        super().__init__()
        self.metric_name = metric
        self.lp_order = float(lp_order)
        assert embed_range in {"unit", "pm1"}, "embed_range must be 'unit' or 'pm1'"
        self.embed_range = embed_range
        self.a = float(a)
        self.c = float(c)
        self.eps_t = float(eps_t)
        self.dtype = dtype
        self.beta_schedule = beta_schedule
        if self.beta_schedule is not None and device is not None:
            self.beta_schedule.to(device=device, dtype=dtype)
        self.use_gumbel = bool(use_gumbel)
        self.gumbel_tau = float(gumbel_tau)
        self.gumbel_hard = bool(gumbel_hard)

        self.embedding = self._build_embedding(
            embedding_path_or_weight, vocab_size, emb_dim, device, dtype
        )
        # Freeze: this table is only for defining the *metric*, not a learnable model stem.
        for p in self.embedding.parameters():
            p.requires_grad_(False)

        self.learnable_metric: Optional[MahalanobisTokenMetric] = None
        self.learnable_lut: Optional[LearnableScalarLUT] = None
        self._lut_num_channels = int(max(1, lut_num_channels))
        self._lut_share_across_channels = bool(lut_share_across_channels)
        self._lut_renorm_to_init_norm = bool(lut_renorm_to_init_norm)
        self._lut_bounded_residual_scale = bool(lut_bounded_residual_scale)
        self._lut_scale_baseline = float(lut_scale_baseline)
        self._lut_scale_epsilon = float(lut_scale_epsilon)
        self._lut_init_method = str(lut_init_method)
        self._lut_init_noise_scale = float(lut_init_noise_scale)
        self._use_normalized_distance = bool(use_normalized_distance)
        # Optional cosine metric scale (applied to 1-cos distance)
        self._lut_cosine_scale: float = float(lut_cosine_scale)

        if learnable_lut:
            if learnable_metric_dim > 0:
                raise ValueError("Cannot enable both learnable_metric and learnable_lut")
            if metric not in {"lp", "euclidean", "cosine"}:
                raise ValueError(
                    "learnable_lut supports metrics: 'lp', 'euclidean', or 'cosine'"
                )
            if metric == "cosine":
                logger.info(
                    "Initializing learnable LUT with cosine distance (scale=%.3f)",
                    self._lut_cosine_scale,
                )
            num_channels = 1 if self._lut_share_across_channels else self._lut_num_channels
            lut_module = LearnableScalarLUT(
                num_channels=num_channels,
                vocab_size=vocab_size,
                emb_dim=int(lut_emb_dim),
                embed_range="pm1" if embed_range == "pm1" else "unit",
                device=device,
                dtype=dtype,
                lp_order=float(lp_order),
                renormalize_to_init_norm=self._lut_renorm_to_init_norm,
                bounded_residual_scale=self._lut_bounded_residual_scale,
                scale_baseline=self._lut_scale_baseline,
                scale_epsilon=self._lut_scale_epsilon,
                init_method=self._lut_init_method,
                init_noise_scale=self._lut_init_noise_scale,
            )
            if device is not None:
                lut_module = lut_module.to(device=device, dtype=dtype)
            else:
                lut_module = lut_module.to(dtype=dtype)
            self.learnable_lut = lut_module

        if learnable_metric_dim > 0:
            init_codes = self._build_metric_init_codes(
                metric_dim=int(learnable_metric_dim),
                device=device if device is not None else self.embedding.weight.device,
                dtype=dtype,
            )
            metric_module = MahalanobisTokenMetric(
                vocab_size,
                metric_dim=int(learnable_metric_dim),
                init_codes=init_codes,
                diag_eps=float(learnable_metric_diag_eps),
            )
            metric_module = metric_module.to(dtype=dtype)
            if device is not None:
                metric_module = metric_module.to(device=device)
            self.learnable_metric = metric_module

        # Caches for speed: token distance table and baseline distance table
        self._cached_dist_table: Optional[Tensor] = None  # [K,K]
        self._base_dist_table_cpu: Optional[Tensor] = None
        self._cached_emb_weight: Optional[Tensor] = None  # reference to detect invalidation
        self._cached_metric_name: Optional[str] = None
        self._cached_lp_order: Optional[float] = None
        # Cache for learned metric during eval (parameters frozen)
        self._cached_learned_dist_table: Optional[Tensor] = None  # [K,K]
        # Cache for learnable LUT distance tables during eval
        self._cached_lut_dist_table: Optional[Tensor] = None  # [C,K,K]

        self.metric_interp_lambda = 0.0
        self.set_metric_interpolation_lambda(metric_interp_lambda)

    # ---------- Embedding handling ----------
    def _build_embedding(
        self,
        src: Optional[Union[str, Tensor, nn.Embedding]],
        vocab_size: int,
        emb_dim: int,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> nn.Embedding:
        def from_weight(w: Tensor) -> nn.Embedding:
            emb = nn.Embedding(w.size(0), w.size(1), _weight=w.to(device=device, dtype=dtype))
            emb.weight.requires_grad_(False)
            return emb

        if isinstance(src, nn.Embedding):
            return src.to(device=device, dtype=dtype)
        if isinstance(src, Tensor):
            assert src.ndim == 2 and src.size(0) == vocab_size, "Bad embedding weight shape"
            return from_weight(src)

        if isinstance(src, str):
            obj = torch.load(src, map_location="cpu")
            if isinstance(obj, nn.Embedding):
                return obj.to(device=device, dtype=dtype)
            if isinstance(obj, dict) and "weight" in obj:
                return from_weight(obj["weight"])
            if isinstance(obj, Tensor):
                return from_weight(obj)
            raise ValueError(f"Unsupported object loaded from {src}")

        # Default deterministic embedding table.
        # For KO CIFAR-10: map tokens to [-1,1] via emb(x) = 2*x/255 - 1
        if self.embed_range == "pm1":
            w = torch.linspace(-1.0, 1.0, steps=vocab_size).unsqueeze(1)  # [K,1]
        else:
            w = torch.linspace(0.0, 1.0, steps=vocab_size).unsqueeze(1)   # [K,1]
        if emb_dim > 1:
            w = w.repeat(1, emb_dim)  # trivial tiling if a wider dim is desired
        return from_weight(w)

    def _build_metric_init_codes(
        self, *, metric_dim: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """
        Initialize metric codes using Classical MDS from L_p distance structure.
        
        This provides a rank-d initialization that captures the intrinsic geometry
        of the L_p metric, enabling faster convergence compared to rank-1 baseline.
        
        For metric_dim=8: Spearman correlation ≈ 0.82 with L_p distances.
        """
        from flow_matching.path.mds_init import initialize_metric_from_lp
        
        init_codes = initialize_metric_from_lp(
            vocab_size=self.vocab_size,
            metric_dim=metric_dim,
            lp_order=self.lp_order,
            embed_range=self.embed_range,
            device=device,
            dtype=dtype
        )
        return init_codes

    def _embedding_to_tokens(self, emb: Tensor) -> Tensor:
        if emb.shape[-1] < 1:
            raise ValueError("Embedding tensor must have at least one dimension")
        values = emb[..., 0]
        if self.embed_range == "pm1":
            scaled = (values + 1.0) * 0.5
        else:
            scaled = values
        scaled = scaled.clamp(0.0, 1.0)
        idx = torch.round(scaled * (self.vocab_size - 1)).to(dtype=torch.long)
        return idx

    @property
    def vocab_size(self) -> int:
        return self.embedding.num_embeddings

    @property
    def emb_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def has_learnable_metric(self) -> bool:
        return self.learnable_metric is not None

    def set_metric_interpolation_lambda(self, value: float) -> None:
        clamped = float(min(max(value, 0.0), 1.0))
        if self.learnable_lut is not None and clamped > 0.0:
            logger.warning(
                "Interpolation lambda has no effect when learnable_lut is enabled; forcing to 0."
            )
            clamped = 0.0
        self.metric_interp_lambda = clamped

    def get_metric_interpolation_lambda(self) -> float:
        return float(self.metric_interp_lambda)

    # ---------- Scheduler β(t) and its derivative (useful later for KO velocities) ----------
    def beta(self, t: Tensor):
        """
        β(t) = c * (t / (1 - t))**a, with clamping for stability.
        Returns (beta_t, d_beta_t) broadcasting over batch.
        """
        if self.beta_schedule is not None:
            return self.beta_schedule.beta_and_derivative(t)

        eps = self.eps_t
        t = t.clamp(min=eps, max=1.0 - eps)  # (B,)
        u = 1.0 - t                            # denominator; safe since t <= 1 - eps
        y = t / u                              # t/(1 - t)
        beta_t = self.c * (y ** self.a)
        dy_dt = 1.0 / (u * u)
        d_beta_t = self.c * self.a * (y ** (self.a - 1.0)) * dy_dt
        return beta_t, d_beta_t

    def schedule_parameters(self) -> Iterable[nn.Parameter]:
        if isinstance(self.beta_schedule, nn.Module):
            yield from self.beta_schedule.parameters()

    def metric_parameters(self) -> Iterable[nn.Parameter]:
        if self.learnable_metric is not None:
            yield from self.learnable_metric.parameters()

    def lut_parameters(self) -> Iterable[nn.Parameter]:
        if self.learnable_lut is not None:
            yield from self.learnable_lut.parameters()

    # ---------- Distance on the embedding space ----------
    def _pairwise_dist(self, z_flat: Tensor, E: Tensor) -> Tensor:
        """
        z_flat: [B*S, D], E: [K, D] -> distances [B*S, K]
        """
        if self.metric_name == "euclidean":
            # cdist is stable and vectorized; returns L2 distance
            return torch.cdist(z_flat, E, p=2.0)
        elif self.metric_name == "lp":
            # General L_p. torch.cdist supports generic p>0.
            return torch.cdist(z_flat, E, p=self.lp_order)
        elif self.metric_name == "cosine":
            z_n = F.normalize(z_flat, p=2, dim=-1)
            E_n = F.normalize(E, p=2, dim=-1)
            # Cosine distance = 1 - cosine similarity
            return 1.0 - (z_n @ E_n.T)
        else:
            raise ValueError(f"Unsupported metric: {self.metric_name}")

    def metric(self, z: Tensor) -> Tensor:
        """
        z: [B, S, D] = E[x1] per site. Returns distances d to all tokens: [B, S, K].
        """
        B, S, D = z.shape
        E = self.embedding.weight.to(device=z.device, dtype=z.dtype)  # [K, D]
        d_flat = self._pairwise_dist(z.view(B * S, D), E)             # [B*S, K]
        return d_flat.view(B, S, self.vocab_size)

    # ---------- Precompute and use fast token-indexed distances ----------
    def _get_base_distance_table(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.learnable_lut is not None:
            return self._build_lut_distance_table(device=device, dtype=dtype)

        E = self.embedding.weight
        metric_changed = (
            self._cached_metric_name != self.metric_name
            or (self.metric_name == "lp" and self._cached_lp_order != self.lp_order)
        )
        weight_changed = (self._cached_emb_weight is not E)

        if self._base_dist_table_cpu is None or metric_changed or weight_changed:
            Ew = E.detach().to("cpu", dtype=torch.float32)
            if self.metric_name in ("euclidean", "lp"):
                p = 2.0 if self.metric_name == "euclidean" else float(self.lp_order)
                dist_cpu = torch.cdist(Ew, Ew, p=p)
            elif self.metric_name == "cosine":
                Ew_n = F.normalize(Ew, p=2, dim=-1)
                dist_cpu = 1.0 - (Ew_n @ Ew_n.T)
            else:
                raise ValueError(f"Unsupported metric: {self.metric_name}")

            self._base_dist_table_cpu = dist_cpu
            self._cached_emb_weight = E
            self._cached_metric_name = self.metric_name
            self._cached_lp_order = float(self.lp_order)

        base = self._base_dist_table_cpu
        assert base is not None
        if base.device != device or base.dtype != dtype:
            base = base.to(device=device, dtype=dtype)
        return base

    def _build_lut_distance_table(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Build pairwise distance table for LUT embeddings.
        
        When use_normalized_distance=True:
            Uses normalized distance: \tilde d = ||E[v] - E[x_1]||_2 / \sqrt{m}
            where m is the embedding dimension (for vector embeddings) or 1 (for scalar)
        
        Returns:
            Tensor of shape [num_channels, vocab_size, vocab_size] containing
            pairwise Lp distances between all token embeddings.
        """
        assert self.learnable_lut is not None
        weight = self.learnable_lut()
        if weight.device != device or weight.dtype != dtype:
            weight = weight.to(device=device, dtype=dtype)
        channels, vocab_size, emb_dim = weight.shape

        # Cosine metric: compute 1 - cosine similarity on unit-normalized embeddings
        if self.metric_name == "cosine":
            wn = F.normalize(weight, p=2, dim=-1, eps=1e-12)  # [C,V,D]
            sim = torch.matmul(wn, wn.transpose(-1, -2))      # [C,V,V]
            dist = (1.0 - sim) * float(self._lut_cosine_scale)
            return dist

        # Compute pairwise differences: [C, V, V, D]
        diff = weight[:, :, None, :] - weight[:, None, :, :]  # [C, V, 1, D] - [C, 1, V, D]

        if self._use_normalized_distance:
            # Normalized distance: ||diff||_2 / sqrt(m)
            # For scalar embeddings (emb_dim=1): m=1, so just ||diff||_2
            # For vector embeddings (emb_dim>1): m=emb_dim
            m = float(emb_dim)  # embedding dimension as normalization factor
            dist = torch.linalg.vector_norm(diff, dim=-1) / math.sqrt(m)  # [C, V, V]
        else:
            # Original Lp norm distance
            p = self.learnable_lut.lp_order
            if emb_dim == 1:
                # Scalar case: simple absolute difference (backward compatible)
                dist = diff.abs().squeeze(-1)  # [C, V, V]
            elif p == 1.0:
                # L1 norm: sum of absolute differences
                dist = diff.abs().sum(dim=-1)  # [C, V, V]
            elif p == 2.0:
                # L2 norm: Euclidean distance
                dist = (diff ** 2).sum(dim=-1).sqrt()  # [C, V, V]
            else:
                # General Lp norm
                dist = (diff.abs() ** p).sum(dim=-1) ** (1.0 / p)  # [C, V, V]
        
        return dist

    def _lut_channel_assignments(self, tokens: Tensor) -> Tensor:
        assert self.learnable_lut is not None
        channels = self.learnable_lut.num_channels
        B, S = tokens.shape
        if channels == 1:
            return torch.zeros((B, S), device=tokens.device, dtype=torch.long)
        if S % channels != 0:
            raise ValueError(
                f"Token sequence length {S} is not divisible by lut channels {channels}."
            )
        per_channel = S // channels
        base = torch.arange(S, device=tokens.device) // per_channel
        return base.unsqueeze(0).expand(B, -1)

    def _lut_rows(self, dist_table: Tensor, token_indices: Tensor) -> Tensor:
        assert self.learnable_lut is not None
        channels = dist_table.shape[0]
        B, S = token_indices.shape
        channel_ids = self._lut_channel_assignments(token_indices).reshape(-1)
        flat_tokens = token_indices.reshape(-1)
        lookup = dist_table.reshape(channels * self.vocab_size, self.vocab_size)
        combined = channel_ids * self.vocab_size + flat_tokens
        rows = lookup.index_select(0, combined)
        return rows.view(B, S, self.vocab_size)

    def _lut_pair_distance(self, dist_table: Tensor, x_tokens: Tensor, x1_tokens: Tensor) -> Tensor:
        assert self.learnable_lut is not None
        channels = dist_table.shape[0]
        assignments = self._lut_channel_assignments(x_tokens).reshape(-1)
        flat = x_tokens.reshape(-1)
        flat1 = x1_tokens.reshape(-1)
        lookup = dist_table.reshape(channels * self.vocab_size, self.vocab_size)
        combined = assignments * self.vocab_size + flat
        rows = lookup.index_select(0, combined)
        dist = rows.gather(1, flat1.view(-1, 1))
        return dist.view(x_tokens.shape + (1,))

    def precompute_lut_distance_table(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Precompute and cache the learnable LUT distance table for evaluation.

        This should be called at the start of evaluation when LUT parameters are
        frozen so that repeated sampling steps can reuse the cached distances.
        """
        if self.learnable_lut is None:
            logger.warning("precompute_lut_distance_table is a no-op (learnable_lut is None)")
            return
        with torch.no_grad():
            table = self._build_lut_distance_table(device=device, dtype=dtype).detach()
        self._cached_lut_dist_table = table
        logger.info(
            "Precomputed learnable LUT distance table [%s] on %s with dtype %s",
            tuple(table.shape),
            device,
            dtype,
        )

    def clear_lut_cache(self) -> None:
        """Clear the cached LUT distance table (if any)."""
        if self._cached_lut_dist_table is not None:
            logger.info("Clearing cached learnable LUT distance table")
            self._cached_lut_dist_table = None

    def precompute_learned_metric_table(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        metric_module: Optional[MahalanobisTokenMetric] = None,
    ) -> None:
        """Precompute and cache learned metric distance table for evaluation.

        Args:
            device: Device to store the cached table on.
            dtype: Data type for the cached table.
            metric_module: Optional override for the metric (e.g., EMA teacher).
                When omitted, uses ``self.learnable_metric``.
        """
        if self.learnable_lut is not None:
            logger.warning(
                "precompute_learned_metric_table is intended for learnable metrics; "
                "call precompute_lut_distance_table when using a learnable LUT instead."
            )
            return
        learned_table = self._learned_distance_table(
            device=device,
            dtype=dtype,
            metric_module=metric_module,
        )
        if learned_table is not None:
            self._cached_learned_dist_table = learned_table
            logger.info(
                f"Precomputed learned metric distance table [{learned_table.shape}] "
                f"on {device} with dtype {dtype}"
            )
        else:
            logger.warning("No learned metric to precompute (learnable_metric is None)")

    def clear_learned_metric_cache(self) -> None:
        """Clear the cached learned metric distance table.
        
        This should be called at the end of evaluation to free memory.
        """
        if self._cached_learned_dist_table is not None:
            logger.info("Clearing cached learned metric distance table")
            self._cached_learned_dist_table = None

    def _learned_distance_table(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        metric_module: Optional[MahalanobisTokenMetric] = None,
    ) -> Optional[Tensor]:
        module = metric_module if metric_module is not None else self.learnable_metric
        if module is None:
            return None

        return module.pairwise_distance_table(device=device, dtype=dtype)

    def _build_distance_table(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        metric_module: Optional[MahalanobisTokenMetric] = None,
        cache_result: bool = True,
        use_cache: bool = False,
    ) -> Tensor:
        if self.learnable_lut is not None:
            if use_cache and self._cached_lut_dist_table is not None:
                table = self._cached_lut_dist_table
                if table.device != device or table.dtype != dtype:
                    table = table.to(device=device, dtype=dtype)
            else:
                table = self._build_lut_distance_table(device=device, dtype=dtype)
            return table

        base = self._get_base_distance_table(device=device, dtype=dtype)

        lam = float(self.metric_interp_lambda)
        learned_table = None
        if lam > 0.0:
            # Try to use cached table first (eval optimization)
            if use_cache and self._cached_learned_dist_table is not None:
                learned_table = self._cached_learned_dist_table
                # Ensure device/dtype match
                if learned_table.device != device or learned_table.dtype != dtype:
                    learned_table = learned_table.to(device=device, dtype=dtype)
            else:
                # Recompute (training or cache miss)
                learned_table = self._learned_distance_table(
                    device=device, dtype=dtype, metric_module=metric_module
                )

        if learned_table is None or lam <= 0.0:
            dist = base
        elif lam >= 1.0:
            dist = learned_table
        else:
            dist = (1.0 - lam) * base + lam * learned_table

        # NOTE: We do NOT cache the learned metric distance table during training
        # because the metric parameters are constantly changing. Caching would
        # use stale distances. Only cache the baseline Lp table (done in _get_base_distance_table).
        # if cache_result and metric_module is None:
        #     self._cached_dist_table = dist
        return dist

    def distances_from_tokens(self, token_indices: Tensor) -> Tensor:
        """Return distances to all vocab tokens for each provided token index.

        Args:
            token_indices: int tensor of shape [B,S]
        Returns:
            dist: [B,S,K] where dist[b,s,:] = d(E[token_indices[b,s]], E[:])
        """
        assert token_indices.dtype in (torch.int32, torch.int64)
        device = token_indices.device
        dtype = self.embedding.weight.dtype
        
        # LUT path: [C, K, K] distance table requires channel-aware indexing
        if self.learnable_lut is not None:
            dist_table = self._build_distance_table(
                device=device, dtype=dtype, use_cache=True
            )
            return self._lut_rows(dist_table, token_indices)
        
        # Standard path: [K, K] distance table
        dist_table = self._build_distance_table(
            device=device, dtype=dtype, use_cache=True
        )
        # Gather rows for each token index
        K = self.vocab_size
        B, S = token_indices.shape[0], token_indices.view(token_indices.shape[0], -1).shape[1]
        flat = token_indices.view(-1)  # [B*S]
        dist_rows = dist_table.index_select(dim=0, index=flat)  # [B*S, K]
        return dist_rows.view(token_indices.shape + (K,))

    def get_prob_distribution_from_tokens(
        self,
        x1_tokens: Tensor,
        t: Tensor,
        *,
        beta_values: Optional[Tensor] = None,
        beta_schedule: Optional[BetaSchedule] = None,
        metric_module: Optional[MahalanobisTokenMetric] = None,
    ) -> Tensor:
        """Fast probability path using token indices directly.

        Args:
            x1_tokens: int tensor [B,S]
            t: [B]
            beta_values: optional precomputed β(t) values of shape [B].
            beta_schedule: optional schedule override used when ``beta_values`` is not
                provided. When omitted, the path's current schedule is used.
            metric_module: optional learned metric override (e.g., EMA teacher).
        Returns:
            probs: [B,S,K]
        """
        B = x1_tokens.shape[0]
        device = x1_tokens.device
        dtype = self.embedding.weight.dtype
        dist_table = self._build_distance_table(
            device=device,
            dtype=dtype,
            metric_module=metric_module,
            cache_result=metric_module is None,
            use_cache=True,  # Always try cache (no-op during training)
        )
        flat_indices = x1_tokens.view(-1)
        if self.learnable_lut is not None:
            d = self._lut_rows(dist_table, x1_tokens)
        else:
            d = dist_table.index_select(0, flat_indices).view(x1_tokens.shape + (self.vocab_size,))
        if beta_values is not None:
            beta_t = beta_values
        elif beta_schedule is not None:
            beta_t, _ = beta_schedule.beta_and_derivative(t)
        else:
            beta_t, _ = self.beta(t)
        beta_t = beta_t.view(B, 1, 1)
        logits = -beta_t * d  # [B, S, K]
        # Numerically stable log-sum-exp softmax (no clamp, fully backward compatible)
        max_logits = logits.max(dim=-1, keepdim=True).values  # [B, S, 1]
        logits = logits - max_logits
        return torch.softmax(logits, dim=-1)

    def pair_distance_tokens(self, x_tokens: Tensor, x1_tokens: Tensor) -> Tensor:
        """Return per-site distance d(E[x], E[x1]) for current state x and target x1.

        Args:
            x_tokens: [B,S]
            x1_tokens: [B,S]
        Returns:
            dist: [B,S,1]
        """
        assert x_tokens.shape == x1_tokens.shape
        device = x_tokens.device
        dtype = self.embedding.weight.dtype
        dist_table = self._build_distance_table(
            device=device, dtype=dtype, use_cache=True
        )
        if self.learnable_lut is not None:
            dist = self._lut_pair_distance(dist_table, x_tokens, x1_tokens)
            return dist

        flat = x_tokens.view(-1)             # [N]
        flat1 = x1_tokens.view(-1)           # [N]
        rows = dist_table.index_select(0, flat)  # [N,K]
        dist = rows.gather(1, flat1.view(-1, 1))              # [N,1]
        return dist.view(x_tokens.shape + (1,))

    # ---------- Conditional distribution p_t(· | x1) ----------
    def get_prob_distribution(self, emb_x1: Tensor, t: Tensor) -> Tensor:
        """
        emb_x1: E[x1] per site, shape [B, S, D]
        t: shape [B]
        returns probs [B, S, K] with last-dim softmax (vocab).
        """
        tokens = self._embedding_to_tokens(emb_x1)
        return self.get_prob_distribution_from_tokens(tokens, t)

    # ---------- API: sample() ----------
    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> DiscretePathSample:
        """
        x_0: (B, ...) ints, not used by this path (metric-induced depends only on x1)
        x_1: (B, ...) ints (e.g., (B, C, H, W))
        t:   (B,) in [0,1]
        returns X_t as integers (B, S)
        """
        assert x_1.dtype in (torch.int32, torch.int64), "x_1 must be integer tokens"
        device = x_1.device
        B = x_1.shape[0]
        orig_shape = x_1.shape  # (B, ...)
        x1_flat = x_1.view(B, -1)
        # Fast path via precomputed table
        probs = self.get_prob_distribution_from_tokens(x1_flat, t)
        S = probs.shape[1]
        x_t_soft = None
        if self.use_gumbel:
            logits = torch.log(probs.clamp_min(1e-12))
            gumbel = F.gumbel_softmax(
                logits,
                tau=self.gumbel_tau,
                hard=self.gumbel_hard,
                dim=-1,
            )
            x_t_soft = gumbel.view(orig_shape + (self.vocab_size,))
            x_t_flat = gumbel.argmax(dim=-1)
        else:
            x_t_flat = torch.multinomial(
                probs.view(B * S, self.vocab_size), num_samples=1, replacement=True
            ).view(B, S)

        x_t = x_t_flat.view(orig_shape).to(device=device, dtype=x_1.dtype)
        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t, x_t_soft=x_t_soft)

    # ---------- API: posterior_to_velocity() ----------
    def posterior_to_velocity(self, posterior_logits: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """
        This path is *not* the mixture path, so the mixture closed-form velocity used by
        Meta's MixtureDiscreteEulerSolver does not apply.
        Implementing KO velocities for this general path requires either:
          - the KO closed-form (sec. 4.1) or
          - solving the Laplacian (eq. 21) once per t/path (costly).

        For pretraining you don't need this; the loss only needs scheduler(t).
        """
        raise NotImplementedError("Use MixtureDiscreteEulerSolver only with MixtureDiscreteProbPath.")

