# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Iterable, Optional, Union

from torch import Tensor


def _inv_softplus_tensor(x: Tensor) -> Tensor:
    """Stable inverse softplus used for positive-diagonal initialization."""

    return torch.log(torch.expm1(x))


class MahalanobisTokenMetric(nn.Module):
    """Learnable PSD metric defined via token codes and a lower-triangular map."""

    def __init__(
        self,
        vocab_size: int,
        metric_dim: int,
        *,
        init_codes: Optional[Tensor] = None,
        diag_eps: float = 1e-4,
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

        self.codes = nn.Parameter(torch.zeros(self.vocab_size, self.metric_dim))
        self._lower_params = nn.Parameter(torch.zeros(self.metric_dim, self.metric_dim))

        self.reset_parameters(init_codes=init_codes)

    @torch.no_grad()
    def reset_parameters(self, init_codes: Optional[Tensor] = None) -> None:
        if init_codes is not None:
            if init_codes.shape != (self.vocab_size, self.metric_dim):
                raise ValueError(
                    "init_codes must have shape (vocab_size, metric_dim)"
                )
            self.codes.copy_(init_codes)
        else:
            self.codes.zero_()

        self._lower_params.zero_()
        inv_sp_one = _inv_softplus_tensor(torch.ones(self.metric_dim, dtype=self._lower_params.dtype))
        torch.diagonal(self._lower_params).copy_(inv_sp_one)

    def cholesky_factor(self) -> Tensor:
        lower = torch.tril(self._lower_params)
        diag = torch.diagonal(lower)
        positive_diag = F.softplus(diag) + self.diag_eps
        lower = lower - torch.diag_embed(diag) + torch.diag_embed(positive_diag)
        return lower

    def transformed_codes(self, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Tensor:
        codes = self.codes
        if device is not None or dtype is not None:
            codes = codes.to(device=device or codes.device, dtype=dtype or codes.dtype)
        lower = self.cholesky_factor().to(device=codes.device, dtype=codes.dtype)
        return codes @ lower.T

    def pairwise_distance_table(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        transformed = self.transformed_codes(device=device, dtype=dtype)
        return torch.cdist(transformed, transformed, p=2.0)

    def get_extra_state(self) -> dict:
        return {"diag_eps": self.diag_eps}

    def set_extra_state(self, state: dict) -> None:  # pragma: no cover - API requirement
        self.diag_eps = float(state.get("diag_eps", self.diag_eps))

from flow_matching.path.beta_schedules import BetaSchedule
from flow_matching.path.path import ProbPath

from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import ConvexScheduler
from flow_matching.utils import expand_tensor_like, unsqueeze_to_match
    
def _offdiag_mad_float(matrix: Tensor) -> float:
    """Return the off-diagonal median absolute deviation of a square matrix."""

    if matrix.ndim != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("matrix must be square")
    if matrix.size(0) <= 1:
        return 0.0

    mat = matrix.detach().to(dtype=torch.float64)
    n = mat.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=mat.device)
    values = mat.masked_select(mask)
    if values.numel() == 0:
        return 0.0

    median = torch.median(values)
    deviations = (values - median).abs()
    mad = torch.median(deviations)
    return float(mad)


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
        base = self.embedding.weight.detach().to(device=device, dtype=dtype)
        codes = torch.zeros(base.size(0), metric_dim, device=device, dtype=dtype)
        dims = min(base.size(1), metric_dim)
        if dims > 0:
            codes[:, :dims] = base[:, :dims]
        return codes

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

    def learnable_parameters(self) -> Iterable[nn.Parameter]:
        params: list[nn.Parameter] = []
        if isinstance(self.beta_schedule, nn.Module):
            params.extend(self.beta_schedule.parameters())
        if self.learnable_metric is not None:
            params.extend(self.learnable_metric.parameters())
        return params

    def schedule_parameters(self) -> Iterable[nn.Parameter]:
        if isinstance(self.beta_schedule, nn.Module):
            yield from self.beta_schedule.parameters()

    def metric_parameters(self) -> Iterable[nn.Parameter]:
        if self.learnable_metric is not None:
            yield from self.learnable_metric.parameters()

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
            try:
                self._base_dist_offdiag_scale = _offdiag_mad_float(dist_cpu)
            except ValueError:
                self._base_dist_offdiag_scale = None

        base = self._base_dist_table_cpu
        assert base is not None
        if base.device != device or base.dtype != dtype:
            base = base.to(device=device, dtype=dtype)
        return base

    def _build_distance_table(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        metric_module: Optional[MahalanobisTokenMetric] = None,
        cache_result: bool = True,
    ) -> Tensor:
        base = self._get_base_distance_table(device=device, dtype=dtype)

        module = metric_module if metric_module is not None else self.learnable_metric
        learned_table = None
        lam = float(self.metric_interp_lambda)
        if module is not None and lam > 0.0:
            learned_table = module.pairwise_distance_table(device=device, dtype=dtype)
            base_scale = self._base_dist_offdiag_scale
            if base_scale is not None and base_scale > 0.0:
                try:
                    learned_scale = _offdiag_mad_float(learned_table)
                except ValueError:
                    learned_scale = None
                if learned_scale is not None and learned_scale > 0.0:
                    scale_value = base_scale / max(learned_scale, 1e-12)
                    if math.isfinite(scale_value) and scale_value > 0.0:
                        scale_tensor = torch.as_tensor(
                            scale_value,
                            device=learned_table.device,
                            dtype=learned_table.dtype,
                        )
                        learned_table = learned_table * scale_tensor

        if learned_table is None or lam <= 0.0:
            dist = base
        elif lam >= 1.0:
            dist = learned_table
        else:
            dist = (1.0 - lam) * base + lam * learned_table

        if cache_result and metric_module is None:
            self._cached_dist_table = dist
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
        dist_table = self._build_distance_table(device=device, dtype=dtype)
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
        )
        flat_indices = x1_tokens.view(-1)
        d = dist_table.index_select(0, flat_indices).view(x1_tokens.shape + (self.vocab_size,))
        if beta_values is not None:
            beta_t = beta_values
        elif beta_schedule is not None:
            beta_t, _ = beta_schedule.beta_and_derivative(t)
        else:
            beta_t, _ = self.beta(t)
        beta_t = beta_t.view(B, 1, 1)
        logits = -beta_t * d
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
        dist_table = self._build_distance_table(device=device, dtype=dtype)
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

