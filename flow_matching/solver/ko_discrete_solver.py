# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from flow_matching.path.mixture import MetricInducedGibbsProbPath
from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical, ModelWrapper

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(*args, **kwargs):  # type: ignore
        return nullcontext()


class KODiscreteGibbsEulerSolver(Solver):
    """KO-style discrete CTMC solver for metric-induced Gibbs path on pixel space.

    This Euler solver advances X_t by:
      1) Sample x_1 ~ p_{1|t}(· | x_t) from the model
      2) Build conditional intensities u_t(· | x_t, x_1) ∝ p_t(· | x_1) * dβ_t * Δd(·; x_t, x_1)
         where Δd(x; x_t, x_1) = relu( d(E[x_t],E[x_1]) - d(E[x],E[x_1]) ) and optionally
         add a symmetric correction term symmetrize * p_t(· | x_1) * dβ_t * |d(E[x],E[x_1]) - d(E[x_t],E[x_1])|
      3) Jump with prob 1 - exp(-h * λ) where λ = Σ_x u_t(x), and draw new state from u/λ.

    Notes:
    - This assumes the provided `path` is MetricInducedGibbsProbPath and provides
      embedding, metric() and beta(t) APIs.
    - The wrapped `model` must output logits of shape (B, ..., K) where the last dim is vocab.
    - This implementation is single-modality and expects x tensors of shape (B, S) or (B, C, H, W)
      with integer tokens.
    """

    def __init__(
        self,
        model: ModelWrapper,
        path: MetricInducedGibbsProbPath,
        vocabulary_size: int,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        symmetrize: Union[float, Callable[[float], float]] = 0.0,
        **model_extras,
    ) -> Tensor:
        # Time discretization
        device = x_init.device
        time_grid = time_grid.to(device=device)
        if step_size is None:
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
        else:
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (t_final - t_init) > step_size, (
                f"Time interval [{t_init}, {t_final}] must be larger than step_size {step_size}."
            )
            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=device,
            )

        # Init
        x_t = x_init.clone()
        res = [x_t.clone()] if return_intermediates else []
        steps_counter = 0

        with (tqdm(total=t_final, desc=f"NFE: {steps_counter}") if verbose else nullcontext()) as pbar:
            for i in range(n_steps):
                t = t_discretization[i : i + 1]  # [1]
                h = t_discretization[i + 1 : i + 2] - t
                # repeat time to match batch size
                B = x_t.shape[0]
                t_batch = t.repeat(B)

                # 1) model posterior p_{1|t}(· | x_t)
                #    Expect logits with last dim K
                logits = self.model(x=x_t, t=t_batch, **model_extras)
                p_1t = torch.softmax(logits, dim=-1)

                # Flatten to (B*S, K) for categorical sampler
                p_1t_flat = p_1t.reshape(-1, p_1t.shape[-1])
                x_1 = categorical(p_1t_flat.to(dtype=dtype_categorical)).view(*p_1t.shape[:-1])

                # Final step: directly set x_t = x_1
                if i == n_steps - 1:
                    x_t = x_1
                    if return_intermediates:
                        res.append(x_t.clone())
                    break

                # 2) Build u_t
                B = x_t.shape[0]
                # shapes: treat tokens as (B, S)
                x_t_tokens = x_t.view(B, -1)
                x_1_tokens = x_1.view(B, -1)

                # Fast probs using token-indexed distances
                probs_xt = self.path.get_prob_distribution_from_tokens(x_1_tokens, t_batch)  # [B, S, K]

                # distances
                # d(E[xt], E[x1]) per-site via distance table
                dist_xt_x1 = self.path.pair_distance_tokens(x_t_tokens, x_1_tokens)  # [B,S,1]

                # d(E[x], E[x1]) for all x in vocab
                dist_x1_to_all = self.path.distances_from_tokens(x_1_tokens)  # [B,S,K]
                delta_d = torch.relu(dist_xt_x1 - dist_x1_to_all)  # [B,S,K]

                # dβ_t
                _, d_beta_t = self.path.beta(t_batch)  # [B]
                d_beta_t = d_beta_t.view(-1, 1, 1)  # [B,1,1] via broadcast

                u = probs_xt * d_beta_t * delta_d  # [B,S,K]

                sym_value: float
                if callable(symmetrize):
                    sym_value = float(symmetrize(float(t.item())))
                else:
                    sym_value = float(symmetrize)

                if sym_value != 0.0:
                    sym_term = torch.abs(dist_xt_x1 - dist_x1_to_all)
                    sym_term = probs_xt * d_beta_t * sym_term
                    u = u + sym_value * sym_term

                # Zero self-transition
                onehot_xt = F.one_hot(x_t_tokens, num_classes=self.vocabulary_size)
                u = torch.where(onehot_xt.bool(), torch.zeros_like(u), u)

                # 3) Jump decision and sampling
                intensity = u.sum(dim=-1)  # [B,S]
                prob_jump = 1.0 - torch.exp(-h * intensity)
                mask_jump = torch.rand_like(intensity, dtype=prob_jump.dtype) < prob_jump

                if mask_jump.any():
                    # sample from u normalized on jumped sites
                    u_flat = u.view(-1, self.vocabulary_size)
                    mask_flat = mask_jump.view(-1)
                    chosen = categorical(u_flat[mask_flat].to(dtype=dtype_categorical))
                    x_new = x_t_tokens.clone()
                    x_new.view(-1)[mask_flat] = chosen
                    x_t_tokens = x_new

                x_t = x_t_tokens.view_as(x_t)

                if return_intermediates:
                    res.append(x_t.clone())

                steps_counter += 1
                if verbose and pbar is not None:
                    pbar.n = (t + h).item()
                    pbar.set_description(f"NFE: {steps_counter}")
                    pbar.refresh()

        if return_intermediates:
            return torch.stack(res, dim=0)
        else:
            return x_t
