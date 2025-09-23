# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from flow_matching.solver.ko_discrete_solver import KODiscreteGibbsEulerSolver


class DummyModel(torch.nn.Module):
    def forward(self, x, t, **extras):
        logits = torch.zeros(*x.shape, 3, dtype=torch.float32, device=x.device)
        logits[..., 0] = 5.0
        logits[..., 1] = 1.0
        logits[..., 2] = 0.5
        return logits


class DummyPath:
    vocab_size = 3

    def get_prob_distribution_from_tokens(self, x1_tokens, t_batch):
        base = torch.tensor(
            [[[0.7, 0.2, 0.1]]], dtype=torch.float32, device=x1_tokens.device
        )
        B, S = x1_tokens.shape
        return base.expand(B, S, -1).clone()

    def pair_distance_tokens(self, x_t_tokens, x1_tokens):
        B, S = x_t_tokens.shape
        return torch.ones(B, S, 1, dtype=torch.float32, device=x_t_tokens.device)

    def distances_from_tokens(self, x1_tokens):
        base = torch.tensor(
            [[[0.0, 1.0, 2.0]]], dtype=torch.float32, device=x1_tokens.device
        )
        B, S = x1_tokens.shape
        return base.expand(B, S, -1).clone()

    def beta(self, t):
        beta = torch.ones_like(t, dtype=torch.float32)
        d_beta = torch.full_like(beta, 2.0)
        return beta, d_beta


class TestKODiscreteGibbsEulerSolver(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.path = DummyPath()
        self.solver = KODiscreteGibbsEulerSolver(
            model=self.model, path=self.path, vocabulary_size=self.path.vocab_size
        )

    def _run_and_capture(self, sym_value):
        x_init = torch.tensor([[1]], dtype=torch.long)
        time_grid = torch.tensor([0.0, 1.0])
        calls = []

        def fake_categorical(probs):
            calls.append(probs.clone())
            return torch.zeros(probs.shape[:-1], dtype=torch.long, device=probs.device)

        def fake_rand_like(tensor, *args, **kwargs):
            dtype = kwargs.get("dtype", tensor.dtype)
            return torch.zeros(tensor.shape, dtype=dtype, device=tensor.device)

        with patch("flow_matching.solver.ko_discrete_solver.categorical", side_effect=fake_categorical), patch(
            "flow_matching.solver.ko_discrete_solver.torch.rand_like",
            side_effect=fake_rand_like,
        ):
            self.solver.sample(
                x_init=x_init,
                step_size=0.5,
                time_grid=time_grid,
                symmetrize=sym_value,
            )

        return calls

    def test_symmetrize_zero_matches_base_intensity(self):
        calls = self._run_and_capture(sym_value=0.0)
        self.assertGreaterEqual(len(calls), 2)
        observed_u = calls[1]

        x_t_tokens = torch.tensor([[1]], dtype=torch.long)
        x1_tokens = torch.tensor([[0]], dtype=torch.long)
        t_batch = torch.tensor([0.0])

        probs = self.path.get_prob_distribution_from_tokens(x1_tokens, t_batch)
        dist_xt_x1 = self.path.pair_distance_tokens(x_t_tokens, x1_tokens)
        dist_x1_all = self.path.distances_from_tokens(x1_tokens)
        _, d_beta = self.path.beta(t_batch)
        d_beta = d_beta.view(-1, 1, 1)
        base_delta = torch.relu(dist_xt_x1 - dist_x1_all)
        base_u = probs * d_beta * base_delta
        mask = F.one_hot(x_t_tokens, num_classes=self.path.vocab_size)
        base_u = torch.where(mask.bool(), torch.zeros_like(base_u), base_u)

        expected = base_u.view(-1, self.path.vocab_size)
        self.assertTrue(torch.allclose(observed_u, expected))

    def test_symmetrize_adds_absolute_distance_term(self):
        calls = self._run_and_capture(sym_value=1.0)
        self.assertGreaterEqual(len(calls), 2)
        observed_u = calls[1]

        x_t_tokens = torch.tensor([[1]], dtype=torch.long)
        x1_tokens = torch.tensor([[0]], dtype=torch.long)
        t_batch = torch.tensor([0.0])

        probs = self.path.get_prob_distribution_from_tokens(x1_tokens, t_batch)
        dist_xt_x1 = self.path.pair_distance_tokens(x_t_tokens, x1_tokens)
        dist_x1_all = self.path.distances_from_tokens(x1_tokens)
        _, d_beta = self.path.beta(t_batch)
        d_beta = d_beta.view(-1, 1, 1)

        base_delta = torch.relu(dist_xt_x1 - dist_x1_all)
        base_u = probs * d_beta * base_delta
        sym_delta = torch.abs(dist_xt_x1 - dist_x1_all)
        sym_u = probs * d_beta * sym_delta
        mask = F.one_hot(x_t_tokens, num_classes=self.path.vocab_size)
        base_u = torch.where(mask.bool(), torch.zeros_like(base_u), base_u)
        sym_u = torch.where(mask.bool(), torch.zeros_like(sym_u), sym_u)

        expected = (base_u + sym_u).view(-1, self.path.vocab_size)
        self.assertTrue(torch.allclose(observed_u, expected))


if __name__ == "__main__":
    unittest.main()
