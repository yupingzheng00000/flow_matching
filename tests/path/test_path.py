# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import math
import unittest

import torch
from flow_matching.path import (
    AffineProbPath,
    CondOTProbPath,
    GeodesicProbPath,
    MetricInducedGibbsProbPath,
    MixtureDiscreteProbPath,
    ExpMonotoneRQSConfig,
    ExpMonotoneRQSSchedule,
    BetaScheduleEMA,
    LearnableMetricEMA,
    MonotoneRQBetaSchedule,
    MonotoneRQConfig,
)
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils.manifolds import FlatTorus, Sphere
from flow_matching.path.beta_schedules import _ExpRQS1D


class TestAffineProbPath(unittest.TestCase):
    def test_affine_prob_path_sample(self):
        scheduler = CondOTScheduler()
        affine_prob_path = AffineProbPath(scheduler)
        x_0 = torch.randn(10, 5)
        x_1 = torch.randn(10, 5)
        t = torch.randn(10)
        sample = affine_prob_path.sample(x_0, x_1, t)
        self.assertEqual(sample.x_t.shape, x_0.shape)
        self.assertEqual(sample.dx_t.shape, x_0.shape)
        self.assertTrue((sample.t == t).all())
        self.assertTrue((sample.x_0 == x_0).all())
        self.assertTrue((sample.x_1 == x_1).all())

    def test_assert_sample_shape(self):
        scheduler = CondOTScheduler()
        path = AffineProbPath(scheduler)
        x_0 = torch.randn(10, 5)
        x_1 = torch.randn(10, 5)
        t = torch.randn(10)
        path.assert_sample_shape(x_0, x_1, t)

        x_0 = torch.randn(10, 5)
        x_1 = torch.randn(10, 5)
        t = torch.randn(5)
        with self.assertRaises(AssertionError):
            path.assert_sample_shape(x_0, x_1, t)

    def test_cond_ot_prob_path_sample(self):
        cond_ot_prob_path = CondOTProbPath()
        scheduler = CondOTScheduler()
        affine_path = AffineProbPath(scheduler)
        x_0 = torch.randn(10, 5)
        x_1 = torch.randn(10, 5)
        t = torch.randn(10)
        sample1 = cond_ot_prob_path.sample(x_0, x_1, t)
        sample2 = affine_path.sample(x_0, x_1, t)
        self.assertTrue(torch.allclose(sample1.x_t, sample2.x_t))

    def test_to_velocity(self):
        path = CondOTProbPath()
        x_1 = torch.randn(10, 5, dtype=torch.float64)
        x_t = torch.randn(10, 5, dtype=torch.float64)
        t = torch.randn(10, 5, dtype=torch.float64)
        velocity = path.target_to_velocity(x_1, x_t, t)
        target = path.velocity_to_target(velocity, x_t, t)
        self.assertTrue(torch.allclose(target, x_1))

    def test_to_epsilon(self):
        path = CondOTProbPath()
        x_1 = torch.randn(10, 5, dtype=torch.float64)
        x_t = torch.randn(10, 5, dtype=torch.float64)
        t = torch.randn(10, 5, dtype=torch.float64)
        epsilon = path.target_to_epsilon(x_1, x_t, t)
        target = path.epsilon_to_target(epsilon, x_t, t)
        self.assertTrue(torch.allclose(target, x_1))

    def test_epsilson_velocity(self):
        path = CondOTProbPath()
        velocity = torch.randn(10, 5, dtype=torch.float64)
        x_t = torch.randn(10, 5, dtype=torch.float64)
        t = torch.randn(10, 5, dtype=torch.float64)

        epsilon = path.velocity_to_epsilon(velocity, x_t, t)
        v = path.epsilon_to_velocity(epsilon, x_t, t)
        self.assertTrue(torch.allclose(v, velocity))


class TestGeodesicProbPath(unittest.TestCase):
    def test_sphere(self):
        manifold = Sphere()
        path = GeodesicProbPath(manifold=manifold, scheduler=CondOTScheduler())

        def wrap(samples):
            center = torch.cat(
                [torch.zeros_like(samples), torch.ones_like(samples[..., 0:1])], dim=-1
            )
            samples = (
                torch.cat([samples, torch.zeros_like(samples[..., 0:1])], dim=-1) / 2
            )
            return manifold.expmap(center, samples)

        x1 = manifold.projx(torch.rand(5, 5, dtype=torch.float64))
        x0 = torch.randn_like(x1)
        x0 = wrap(x0)
        x1 = wrap(x1)
        t = torch.rand(x0.size(0), dtype=torch.float64)

        sample = path.sample(t=t, x_0=x0, x_1=x1)

        # Check that x_t is on the sphere
        self.assertTrue(
            torch.allclose(
                sample.x_t.norm(2, -1), torch.ones(x0.size(0), dtype=torch.float64)
            )
        )

    def test_torus(self):
        manifold = FlatTorus()
        path = GeodesicProbPath(manifold=manifold, scheduler=CondOTScheduler())

        def wrap(samples):
            center = torch.zeros_like(samples)
            return manifold.expmap(center, samples)

        batch_size = 5
        coord1 = torch.rand(batch_size, dtype=torch.float64) * 4 - 2
        coord2_ = (
            torch.rand(batch_size, dtype=torch.float64)
            - torch.randint(high=2, size=(batch_size,), dtype=torch.float64) * 2
        )
        coord2 = coord2_ + (torch.floor(coord1) % 2)

        x1 = torch.stack([coord1, coord2], dim=1)
        x0 = torch.randn_like(x1)
        x0 = wrap(x0)
        x1 = wrap(x1)
        t = torch.rand(x0.size(0), dtype=torch.float64)

        sample = path.sample(t=t, x_0=x0, x_1=x1)

        self.assertTrue((sample.x_t < 2 * math.pi).all())


class TestMixtureDiscreteProbPath(unittest.TestCase):
    def test_mixture_discrete_prob_path_sample(self):
        scheduler = CondOTScheduler()
        discrete_prob_path = MixtureDiscreteProbPath(scheduler)
        x_0 = torch.randn(10, 5)
        x_1 = torch.randn(10, 5)
        t = torch.randn(10)
        sample = discrete_prob_path.sample(x_0, x_1, t)
        self.assertEqual(sample.x_t.shape, x_0.shape)
        self.assertTrue((sample.t == t).all())
        self.assertTrue((sample.x_0 == x_0).all())
        self.assertTrue((sample.x_1 == x_1).all())

        # Test at t=0
        t = torch.zeros(10)
        sample = discrete_prob_path.sample(x_0, x_1, t)
        self.assertTrue(torch.allclose(sample.x_t, x_0))
        # Test at t=1
        t = torch.ones(10)
        sample = discrete_prob_path.sample(x_0, x_1, t)
        self.assertTrue(torch.allclose(sample.x_t, x_1))

    def test_posterior_to_velocity(self):
        scheduler = CondOTScheduler()
        discrete_prob_path = MixtureDiscreteProbPath(scheduler)
        posterior_logits = torch.randn(10, 5)
        x_t = torch.randint(0, 5, size=[10])
        t = torch.randn(10)
        x_t_one_hot = torch.nn.functional.one_hot(x_t, num_classes=5)
        velocity = discrete_prob_path.posterior_to_velocity(posterior_logits, x_t, t)
        expected_velocity = (torch.softmax(posterior_logits, dim=-1) - x_t_one_hot) / (
            1 - t
        ).unsqueeze(-1)
        self.assertTrue(torch.allclose(velocity, expected_velocity))


class TestMetricInducedProbPath(unittest.TestCase):
    def test_gumbel_sample_has_soft_assignments(self):
        path = MetricInducedGibbsProbPath(
            vocab_size=4,
            emb_dim=1,
            metric="lp",
            lp_order=2.0,
            embed_range="unit",
            a=1.0,
            c=1.0,
            use_gumbel=True,
            gumbel_tau=0.7,
        )
        x1 = torch.randint(0, 4, size=(3, 1, 1, 1), dtype=torch.long)
        x0 = torch.zeros_like(x1)
        t = torch.full((3,), 0.5)
        sample = path.sample(x_0=x0, x_1=x1, t=t)
        self.assertIsNotNone(sample.x_t_soft)
        assert sample.x_t_soft is not None  # satisfy type checker
        self.assertEqual(sample.x_t_soft.shape, x1.shape + (path.vocab_size,))
        probs = sample.x_t_soft.sum(dim=-1)
        self.assertTrue(torch.allclose(probs, torch.ones_like(probs)))

    def test_learnable_metric_warm_start_matches_lp(self):
        torch.manual_seed(0)
        base_path = MetricInducedGibbsProbPath(
            vocab_size=8,
            emb_dim=1,
            metric="lp",
            lp_order=3.0,
            embed_range="pm1",
        )
        learned_path = MetricInducedGibbsProbPath(
            vocab_size=8,
            emb_dim=1,
            metric="lp",
            lp_order=3.0,
            embed_range="pm1",
            learnable_metric_dim=4,
            metric_interp_lambda=0.0,
        )

        tokens = torch.tensor([[0, 2, 4, 6]], dtype=torch.long)
        t = torch.tensor([0.3])
        base_probs = base_path.get_prob_distribution_from_tokens(tokens, t)
        warm_probs = learned_path.get_prob_distribution_from_tokens(tokens, t)
        self.assertTrue(torch.allclose(base_probs, warm_probs, atol=1e-6))

        learned_path.set_metric_interpolation_lambda(1.0)
        full_probs = learned_path.get_prob_distribution_from_tokens(tokens, t)
        self.assertTrue(torch.allclose(base_probs, full_probs, atol=1e-6))

    def test_learnable_metric_cache_refreshes_after_update(self):
        torch.manual_seed(0)
        path = MetricInducedGibbsProbPath(
            vocab_size=6,
            emb_dim=1,
            metric="lp",
            lp_order=3.0,
            embed_range="pm1",
            learnable_metric_dim=3,
            metric_interp_lambda=1.0,
        )
        tokens = torch.tensor([[1, 3, 5]], dtype=torch.long)
        dist_before = path.distances_from_tokens(tokens).detach()
        with torch.no_grad():
            assert path.learnable_metric is not None
            path.learnable_metric.codes.mul_(1.1)
        dist_after = path.distances_from_tokens(tokens).detach()
        diff = (dist_after - dist_before).abs().sum()
        self.assertGreater(diff.item(), 0.0)


class TestScheduleUtilities(unittest.TestCase):
    def test_metric_induced_beta_override_matches_manual(self):
        path = MetricInducedGibbsProbPath(
            vocab_size=4,
            emb_dim=1,
            metric="lp",
            lp_order=2.0,
            embed_range="unit",
            a=1.0,
            c=1.0,
        )
        x1 = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        t = torch.tensor([0.25, 0.75])
        beta_values = torch.tensor([0.5, 2.0])
        manual_dist = path.distances_from_tokens(x1)
        expected = torch.softmax(-beta_values.view(-1, 1, 1) * manual_dist, dim=-1)
        result = path.get_prob_distribution_from_tokens(x1, t, beta_values=beta_values)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

        if path.beta_schedule is None:
            schedule = ExpMonotoneRQSSchedule(ExpMonotoneRQSConfig(num_bins=4, tail_bound=3.0))
            path.beta_schedule = schedule
        default_probs = path.get_prob_distribution_from_tokens(x1, t)
        override_probs = path.get_prob_distribution_from_tokens(
            x1, t, beta_schedule=path.beta_schedule
        )
        self.assertTrue(torch.allclose(default_probs, override_probs, atol=1e-6))

    def test_beta_schedule_ema_tracks_schedule(self):
        schedule = ExpMonotoneRQSSchedule(
            ExpMonotoneRQSConfig(num_bins=3, tail_bound=2.0, init_c=1.0, init_a=2.0)
        )
        ema = BetaScheduleEMA(schedule, decay=0.5)
        ema.synchronize_from(schedule)

        t = torch.tensor([0.3])
        base_beta, _ = schedule.beta_and_derivative(t)
        ema_beta, _ = ema.beta_and_derivative(t)
        self.assertTrue(torch.allclose(base_beta, ema_beta, atol=1e-6))

        with torch.no_grad():
            schedule.y0.add_(1.0)
        ema.update(schedule)
        effective_decay = min(0.5, (1 + ema.num_updates.item()) / (10 + ema.num_updates.item()))
        expected_param = (1.0 - effective_decay) * 1.0  # previous teacher value was zero
        self.assertAlmostEqual(float(ema.teacher.y0), expected_param, places=6)
        self.assertEqual(int(ema.num_updates.item()), 1)

        ema_beta_after, _ = ema.beta_and_derivative(t)
        self.assertFalse(torch.allclose(base_beta, ema_beta_after))

        ema.synchronize_from(schedule)
        self.assertEqual(int(ema.num_updates.item()), 0)
        synced_beta, _ = ema.beta_and_derivative(t)
        schedule_beta, _ = schedule.beta_and_derivative(t)
        self.assertTrue(torch.allclose(synced_beta, schedule_beta, atol=1e-6))

    def test_learnable_metric_ema_tracks_metric(self):
        path = MetricInducedGibbsProbPath(
            vocab_size=5,
            emb_dim=1,
            metric="lp",
            lp_order=3.0,
            embed_range="pm1",
            learnable_metric_dim=2,
            metric_interp_lambda=1.0,
        )
        assert path.learnable_metric is not None
        metric_module = path.learnable_metric
        ema = LearnableMetricEMA(metric_module, decay=0.5)
        ema.synchronize_from(metric_module)

        with torch.no_grad():
            metric_module.codes.add_(0.5)
        ema.update(metric_module)
        self.assertEqual(int(ema.num_updates.item()), 1)
        self.assertFalse(
            torch.allclose(ema.teacher.codes, metric_module.codes, atol=1e-6)
        )

        ema.synchronize_from(metric_module)
        self.assertTrue(torch.allclose(ema.teacher.codes, metric_module.codes, atol=1e-6))


class TestMonotoneRQSchedule(unittest.TestCase):
    def test_schedule_monotonicity_and_bounds(self):
        config = MonotoneRQConfig(num_bins=8, beta_min=0.0, beta_max=5.0)
        schedule = MonotoneRQBetaSchedule(config=config)
        t = torch.linspace(1e-3, 1 - 1e-3, steps=32, requires_grad=True)
        beta, d_beta = schedule.beta_and_derivative(t)
        self.assertTrue(torch.all(beta >= config.beta_min - 1e-5))
        self.assertTrue(torch.all(beta <= config.beta_max + 1e-5))
        self.assertTrue(torch.all(d_beta > 0))
        self.assertTrue(torch.all(beta[1:] >= beta[:-1]))
        loss = beta.sum()
        loss.backward()
        grads = [param.grad for param in schedule.parameters()]
        self.assertTrue(all(g is not None for g in grads))


class TestExpMonotoneRQSchedule(unittest.TestCase):
    def test_warm_start_matches_baseline(self):
        config = ExpMonotoneRQSConfig(
            num_bins=5,
            tail_bound=4.0,
            init_c=1.3,
            init_a=3.5,
            t_eps=1e-5,
            logit_eps=1e-6,
        )
        schedule = ExpMonotoneRQSSchedule(config=config).to(dtype=torch.float64)
        t = torch.linspace(1e-3, 1 - 1e-3, steps=64, dtype=torch.float64)
        beta, d_beta = schedule.beta_and_derivative(t)

        t_clamped = t.clamp(min=config.t_eps, max=1.0 - config.t_eps)
        u = 1.0 - t_clamped
        ratio = t_clamped / u
        baseline = config.init_c * (ratio ** config.init_a)
        baseline_deriv = (
            config.init_c
            * config.init_a
            * (ratio ** (config.init_a - 1.0))
            * (1.0 / (u * u))
        )

        self.assertTrue(torch.allclose(beta, baseline, atol=1e-7, rtol=1e-5))
        self.assertTrue(torch.allclose(d_beta, baseline_deriv, atol=1e-7, rtol=1e-5))
        self.assertTrue(torch.all(d_beta > 0))

    def test_sample_uniform_logbeta_round_trip_and_weight(self):
        seed = 1234
        torch.manual_seed(seed)
        config = ExpMonotoneRQSConfig(
            num_bins=4,
            tail_bound=3.0,
            init_c=1.2,
            init_a=2.3,
            t_eps=1e-5,
            logit_eps=1e-6,
        )
        schedule = ExpMonotoneRQSSchedule(config=config).to(dtype=torch.float64)

        batch_shape = (16,)
        lmin, lmax = -1.5, 1.8
        t, weight = schedule.sample_t_uniform_logbeta(batch_shape, lmin, lmax)

        self.assertEqual(t.shape, torch.Size(batch_shape))
        self.assertEqual(weight.shape, torch.Size(batch_shape))
        self.assertTrue(torch.all(t >= config.t_eps))
        self.assertTrue(torch.all(t <= 1.0 - config.t_eps))

        beta, d_beta = schedule.beta_and_derivative(t)
        ell_recovered = beta.log()

        self.assertTrue(torch.all(ell_recovered >= lmin - 1e-6))
        self.assertTrue(torch.all(ell_recovered <= lmax + 1e-6))

        expected_weight = beta / d_beta
        self.assertTrue(torch.all(expected_weight > 0))
        self.assertTrue(
            torch.allclose(weight, expected_weight, atol=1e-6, rtol=1e-5)
        )

        torch.manual_seed(seed)
        expected_ell = torch.empty(
            batch_shape, dtype=ell_recovered.dtype, device=ell_recovered.device
        ).uniform_(lmin, lmax)
        self.assertTrue(
            torch.allclose(ell_recovered, expected_ell, atol=1e-6, rtol=1e-5)
        )

    def test_sample_uniform_logbeta_interval_validation(self):
        schedule = ExpMonotoneRQSSchedule()
        with self.assertRaises(ValueError):
            schedule.sample_t_uniform_logbeta((4,), 1.0, 0.0)


class TestExpRQSInverse(unittest.TestCase):
    def test_inverse_round_trip(self):
        torch.manual_seed(0)
        rqs = _ExpRQS1D(num_bins=5, tail_bound=2.5).to(dtype=torch.float64)
        with torch.no_grad():
            rqs.theta_w.copy_(torch.randn_like(rqs.theta_w))
            rqs.theta_h.copy_(torch.randn_like(rqs.theta_h))
            if rqs.theta_d.numel() > 0:
                rqs.theta_d.copy_(torch.randn_like(rqs.theta_d))

        inputs = torch.linspace(-3.0, 3.0, steps=41, dtype=torch.float64)
        outputs, derivatives = rqs(inputs)
        recovered, recovered_derivatives = rqs.inverse(outputs)

        max_error = (recovered - inputs).abs().max().item()
        self.assertLessEqual(max_error, 5e-6)
        self.assertTrue(torch.all(recovered_derivatives > 0.0))
        self.assertTrue(
            torch.allclose(
                recovered_derivatives, derivatives, atol=1e-6, rtol=1e-5
            )
        )


if __name__ == "__main__":
    unittest.main()
