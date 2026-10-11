"""Artificial-data checks of N01 empirical objectives; no scientific data read."""
import inspect
import unittest

import torch

from n01.losses import (
    N01Loss, PopulationWeights, WaveformDiscriminator, d_hinge,
    fit_loss_statistics, g_hinge, weighted_wasserstein_1d, window_charge,
)


def pulses(count: int = 256, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(480).float()
    shape = torch.where(t < 120, ((t - 100) / 20).clamp_min(0), torch.exp(-(t - 120) / 35))
    amplitudes = torch.linspace(0.06, 0.30, count)[:, None]
    baseline = 0.0006 * torch.randn(count, 480, generator=generator)
    return amplitudes * shape + baseline


class EmpiricalW1Tests(unittest.TestCase):
    def test_known_unequal_count_answer_and_gradients(self):
        x = torch.tensor([0.0, 2.0], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0.5], dtype=torch.float64)
        weights = torch.tensor([0.75, 0.25], dtype=torch.float64, requires_grad=True)
        loss = weighted_wasserstein_1d(x, y, weights)
        self.assertAlmostEqual(loss.item(), 0.75, places=12)
        loss.backward()
        torch.testing.assert_close(x.grad, torch.tensor([-0.75, 0.25], dtype=torch.float64))
        torch.testing.assert_close(weights.grad, torch.tensor([-0.25, 0.75], dtype=torch.float64))
        self.assertAlmostEqual(weighted_wasserstein_1d(x.detach(), torch.tensor([1.0], dtype=torch.float64)).item(), 1.0)

    def test_identical_weighted_distribution_and_duplicates(self):
        x = torch.tensor([[0., 2.], [1., 3.], [1., 3.]], dtype=torch.float64)
        weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        perm = torch.tensor([2, 0, 1])
        self.assertLess(weighted_wasserstein_1d(x, x[perm], weights, weights[perm]).item(), 1e-12)
        self.assertLess(weighted_wasserstein_1d(torch.tensor([0., 1.]), torch.tensor([0., 0., 1., 1.])).item(), 1e-7)

    def test_weight_errors_are_explicit(self):
        for weights in (torch.tensor([-1., 2.]), torch.tensor([0., 0.]), torch.tensor([1.])):
            with self.assertRaises(ValueError):
                weighted_wasserstein_1d(torch.tensor([0., 1.]), torch.tensor([0.]), weights)


class N01LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.real = pulses()
        cls.source = pulses(seed=2)
        cls.stats = fit_loss_statistics(cls.real, cls.source)

    def test_statistics_all_rows_and_train_only(self):
        stats = self.stats
        self.assertEqual(stats["fit_split"], "TRAIN")
        self.assertEqual(stats["validity"]["row_selection"], "none")
        self.assertEqual(sum(stats["validity"]["real_charge_bin_counts"]), len(self.real))
        self.assertEqual(stats["validity"]["real_nonpositive_q_rows"], 0)
        self.assertTrue(all((value["scale"] > 0).all() for value in stats["views"].values()))
        invalid = self.source.clone()
        invalid[0, 5] = float("nan")
        with self.assertRaisesRegex(ValueError, "no rows were dropped"):
            fit_loss_statistics(self.real, invalid)
        bad_source = self.source.clone()
        bad_source[0] = 0
        with self.assertRaisesRegex(ValueError, "no rows dropped"):
            fit_loss_statistics(self.real, bad_source)

    def test_same_distribution_zero_and_component_names(self):
        objective = N01Loss(self.stats)
        values = objective.components({"waveform": self.real}, self.real)
        self.assertEqual(set(values), {"wave", "QPSD", "noise", "regular", "trend"})
        for value in values.values():
            self.assertLess(value.abs().item(), 2e-6)
        self.assertTrue(objective.last_diagnostics["conditional_coverage_complete"])

    def test_finite_generator_gradient_and_detached_comparison_weights(self):
        objective = N01Loss(self.stats)
        fake = (self.source * 1.03 + 1e-5).clone().requires_grad_()
        weights = torch.ones(len(fake), requires_grad=True)
        components = objective.components({"waveform": fake}, self.real, weights)
        total = sum(components.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(fake.grad)
        self.assertTrue(torch.isfinite(fake.grad).all())
        self.assertGreater(fake.grad.norm().item(), 0)
        self.assertIsNone(weights.grad)

    def test_population_gradient_response_detached_and_unequal_counts(self):
        objective = N01Loss(self.stats)
        fake = self.source[:173].clone().requires_grad_()
        population = PopulationWeights(window_charge(fake), torch.linspace(0, 1, len(fake)))
        value = objective.population_objective(fake, self.real, population())
        value.backward()
        self.assertIsNone(fake.grad)
        self.assertTrue(torch.isfinite(population.theta.grad).all())
        self.assertGreater(population.theta.grad.norm().item(), 0)
        self.assertEqual(objective.last_diagnostics["fake_global_rows"], 173)

    def test_outside_bin_rows_kept_and_missing_support_named(self):
        objective = N01Loss(self.stats)
        fake = self.source.clone()
        fake[0, 100:400] *= -1
        fake[-1] *= 20
        output = objective.components({"waveform": fake[:15]}, self.real)
        diagnostics = objective.last_diagnostics
        self.assertTrue(torch.isfinite(output["QPSD"]))
        self.assertEqual(diagnostics["fake_global_rows"], 15)
        self.assertEqual(diagnostics["fake_nonpositive_q_rows"], 1)
        self.assertEqual(diagnostics["conditional_bins_used"], 0)
        self.assertTrue(diagnostics["conditional_rebatch_needed"])
        self.assertTrue(all(not row["supported"] for row in diagnostics["conditional_bins"]))
        objective.components({"waveform": fake}, self.real)
        bins = objective.last_diagnostics["conditional_bins"]
        self.assertEqual(sum(row["fake_count"] for row in bins), len(fake))
        self.assertGreater(bins[0]["fake_count"], 0)
        self.assertGreater(bins[-1]["fake_count"], 0)

    def test_nonpositive_real_q_is_diagnosed_not_removed(self):
        real = self.real.clone()
        real[0, 100:400] *= -1
        stats = fit_loss_statistics(real, self.source)
        self.assertEqual(stats["validity"]["real_nonpositive_q_rows"], 1)
        self.assertEqual(stats["validity"]["real_rows"], len(real))
        self.assertEqual(sum(stats["validity"]["real_charge_bin_counts"]), len(real))

    def test_mean_stage_uses_means_without_width_match(self):
        objective = N01Loss(self.stats)
        fake = self.real * 1.04
        mean = objective.components({"waveform": fake}, self.real, stage="mean_response")
        self.assertTrue(all(torch.isfinite(value) for value in mean.values()))
        self.assertEqual(objective.last_diagnostics["conditional_term"], "mean_PSD")

    def test_noise_feature_shape_baseline_equivalence_and_constant_finiteness(self):
        objective = N01Loss(self.stats)
        features = objective.noise_features(self.real)
        self.assertEqual(tuple(features.shape), (len(self.real), 32))
        torch.testing.assert_close(features, objective.noise_features(self.real[:, :100]))
        self.assertTrue(torch.isfinite(objective.noise_features(torch.zeros(4, 100))).all())

    def test_directions_frozen_seeded_and_global_rng_unchanged(self):
        rng = torch.random.get_rng_state().clone()
        first, second = N01Loss(self.stats, seed=47), N01Loss(self.stats, seed=47)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        torch.testing.assert_close(first.shape_directions, second.shape_directions)
        self.assertEqual(tuple(first.shape_directions.shape), (480, 32))
        self.assertEqual(tuple(first.qpsd_directions.shape), (2, 16))

    def test_discriminator_both_heads_and_all_parameter_gradients(self):
        discriminator = WaveformDiscriminator(self.stats)
        fake = self.source[:8].clone().requires_grad_()
        real_predictions = discriminator(self.real[:9])
        fake_predictions = discriminator(fake)
        self.assertEqual(tuple(fake_predictions[0].shape), (8,))
        self.assertEqual(tuple(fake_predictions[1].shape), (8, 60))
        loss = d_hinge(fake_predictions, real_predictions) + g_hinge(fake_predictions)
        loss.backward()
        for name, parameter in discriminator.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertTrue(torch.isfinite(fake.grad).all())


class PopulationWeightTests(unittest.TestCase):
    def test_bounds_global_normalization_uniform_initialization_no_ids(self):
        q = torch.exp(torch.linspace(-3, 3, 401))
        h = torch.linspace(0, 1, 401)
        population = PopulationWeights(q, h)
        torch.testing.assert_close(population(), torch.ones_like(q))
        with torch.no_grad():
            population.theta.copy_(torch.tensor([100., -100., 100.]))
        weights = population()
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=6)
        self.assertGreaterEqual(weights.min().item(), 0.5 + 0.5 / torch.e - 1e-6)
        self.assertLessEqual(weights.max().item(), 0.5 + 0.5 * torch.e + 1e-6)
        selected = torch.tensor([0, 3, 3, 21])
        torch.testing.assert_close(population(selected), weights[selected])
        self.assertNotAlmostEqual(population(selected).mean().item(), 1.0, places=3)
        self.assertEqual(list(inspect.signature(PopulationWeights).parameters), ["q_source", "h"])
        self.assertEqual(tuple(population.features.shape), (401, 3))
        self.assertEqual(population.theta.numel(), 3)
        self.assertTrue(torch.isfinite(population.penalty()))


if __name__ == "__main__":
    unittest.main()
