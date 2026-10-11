"""Artificial CPU semantic tests, not a physical or real-data acceptance gate."""
from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from n01.model import (ADC_LEVELS, N01Response, InvalidWaveformError,
                       PhysicalHead, charge, digitize, causal_fir,
                       cubic_expansion, relative_time_warp, support_gate,
                       deterministic_linear_upsample_2x, deterministic_control_average)


torch.set_num_threads(2)


def pulse(batch: int = 3) -> torch.Tensor:
    t = torch.arange(480, dtype=torch.float64)
    elapsed = (t - 116.0).clamp_min(0.0)
    shape = (1 - torch.exp(-elapsed / 6.0)) * torch.exp(-elapsed / 50.0)
    shape = shape / shape.max()
    amplitudes = torch.linspace(0.04, 0.12, batch, dtype=torch.float64)
    return (amplitudes[:, None] * shape[None]).float()


def seeded_model(variant: str = "full") -> N01Response:
    torch.manual_seed(734)
    return N01Response(variant=variant)


def draws(model: N01Response, batch: int) -> dict[str, torch.Tensor]:
    return model.sample_noise(batch, "cpu", torch.Generator().manual_seed(291))


def number(value: torch.Tensor) -> float:
    return float(value.detach())


class ModelTests(unittest.TestCase):
    def test_charge_identity_electronics_retained_and_canonical_adc(self):
        model = seeded_model()
        output = model(pulse(), noise=draws(model, 3))
        self.assertEqual(output["waveform"].shape, (3, 480))
        self.assertEqual(output["pre_adc"].dtype, torch.float64)
        self.assertLess(number((charge(output["unit_shape"]) - 1).abs().max()), 1e-12)
        self.assertLess(number(charge(output["signal_noise"]).abs().max()), 1e-13)
        self.assertLess(number(output["charge_identity_residual"].abs().max()), 1e-12)
        self.assertGreater(number(output["electronics_charge"].abs().max()), 1e-3)
        predicted_qsig = output["qbar"] * (output["physical_eta"][:, 0] - .5 * output["sigma_q2"]).exp()
        self.assertTrue(torch.equal(output["qsig"], predicted_qsig))
        self.assertTrue(torch.equal(output["waveform"], digitize(output["pre_adc"], False)))
        index = (output["waveform"] + .5) * ADC_LEVELS
        self.assertLess(number((index - index.round()).abs().max()), 1e-10)

    def test_deterministic_upsample_forward_and_gradient_agree(self):
        generator = torch.Generator().manual_seed(771)
        for dtype in (torch.float32, torch.float64):
            for length in (1, 3, 30, 60, 120, 240):
                x = torch.randn(2, 4, length, dtype=dtype, generator=generator, requires_grad=True)
                fixed = deterministic_linear_upsample_2x(x)
                reference = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
                tolerance = 3e-7 if dtype == torch.float32 else 2e-15
                self.assertTrue(torch.allclose(fixed, reference, atol=tolerance, rtol=tolerance))
                upstream = torch.randn(fixed.shape, dtype=dtype, generator=generator)
                grad_fixed = torch.autograd.grad(fixed, x, upstream, retain_graph=True)[0]
                grad_reference = torch.autograd.grad(reference, x, upstream)[0]
                self.assertTrue(torch.allclose(grad_fixed, grad_reference, atol=tolerance, rtol=tolerance))

    def test_deterministic_control_average_forward_and_gradient_agree(self):
        generator = torch.Generator().manual_seed(887)
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(2, 8, 480, dtype=dtype, generator=generator, requires_grad=True)
            fixed = deterministic_control_average(x)
            reference = F.adaptive_avg_pool1d(x, 60)
            tolerance = 2e-7 if dtype == torch.float32 else 1e-15
            self.assertTrue(torch.allclose(fixed, reference, atol=tolerance, rtol=tolerance))
            upstream = torch.randn(fixed.shape, dtype=dtype, generator=generator)
            grad_fixed = torch.autograd.grad(fixed, x, upstream, retain_graph=True)[0]
            grad_reference = torch.autograd.grad(reference, x, upstream)[0]
            self.assertTrue(torch.allclose(grad_fixed, grad_reference, atol=tolerance, rtol=tolerance))

    def test_adc_ste_preserves_forward_and_masks_saturation(self):
        x = torch.tensor([-.6, -.5, -.1, .0, .2, .5, .7], dtype=torch.float64, requires_grad=True)
        y = digitize(x, True)
        self.assertTrue(torch.equal(y, digitize(x, False)))
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([0., 1., 1., 1., 1., 1., 0.], dtype=torch.float64)))

    def test_monotone_fixed_shape_spline_including_extrapolation(self):
        model = seeded_model()
        logq = torch.linspace(math.log(.001), math.log(1000), 151, dtype=torch.float64, requires_grad=True)
        u = pulse(1).double() / pulse(1).double().amax()
        shape = u.expand(logq.numel(), -1)
        initial = model.mean_charge(logq, shape)
        self.assertLess(number((initial / logq.exp() - 1).abs().max()), 1e-12)
        with torch.no_grad():
            model.charge_head.parameters_from_shape.weight.normal_(0.0, .15)
            model.charge_head.parameters_from_shape.bias.uniform_(-.8, .8)
        q = model.mean_charge(logq, shape)
        self.assertTrue(bool((q[1:] > q[:-1]).all()))
        gradient = torch.autograd.grad(q.log().sum(), logq)[0]
        self.assertTrue(bool((gradient > 0).all()))
        # The same normalized shape is its only parameter-conditioning input.
        encoded = model.shape_encoder(model._features(u[:, None]))
        scaled = model.shape_encoder(model._features((u * 7 / 7)[:, None]))
        self.assertTrue(torch.equal(encoded, scaled))

    def test_joint_covariance_and_lognormal_mean_variance(self):
        torch.manual_seed(221)
        head = PhysicalHead(10)
        with torch.no_grad():
            head.base[-1].bias[1] = .7  # nonzero Q/time covariance
            head.base[-1].bias[3] = -.4  # Q/shape covariance
        lower = head(torch.zeros(1, 3)).double()[0]
        self.assertTrue(torch.equal(lower, lower.tril()))
        self.assertEqual(head.base[-1].out_features, 55)
        latent = torch.randn(65536, 10, dtype=torch.float64)
        eta = latent @ lower.T
        expected_cov = lower @ lower.T
        observed_cov = torch.cov(eta.T)
        self.assertLess(number((observed_cov - expected_cov).abs().max()), 2e-4)
        relative_q = (eta[:, 0] - .5 * expected_cov[0, 0]).exp()
        relative_a = (eta[:, 1] - .5 * expected_cov[1, 1]).exp()
        self.assertLess(abs(number(relative_q.mean()) - 1), 8e-4)
        expected_variance = expected_cov[0, 0].exp() - 1
        self.assertLess(abs(number(relative_q.var() / expected_variance) - 1), .05)
        observed_qa = torch.cov(torch.stack((relative_q, relative_a)))[0, 1]
        expected_qa = expected_cov[0, 1].exp() - 1
        self.assertLess(abs(number(observed_qa - expected_qa)), 6e-5)

    def test_initial_shape_scale_is_stable_for_an_artificial_narrow_pulse(self):
        model = seeded_model()
        t = torch.arange(480, dtype=torch.float64)
        r = (t - 116.).clamp_min(0.)
        x = (1 - torch.exp(-r / 4.)) * torch.exp(-r / 13.)
        x = (.05 * x / x.max()).float()[None]
        noise = {key: torch.zeros_like(value) for key, value in draws(model, 1).items()}
        # All draws share the same input-dependent basis. Evaluate that basis
        # once, then probe the exact vectorized stochastic shape assembly.
        with torch.no_grad():
            output = model(x, noise=noise)
            latent = torch.randn(2048, 10, dtype=torch.float64, generator=torch.Generator().manual_seed(123))
            lower = output["physical_cholesky"][0]
            eta = latent @ lower.T
            scale = (eta[:, 1] - .5 * lower[1].square().sum()).exp()
            u = (x.double() / x.double().amax())[:, None]
            source = model.response_filter(u).expand(2048, -1, -1)
            warped = relative_time_warp(source, scale)
            qv = charge(warped).squeeze(1) + eta[:, 2:] @ charge(output["basis"][0])
            qv = qv + charge(output["mean_residual"])[0]
            self.assertTrue(bool((qv > 1e-6).all()))

    def test_explicit_replay_and_batch_permutation(self):
        model = seeded_model()
        x, noise = pulse(4), draws(model, 4)
        first = model(x, noise=noise)
        # Automatic RNG state changes do not affect an explicit-noise replay.
        torch.randn(71)
        second = model(x, noise=noise)
        for key in ("waveform", "pre_adc", "qsig", "basis", "signal_noise"):
            self.assertTrue(torch.equal(first[key], second[key]), key)
        order = torch.tensor([2, 0, 3, 1])
        shuffled = model(x[order], noise={key: value[order] for key, value in noise.items()})
        for key in ("waveform", "pre_adc", "qsig", "basis", "signal_noise"):
            self.assertTrue(torch.allclose(first[key][order], shuffled[key], atol=1e-12, rtol=1e-10), key)

    def test_causal_fir_has_no_wrap_and_valid_history(self):
        x = torch.zeros(1, 1, 480, dtype=torch.float64)
        x[..., -1] = 1
        taps = torch.zeros(65, dtype=torch.float64)
        taps[2] = 1
        self.assertEqual(number(causal_fir(x, taps).abs().sum()), 0.)
        x[..., -1] = 0
        x[..., 130] = 1
        result = causal_fir(x, taps)
        self.assertEqual(number(result[..., 132]), 1.)
        self.assertEqual(number(result[..., :130].abs().sum()), 0.)
        extended = torch.zeros(1, 1, 544, dtype=torch.float64)
        extended[..., 0] = 1
        taps.zero_()
        taps[-1] = 1
        valid = causal_fir(extended, taps, extended=True)
        self.assertEqual(valid.shape, (1, 1, 480))
        self.assertEqual(number(valid[..., 0]), 1.)
        self.assertEqual(number(valid[..., 1:].abs().sum()), 0.)

    def test_smooth_bases_and_no_added_baseline_leakage(self):
        model = seeded_model()
        with torch.no_grad():
            model.mean_head.bias.fill_(.1)
        output = model(pulse(), noise=draws(model, 3))
        for key in ("basis", "mean_residual", "signal_noise", "signal_envelope"):
            self.assertEqual(number(output[key][..., :100].abs().sum()), 0., key)
            self.assertEqual(number(output[key][..., -1].abs().sum()), 0., key)
        self.assertLess(number((output["basis"].square().mean(-1) - 1).abs().max()), 1e-12)
        matrix = cubic_expansion()
        self.assertTrue(bool((matrix >= 0).all()))
        self.assertLess(number((matrix.sum(1) - 1).abs().max()), 1e-12)
        points = torch.tensor([99.9, 100., 100.1, 114.9, 115., 115.1], dtype=torch.float64, requires_grad=True)
        gate = support_gate(points)
        gradient = torch.autograd.grad(gate.sum(), points, create_graph=True)[0]
        curvature = torch.autograd.grad(gradient.sum(), points)[0]
        self.assertEqual(number(gradient[1]), 0.)
        self.assertLess(abs(number(curvature[1])), 1e-12)
        self.assertLess(abs(number(gradient[4])), 1e-12)
        self.assertLess(abs(number(curvature[4])), 1e-12)

    def test_warp_anchor_and_finite_scale_gradient(self):
        x = pulse(1).double()[:, None]
        scales = torch.tensor([.8, 1., 1.3], dtype=torch.float64, requires_grad=True)
        output = relative_time_warp(x.expand(3, -1, -1), scales)
        self.assertTrue(torch.equal(output[:, 0, 120], x[0, 0, 120].expand(3)))
        self.assertTrue(torch.equal(output[1], x[0]))
        output.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(scales.grad).all()))
        self.assertGreater(number(scales.grad.abs().sum()), 0.)

    def test_all_variants_and_stage_gradients(self):
        for variant in ("full", "deterministic", "simple"):
            for stage in N01Response.stages:
                with self.subTest(variant=variant, stage=stage):
                    model = seeded_model(variant)
                    model.set_stage(stage)
                    output = model(pulse(2), noise=draws(model, 2))
                    for key, value in output.items():
                        self.assertTrue(bool(torch.isfinite(value).all()), key)
                    objective = output["pre_adc"].square().mean() + output["qsig"].log().mean() + .05 * output["regular"]
                    objective.backward()
                    for name, p in model.named_parameters():
                        if p.requires_grad:
                            self.assertIsNotNone(p.grad, name)
                            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
                        else:
                            self.assertIsNone(p.grad, name)
                    self.assertEqual(number(output["trend"]), 0.)
                    if stage == "baseline":
                        self.assertTrue(model.electronics.log_rms.requires_grad)
                    else:
                        self.assertFalse(model.electronics.log_rms.requires_grad)
                    if variant == "deterministic" or stage == "mean_response":
                        self.assertTrue(torch.equal(output["qsig"], output["qbar"]))
                        self.assertEqual(number(output["physical_cholesky"].abs().sum()), 0.)
                    else:
                        self.assertGreater(number(output["sigma_q2"].min()), 0.)
                    if variant == "simple":
                        self.assertIsNone(model.unet)
                        self.assertEqual(number(output["basis"].abs().sum()), 0.)

    def test_invalid_sources_and_invalid_generated_shape_are_explicit(self):
        model = seeded_model()
        x = pulse()
        x[1] *= 1e-9
        with self.assertRaises(InvalidWaveformError) as caught:
            model(x, noise=draws(model, 3))
        self.assertEqual(caught.exception.rows, [1])
        x = pulse()
        x[2] *= -1
        with self.assertRaises(InvalidWaveformError):
            model(x, noise=draws(model, 3))
        model.set_stage("mean_response")
        with torch.no_grad():
            model.mean_head.bias.fill_(-100.)
            model.mean_time[-1].bias.fill_(-100.)
        with self.assertRaises(InvalidWaveformError) as caught:
            model(pulse(), noise=draws(model, 3))
        self.assertIn("Q(v)", caught.exception.reason)

    def test_baseline_shared_copy_including_covariance_and_electronics(self):
        baseline = seeded_model()
        baseline.set_stage("baseline")
        with torch.no_grad():
            baseline.physical_head.base[-1].bias[:3].add_(.3)
            baseline.electronics.offset_mean.fill_(1e-4)
            baseline.charge_head.parameters_from_shape.bias.fill_(.1)
        noise = draws(baseline, 3)
        reference = baseline(pulse(), noise=noise)
        for variant in ("full", "deterministic", "simple"):
            target = seeded_model(variant)
            target.load_shared_baseline(baseline)
            target.set_stage("baseline")
            result = target(pulse(), noise=noise)
            self.assertTrue(torch.equal(reference["electronics_noise"], result["electronics_noise"]))
            self.assertTrue(torch.equal(reference["qbar"], result["qbar"]))
            if variant != "deterministic":
                self.assertTrue(torch.equal(reference["physical_cholesky"], result["physical_cholesky"]))
                self.assertTrue(torch.equal(reference["pre_adc"], result["pre_adc"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
