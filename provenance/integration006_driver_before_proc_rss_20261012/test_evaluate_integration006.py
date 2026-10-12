"""ArtificialCPU driver tests; no scientific input/checkpoint decode or fits."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from n01 import evaluate_integration006 as e
from n01.electronics_v3 import ObservedBaselineNoiseV3
from n01.extension006 import ConditionalExtension006, HYPOTHESES006
from n01.model import charge, digitize

TMP = Path("E:/GAN/runtime_tmp/N01_integration006_tests")


def configuration():
    return dict(evaluation=dict(draws_per_parent=4, chunk_rows=128, bootstrap_draws=25,
                minimum_parents_per_fixed_bin=1, minimum_rows_per_fixed_bin=4, noise_diagnostic_rows=128),
        seeds=dict(TRAIN_response=101, TRAIN_noise005=102, TRAIN_extension=103,
                   DEV_response=104, DEV_noise005=105, DEV_extension=106, bootstrap_TRAIN=107, bootstrap_DEV=108),
        decision=dict(Q_mean_range_over_REAL_sd_max=.05, PSD_mean_range_over_REAL_sd_max=.05,
                      fixed_bin_PSD_sd_ratio_max=1.10, noise_tail_variance_ratio_max=1.25,
                      signal_baseline_absolute_max_V=1 / 16383, bootstrap_interval=[.025, .975]))


def noise_model():
    index = torch.arange(100, dtype=torch.float64)
    lower = torch.diag(1e-4 * (.7 + index / 200))
    init = dict(length=100, mean_V=(index * 1e-8).tolist(), cholesky_V=lower.tolist(),
                voltage_scale=1e-4, normalized_diagonal_floor=1e-6, initial_log_scale_std=.2,
                metadata=dict(role="fabricated_for_CPU_driver_test"))
    return ObservedBaselineNoiseV3(init).eval().requires_grad_(False)


class FakeResponse:
    """Synthetic components expose the explicit-input candidate dataflow."""
    def __init__(self, multiplier=1.):
        self.multiplier = multiplier
        self.received = []

    def __call__(self, source, noise, adc_ste=False):
        self.received.append(e.tensor_dict_sha(noise))
        batch = len(source)
        shape = torch.zeros((batch, 480), dtype=torch.float64)
        shape[:, 100:400] = 1 / 300
        q = charge(source.to(torch.float64)) * self.multiplier * (1 + .01 * noise["phys"][:, 0].to(torch.float64))
        local = torch.zeros_like(shape)
        electronic = noise["electronics_offset"].to(torch.float64).reshape(batch, 1).expand(-1, 480) * 1e-4
        signal = q[:, None] * shape + local
        pre = signal + electronic
        return dict(qsig=q, unit_shape=shape, signal_noise=local, electronics_noise=electronic,
                    pre_adc=pre, waveform=digitize(pre, ste=adc_ste))


class Integration006Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        TMP.mkdir(parents=True, exist_ok=True)
        cls.legacy_hashes = {file: e.sha256(file) for file in (
            ROOT / "src/n01/data.py", ROOT / "src/n01/model.py", ROOT / "src/n01/evaluate.py",
            ROOT / "src/n01/electronics_v3.py", ROOT / "integration006_config.json", ROOT / "docs/INTEGRATION_006_PROTOCOL.md")}

    @classmethod
    def tearDownClass(cls):
        for file, expected in cls.legacy_hashes.items():
            if e.sha256(file) != expected:
                raise AssertionError("Frozen source/config/protocol changed: " + str(file))

    def test_readouts_signed_guards_rawPSD_and_saturation_retain_rows(self):
        pre = np.zeros((4, 480), np.float64)
        pre[1, 100:400] = -.001
        pre[2, 100:400] = 1e-7
        pre[3, 120] = .7
        wave = e.array(digitize(torch.from_numpy(pre), ste=False))
        observed = e.readouts(wave, .001, pre_adc=pre)
        self.assertEqual(observed["Q"].shape, (4,))
        np.testing.assert_array_equal(observed["Q_nonpositive"], observed["Q"] <= 0)
        np.testing.assert_array_equal(observed["PSD_guarded"], abs(observed["Q"]) < .001)
        np.testing.assert_allclose(observed["tail"], wave[:, 150:400].sum(1) - 2.5 * wave[:, :100].sum(1))
        signed = np.where(observed["Q"] < 0, -1., 1.) * np.maximum(abs(observed["Q"]), .001)
        np.testing.assert_allclose(observed["PSD"], observed["tail"] / signed)
        self.assertEqual(observed["saturation_samples"][3], 1)
        np.testing.assert_allclose(observed["ADCcharge_change"], observed["Q"] - e.charge_np(pre))
        zero = e.readouts(np.zeros((1, 480)), .001)
        self.assertTrue(np.isnan(zero["PSD_raw"][0]))
        self.assertTrue(zero["Q_nonpositive"][0] and zero["PSD_guarded"][0])

    def test_bins_equal_boundary_goes_left_and_nonpositive_is_retained(self):
        stats = dict(q_reference=1., q_floor=.01, charge_bin_boundaries=[-1.5, -1., -.5, 0., .5, 1., 1.5])
        q = np.array([-1., 0., .01, 1., np.exp(3.)])
        bins = e.charge_bins(q, stats)
        self.assertEqual(bins[3], 3)
        self.assertEqual(bins[-1], 7)
        self.assertEqual(len(bins), len(q))
        coordinate = np.where(q >= .01, np.log(np.maximum(q, .01)), np.log(.01) + np.arcsinh((q - .01) / .01))
        np.testing.assert_array_equal(bins, np.searchsorted(stats["charge_bin_boundaries"], coordinate, side="left"))

    def test_centered_online_moments_equal_batch_and_empty_bins_undefined(self):
        rng = np.random.default_rng(3)
        x = 100 + rng.normal(size=(17, 5, 480)) * 1e-3
        moments = e.Moments((5, 480))
        moments.add(x[:8])
        moments.add(x[8:])
        mean, variance = moments.values()
        np.testing.assert_allclose(mean, x.mean(0), atol=1e-12)
        np.testing.assert_allclose(variance, x.var(0), atol=1e-14)
        self.assertTrue(np.isnan(e.Moments((5, 480)).values()[0]).all())
        covariance = e.CovarianceMoments(3)
        y = x[:, 0, :3]
        covariance.add(y[:8])
        covariance.add(y[8:])
        np.testing.assert_allclose(covariance.values()[1], np.cov(y, rowvar=False, ddof=0), atol=1e-14)

    def test_three_independent_streams_explicit_pairing_and_raw_hash_contract(self):
        cfg = configuration()
        _, first = e.streams_for_split("TRAIN", cfg)
        _, second = e.streams_for_split("TRAIN", cfg)
        global_before = torch.get_rng_state().clone()
        for rows in (128, 4):
            a, b, c = e.draw_latents(rows, first)
            aa, bb, cc = e.draw_latents(rows, second)
            self.assertEqual(e.tensor_dict_sha(a), e.tensor_dict_sha(aa))
            self.assertEqual(e.tensor_dict_sha(b), e.tensor_dict_sha(bb))
            self.assertEqual(e.array_sha(c), e.array_sha(cc))
            self.assertTrue(all(value.dtype == torch.float32 and value.device.type == "cpu" for value in a.values()))
            expected = hashlib.sha256()
            for key in sorted(a):
                expected.update(key.encode() + b"\0" + hashlib.sha256(a[key].numpy().tobytes()).hexdigest().encode() + b"\n")
            self.assertEqual(e.tensor_dict_sha(a), expected.hexdigest())
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        for key in first:
            self.assertTrue(torch.equal(first[key].get_state(), second[key].get_state()))

    def test_response_candidates_use_direct_components_and_single_ADC(self):
        cfg = configuration()
        _, streams = e.streams_for_split("TRAIN", cfg)
        response, baseline, epsilon = e.draw_latents(4, streams)
        model = noise_model()
        ext = ConditionalExtension006(model, list(HYPOTHESES006))
        counts, engineering = {}, {}
        prefix, scale, noises = e.paired_noise(4, model, ext, baseline, epsilon, split="TRAIN", counts=counts, engineering=engineering)
        source = torch.zeros(4, 480, dtype=torch.float32)
        source[:, 100:400] = .001
        result = e.response_chunk(FakeResponse(), source, response, noises, prefix, split="TRAIN", arm="full", counts=counts, engineering=engineering)
        for i, h in enumerate(e.HYPOTHESES):
            expected_pre = result["signal_pre"] + noises[h][:, 0]
            self.assertTrue(torch.equal(result["pre_adc"][i + 2], expected_pre))
            self.assertTrue(torch.equal(result["waveform"][i + 2], digitize(expected_pre, ste=False)))
            self.assertEqual(counts["candidate_ADC"]["TRAIN/full/" + h], dict(batches=1, events=4))
            self.assertEqual(counts["candidate_ADC_attempted"]["TRAIN/full/" + h], dict(batches=1, events=4))
        self.assertTrue(all(item["passed"] for item in engineering.values()))
        self.assertTrue(torch.equal(result["waveform"][2, :, :100], result["waveform"][4, :, :100]))

    def test_failed_forward_is_attempted_not_successful(self):
        def failure(*args, **kwargs):
            raise RuntimeError("artificial forward failure")
        counts = {}
        with self.assertRaisesRegex(RuntimeError, "artificial"):
            e.response_chunk(failure, torch.zeros(4, 480), {}, {}, torch.zeros(4, 1, 100),
                             split="TRAIN", arm="full", counts=counts, engineering={})
        self.assertEqual(counts["model_forward_attempted"]["TRAIN/full"]["events"], 4)
        self.assertNotIn("model_forward", counts)

    def test_artificial_split_parent_coverage_chunks_replay_and_fixed_bins(self):
        cfg = configuration()
        source = np.zeros((40, 480), np.float32)
        source[:, 100:400] = np.linspace(.0007, .004, 40)[:, None]
        real = e.array(digitize(torch.from_numpy(source[:12]).to(torch.float64), ste=False))
        data = dict(source=source, real=real, source_ids=np.arange(40, dtype=np.int64) + 50,
                    real_ids=np.arange(12, dtype=np.int64) + 100, real_groups=np.arange(12, dtype=np.int64) + 200,
                    h=np.linspace(0, 1, 40))
        stats = dict(loss=dict(q_reference=.5, q_floor=1e-5, charge_bin_boundaries=np.linspace(-1, .9, 7).tolist()))
        model = noise_model()
        ext = ConditionalExtension006(model, list(HYPOTHESES006))
        models = dict(deterministic=FakeResponse(), full=FakeResponse(1.1))
        class FakeRuntime:
            device = torch.device("cpu")
            def check(self, out=None):
                pass
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            counts, engineering = {}, {}
            obs, curves, replay, samples = e.split_evaluation("TRAIN", data, cfg, models, model, ext, stats,
                FakeRuntime(), Path(folder), counts, engineering)
            self.assertEqual(models["deterministic"].received, models["full"].received)
            np.testing.assert_array_equal(obs["TRAIN_source_parent_row"], np.repeat(np.arange(40), 4))
            np.testing.assert_array_equal(obs["TRAIN_draw_index"], np.tile(np.arange(4), 40))
            saved = torch.load(Path(folder) / "latents/TRAIN.pt", weights_only=False)
            self.assertEqual([(x["row_start"], x["row_stop"]) for x in saved["chunks"]], [(0, 128), (128, 160)])
            self.assertEqual(replay["source"].shape, (128, 480))
            self.assertEqual(replay["arms"]["full"]["expected_views"].shape, (5, 128, 480))
            self.assertEqual(samples["arms"]["full"]["noise_pre"].shape, (3, 128, 480))
            for arm in e.ARMS:
                p = "TRAIN_" + arm + "_"
                self.assertEqual(obs[p + "Q"].shape, (5, 160))
                np.testing.assert_array_equal(obs[p + "fixed_bins"], e.charge_bins(obs[p + "signal_pre_Q"], stats["loss"]))
                self.assertEqual(curves[p + "fixed_bin_mean"].shape, (8, 5, 480))
                np.testing.assert_array_equal(obs[p + "migration"].sum((1, 2)), np.full(5, 160))
                self.assertEqual(counts["model_forward"]["TRAIN/" + arm], dict(batches=2, events=160))
                self.assertEqual(counts["model_forward_attempted"]["TRAIN/" + arm], dict(batches=2, events=160))

    def test_bootstrap_matches_explicit_parent_resampling_and_constant_point3_is_undefined(self):
        cfg = configuration()
        n = 16
        parent = np.repeat(np.arange(n), 4)
        obs = {"TRAIN_source_parent_ids": np.arange(n), "TRAIN_source_parent_row": parent}
        rng = np.random.default_rng(40)
        for arm in e.ARMS:
            p = "TRAIN_" + arm + "_"
            base = rng.normal(.3, .02, (n * 4,))
            obs[p + "Q"] = np.array([base + i / 100 for i in range(5)]) + 1
            obs[p + "PSD"] = np.array([base * (1 + i / 200) for i in range(5)])
            obs[p + "fixed_bins"] = parent % 8
        payload, summary = e.parent_bootstrap(obs, "TRAIN", cfg, (.5, .1))
        generator = np.random.default_rng(cfg["seeds"]["bootstrap_TRAIN"])
        for draw in range(cfg["evaluation"]["bootstrap_draws"]):
            sampled = generator.integers(0, n, size=n)
            expanded = (sampled[:, None] * 4 + np.arange(4)).reshape(-1)
            for j, arm in enumerate(e.ARMS):
                p = "TRAIN_" + arm + "_"
                values = payload["TRAIN_metric_values"][draw, j * 11:(j + 1) * 11]
                self.assertAlmostEqual(values[0], np.ptp(obs[p + "Q"][2:, expanded].mean(1)) / .5)
                self.assertAlmostEqual(values[1], np.ptp(obs[p + "PSD"][2:, expanded].mean(1)) / .1)
                for b in range(8):
                    selected = expanded[obs[p + "fixed_bins"][expanded] == b]
                    expected = e.sd_ratio(obs[p + "PSD"][2:, selected].std(1)) if len(selected) else np.nan
                    np.testing.assert_allclose(values[b + 2], expected, atol=1e-12, equal_nan=True)
            self.assertEqual(payload["TRAIN_resample_parent_hashes"][draw], e.array_sha(sampled))
        for arm in e.ARMS:
            obs["TRAIN_" + arm + "_PSD"][:] = .3
        payload, _ = e.parent_bootstrap(obs, "TRAIN", cfg, (.5, .1))
        self.assertTrue(np.isnan(payload["TRAIN_point_metrics"][[2, 10, 13, 21]]).all())
        self.assertTrue(np.isnan(payload["TRAIN_intervals"][[2, 10, 13, 21]]).all())
        # Near-constant widths need centered arithmetic too: the literal
        # second-moment subtraction loses these1e-10 fluctuations around.3.
        deviation = (np.arange(n * 4) % 4 - 1.5) * 1e-10
        for arm in e.ARMS:
            for h in range(3):
                obs["TRAIN_" + arm + "_PSD"][h + 2] = .3 + (h + 1) * deviation
        payload, _ = e.parent_bootstrap(obs, "TRAIN", cfg, (.5, .1))
        for j, arm in enumerate(e.ARMS):
            p = "TRAIN_" + arm + "_"
            for b in range(8):
                selected = obs[p + "PSD"][2:, obs[p + "fixed_bins"] == b]
                centered = selected - selected[:, :1]
                expected = e.sd_ratio(centered.std(1, ddof=0))
                self.assertAlmostEqual(payload["TRAIN_point_metrics"][j * 11 + 2 + b], expected, places=12)

    def test_extended_quantiles_preserve_inf_and_never_drop_nan(self):
        values = np.column_stack((np.arange(10), np.full(10, np.inf), np.full(10, np.nan)))
        intervals = e.extended_quantile(values, [.025, .975])
        np.testing.assert_allclose(intervals[0], np.quantile(values[:, 0], [.025, .975], method="linear"))
        self.assertTrue(np.isinf(intervals[1]).all())
        self.assertTrue(np.isnan(intervals[2]).all())
        self.assertTrue(np.isnan(e.sd_ratio(np.zeros(3))))
        self.assertEqual(e.sd_ratio(np.array([0., 1., 2.])), np.inf)

    def test_decision_priority_support_zero_width_and_DEV_descriptive_only(self):
        cfg = configuration()
        obs = {}
        for split in e.SPLITS:
            for arm in e.ARMS:
                p = split + "_" + arm + "_"
                obs[p + "signal_baseline_max_per_event"] = np.array([0. if split == "TRAIN" else 10.])
                obs[p + "Q_nonpositive"] = np.zeros((5, 4), bool)
                obs[p + "PSD_guarded"] = np.zeros((5, 4), bool)
        names = [arm + "/" + metric for arm in e.ARMS for metric in e.METRIC_ORDER]
        points = np.tile(np.r_[.01, .01, np.full(9, 1.01)], 2)
        train = dict(metric_names=names, point_metrics=points, intervals=np.repeat(points[:, None], 2, axis=1),
            support={arm: dict(rows=np.full(8, 4), parents=np.ones(8), supported=np.ones(8, bool)) for arm in e.ARMS})
        outcome = lambda t=train, ratio=1.: e.make_decision(cfg, obs, t, ratio, {})["outcome"]
        self.assertEqual(outcome(), "sensitivity_within_family")
        obs["TRAIN_full_Q_nonpositive"][1, 0] = True
        self.assertEqual(outcome(), "sensitivity_within_family")
        obs["TRAIN_full_Q_nonpositive"][2, 0] = True
        self.assertEqual(outcome(), "interface_fail")
        obs["TRAIN_full_Q_nonpositive"][2, 0] = False
        self.assertEqual(outcome(ratio=1.3), "sensitivity_detected")
        train["intervals"][0] = [.04, .06]
        self.assertEqual(outcome(), "MC_inconclusive")
        train["intervals"][0] = [.06, .07]
        self.assertEqual(outcome(), "sensitivity_detected")
        train["intervals"][0] = [.01, .02]
        train["intervals"][2] = [np.nan, np.nan]
        self.assertEqual(outcome(), "MC_inconclusive")
        train["intervals"][2] = [np.inf, np.inf]
        self.assertEqual(outcome(), "sensitivity_detected")
        train["intervals"][2] = [1., 1.05]
        train["support"]["full"]["supported"][7] = False
        self.assertEqual(outcome(), "MC_inconclusive")
        obs["TRAIN_full_signal_baseline_max_per_event"][0] = .001
        self.assertEqual(outcome(), "interface_fail")

    def test_binding_failure_never_decodes_and_preserves_failure_no_retry(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            folder = Path(folder)
            config = folder / "config.json"
            config.write_text("{}", encoding="utf8")
            manifest = folder / "manifest.json"
            manifest.write_text("{}", encoding="utf8")
            out = folder / "output"
            with patch.object(e, "frozen_models", side_effect=AssertionError("Must never decode checkpoints")), \
                 patch.object(e, "load_training", side_effect=AssertionError("Must never decode arrays")), \
                 patch.object(e, "load_validation", side_effect=AssertionError("Must never decode arrays")):
                with self.assertRaisesRegex(ValueError, "config mismatch"):
                    e.run(config, folder / "unused.npz", folder / "ids.npz", out, source_manifest=manifest, device="cpu")
            failure = json.loads((out / "FAILURE.json").read_text(encoding="utf8"))
            self.assertEqual(failure["actual_counts"], {})
            self.assertEqual(failure["updates"], e.UPDATES)
            original = (out / "FAILURE.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "no retry"):
                e.run(config, folder / "unused.npz", folder / "ids.npz", out, source_manifest=manifest, device="cpu")
            self.assertEqual((out / "FAILURE.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
