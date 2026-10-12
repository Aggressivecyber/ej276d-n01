"""ArtificialCPU driver tests; no scientific input/checkpoint decode or fits."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
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

    def state_dict(self):
        return {"artificial_multiplier": torch.tensor(self.multiplier, dtype=torch.float64)}


def artificial_dataset(n, real_rows, split, *, id_kind="U"):
    source = np.zeros((n, 480), np.float32)
    source[:, 100:400] = np.linspace(.0007, .004, n)[:, None]
    levels = np.linspace(.0008, .0045, real_rows)
    real_pre = np.zeros((real_rows, 480), np.float64)
    real_pre[:, 100:150] = levels[:, None]
    real_pre[:, 150:400] = (levels * np.linspace(.65, 1.1, real_rows))[:, None]
    real = e.array(digitize(torch.from_numpy(real_pre), ste=False))
    if id_kind == "U":
        source_ids = np.array([split + "_源父_" + str(i) for i in range(n)], dtype="U" + ("24" if split == "TRAIN" else "30"))
        real_ids = np.array([split + "_实测原件_" + str(i) for i in range(real_rows)], dtype="U" + ("26" if split == "TRAIN" else "32"))
    elif id_kind == "S":
        source_ids = np.array([split + "_parent_" + str(i) for i in range(n)], dtype="S24")
        real_ids = np.array([split + "_REAL_" + str(i) for i in range(real_rows)], dtype="S24")
    else:
        offset = 1000 if split == "TRAIN" else 2000
        source_ids = np.arange(n, dtype=np.int32) + offset
        real_ids = np.arange(real_rows, dtype=np.int64) + offset * 10
    group_offset = 10000 if split == "TRAIN" else 20000
    return dict(source=source, real=real, source_ids=source_ids, real_ids=real_ids,
                real_groups=np.arange(real_rows, dtype=np.int64) + group_offset, h=np.linspace(0, 1, n))


def independent_verifier():
    spec = importlib.util.spec_from_file_location("artificial006_verifier", ROOT / "scripts/verify_integration006.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Integration006Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        TMP.mkdir(parents=True, exist_ok=True)
        cls.legacy_hashes = {file: e.sha256(file) for file in (
            ROOT / "src/n01/data.py", ROOT / "src/n01/model.py", ROOT / "src/n01/evaluate.py",
            ROOT / "src/n01/electronics_v3.py", ROOT / "integration006_config.json", ROOT / "docs/INTEGRATION_006_PROTOCOL.md",
            ROOT / "scripts/supervise.py")}

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

    def test_exact_metadata_tensor_JSON_scalar_vector_float32_float64(self):
        tensors = {"float32_scalar": torch.tensor(.3, dtype=torch.float32),
                   "float64_scalar": torch.tensor(.3, dtype=torch.float64),
                   "float32_vector": torch.tensor([.1, -.3, -0.], dtype=torch.float32),
                   "nested": {"float64_vector": torch.tensor([[.1, -.3], [1., -0.]], dtype=torch.float64)},
                   "integer_scalar": torch.tensor(7), "boolean_vector": torch.tensor([True, False])}
        native = {"float32_scalar": tensors["float32_scalar"].tolist(),
                  "float64_scalar": .3, "float32_vector": tensors["float32_vector"].tolist(),
                  "nested": {"float64_vector": [[.1, -.3], [1., -0.]]},
                  "integer_scalar": 7, "boolean_vector": [True, False]}
        # Parse a JSON round trip as the frozen statistics file would be read.
        native = json.loads(json.dumps(native, allow_nan=False))
        with patch.object(e, "clean", side_effect=AssertionError("Binding must not use clean")):
            encoded = e.metadata_json_bytes(tensors)
            self.assertEqual(encoded, e.metadata_json_bytes(native))
            self.assertEqual(encoded, e.metadata_json_bytes(dict(reversed(list(native.items())))))
        self.assertIn(b"-0.0", encoded)
        self.assertEqual(e.metadata_json_bytes(torch.tensor(-0., dtype=torch.float64)), b"-0.0")
        self.assertNotEqual(e.metadata_json_bytes(torch.tensor(-0., dtype=torch.float64)), e.metadata_json_bytes(0.))

    def test_exact_metadata_rejects_one_ULP_shape_key_and_nonfinite_changes(self):
        for dtype in (torch.float32, torch.float64):
            scalar = torch.tensor(.3, dtype=dtype)
            changed = torch.nextafter(scalar, torch.tensor(float("inf"), dtype=dtype))
            expected = e.metadata_json_bytes({"value": scalar.tolist()})
            self.assertNotEqual(e.metadata_json_bytes({"value": changed}), expected)
            self.assertNotEqual(e.metadata_json_bytes({"value": scalar[None]}), expected)
            self.assertNotEqual(e.metadata_json_bytes({"renamed": scalar}), expected)
        expected = e.metadata_json_bytes({"nested": [[1., 2.], [3., 4.]]})
        self.assertNotEqual(e.metadata_json_bytes({"nested": torch.tensor([1., 2., 3., 4.], dtype=torch.float64)}), expected)
        self.assertNotEqual(e.metadata_json_bytes({"nested": [[1., 2.], [3., 4.]], "extra": None}), expected)
        for value in (float("nan"), float("inf"), -float("inf")):
            for item in (value, torch.tensor(value, dtype=torch.float32), torch.tensor([value], dtype=torch.float64)):
                with self.subTest(value=value, metadata_type=type(item).__name__):
                    with self.assertRaises(ValueError):
                        e.metadata_json_bytes({"nested": [item]})
        with self.assertRaisesRegex(ValueError, "keys must be strings"):
            e.metadata_json_bytes({1: .3})

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

    def test_linux_RSS_stdlib_cache_descendants_and_limit_without_psutil(self):
        fake_proc = {
            "/proc/101/status": "Name: artificial\nVmRSS:\t1024 kB\n",
            "/proc/101/task/101/children": "102 103",
            "/proc/102/status": "VmRSS:\t2048 kB\n",
            "/proc/102/task/102/children": "104",
            "/proc/103/status": "VmRSS:\t4096 kB\n",
            "/proc/103/task/103/children": "105",
            "/proc/104/status": "VmRSS:\t8192 kB\n",
            "/proc/104/task/104/children": "",
        }
        def read_proc(path, *args, **kwargs):
            key = path.as_posix()
            if key not in fake_proc:
                raise FileNotFoundError(key)
            return fake_proc[key]
        original_import = __import__
        def no_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ModuleNotFoundError("Artificial remote runtime has no psutil")
            return original_import(name, *args, **kwargs)
        e.supervisor_module.cache_clear()
        with patch.object(e.sys, "platform", "linux"), patch.object(e.os, "getpid", return_value=101), \
             patch.object(Path, "read_text", new=read_proc), patch("builtins.__import__", side_effect=no_psutil):
            self.assertEqual(e.process_tree_RSS(), 15 * 1024**2)
            self.assertIs(e.supervisor_module(), e.supervisor_module())
            # A synthetic Runtime object isolates resource logic; no source,
            # checkpoints, scientific arrays, or native model execution.
            runtime = object.__new__(e.Runtime)
            runtime.cfg = dict(wall_limit_s=1800, job_tree_RSS_limit_GiB=8)
            runtime.start = e.time.monotonic()
            runtime.device = torch.device("cpu")
            runtime.maximum_RSS_bytes = runtime.maximum_CUDA_bytes = 0
            runtime.check()
            self.assertEqual(runtime.maximum_RSS_bytes, 15 * 1024**2)
            with patch.object(e, "process_tree_RSS", return_value=9 * 1024**3):
                with self.assertRaisesRegex(ValueError, "RSS bound exceeded"):
                    runtime.check()
        e.supervisor_module.cache_clear()

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

    def test_ID_dtype_preserved_in_split_torch_roundtrip_and_verifier_exact(self):
        verifier = independent_verifier()
        class FakeRuntime:
            device = torch.device("cpu")
            def check(self, out=None):
                pass
        stats = dict(loss=dict(q_reference=.5, q_floor=1e-5, charge_bin_boundaries=np.linspace(-1, .9, 7).tolist()))
        model = noise_model()
        extension = ConditionalExtension006(model, list(HYPOTHESES006))
        for kind in ("U", "S", "numeric"):
            with self.subTest(ID_kind=kind), tempfile.TemporaryDirectory(dir=TMP) as folder:
                data = artificial_dataset(32, 12, "TRAIN", id_kind=kind)
                models = dict(deterministic=FakeResponse(), full=FakeResponse(1.1))
                obs, curves, replay, samples = e.split_evaluation("TRAIN", data, configuration(), models, model,
                    extension, stats, FakeRuntime(), Path(folder), {}, {})
                expected = data["source_ids"][np.repeat(np.arange(32, dtype=np.int64), 4)].copy()
                self.assertIsInstance(replay["source_parent_ids"], np.ndarray)
                verifier.exact(verifier.cpu(replay["source_parent_ids"]), expected, "original typed SOURCE IDs")
                e.atomic_save(replay, Path(folder) / "typed_replay.pt")
                e.atomic_save(samples, Path(folder) / "typed_samples.pt")
                for path, key in (("latents/TRAIN.pt", "first128"), ("typed_replay.pt", None), ("typed_samples.pt", None)):
                    saved = torch.load(Path(folder) / path, map_location="cpu", weights_only=False)
                    if key:
                        saved = saved[key]
                    self.assertEqual(saved["source_parent_ids"].dtype, data["source_ids"].dtype)
                    verifier.exact(verifier.cpu(saved["source_parent_ids"]), expected, path + " exact IDs")
                    np.testing.assert_array_equal(verifier.cpu(saved["source_parent_row"]), np.repeat(np.arange(32), 4))
                    np.testing.assert_array_equal(verifier.cpu(saved["draw_index"]), np.tile(np.arange(4), 32))
                changed = expected.copy()
                changed[0] = "changed" if kind == "U" else b"changed" if kind == "S" else 9999
                with self.assertRaises(AssertionError):
                    verifier.exact(changed, expected, "different identity must fail")
                with self.assertRaises(AssertionError):
                    verifier.exact(expected.reshape(32, 4), expected, "different identity shape must fail")
                dtype_changed = expected.astype("U40" if kind == "U" else "S40" if kind == "S" else np.int64)
                with self.assertRaises(AssertionError):
                    verifier.exact(dtype_changed, expected, "different identity dtype must fail")

    def test_complete_artificial_driver_run_serialization_reopen_and_verifier_interop(self):
        cfg = configuration()
        cfg.update(cpu_threads=2, gpu_memory_limit_GiB=10, job_tree_RSS_limit_GiB=8, wall_limit_s=1800, output_limit_MiB=256)
        cfg["evaluation"].update(arms=list(e.ARMS), train_source_parents=40, train_real_rows=12,
                                 dev_source_parents=32, dev_real_rows=8, noise_diagnostic_rows=8192)
        cfg["seeds"].update(noise_diagnostic005=109, noise_diagnostic_extension=110)
        data = {"TRAIN": artificial_dataset(40, 12, "TRAIN"), "DEV": artificial_dataset(32, 8, "DEV")}
        stats = dict(loss=dict(q_reference=.5, q_floor=1e-5, peak_floor=1e-5,
                               charge_bin_boundaries=np.linspace(-1, .9, 7).tolist()))
        class IDGuardResponse(FakeResponse):
            def __init__(self, multiplier):
                super().__init__(multiplier)
                self.input_waveforms = []
            def __call__(self, source, noise, adc_ste=False):
                if source.dtype != torch.float32 or source.shape[1:] != (480,) or set(noise) != set(e.N01Response.noise_shapes):
                    raise AssertionError("Only SOURCE voltage and explicit model latents may enter a response")
                self.input_waveforms.append(e.cpu(source))
                return super().__call__(source, noise=noise, adc_ste=adc_ste)
        models = dict(deterministic=IDGuardResponse(1.), full=IDGuardResponse(1.1))
        noise = noise_model()
        extension = ConditionalExtension006(noise, list(HYPOTHESES006))
        snapshots = {"noise005": e.tensor_dict_sha(noise.state_dict())}
        snapshots.update({arm: e.tensor_dict_sha(model.state_dict()) for arm, model in models.items()})
        provenance = dict(kind="artificial_end_to_end_no_frozen_data_or_weights", config_sha256="artificial_config")
        verifier = independent_verifier()
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            out = Path(folder) / "complete_run"
            with patch.object(e, "bind", return_value=(cfg, provenance)) as binding, \
                 patch.object(e, "frozen_models", return_value=(models, noise, extension, stats, snapshots)), \
                 patch.object(e, "load_training", return_value=data["TRAIN"]), \
                 patch.object(e, "load_validation", return_value=data["DEV"]), \
                 patch.object(e, "process_tree_RSS", return_value=64 * 1024**2):
                result = e.run("artificial_config", "artificial_container", "artificial_identities", out,
                               source_manifest="artificial_manifest", device="cpu")
            self.assertEqual(binding.call_count, 2)
            self.assertEqual(result["status"], "zero_training_completed")
            expected_payloads = {"observables.npz", "curves.npz", "theory.npz", "bootstrap.npz", "replay128.pt", "sample_waveforms.pt",
                                 "latents/TRAIN.pt", "latents/DEV.pt", "latents/NOISE_DIAGNOSTIC.pt", "SUMMARY.json", "DECISION.json"}
            self.assertEqual(set(result["payload_sha256"]), expected_payloads)
            self.assertEqual({file.relative_to(out).as_posix() for file in out.rglob("*") if file.is_file()}, expected_payloads | {"COMPLETE.json"})
            complete = json.loads((out / "COMPLETE.json").read_text(encoding="utf8"))
            summary = json.loads((out / "SUMMARY.json").read_text(encoding="utf8"))
            decision = json.loads((out / "DECISION.json").read_text(encoding="utf8"))
            self.assertEqual(complete["actual_counts"], summary["actual_counts"])
            self.assertEqual(complete["updates"], e.UPDATES)
            self.assertEqual(decision["outcome"], complete["decision_outcome"])
            self.assertIn(decision["outcome"], ("interface_fail", "sensitivity_detected", "MC_inconclusive", "sensitivity_within_family"))
            for relative, digest in complete["payload_sha256"].items():
                self.assertEqual(e.sha256(out / relative), digest)
            obs, curves, theory, bootstrap = (verifier.read_arrays(out / name) for name in (
                "observables.npz", "curves.npz", "theory.npz", "bootstrap.npz"))
            replay = torch.load(out / "replay128.pt", map_location="cpu", weights_only=False)
            samples = torch.load(out / "sample_waveforms.pt", map_location="cpu", weights_only=False)
            for split in e.SPLITS:
                parent = np.repeat(np.arange(len(data[split]["source"]), dtype=np.int64), 4)
                expected_ids = data[split]["source_ids"][parent[:128]].copy()
                verifier.exact(obs[split + "_source_parent_ids"], data[split]["source_ids"], split + " original IDs")
                verifier.exact(obs[split + "_REAL_ids"], data[split]["real_ids"], split + " REAL IDs")
                latent = torch.load(out / "latents" / (split + ".pt"), map_location="cpu", weights_only=False)
                for record in (latent["first128"], replay["splits"][split], samples["splits"][split]):
                    verifier.exact(verifier.cpu(record["source_parent_ids"]), expected_ids, split + " original Unicode first128")
                    verifier.exact(verifier.cpu(record["source_parent_row"]), parent[:128], split + " first128 row order")
                    verifier.exact(verifier.cpu(record["draw_index"]), np.tile(np.arange(4, dtype=np.int64), 32), split + " first128 draws")
                for arm in e.ARMS:
                    p = split + "_" + arm + "_"
                    self.assertEqual(obs[p + "Q"].shape, (5, len(parent)))
                    self.assertEqual(curves[p + "fixed_bin_mean"].shape, (8, 5, 480))
                    self.assertEqual(replay["splits"][split]["arms"][arm]["expected_views"].shape, (5, 128, 480))
                    self.assertEqual(samples["splits"][split]["arms"][arm]["noise_pre"].shape, (3, 128, 480))
                    self.assertEqual(complete["actual_counts"]["model_forward"][split + "/" + arm]["events"], len(parent))
            for hypothesis in e.HYPOTHESES:
                self.assertEqual(theory[hypothesis + "_diagnostic_Q"].shape, (8192,))
            verifier.check_observables(obs, cfg, stats["loss"], data)
            verifier.check_real_curves(curves, obs, data)
            verified_bootstrap = verifier.check_bootstrap(bootstrap, obs, cfg)
            leakage = {split: {arm: dict(absmax=float(obs[split + "_" + arm + "_signal_baseline_max_per_event"].max()))
                                for arm in e.ARMS} for split in e.SPLITS}
            engineering = {key: value for key, value in summary["engineering"].items() if isinstance(value, dict) and "passed" in value}
            independently_decided = verifier.independent_decision(cfg, obs, verified_bootstrap, theory, leakage, engineering)
            self.assertEqual(decision["outcome"], independently_decided["outcome"])
            verifier.check_actual_counts(cfg, complete, summary, data)
            expected_inputs = np.concatenate([data[split]["source"][np.repeat(np.arange(len(data[split]["source"])), 4)] for split in e.SPLITS])
            for model in models.values():
                verifier.exact(e.array(torch.cat(model.input_waveforms)), expected_inputs, "No IDs/h/groups in response inputs")

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
