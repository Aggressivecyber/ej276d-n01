"""Artificial-only semantic checks for the independent TRAIN diagnosis driver.

No trained checkpoint or actual SOURCE/REAL container is opened.  Default mode
executes one complete128-row A/B/C family across all three hypotheses, including
raw storage, covariance factors, paired means, and fixed empirical controls.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import torch
from torch import nn

from n01 import diagnose_mean_random009 as diag
from n01.data import sha256
from n01.losses008 import fit_loss_statistics
from n01.observation008 import observe_generated_carrier


def analytic_source(n):
    t = np.arange(384, dtype=np.float64)[None]
    k = np.arange(n, dtype=np.float64)[:, None]
    width = 3.+k % 3
    decay = 20.+k % 11
    pulse = np.where(t <= 126, np.exp(-.5*np.square((t-126)/width)), np.exp(-(t-126)/decay))
    pulse[:, :100] = 0.
    return (.008+.025*(k+1)/(n+1))*pulse


class ArtificialResponse(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.register_buffer('gain', torch.tensor(1. if variant == 'A' else 1.02, dtype=torch.float64))
        t = torch.arange(384, dtype=torch.float64)
        projection = torch.zeros(384, dtype=torch.float64)
        projection[110:366] = 1./256.
        basis = torch.sin((t-126)/12.)*torch.exp(-((t-155)/45.).square())
        basis[:100] = 0.
        basis = basis-basis[110:366].sum()*projection
        self.register_buffer('basis', basis)
        self.register_buffer('projection', projection)

    def calculate(self, source, response, random=True):
        centered = source-source[:, :100].mean(1, keepdim=True)
        q = centered[:, 110:366].sum(1)
        qbar = q*self.gain
        stochastic = random and self.variant == 'B'
        qsig = qbar*torch.exp(.025*response['phys'][:, 0].double()-.5*.025**2) if stochastic else qbar
        shape = .0002*response['phys'][:, 2, None].double()*self.basis[None] if stochastic else torch.zeros_like(source)
        local = .000001*response['local_white'][:, 0, :384].double() if stochastic else torch.zeros_like(source)
        local[:, :100] = 0.
        local = local-local[:, 110:366].sum(1, keepdim=True)*self.projection[None]
        mean_component = centered*(qsig/q)[:, None]
        signal = mean_component+shape+local
        return dict(signal_pre=signal, signal_noise=local, qbar=qbar, qsig=qsig,
            time_scale=torch.ones_like(q), q_source=q,
            canonical_charge_identity_residual=signal[:, 110:366].sum(1)-qsig,
            canonical_local_charge=local[:, 110:366].sum(1),
            canonical_charge=signal[:, 110:366].sum(1),
            base_shape_Q=q, qsig_over_base_shape_Q=qsig/q,
            mean_component_pre_V=mean_component, stochastic_shape_pre_V=shape,
            stochastic_shape_voltage_RMS_V=shape.square().mean(1).sqrt(),
            local_RMS_V=local.square().mean(1).sqrt())

    def signal_forward(self, source, response):
        return self.calculate(source, response)


class ArtificialAblation:
    def __init__(self, model):
        self.model = model
        self.before = diag.state_digest(model.state_dict())

    def signal_forward(self, source, response, *, disable=()):
        assert disable in ((), diag.ALL_OFF)
        return self.model.calculate(source, response, random=not disable)

    def assert_unchanged(self, deep=False):
        assert diag.state_digest(self.model.state_dict()) == self.before


class ArtificialNoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale_std', torch.tensor(.1, dtype=torch.float64))


class ArtificialExtension(nn.Module):
    def __init__(self):
        super().__init__()
        self.model005 = ArtificialNoise()
        self.before = diag.state_digest(self.state_dict())

    def assert_frozen(self):
        assert diag.state_digest(self.state_dict()) == self.before

    def sample_prefix(self, values):
        std = self.model005.scale_std
        scale = torch.exp(std*values['baseline_scale'].double()-std.square()).reshape(-1)
        return dict(prefix_pre=.0002*scale[:, None]*values['baseline_gaussian'][:, 0].double(), scale=scale)

    def extend_prefix(self, prefix, scale, epsilon, hypothesis):
        tail = dict(fast=1., slow=1.2, common_level=.8)[hypothesis]*.0002*scale[:, None]*epsilon[:, 0].double()
        if hypothesis == 'common_level':
            tail = tail+prefix[:, -1, None]
        return dict(noise_pre=torch.cat((prefix, tail), 1))


def artificial_fixture():
    ns, nr = 128, 512
    source = analytic_source(ns)
    source_window = source[:, 110:366].copy()
    real_source = analytic_source(nr)
    rng = np.random.default_rng(44109)
    noise = rng.normal(0, .0002, (nr, 384))
    real_carrier = 1.01*real_source+noise
    obs = observe_generated_carrier(torch.from_numpy(real_carrier), ste=False, return_carrier=True)
    real = obs['window_V'].numpy()
    baseline = obs['baseline100'].numpy()
    data = dict(source=source_window, source_carrier384=source, real=real,
        real_baseline100=baseline, source_ids=np.arange(ns, dtype=np.int64)+100000,
        real_ids=np.arange(nr, dtype=np.int64)+200000, real_groups=np.arange(nr, dtype=np.int64) % 2,
        source_role=np.full(ns, 'TRAIN'), real_role=np.full(nr, 'TRAIN'),
        original_source_weights=np.ones(ns), h=np.linspace(0, 1, ns))
    loss = fit_loss_statistics(torch.from_numpy(real), torch.from_numpy(source_window), torch.from_numpy(baseline))
    statistics = dict(loss=loss)
    data_statistics = dict(domains=dict(source=dict(Q256_eight_quantile_bin_edges=
        np.quantile(source_window.sum(1), np.linspace(0, 1, 9)))))
    return data, statistics, data_statistics


class GuardArchive:
    def __init__(self, data):
        self.data = data
        self.files = list(data)+['source_val', 'target_val', 'hval', 'source_val_ids', 'real_val_original_index']
        self.read = []

    def __getitem__(self, key):
        assert key in diag.TRAIN_KEYS+diag.TRAIN_OPTIONAL_KEYS, 'Forbidden DEV key decoded: '+key
        self.read.append(key)
        return self.data[key]


def expect_failure(action):
    try:
        action()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('Expected binding/semantic rejection did not occur')


def check_binding_order(out, official_policy):
    root, old = out/'artificial_root', out/'artificial_immutable_root'
    (root/'src/n01').mkdir(parents=True)
    old.mkdir()
    for name in diag.SOURCE_DEPENDENCIES:
        (root/'src/n01'/(name+'.py')).write_text('explicit artificial binding payload\n', encoding='utf8')
    policy = copy.deepcopy(official_policy)
    policy['remote_root'], policy['immutable_endpoint_root'] = str(root.resolve()), str(old.resolve())
    policy_path = root/'policy.json'
    diag.save_json(policy_path, policy)
    manifest_path = root/'manifest.json'
    diag.save_json(manifest_path, dict(files={str(v.relative_to(root)).replace('\\', '/'): sha256(v)
        for v in sorted((root/'src/n01').glob('*.py'))}))
    files = {}
    for name in ('dataset', 'statistics', 'data_statistics', 'noise005', 'deterministic', 'full', 'base_config'):
        path = (old if name in ('deterministic', 'full', 'noise005') else root)/(name+'.artificial')
        path.write_bytes(('not a science file: '+name).encode())
        files[name] = dict(path=str(path.resolve()), sha256=sha256(path))
    binding_path = root/'bindings.json'
    diag.save_json(binding_path, dict(schema=diag.BINDINGS_SCHEMA, policy_sha256=sha256(policy_path),
        roots=[str(root.resolve()), str(old.resolve())],
        source_manifest=dict(path=str(manifest_path.resolve()), sha256=sha256(manifest_path)), files=files))
    diag.bind_inputs(policy_path, binding_path, root/'output', root=root)
    assert not (root/'output').exists()
    counts = dict(np_load=0, torch_load=0)
    np_load, torch_load = diag.np.load, diag.torch.load
    def forbidden_np(*args, **kwargs):
        counts['np_load'] += 1
        raise AssertionError('TRAIN decode occurred before failed binding')
    def forbidden_torch(*args, **kwargs):
        counts['torch_load'] += 1
        raise AssertionError('Checkpoint load occurred before failed binding')
    diag.np.load, diag.torch.load = forbidden_np, forbidden_torch
    try:
        Path(files['full']['path']).write_bytes(b'artificial mutation')
        expect_failure(lambda: diag.bind_inputs(policy_path, binding_path, root/'output', root=root))
    finally:
        diag.np.load, diag.torch.load = np_load, torch_load
    assert counts == dict(np_load=0, torch_load=0)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    out = Path(args.output).resolve()
    assert not out.exists()
    if out.drive:
        assert out.drive.upper() in ('E:', 'D:')
    out.mkdir(parents=True)
    began = time.monotonic()
    torch.set_num_threads(4)
    official_policy = diag.fixed_policy(ROOT/'mean_random009_policy.json')
    checks = []
    changed = copy.deepcopy(official_policy)
    changed['seeds']['response'] += 1
    expect_failure(lambda: diag.fixed_policy(changed))
    checks.append('changed_policy_stream_rejected')
    binding_counts = check_binding_order(out, official_policy)
    checks.append('input_mutation_rejected_before_numpy_or_checkpoint_decode')
    data, statistics, data_statistics = artificial_fixture()
    archive = GuardArchive(data)
    loaded = diag.read_train_members(archive)
    assert set(archive.read) == set(data) and len(archive.read) == len(data)
    assert not any('val' in key or 'DEV' in key for key in archive.read)
    checks.append('TRAIN_whitelist_never_reads_forbidden_DEV_arrays')
    q = diag.validate_train(loaded, source_rows=128, real_rows=512)
    selected, bins, counts, post = diag.select_parents(q,
        data_statistics['domains']['source']['Q256_eight_quantile_bin_edges'], parents_per_bin=4)
    assert len(selected) == 32 and len(np.unique(selected)) == 32
    assert np.array_equal(np.bincount(bins, minlength=8), np.full(8, 4)) and np.isclose(post.sum(), 32)
    checks.append('SOURCE_TRAIN_Q8_equal_quota_unique_parents')
    a, b = diag.select_real(512, 128, 2026101414)
    assert not np.intersect1d(a, b).size and len(a) == len(b) == 128
    checks.append('matched_REAL_REAL_control_disjoint')
    bank1, values1, binding1 = diag.explicit_bank(128, 128, diag.SEEDS)
    bank2, values2, binding2 = diag.explicit_bank(128, 128, diag.SEEDS)
    assert binding1 == binding2 and all(np.array_equal(values1[k], values2[k]) for k in values1)
    checks.append('explicit_fixed_chunk_parent_major_bank_replays')
    rng = np.random.default_rng(4)
    z = rng.normal(size=(8, 4, 7))
    factor = diag.covariance_factor(z)
    for parent in range(8):
        f = factor['centered_draw_covariance_factor'][parent]
        assert np.allclose(f.T@f/3, np.cov(z[parent], rowvar=False, ddof=1), atol=1e-14)
        assert np.allclose(np.diag(f.T@f/3), factor['sample_variance'][parent], atol=1e-14)
    checks.append('complete_wave_covariance_factor_reconstructs_full_sample_covariance')
    shifted = torch.from_numpy(np.asarray([.03*np.exp(-.5*((np.arange(384)-peak)/3.)**2) for peak in (122, 130)]))
    mean_individual = observe_generated_carrier(shifted)['window_V'].mean(0)
    observation_mean = observe_generated_carrier(shifted.mean(0, keepdim=True))['window_V'][0]
    assert float((mean_individual-observation_mean).abs().max()) > .001
    checks.append('mean_hard_observation_distinct_from_observation_of_mean')
    t = np.arange(256)
    wide = .03*np.exp(-.5*((t-16)/10.)**2)
    morph = diag.morphology(np.stack((wide, -np.ones(256)*.01)))
    assert morph['rise_prepeak_censored'][0] and not morph['temporal_nonpositive_peak'][0]
    assert morph['temporal_nonpositive_peak'][1] and morph['rise_undefined'].sum() == 2
    checks.append('positive_prepeak_censoring_separate_from_nonpositive_peak')
    models = dict(A=ArtificialResponse('A').eval(), B=ArtificialResponse('B').eval())
    for model in models.values():
        model.requires_grad_(False)
    extension = ArtificialExtension().eval()
    helper = ArtificialAblation(models['B'])
    policy = copy.deepcopy(official_policy)
    policy.update(parents=32, repeats=4, chunk_rows=128, TRAIN_counts=dict(source=128, real=512),
                  matched_real_rows=128, REAL_REAL_rows_each=128, bootstrap_replicates=20)
    policy['selection']['parents_per_bin'] = 4
    summary = diag.run_arrays(data, models, helper, extension, statistics, data_statistics,
        policy, out/'artificial_driver', device='cpu', artificial=True)
    run = out/'artificial_driver'
    assert not (run/'START_PROOF.json').exists() and (run/'ARTIFICIAL_START_PROOF.json').exists()
    proof = diag.read_json(run/'ARTIFICIAL_START_PROOF.json')
    assert proof['actual_family_rows'] == 128 and proof['counters']['hard_observation_rows'] == 1152
    assert summary['counters']['signal_forward_rows'] == 384
    assert summary['counters']['hard_observation_rows'] == 1152
    assert summary['counters']['noise005_rows'] == 128 and summary['counters']['D_model_forward_rows'] == 0
    assert summary['counters']['optimizer_updates'] == 0 and summary['frozen_tensors_unchanged']
    assert summary['DEV_numeric_read'] is False and summary['artificial_only'] is True
    checks.append('first_complete128_row_A_B_C_three_hypothesis_receipt_and_zero_update_counters')
    with np.load(run/'B_SIGNAL_DRAWS.npz', allow_pickle=False) as values:
        signal = values['signal_pre_V'].reshape(32, 4, 384)
    with np.load(run/'D_SIGNAL_MC_MEAN.npz', allow_pickle=False) as values:
        assert np.array_equal(values['pure_signal_pre_V'], signal.mean(1))
    for h in diag.HYPOTHESES:
        with np.load(run/('B_'+h+'_OBSERVED_DRAWS.npz'), allow_pickle=False) as values:
            hard = values['hard_observed_waveform_V'].reshape(32, 4, 256)
        with np.load(run/('D_'+h+'_MC_MEANS.npz'), allow_pickle=False) as values:
            assert np.array_equal(values['mean_of_individual_hard_observations_V'], hard.mean(1))
    assert 'D' not in summary['distances']
    checks.append('D_derived_exactly_from_saved_B_rows_without_distribution_scoring')
    with np.load(run/'SHARED_BASELINE100_ADC.npz', allow_pickle=False) as values:
        shared = values['baseline100_ADC']
    for v in diag.VIEWS:
        for h in diag.HYPOTHESES:
            with np.load(run/(v+'_'+h+'_OBSERVED_DRAWS.npz'), allow_pickle=False) as values:
                assert np.array_equal(values['baseline_mean_V'], shared.mean(1))
                assert np.max(np.abs(values['Q']-values['Q_signal']-values['Q_noise']-values['Q_ADC_delta'])) < 1e-10
                if v == 'A':
                    assert np.isnan(values['base_shape_Q']).all() and np.isnan(values['qsig_over_base_shape_Q']).all()
    checks.append('common_electronic_noise_same_crop_Q_identity_and_A_unknown_availability')
    class SecondFamilyFailure(ArtificialAblation):
        def __init__(self, model):
            super().__init__(model)
            self.normal_calls = 0

        def signal_forward(self, source, response, *, disable=()):
            if not disable:
                self.normal_calls += 1
                if self.normal_calls == 2:
                    raise RuntimeError('artificial B failure in second family')
            return super().signal_forward(source, response, disable=disable)
    failure_policy = copy.deepcopy(policy)
    failure_policy.update(parents=64, matched_real_rows=256, REAL_REAL_rows_each=256)
    failure_policy['selection']['parents_per_bin'] = 8
    failure_models = dict(A=ArtificialResponse('A').eval(), B=ArtificialResponse('B').eval())
    for model in failure_models.values():
        model.requires_grad_(False)
    failure_root = out/'artificial_second_family_failure'
    try:
        diag.run_arrays(data, failure_models, SecondFamilyFailure(failure_models['B']),
            ArtificialExtension().eval(), statistics, data_statistics, failure_policy,
            failure_root, device='cpu', artificial=True)
    except RuntimeError as error:
        assert 'artificial B failure' in str(error)
    else:
        raise AssertionError('Artificial B failure was not triggered')
    failed = diag.read_json(failure_root/'FAILED.json')
    status = diag.read_json(failure_root/'status.json')
    assert failed == status and failed['status'] == 'failed_frozen_diagnosis'
    assert failed['counters']['A_signal_forward_rows'] == 256
    assert failed['counters']['B_signal_forward_rows'] == failed['counters']['C_ablation_forward_rows'] == 128
    assert failed['counters']['signal_forward_rows'] == 512
    assert failed['attempted_operation_counters']['B_signal_forward_rows'] == 256
    assert failed['completed_family_rows'] == 128 and failed['completed_families'] == 1
    assert failed['in_flight']['view'] == 'B' and failed['in_flight']['start'] == 128
    assert failed['attempted_in_flight_rows'] == 128 and not (failure_root/'SUMMARY.json').exists()
    assert len((failure_root/'CHUNK_RECEIPTS.jsonl').read_text().splitlines()) == 1
    checks.append('second_family_failure_preserves_completed_A_and_unknown_attempted_B_counts')
    receipt = dict(status='artificial_mean_random009_driver_checks_passed', checks=checks, passed=len(checks),
        scientific_inputs_read=0, trained_checkpoints_loaded=0, scientific_model_forwards=0,
        artificial_signal_forward_rows=896, artificial_hard_observation_rows=2304, optimizer_updates=0,
        failed_binding_decode_counters=binding_counts, forbidden_archive_fields_read=[],
        driver_sha256=sha256(ROOT/'src/n01/diagnose_mean_random009.py'),
        script_sha256=sha256(Path(__file__)), elapsed_seconds=time.monotonic()-began,
        summary=str(run/'SUMMARY.json'), artifacts={str(v.relative_to(out)): dict(sha256=sha256(v), bytes=v.stat().st_size)
            for v in sorted(out.rglob('*')) if v.is_file()})
    diag.save_json(out/'ENGINEERING_RECEIPT.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'passed', 'elapsed_seconds', 'scientific_inputs_read', 'optimizer_updates')}, allow_nan=False))


if __name__ == '__main__':
    main()
