"""Artificial-only binding, full-coverage, update and failure checks for010."""
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
import check_mean_random009 as fixtures
from n01 import train_shape010 as driver
from n01.model008 import canonical_charge008
from n01.diagnose_mean_random009 import read_json, save_json, sha256, state_digest


class ArtificialShape010(nn.Module):
    """Small trainable contract fixture, never the scientific model class."""
    def __init__(self):
        super().__init__()
        self.baseline = fixtures.ArtificialResponse('A').eval()
        self.baseline.requires_grad_(False)
        self.before = state_digest(self.baseline.state_dict())
        self.basis_head = nn.Conv1d(1, 8, 1)
        with torch.no_grad():
            self.basis_head.weight.fill_(.02)
            self.basis_head.bias.fill_(.001)
        self.forward_calls = 0
        self.fail_call = None

    def train(self, mode=True):
        self.training = mode
        self.basis_head.train(mode)
        self.assert_frozen_unchanged()
        return self

    def trainable_parameters(self):
        return tuple(self.basis_head.parameters())

    def trainable_parameter_names(self):
        return ('basis_head.weight', 'basis_head.bias')

    def assert_optimizer_membership(self, optimizer):
        assert {id(v) for g in optimizer.param_groups for v in g['params']} == {id(v) for v in self.basis_head.parameters()}

    def assert_frozen_unchanged(self, deep=False):
        assert not self.baseline.training and all(not p.requires_grad and p.grad is None for p in self.baseline.parameters())
        assert state_digest(self.baseline.state_dict()) == self.before

    def random_forward(self, source, z, enabled=True):
        self.forward_calls += 1
        if self.fail_call == self.forward_calls:
            raise RuntimeError('artificial010 injected forward failure')
        with torch.no_grad():
            original = self.baseline.signal_forward(source, dict(phys=torch.zeros(len(source), 10), local_white=torch.zeros(len(source), 1, 400)))
        base = original['signal_pre']
        raw = self.basis_head(source[:, None].float()).double()*.05
        gate = torch.ones(384, dtype=torch.float64)
        gate[:100] = 0.
        raw = raw*gate
        basis = raw-canonical_charge008(raw)[..., None]*self.baseline.projection[None, None]
        random_shape = torch.einsum('bkt,bk->bt', basis, z.double()) if enabled else torch.zeros_like(base)
        signal = base+random_shape
        regular = random_shape.square().mean()/.02**2+torch.diff(random_shape, n=2).square().mean()/.02**2
        return dict(original, signal_pre=signal, base_signal_pre_V=base, random_shape_pre_V=random_shape,
            random_RMS_V=random_shape.square().mean(1).sqrt(), regular=regular,
            basis_canonical_charge=canonical_charge008(basis), canonical_random_shape_charge=canonical_charge008(random_shape),
            canonical_charge_identity_residual=canonical_charge008(signal)-original['qsig'])


def expect_failure(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError('Expected010 binding rejection missing')


def check_bindings(out, official):
    root, old = out/'artificial_root', out/'artificial_immutable_root'
    (root/'src/n01').mkdir(parents=True)
    old.mkdir()
    dependencies = set(driver.SOURCE_DEPENDENCIES) | {'train_shape010', 'model010'}
    for name in dependencies:
        (root/'src/n01'/(name+'.py')).write_text('explicit artificial010 source binding\n')
    policy = copy.deepcopy(official)
    policy['remote_root'], policy['immutable_endpoint_root'] = str(root.resolve()), str(old.resolve())
    pp, bp, mp = root/'policy.json', root/'bindings.json', root/'manifest.json'
    save_json(pp, policy)
    files = {}
    for name in ('dataset', 'statistics', 'data_statistics', 'noise005', 'deterministic', 'base_config', 'training_complete'):
        path = (old if name in ('noise005', 'deterministic', 'training_complete') else root)/(name+'.artificial')
        path.write_bytes(('artificial010 bound bytes '+name).encode())
        files[name] = dict(path=str(path.resolve()), sha256=sha256(path))
    save_json(mp, dict(schema='n01/shape010-source/v1', run_id=driver.RUN_ID, policy_sha256=sha256(pp),
        files={str(v.relative_to(root)).replace('\\', '/'): sha256(v) for v in (root/'src/n01').glob('*.py')},
        bound_inputs={name: row['sha256'] for name, row in files.items()}))
    save_json(bp, dict(schema='n01/shape010-bindings/v1', run_id=driver.RUN_ID, policy_sha256=sha256(pp),
        roots=[str(root.resolve()), str(old.resolve())], source_manifest=dict(path=str(mp.resolve()), sha256=sha256(mp)), files=files))
    driver.bind_inputs(pp, bp, root/'unused_output', root=root)
    assert not (root/'unused_output').exists()
    original_np, original_torch = driver.np.load, driver.torch.load
    decode = dict(np_load=0, torch_load=0)
    def forbidden_np(*args, **kwargs):
        decode['np_load'] += 1
        raise AssertionError('Numeric decode before rejected binding')
    def forbidden_torch(*args, **kwargs):
        decode['torch_load'] += 1
        raise AssertionError('Checkpoint load before rejected binding')
    driver.np.load, driver.torch.load = forbidden_np, forbidden_torch
    try:
        Path(files['deterministic']['path']).write_bytes(b'artificial mutation')
        expect_failure(lambda: driver.bind_inputs(pp, bp, root/'unused_output', root=root))
    finally:
        driver.np.load, driver.torch.load = original_np, original_torch
    assert decode == dict(np_load=0, torch_load=0)
    return decode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    out = Path(args.output).resolve()
    assert not out.exists() and (not out.drive or out.drive.upper() in ('E:', 'D:'))
    out.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    official = driver.fixed_policy(ROOT/'shape010_policy.json')
    checks = []
    wrong = copy.deepcopy(official)
    wrong['training']['adversarial_weight'] = .1
    expect_failure(lambda: driver.fixed_policy(wrong))
    checks.append('noGAN_recipe_changes_rejected')
    rejected_decode = check_bindings(out, official)
    checks.append('manifest_input_mutation_rejected_before_np_or_checkpoint_decode')
    data, stats, domain = fixtures.artificial_fixture()
    archive = fixtures.GuardArchive(data)
    data = driver.read_train_members(archive)
    assert not any('val' in v or 'DEV' in v for v in archive.read)
    checks.append('TRAIN_whitelist_reads_no_DEV_members')
    p = copy.deepcopy(official)
    p.update(G_updates=9, TRAIN_counts=dict(source=128, real=512))
    p['training'].update(source_parents_per_update=32, draws_per_parent=2, real_rows_per_update=64,
        forward_chunk_rows=32, checkpoint_interval=5, status_interval=1)
    p['evaluation'].update(source_parents=16, draws_per_parent=4, matched_rows=64, chunk_rows=32,
        real_reference_rows=64, real_control_rows=64, final_evaluation_after_update=9)
    model = ArtificialShape010()
    extension = fixtures.ArtificialExtension().eval()
    result = driver.run_arrays(data, model, extension, stats, p, out/'artificial_training', artificial=True)
    assert result['G_updates'] == result['counters']['attempted_G_updates'] == 9
    assert result['counters']['train_response_rows'] == 9*64
    assert result['counters']['train_hard_observation_rows'] == 9*64*3
    assert result['coverage']['source_unique'] == 128 and result['coverage']['real_unique'] == 512
    assert result['coverage']['source_parent_visits'] == 9*32 and result['coverage']['real_row_visits'] == 9*64
    assert all(result['counters'][k] == 0 for k in ('D_updates', 'noise_updates', 'population_updates', 'C2ST_updates'))
    assert result['frozen_A_005_unchanged'] and result['DEV_numeric_read'] is False
    checks.append('full_legal_TRAIN_sampler_coverage_with_exact_successful_G_count_and_no_D')
    proof = read_json(out/'artificial_training/FIRST_UPDATE_PROOF.json')
    assert proof['actual_G_updates'] == 1 and proof['new_head_gradient_nonzero'] and proof['new_head_parameters_changed']
    assert proof['frozen_A_005_unchanged'] and proof['metrics']['signal_prefix_absmax_V'] == 0
    checks.append('real_first_step_semantics_on_artificial_new_head_only')
    initial = read_json(out/'artificial_training/INITIAL_EVALUATION.json')
    final = read_json(out/'artificial_training/FINAL_EVALUATION.json')
    assert initial['projection_context'] == final['projection_context']
    with np.load(out/'artificial_training/EVALUATION_BANK.npz', allow_pickle=False) as values:
        assert not np.intersect1d(values['real_reference_rows'], values['real_control_rows']).size
        assert len(np.unique(values['selected_source_ids'])) == 16
    for h in driver.HYPOTHESES:
        with np.load(out/('artificial_training/frozen_A_'+h+'_TRAIN.npz'), allow_pickle=False) as a:
            a_signal = a['pure_signal_V'].reshape(16, 4, 384)
            assert np.array_equal(a_signal, np.repeat(a_signal[:, :1], 4, axis=1))
        with np.load(out/('artificial_training/shape010_final_'+h+'_TRAIN.npz'), allow_pickle=False) as v:
            assert v['waveform_V'].shape == (64, 256)
            assert np.max(np.abs(v['Q']-v['Q_signal']-v['Q_noise']-v['Q_ADC_delta'])) < 1e-10
            assert np.array_equal(v['crop_start'], v['peak_index']-16)
    checks.append('same_frozen_eval_projection_parent_bank_disjoint_REAL_and_observed_crop_Q_components')
    # Initial evaluation invokes two chunks x two views=4 forwards. Train G1
    # invokes two chunks; fail the first forward of attempted G2 after G1 returns.
    failure_model = ArtificialShape010()
    failure_model.fail_call = 7
    try:
        driver.run_arrays(data, failure_model, fixtures.ArtificialExtension().eval(), stats, p,
            out/'artificial_failure', artificial=True)
    except RuntimeError as error:
        assert 'injected forward failure' in str(error)
    else:
        raise AssertionError('010 artificial failure did not trigger')
    failed = read_json(out/'artificial_failure/FAILED.json')
    assert failed['completed_G_updates'] == 1
    assert failed['progress']['counters']['attempted_G_updates'] == 2
    assert failed['progress']['counters']['G_updates'] == 1
    assert failed['progress']['in_flight']['operation'] == 'training_new_head_forward'
    checkpoint = torch.load(out/'artificial_failure/failure.pt', weights_only=False)
    assert checkpoint['G_updates'] == 1 and checkpoint['complete_step_boundary'] is False
    assert checkpoint['automatic_restart'] is False and not (out/'artificial_failure/COMPLETE.json').exists()
    checks.append('failure_keeps_completed_G_and_attempted_partial_state_without_restart')
    checkpoint_failure_model = ArtificialShape010()
    checkpoint_failure_model.fail_call = 7
    original_save = driver.torch.save
    def fail_failure_checkpoint(value, stream, *args, **kwargs):
        if isinstance(value, dict) and value.get('purpose') == 'failure':
            raise OSError('artificial failure checkpoint write rejected')
        return original_save(value, stream, *args, **kwargs)
    driver.torch.save = fail_failure_checkpoint
    try:
        try:
            driver.run_arrays(data, checkpoint_failure_model, fixtures.ArtificialExtension().eval(), stats, p,
                out/'artificial_checkpoint_failure', artificial=True)
        except RuntimeError as error:
            assert 'injected forward failure' in str(error)
        else:
            raise AssertionError('Combined forward/checkpoint failure not triggered')
    finally:
        driver.torch.save = original_save
    failed_cp = read_json(out/'artificial_checkpoint_failure/FAILED.json')
    error_cp = read_json(out/'artificial_checkpoint_failure/FAILURE_CHECKPOINT_ERROR.json')
    assert failed_cp['completed_G_updates'] == error_cp['completed_G_updates'] == 1
    assert failed_cp['error_type'] == 'RuntimeError' and error_cp['error_type'] == 'OSError'
    assert (out/'artificial_checkpoint_failure/initial.pt').exists()
    assert not (out/'artificial_checkpoint_failure/failure.pt').exists()
    checks.append('failure_checkpoint_write_error_preserves_original_error_and_actual_completed_G')
    receipt = dict(status='artificial_shape010_driver_checks_passed', passed=len(checks), checks=checks,
        scientific_inputs_read=0, scientific_checkpoints_loaded=0, scientific_optimizer_updates=0,
        artificial_completed_G_updates=11, failed_binding_decode_counters=rejected_decode,
        driver_sha256=sha256(ROOT/'src/n01/train_shape010.py'), checker_sha256=sha256(Path(__file__)),
        elapsed_seconds=time.monotonic()-started,
        artifacts={str(v.relative_to(out)): dict(sha256=sha256(v), bytes=v.stat().st_size) for v in out.rglob('*') if v.is_file()})
    save_json(out/'ENGINEERING_RECEIPT.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'passed', 'elapsed_seconds', 'scientific_optimizer_updates')}))


if __name__ == '__main__':
    main()
