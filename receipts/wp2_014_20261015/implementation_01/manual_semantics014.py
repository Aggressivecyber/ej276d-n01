"""Artificial-only semantic checks; never private arrays or trained weights."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

import numpy as np
import torch
import check_model010 as mf
import check_mean_random009 as fixture
import run_wp2_014 as driver
from n01.model010 import FrozenDeterministicVoltageShape010, tensor_state_digest010, INITIALIZATION_SEED010
from n01.model014 import FrozenDeterministicVoltageShape014, calibrate_initial_head014
from n01.losses008 import ADC_STEP
from n01.losses008_v2 import N01Loss008
from n01.losses014 import N01Loss014, fit_integral_statistics014, integral_features014, integral_directions014
from n01.observation008 import observe_generated_carrier
from n01.train_shape010 import Streams010, cpu_tree
from n01.response008 import loss_tensor_statistics
from n01.data import FullCoverageSampler
from wp1_ops import write_new, clean, sha


def assert_raises(action, word):
    try:
        action()
    except (ValueError, RuntimeError) as e:
        assert word in str(e), str(e)
        return str(e)
    raise AssertionError('Expected rejection: '+word)


def main():
    out = Path(sys.argv[1]).resolve()
    assert out.drive.upper() in ('E:', 'D:') and not out.exists()
    out.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    checks = []
    protected = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
        (ROOT/'src/n01').glob('*.py') if p.name not in ('model014.py', 'losses014.py')}
    source, z = mf.fixture()
    baseline = mf.baseline_fixture()
    frozen = tensor_state_digest010(baseline)
    cpu_rng = torch.get_rng_state().clone()
    old = FrozenDeterministicVoltageShape010(baseline)
    new = FrozenDeterministicVoltageShape014(baseline, seed=INITIALIZATION_SEED010)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert tensor_state_digest010(old.basis_head) == tensor_state_digest010(new.basis_head)
    with torch.no_grad():
        old_out, new_out = old(source, z), new(source, z)
        for name, value in old_out.items():
            if isinstance(value, torch.Tensor):
                assert mf.bytes_equal(value, new_out[name]), name
        zero = new(source, torch.zeros_like(z))
        assert mf.bytes_equal(zero['signal_pre'], zero['base_signal_pre_V'])
        off = new(source, z, enabled=False)
        assert mf.bytes_equal(off['signal_pre'], off['base_signal_pre_V'])
        assert (new_out['signal_pre'][:, :100] == 0).all()
        assert new_out['canonical_charge_identity_residual'].abs().max() < 1e-10
    assert sum(p.numel() for p in new.trainable_parameters()) == 1288
    assert new.trainable_parameter_names() == ('basis_head.weight', 'basis_head.bias')
    checks.append(dict(name='exact_010_layout_and_arithmetic_same_seed', tensors_bitwise_equal=True,
        zero_latent_equals_A_bytes=True, global_CPU_RNG_preserved=True, new_head_parameters=1288))

    data, stats, _ = fixture.artificial_fixture()
    integral_stats = fit_integral_statistics014(torch.as_tensor(data['real']))
    native = N01Loss008(loss_tensor_statistics(stats['loss']), seed=1231, evaluation_seed=1232)
    loss = N01Loss014(loss_tensor_statistics(stats['loss']), integral_stats,
        seed=1231, evaluation_seed=1232, integral_seed=1233)
    fake = torch.as_tensor(data['real'][:32]).clone().requires_grad_(True)
    real = torch.as_tensor(data['real'][32:64])
    fb, rb = map(torch.as_tensor, (data['real_baseline100'][:32], data['real_baseline100'][32:64]))
    context = native.create_projection_context(1)
    expected = native.components(dict(waveform=fake, baseline100=fb), real,
        projection_context=context, real_baseline100=rb)
    actual = loss.training_components(dict(waveform=fake, baseline100=fb), real,
        candidate='C0', projection_context=context, real_baseline100=rb)
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
    go = torch.autograd.grad(expected['wave']+2*expected['QPSD'], fake, retain_graph=True)[0]
    gn = torch.autograd.grad(actual['wave']+2*actual['distribution'], fake)[0]
    assert torch.equal(go, gn)
    checks.append(dict(name='C0_native_010_loss_and_input_gradient_exact_equality', all_native_components_equal=True,
        max_gradient_difference=float((go-gn).abs().max())))

    y = torch.zeros(8, 256, dtype=torch.float64)
    y[1] = -.002
    y[2, :46], y[2, 46:] = .003, .001
    y[3, :46], y[3, 46:] = -.001, .002
    y[4, :46], y[4, 46:] = .002, -.001
    y[5, :46], y[5, 46:] = -.004, .0001
    y[6] = .0005
    y[7, 0], y[7, 46] = -.0625, .0625
    y.requires_grad_(True)
    features = integral_features014(y)
    q, tail = y.sum(1), y[:, 46:].sum(1)
    nz = q != 0
    assert torch.allclose(features[nz, 1]/features[nz, 0], tail[nz]/q[nz], rtol=0, atol=1e-14)
    transformed = torch.asinh(features/loss.integral_scale)
    recovered = torch.sinh(transformed)*loss.integral_scale
    assert torch.allclose(recovered, features, rtol=1e-13, atol=1e-14)
    dist = loss.integral_loss(y, torch.as_tensor(data['real'][:8]), None, context)
    gy = torch.autograd.grad(dist, y)[0]
    assert len(features) == 8 and torch.isfinite(gy).all()
    assert loss.last_integral_diagnostics['fake_nonpositive_Q_rows'] == int((q <= 0).sum())
    checks.append(dict(name='C1_signed_integral_inversion_and_zero_negative_Q_retention', rows=8,
        nonpositive_Q_rows=int((q <= 0).sum()), exactly_zero_Q_rows=int((q == 0).sum()),
        denominator_free_gradient_finite=True, max_inverse_error=float((features-recovered).detach().abs().max())))

    before = torch.get_rng_state().clone()
    fingerprint = context.fingerprint
    d1 = integral_directions014(context, 1233)
    d2 = integral_directions014(context, 1233)
    assert torch.equal(d1, d2) and torch.equal(before, torch.get_rng_state())
    assert context.validate().fingerprint == fingerprint
    left, right = Streams010(50, 51, 52), Streams010(50, 51, 52)
    ls, rs = FullCoverageSampler(128, 53), FullCoverageSampler(128, 53)
    for count in (32, 32, 16, 64):
        a, an, ae = left.draw(count, 'cpu')
        loss.integral_loss(y.detach(), torch.as_tensor(data['real'][:8]), None, context)
        b, bn, be = right.draw(count, 'cpu')
        assert torch.equal(a, b) and torch.equal(ae, be) and all(torch.equal(an[k], bn[k]) for k in an)
        assert np.array_equal(ls.draw(count), rs.draw(count))
    checks.append(dict(name='paired_source_latent_noise_extension_projection_streams', same_streams=True,
        extra_C1_directions_advance_no_shared_RNG=True))

    calibration_model = FrozenDeterministicVoltageShape014(baseline, seed=2026101540)
    with torch.no_grad():
        # Deliberately large artificial weights ensure the lower-bracket branch.
        calibration_model.basis_head.weight.mul_(100)
        original, decoded = calibration_model._baseline_once(source[::2])
        zbanks = torch.stack([torch.randn(4, 4, 8, generator=torch.Generator().manual_seed(s)) for s in (101, 102)])
        cal = calibrate_initial_head014(calibration_model, decoded, original['signal_pre'], zbanks,
            target_RMS_V=ADC_STEP, bisections=16)
        logits = calibration_model.cached_initial_logits(decoded).repeat_interleave(4, 0)
        actual_random = torch.stack([calibration_model.random_from_logits(logits,
            original['signal_pre'].repeat_interleave(4, 0), bank.reshape(-1, 8)) for bank in zbanks])
        actual_rms = float(actual_random.square().mean().sqrt())
        assert actual_rms == cal['actual_RMS_V'] and actual_rms <= ADC_STEP
        assert cal['calibration_status'] == 'lower_bracket_scaled' and cal['actual_bisections'] == 16
        assert cal['initial_weight_multiplier'] == cal['bisection_lower']
        assert calibration_model.amplitude_fraction == .05
        below = FrozenDeterministicVoltageShape014(baseline, seed=2026101640)
        below.basis_head.weight.zero_()
        cal_below = calibrate_initial_head014(below, decoded, original['signal_pre'], zbanks,
            target_RMS_V=ADC_STEP, bisections=16)
        assert cal_below['calibration_status'] == 'already_below' and cal_below['initial_weight_multiplier'] == 1
    checks.append(dict(name='actual_voltage_RMS_both_finite_solver_branches', calibrated=cal, already_below=cal_below,
        later_gain_unchanged=.05, persistent_scale_state=False))

    opt = torch.optim.AdamW(list(new.trainable_parameters()), lr=.0002)
    new.assert_optimizer_membership(opt)
    extra = torch.nn.Parameter(torch.ones(()))
    wrong_opt = torch.optim.AdamW(list(new.trainable_parameters())+[extra], lr=.0002)
    assert_raises(lambda: new.assert_optimizer_membership(wrong_opt), 'exactly')
    assert tensor_state_digest010(baseline) == frozen
    checks.append(dict(name='optimizer_whitelist_excludes_everything_but_head', wrong_parameter_rejected=True,
        frozen_A_bytes_unchanged=True))

    # A small real semantic path with artificial arrays/A/005; two paired head
    # updates are engineering fixtures and are explicitly labelled artificial.
    p = driver.settings014()
    p['G_updates'] = 1
    p['training'].update(source_parents_per_update=8, draws_per_parent=2, real_rows_per_update=16,
        forward_chunk_rows=8, status_interval=1, checkpoint_steps=[1])
    p['calibration'].update(parents=4, draws=4)
    p['probe'].update(steps=[0, 1], parents=4, single_parent_positions=[0, 1, 2, 3], real_reference_rows=16)
    p['evaluation'].update(parents=128, draws=2, chunk_rows=32,
        projection_seed=2026101590, bank_seeds=[dict(shape=191, noise005=192, extension=193), dict(shape=194, noise005=195, extension=196)])
    p['contracts'] = dict(tolerance_table=dict(path='artificial_table_only', sha256='artificial'))
    p['budget_caps'] = driver.budget_caps014(p)
    store = driver.Store014(out/'ARTIFICIAL_BATCH', p, {}, artificial=True)
    info = dict(source_id=data['source_ids'], role=data['source_role'], canonical_Q_Vns=data['source'].sum(1),
        amplitude_384_V=data['source_carrier384'].max(1), PC_scores_384_Qnorm=np.zeros((128, 2)),
        PC_valid_384_Qnorm=np.ones(128, bool), physical_weight_known=np.zeros(128, bool))
    rt = driver.Runtime014(p, data, baseline, fixture.ArtificialExtension().eval(), stats, integral_stats, store, 'cpu', info)
    state, calibration = rt.calibrate(p['seed_sets'][0], store.root/'seed0_CALIBRATION')
    start_hashes, first_updates = [], []
    for candidate in ('C0', 'C1'):
        engine = driver.Engine014(rt, p['seed_sets'][0], candidate, state, calibration, store.root/('seed0_'+candidate))
        start_hashes.append(tensor_state_digest010(engine.model.basis_head))
        engine.checkpoint('initial.pt', 'initial')
        initial_metrics = engine.evaluate(initial=True)
        records0 = engine.probe()
        engine.step()
        engine.checkpoint('final.pt', 'final')
        records1 = engine.probe()
        final_metrics = engine.evaluate(initial=False)
        assert store.counts['G_updates'] == len(first_updates)+1
        proof = json.loads((engine.out/'FIRST_UPDATE_PROOF.json').read_text('utf8'))
        assert proof['artificial_only'] is True and proof['actual_G_updates'] == 1
        assert proof['gradient_nonzero'] and proof['actual_parameter_update_nonzero'] and proof['frozen_A_005_unchanged']
        assert len(records0) == len(records1) == 2*3*13
        assert sum(r['AD_queries'] for r in records0) <= 234
        assert sum(r['AD_queries'] for r in records1) <= 234
        assert all(r.get('AD_queries') in (0, 3) for r in records0+records1)
        assert len(initial_metrics) == 12 and len(final_metrics) == 6
        first_updates.append(proof)
    assert start_hashes[0] == start_hashes[1] == calibration['calibrated_head_sha256']
    assert store.counts['G_updates'] == store.counts['optimizer_steps'] == store.counts['training_backward_queries'] == 2
    assert store.counts['new_AD_queries'] == store.counts['training_backward_queries']+store.counts['probe_AD_queries']
    assert store.counts['head_total_kernel_rows'] == store.counts['head_actual_parameter_forward_rows']+store.counts['calibration_map_rows']
    assert not store.inflight and store.entered == store.completed
    rt.verify(files=True)
    checks.append(dict(name='two_paired_artificial_native_engine_steps_and_full_semantics', initial_head_SHA_identical=True,
        synthetic_G_updates=2, first_update_proof_both_artificial=True, counters=store.counts,
        entered=store.entered, completed=store.completed, probe_strata_and_empty_reasons=True,
        full_hard_observation_reports_and_lossless_rows=True))

    # Refuse a draft before checkpoint/array loaders; no output is created.
    draft = driver.settings014()
    cp = out/'DRAFT.json'
    write_new(cp, draft)
    calls = dict(np_load=0, torch_load=0)
    np_original, torch_original = np.load, torch.load
    def forbidden_np(*a, **kw):
        calls['np_load'] += 1
        raise AssertionError('unexpected private array decode')
    def forbidden_torch(*a, **kw):
        calls['torch_load'] += 1
        raise AssertionError('unexpected trained CP load')
    np.load, torch.load = forbidden_np, forbidden_torch
    try:
        assert_raises(lambda: driver.run014(cp, sha(cp), out/'must_not_exist', 'cpu'), 'draft')
    finally:
        np.load, torch.load = np_original, torch_original
    assert calls == dict(np_load=0, torch_load=0) and not (out/'must_not_exist').exists()
    checks.append(dict(name='draft_fail_closed_before_private_numeric_load', calls=calls))

    error_store = driver.Store014(out/'ARTIFICIAL_FAILURE_ACCOUNTING', p, {}, artificial=True)
    def injected():
        raise RuntimeError('artificial native operation did not return')
    assert_raises(lambda: error_store.call('injected_head', dict(head_actual_parameter_forward_rows=2), injected), 'did not return')
    assert error_store.counts['head_actual_parameter_forward_rows'] == 0
    assert error_store.attempted['head_actual_parameter_forward_rows'] == 2
    assert error_store.inflight and error_store.entered == 1 and error_store.completed == 0
    checks.append(dict(name='entered_completed_inflight_failure_cost_preserved', completed=0,
        attempted=2, actual_completion_unknown=True, no_automatic_restart=True))

    assert protected == {rel:sha(ROOT/rel) for rel in protected}
    assert tensor_state_digest010(baseline) == frozen
    result = dict(schema='n01/wp2-manual-semantics/v1', status='PASS', artificial_only=True,
        private_data_reads=0, trained_checkpoint_loads=0, real_NN_rows=0, real_AD_queries=0,
        scientific_G_updates=0, SSH_calls=0, elapsed_seconds=time.monotonic()-started,
        checks=checks, protected_original_n01_source_files=len(protected), protected_sources_unchanged=True,
        new_source_SHA256={rel:sha(ROOT/rel) for rel in driver.OWNED_FILES})
    write_new(out/'MANUAL_SEMANTICS.json', result)
    print(json.dumps(dict(status=result['status'], checks=len(checks), elapsed_seconds=result['elapsed_seconds']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
