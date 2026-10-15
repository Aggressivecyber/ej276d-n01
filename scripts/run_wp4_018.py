"""CPU-only saved-array replay of WP4-017 and the prospective WP4-018 contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

# Import only the three pure numerical modules, without n01/__init__'s model imports.
package = types.ModuleType('n01')
package.__path__ = [str(Path(__file__).resolve().parents[1]/'src'/'n01')]
sys.modules['n01'] = package
from n01.saved_array_kernels017 import scalar_moments
from n01.stats017 import (FEATURES017, TIME017, charge_identifiability017, coverage017,
    coverage_cells017, covariance_cell017, energy_cell017, groups_weights017,
    missingness_cells017, pooled_wave017, scalar_banks017, stage_gate017,
    timing_cell017, variance_curve_cell017, waveform_cells017)
from n01.stats018 import (DISTANCE_COUNTS018, OBSERVABLE_FIELDS018, TIME_FIELDS018,
    covariance8_cell018, energy_discrepancy018, observable018, observable_cell018,
    pair_distances018, scalar_features018, self_mean018, stage_gate018)
from run_wp4_017 import compact_report017
from wp1_ops import atomic_json, now, read_json, require, sha, write_new

RUN_ID = 'wp4_018_20261016'
HYPOTHESES = ('fast', 'slow', 'common_level')
TRIALS = ('T1_paired_seed0', 'T1_paired_seed1')
PREFIX = 'runs/primary/artifacts/'
REQUIRED_SOURCES = ('scripts/run_wp4_018.py', 'src/n01/stats018.py',
                    'scripts/run_wp4_017.py', 'scripts/wp1_ops.py',
                    'src/n01/stats017.py', 'src/n01/saved_array_kernels017.py')


def phase018(output, phase, started, p, **detail):
    elapsed = time.monotonic()-started
    size = sum(q.stat().st_size for q in output.rglob('*') if q.is_file())
    require(elapsed < p['resource_caps']['wall_seconds'], '018 wall budget exceeded')
    require(size <= p['resource_caps']['output_MiB']*1024**2, '018 output budget exceeded')
    atomic_json(output/'PHASE.json', dict(run_id=RUN_ID, UTC=now(), phase=phase,
        elapsed_seconds=elapsed, output_bytes=size, distance_counts=DISTANCE_COUNTS018, **detail))


def subset018(obj, rows):
    return {k: ([subset018(v, rows) for v in value] if isinstance(value, list)
                else value[rows] if isinstance(value, np.ndarray) else value) for k, value in obj.items()}


def load_kernel018(saved, directory, h, ids, p, output, started, key):
    """All scientific values are rederived from saved Y; REPORT is a check only."""
    n, r = len(ids), 32
    waves, scalars, flags, embeddings, states, times, masks, bounds = [], [], [], [], [], [], [], []
    native_means, native_variances = [], []
    rows, draws = np.repeat(np.arange(n), r), np.tile(np.arange(r), n)
    for bank in range(2):
        path = saved/PREFIX/directory/f'bank{bank}_{h}_HARD_ROWS.npz'
        with np.load(path, allow_pickle=False) as ar:
            np.testing.assert_array_equal(ar['source_parent_row'], rows)
            np.testing.assert_array_equal(ar['source_parent_ids'], ids[rows])
            np.testing.assert_array_equal(ar['draw_index'], draws)
            np.testing.assert_array_equal(ar['bank_index'], np.full(len(rows), bank))
            codes, mean, peaks = ar['ADC_crop_codes'], ar['baseline_mean_V'], ar['peak_index']
            require(codes.shape == (n*r, 256) and codes.dtype.kind == 'u', '018 original code256 schema')
            y = codes.astype(np.float64)/16383-.5-mean[:, None]
            if 'IEEE754_reconstruction_XOR_uint64' in ar.files:
                y.view(np.uint64)[:] ^= ar['IEEE754_reconstruction_XOR_uint64']
            original_flags = {name: ar[name] for name in ar.files if name.startswith('flag__')}
        with np.load(saved/PREFIX/directory/f'bank{bank}_BASELINE100.npz', allow_pickle=False) as ar:
            np.testing.assert_array_equal(ar['source_parent_row'], rows)
            np.testing.assert_array_equal(ar['source_parent_ids'], ids[rows])
            np.testing.assert_array_equal(ar['draw_index'], draws)
            scalar, morph = scalar_features018(y, ar['baseline100'], peaks)
        for name, value in morph.items():
            if value.dtype.kind == 'b':
                np.testing.assert_array_equal(value, original_flags['flag__'+name])
        obs = observable018(y, scalar, morph, p['scales'][h])
        # The identical map is evaluated again on the same complete teacher rows.
        if key == 'T1':
            identical = observable018(y, scalar, morph, p['scales'][h])
            for field in obs:
                np.testing.assert_array_equal(obs[field], identical[field])
            del identical
        scalar = scalar.reshape(n, r, 9)
        wave = y.reshape(n, r, 256)
        wave_mean, wave_variance = wave.mean(1), wave.var(1, ddof=1)
        native = scalar_moments(scalar)
        with np.load(saved/PREFIX/directory/f'bank{bank}_{h}_REPORT.npz', allow_pickle=False) as ar:
            np.testing.assert_array_equal(scalar, ar['scalar9'])
            np.testing.assert_array_equal(wave_mean, ar['wave_mean'])
            np.testing.assert_array_equal(wave_variance, ar['wave_variance'])
            for name, value in native.items():
                np.testing.assert_array_equal(value, ar['scalar_'+name])
        waves.append(wave)
        scalars.append(scalar)
        native_means.append(wave_mean)
        native_variances.append(wave_variance)
        flags.append({name: value.reshape(n, r).mean(1) for name, value in original_flags.items()})
        embeddings.append(obs['embedding'].reshape(n, r, 19))
        states.append(obs['state'].reshape(n, r))
        times.append(obs['timing_values_ns'].reshape(n, r, 6))
        masks.append(obs['timing_defined'].reshape(n, r, 6))
        bounds.append(obs['conditional_bound_defined'].reshape(n, r))
        if key == 'T1' and h == HYPOTHESES[0] and bank == 0:
            phase018(output, 'first_teacher_bank_decoded', started, p,
                parents=n, original_bank=bank, draws=r, exact_REPORT_checks=True,
                decoded_hard_rows=len(y), NN_forward_rows=0, new_RNG_requests=0)
        del y, codes, scalar, morph, obs, original_flags, native, wave
    scalar = np.stack(scalars, axis=1)
    mean, variance = pooled_wave017(native_means[0], native_variances[0], native_means[1], native_variances[1])
    data = dict(waves=np.stack(waves, axis=1), scalar=scalar, moments=scalar_banks017(scalar),
                wave_mean=mean, wave_variance=variance, embedding=np.stack(embeddings, axis=1),
                state=np.stack(states, axis=1), timing=np.stack(times, axis=1),
                defined=np.stack(masks, axis=1), conditional_bound=np.stack(bounds, axis=1),
                flags={name: .5*(flags[0][name]+flags[1][name]) for name in flags[0]})
    np.savez_compressed(output/f'{key}_{h}_OBSERVABLE.npz', embedding=data['embedding'],
        state=data['state'], timing_values_ns=data['timing'], timing_defined=data['defined'],
        conditional_bound_defined=data['conditional_bound'],
        original_full64_undefined_fraction=data['moments']['full_slot_undefined_fraction'],
        **{'original_full64_'+name: value for name, value in data['flags'].items()})
    return data


def wave_features018(y):
    delta = 1/16383
    return np.concatenate((y/(16*delta), y.sum(-1, keepdims=True)/(256*delta)), axis=-1)


def wave_reference018(teacher):
    A, B = wave_features018(teacher[:, 0]), wave_features018(teacher[:, 1])
    return pair_distances018(A, B).mean((1, 2))-.25*(self_mean018(A)+self_mean018(B))


def wave_energy018(student, teacher):
    scores = []
    for bank in range(2):
        S, T = wave_features018(student[:, bank]), wave_features018(teacher[:, bank])
        scores.append(pair_distances018(S, T).mean((1, 2))-.5*self_mean018(S))
    return .5*(scores[0]+scores[1])


def replay_cells018(S, T, U, test, groups, weights, energies, h):
    coverage = coverage017(S['scalar'], T['scalar'], U)
    sm, tm = subset018(S['moments'], test), subset018(T['moments'], test)
    selected_S, selected_T = S['scalar'][test], T['scalar'][test]
    C = {k: (v[test] if isinstance(v, np.ndarray) else v) for k, v in coverage.items()}
    cells, extra, cov8 = [], [], []
    for group, g in groups.items():
        for weight, w in weights.items():
            tag = dict(group=group, weights=weight, hypothesis=h)
            for name, item in waveform_cells017(S['wave_mean'][test], S['wave_variance'][test],
                    T['wave_mean'][test], T['wave_variance'][test], g, w).items():
                cells.append(dict(item, metric='hard_wave_'+name, **tag))
            cells.append(dict(covariance_cell017(selected_S, selected_T, g, w, moments=(sm, tm)),
                              metric='conditional_cov9', **tag))
            cov8.append(dict(covariance8_cell018(selected_S, selected_T, g, w, moments=(sm, tm)),
                             metric='conditional_cov8', **tag))
            for f in TIME017:
                cells.append(dict(timing_cell017(selected_S, selected_T, f, g, w, moments=(sm, tm)),
                                  metric='time_'+FEATURES017[f], **tag))
            for item in coverage_cells017(C, g, w):
                cells.append(dict(item, metric='PIT_coverage', **tag))
            for item in missingness_cells017(sm, tm,
                    {k: v[test] for k, v in S['flags'].items()},
                    {k: v[test] for k, v in T['flags'].items()}, g, w):
                (cells if item['required_for_gate'] else extra).append(dict(item,
                    metric='missingness_or_censor' if item['required_for_gate'] else 'other_retained_flag_fraction', **tag))
            cells.append(dict(energy_cell017(energies['final'][test], energies['initial'][test],
                energies['reference'][test], g, w), metric='conditional_energy', **tag))
    return cells, extra, cov8, coverage


def compare_value018(actual, expected, arrays, metrics, path, *, energy=False):
    """Resolve producer array references and check every cell value, not just counts."""
    if isinstance(expected, dict) and 'array' in expected:
        expected = arrays[expected['array']]
    if isinstance(actual, np.ndarray):
        expected = np.asarray(expected)
        require(actual.shape == expected.shape and actual.dtype == expected.dtype, '018 replay array schema: '+path)
        if actual.dtype.kind in 'fc':
            np.testing.assert_allclose(actual, expected, rtol=1e-10 if energy else 5e-12,
                atol=1e-9 if energy else 5e-14, equal_nan=True, err_msg=path)
            finite = np.isfinite(actual) & np.isfinite(expected)
            if finite.any():
                metrics['maximum_absolute_numeric_difference'] = max(metrics['maximum_absolute_numeric_difference'],
                    float(np.abs(actual[finite]-expected[finite]).max()))
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=path)
        metrics['array_checks'] += 1
    elif isinstance(actual, dict):
        require(actual.keys() == expected.keys(), '018 replay field schema: '+path)
        for key in actual:
            compare_value018(actual[key], expected[key], arrays, metrics, path+'/'+str(key), energy=energy)
    elif isinstance(actual, (tuple, list)):
        require(len(actual) == len(expected), '018 replay sequence length: '+path)
        for i, value in enumerate(actual):
            compare_value018(value, expected[i], arrays, metrics, path+'/'+str(i), energy=energy)
    elif isinstance(actual, (float, np.floating)):
        if expected is None:
            require(not np.isfinite(actual), '018 replay unexpected undefined: '+path)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-10 if energy else 5e-12,
                atol=1e-9 if energy else 5e-14, err_msg=path)
            metrics['maximum_absolute_numeric_difference'] = max(metrics['maximum_absolute_numeric_difference'], abs(float(actual)-expected))
        metrics['scalar_float_checks'] += 1
    else:
        require(actual == expected, '018 replay exact value differs: '+path)
        metrics['exact_checks'] += 1


def fixed_censored_Q_control018(T, teacher_self, teacher_null, test, groups, weights, h):
    """Only censored-group Q/time coupling changes; both Q and state marginals stay."""
    z = T['embedding'].reshape(len(T['embedding']), 64, 19)
    control = z.copy()
    state, timing = T['state'].reshape(-1, 64), T['timing'].reshape(-1, 64, 6)
    affected = np.zeros(len(z), np.int64)
    eligible = np.zeros(len(z), np.int64)
    before, after = np.full(len(z), np.nan), np.full(len(z), np.nan)
    for parent in range(len(z)):
        slots = np.flatnonzero((state[parent] == 1) & np.isfinite(timing[parent, :, 3]))
        eligible[parent] = len(slots)
        if len(slots) < 2:
            continue
        order = slots[np.lexsort((slots, timing[parent, slots, 3]))]
        values = z[parent, slots, 15]
        control[parent, order, 15] = np.sort(values)[::-1]
        affected[parent] = np.count_nonzero(control[parent, slots, 15] != values)
        q, t = z[parent, slots, 15], timing[parent, slots, 3]
        changed = control[parent, slots, 15]
        before[parent] = np.mean((q-q.mean())*(t-t.mean()))
        after[parent] = np.mean((changed-changed.mean())*(t-t.mean()))
        np.testing.assert_array_equal(np.sort(control[parent, :, 15]), np.sort(z[parent, :, 15]))
    np.testing.assert_array_equal(control[state != 1], z[state != 1])
    np.testing.assert_array_equal(control[:, :, :15], z[:, :, :15])
    np.testing.assert_array_equal(control[:, :, 16:], z[:, :, 16:])
    d = energy_discrepancy018(control, z, teacher_self=teacher_self)
    cells = [dict(observable_cell018(d[test], teacher_null[test], g, weights[w]),
                  group=name, weights=w, hypothesis=h) for name, g in groups.items() for w in weights]
    return dict(hypothesis=h, affected_parents=int((affected > 0).sum()), affected_slots=int(affected.sum()),
        evaluable_parents=int((eligible >= 2).sum()), evaluable_slots=int(eligible[eligible >= 2].sum()),
        ineligible_parents_with_undefined_coupling=int((eligible < 2).sum()),
        max_abs_Q_change_in_T_units=float(np.abs(control[:, :, 15]-z[:, :, 15]).max()),
        overall_Q_marginal_exact=True, state_marginal_exact=True, censored_Q_marginal_exact=True,
        complete_and_other_rows_exact=True, tests_Q_visible_time_coupling_only=True,
        tests_state_Q_dependence=False, new_required_gate=False, cells=cells), dict(
            discrepancy=d, affected_slots_per_parent=affected,
            eligible_slots_per_parent=eligible, covariance_defined=eligible >= 2,
            censored_Q_t90_covariance_before=before, censored_Q_t90_covariance_after=after)


def run018(config, config_sha256, output):
    started = time.monotonic()
    cp = Path(config).resolve()
    require(sha(cp) == config_sha256, '018 config SHA differs')
    p = read_json(cp)
    require(p['run_id'] == RUN_ID and p['status'] == 'frozen', '018 requires frozen configuration')
    require(tuple(p['hypotheses']) == HYPOTHESES and p['parents'] == 6593 and p['banks'] == 2
            and p['draws_per_bank'] == 32, '018 fixed saved-array axes')
    require(p['time_scale_software_ns'] == 1 and p['new_observable_energy_tolerance'] == .10
            and p['new_observable_MC_half_width_limit'] == .05, '018 prospective representation/thresholds differ')
    require(set(p['scientific_budget'].values()) == {0}, '018 has no new scientific generation budget')
    require(set(REQUIRED_SOURCES) == set(p['source_files']), '018 exact pure saved-evaluator source closure')
    for name, record in p['source_files'].items():
        source = Path(record['path'])
        require(source.stat().st_size == record['bytes'] and sha(source) == record['sha256'], '018 frozen source differs: '+name)
    require('torch' not in sys.modules, '018 pure numerical imports unexpectedly loaded torch')
    out, saved = Path(output).absolute(), Path(p['saved_root'])
    require(str(out) == p['remote_output'] and not out.exists(), '018 fresh exact owned output required')
    require(os.name != 'nt' or out.drive.upper() in ('E:', 'D:'), '018 all Windows output stays E/D')
    if os.name == 'posix':
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (6*1024**3, 6*1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (21590, 21600))
    out.mkdir(parents=True)
    DISTANCE_COUNTS018.clear()
    write_new(out/'STARTED.json', dict(run_id=RUN_ID, UTC=now(), config_sha256=config_sha256,
        phase='verify_frozen_saved_inputs', scientific_budget=p['scientific_budget'], PID=os.getpid()))
    manifest_path = cp.parent/p['saved_input_manifest']
    require(sha(manifest_path) == p['saved_input_manifest_sha256'], '018 input manifest differs')
    manifest = read_json(manifest_path)
    require(manifest['file_count'] == 75 and manifest['saved_root'] == str(saved), '018 exact 75-file bound')
    for relative, record in manifest['entries'].items():
        path = saved/relative
        require(path.resolve().is_relative_to(saved.resolve()) and not path.is_symlink(), '018 input escaped saved root')
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], '018 saved input differs: '+relative)
    write_new(out/'INPUT_VERIFIED.json', dict(manifest_sha256=sha(manifest_path), file_count=75,
        compressed_input_bytes=manifest['compressed_input_bytes'], entries=manifest['entries']))
    phase018(out, 'manifest_verified', started, p, file_count=75)
    with np.load(saved/PREFIX/'cache/PARENT_SPLIT.npz', allow_pickle=False) as ar:
        fit, test, ids = ar['fit_rows'], ar['test_rows'], ar['source_ids']
    Q = np.load(saved/PREFIX/'cache/Q.npy', allow_pickle=False)
    require(len(ids) == len(Q) == 6593 and len(fit) == 5274 and len(test) == 1319, '018 original parent denominator')
    with np.load(saved/PREFIX/'cache/ORIGINAL_PHYSICAL_WEIGHTS.npz', allow_pickle=False) as ar:
        np.testing.assert_array_equal(ar['source_ids'], ids)
        original_weights = ar['weights']
    groups, weights, cuts = groups_weights017(Q[fit], Q[test], ids[test])
    np.savez_compressed(out/'PARENT_AXES.npz', source_ids=ids, fit_rows=fit, test_rows=test,
        fixed_A_Q_Vns=Q, original_physical_weights=original_weights,
        **{'group_'+k: v for k, v in groups.items()}, **{'comparison_weight_'+k: v for k, v in weights.items()})
    with np.load(saved/PREFIX/'T1/PIT_UNIFORMS.npz', allow_pickle=False) as ar:
        uniforms = ar['uniforms']
    trials, old_trials, verification = [], [], []
    for trial in TRIALS:
        metadata = read_json(saved/PREFIX/p['kernels'][trial]/'EVALUATION.json')
        a, b = metadata['a_Vns'], metadata['b']
        analytic = [dict(variance_curve_cell017(Q[test], a, b, g, weights[w]),
            metric='analytic_charge_variance', group=name, weights=w) for name, g in groups.items() for w in weights]
        old_trials.append(dict(trial=trial, cells=analytic.copy()))
        trials.append(dict(trial=trial, cells=analytic.copy(), selection_diagnostics=[],
            other_retained_flag_diagnostics=[], parameter_recovery_secondary=charge_identifiability017(Q[fit], Q[test], a, b)))
    negative_controls, identities, fixed_controls = [], [], []
    for hi, h in enumerate(HYPOTHESES):
        T = load_kernel018(saved, p['kernels']['T1'], h, ids, p, out, started, 'T1')
        zt = T['embedding'].reshape(len(ids), 64, 19)
        teacher_self = self_mean018(zt)
        teacher_null = energy_discrepancy018(T['embedding'][:, 0], T['embedding'][:, 1])
        identity = energy_discrepancy018(zt, zt, student_self=teacher_self, teacher_self=teacher_self)
        np.testing.assert_allclose(identity, -teacher_self/64, rtol=5e-12, atol=5e-14)
        identities.append(dict(hypothesis=h, all_parent_mapping_exact=True, all_parent_U_diagonal_correction_verified=True,
            maximum_abs_U_identity_residual=float(np.abs(identity+teacher_self/64).max()), U_forced_zero=False))
        null_cells = [dict(observable_cell018(teacher_null[test], teacher_null[test], g, weights[w]),
            group=name, weights=w, hypothesis=h) for name, g in groups.items() for w in weights]
        negative_controls.extend(null_cells)
        reference = wave_reference018(T['waves'])
        phase018(out, 'teacher_ready', started, p, hypothesis=h, all_parents=len(ids), identity_verified=True)
        fixed_report, fixed_arrays = fixed_censored_Q_control018(T, teacher_self, teacher_null, test, groups, weights, h)
        fixed_controls.append(fixed_report)
        np.savez_compressed(out/f'T1_{h}_CONTROLS.npz', teacher_null=teacher_null,
                            identity_U=identity, fullwave_teacher_reference=reference, **fixed_arrays)
        I = load_kernel018(saved, p['kernels']['initial'], h, ids, p, out, started, 'initial')
        initial = wave_energy018(I['waves'], T['waves'])
        initial_observable = energy_discrepancy018(I['embedding'].reshape(len(ids), 64, 19), zt, teacher_self=teacher_self)
        np.savez_compressed(out/f'initial_{h}_ENERGY.npz', fullwave_energy=initial, observable_discrepancy=initial_observable)
        del I
        phase018(out, 'initial_scored', started, p, hypothesis=h)
        for si, trial in enumerate(TRIALS):
            S = load_kernel018(saved, p['kernels'][trial], h, ids, p, out, started, trial)
            final = wave_energy018(S['waves'], T['waves'])
            discrepancy = energy_discrepancy018(S['embedding'].reshape(len(ids), 64, 19), zt, teacher_self=teacher_self)
            cells, extra, cov8, coverage = replay_cells018(S, T, uniforms[:, :, :, :, hi], test, groups, weights,
                dict(final=final, initial=initial, reference=reference), h)
            old_trials[si]['cells'].extend(cells)
            diagnostics = [c for c in cells if c['metric'] in ('conditional_cov9', 'time_rise10_90') or
                           (c['metric'] == 'PIT_coverage' and c['feature'] == 'rise10_90')]
            trials[si]['selection_diagnostics'].extend(diagnostics)
            trials[si]['cells'].extend([c for c in cells if c['metric'] not in ('conditional_cov9', 'time_rise10_90')
                and not (c['metric'] == 'PIT_coverage' and c['feature'] == 'rise10_90')])
            trials[si]['cells'].extend(cov8)
            trials[si]['other_retained_flag_diagnostics'].extend(extra)
            trials[si]['cells'].extend([dict(observable_cell018(discrepancy[test], teacher_null[test], g, weights[w]),
                metric='observable_joint_energy', group=name, weights=w, hypothesis=h) for name, g in groups.items() for w in weights])
            np.savez_compressed(out/f'{trial}_{h}_PARENT_COMPARISON.npz', energy_final=final,
                energy_initial=initial, energy_teacher_reference=reference, observable_discrepancy=discrepancy,
                **{'coverage_'+k: v for k, v in coverage.items() if isinstance(v, np.ndarray)})
            metrics = dict(trial=trial, hypothesis=h, array_checks=0, scalar_float_checks=0,
                           exact_checks=0, maximum_absolute_numeric_difference=0.)
            with np.load(saved/PREFIX/p['kernels'][trial]/f'PARENT_COMPARISON_{h}.npz', allow_pickle=False) as ar:
                for name, value in (('parent_ids', ids), ('test_rows', test), ('energy_final', final),
                                    ('energy_initial', initial), ('energy_teacher_reference', reference)):
                    compare_value018(value, ar[name], {}, metrics, name, energy=name.startswith('energy_'))
                for name, value in coverage.items():
                    if isinstance(value, np.ndarray):
                        compare_value018(value, ar['coverage_'+name], {}, metrics, name)
            verification.append(metrics)
            del S
            phase018(out, f'seed{si}_scored', started, p, hypothesis=h, trial=trial)
        del T, zt
        phase018(out, 'hypothesis_complete', started, p, hypothesis=h)
    legacy_gate = stage_gate017(old_trials)
    for si, trial in enumerate(TRIALS):
        producer = read_json(saved/PREFIX/trial/'COMPARISON.json')
        actual = old_trials[si]['cells']
        require(len(actual) == len(producer['cells']) == 1344, '018 complete original primary count')
        metrics = dict(trial=trial, cells=1344, array_checks=0, scalar_float_checks=0,
                       exact_checks=0, maximum_absolute_numeric_difference=0.)
        with np.load(saved/PREFIX/trial/'COMPARISON.npz', allow_pickle=False) as ar:
            for ci, cell in enumerate(actual):
                compare_value018(cell, producer['cells'][ci], ar, metrics, f'cells/{ci}',
                                 energy=cell['metric'] == 'conditional_energy')
            compare_value018(trials[si]['parameter_recovery_secondary'], producer['parameter_recovery_secondary'],
                             ar, metrics, 'parameter_recovery_secondary')
        verification.append(metrics)
        compact_report017(out/f'{trial}_REPLAY017.json', old_trials[si])
        compact_report017(out/f'{trial}_COMPARISON018.json', trials[si])
    verdict = stage_gate018(trials)
    require(p['distance_budget']['fullwave257_ordered_pair_distances'] == 303805440
        and p['distance_budget']['O19_ordered_pair_distances'] == 870908928,
        '018 frozen distance budget differs')
    require(DISTANCE_COUNTS018 == {257: 303805440, 19: 870908928}, '018 exact distance budget differs')
    # Immutable readback boundary: all 75 original byte streams remain unchanged.
    for relative, record in manifest['entries'].items():
        path = saved/relative
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], '018 changed saved input: '+relative)
    compact_report017(out/'CONTROLS018.json', dict(identity=identities, independent_teacher32=negative_controls,
        fixed_censored_Q_time_sensitivity=fixed_controls, no_oracle_or_hardware_claim=True))
    write_new(out/'REPLAY017_VERIFICATION.json', dict(all_2688_cells_recomputed=True,
        all_signatures_checked=True, all_original_parent_energy_vectors_recomputed_from_hard=True,
        native_scalar9_and_wave_moments_exact=True, old_gate=legacy_gate,
        comparisons=verification, shared_original_statistical_formulas=True,
        independent_full_statistical_implementation=False, old017_conclusion_overwritten=False))
    write_new(out/'STAGE1_GATE018.json', verdict)
    phase018(out, 'completed', started, p, all_2688_old_cells_verified=True, required_new_cells=2544)
    inventory = {str(q.relative_to(out)): dict(bytes=q.stat().st_size, sha256=sha(q))
                 for q in sorted(out.rglob('*')) if q.is_file()}
    write_new(out/'OUTPUT_INVENTORY.json', inventory)
    peak = None
    if os.name == 'posix':
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(run_id=RUN_ID, UTC=now(), native_status='COMPLETE', config_sha256=config_sha256,
        scientific_budget=p['scientific_budget'], distance_counts=DISTANCE_COUNTS018,
        elapsed_seconds=time.monotonic()-started, peak_RSS_KiB=peak,
        old017_status_preserved=True, new_018_gate=verdict, scientific_acceptance=False,
        full_per_parent_arrays_remain_owned_output=True, input_files_unchanged=75)
    write_new(out/'COMPLETE.json', result)
    require(sum(q.stat().st_size for q in out.rglob('*') if q.is_file()) <=
            p['resource_caps']['output_MiB']*1024**2, '018 final output budget exceeded')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = run018(args.config, args.config_sha256, args.output)
    print(json.dumps(dict(native_status=result['native_status'],
        new_018_status=result['new_018_gate']['status'], new_scientific_updates=0), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
