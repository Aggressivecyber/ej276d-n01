"""Read-only audit of the interrupted derivative prefix; payloads stay on NAS."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import os
import signal
import time
import zipfile
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
import numpy as np

started = time.monotonic()
signal.alarm(900)
root = Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_diagnostic_20261020')
p = INPUT['config']
manifest = INPUT['manifest']
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot == '83c9f761-c124-42e1-89b8-5f45f0b3acc4'
assert not any(Path('/proc', str(pid)).exists() for pid in (201218, 201219, 201226, 201227))
print(json.dumps(dict(stage='PREFIX_AUDIT_STARTED', UTC=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19], boot_id=boot, source_files=len(manifest['files']), maximum_seconds=900, workers=4)), flush=True)
kept = {}
source_hashes = {}
source_errors = []
with ThreadPoolExecutor(max_workers=4) as pool:
    pending = [(name, entry, pool.submit((root/name).read_bytes)) for name, entry in manifest['files'].items()]
    for i, (name, entry, future) in enumerate(pending, 1):
        try:
            raw = future.result()
        except OSError as error:
            source_errors.append(dict(path=name, error_type=type(error).__name__, error=str(error)))
            print(json.dumps(dict(stage='SOURCE_READ_ERROR', index=i, path=name, error=str(error))), flush=True)
            continue
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == entry['bytes'] and digest == entry['sha256'], name
        source_hashes[name] = digest
        if name in p['saved_arrays'] or name == p['roles_file'] or name.endswith('stable_projection_diagnostic_01/terminal_01/TERMINAL_REVIEW.json'):
            kept[name] = raw
        if i % 12 == 0 or i == len(pending):
            print(json.dumps(dict(stage='SOURCE_HASH_PROGRESS', verified=len(source_hashes), processed=i, elapsed_seconds=time.monotonic()-started, UTC=datetime.now(timezone.utc).isoformat())), flush=True)
assert not source_errors, source_errors
assert len(source_hashes) == 185
with np.load(io.BytesIO(kept[p['roles_file']]), allow_pickle=False) as z:
    roles = dict(z)
review_name = 'receipts/joint_response_01_20261020/stable_projection_diagnostic_01/terminal_01/TERMINAL_REVIEW.json'
previous_review = json.loads(kept[review_name])
names = [f.relative_to(root).as_posix() for sub in ('run_01', 'control_01') for f in (root/sub).iterdir() if f.is_file()]
names.append('dispatcher.log')
output_hashes = {}
payloads = {}
with ThreadPoolExecutor(max_workers=4) as pool:
    pending = [(name, pool.submit((root/name).read_bytes)) for name in names]
    for i, (name, future) in enumerate(pending, 1):
        raw = future.result()
        output_hashes[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        payloads[name] = raw
        if i % 8 == 0 or i == len(pending):
            print(json.dumps(dict(stage='OUTPUT_HASH_PROGRESS', verified=i, total=len(pending), elapsed_seconds=time.monotonic()-started)), flush=True)
failed = json.loads(payloads['run_01/FAILED.json'])
assert failed == INPUT['failed']
assert 'run_01/COMPLETE.json' not in payloads and 'run_01/RESULT.json' not in payloads
assert failed['execution_counts']['finite_difference_columns'] == 327
with np.load(io.BytesIO(payloads['run_01/REAL_CALIBRATION_ROLES.npz']), allow_pickle=False) as z:
    for key in z.files:
        np.testing.assert_array_equal(z[key], roles[key])
counts = Counter()
summaries = {}
payload_errors = []
for seed in range(2):
    with np.load(io.BytesIO(kept[p['saved_arrays'][seed]]), allow_pickle=False) as z:
        original = dict(z)
    for label in ('FIT0', 'FIT1', 'FIT2', 'IID0', 'IID1', 'IID2', 'IID3'):
        stem = f'run_01/seed{seed}_{label}_RECENTERED'
        if stem+'.npz' not in payloads:
            continue
        try:
            with np.load(io.BytesIO(payloads[stem+'.npz']), allow_pickle=False) as z:
                a = dict(z)
        except (zipfile.BadZipFile, ValueError, EOFError) as error:
            payload_errors.append(dict(path=stem+'.npz', error_type=type(error).__name__, error=str(error)))
            continue
        for value in a.values():
            assert np.isfinite(value).all()
            counts['finite_arrays'] += 1
        np.testing.assert_array_equal(a['theta'], original['curvature_update_control'])
        np.testing.assert_array_equal(a['previous_theta'], original['theta'])
        for key in ('initial', 'lower', 'upper'):
            np.testing.assert_array_equal(a[key], original[key])
        for key, size, role in [('source_parents', 1024, 'source_train_rows'), ('real_rows', 4096, 'real_train_rows')]:
            np.testing.assert_array_equal(a[key], original[label+'_'+key])
            assert len(a[key]) == size and np.isin(a[key], roles[role]).all()
            counts['role_entries_replayed'] += size
        np.testing.assert_array_equal(a['cholesky'], original[label+'_cholesky'])
        assert a['random_seed'] == original[label+'_random_seed']
        for key, old_key in [('previous_residual', label+'_baseline_residual'), ('previous_native', label+'_baseline_native'), ('initial_native', label+'_initial_native'), ('baseline_residual', 'curvature_update_control_'+label+'_actual_residual'), ('baseline_native', 'curvature_update_control_'+label+'_actual_native')]:
            np.testing.assert_allclose(a[key], original[old_key], rtol=1e-12, atol=1e-12)
        k = a['FD_plus_theta'].shape[1]
        assert 1 <= k <= 53
        plus = np.repeat(a['theta'][:, None], k, axis=1)
        minus = plus.copy()
        for j in range(k):
            plus[j, j] = min(a['upper'][j], a['theta'][j]+.004)
            minus[j, j] = max(a['lower'][j], a['theta'][j]-.004)
        np.testing.assert_array_equal(a['FD_plus_theta'], plus)
        np.testing.assert_array_equal(a['FD_minus_theta'], minus)
        width = np.diag(plus[:k])-np.diag(minus[:k])
        J = (a['FD_plus_residual']-a['FD_minus_residual'])/width
        N = (a['FD_plus_native']-a['FD_minus_native'])/width
        assert J.shape == (816, k) and N.shape == (9, k)
        assert not N[2::3].any()
        for key in ('FD_plus_native', 'FD_minus_native'):
            np.testing.assert_array_equal(a[key][2::3], np.repeat(a['baseline_native'][2::3, None], k, axis=1))
        complete = k == 53 and 'jacobian' in a and 'native_jacobian' in a and stem+'.json' in payloads
        if complete:
            np.testing.assert_allclose(a['jacobian'], J, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(a['native_jacobian'], N, rtol=1e-12, atol=1e-12)
            report = json.loads(payloads[stem+'.json'])
            assert report['FD_columns_completed'] == 53
            check = report['center_check']
            L = a['cholesky']
            variance = np.square(L).sum(1)
            for vector, recorded in [(a['previous_residual'], report['original_risks']), (a['baseline_residual'], check['risks'])]:
                for ih, h in enumerate(('fast', 'slow', 'common_level')):
                    white = vector[ih*272:(ih+1)*272]*np.sqrt(3)
                    raw = L@white
                    np.testing.assert_allclose(recorded[h]['joint_whitened'], np.mean(white**2), rtol=1e-11)
                    for ib, block in enumerate(('wave', 'charge_morphology', 'peak_codes', 'baseline', 'joint')):
                        take = slice(56*ib, 56*(ib+1)) if ib < 4 else slice(224, 272)
                        np.testing.assert_allclose(recorded[h][block], np.mean(raw[take]**2/variance[take]), rtol=1e-10, atol=1e-12)
                    counts['risk_views_recomputed'] += 1
            ratios = (a['baseline_native']/a['previous_native']).reshape(3, 3)
            anchors = (a['baseline_native']/a['initial_native']).reshape(3, 3)
            np.testing.assert_allclose(check['native_ratios'], ratios, rtol=1e-12)
            np.testing.assert_allclose(check['initial_anchor_ratios'], anchors, rtol=1e-12)
            ng = bool((ratios <= [1.01, 1.01, 1.10]).all() and (anchors <= [1.01, 1.01, 1.10]).all())
            old, new = report['original_risks'], check['risks']
            joint = np.mean([v['joint_whitened'] for v in new.values()])/np.mean([v['joint_whitened'] for v in old.values()])
            macro = all(new[h][b] <= 1.15*old[h][b]+2/4096 for h in old for b in ('wave', 'charge_morphology', 'peak_codes', 'baseline'))
            trust = check['trust']
            tr = trust['mean_relative_RMS'] <= .08 and trust['p99_relative_RMS'] <= .25 and max(v for d in trust['flag_count_changes'].values() for v in d.values()) <= .005*32768
            assert tr == trust['pass'] and trust['denominator'] == 32768
            jg = bool(joint <= .99 and macro and tr)
            assert ng == check['native_pass'] and jg == check['original_joint_accept'] and (ng and jg) == check['accepted']
            assert check['accepted'] == previous_review['summaries'][str(seed)]['by_model']['curvature_update_control'][label]['accepted']
            counts['control_checks_recomputed'] += 1
        counts['finite_difference_columns_recomputed'] += k
        counts['libraries_with_valid_arrays'] += 1
        counts['complete_libraries'] += int(complete)
        summaries[f'seed{seed}_{label}'] = dict(columns=k, complete=complete, joint_J_norm=float(np.linalg.norm(J)), native_N_norm=float(np.linalg.norm(N)))
elapsed = time.monotonic()-started
result = dict(status='SAVED_PREFIX_AUDITED_INCOMPLETE_JOB', UTC=datetime.now(timezone.utc).isoformat(), elapsed_seconds=elapsed, source_files_verified=185, source_pins=184, output_hashes=output_hashes, audit_counts=dict(counts), summaries=summaries, payload_errors=payload_errors, driver_reported_failure_counts=failed['execution_counts'], unavailable_executed_columns=327-counts['finite_difference_columns_recomputed'], unexecuted_registered_columns=742-327, native_exit=json.loads(payloads['control_01/exit_receipt.json']) if 'control_01/exit_receipt.json' in payloads else None, new_solves=0, new_generator_calls=0, parameter_updates=0, restoration_eligible=False, scientific_acceptance=False, evidence_limit='Available saved arrays, identities and gate arithmetic audited. Missing or corrupt payloads remain explicit. No full successful collection, full-waveform regeneration or restoration solve.')
assert result['unavailable_executed_columns'] >= 0
print(json.dumps(dict(stage='PREFIX_AUDIT_COMPLETE', result=result)), flush=True)
