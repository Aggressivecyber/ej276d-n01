from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json
import numpy as np
import torch
from scipy.linalg import eigh

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
T = HERE/'terminal_01'
OUT = T/'collected/run_01'
BASE = HERE.parent/'original_precision_REAL_pilot_01/terminal_01/collected/run_01'
CONTROL = HERE.parent/'curvature_update_diagnostic_01/terminal_01/collected/run_01'
RANDOM = HERE.parent/'random_precision_diagnostic_01'
config = ROOT/'configs/joint_response_01_stable_projection_diagnostic_20261020.json'
p = json.loads(config.read_bytes())
m = json.loads((T/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
s = json.loads((HERE/'SOURCE_MANIFEST.json').read_bytes())
result = json.loads((OUT/'RESULT.json').read_bytes())
complete = json.loads((OUT/'COMPLETE.json').read_bytes())
assert m['native_exit']['exit_code'] == 0 and not m['processes'] and m['complete_present'] and not m['failed_present']
assert m['source_files_verified'] == len(s['files']) == 902
assert hashlib.sha256(config.read_bytes()).hexdigest() == s['config_sha256'] == result['config_sha256']
assert hashlib.sha256((OUT/'RESULT.json').read_bytes()).hexdigest() == complete['result_sha256']
local_hashes = 0
for name, entry in s['files'].items():
    if (ROOT/name).is_file():
        raw = (ROOT/name).read_bytes()
        assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
        local_hashes += 1
for name, digest in p['source_pins'].items():
    assert s['files'][name]['sha256'] == digest
for name, path in json.loads((HERE/'INPUT_MAP.json').read_bytes()).items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == s['files'][name]['sha256']
for name, entry in m['outputs'].items():
    raw = (T/'collected'/name).read_bytes()
    assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
raw = (T/'terminal.tar').read_bytes()
assert len(raw) == m['archive_bytes'] and hashlib.sha256(raw).hexdigest() == m['archive_sha256']
assert m['native_exit']['peak_rss_bytes'] <= 12*1024**3
assert result['peak_GPU_reserved_bytes'] <= 4*1024**3 and result['elapsed_seconds'] <= 360
assert sum(v['bytes'] for v in m['outputs'].values()) <= 64*1024**2
with np.load(OUT/'REAL_CALIBRATION_ROLES.npz', allow_pickle=False) as z:
    roles = dict(z)
with np.load(ROOT/'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz', allow_pickle=False) as z:
    for key in z.files:
        np.testing.assert_array_equal(roles[key], z[key])
with np.load(RANDOM/'STABLE_PROJECTION_PROPOSALS.npz', allow_pickle=False) as z:
    proposals = dict(z)
labels = ('FIT0', 'FIT1', 'FIT2', 'CHECK', 'IID0', 'IID1', 'IID2', 'IID3', 'FRESH')
hypotheses = ('fast', 'slow', 'common_level')
blocks = ('wave', 'charge_morphology', 'peak_codes', 'baseline', 'joint')
counts, summary, projections = Counter(), {}, {}
for seed in range(2):
    with np.load(OUT/f'seed{seed}_STABLE_PROJECTION.npz', allow_pickle=False) as z:
        a = dict(z)
    with np.load(BASE/f'seed{seed}_joint_ITER02.npz', allow_pickle=False) as z:
        old = dict(z)
    with np.load(CONTROL/f'seed{seed}_CURVATURE_UPDATE.npz', allow_pickle=False) as z:
        control = dict(z)
    replays = []
    for rep in range(4):
        with np.load(RANDOM/f'terminal_01/collected/run_01/seed{seed}_rep{rep}_IID.npz', allow_pickle=False) as z:
            replays.append(dict(z))
    report = json.loads((OUT/f'seed{seed}_STABLE_PROJECTION.json').read_bytes())
    for value in a.values():
        assert np.isfinite(value).all()
        counts['finite_arrays'] += 1
    np.testing.assert_array_equal(a['theta'], old['theta'])
    np.testing.assert_array_equal(a['initial'], np.zeros(54))
    np.testing.assert_array_equal(a['curvature_update_control'], control['curvature_update'])
    np.testing.assert_array_equal(a['stable_projection'], proposals[f'seed{seed}_stable_projection'])
    scale = np.sqrt(sum(v['baseline_residual']@v['baseline_residual'] for v in replays)/4)
    jacobians = np.stack([v['jacobian'] for v in replays])*.0125/scale
    mean = jacobians.mean(0)
    B = mean.T@mean
    K = sum((v-mean).T@(v-mean) for v in jacobians)/3
    eigen, V = eigh(B[:53, :53], K[:53, :53]/4+.02*np.trace(B)/54*np.eye(53))
    U, _ = np.linalg.qr(V[:, eigen >= 4.])
    delta = control['curvature_update']-old['theta']
    expected = old['theta'].copy()
    expected[:53] += U@(U.T@delta[:53])
    np.testing.assert_allclose(a['stable_projection'], expected, rtol=1e-10, atol=1e-12)
    assert a['stable_projection'][-1] == old['theta'][-1]
    assert np.all(a['stable_projection'] >= a['lower']) and np.all(a['stable_projection'] <= a['upper'])
    projections[str(seed)] = dict(dimensions=U.shape[1],threshold=4.,
        original_step_norm=float(np.linalg.norm(delta)),
        projected_step_norm=float(np.linalg.norm(expected-old['theta'])))
    counts['orthogonal_projections_reconstructed'] += 1
    risk_records, by_model = [], {name: {} for name in report['checks']}
    for label in labels:
        replay = replays[int(label[-1])] if label.startswith('IID') else None
        for name, number, offset, key in [('source', 1024, 100, 'source_parents'), ('real', 4096, 200, 'real_rows')]:
            if label == 'FRESH':
                pool = roles[name+'_train_rows']
                generator = torch.Generator().manual_seed(p['fresh_seed']+p['fresh_head_stride']*seed+offset)
                expected_rows = pool[torch.randperm(len(pool), generator=generator).numpy()[:number]]
            elif replay is not None:
                expected_rows = replay[key]
            else:
                expected_rows = old[label+'_'+key]
            np.testing.assert_array_equal(a[label+'_'+key], expected_rows)
            assert np.isin(expected_rows, roles[name+'_train_rows']).all()
            counts['role_entries_replayed'] += len(expected_rows)
        expected_seed = p['fresh_seed']+p['fresh_head_stride']*seed+1000 if label == 'FRESH' else (replay['random_seed'] if replay is not None else old[label+'_random_seed'])
        assert a[label+'_random_seed'] == expected_seed
        counts['random_bank_seeds_verified'] += 1
        map_label = 'FIT0' if replay is not None else label
        assert report['projection_contexts'][map_label]['seed'] == p['native_projection_seeds'][map_label]
        for key in ('baseline_residual', 'baseline_native', 'initial_native', 'cholesky'):
            if label != 'FRESH':
                np.testing.assert_allclose(a[label+'_'+key], replay[key] if replay is not None else old[label+'_'+key], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(a[label+'_baseline_native'], report['baseline_native'][label])
        np.testing.assert_array_equal(a[label+'_initial_native'], report['initial_native'][label])
        risk_records.append((label, a[label+'_baseline_residual'], report['baseline_risks'][label]))
        for name in report['checks']:
            check = report['checks'][name][label]
            native = a[name+'_'+label+'_actual_native']
            risk_records.append((label, a[name+'_'+label+'_actual_residual'], check['risks']))
            np.testing.assert_array_equal(native, check['native'])
            ratio = (native/a[label+'_baseline_native']).reshape(3, 3)
            anchor = (native/a[label+'_initial_native']).reshape(3, 3)
            np.testing.assert_allclose(ratio, check['native_ratios'], rtol=1e-12)
            np.testing.assert_allclose(anchor, check['initial_anchor_ratios'], rtol=1e-12)
            if label != 'FRESH' and name == 'curvature_update_control':
                for key in ('residual', 'native'):
                    expected_value = replay['one_rank_actual_'+key] if replay is not None else control['curvature_update_'+label+'_actual_'+key]
                    np.testing.assert_allclose(a[name+'_'+label+'_actual_'+key], expected_value, rtol=1e-12, atol=1e-12)
                counts['control_candidates_replayed'] += 1
            base, risk, trust = report['baseline_risks'][label], check['risks'], check['trust']
            joint = np.mean([risk[h]['joint_whitened'] for h in hypotheses])/np.mean([base[h]['joint_whitened'] for h in hypotheses])
            macro = all(risk[h][block] <= 1.15*base[h][block]+2/4096 for h in hypotheses for block in blocks[:4])
            assert trust['denominator'] == 32768
            tr = trust['mean_relative_RMS'] <= .08 and trust['p99_relative_RMS'] <= .25 and max(v for hh in trust['flag_count_changes'].values() for v in hh.values()) <= max(4, .005*32768)
            ng = bool((ratio <= np.array([1.01, 1.01, 1.10])).all() and (anchor <= np.array([1.01, 1.01, 1.10])).all())
            jg = bool(joint <= .99 and macro and tr)
            assert tr == trust['pass'] and ng == check['native_pass'] and jg == check['original_joint_accept'] and (ng and jg) == check['accepted']
            by_model[name][label] = dict(accepted=check['accepted'], joint_ratio=float(joint), macro_pass=macro,
                trust_pass=tr, native_pass=ng, native_ratio_max=ratio.max(0).tolist(), initial_anchor_ratio_max=anchor.max(0).tolist())
            counts['acceptance_checks_recomputed'] += 1
    for label, vector, recorded in risk_records:
        L = a[label+'_cholesky']
        variance = np.square(L).sum(1)
        for ih, h in enumerate(hypotheses):
            white = vector[ih*272:(ih+1)*272]*np.sqrt(3)
            raw = L@white
            np.testing.assert_allclose(recorded[h]['joint_whitened'], np.mean(white**2), rtol=1e-11)
            for ib, block in enumerate(blocks):
                take = slice(56*ib, 56*(ib+1)) if ib < 4 else slice(224, 272)
                np.testing.assert_allclose(recorded[h][block], np.mean(raw[take]**2/variance[take]), rtol=1e-10, atol=1e-12)
            counts['risk_views_recomputed'] += 1
    all_pass = {name: all(v['accepted'] for v in rows.values()) for name, rows in by_model.items()}
    assert all_pass == report['all_local_gates'] == result['summaries'][str(seed)]
    summary[str(seed)] = dict(all_pass=all_pass, by_model=by_model)
for name, maximum in p['operation_budget'].items():
    assert result['execution_counts'].get(name, 0) == maximum, (name, result['execution_counts'].get(name, 0), maximum)
assert set(result['execution_counts']) <= set(p['operation_budget']) and result['native_timing']['calls'] == 72
passed = all(v['all_pass']['stable_projection'] for v in summary.values())
assert passed == result['two_seed_stable_projection_all_local_pass']
assert not result['scientific_acceptance'] and result['committed_parameter_updates'] == 0 and not result['automatic_training']
review = dict(status='PASS_SAVED_STABLE_PROJECTION_DIAGNOSTIC_AUDIT', UTC=datetime.now(timezone.utc).isoformat(),
    source_files_verified=902, source_pins=len(p['source_pins']), local_source_hashes=local_hashes,
    output_files_rehashed=len(m['outputs']), output_bytes=sum(v['bytes'] for v in m['outputs'].values()),
    native_exit=m['native_exit'], elapsed_seconds=result['elapsed_seconds'], config_sha256=result['config_sha256'],
    result_sha256=complete['result_sha256'], audit_counts=dict(counts), execution_counts=result['execution_counts'],
    projection_reconstruction=projections, summaries=summary, two_seed_stable_projection_all_local_pass=passed,
    scientific_acceptance=False, committed_parameter_updates=0,
    evidence_limit='Saved vectors, orthogonal projection construction, role IDs, paired controls and gate arithmetic independently audited. Complete candidate waveforms and trust extraction were not independently regenerated. FRESH uses historically exposed fitting records.')
(T/'TERMINAL_REVIEW.json').write_text(json.dumps(review, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status=review['status'], audit_counts=dict(counts), summary=summary), indent=2))
