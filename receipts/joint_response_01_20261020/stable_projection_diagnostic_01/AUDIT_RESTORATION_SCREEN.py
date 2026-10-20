"""Independently reconstruct the saved stale-derivative screen; no new solves."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
DESIGN = json.loads((HERE/'RECENTERED_RESTORATION_SCREEN_DESIGN.json').read_bytes())
SCREEN = json.loads((HERE/'RECENTERED_RESTORATION_SCREEN.json').read_bytes())
BASE = HERE.parent/'original_precision_REAL_pilot_01/terminal_01/collected/run_01'
RANDOM = HERE.parent/'random_precision_diagnostic_01/terminal_01/collected/run_01'
reports = []
for record in SCREEN['reports']:
    seed = record['seed']
    for name, digest in record['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    payload = Path(record['evidence_NAS_path'])
    remote = '/mnt/MiNas/GAN/research/N01/receipts/joint_response_01_20261020/stable_projection_diagnostic_01/'+payload.name
    request = 'from pathlib import Path\nimport sys\nsys.stdout.buffer.write(Path('+repr(remote)+').read_bytes())\n'
    transfer = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        'rylan@100.82.231.102', 'python3 -'], input=request.encode(), capture_output=True, timeout=40)
    transfer.check_returncode()
    assert hashlib.sha256(transfer.stdout).hexdigest() == record['evidence_sha256']
    with np.load(io.BytesIO(transfer.stdout), allow_pickle=False) as z:
        saved = dict(z)
    with np.load(BASE/f'seed{seed}_joint_ITER02.npz', allow_pickle=False) as z:
        old = dict(z)
    with np.load(HERE/f'terminal_01/collected/run_01/seed{seed}_STABLE_PROJECTION.npz', allow_pickle=False) as z:
        measured = dict(z)
    radius = DESIGN['radius']
    center = measured['curvature_update_control']
    np.testing.assert_array_equal(saved['center'], center)
    u = saved['u']
    step = radius*u
    A = saved['A']
    b = saved['b']
    expected_A = [np.r_[np.eye(53), -np.eye(53)]]
    expected_b = [np.r_[old['upper'][:53]-center[:53], center[:53]-old['lower'][:53]]/radius]
    native_A = []
    native_b = []
    sizes = [106]
    max_joint = 0.
    max_macro = 0.
    max_native = 0.
    for label in DESIGN['libraries']:
        if label.startswith('IID'):
            with np.load(RANDOM/f'seed{seed}_rep{label[-1]}_IID.npz', allow_pickle=False) as z:
                J = z['jacobian'][:, :53]
                N = z['native_jacobian'][:, :53]
        else:
            i = int(label[-1])
            J = old['original_jacobian'][i*816:(i+1)*816, :53]*np.sqrt(3)
            N = old['original_native_jacobian'][i*9:(i+1)*9, :53]
        r0 = measured[label+'_baseline_residual']
        r = measured['curvature_update_control_'+label+'_actual_residual']
        n0 = measured[label+'_baseline_native']
        anchor = measured[label+'_initial_native']
        n = measured['curvature_update_control_'+label+'_actual_native']
        predicted = r+J@step
        joint_limit = DESIGN['joint_cap']*(r0@r0)
        denom = np.sqrt(joint_limit)
        expected_A.append(np.vstack((np.zeros((1, 53)), -radius*J/denom)))
        expected_b.append(np.r_[1., r/denom])
        sizes.append(817)
        max_joint = max(max_joint, float(predicted@predicted/joint_limit))
        L = measured[label+'_cholesky']
        scales = np.square(L).sum(axis=1)
        for h in range(3):
            section = slice(272*h, 272*(h+1))
            old_raw = np.sqrt(3)*(L@r0[section])
            new_raw = np.sqrt(3)*(L@r[section])
            derivative_raw = np.sqrt(3)*(L@J[section])
            for k in range(4):
                rows = slice(56*k, 56*(k+1))
                cap = DESIGN['macro_cap']*np.mean(old_raw[rows]**2/scales[rows])+DESIGN['macro_sampling_allowance']
                scale = np.sqrt(56*cap*scales[rows])
                expected_A.append(np.vstack((np.zeros((1, 53)), -radius*derivative_raw[rows]/scale[:, None])))
                expected_b.append(np.r_[1., new_raw[rows]/scale])
                sizes.append(57)
                max_macro = max(max_macro, float(np.mean((new_raw[rows]+derivative_raw[rows]@step)**2/scales[rows])/cap))
        keep = [0, 1, 3, 4, 6, 7]
        cap = DESIGN['native_restoration_factor']*np.minimum(n0, anchor)
        native_A.append(radius*N[keep]/n0[keep, None])
        native_b.append((cap[keep]-n[keep])/n0[keep])
        max_native = max(max_native, float(np.max((n+N@step)[keep]/cap[keep])))
    expected_A.extend([np.vstack(native_A), np.vstack((np.zeros((1, 53)), -np.eye(53)))])
    expected_b.extend([np.concatenate(native_b), np.r_[1., np.zeros(53)]])
    sizes.extend([42, 54])
    np.testing.assert_allclose(A, np.vstack(expected_A), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b, np.concatenate(expected_b), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(saved['cone_sizes'], sizes)
    slack = b-A@u
    dual = saved['dual']
    primal_margins = []
    dual_margins = []
    cursor = 0
    for i, size in enumerate(sizes):
        s, z = slack[cursor:cursor+size], dual[cursor:cursor+size]
        cursor += size
        linear = i in (0, len(sizes)-2)
        primal_margins.append(float(s.min() if linear else s[0]-np.linalg.norm(s[1:])))
        dual_margins.append(float(z.min() if linear else z[0]-np.linalg.norm(z[1:])))
    stationarity = float(np.linalg.norm(u+A.T@dual, ord=np.inf))
    complementarity = float(abs(slack@dual))
    assert cursor == len(b) == 10709
    assert min(primal_margins) >= -1e-8 and min(dual_margins) >= -1e-8
    assert stationarity < 1e-8 and complementarity < 1e-8
    assert max(max_joint, max_macro, max_native) <= 1+1e-8
    assert np.linalg.norm(step) <= radius+1e-10 and bool(saved['stale_derivative_screen_only'])
    assert record['KKT_pass'] and record['solver_status'] == 'Solved'
    np.testing.assert_allclose(record['correction_norm'], np.linalg.norm(step), rtol=1e-12)
    reports.append(dict(seed=seed, reconstructed_rows=cursor, libraries=7, joint_constraints=7,
        macro_constraints=84, native_linear_constraints=42,
        minimum_primal_margin=min(primal_margins), minimum_dual_margin=min(dual_margins),
        stationarity=stationarity, complementarity=complementarity,
        maximum_joint_fraction_of_cap=max_joint, maximum_macro_fraction_of_cap=max_macro,
        maximum_native_fraction_of_cap=max_native, correction_norm=float(np.linalg.norm(step)),
        payload_sha256=record['evidence_sha256']))
result = dict(status='PASS_INDEPENDENT_SAVED_RESTORATION_SCREEN_AUDIT',
    UTC=datetime.now(timezone.utc).isoformat(), reports=reports,
    new_solves=0, new_generator_calls=0, new_candidates=0, parameter_updates=0,
    scientific_acceptance=False,
    NAS_access='Read exact registered payloads through server SSH into memory; Windows Z mapping absent. No local payload files created.',
    evidence_limit='Only the saved stale-derivative local convex model is certified; no fresh derivative or physical candidate acceptance.')
(HERE/'RECENTERED_RESTORATION_SCREEN_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result))
