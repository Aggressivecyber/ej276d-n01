"""Production numerical semantics on synthetic events; no data/model loads."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import osqp
import scipy
import torch

from n01.finalization.smooth_mean import constraint_space, fit_mean, SOLVER_SETTINGS
from n01.finalization.linear_random_transport import raw_warp_delta, project_random
from n01.finalization.native_storage import save_native
from n01.observation008 import observe_generated_carrier
from run_response069 import level_warp069


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(2026101902)
    grid = np.arange(384)
    cases = []
    for peak in (112, 126, 143):
        mean = .16*np.where(grid <= peak, np.exp(-((grid-peak)/5.)**2),
                            np.exp(-(grid-peak)/20.))
        cases.append((f'peak_{peak}', mean))
    base = cases[1][1]
    for name in ('plateau', 'two_crests', 'negative_support', 'identity', 'zero_dof',
                 'active_peak_constraint', 'active_positive_constraint'):
        mean = base.copy()
        if name == 'plateau': mean[127] = mean[126]
        if name == 'two_crests': mean[130] = .999*mean[126]
        if name == 'negative_support': mean[[114, 115, 158, 159]] = -.001
        if name == 'zero_dof': mean[:] = 0.; mean[126] = .16; mean[150] = .01
        cases.append((name, mean))
    # Profiles are constructed here, never decoded from SOURCE or REAL.
    t = np.arange(256)-16
    sp = np.where(t <= 0, np.exp(-(t/.9)**2), np.exp(-t/4.5))
    tp = np.where(t <= 0, np.exp(-(t/.7)**2), np.exp(-t/4.))
    centers = np.geomspace(.1, 30., 8)
    template = dict(source_Q_centers_Vns=centers.tolist(), target_Q_centers_Vns=centers.tolist(),
                    source_unit_Q_profiles=np.tile(sp/sp.sum(), (8, 1)).tolist(),
                    target_unit_Q_profiles=np.tile(tp/tp.sum(), (8, 1)).tolist())
    means = np.stack([m for _, m in cases])
    random = .001*rng.normal(size=means.shape)
    tm = torch.as_tensor(means, device=args.device)
    tr = torch.as_tensor(random, device=args.device)
    q = torch.as_tensor(np.linspace(.5, 10., len(cases)), device=args.device)
    old, _, rows, queries, _, _ = level_warp069(tm, tr, q, template)
    old, queries = old.cpu().numpy(), queries.cpu().numpy()
    order = np.arange(len(cases))[::-1].copy()
    permuted, _, _, permuted_queries, _, _ = level_warp069(tm[order],tr[order],q[order],template)
    permutation_error = max(float(np.max(np.abs(permuted.cpu().numpy()[order]-old))),
                            float(np.max(np.abs(permuted_queries.cpu().numpy()[order]-queries))))
    chunked = []
    for start in range(0,len(cases),3):
        chunked.append(level_warp069(tm[start:start+3],tr[start:start+3],q[start:start+3],template)[0].cpu().numpy())
    chunk_error = float(np.max(np.abs(np.concatenate(chunked)-old)))
    assert max(permutation_error,chunk_error) < 1e-12
    scales = (.15554293597289215, 3.66896364775571, .6947215402546918)
    reports, deltas, random_deltas = [], [], []
    for i, (name, mean) in enumerate(cases):
        space = constraint_space(mean, scales)
        target = np.zeros(66) if name == 'identity' else old[i, 110:176]
        if name == 'active_peak_constraint': target = np.zeros(66); target[20:24] = .4
        if name == 'active_positive_constraint': target = np.zeros(66); target[40:55] = -.4
        delta, coefficient, info = fit_mean(mean, target, space)
        raw = raw_warp_delta(random[i, 110:176], queries[i], float(rows['amplitude_scale'][i]), space['allowed'])
        raw_mean = raw_warp_delta(mean[110:176],queries[i],float(rows['amplitude_scale'][i]),space['allowed'])
        assert np.max(np.abs(raw_mean-old[i,110:176])) < 1e-12
        projected = project_random(raw, space)
        adapted = mean.copy(); adapted[110:176] += delta
        a, b = rng.normal(size=(2, 66))*.01
        ra = raw_warp_delta(a, queries[i], float(rows['amplitude_scale'][i]), space['allowed'])
        rb = raw_warp_delta(b, queries[i], float(rows['amplitude_scale'][i]), space['allowed'])
        rab = raw_warp_delta(a+b, queries[i], float(rows['amplitude_scale'][i]), space['allowed'])
        linearity = np.max(np.abs(project_random(rab, space)-project_random(ra, space)-project_random(rb, space)))
        eye = np.eye(66)
        operator = np.column_stack([project_random(raw_warp_delta(row, queries[i],
            float(rows['amplitude_scale'][i]), space['allowed']), space) for row in eye])
        transport = eye+operator
        factor = rng.normal(size=(66, 6))*.001
        covariance = np.max(np.abs((transport@factor)@(transport@factor).T-
                                  transport@(factor@factor.T)@transport.T))
        paired = np.column_stack((factor, -factor))
        assert np.max(np.abs((transport@paired).mean(1))) < 1e-12
        eq = max(abs(delta[space['front']].sum()), abs(delta[space['tail']].sum()),
                 abs(projected[space['front']].sum()), abs(projected[space['tail']].sum()))
        assert eq < 1e-12 and linearity < 1e-12 and covariance < 1e-12
        assert np.all(delta[space['structural']] == 0.)
        assert np.all(projected[space['structural']] == 0.)
        assert np.array_equal(adapted[:110], mean[:110])
        assert np.array_equal(adapted[176:], mean[176:])
        assert np.array_equal(adapted[mean <= 0.], mean[mean <= 0.])
        assert adapted[112:144].argmax() == mean[112:144].argmax()
        assert np.all(adapted[110:176][space['allowed']] >= 0.)
        if name == 'identity': assert np.array_equal(adapted, mean)
        # Repeating an event independently cannot depend on previous active sets.
        again, _, _ = fit_mean(mean, target, constraint_space(mean, scales))
        assert np.array_equal(delta, again)
        reports.append(dict(name=name, **info, equality_residual_Vns=float(eq),
                            linearity_error_V=float(linearity), covariance_error=float(covariance)))
        deltas.append(delta); random_deltas.append(projected)
    full_delta = np.zeros_like(means); full_delta[:, 110:176] = deltas
    full_random = np.zeros_like(means); full_random[:, 110:176] = random_deltas
    carrier = means+full_delta+random+full_random
    # One explicit noisy competitor proves the readout remains free to reselect.
    carrier[1, 130] = carrier[1, 126]+.1
    obs = observe_generated_carrier(torch.as_tensor(carrier, device=args.device),
        ste=False, q_floor=.01, return_carrier=True, return_indices=True)
    adc = obs['carrier_adc'].cpu().numpy()
    baseline = adc[:, :100].mean(1)
    centered = adc-baseline[:, None]
    peak = centered[:, 112:144].argmax(1)+112
    wave = np.take_along_axis(centered, peak[:, None]-16+np.arange(256), axis=1)
    assert np.array_equal(peak, obs['peak_index'].cpu().numpy())
    assert peak[1] == 130
    error = float(np.max(np.abs(wave-obs['window_V'].cpu().numpy())))
    assert error < 1e-12
    storage = save_native(args.out/'SYNTHETIC_NATIVE.npz', obs)
    np.savez_compressed(args.out/'SYNTHETIC_ARRAYS.npz', mean=means, correction=full_delta,
                        random_correction=full_random, carrier=carrier, adc=adc,
                        baseline=obs['baseline100'].cpu().numpy(), wave=obs['window_V'].cpu().numpy(),
                        peak_index=peak, queries=queries)
    result = dict(status='PASS', UTC=datetime.now(timezone.utc).isoformat(), device=args.device,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        torch=torch.__version__, osqp=osqp.__version__, solver_settings=SOLVER_SETTINGS,
        cases=reports, native_numpy_error_V=error, noisy_peak_reselected=True,
        frozen069_permutation_error=permutation_error,frozen069_chunk_error=chunk_error,
        checkpoint_loads=0, SOURCE_REAL_archive_decodes=0, model_forwards=0,
        scientific_candidate_evaluations=0, native_storage=storage,
        arrays_sha256=hashlib.sha256((args.out/'SYNTHETIC_ARRAYS.npz').read_bytes()).hexdigest())
    (args.out/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
