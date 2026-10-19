"""Independent NumPy/SciPy replay of saved hard-output native distances."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent
OUT = HERE / 'collected/run_01'
started = time.monotonic()
result = json.loads((OUT / 'RESULT.json').read_text())
metadata = json.loads((OUT / 'NATIVE_METRIC_STATE.json').read_text())
with np.load(OUT / 'NATIVE_METRIC_STATE.npz', allow_pickle=False) as archive:
    state = {key: archive[key].astype(np.float64) for key in archive.files}
names = ('voltage', 'shape', 'difference', 'curvature', 'difference_smooth_5', 'difference_smooth_17')


def standard(name, values):
    return (values - state['loss/' + name + '_center']) / state['loss/' + name + '_scale']


def w1(x, y):
    return float(np.abs(np.sort(x, axis=0) - np.sort(y, axis=0)).mean())


def wave_views(wave):
    shape = wave / np.maximum(wave.max(1, keepdims=True), state['loss/peak_floor'])
    views = dict(voltage=wave, shape=shape, difference=np.diff(shape, axis=1), curvature=np.diff(shape, n=2, axis=1))
    for size in (5, 17):
        padded = np.pad(shape, ((0, 0), (size // 2, size // 2)), mode='edge')
        smooth = sum(padded[:, i:i + 256] for i in range(size)) / size
        views['difference_smooth_' + str(size)] = np.diff(smooth, axis=1)
    return {name: standard(name, value) for name, value in views.items()}


def qpsd(wave):
    charge, tail = wave.sum(1), wave[:, 46:].sum(1)
    floor, reference = state['loss/q_floor'], state['loss/q_reference']
    logq = np.where(charge >= floor, np.log(np.maximum(charge, floor) / reference),
        np.log(floor / reference) + np.arcsinh((charge - floor) / floor))
    raw = np.stack((logq, tail / (np.where(charge < 0, -1., 1.) * np.maximum(abs(charge), floor))), 1)
    return raw, standard('qpsd', raw)


def noise_features(baseline):
    mean = baseline.mean(1, keepdims=True)
    centered = baseline - mean
    t = np.linspace(-1, 1, 100)
    slope = (centered * t).mean(1, keepdims=True) / np.square(t).mean()
    floor = (1. / (2**14 - 1))**2 / 12.
    variance = np.maximum(np.square(centered).mean(1, keepdims=True), floor)
    acf = np.concatenate([(centered[:, :-lag] * centered[:, lag:]).mean(1, keepdims=True) / variance
        for lag in range(1, 17)], 1)
    quantiles = np.quantile(centered, [.01, .50, .99], axis=1).T
    power = np.abs(np.fft.rfft(centered, axis=1))**2
    total = np.maximum(power[:, 1:].sum(1, keepdims=True), 100 * floor)
    bands = np.concatenate([power[:, lo:hi].sum(1, keepdims=True) / total
        for lo, hi in ((1, 3), (3, 6), (6, 11), (11, 21), (21, 51))], 1)
    local = centered.reshape(-1, 5, 20)
    local_log = np.log(np.maximum(np.square(local - local.mean(2, keepdims=True)).mean(2), floor))
    return standard('noise', np.concatenate((mean, slope, .5 * np.log(variance), acf, quantiles, bands, local_log), 1))


checks, differences, decompositions = [], [], []
for seed in (0, 1):
    full = np.empty((3, 3, 3, 3, 3))
    for bi, bank in enumerate(('FIT0', 'FIT1', 'FIT2')):
        with np.load(OUT / f'seed{seed}_{bank}.npz', allow_pickle=False) as archive:
            target = archive['target_window_V']
            target_views = wave_views(target)
            target_raw, target_qpsd = qpsd(target)
            target_noise = noise_features(archive['target_baseline100'])
            target_bins = np.searchsorted(state['loss/charge_bin_boundaries'], target_raw[:, 0], side='left')
            for pi, point in enumerate(('M0', 'trial0', 'trial1')):
                saved_native = archive[point + '_native_cross']
                for hi, hypothesis in enumerate(metadata['hypothesis_order']):
                    wave = archive[point + f'_h{hi}_window_V']
                    views = wave_views(wave)
                    raw, transformed = qpsd(wave)
                    noise = noise_features(archive[point + f'_h{hi}_baseline100'])
                    bins = np.searchsorted(state['loss/charge_bin_boundaries'], raw[:, 0], side='left')
                    conditional = []
                    for bin_index in range(8):
                        fm, rm = bins == bin_index, target_bins == bin_index
                        if min(fm.sum(), rm.sum()) >= metadata['minimum_support']:
                            conditional.append(wasserstein_distance(transformed[fm, 1], target_qpsd[rm, 1]))
                    coordinate = np.mean([w1(views[name], target_views[name]) for name in ('voltage', 'shape')])
                    noise_coordinate = w1(noise, target_noise)
                    qpsd_coordinate = w1(transformed, target_qpsd)
                    for ci, context in enumerate(('FIT0', 'FIT1', 'FIT2')):
                        swd = np.mean([w1(views[name] @ state[context + '/wave:' + name],
                            target_views[name] @ state[context + '/wave:' + name]) for name in names])
                        lag_terms = []
                        for name in ('voltage', 'shape'):
                            for lag in (1, 2, 4, 8):
                                directions = state[context + f'/lag:{name}:{lag}']
                                a = (views[name][:, :-lag, None] * directions[0] + views[name][:, lag:, None] * directions[1]).reshape(len(wave), -1)
                                b = (target_views[name][:, :-lag, None] * directions[0] + target_views[name][:, lag:, None] * directions[1]).reshape(len(target), -1)
                                lag_terms.append(w1(a, b))
                        wave_metric = .5 * swd + .25 * coordinate + .25 * np.mean(lag_terms)
                        directions = state[context + '/qpsd']
                        global_qpsd = .5 * (qpsd_coordinate + w1(transformed @ directions, target_qpsd @ directions))
                        qpsd_metric = .5 * (global_qpsd + np.mean(conditional)) if conditional else global_qpsd
                        directions = state[context + '/noise']
                        noise_metric = .5 * (noise_coordinate + w1(noise @ directions, target_noise @ directions))
                        actual = np.array([wave_metric, qpsd_metric, noise_metric])
                        difference = abs(actual - saved_native[ci, hi])
                        full[pi, bi, ci, hi] = actual
                        differences.append(difference)
                        checks.append(dict(seed=seed, bank=bank, point=point, hypothesis=hypothesis,
                            context=context, values=actual.tolist(), max_abs_difference=float(difference.max())))
                        if difference.max() > 1e-10:
                            raise ValueError('Independent hard metric replay exceeds 1e-10: ' + str(checks[-1]))
                        if time.monotonic() - started > 300:
                            raise TimeoutError('Saved-output independent replay 300-second budget exceeded')
        (HERE / 'AUDIT_PROGRESS.json').write_text(json.dumps(dict(seed=seed, bank=bank, vector_checks=len(checks),
            elapsed_seconds=time.monotonic() - started), indent=2) + '\n')
    np.testing.assert_allclose(full, result['seeds'][str(seed)]['native_grid'], rtol=0, atol=1e-10)
    ratios = full[1:] / full[0]
    for item in result['seeds'][str(seed)]['contrasts']:
        pi = ('trial0', 'trial1').index(item['point'])
        hi = metadata['hypothesis_order'].index(item['hypothesis'])
        mi = metadata['metric_order'].index(item['metric'])
        values = ratios[pi, :, :, hi, mi]
        np.testing.assert_allclose(values, item['ratios'], rtol=0, atol=1e-10)
        saved = np.array(item['ratios'])
        grand = saved.mean()
        be, pe = saved.mean(1) - grand, saved.mean(0) - grand
        interaction = saved - grand - be[:, None] - pe[None, :]
        ss = np.array([3 * np.square(be).sum(), 3 * np.square(pe).sum(), np.square(interaction).sum()])
        np.testing.assert_allclose(ss, item['sum_squares_bank_projection_interaction'], rtol=0, atol=1e-18)
        assert abs(ss.sum() - np.square(saved - grand).sum()) < 1e-17
        decompositions.append(dict(seed=seed, point=item['point'], hypothesis=item['hypothesis'], metric=item['metric'],
            sum_squares_identity_residual=float(ss.sum() - np.square(saved - grand).sum())))
assert len(checks) == 162 and len(decompositions) == 36
receipt = dict(status='PASS_INDEPENDENT_NUMPY_SCIPY_HARD_OUTPUT_REPLAY', UTC=datetime.now(timezone.utc).isoformat(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), result_sha256=hashlib.sha256((OUT / 'RESULT.json').read_bytes()).hexdigest(),
    vector_checks=162, scalar_metric_checks=486, fixed_factor_decompositions=36,
    maximum_absolute_difference=float(np.max(differences)),
    maximum_absolute_difference_by_wave_QPSD_noise=np.max(differences, axis=0).tolist(),
    checks=checks, decompositions=decompositions, elapsed_seconds=time.monotonic() - started,
    generator_calls=0, parameter_updates=0, boundary='Saved hard outputs replayed independently; original generators and raw RNG were not rerun by this CPU audit.')
(HERE / 'HARD_OUTPUT_AUDIT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key: value for key, value in receipt.items() if key not in ('checks', 'decompositions')}))
