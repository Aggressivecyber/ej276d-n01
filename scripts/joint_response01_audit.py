"""Bounded stage-1 audit of already saved native outputs and REAL TRAIN."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from n01.finalization.native_storage import load_native
from n01.joint01_observables import peak_code_features, aggregate_code_features

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--config-sha256', required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
assert hashlib.sha256(args.config.read_bytes()).hexdigest() == args.config_sha256
p = json.loads(args.config.read_text(encoding='utf-8'))
root = Path(__file__).resolve().parents[1]
for name, digest in p['source_pins'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
args.out.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
input_hashes = {}
for name, binding in p['inputs'].items():
    path = Path(binding['path'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.stat().st_size == binding['bytes'] and digest == binding['sha256'], name
    input_hashes[name] = digest
counts = dict(TRAIN_archives_decoded=0, TRAIN_members_decoded=0, saved_native_archives_decoded=0,
              saved_native_rows=0, REAL_rows=0, response_forwards=0, checkpoint_loads=0,
              new_random_draws=0, AD_queries=0, optimizer_updates=0, DEV_waveforms_decoded=0)
with np.load(p['inputs']['TRAIN']['path'], allow_pickle=False) as data:
    members = {name: data[name].copy() for name in p['TRAIN_members']}
counts['TRAIN_archives_decoded'] = 1
counts['TRAIN_members_decoded'] = len(members)
assert (members['real_role'] == 'TRAIN').all() and (members['source_role'] == 'TRAIN').all()
assert np.array_equal(members['original_source_weights'], np.ones(6593))
assert len(members['real']) == 21047 and len(members['source_role']) == 6593
real_adc, real_wave, real_prefix = [members[name] for name in ('real_carrier384', 'real', 'real_baseline100')]
real_peaks = real_adc[:, 112:144].argmax(1) + 112
assert np.array_equal(real_prefix, real_adc[:, :100])
real_columns, real_flags = peak_code_features(real_adc, real_wave, real_prefix, real_peaks)
result = dict(schema='n01/joint01-observable-audit/v1', real=aggregate_code_features(real_columns, real_flags),
              generated={}, source_nominal_count=6593, REAL_groups=int(len(np.unique(members['real_groups']))),
              group_scope='logical groups within one historical acquisition; not independent acquisitions')
counts['REAL_rows'] = len(real_wave)
real_codes = np.rint((real_adc + .5) * 16383.)
assert np.max(np.abs(real_adc - (real_codes / 16383. - .5))) < 1.e-15
result['REAL_grid_reconstruction_max_abs_V'] = float(np.max(np.abs(real_adc - (real_codes / 16383. - .5))))
baseline_constraints = {}
groups = members['real_groups']
for key in ['baseline_mean_V', 'baseline_RMS_V']:
    value = real_columns[key]
    total = len(value)
    residual_sums = np.array([(value[groups == g] - value.mean()).sum() for g in np.unique(groups)])
    G = len(residual_sums)
    clustered_se = np.sqrt(G / (G - 1) * np.sum((residual_sums / total) ** 2))
    baseline_constraints[key] = dict(REAL_mean=float(value.mean()), cluster_SE=float(clustered_se),
                                    cluster_scope='within-capture logical groups; not hardware calibration precision')
for name, binding in p['inputs'].items():
    if name == 'TRAIN':
        continue
    with np.load(binding['path'], allow_pickle=False) as saved:
        adc, wave, prefix = load_native(saved)
        peaks = saved['peak_index'].copy()
    assert len(wave) == 6593 and np.array_equal(adc[:, 112:144].argmax(1) + 112, peaks), name
    columns, flags = peak_code_features(adc, wave, prefix, peaks)
    summary = aggregate_code_features(columns, flags)
    names = list(columns)
    values = np.column_stack([columns[key] for key in names])
    centered = values - values.mean(0)
    summary['joint_feature_order'] = names
    summary['joint_covariance_population'] = (centered.T @ centered / len(values)).tolist()
    # Whole observed distributions, not source/REAL event pairing.
    summary['observable_scalar_W1_to_REAL'] = {}
    for key in ['top_two_code_gap_LSB', 'maximum_code_tied_samples', 'maximum_code_longest_plateau',
                'peak_curvature_LSB', 'baseline_mean_V', 'baseline_RMS_V']:
        from scipy.stats import wasserstein_distance
        summary['observable_scalar_W1_to_REAL'][key] = float(wasserstein_distance(columns[key], real_columns[key]))
    result['generated'][name] = summary
    counts['saved_native_archives_decoded'] += 1
    counts['saved_native_rows'] += len(wave)
    assert time.monotonic() - started < p['resources']['internal_seconds']
    (args.out / 'PROGRESS.json').write_text(json.dumps(dict(last_completed=name, counts=counts)) + '\n')

for key, info in baseline_constraints.items():
    references = [result['generated'][f'seed{seed}_M0_fast_NATIVE.npz']['features'][key]['mean'] for seed in (0, 1)]
    info['M0_saved_library_means'] = references
    info['M0_library_mean'] = float(np.mean(references))
    info['REAL_minus_M0'] = info['REAL_mean'] - info['M0_library_mean']
    info['between_two_saved_library_half_difference'] = .5 * abs(references[0] - references[1])
result['baseline_evidence'] = baseline_constraints
result['composition_scenarios'] = {}
for name, sign in [('nominal', 0.), ('low_h', -1.), ('high_h', 1.)]:
    raw = 1. + sign * .2 * (2 * members['h'] - 1)
    weight = raw / raw.sum()
    result['composition_scenarios'][name] = dict(raw_min=float(raw.min()), raw_max=float(raw.max()),
        parent_ESS=float(1. / (weight @ weight)), reference='026 existing fixed sensitivity scenarios',
        learned_population=False)
assert counts == p['operation_counts_exact'], (counts, p['operation_counts_exact'])
result.update(status='PASS_NEW_OBSERVABLE_AUDIT', UTC=datetime.now(timezone.utc).isoformat(),
              elapsed_seconds=time.monotonic() - started, counts=counts, scientific_acceptance=False,
              historical_S_decision_changed=False, input_SHA256=input_hashes)
target = args.out / 'RESULT.json'
target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
assert sum(f.stat().st_size for f in args.out.rglob('*') if f.is_file()) < p['resources']['output_MiB'] * 1024**2
(args.out / 'COMPLETE.json').write_text(json.dumps(dict(status='PASS', result_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), counts=counts)) + '\n')
print(json.dumps(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds'], counts=counts,
                     baseline_evidence=baseline_constraints), ensure_ascii=False))
