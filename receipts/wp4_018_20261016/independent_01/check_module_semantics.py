"""Independent deterministic 018 semantic examples; no private arrays or random draws."""
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np

root = Path('E:/GAN/research/N01')
sys.path.insert(0, str(root/'src'))
package = types.ModuleType('n01')
package.__path__ = [str(root/'src/n01')]
sys.modules['n01'] = package
from n01.saved_array_kernels017 import morphology
from n01.stats017 import REQUIRED_MISSING017, covariance_cell017, scalar_banks017
from n01.stats018 import (covariance8_cell018, energy_discrepancy018, morphology018,
                         observable018, observable_cell018, pair_distances018,
                         scalar_features018, self_mean018, stage_gate018)

index = np.arange(256, dtype=np.float64)
triangle = np.maximum(1.-np.maximum(index-16., 0.)/64., 0.)
triangle[:17] = index[:17]/16.
waves = np.tile(triangle, (9, 1))
waves[1, :17] = .2+.8*index[:17]/16.
waves[2, :17] = .95+.05*index[:17]/16.
waves[3] = 0.
waves[4] = -1.
waves[5, 8:10] = (.95, .05)
waves[6, 4] = 1.2
waves[7, :17] = .1+.9*index[:17]/16.
waves[8, 17:] = .7
old, new = morphology(waves), morphology018(waves)
assert old.keys() == new.keys()
for key in old:
    assert np.array_equal(old[key], new[key], equal_nan=True), key
scalar, new = scalar_features018(waves, np.zeros((len(waves), 100)), np.full(len(waves), 128))
observed = observable018(waves, scalar, new,
    dict(Q_Vns=1., peak_voltage_V=1., FWHM_ns=1.))
assert np.array_equal(observed['state'], (0, 1, 1, 2, 2, 0, 0, 0, 0))
assert observed['conditional_bound_defined'].tolist() == [False, True]+[False]*7
assert abs(observed['timing_values_ns'][1, 5]-14.) < 1e-14
assert abs(observed['timing_values_ns'][1, 3]+2.) < 1e-14
assert observed['timing_values_ns'][7, 1] == -16.
assert observed['embedding'][8, -1] == 0.
assert observed['embedding'][8, -2] == 0.
assert np.all(observed['embedding'][:, 9:15][~observed['timing_defined']] == 0.)
assert np.all(observed['embedding'][:, :3].sum(1) == 1.)

draws = np.asarray((-1., 0., 1., 2.))[None, :, None]
self_u = self_mean018(draws)
identity_u = energy_discrepancy018(draws, draws)
assert np.allclose(identity_u, -self_u/4., rtol=0, atol=1e-15)
all_pairs = pair_distances018(draws, draws).mean((1, 2))
identity_v = all_pairs-.5*all_pairs-.5*all_pairs
assert np.array_equal(identity_v, (0.,))
assert np.array_equal(energy_discrepancy018(np.zeros((1, 4, 1)), np.full((1, 4, 1), 3.)), (3.,))

group, weights = np.ones(64, bool), np.ones(64)
assert observable_cell018(np.zeros(64), np.zeros(64), group, weights)['status'] == 'pass'
assert observable_cell018(np.full(64, .11), np.zeros(64), group, weights)['status'] == 'fail'
assert observable_cell018(np.arange(64)/63.-.5, np.zeros(64), group, weights)['reasons'] == ['parent_cluster_MC_half_width_above005']
assert observable_cell018(.09+np.tile((-.05, .05), 32), np.zeros(64), group, weights)['reasons'] == ['interval_intersects_tolerance']
assert observable_cell018(np.full(64, 2.), np.full(64, .06), group, weights)['reasons'] == ['teacher32_null_unresolved']

scalar_example = np.broadcast_to(np.arange(32)[None, None, :, None], (64, 2, 32, 9)).astype(float).copy()
scalar_example[:, :, :, 6] = np.nan
moments = scalar_banks017(scalar_example)
cov8 = covariance8_cell018(scalar_example, scalar_example, group, weights, moments=(moments, moments))
cov9 = covariance_cell017(scalar_example, scalar_example, group, weights, moments=(moments, moments))
assert cov8['status'] == 'pass' and cov8['supported_parents'] == 64
assert cov9['status'] == 'pending' and cov9['supported_parents'] == 0

cells = []
for g in ('all_heldout', 'fit_Q_tertile0_heldout', 'fit_Q_tertile1_heldout', 'fit_Q_tertile2_heldout'):
    for w in ('nominal', 'low', 'high'):
        cells.append(dict(metric='analytic_charge_variance', group=g, weights=w, status='pass', reasons=[]))
        for h in ('fast', 'slow', 'common_level'):
            for metric in ('hard_wave_mean', 'hard_wave_variance', 'conditional_cov8', 'conditional_energy',
                           'time_peak_index', 'time_FWHM', 'observable_joint_energy'):
                cells.append(dict(metric=metric, group=g, weights=w, hypothesis=h, status='pass', reasons=[]))
            for feature in ('Q', 'tail', 'peak_voltage', 'peak_index', 'FWHM'):
                for level in (.5, .9):
                    cells.append(dict(metric='PIT_coverage', group=g, weights=w, hypothesis=h,
                                      feature=feature, level=level, status='pass', reasons=[]))
            for feature in REQUIRED_MISSING017:
                cells.append(dict(metric='missingness_or_censor', group=g, weights=w, hypothesis=h,
                                  feature=feature, status='pass', reasons=[]))
assert len(cells) == 1272
trials = [dict(trial=name, cells=[c.copy() for c in cells]) for name in ('T1_paired_seed0', 'T1_paired_seed1')]
assert stage_gate018(trials)['T1_paired_018_qualified']
trials[0]['cells'][0]['status'] = 'pending'
assert not stage_gate018(trials)['T1_paired_018_qualified']
trials[0]['cells'] = [c.copy() for c in cells[:-1]]
try:
    stage_gate018(trials)
except ValueError as error:
    assert 'missing/duplicate primary signature' in str(error)
else:
    raise AssertionError('missing cell accepted')
trials[0]['cells'].append(cells[0].copy())
try:
    stage_gate018(trials)
except ValueError as error:
    assert 'missing/duplicate primary signature' in str(error)
else:
    raise AssertionError('duplicate replacement accepted')

report = dict(status='pass', scope='9 deterministic morphology/LB/mask rows and analytic U/V examples',
    morphology_equivalent_to017=True, conditional_bound_crop_ns=float(observed['timing_values_ns'][1, 5]),
    relative_t90_ns=float(observed['timing_values_ns'][1, 3]), identity_U=identity_u.tolist(),
    identity_V=identity_v.tolist(), threshold_branch_examples=5, cov8_full_support_with_rise_missing=True,
    cov9_original_rule_preserved=True, exact2544_stage_schema_checked=True, missing_duplicate_rejected=True,
    pending_closes_qualification=True, new_RNG=0, private_array_reads=0, NN=0, AD=0, G=0,
    source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (root/'src/n01/stats018.py', root/'src/n01/saved_array_kernels017.py')})
output = Path(__file__).with_name('INDEPENDENT_MODULE_SEMANTICS.json')
output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
