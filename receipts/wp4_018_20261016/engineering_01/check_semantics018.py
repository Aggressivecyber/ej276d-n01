"""Only the new observable semantics and four fixed feature-space controls. No private arrays."""
import hashlib
import json
from pathlib import Path
import sys
import types
import faulthandler

faulthandler.dump_traceback_later(20)
print('018 artificial check: importing NumPy', flush=True)

import numpy as np

root = Path('E:/GAN/research/N01')
package = types.ModuleType('n01')
package.__path__ = [str(root/'src'/'n01')]
sys.modules['n01'] = package
from n01.saved_array_kernels017 import morphology
from n01.stats017 import scalar_banks017, covariance_cell017
from n01.stats018 import (DISTANCE_COUNTS018, covariance8_cell018, energy_discrepancy018,
    morphology018, observable018, observable_cell018, scalar_features018, self_mean018)
print('018 artificial check: numerical modules imported', flush=True)

assert 'torch' not in sys.modules
scales = json.loads((root/'wp4_018_config.json').read_text(encoding='utf-8'))['scales']['fast']
x = np.arange(256, dtype=np.float64)
waves = [np.exp(-.5*((x-16)/sigma)**2) for sigma in (4., 10., 45.)]
waves += [-np.ones(256), np.zeros(256)]
nonmonotonic = np.zeros(256)
nonmonotonic[5:17] = [.2, .8, .05, .3, .6, .02, .15, .1, .1, .4, .9, 1.]
nonmonotonic[17:27] = [.9, .7, .4, .6, .3, .2, .1, .1, .02, 0.]
waves.append(nonmonotonic)
waves.append(np.ones(256))
post_censored = np.interp(x, (0., 16., 17., 255.), (0., 1., .8, .8))
waves.append(post_censored)
plateau = np.interp(x, (0., 4., 5., 6., 7., 12., 16., 24., 255.), (0., .1, .1, .1, .2, .5, 1., 0., 0.))
waves.append(plateau)
waves = np.stack(waves)
old, new = morphology(waves), morphology018(waves)
assert old.keys() == new.keys()
for key in old:
    assert np.array_equal(new[key], old[key], equal_nan=True)
print('018 artificial check: crossings exact', flush=True)
scalar, morph = scalar_features018(waves, np.zeros((len(waves), 100)), np.full(len(waves), 128))
observed = observable018(waves, scalar, morph, scales)
assert observed['state'].tolist() == [0, 1, 1, 2, 2, 0, 1, 0, 0]
assert observed['conditional_bound_defined'][1]
assert not observed['conditional_bound_defined'][[0, 2, 3, 4, 6]].any()
assert observed['timing_values_ns'][1, 5] == new['left_crossing90_ns'][1]
assert observed['embedding'][7, 18] == 0
assert np.array_equal(observed['embedding'][:, 9:15],
    np.where(observed['timing_defined'], observed['timing_values_ns'], 0.))
assert np.isnan(observed['timing_values_ns'][~observed['timing_defined']]).all()

# Original rise support must not block the separate eight-dimensional covariance.
repeat = np.sin(np.arange(32, dtype=np.float64))
cov_scalar = np.broadcast_to(repeat[None, None, :, None]+np.arange(9), (128, 2, 32, 9)).copy()
cov_scalar[:, :, :, 6] = np.nan
moments = scalar_banks017(cov_scalar)
group, weights = np.ones(128, bool), np.ones(128)
cov9 = covariance_cell017(cov_scalar, cov_scalar, group, weights, moments=(moments, moments))
cov8 = covariance8_cell018(cov_scalar, cov_scalar, group, weights, moments=(moments, moments))
assert cov9['status'] == 'pending' and cov8['status'] == 'pass'
assert cov8['supported_parents'] == cov8['original_parents'] == 128
assert cov8['group_relative_Frobenius'] == cov8['teacher32_split_resolution'] == 0
print('018 artificial check: cov8 isolation exact', flush=True)

# Construct coherent observable tuples. These are feature-space distributions,
# not a call to the private hard observation/model pipeline or a hardware test.
z = np.zeros((128, 64, 19))
templates = np.asarray([[8., -12., -8., -4., 6., np.nan],
                        [np.nan, np.nan, -14., -10., 6., 6.],
                        [np.nan, np.nan, -6., -2., 6., 14.]])
row_type = np.r_[np.zeros(32, np.int64), np.ones(16, np.int64), np.full(16, 2, np.int64)]
state = (row_type > 0).astype(np.int64)
timing = templates[row_type]
defined = np.isfinite(timing)
z[:, :, :3] = np.eye(3)[state]
z[:, :, 3:9] = defined
z[:, :, 9:15] = np.where(defined, timing, 0.)
z[:, :, 15] = np.where(row_type == 2, 22., 2.)
z[:, :, 16] = 4.
z[:, :, 17] = (timing[:, 4]-timing[:, 2])/scales['FWHM_ns']
z[:, :, 18] = 1.
assert np.allclose(z[:, :, 17]*scales['FWHM_ns'], timing[None, :, 4]-timing[None, :, 2],
                   rtol=5e-12, atol=5e-14)
self_u = self_mean018(z)
identity = energy_discrepancy018(z, z, student_self=self_u, teacher_self=self_u)
assert np.allclose(identity, -self_u/64, rtol=5e-12, atol=5e-14)
assert (identity < 0).all()

controls = []
for name in ('censor_rate_plus025', 'censored_Q_visible_time_coupling',
             'measurable_rise_plus4ns', 'state_Q_dependence'):
    bad = z.copy()
    if name == 'censor_rate_plus025':
        bad[:, :16, 0] = 0.
        bad[:, :16, 1] = 1.
        bad[:, :16, 3:5] = 0.
        bad[:, :16, 9:11] = 0.
        bad[:, :16, 8] = 1.
        bad[:, :16, 14] = 12.
        effect = dict(censor_rate_before=.5, censor_rate_after=.75, rate_difference=.25)
    elif name == 'censored_Q_visible_time_coupling':
        bad[:, 32:, 15] = z[:, 32:, 15][:, ::-1]
        assert np.array_equal(np.sort(bad[:, :, 15], axis=1), np.sort(z[:, :, 15], axis=1))
        assert np.array_equal(np.sort(bad[:, 32:, 15], axis=1), np.sort(z[:, 32:, 15], axis=1))
        assert np.array_equal(bad[:, :, :15], z[:, :, :15])
        assert np.array_equal(bad[:, :, 16:], z[:, :, 16:])
        assert np.array_equal(bad[:, :32], z[:, :32])
        effect = dict(Q_separation_in_T_units=20., t90_separation_ns=8.,
            censored_Q_t90_covariance_before=40., censored_Q_t90_covariance_after=-40.,
            overall_Q_marginal_exact=True, censored_Q_marginal_exact=True,
            state_marginal_exact=True, measurable_rows_exact=True,
            state_Q_dependence_unchanged=True, tested='Q and visible time within the censored state')
    elif name == 'measurable_rise_plus4ns':
        bad[:, :32, 9] += 4.
        bad[:, :32, 10] -= 4.
        assert np.array_equal(bad[:, :32, 9], bad[:, :32, 12]-bad[:, :32, 10])
        assert np.array_equal(bad[:, :, 15:], z[:, :, 15:])
        effect = dict(measurable_rise_shift_ns=4., left10_shift_ns=-4.,
            crossing_identity_preserved=True, state_Q_peak_FWHM_unchanged=True)
    else:
        bad[:, :16, 15], bad[:, 48:, 15] = z[:, 48:, 15], z[:, :16, 15]
        assert np.array_equal(np.sort(bad[:, :, 15], axis=1), np.sort(z[:, :, 15], axis=1))
        assert np.array_equal(bad[:, :, :15], z[:, :, :15])
        assert np.array_equal(bad[:, :, 16:], z[:, :, 16:])
        effect = dict(overall_Q_marginal_exact=True, state_marginal_exact=True,
            measurable_Q_mean_before=2., measurable_Q_mean_after=12.,
            censored_Q_mean_before=12., censored_Q_mean_after=2.,
            tested='state and Q dependence at fixed marginals')
    discrepancy = energy_discrepancy018(bad, z, teacher_self=self_u)
    # This zero-null device only exercises the fixed fail branch. It is not an
    # independent stochastic teacher-bank precision check; the real job does that.
    cell = observable_cell018(discrepancy, np.zeros(128), group, weights)
    assert (discrepancy > .10).all() and ((discrepancy-identity) > .10).all()
    assert cell['status'] == 'fail'
    controls.append(dict(name=name, effect=effect, D_U=float(discrepancy[0]),
        D_U_increase_from_identity=float(discrepancy[0]-identity[0]), detected=True,
        independent_teacher_precision_test=False, feature_space_artificial_only=True))
    print('018 artificial check: detected '+name, flush=True)

result = dict(schema='n01/wp4-018-semantic-check/v1', status='PASS',
    python_executable=sys.executable, python_version=sys.version,
    scope='9 fixed artificial hard256 rows and four fixed feature-space distribution controls only',
    private_arrays_read=0, SOURCE_REAL_DEV_CP_reads=0, NN_AD_G_new_RNG=0,
    old_full_engineering_suite_rerun=False, pure_NumPy_import_no_torch=True,
    vectorized_crossings_old_definition_all_fields_exact=True, boundary_rows=len(waves),
    conditional_bound_does_not_assert_unseen_crossing_exists=True,
    cov8_pass_with_all_rise_missing_and_original_denominator=True,
    same_array_U_analytic_negative_verified=True, identity_U=float(identity[0]),
    controls=controls, artificial_ordered_distance_counts=DISTANCE_COUNTS018,
    source_files={relative:hashlib.sha256((root/relative).read_bytes()).hexdigest()
        for relative in ('src/n01/stats018.py','scripts/run_wp4_018.py','wp4_018_config.json',
                         'WP4_018_SAVED_INPUTS.json','docs/WP4_018_CONFIGURATION.md',
                         'receipts/wp4_018_20261016/engineering_01/check_semantics018.py')})
destination = Path(__file__).with_name('SEMANTIC_CHECK.json')
faulthandler.cancel_dump_traceback_later()
with destination.open('x', encoding='utf-8', newline='\n') as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps(result, indent=2, allow_nan=False))
