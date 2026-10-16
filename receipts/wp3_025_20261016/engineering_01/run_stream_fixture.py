"""One fixed, public synthetic comparison of025 streaming and the mature full CDF path."""
import json
import os
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root/'scripts'), str(root/'src/n01')]
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from run_wp3_025 import settings025, budget_caps025, features025, stream_score025, MARGINALS
from stability011_metrics import Representation, WholeProxyCache, VIEWS, DIMENSIONS, COORDINATE_VIEWS, LAGS
from wp1_evaluation import raw_distance
from wp1_ops import Store, require, sha

started = time.monotonic()
t = np.arange(256, dtype=np.float64)
profile = np.where(t <= 16, (t/16)**2, np.exp(-(t-16)/37))
famp = np.repeat(np.arange(1, 9), 17).astype(np.float64)
ramp = np.repeat(np.arange(1, 9)+.1, 19).astype(np.float64)
f = famp[:, None]*profile
r = ramp[:, None]*profile
f[0], f[1] = -.1*profile, 0.
r[0], r[1] = -.08*profile, 0.
stats = dict(peak_floor=1e-6, q_floor=.002, q_reference=1., fit_split='TRAIN', minimum_conditional_support=16,
    views={name: dict(center=np.linspace(-.11, .09, dim), scale=np.linspace(.7, 1.2, dim))
           for name, dim in zip(VIEWS, DIMENSIONS)},
    qpsd=dict(center=[.1, .03], scale=[.8, 1.1]),
    charge_bin_boundaries=np.log((np.arange(1, 8)+.5)*profile.sum()))
banks = {}
for name, dim in zip(VIEWS, DIMENSIONS):
    a = np.sin((np.arange(dim)[:, None]+1)*(np.arange(32)[None, :]+1)*.013)+.1
    banks['wave:'+name] = a/np.linalg.norm(a, axis=0, keepdims=True)
for name in COORDINATE_VIEWS:
    for lag in LAGS:
        a = np.array([[1., 1., -1., 2.], [1., -2., 1., 1.]])
        banks[f'lag:{name}:{lag}'] = a/np.linalg.norm(a, axis=0, keepdims=True)
a = np.stack((np.cos(np.arange(16)*.3), np.sin(np.arange(16)*.3)))
banks['qpsd'] = a/np.linalg.norm(a, axis=0, keepdims=True)
p = settings025()
p['budget_caps'] = budget_caps025(p)
store = Store(root/'receipts/wp3_025_20261016/engineering_01/fixture_01', p, {}, artificial=True)
fake = features025(f, stats)
real = features025(r, stats)
actual = stream_score025(fake, real, banks, store, 'artificial_fixed_unequal_N_ties')
reference = WholeProxyCache.create(Representation.create(f, stats, banks), Representation.create(r, stats, banks)).evaluate(
    np.ones(len(f), np.int64), np.ones(len(r), np.int64), column_chunk=16)
errors = {}
for name in ('wave', 'QPSD'):
    errors[name] = abs(actual['values'][name]-reference['components'][name])
for name in ('swd', 'coordinate', 'short_lag'):
    errors[name] = abs(actual['wave_subcomponents'][name]-reference['wave_subcomponents'][name])
for name in MARGINALS:
    detail = raw_distance(fake['observables'][name], real['observables'][name])
    errors[name] = abs(actual['defined_population'][name]['value']-detail['value'])
    require(all(actual['defined_population'][name][key] == detail[key] for key in
                ('fake_rows', 'real_rows', 'fake_defined_rows', 'real_defined_rows')), 'Fixture defined populations differ')
np.testing.assert_allclose(list(errors.values()), 0., atol=2e-12, rtol=0.)
require(actual['conditional_counts'] == reference['conditional_counts'] and actual['conditional_coverage_complete'],
        'Fixture conditional support differs or misses a bin')
require(fake['flags']['Q_nonpositive'].sum() == real['flags']['Q_nonpositive'].sum() == 2 and
        fake['flags']['PSD_undefined'].sum() == real['flags']['PSD_undefined'].sum() == 1,
        'Fixture zero/negative Q did not reach the global/flag path')
require(store.counts['CDF_create_calls_cap'] == 568 and
        store.counts['wave_scalar_W1_columns_exact'] == 8776 and
        store.counts['global_QPSD_scalar_W1_columns_exact'] == 18 and
        store.counts['conditional_scalar_W1_columns_cap'] == 8 and
        store.counts['marginal_scalar_W1_columns_cap'] == 8 and not store.inflight,
        'Fixture streaming budget algebra differs')
if os.name == 'nt':
    import ctypes
    memory = ctypes.create_string_buffer(72)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    require(ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(memory), len(memory)) != 0, 'Fixture native process memory read failed')
    peak_rss_KiB = (ctypes.c_size_t.from_buffer(memory, 8).value+1023)//1024
else:
    import resource
    peak_rss_KiB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
store.json('RESULT.json', dict(schema='n01/wp3-025-artificial-stream-check/v1', status='PASS',
    driver=dict(bytes=(root/'scripts/run_wp3_025.py').stat().st_size, sha256=sha(root/'scripts/run_wp3_025.py')),
    fixture_sha256=sha(Path(__file__)), fake_rows=len(f), real_rows=len(r),
    checks='one combined real function path: sixview grouping,8lags,unequalN,ties,8Qbins,zeroQ,negativeQ,defined morphology',
    maximum_absolute_difference=max(errors.values()), errors=errors,
    exact_conditional_counts=actual['conditional_counts'], stream_counts=store.counts,
    reference_CDF_columns=8810, reference_CDF_support_scalar_values=store.counts['CDF_support_scalar_values_cap'],
    total_artificial_CDF_columns=2*8810, total_artificial_support_values=2*store.counts['CDF_support_scalar_values_cap'],
    synthetic_wave_rows=288, feature_constructions=4, private_array_reads=0,
    NN_CP_AD_G_RNG_ADC_increment=0, wall_seconds=time.monotonic()-started,
    peak_RSS_KiB=peak_rss_KiB,
    actual=actual, reference=reference))
print(json.dumps(dict(status='PASS', maximum_absolute_difference=max(errors.values()),
    stream_CDF_calls=store.counts['CDF_create_calls_cap'], fixture=str(store.root)), indent=2))
