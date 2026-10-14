"""Independent synthetic literal-replication checks of the011 metric cache.

No scientific data or saved waveforms/checkpoints are loaded. No NN/gradients.
"""
import os
os.environ['TEMP'] = os.environ['TMP'] = os.environ['TMPDIR'] = 'E:/GAN/runtime_tmp'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '2'
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from artificial_reference import literal_w1, literal_loss, expand_indices, artificial_fixture

ROOT = Path('E:/GAN/research/N01')
OWN = Path(__file__).resolve().parent
OUTPUT = OWN/'metrics_synthetic_01'
OUTPUT.mkdir(exist_ok=False)
SOURCE = ROOT/'scripts/stability011_metrics.py'
SOURCE_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('independent_stability011_metrics', SOURCE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
checks = []
max_error = 0.


def run_test(name, test):
    test()
    checks.append(dict(name=name, passed=True))


def near(a, b, tolerance=3e-12):
    global max_error
    error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    max_error = max(max_error, error)
    assert error < tolerance, (a, b, error)


def compare_full(cached, fake, real, stats, banks, fw, rw):
    actual = cached.evaluate(fw, rw, column_chunk=7)
    reference = literal_loss(fake[expand_indices(fw)], real[expand_indices(rw)], stats, banks)
    for key in ('wave', 'QPSD'):
        near(actual['components'][key], reference[key])
    for key in ('swd', 'coordinate', 'short_lag'):
        near(actual['wave_subcomponents'][key], reference[key])
    assert actual['conditional_counts'] == reference['conditional_counts']
    assert actual['conditional_bins_used'] == reference['conditional_bins_used']
    assert actual['conditional_coverage_complete'] == reference['full_coverage']
    assert actual['fake_rows'] == int(np.sum(fw)) and actual['real_rows'] == int(np.sum(rw))
    near(actual['proxy'], reference['wave']+2*reference['QPSD'])
    return actual


def analytical_ties_unequal_domain_masses():
    f = np.array([[0., 2.], [0., 2.], [2., 0.]])
    r = np.array([[0., 1.], [1., 1.], [3., -1.]])
    fw, rw = np.array([0, 3, 1]), np.array([1, 0, 2])
    c = m.CDFCache.create(f, r)
    assert not c.order.flags.writeable and not c.gap.flags.writeable
    original_order, original_gap = c.order.copy(), c.gap.copy()
    for chunk in (1, 2, 256):
        near(c.distance(fw, rw, column_chunk=chunk), 5/3)
        near(c.distance(fw*3, rw*7, column_chunk=chunk), 5/3)
    near(c.distance(fw, rw), literal_w1(f[expand_indices(fw)], r[expand_indices(rw)]))
    assert np.array_equal(c.order, original_order) and np.array_equal(c.gap, original_gap)


def full_parent_packs_and_variable_REAL_groups():
    f, r, stats, banks = artificial_fixture()
    source_parent_m = np.array([2, 0, 1, 3, 0, 1, 0, 1, 1])
    real_groups = np.repeat(np.arange(5), [1, 4, 7, 2, 15])
    real_group_m = np.array([2, 0, 1, 3, 2])
    fw, rw = np.repeat(source_parent_m, 4), real_group_m[real_groups]
    assert np.array_equal(fw.reshape(9, 4), np.repeat(source_parent_m[:, None], 4, axis=1))
    assert fw.sum() == 36 and rw.sum() == 45 and len(rw) == 29
    cache = m.WholeProxyCache.create(m.Representation.create(f, stats, banks), m.Representation.create(r, stats, banks))
    actual = compare_full(cache, f, r, stats, banks, fw, rw)
    base = compare_full(cache, f, r, stats, banks, np.ones(36, int), np.ones(29, int))
    near(cache.evaluate(fw*2, rw*3)['components']['wave'], actual['components']['wave'])
    near(cache.evaluate(fw*2, rw*3)['QPSD_global'], actual['QPSD_global'])
    assert np.isfinite(base['proxy']) and np.isfinite(actual['proxy'])


def occurrence_support_strict_15_16():
    f, r, stats, banks = artificial_fixture()
    stats['charge_bin_boundaries'] = np.full(7, 100.)
    cache = m.WholeProxyCache.create(m.Representation.create(f, stats, banks), m.Representation.create(r, stats, banks))
    fw, rw = np.zeros(36, int), np.zeros(29, int)
    fw[:4] = 4
    rw[0] = 16
    actual = compare_full(cache, f, r, stats, banks, fw, rw)
    assert actual['conditional_counts'][0] == dict(bin=0, fake_count=16, real_count=16, supported=True)
    assert actual['conditional_bins_used'] == 1
    fw[3] = 3
    actual = compare_full(cache, f, r, stats, banks, fw, rw)
    assert actual['conditional_counts'][0] == dict(bin=0, fake_count=15, real_count=16, supported=False)
    assert actual['conditional_bins_used'] == 0 and actual['QPSD_conditional_mean'] is None
    near(actual['components']['QPSD'], actual['QPSD_global'])
    fw[3], rw[0] = 4, 15
    actual = compare_full(cache, f, r, stats, banks, fw, rw)
    assert actual['conditional_counts'][0] == dict(bin=0, fake_count=16, real_count=15, supported=False)
    assert actual['conditional_bins_used'] == 0


def full_partial_all_missing_bins_literal_loss():
    _, _, stats, banks = artificial_fixture()
    t = np.arange(256)
    template = np.exp(-((t-16)/7)**2)+.12*np.exp(-np.maximum(t-16, 0)/30)
    charges = np.repeat(np.exp(np.array([-4., -2.5, -1.5, -.5, .5, 1.5, 2.5, 4.])), 16)
    f = charges[:, None]/template.sum()*template
    real_template = template*(1+0.03*np.sin(t/9))
    r = (charges*1.02)[:, None]/real_template.sum()*real_template
    cache = m.WholeProxyCache.create(m.Representation.create(f, stats, banks), m.Representation.create(r, stats, banks))
    fw, rw = np.ones(128, int), np.ones(128, int)
    full = compare_full(cache, f, r, stats, banks, fw, rw)
    assert full['conditional_bins_used'] == 8 and full['conditional_coverage_complete']
    fw[0] = 0
    partial = compare_full(cache, f, r, stats, banks, fw, rw)
    assert partial['conditional_bins_used'] == 7 and not partial['conditional_coverage_complete']
    assert partial['conditional_counts'][0]['fake_count'] == 15
    fw = np.tile(np.r_[np.ones(12, int), np.zeros(4, int)], 8)
    missing = compare_full(cache, f, r, stats, banks, fw, rw)
    assert missing['conditional_bins_used'] == 0 and not missing['conditional_coverage_complete']
    assert missing['fake_rows'] == 96 and missing['real_rows'] == 128
    near(missing['components']['QPSD'], missing['QPSD_global'])


def pooled_CDF_not_mean_parent_CDF():
    f, r = np.array([0.]*4+[2.]*4), np.array([0., 2.])
    pooled = m.CDFCache.create(f, r).distance(np.ones(8, int), np.ones(2, int))
    parent_average = (literal_w1(f[:4], r)+literal_w1(f[4:], r))/2
    near(pooled, 0.)
    near(parent_average, 1.)
    assert pooled != parent_average


def paired_proxy_and_nonconstant_source_guards():
    metrics = [[dict(proxy=v, conditional_coverage_complete=True) for v in (1., 2., 3.)],
               [dict(proxy=v, conditional_coverage_complete=True) for v in (.9, 1.8, 2.7)]]
    record = m.paired_record(metrics)
    near(record['S_A'], 2.)
    near(record['S_final'], 1.8)
    near(record['Delta'], -.2)
    near(record['relative_contrast_percent'], -10.)
    assert record['conditional_coverage_complete']
    metrics[1][2]['conditional_coverage_complete'] = False
    assert not m.paired_record(metrics)['conditional_coverage_complete']
    for invalid in (np.array([0, 0]), np.array([1., .5]), np.array([1, -1]), np.array([np.nan, 1])):
        try:
            m.counts(invalid, 2)
        except ValueError:
            pass
        else:
            raise AssertionError('Bad multiplicity accepted')


try:
    for name, test in [('analytical_ties_unequal_domain_masses', analytical_ties_unequal_domain_masses),
            ('whole_parent4_pack_variable_REAL_group_N', full_parent_packs_and_variable_REAL_groups),
            ('occurrence_support_15_16_with_duplicate_rows', occurrence_support_strict_15_16),
            ('complete_partial_all_missing_bins_fallback', full_partial_all_missing_bins_literal_loss),
            ('pooled_W1_not_mean_parent_distance', pooled_CDF_not_mean_parent_CDF),
            ('paired_proxy_and_bad_multiplicity_guard', paired_proxy_and_nonconstant_source_guards)]:
        run_test(name, test)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    receipt = dict(schema='n01/stability011-independent-metrics-artificial/v1',
        UTC=datetime.now(timezone.utc).isoformat(), status='passed', source_sha256=SOURCE_SHA,
        checks=checks, passed_checks=len(checks), maximum_numeric_absolute_error=max_error,
        artificial_only=True, actual_scientific_bootstrap_replicates=0, saved_scientific_array_reads=0,
        original_data_or_weight_reads=0, model_instances=0, model_forwards=0, gradient_calls=0,
        optimizer_updates=0, DEV_reads=0, temporary_root=os.environ['TEMP'],
        numerical_threads_requested=2, literal_reference_sha256=hashlib.sha256((OWN/'artificial_reference.py').read_bytes()).hexdigest())
    with (OUTPUT/'ENGINEERING_RECEIPT.json').open('x',encoding='utf-8') as f:
        json.dump(receipt,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')
    print(json.dumps(receipt))
except BaseException as error:
    with (OUTPUT/'FAILURE_RECEIPT.json').open('x',encoding='utf-8') as f:
        json.dump(dict(error_type=type(error).__name__, error=str(error), completed_checks=checks,
                      artificial_only=True, model_forwards=0, scientific_bootstrap_replicates=0), f,indent=2)
    raise
