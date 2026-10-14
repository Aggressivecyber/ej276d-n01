"""Artificial-only semantic checks for the two actual011 analysis scripts."""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
import stability011_metrics as metrics
import run_stability011 as runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference():
    path = ROOT/'receipts/shape010_20261014/postrun_independent_01/readback_attempt02.py'
    nodes = [v for v in ast.parse(path.read_text(encoding='utf8')).body
        if isinstance(v, ast.FunctionDef) and v.name in {'w1', 'wave_views', 'q_features', 'loss_statistics'}]
    def require(flag, message):
        if not flag:
            raise AssertionError(message)
    namespace = {'np': np, 'require': require}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def fixture():
    rng = np.random.default_rng(2026101455)
    target_q = np.exp([-2.2, -1.7, -1.2, -.7, -.2, .3, .8, 1.3])
    pulse = np.exp(-((np.arange(256)-16)/12.)**2)+.08*np.exp(-np.arange(256)/60.)
    pulse /= pulse.sum()
    fake = target_q[np.arange(128)//16, None]*pulse[None]
    rgroup = np.repeat(np.arange(16), np.tile(np.arange(8, 16), 2))
    real = target_q[np.arange(len(rgroup))%8, None]*pulse[None]*(1+rng.normal(0, .0001, (len(rgroup), 256)))
    stats = dict(peak_floor=1e-9, q_floor=1e-6, q_reference=1.,
        charge_bin_boundaries=[-2., -1.5, -1., -.5, 0., .5, 1.], minimum_conditional_support=16,
        views={name: dict(center=np.zeros(d), scale=np.ones(d)) for name, d in zip(metrics.VIEWS, metrics.DIMENSIONS)},
        qpsd=dict(center=np.zeros(2), scale=np.ones(2)))
    def directions(d, count):
        value = rng.normal(size=(d, count))
        return value/np.linalg.norm(value, axis=0)
    banks = {'wave:'+name: directions(d, 32) for name, d in zip(metrics.VIEWS, metrics.DIMENSIONS)}
    banks.update({f'lag:{name}:{lag}': directions(2, 4) for name in metrics.COORDINATE_VIEWS for lag in metrics.LAGS})
    banks['qpsd'] = directions(2, 16)
    real_repr = metrics.Representation.create(real, stats, banks)
    a = metrics.WholeProxyCache.create(metrics.Representation.create(fake, stats, banks), real_repr)
    final = metrics.WholeProxyCache.create(metrics.Representation.create(1.5*fake, stats, banks), real_repr)
    return fake, real, np.repeat(np.arange(32), 4), rgroup, stats, banks, a, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    assert not out.exists()
    out.mkdir(parents=True)
    refs = reference()
    tests = []
    rng = np.random.default_rng(2026101456)
    errors = []
    for n, m in ((9, 9), (9, 13)):
        x, y = np.round(rng.normal(size=(n, 7)), 1), np.round(rng.normal(size=(m, 7)), 1)
        fc, rc = rng.integers(0, 4, n), rng.integers(0, 4, m)
        fc[0], rc[0] = 2, 1
        value = metrics.CDFCache.create(x, y)
        error = abs(value.distance(fc, rc, column_chunk=3)-refs['w1'](np.repeat(x, fc, axis=0), np.repeat(y, rc, axis=0)))
        assert error < 1e-12
        errors.append(error)
    tests.append(dict(name='CDF_literal_repeat_equal_unequal_N_ties_zero_counts', passed=True, maximum_error=max(errors)))
    cached = metrics.CDFCache.create(np.arange(5.)[:, None], np.arange(7.)[:, None])
    rejects = 0
    for counts in (np.ones(4), [-1, 1, 1, 1, 1], [.5, 1, 1, 1, 1], [np.nan, 1, 1, 1, 1], np.zeros(5)):
        try:
            cached.distance(counts, np.ones(7))
        except ValueError:
            rejects += 1
    assert rejects == 5
    tests.append(dict(name='CDF_rejects_mismatch_negative_noninteger_nonfinite_zero_mass', passed=True, rejected=rejects))

    fake, real, parent, groups, stats, banks, a, final = fixture()
    fw = np.repeat(np.tile([0, 1, 2, 1], 8), 4)
    rw = np.array([0, 2, 1, 1, 1, 1, 1, 1]*2)[groups]
    assert fw.sum() == 128 and rw.sum() == 186 != len(real)
    for name, wave, value in (('A', fake, a), ('final_specific_Qbins', 1.5*fake, final)):
        exact = refs['loss_statistics'](np.repeat(wave, fw, axis=0), np.repeat(real, rw, axis=0), stats, banks)
        obtained = value.evaluate(fw, rw, column_chunk=128)
        assert obtained['conditional_counts'] == exact['conditional_counts']
        err = max(abs(obtained['components'][key]-exact['components'][key]) for key in ('wave', 'QPSD'))
        err = max(err, *[abs(obtained['wave_subcomponents'][key]-exact['wave_subcomponents'][key]) for key in ('swd', 'coordinate', 'short_lag')])
        assert err < 1e-12
        tests.append(dict(name='whole_proxy_literal_repeat_'+name, passed=True, maximum_error=err,
            real_repeated_rows=obtained['real_rows'], conditional_bins_used=obtained['conditional_bins_used']))
    assert a.evaluate(fw, rw)['conditional_counts'][0]['fake_count'] == 16
    assert np.count_nonzero(fw[a.fake.q_bins == 0]) == 12
    tests.append(dict(name='support_16_is_occurrence_count_not_12_unique_rows', passed=True))
    fonly = np.where(a.fake.q_bins == 0, 1, 0)*8
    ronly = np.where(a.real.q_bins == 7, 1, 0)*8
    exact = refs['loss_statistics'](np.repeat(fake, fonly, axis=0), np.repeat(real, ronly, axis=0), stats, banks)
    obtained = a.evaluate(fonly, ronly)
    assert obtained['conditional_bins_used'] == 0 and not obtained['conditional_coverage_complete']
    assert abs(obtained['components']['QPSD']-exact['components']['QPSD']) < 1e-12
    assert obtained['components']['QPSD'] == obtained['QPSD_global']
    tests.append(dict(name='all_missing_Qbins_global_only_exact_fallback', passed=True))
    guarded = np.concatenate((np.zeros((1, 256)), -fake[:1], fake[:1]*1e-8))
    np.testing.assert_allclose(metrics.q_features(guarded, stats)[0], refs['q_features'](guarded, stats), rtol=1e-15, atol=1e-15)
    tests.append(dict(name='zero_negative_nearzero_Q_guard_retains_rows', passed=True))
    before = sha(ROOT/'scripts/stability011_metrics.py')
    original_cache_digest = hashlib.sha256(cached.order.tobytes()+cached.gap.tobytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=3) as pool:
        values = list(pool.map(lambda _: cached.distance(np.ones(5, int), np.ones(7, int)), range(9)))
    assert len(set(values)) == 1 and original_cache_digest == hashlib.sha256(cached.order.tobytes()+cached.gap.tobytes()).hexdigest()
    assert not cached.order.flags.writeable and not cached.gap.flags.writeable
    tests.append(dict(name='threaded_CDF_cache_immutable', passed=True))
    assert metrics.WAVE_COLUMNS == 8776
    assert sum(c.order.shape[0] for grp in a.groups.values() for c in grp) == 8776
    tests.append(dict(name='all_8776_wave_columns_hierarchical_weights_preserved', passed=True))

    policy = dict(bootstrap_replicates=6, source_parent_clusters=32, REAL_reference_original_groups=16,
        seeds=dict(source_parent_bootstrap=2026101451, REAL_group_bootstrap=2026101452),
        interval=dict(quantiles=[.025, .975], method='linear'),
        resource_limits=dict(CPU_threads=3, wall_seconds=120, output_MiB=16, job_tree_RSS_GiB=1))
    global_before = np.random.get_state()
    fc, rc = runner.multiplicities(policy)
    global_after = np.random.get_state()
    assert all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(global_before, global_after))
    source_draws = np.random.Generator(np.random.PCG64(2026101451)).integers(0, 32, size=(6, 32))
    real_draws = np.random.Generator(np.random.PCG64(2026101452)).integers(0, 16, size=(6, 16))
    np.testing.assert_array_equal(fc, np.stack([np.bincount(row, minlength=32) for row in source_draws]))
    np.testing.assert_array_equal(rc, np.stack([np.bincount(row, minlength=16) for row in real_draws]))
    tests.append(dict(name='exact_PCG64_integers_bincount_and_global_RNG_unchanged', passed=True))
    store = runner.Store(out/'synthetic_success', policy, artificial=True)
    store.point_passed = True
    protected_source = str(ROOT/'scripts/stability011_metrics.py')
    store.protection = dict(inputs_sha256={protected_source: sha(protected_source)},
        sources_sha256={protected_source: sha(protected_source)})
    records = runner.run_bootstrap([[a]*3, [final]*3], parent, groups, policy, store, {'synthetic_only': True},
        lambda: runner.assert_unchanged(store.protection))
    runner.save_results(store, records)
    result = runner.summarize(records, policy)
    assert len(records) == store.saved == store.attempted == store.finished == 6
    assert (store.root/'FIRST_BOOTSTRAP_PROOF.json').exists()
    assert len((store.root/'bootstrap.jsonl').read_text().splitlines()) == 6
    assert len((store.root/'bootstrap_attempts.jsonl').read_text().splitlines()) == 6
    assert len((store.root/'bootstrap_computed.jsonl').read_text().splitlines()) == 6
    assert result['classification'] == 'coverage_limited'
    actual_status = json.loads((store.root/'status.json').read_text())
    assert actual_status['inputs_sha256'] == store.protection['inputs_sha256']
    assert actual_status['sources_sha256'] == store.protection['sources_sha256']
    tests.append(dict(name='actual_Store_status_and_base_share_protected_hash_fields', passed=True))
    tests.append(dict(name='actual_driver_bounded_threads_saved_counts_wholeproxy_coverage_classification', passed=True,
        synthetic_replicates=6, optimizer_updates=0, NN_forward_rows=0))
    for index, record in enumerate(records):
        explicit = []
        for wave in (fake, 1.5*fake):
            p = refs['loss_statistics'](np.repeat(wave, fc[index][parent], axis=0), np.repeat(real, rc[index][groups], axis=0), stats, banks)
            explicit.append(p['components']['wave']+2*p['components']['QPSD'])
        assert abs(record['paired']['Delta']-(explicit[1]-explicit[0])) < 1e-12
    tests.append(dict(name='all_six_driver_replicates_match_explicit_pooled_proxy', passed=True))
    class FailAfterFirst:
        def __init__(self, original):
            self.original, self.calls = original, 0
        def evaluate(self, fw, rw):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('deliberate synthetic second-replicate failure')
            return self.original.evaluate(fw, rw)
    failed_policy = dict(policy, resource_limits=dict(policy['resource_limits'], CPU_threads=1))
    failed = runner.Store(out/'synthetic_partial_failure', failed_policy, artificial=True)
    failed.point_passed = True
    try:
        runner.run_bootstrap([[FailAfterFirst(a), a, a], [final]*3], parent, groups, failed_policy, failed,
            {'synthetic_only': True}, lambda: None)
        raise AssertionError('Expected synthetic failure')
    except RuntimeError as error:
        assert 'deliberate synthetic' in str(error)
        runner.write_failure(failed, error)
    assert failed.saved == 1 and failed.attempted == 2 and failed.finished == 1
    assert len((failed.root/'bootstrap.jsonl').read_text().splitlines()) == 1
    assert not (failed.root/'COMPLETE.json').exists()
    assert len((failed.root/'bootstrap_attempts.jsonl').read_text().splitlines()) == 2
    assert len((failed.root/'bootstrap_computed.jsonl').read_text().splitlines()) == 1
    assert (failed.root/'FAILED.json').exists()
    tests.append(dict(name='partial_failure_keeps_one_saved_two_attempts_no_fill_restart_or_complete', passed=True))
    native_fixture = '''import sys, threading
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_stability011 as r
import numpy as np
mode, output = sys.argv[2], sys.argv[3]
p = dict(bootstrap_replicates=4,source_parent_clusters=8,REAL_reference_original_groups=6,
seeds=dict(source_parent_bootstrap=2026101451,REAL_group_bootstrap=2026101452),
resource_limits=dict(CPU_threads=2,wall_seconds=120,output_MiB=8,job_tree_RSS_GiB=1))
s=r.Store(output,p,artificial=True); s.point_passed=True
class ArtificialFailure:
 def evaluate(self, fw, rw):
  if mode=='kill': threading.Event().wait(30)
  raise RuntimeError('deliberate artificial first-metric interruption')
try:
 r.run_bootstrap([[ArtificialFailure()]*3]*2,np.repeat(np.arange(8),4),np.arange(12)%6,p,s,dict(artificial_only=True),lambda:None)
except BaseException as e:
 r.write_failure(s,e)
 sys.exit(7)
sys.exit(0)
'''
    command = [sys.executable, '-B', '-', str(ROOT/'scripts')]
    native_fail = subprocess.run(command+['fail', str(out/'synthetic_native_nonzero')], input=native_fixture,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    assert native_fail.returncode == 7
    failure_value = json.loads((out/'synthetic_native_nonzero/FAILED.json').read_text())
    entered = (out/'synthetic_native_nonzero/bootstrap_attempts.jsonl').read_text().splitlines()
    assert failure_value['bootstrap_replicates_attempted'] == len(entered) >= 1
    assert failure_value['bootstrap_replicates_computed_successfully'] == failure_value['bootstrap_replicates_completed'] == 0
    assert not (out/'synthetic_native_nonzero/COMPLETE.json').exists()
    tests.append(dict(name='native_nonzero_before_first_completion_retains_durable_attempts_and_FAILED', passed=True,
        native_exit_code=native_fail.returncode, durable_attempts=len(entered)))
    killed = subprocess.Popen(command+['kill', str(out/'synthetic_native_kill')], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    killed.stdin.write(native_fixture)
    killed.stdin.close()
    start = time.monotonic()
    attempt_file = out/'synthetic_native_kill/bootstrap_attempts.jsonl'
    while (not attempt_file.exists() or not attempt_file.read_text().strip()) and killed.poll() is None and time.monotonic()-start < 10:
        time.sleep(.01)
    assert attempt_file.exists() and attempt_file.read_text().strip()
    killed.kill()
    killed.wait(timeout=10)
    assert killed.returncode != 0
    persisted = json.loads((out/'synthetic_native_kill/status.json').read_text())
    durable_entries = attempt_file.read_text().splitlines()
    assert len(durable_entries) >= 1 and persisted['bootstrap_replicates_completed'] == 0
    assert not (out/'synthetic_native_kill/COMPLETE.json').exists()
    tests.append(dict(name='hard_kill_during_first_inflight_keeps_start_journal_no_zero_attempt_inference', passed=True,
        native_exit_code=killed.returncode, durable_attempt_lower_bound=len(durable_entries),
        status_snapshot_attempts=persisted['bootstrap_replicates_attempted'], unfinished_actual_compute_state='unknown'))
    try:
        runner.Store(store.root, policy, artificial=True)
        raise AssertionError('Existing output overwritten')
    except ValueError:
        pass
    tests.append(dict(name='fresh_output_no_overwrite', passed=True))
    assert before == sha(ROOT/'scripts/stability011_metrics.py')
    receipt = dict(schema='n01/stability011-native-check/v1', artificial_only=True, science_bootstrap_replicates=0,
        private_array_decodes=0, checkpoint_decodes=0, model_forward_rows=0, optimizer_updates=0,
        passed=len(tests), total=len(tests), tests=tests,
        source_sha256={name: sha(ROOT/name) for name in ('scripts/stability011_metrics.py', 'scripts/run_stability011.py')})
    (out/'SYNTHETIC_CHECK.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()
