"""Tiny new014 saved-layout semantics only; no real inputs, CP, model or AD."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def require(value, text):
    if not value:
        raise AssertionError(text)


def fails(fn):
    try:
        fn()
    except (ValueError, AssertionError):
        return
    raise AssertionError('Expected malformed artificial input to fail')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    out = a.out.resolve()
    require(out.drive.upper() in ('E:', 'D:') and not out.exists(), 'fresh E/D tiny engineering directory')
    out.mkdir(parents=True)
    os.environ.update(TEMP=str(out), TMP=str(out), TMPDIR=str(out), PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    started = time.monotonic()
    here = Path(__file__).resolve().parent
    w = module(here/'readback_saved_arrays.py', 'artificial_readback014')
    require(not {'numpy', 'torch', 'n01'} & set(sys.modules), 'worker import must remain stdlib')
    import numpy as np
    k = module(here/'saved_array_kernels.py', 'artificial_statistical_kernels014')
    w.np, w.kernels = np, k
    checks = []

    codes = np.tile(np.arange(256, dtype=np.uint16)+8000, (8, 1))
    mean = np.linspace(-.001, .001, 8, dtype=np.float64)
    value, error = w.decode_window(codes, mean)
    require(np.array_equal(value, codes.astype(np.float64)/16383-.5-mean[:, None]) and error == 0, 'no-XOR reversible values')
    checks.append('actual_decoder_without_XOR_exact_codes')
    target = np.nextafter(value, np.inf)
    xor = value.view(np.uint64) ^ target.view(np.uint64)
    decoded, error = w.decode_window(codes, mean, xor)
    require(np.array_equal(decoded.view(np.uint64), target.view(np.uint64)) and 0 < error < 4e-15, 'actual IEEE XOR inverse bytes')
    checks.append('actual_decoder_optional_IEEE_XOR_preserves_original_synthetic_double_bits')

    parent, draw = np.repeat(np.arange(2), 4), np.tile(np.arange(4), 2)
    ids, banks = np.repeat(np.array(['synthetic:a', 'synthetic:b']), 4), np.zeros(8, np.int64)
    require(np.array_equal(w.lineage(parent, draw, banks, ids, parents=2, draws=4, bank=0), ids[::4]), 'valid lineage')
    wrong = draw.copy(); wrong[0] = 1
    fails(lambda: w.lineage(parent, wrong, banks, ids, parents=2, draws=4, bank=0))
    wrong = parent.copy(); wrong[-1] = 0
    fails(lambda: w.lineage(wrong, draw, banks, ids, parents=2, draws=4, bank=0))
    fails(lambda: w.lineage(parent, draw, banks.astype(float), ids, parents=2, draws=4, bank=0))
    fails(lambda: w.lineage(parent, draw, banks+1, ids, parents=2, draws=4, bank=0))
    checks.append('actual_parent_draw_bank_lineage_rejects_wrong_or_duplicated_identity')

    x = np.arange(2*4*256, dtype=np.float64).reshape(2, 4, 256)/1000
    m0, v0, f0 = w.wave_moments(x)
    m1, v1, _ = w.wave_moments(x+.4)
    m8, v8, f8 = w.wave_moments(np.concatenate((x, x+.4), axis=1))
    require(np.isclose(v0[0, 0], .32768/3, rtol=1e-14), 'r4 denominator3 literal variance')
    require(np.allclose(m8, (m0+m1)/2, atol=2e-15, rtol=0)
            and np.allclose(v8, (3*(v0+v1)+2*(m0-m1)**2)/7, atol=2e-15, rtol=0), 'r8 denominator7 literal merge')
    require(f0.shape == (2, 4, 256) and f8.shape == (2, 8, 256)
            and np.allclose(np.sqrt(v8/8), np.sqrt((f8*f8).sum(1)/56)), 'factors and MCSE')
    checks.append('actual_r4_r8_centered_factors_denominators_and_MCSE')

    scalar = np.empty((2, 8, 3), np.float64)
    scalar[:, :, 0] = np.arange(1., 9.)
    scalar[:, :, 1] = 2*scalar[:, :, 0]
    scalar[:, :, 2] = scalar[:, ::-1, 0]
    scalar[0, 1::2, 1] = np.nan; scalar[0, :, 2] = np.nan
    moments = k.scalar_moments(scalar)
    require(moments['jointly_defined_draws'][0, 0, 1] == 4
            and np.isclose(moments['sample_covariance'][0, 0, 1], 40/3)
            and moments['defined_draws'][0, 0] == 8 and moments['defined_draws'][0, 1] == 4
            and moments['defined_draws'][0, 2] == 0 and np.isnan(moments['sample_variance'][0, 2])
            and moments['jointly_defined_draws'][0, 2, 2] == 0, 'pair-specific NaN support and covariance literal answer')
    require(np.allclose(moments['MonteCarlo_SE'][0, :2], np.sqrt(moments['sample_variance'][0, :2]/[8, 4])), 'per-feature finite count MCSE')
    checks.append('mature_pairwise_finite_covariance_counts_NaNs_and_feature_specific_MCSE')

    wave = np.arange(1288, dtype=np.float64)/1288
    distribution, regular = -.5*wave, np.full(1288, 1e-6)
    require(np.array_equal(w.add_three_gradients(wave, distribution, regular), .5*wave+regular), 'three weighted vectors sum')
    fails(lambda: w.add_three_gradients(wave[:-1], distribution, regular))
    checks.append('actual_1288_three_vector_sum_without_AD')

    original = out/'synthetic_original'; original.mkdir()
    audit = w.Audit(SimpleNamespace(run_root=original, out=out/'synthetic_sidecar/artifacts', artificial_only=True))
    audit.op('SYNTHETIC_SUCCESS_NO_REAL_ARRAY', dict(bank_pairs_verified=1), lambda: 1)
    def fail():
        raise ValueError('synthetic incomplete decode')
    fails(lambda: audit.op('SYNTHETIC_FAILED_NO_REAL_ARRAY', dict(hard_rows_files_decoded=1), fail))
    state = json.loads((audit.out/'status.json').read_text(encoding='utf8'))
    rows = [json.loads(line) for line in (audit.out/'operations.jsonl').read_text(encoding='utf8').splitlines()]
    require(state['artificial_only'] and state['operations_entered'] == 2 and state['operations_completed'] == 1
            and state['counters']['hard_rows_files_decoded'] == 0 and state['attempted_counters']['hard_rows_files_decoded'] == 1
            and list(state['inflight']) == ['1'] and [v['event'] for v in rows] == ['entered', 'completed', 'entered', 'failed'], 'durable failure lower bound/unknowninflight')
    checks.append('actual_durable_writer_completed_attempted_failed_and_unknown_inflight')
    audit.equal(np.array([2e-14]), np.array([0.]), 'same_key')
    audit.equal(np.array([1e-14]), np.array([0.]), 'same_key')
    require(audit.errors['same_key'] == 2e-14, 'maximum comparison error not overwritten by later smaller value')
    checks.append('repeated_error_names_accumulate_maximum')
    require(not {'torch', 'n01'} & set(sys.modules), 'no neural package imported')
    result = dict(schema='n01/wp2-readback-tiny-artificial-QA/v1', status='PASS', artificial_only=True,
        checks=checks, passed_checks=len(checks), native_test_process_exit=0, elapsed_seconds=time.monotonic()-started,
        worker_sha256=sha(here/'readback_saved_arrays.py'), kernel_sha256=sha(here/'saved_array_kernels.py'),
        CPU_requested=2, OMP_MKL_OPENBLAS_NUMEXPR_effective_environment=2,
        artificial_waveforms=16, real_saved_array_reads=0, scientific_input_reads=0,
        CP_loads=0, NN_rows=0, AD_queries=0, training_updates=0, SSH_calls=0)
    (out/'ENGINEERING_RECEIPT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(status='PASS', checks=len(checks), receipt=str(out/'ENGINEERING_RECEIPT.json'),
                         sha256=sha(out/'ENGINEERING_RECEIPT.json'), elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
