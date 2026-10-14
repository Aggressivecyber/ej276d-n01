"""Synthetic-only 011 CDF equivalence and native CPU cost fixture.

Never decodes project arrays/checkpoints, invokes an NN, or runs a scientific
bootstrap. All generated numbers are explicitly artificial. The independent
reference is extracted as functions only from the already verified NumPy
readback source; its module-level scientific resampling bank is not executed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np


BENCH_CACHE = None


def cache(x, y):
    """Column-major sorted union; IDs refer to original row occurrences."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    joined = np.concatenate((x.T, y.T), axis=1)
    order = np.argsort(joined, axis=1, kind="stable")
    delta = np.diff(np.take_along_axis(joined, order, axis=1), axis=1)
    small_type = np.uint16 if joined.shape[1] <= 65536 else np.uint32
    return order.astype(small_type), np.ascontiguousarray(delta), len(x), len(y)


def reduce_cache(cached, fake_counts, real_counts):
    order, delta, nf, nr = cached
    fw = np.asarray(fake_counts, np.float64)
    rw = np.asarray(real_counts, np.float64)
    assert fw.shape == (nf,) and rw.shape == (nr,)
    assert np.min(fw) >= 0 and np.min(rw) >= 0 and fw.sum() > 0 and rw.sum() > 0
    mass = np.concatenate((fw / fw.sum(), -rw / rw.sum()))
    cumulative = mass[order]
    np.cumsum(cumulative, axis=1, out=cumulative)
    work = cumulative[:, :-1]
    np.abs(work, out=work)
    np.multiply(work, delta, out=work)
    return float(work.sum(axis=1).mean())


def reference_functions(path):
    source = Path(path).read_text(encoding="utf8")
    names = {"w1", "wave_views", "q_features", "loss_statistics"}
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"np": np, "require": lambda flag, message: None if flag else (_ for _ in ()).throw(AssertionError(message))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def prepare_proxy(fake, real, stats, banks, refs):
    fv = refs["wave_views"](fake, stats["peak_floor"])
    rv = refs["wave_views"](real, stats["peak_floor"])
    f = {name: (value-np.asarray(stats["views"][name]["center"]))/np.asarray(stats["views"][name]["scale"]) for name, value in fv.items()}
    r = {name: (value-np.asarray(stats["views"][name]["center"]))/np.asarray(stats["views"][name]["scale"]) for name, value in rv.items()}
    pieces = {"swd": [], "coordinate": [], "short_lag": []}
    for name in fv:
        pieces["swd"].append(cache(f[name] @ banks["wave:"+name], r[name] @ banks["wave:"+name]))
    for name in ("voltage", "shape"):
        pieces["coordinate"].append(cache(f[name], r[name]))
        for lag in (1, 2, 4, 8):
            direction = banks[f"lag:{name}:{lag}"]
            def pair(value):
                return (value[:, :-lag, None]*direction[0]+value[:, lag:, None]*direction[1]).reshape(len(value), -1)
            pieces["short_lag"].append(cache(pair(f[name]), pair(r[name])))
    fq = refs["q_features"](fake, stats)
    rq = refs["q_features"](real, stats)
    fz = (fq-np.asarray(stats["qpsd"]["center"]))/np.asarray(stats["qpsd"]["scale"])
    rz = (rq-np.asarray(stats["qpsd"]["center"]))/np.asarray(stats["qpsd"]["scale"])
    pieces["qglobal"] = [cache(fz, rz), cache(fz @ banks["qpsd"], rz @ banks["qpsd"])]
    fb = np.searchsorted(stats["charge_bin_boundaries"], fq[:, 0], side="left")
    rb = np.searchsorted(stats["charge_bin_boundaries"], rq[:, 0], side="left")
    pieces["conditional"] = [(fb == i, rb == i, cache(fz[fb == i, 1], rz[rb == i, 1]))
        if np.any(fb == i) and np.any(rb == i) else (fb == i, rb == i, None) for i in range(8)]
    return pieces


def reduce_proxy(prepared, fw, rw, minimum_support):
    parts = {name: float(np.mean([reduce_cache(value, fw, rw) for value in prepared[name]]))
        for name in ("swd", "coordinate", "short_lag")}
    global_q = .5*sum(reduce_cache(value, fw, rw) for value in prepared["qglobal"])
    counts, conditional = [], []
    for index, (fm, rm, value) in enumerate(prepared["conditional"]):
        nf, nr = int(fw[fm].sum()), int(rw[rm].sum())
        supported = nf >= minimum_support and nr >= minimum_support
        counts.append(dict(bin=index, fake_count=nf, real_count=nr, supported=supported))
        if supported:
            conditional.append(reduce_cache(value, fw[fm], rw[rm]))
    return dict(components=dict(wave=.5*parts["swd"]+.25*parts["coordinate"]+.25*parts["short_lag"],
        QPSD=.5*(global_q+float(np.mean(conditional))) if conditional else global_q),
        wave_subcomponents=parts, conditional_counts=counts, conditional_bins_used=len(conditional))


def check(reference_path):
    refs, digest = reference_functions(reference_path)
    rng = np.random.default_rng(2026101446)
    tests = []
    for name, n, m in (("equal_original_N", 9, 9), ("unequal_N_variable_REAL", 9, 13)):
        x = np.round(rng.normal(size=(n, 7)), 1)
        y = np.round(rng.normal(size=(m, 7)), 1)
        fw = rng.integers(0, 4, n)
        rw = rng.integers(0, 4, m)
        fw[0], rw[0] = 2, 1
        expanded = refs["w1"](np.repeat(x, fw, axis=0), np.repeat(y, rw, axis=0))
        weighted = reduce_cache(cache(x, y), fw, rw)
        error = abs(expanded-weighted)
        assert error < 1e-12
        tests.append(dict(name=name, passed=True, literal_repeat_error=error,
            original_rows=[n, m], repeated_rows=[int(fw.sum()), int(rw.sum())]))
    ties = np.array([[0., 1.], [0., 1.], [2., 1.], [7., 3.]])
    identical = reduce_cache(cache(ties, ties), [0, 2, 1, 0], [0, 2, 1, 0])
    assert abs(identical) < 1e-14
    tests.append(dict(name="ties_zero_multiplicity_identical", passed=True, W1=identical))

    nfake = 128
    sizes = np.tile(np.arange(8, 16), 2)
    rgroup = np.repeat(np.arange(16), sizes)
    nreal = len(rgroup)
    target_q = np.exp(np.array([-2.2, -1.7, -1.2, -.7, -.2, .3, .8, 1.3]))
    pulse = np.exp(-((np.arange(256)-16)/12.)**2)+.08*np.exp(-np.arange(256)/60.)
    pulse /= pulse.sum()
    fake = target_q[np.arange(nfake)//16, None]*pulse[None]
    real = target_q[np.arange(nreal)%8, None]*pulse[None]*(1.+rng.normal(0, .0001, (nreal, 256)))
    dimensions = dict(voltage=256, shape=256, difference=255, curvature=254,
        difference_smooth_5=255, difference_smooth_17=255)
    stats = dict(peak_floor=1e-9, q_floor=1e-6, q_reference=1.,
        charge_bin_boundaries=np.array([-2., -1.5, -1., -.5, 0., .5, 1.]),
        minimum_conditional_support=16,
        views={name: dict(center=np.zeros(d), scale=np.ones(d)) for name, d in dimensions.items()},
        qpsd=dict(center=np.zeros(2), scale=np.ones(2)))
    def directions(d, count):
        value = rng.normal(size=(d, count))
        return value/np.linalg.norm(value, axis=0)
    banks = {"wave:"+name: directions(d, 32) for name, d in dimensions.items()}
    banks.update({f"lag:{name}:{lag}": directions(2, 4) for name in ("voltage", "shape") for lag in (1, 2, 4, 8)})
    banks["qpsd"] = directions(2, 16)
    fp = np.tile([0, 1, 2, 1], 8)
    rg = np.array([0, 2, 1, 1, 1, 1, 1, 1]*2)
    fw = np.repeat(fp, 4)
    rw = rg[rgroup]
    assert fw.sum() == 128 and rw.sum() != nreal
    maximum_error = 0.
    prepared_by_arm = []
    for label, value, threshold in (("original_bins", fake, 16),
                                  ("arm_specific_bins", 1.5*fake, 16),
                                  ("no_supported_bin_global_only", fake, 10000)):
        current_stats = dict(stats, minimum_conditional_support=threshold)
        prepared = prepare_proxy(value, real, current_stats, banks, refs)
        prepared_by_arm.append(prepared)
        repeated = refs["loss_statistics"](np.repeat(value, fw, axis=0), np.repeat(real, rw, axis=0), current_stats, banks)
        weighted = reduce_proxy(prepared, fw, rw, threshold)
        assert weighted["conditional_counts"] == repeated["conditional_counts"]
        assert weighted["conditional_bins_used"] == repeated["conditional_bins_used"]
        errors = {key: abs(weighted["components"][key]-repeated["components"][key]) for key in ("wave", "QPSD")}
        errors.update({key: abs(weighted["wave_subcomponents"][key]-repeated["wave_subcomponents"][key])
            for key in ("swd", "coordinate", "short_lag")})
        maximum_error = max(maximum_error, *errors.values())
        assert max(errors.values()) < 1e-12
        tests.append(dict(name="full_proxy_"+label, passed=True, repeated_rows=[int(fw.sum()), int(rw.sum())],
            max_abs_error=max(errors.values()), conditional_bins_used=weighted["conditional_bins_used"]))
    assert prepared_by_arm[0]["conditional"][0][0].sum() == 16
    assert np.count_nonzero(fw[prepared_by_arm[0]["conditional"][0][0]]) == 12
    assert prepared_by_arm[0]["conditional"][0][0].dot(fw) == 16
    tests.append(dict(name="support_uses_repeated_occurrences_not_unique_rows", passed=True,
        original_bin_rows=16, nonzero_unique_rows=12, repeated_count=16, minimum_support=16))
    return dict(schema="n01/stability011-design-synthetic/v1", artificial_only=True,
        real_arrays_decoded=False, checkpoint_decoded=False, science_bootstrap_replicates=0,
        new_model_forwards=0, optimizer_updates=0, tests=tests, passed=len(tests), total=len(tests),
        reference_source_sha256=digest, maximum_full_proxy_literal_repeat_error=maximum_error)


def bench_worker(task):
    seed, count = task
    rng = np.random.default_rng(seed)
    fake_parent = np.repeat(np.arange(512), 4)
    real_group = np.arange(2048) % 419
    values = []
    start = time.perf_counter()
    for _ in range(count):
        fc = rng.multinomial(512, np.full(512, 1./512))
        rc = rng.multinomial(419, np.full(419, 1./419))
        values.append(reduce_cache(BENCH_CACHE, fc[fake_parent], rc[real_group]))
    return dict(seconds=time.perf_counter()-start, synthetic_reductions=count, checksum=float(sum(values)))


def benchmark(workers):
    global BENCH_CACHE
    allowed = sorted(os.sched_getaffinity(0))
    assert 1 <= workers <= len(allowed)
    os.sched_setaffinity(0, allowed[:workers])
    rng = np.random.default_rng(2026101447)
    start = time.perf_counter()
    BENCH_CACHE = cache(np.round(rng.normal(size=(2048, 256)), 3), np.round(rng.normal(size=(2048, 256)), 3))
    preparation = time.perf_counter()-start
    serial = bench_worker((2026101448, 32))
    start = time.perf_counter()
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(bench_worker, [(2026101450+i, 32) for i in range(workers)])
    parallel_wall = time.perf_counter()-start
    reductions = sum(x["synthetic_reductions"] for x in results)
    estimated_core = parallel_wall/reductions*(500*6*(8776+18)/256)
    import resource
    return dict(schema="n01/stability011-design-native-synthetic-benchmark/v1", artificial_only=True,
        real_arrays_decoded=False, checkpoint_decoded=False, science_bootstrap_replicates=0,
        new_model_forwards=0, optimizer_updates=0, platform=platform.platform(), python=sys.version,
        numpy=np.__version__, CPU_workers=workers, CPU_affinity=sorted(os.sched_getaffinity(0)), OMP_threads_per_worker=1,
        original_synthetic_rows=[2048, 2048], columns=256, cache_prepare_seconds=preparation,
        serial=serial, parallel_wall_seconds=parallel_wall, parallel_synthetic_reductions=reductions,
        children=results, parent_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        child_maxrss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        proposed_A_final_3h_500_cache_reduce_seconds_linear_extrapolation=estimated_core,
        runtime_estimate_scope="Synthetic cached reducer only; excludes true-array decode, projection/sort preparation, QA, supervisor/hash/output overhead; six A/final x three-h comparisons; measured fork workers on synthetic arrays.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "benchmark"))
    parser.add_argument("--reference-source")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = check(args.reference_source) if args.mode == "check" else benchmark(args.workers)
    if args.out:
        with Path(args.out).open("x", encoding="utf8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
