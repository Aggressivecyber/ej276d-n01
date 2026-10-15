"""Independent artificial WP4 checks. Default run imports no torch/project init.

All fixtures are deterministic manufactured values, not scientific data.
The lazy --with-torch group is prepared for the final review, not run in parallel
with engineering's current import investigation.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import types

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
TEMP = HERE/'tmp'
TEMP.mkdir(parents=True, exist_ok=True)
for key in ('TEMP', 'TMP', 'TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR'):
    os.environ[key] = str(TEMP)
os.environ.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
sys.dont_write_bytecode = True


def record(path):
    assert path.suffix.lower() in ('.py', '.json')
    b = path.read_bytes()
    return dict(sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def load_numpy_modules():
    # n01.__init__ imports the full historical model. Bypass it for NumPy-only
    # preparation; the two actual source modules keep their relative imports.
    name = '_wp4_independent_numpy'
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT/'src/n01')]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name+'.stats017', ROOT/'src/n01/stats017.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def numpy_checks():
    import numpy as np
    assert 'torch' not in sys.modules, 'Default preparation must not import torch'
    stats = load_numpy_modules()
    kernel = sys.modules[stats.__package__+'.saved_array_kernels017']
    results = []

    def check(name, condition, **detail):
        assert bool(condition), name
        results.append(dict(name=name, passed=True, **detail))

    def rejected(name, callback):
        try:
            callback()
        except ValueError:
            check(name, True)
        else:
            raise AssertionError(name)

    # Independent weighted scalar oracle, using three unequal parent weights.
    values = [1., 3., 5.]
    weights = [1., 2., 3.]
    total = math.fsum(weights)
    alpha = [w/total for w in weights]
    mu = math.fsum(a*x for a, x in zip(alpha, values))
    se = math.sqrt(3/2*math.fsum(a*a*(x-mu)**2 for a, x in zip(alpha, values)))
    actual = stats.weighted017(values, weights)
    check('weighted_HC1_and_ESS_independent_three_parent_oracle',
          math.isclose(actual['mean'], mu, rel_tol=1e-14)
          and math.isclose(actual['SE'], se, rel_tol=1e-14)
          and math.isclose(actual['ESS'], total**2/math.fsum(w*w for w in weights), rel_tol=1e-14),
          actual=actual)
    check('paired_constant_difference_has_zero_SE', stats.weighted017([2., 2., 2.], weights)['SE'] == 0.)

    N = 100
    group = np.ones(N, bool)
    w = np.ones(N)
    support95 = np.arange(N) < 95
    positive = stats.support_cell017(group, w, support95)
    check('original_group_95_count_and_mass_support', positive['status'] == 'supported'
          and positive['original_parents'] == 100 and positive['supported_parents'] == 95
          and positive['supported_weight_fraction'] == .95)
    check('94_percent_support_pending', stats.support_cell017(group, w, np.arange(N) < 94)['status'] == 'pending')
    heavy_missing = np.where(support95, 1., 2.)
    check('95_percent_count_but_insufficient_original_weight_mass_pending',
          stats.support_cell017(group, heavy_missing, support95)['status'] == 'pending')
    check('63_original_parents_pending', stats.support_cell017(group[:63], w[:63], group[:63])['status'] == 'pending')
    skew = w.copy(); skew[0] = 1000.
    check('insufficient_ESS_pending', stats.support_cell017(group, skew, group)['status'] == 'pending',
          fixture='Deliberately illegal population skew tests the generic ESS guard; formal low/high weights remain bounded')

    # Actual nS=48, target nT=64, L=48, E=0 gives PIT=48.5/49.
    # A fixed denominator65 would wrongly put the target in both intervals.
    S = np.ones((N, 2, 32, 9)); T = S.copy()
    S[:, :, 24:, 0] = np.nan
    T[:, :, :, 0] = 2.
    U = np.full((N, 2, 32, 6), .5)
    cov = stats.coverage017(S, T, U)
    check('PIT_actual_48_plus1_rejects_fixed65_false_coverage',
          np.all(cov['n_student_finite'][:, 0] == 48)
          and np.all(cov['n_teacher_defined'][:, 0] == 64)
          and np.all(cov['supported'][:, 0]) and np.all(cov['hits'][:, 0] == 0),
          correct_PIT=48.5/49, incorrect_fixed65_PIT=48.5/65)
    zeroS = S.copy(); zeroS[0, :, :, 0] = np.nan
    zeroT = T.copy(); zeroT[1, :, :, 0] = np.nan
    unsupported = stats.coverage017(zeroS, zeroT, U)
    check('zero_prediction_or_undefined_target_not_uniform_PASS',
          not unsupported['supported'][0, 0] and not unsupported['supported'][1, 0]
          and np.isnan(unsupported['rates'][0, 0]).all() and np.all(unsupported['hits'][0, 0] == -1)
          and unsupported['n_student_finite'][0, 0] == 0 and unsupported['n_teacher_defined'][1, 0] == 0)
    imbalanced = S.copy()
    imbalanced[2, 0, 23, 0] = np.nan
    imbalanced[2, 1, 24, 0] = 1.
    badbank = stats.coverage017(imbalanced, T, U)
    check('23_plus25_combined48_does_not_meet_per_bank24',
          badbank['n_student_finite'][2, 0] == 48 and not badbank['supported'][2, 0])

    # Different valid target counts must not implicitly reweight parents.
    equalS = np.ones((N, 2, 32, 9)); equalT = equalS.copy()
    uniform = np.full((N, 2, 32, 6), .99)
    for i in range(N):
        if i < N//2:
            equalS[i, :, 24:, :] = np.nan
            equalT[i, :, 24:, :] = np.nan
            # 24/48 hits across the original two banks.
            uniform[i, 0, :24, :] = .5
        else:
            # 56/64 hits.
            uniform[i, 0, :, :] = .5
            uniform[i, 1, :24, :] = .5
    rates = stats.coverage017(equalS, equalT, uniform)
    parent_rate = stats.weighted017(rates['rates'][:, 0, 0], w)['mean']
    pooled_rate = rates['hits'][:, 0, 0].sum()/rates['n_teacher_defined'][:, 0].sum()
    check('coverage_parent_average_not_pooled_valid_draws',
          math.isclose(parent_rate, .6875, rel_tol=2e-14) and math.isclose(pooled_rate, 5/7) and parent_rate != pooled_rate,
          parent_weighted=parent_rate, pooled_draw_weighted=float(pooled_rate))

    # Two sets of conditional variances cancel AFTER normalization/aggregation.
    # Large, different parent means make pooled population covariance disagree.
    z = np.tile(np.asarray([-1., 1.]), 16)
    teacher = np.empty((N, 2, 32, 9)); student = np.empty_like(teacher)
    for i in range(N):
        teacher[i] = 10.*i+z[:, None]
        scale = math.sqrt(.5 if i < N//2 else 1.5)
        student[i] = 30.*i+scale*z[:, None]
    cc = stats.covariance_cell017(student, teacher, group, w)
    check('conditional_covariance_normalize_parent_then_group',
          cc['status'] == 'pass' and cc['group_relative_Frobenius'] < 1e-10
          and np.allclose(cc['teacher_normalized_conditional_covariance'], np.ones((9, 9)), atol=1e-12, rtol=0)
          and np.allclose(cc['parent_errors'], .5, atol=1e-10, rtol=0),
          group_error=cc['group_relative_Frobenius'], parent_error_mean=float(np.mean(cc['parent_errors'])))
    pooled_error = abs(np.var(student[:, :, :, 0], ddof=1)/np.var(teacher[:, :, :, 0], ddof=1)-1)
    check('between_parent_scatter_is_not_used_by_covariance_gate',
          pooled_error > 1 and cc['no_between_parent_scatter'], pooled_population_variance_error=float(pooled_error))

    unresolved_teacher = teacher.copy()
    for i in range(N):
        unresolved_teacher[i, 1] = 10.*i+2*z[:, None]
    pending_cov = stats.covariance_cell017(unresolved_teacher, unresolved_teacher, group, w)
    check('identical_student_teacher_still_pending_if_cov_MC_unresolved',
          pending_cov['status'] == 'pending' and pending_cov['reasons'] == ['MC_unresolved']
          and pending_cov['group_relative_Frobenius'] == 0.,
          teacher_split=pending_cov['teacher32_split_resolution'])
    zero = np.zeros_like(teacher)
    zero_ref = stats.covariance_cell017(zero, zero, group, w)
    check('zero_covariance_reference_pending_not_zero_error_PASS', zero_ref['status'] == 'pending')
    unresolved_time = teacher.copy()
    unresolved_time[:, 1, :, 8] += 2.
    pending_time = stats.timing_cell017(unresolved_time, unresolved_time, 8, group, w)
    check('identical_student_teacher_still_pending_if_time_MC_unresolved',
          pending_time['status'] == 'pending' and pending_time['reasons'] == ['MC_unresolved']
          and pending_time['aggregate_abs_error_ns'] == 0.,
          teacher_split_abs=pending_time['teacher32_split_aggregate_abs_ns'])

    # Driver is intentionally not imported yet. This oracle will bind its
    # eventual stage1 decision to actual module-produced pending fixture rows.
    def stage1_oracle(rows_by_seed):
        return set(rows_by_seed) == {0, 1} and all(
            row['status'] == 'pass' for rows in rows_by_seed.values() for row in rows)
    check('MC_unresolved_fixture_requires_closed_stage2_oracle',
          not stage1_oracle({0: [cc], 1: [pending_cov]})
          and not stage1_oracle({0: [cc], 1: [pending_time]}),
          driver_executed=False, driver_binding='Deferred until final ENGINEERING_READY; no conclusion on unfinished driver')

    e = stats.energy_cell017(np.ones(N), np.full(N, 3.), np.ones(N), group, w)
    check('same_parent_energy_delta_HC1_and_fixed_z',
          e['status'] == 'pass' and e['upper95'] == -2. and e['improvement']['SE'] == 0.)
    no_improvement = stats.energy_cell017(np.ones(N), np.ones(N), np.ones(N), group, w)
    check('zero_energy_improvement_not_strict_gate_PASS', no_improvement['status'] == 'fail')
    nonfinite = np.ones(N); nonfinite[0] = np.nan
    rejected('fullwave_energy_nonfinite_is_engineering_error',
             lambda: stats.energy_cell017(nonfinite, np.ones(N), np.ones(N), group, w))

    # Explicitly missing pairwise covariance, retaining a non-PSD matrix.
    x = np.asarray([[[-1., -1., np.nan], [1., 1., np.nan],
                     [-1., np.nan, -1.], [1., np.nan, 1.],
                     [np.nan, -1., 1.], [np.nan, 1., -1.]]])
    moment = kernel.scalar_moments(x)
    expected = np.asarray([[4/3, 2., 2.], [2., 4/3, -2.], [2., -2., 4/3]])
    check('pairwise_centered_covariance_uses_actual_joint_counts_no_PSD_clip',
          np.allclose(moment['sample_covariance'][0], expected, atol=1e-14, rtol=0)
          and np.array_equal(moment['jointly_defined_draws'][0], [[4, 2, 2], [2, 4, 2], [2, 2, 4]])
          and np.linalg.eigvalsh(moment['sample_covariance'][0]).min() < 0)

    ids = np.arange(150, dtype=np.int64)
    fit, test = stats.partition017(ids, heldout=40)
    f2, t2 = stats.partition017(ids[::-1], heldout=40)
    check('parent_ID_split_isolation_full_union_and_order_invariance',
          not set(fit)&set(test) and len(fit) == 110 and len(test) == 40
          and set(ids[test]) == set(ids[::-1][t2]) and set(fit)|set(test) == set(range(150)))
    rejected('duplicate_parent_ID_rejected', lambda: stats.partition017([1, 1, 2], heldout=1))
    groups, weights_by_name, cuts = stats.groups_weights017(np.arange(1., 151.), np.arange(1., 101.), ids[:100])
    check('groups_use_fit_Q_cuts_and_all_parent_partition',
          np.array_equal(cuts, np.quantile(np.arange(1., 151.), [1/3, 2/3], method='linear'))
          and np.all(sum(groups[f'fit_Q_tertile{i}_heldout'] for i in range(3)) == 1)
          and groups['all_heldout'].all())
    check('comparison_weights_positive_bounded_and_no_fitted_population',
          all(np.all((v >= .5)&(v <= 2.)) for v in weights_by_name.values()))
    assert 'torch' not in sys.modules, 'Default NumPy group unexpectedly imported torch'
    return dict(checks=results, count=len(results), torch_imported=False,
                gate_fixture_expectations=dict(cov_MC_unresolved=pending_cov['status'],
                    time_MC_unresolved=pending_time['status'], zero_reference=zero_ref['status'],
                    stage2_expected_open=False, actual_driver_tested=False))


def torch_checks():
    # Deliberately lazy: root asked to avoid a concurrent second torch import.
    # Root later approved this fixture-process-only Windows platform fallback
    # after engineering traced the import stall to platform._wmi_query.
    import platform
    workaround = os.name == 'nt' and hasattr(platform, '_wmi')
    if workaround:
        platform._wmi = None
    import torch
    import numpy as np
    torch.set_num_threads(2)
    before_rng = torch.get_rng_state().clone()
    package_name = '_wp4_independent_torch'
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT/'src/n01')]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name+'.model017', ROOT/'src/n01/model017.py')
    model_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model_module
    spec.loader.exec_module(model_module)
    results = []

    def check(name, condition, **detail):
        assert bool(condition), name
        results.append(dict(name=name, passed=True, **detail))

    def energy_oracle(S, T):
        def fmap(row):
            return [float(x)*16383/16 for x in row]+[math.fsum(float(x) for x in row)*16383/256]
        a, b = [fmap(row) for row in S], [fmap(row) for row in T]
        def dist(x, y): return math.sqrt(math.fsum((u-v)**2 for u, v in zip(x, y)))
        cross = math.fsum(dist(x, y) for x in a for y in b)/(len(a)*len(b))
        self_term = math.fsum(dist(a[i], a[j]) for i in range(len(a)) for j in range(len(a)) if i != j)/(2*len(a)*(len(a)-1))
        return cross-self_term

    S = torch.zeros((4, 256), dtype=torch.float64, requires_grad=True)
    T = torch.zeros((3, 256), dtype=torch.float64)
    with torch.no_grad():
        S[1, 9] = .002; S[2, 20] = -.003; S[3, 9] = .002
        T[0, 9] = .001; T[1, 21] = -.004
    score = model_module.energy_score_U017(S, T, block_rows=2)
    expected = energy_oracle(S.detach().tolist(), T.tolist())
    check('actual_U_score_matches_independent_257D_oracle',
          math.isclose(float(score.detach()), expected, rel_tol=2e-13, abs_tol=1e-12))
    score.backward()
    check('self_diagonal_and_offdiagonal_zero_distance_gradients_finite', torch.isfinite(S.grad).all())
    Z = torch.zeros((2, 2, 256), dtype=torch.float64, requires_grad=True)
    zero = model_module.energy_score_U017(Z, torch.zeros_like(Z), paired=True, block_rows=1).sum()
    zero.backward()
    check('zero_distance_exact_zero_score_and_gradient', float(zero) == 0 and torch.isfinite(Z.grad).all() and not Z.grad.any())

    # Gaussian three-node quadrature reproduces mean/variance/fourth moment;
    # independent Rademacher signs make six deterministic weighted states.
    quadrature = [(-math.sqrt(3), 1/6), (0., 2/3), (math.sqrt(3), 1/6)]
    eta = []; signs = []; probability = []
    for sign in (-1, 1):
        for z, weight in quadrature:
            eta.append([z, .25, -.5, .75, -1., 1.25, -1.5, 1.75])
            signs.append(sign); probability.append(weight/2)
    eta = torch.tensor(eta, dtype=torch.float64)
    mixed = model_module.teacher_shape_coefficients017(eta, teacher='T2', sign=torch.tensor(signs))
    mean = math.fsum(w*float(x) for w, x in zip(probability, mixed[:, 0]))
    var = math.fsum(w*float(x)**2 for w, x in zip(probability, mixed[:, 0]))
    fourth = math.fsum(w*float(x)**4 for w, x in zip(probability, mixed[:, 0]))
    check('actual_T2_function_centered_unit_variance_fourth_moment',
          abs(mean) < 1e-14 and abs(var-1) < 1e-14 and abs(fourth-1.6878) < 1e-13
          and torch.equal(mixed[:, 1:], eta[:, 1:]), mean=mean, variance=var, fourth=fourth)

    # Actual two-parameter forward, proper energy loss, autograd and AdamW.
    # Manufactured continuous crop only; hard ADC/STE driver is tested later.
    shape = torch.zeros(384, dtype=torch.float64)
    shape[110:143] = torch.tensor([1-abs(j-16)/17 for j in range(33)], dtype=torch.float64)*.02
    A0 = torch.stack([shape*s for s in (.4, .8, 1.6, 2.4)])
    B0 = torch.zeros((4, 8, 384), dtype=torch.float64)
    B0[:, 0, 130] = .0004; B0[:, 0, 131] = -.0004
    A = A0.repeat_interleave(2, dim=0); B = B0.repeat_interleave(2, dim=0)
    Q = A[:, 110:366].sum(1)
    zs = torch.zeros((8, 8), dtype=torch.float64); zs[:, 0] = torch.tensor([-.4, .7, -1., .2, .8, -.6, .3, -.9])
    zq = torch.tensor([-1.2, .4, .7, -.8, 1.1, -.6, .9, -1.4], dtype=torch.float64)
    tz = torch.zeros_like(zs); tz[:, 0] = torch.tensor([.3, -.2, .6, -.7, -.5, .8, -.9, .1])
    tq = torch.tensor([.55, -1.6, .95, -.25, 1.3, -.95, .2, -.55], dtype=torch.float64)
    fixed_before = [hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in (A, B, Q)]
    student = model_module.CachedFrozenQ1Charge017()
    check('exactly_two_named_trainable_parameters_and_thin_state',
          list(dict(student.named_parameters())) == ['theta_a', 'theta_b']
          and set(student.state_dict()) == {'theta_a', 'theta_b', 'q_reference'})
    target = model_module.assemble_cached017(A, B, Q, tz, tq,
             torch.tensor(.01*3.7103998046755806), torch.tensor(.0025))['signal_pre'][:, 110:366].reshape(4, 2, 256)

    def loss():
        generated = student(A, B, Q, zs, zq)['signal_pre'][:, 110:366].reshape(4, 2, 256)
        return model_module.energy_score_U017(generated, target, paired=True, block_rows=1).mean()

    optimizer = torch.optim.AdamW(student.trainable_parameters(), lr=.01, weight_decay=0.)
    student.assert_optimizer_membership(optimizer)
    optimizer.zero_grad(set_to_none=True)
    actual_loss = loss(); actual_loss.backward()
    gradients = [float(v.grad) for v in student.trainable_parameters()]
    finite_differences = []
    for parameter in student.trainable_parameters():
        with torch.no_grad():
            original = parameter.clone()
            parameter.copy_(original+1e-5); plus = float(loss())
            parameter.copy_(original-1e-5); minus = float(loss())
            parameter.copy_(original)
        finite_differences.append((plus-minus)/2e-5)
    check('actual_two_theta_gradients_match_finite_differences',
          all(math.isfinite(g) and g != 0 and math.isclose(g, fd, rel_tol=3e-5, abs_tol=1e-8)
              for g, fd in zip(gradients, finite_differences)), gradients=gradients, finite_differences=finite_differences)
    before = [v.detach().clone() for v in student.trainable_parameters()]
    torch.nn.utils.clip_grad_norm_(student.trainable_parameters(), 5.)
    optimizer.step()
    check('actual_AdamW_changes_both_theta_and_no_cache_bytes',
          all(not torch.equal(a, b) for a, b in zip(before, student.trainable_parameters()))
          and fixed_before == [hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in (A, B, Q)])
    out = student(A, B, Q, zs, zq); off = student(A, B, Q, zs, None, enabled=False)
    check('off_exact_M0_bits_without_charge_latent',
          torch.equal(off['signal_pre'].view(torch.int64), out['M0_reference_signal_pre_V'].view(torch.int64)))
    check('shape_not_scaled_by_q', torch.allclose(out['signal_pre']-A*(out['qsig']/Q)[:, None],
          (B*zs[:, :, None]).sum(1), rtol=1e-11, atol=1e-15))
    check('no_global_CPU_RNG_advance', torch.equal(before_rng, torch.get_rng_state()))
    return dict(checks=results, count=len(results), CPU_only=True, actual_scientific_operations=0,
                Windows_fixture_process_platform_wmi_disabled=workaround,
                system_libraries_implementation_and_Linux_native_unchanged=True,
                driver_ADC_STE_streams_stage_gate_resource_memory_tests='not yet executed; wait final ENGINEERING_READY')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-torch', action='store_true')
    parser.add_argument('--output', default='PREPARED_SEMANTIC_CHECKS.json')
    args = parser.parse_args()
    paths = [ROOT/'src/n01/model017.py', ROOT/'src/n01/stats017.py', ROOT/'src/n01/saved_array_kernels017.py',
             ROOT/'receipts/wp4_017_20261015/planning_01/ENGINEERING_PROPOSAL.json',
             ROOT/'receipts/wp4_017_20261015/planning_01/INDEPENDENT_CONTRACT_REVIEW.json']
    sources = {p.relative_to(ROOT).as_posix(): record(p) for p in paths}
    assert sources[paths[3].relative_to(ROOT).as_posix()]['sha256'] == 'b2db64dfd3549f6cd22cb5253a1e374738248012e22554308ec26885d2d42e8e'
    assert sources[paths[4].relative_to(ROOT).as_posix()]['sha256'] == '31a0a106bf033ec7d363abc13811a7f0e6dd87e79d139ea7ad6c867a67927fd7'
    numeric = numpy_checks()
    torch_result = torch_checks() if args.with_torch else dict(executed=False,
        reason='Root requested no second torch import while engineering diagnoses its import stall',
        prepared_groups=['full metric unbiased U oracle', 'zero distances and gradients', 'actual T2 fourth moment',
                         'true two theta AD/AdamW update', 'off exact bytes and q-independent B'])
    assert sources == {p.relative_to(ROOT).as_posix(): record(p) for p in paths}, 'Sources changed during preparation; do not issue final review'
    result = dict(schema='n01/wp4-017-independent-semantic-preparation/v1', preparation_complete=True,
                  overall_SOURCE_REVIEW_issued=False, sources=sources,
                  check_source=dict(path=Path(__file__).relative_to(ROOT).as_posix(), **record(Path(__file__))),
                  NumPy_artificial=numeric, torch_artificial=torch_result,
                  no_conclusion_on_unfinished_driver_controller=True,
                  actual_scientific_work=dict(CP_SOURCE_REAL_DEV_NPZ_NN_AD_G_RNG_GPU_SSH=0),
                  scope='Artificial local preparation for the same final code review; no new protocol/gate/native or scientific acceptance')
    target = HERE/args.output
    assert target.resolve().parent == HERE and not target.exists()
    target.write_bytes((json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps(dict(preparation_complete=True, NumPy_checks=numeric['count'], torch_executed=args.with_torch,
        receipt=str(target), **record(target)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
