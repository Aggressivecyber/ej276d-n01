"""Fixed saved-charge variance decomposition; no model, fit, RNG or hard replay."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

RUN_ID = 'wp5_021_20261016'
GROUPS = ('all_heldout', 'fit_Q_tertile0_heldout', 'fit_Q_tertile1_heldout', 'fit_Q_tertile2_heldout')
WEIGHTS = ('nominal', 'low', 'high')
TRIALS = ('T1_paired_seed0', 'T1_paired_seed1', 'T1_unpaired_seed0', 'T1_unpaired_seed1')
RESPONSES = ('known_T1',) + TRIALS
CAPS = dict(population_group_weight_cases=12, variance_decompositions=60,
    response_parent_contributions=39570, sensitivity_parent_contributions=7914,
    two_by_two_eigendecompositions=24, numeric_NPZ_decodes=1,
    NN_forward_rows=0, new_AD_queries=0, G_updates=0, new_RNG_draws=0, checkpoint_loads=0,
    new_original_SOURCE_REAL_DEV_waveform_reads=0, parameter_fits=0, optimizer_steps=0, new_pair_distances=0)
RESOURCES = dict(CPU_threads_max=1, wall_seconds_max=120, RSS_GiB_max=2, output_MiB_max=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    config_path = Path(args.config).resolve()
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != args.config_sha256:
        raise ValueError('021 frozen config SHA differs')
    p = json.loads(config_path.read_text(encoding='utf-8'))
    if not (p['schema'] == 'n01/wp5-021-config/v1' and p['run_id'] == RUN_ID and p['status'] == 'frozen'
            and tuple(p['groups']) == GROUPS and tuple(p['comparison_weights']) == WEIGHTS
            and tuple(p['responses']) == RESPONSES and p['budget_caps'] == CAPS and p['resources'] == RESOURCES
            and p['closure_relative_tolerance'] == 1e-12):
        raise ValueError('021 exact frozen scope/budget required')
    output = Path(args.output).resolve()
    if os.name != 'nt' or config_path.drive.upper() not in ('E:', 'D:') or output.drive.upper() not in ('E:', 'D:'):
        raise ValueError('021 registered local Windows E/D execution required')
    if output != Path(p['output_directory']).resolve() or os.path.lexists(output):
        raise ValueError('021 requires the exact fresh output; preserve previous results')
    source_root = Path(__file__).resolve().parents[1]
    required_sources = ('scripts/run_wp5_021.py', 'docs/WP5_021_CONFIGURATION.md',
        'receipts/wp5_021_20261016/implementation_01/input_sources/INPUT_SOURCES.json')
    if not set(required_sources) <= set(p['source_files']):
        raise ValueError('021 owned source/contract closure missing')
    if any(Path(p['source_files'][name]['path']).resolve() != source_root/name for name in required_sources):
        raise ValueError('021 bound source is not the actual owned source')
    checked = {str(config_path): args.config_sha256}
    records = {**p['source_files'], **{'input:'+k: v for k, v in p['inputs'].items()},
        **{'contract:'+k: v for k, v in p['contracts'].items()}}
    for label, record in records.items():
        path = Path(record['path']).resolve()
        if path.drive.upper() not in ('E:', 'D:') or path.stat().st_size != record['bytes']:
            raise ValueError('021 bound input/source bytes or storage differs: '+label)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record['sha256']:
            raise ValueError('021 bound input/source SHA differs: '+label)
        checked[str(path)] = digest
    input_manifest = json.loads(Path(p['input_manifest']).read_text(encoding='utf-8'))
    if input_manifest['inputs'] != p['inputs'] or input_manifest['file_count'] != 6 or input_manifest['total_bytes'] != 6711911:
        raise ValueError('021 exact six saved inputs differ')
    final020 = json.loads(Path(p['contracts']['final020']['path']).read_text(encoding='utf-8'))
    if not (final020['schema'] == 'n01/wp4-020-final-result/v1' and final020['run_id'] == 'wp4_020_20261016'
            and final020['native_exit_code'] == 0 and final020['qualified'] is False
            and final020['required_cells'] == 2544 and final020['independent_review_pass'] is True):
        raise ValueError('021 requires the independently closed actual020 nonpass')
    authorization = json.loads(Path(p['contracts']['user_authorization']['path']).read_text(encoding='utf-8'))
    if not (authorization['persistent_for_N01_master_plan'] is True and authorization['resume_authorized'] is True
            and authorization['necessary_additional_NN_AD_G_RNG_computation_authorized'] is True):
        raise ValueError('021 persistent direct human authorization missing')
    scope = json.loads(Path(p['contracts']['prospective_scope']['path']).read_text(encoding='utf-8'))
    if scope['schema'] != 'n01/wp5-021-prospective-analysis-scope/v1' or scope['run_id'] != RUN_ID:
        raise ValueError('021 prospectively registered task identity differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    cache_directory = output/'runtime_cache'
    cache_directory.mkdir()
    for name in ('TEMP', 'TMP', 'TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'MPLCONFIGDIR'):
        os.environ[name] = str(cache_directory)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    if not sys.dont_write_bytecode:
        raise ValueError('021 must use python -B')
    # Native OS boundary only; no psutil installation or controller is needed.
    process = wintypes.HANDLE(-1)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    affinity_query = kernel32.GetProcessAffinityMask
    affinity_query.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    affinity_query.restype = wintypes.BOOL
    affinity_set = kernel32.SetProcessAffinityMask
    affinity_set.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
    affinity_set.restype = wintypes.BOOL
    allowed_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not affinity_query(process, ctypes.byref(allowed_mask), ctypes.byref(system_mask)):
        raise OSError(ctypes.get_last_error(), '021 owned process affinity query failed')
    initial_affinity_mask = allowed_mask.value
    selected_cpu_mask = initial_affinity_mask & -initial_affinity_mask
    if not affinity_set(process, selected_cpu_mask):
        raise OSError(ctypes.get_last_error(), '021 owned process single-CPU binding failed')
    (output/'STARTED.json').write_text(json.dumps(dict(schema='n01/wp5-021-started/v1', run_id=RUN_ID,
        phase='all_frozen_bytes_verified_before_numeric_decode', configuration_sha256=args.config_sha256,
        immutable_inputs=checked, initial_affinity_mask=initial_affinity_mask, selected_cpu_mask=selected_cpu_mask), indent=2)+'\n', encoding='utf-8')
    import numpy as np

    original020 = json.loads(Path(p['inputs']['teacher_contract020']['path']).read_text(encoding='utf-8'))
    if original020['run_id'] != 'wp4_020_20261016' or original020['qref'] != 3.7103998046755806:
        raise ValueError('021 original teacher contract differs')
    teacher_a = original020['teachers']['T1']['a_fraction_qref']*original020['qref']
    teacher_b = original020['teachers']['T1']['b']
    coefficients = {'known_T1': dict(a_Vns=teacher_a, b=teacher_b, source='original020 T1 contract')}
    for trial in TRIALS:
        report = json.loads(Path(p['inputs'][trial]['path']).read_text(encoding='utf-8'))
        if report['trial'] != trial or len(report['cells']) != 1272:
            raise ValueError('021 fixed endpoint source identity differs: '+trial)
        item = report['parameter_recovery_secondary']
        if not (np.isfinite(item['a']) and np.isfinite(item['b']) and item['a'] > 0 and item['b'] > 0
                and item['q_reference_Vns'] == original020['qref']):
            raise ValueError('021 fixed endpoint coefficients invalid: '+trial)
        coefficients[trial] = dict(a_Vns=item['a'], b=item['b'], source=p['inputs'][trial])
    axes_path = Path(p['inputs']['parent_axes']['path'])
    with np.load(axes_path, allow_pickle=False) as archive:
        ids, fit, test = archive['source_ids'], archive['fit_rows'], archive['test_rows']
        full_Q, original_physical_weights = archive['fixed_A_Q_Vns'], archive['original_physical_weights']
        groups = {name: archive['group_'+name] for name in GROUPS}
        weights = {name: archive['comparison_weight_'+name] for name in WEIGHTS}
    if not (ids.shape == (6593,) and full_Q.shape == (6593,) and fit.shape == (5274,) and test.shape == (1319,)
            and original_physical_weights.shape == (6593,) and len(np.unique(ids)) == 6593
            and np.array_equal(np.sort(np.concatenate((fit, test))), np.arange(6593))):
        raise ValueError('021 original parent identity/role/physical-weight axes differ')
    Q = full_Q[test].astype(np.float64, copy=False)
    if not (np.isfinite(Q).all() and (Q > 0).all()
            and all(g.shape == (1319,) and g.dtype.kind == 'b' for g in groups.values())
            and groups['all_heldout'].all()
            and np.array_equal(sum(groups[name].astype(np.int8) for name in GROUPS[1:]), np.ones(1319, np.int8))
            and all(w.shape == (1319,) and np.isfinite(w).all() and (w > 0).all() for w in weights.values())):
        raise ValueError('021 existing heldout group/comparison-weight axes invalid')
    shutil.copyfile(axes_path, output/'PRESERVED_PARENT_AXES.npz')
    cases, sensitivity = [], []
    nominal = {}
    response_parent_contributions = sensitivity_parent_contributions = eigendecompositions = 0
    closure_max = dict(within_relative=0., total_relative=0., second_moment_scaled=0., unit_Gram_absolute=0.)
    for group in GROUPS:
        mask = groups[group]
        q = Q[mask]
        q2 = q*q
        for weight in WEIGHTS:
            w = weights[weight][mask].astype(np.float64, copy=False)
            alpha = w/w.sum()
            count = len(q)
            EQ, EQ2 = float(alpha@q), float(alpha@q2)
            between = float(alpha@((q-EQ)**2))
            ESS = float(w.sum()**2/(w@w))
            columns = np.column_stack((teacher_a*q, teacher_b*q2))
            raw_Gram = (columns*alpha[:, None]).T@columns
            unit_Gram = raw_Gram/np.sqrt(np.outer(np.diag(raw_Gram), np.diag(raw_Gram)))
            raw_eigenvalues = np.linalg.eigvalsh(raw_Gram)
            unit_eigenvalues = np.linalg.eigvalsh(unit_Gram)
            eigendecompositions += 2
            sensitivity_parent_contributions += count
            cosine = float(unit_Gram[0, 1])
            unit_error = max(float(np.max(np.abs(np.diag(unit_Gram)-1.))),
                float(np.max(np.abs(unit_eigenvalues-np.array([1.-cosine, 1.+cosine])))))
            closure_max['unit_Gram_absolute'] = max(closure_max['unit_Gram_absolute'], unit_error)
            if unit_error > p['closure_relative_tolerance']:
                raise ArithmeticError('021 unit-column Gram algebra did not close')
            spectra = {}
            for name, matrix, eigenvalues in (('raw', raw_Gram, raw_eigenvalues), ('unit_column', unit_Gram, unit_eigenvalues)):
                rank_cutoff = 2*np.finfo(np.float64).eps*float(eigenvalues[-1])
                rank = int(np.sum(eigenvalues > rank_cutoff))
                spectra[name] = dict(Gram=matrix.tolist(), eigenvalues=eigenvalues.tolist(),
                    lambda_min_over_max=float(eigenvalues[0]/eigenvalues[-1]),
                    numerical_rank=rank, numerical_rank_cutoff=rank_cutoff,
                    condition_number=float(eigenvalues[-1]/eigenvalues[0]) if rank == 2 else None,
                    status='full_rank_at_float64_resolution' if rank == 2 else 'rank_deficient_at_float64_resolution')
            sensitivity.append(dict(group=group, comparison_weights=weight, original_parents=count,
                comparison_weight_sum=float(w.sum()), ESS=ESS, columns=['a_T1*Q', 'b_T1*Q^2'],
                raw_columns_unit='Vns^2; derivative of canonical variance wrt fractional parameter changes',
                centered=False, weighted_column_cosine=cosine, not_Pearson_correlation=True,
                spectra=spectra, actual_training_Hessian=False, new_identifiability_gate=False))
            for response in RESPONSES:
                a, b = coefficients[response]['a_Vns'], coefficients[response]['b']
                a_term, b_term = a*EQ, b*EQ2
                within = float(alpha@(a*q+b*q2))
                total = between+within
                total_centered = float(alpha@((q-EQ)**2+a*q+b*q2))
                second_moment = float(alpha@(q2+a*q+b*q2))
                second_total = second_moment-EQ**2
                errors = dict(within_relative=abs(within-a_term-b_term)/within,
                    total_relative=abs(total-total_centered)/total,
                    second_moment_scaled=abs(total-second_total)/max(second_moment, EQ**2, total))
                if max(errors.values()) > p['closure_relative_tolerance']:
                    raise ArithmeticError('021 float64 variance decomposition did not close')
                for name, value in errors.items():
                    closure_max[name] = max(closure_max[name], value)
                row = dict(group=group, comparison_weights=weight, response=response, original_parents=count,
                    comparison_weight_sum=float(w.sum()), supported_parents=count, removed_parents=0, ESS=ESS,
                    a_Vns=a, b=b, E_Q_Vns=EQ, E_Q_squared_Vns2=EQ2,
                    between_parent_variance_Vns2=between, within_a_Vns2=a_term, within_b_Vns2=b_term,
                    within_parent_variance_Vns2=within, total_canonical_variance_Vns2=total,
                    between_fraction=between/total, kernel_fraction=within/total,
                    a_fraction_of_kernel=a_term/within, b_fraction_of_kernel=b_term/within,
                    delta_a_E_Q_vs_teacher_Vns2=(a-teacher_a)*EQ,
                    delta_b_E_Q_squared_vs_teacher_Vns2=(b-teacher_b)*EQ2,
                    within_difference_vs_teacher_Vns2=within-teacher_a*EQ-teacher_b*EQ2,
                    algebra_closure=errors)
                if weight == 'nominal':
                    nominal[group, response] = row
                else:
                    relative = {}
                    for field in ('E_Q_Vns', 'between_parent_variance_Vns2', 'within_parent_variance_Vns2', 'total_canonical_variance_Vns2'):
                        reference = nominal[group, response][field]
                        relative[field] = dict(absolute_change=row[field]-reference,
                            relative_change=(row[field]-reference)/reference if reference != 0 else None,
                            status='defined' if reference != 0 else 'undefined_zero_nominal_denominator')
                    row['relative_to_same_group_fixed_kernel_nominal'] = relative
                cases.append(row)
                response_parent_contributions += count
    counts = dict(CAPS, population_group_weight_cases=len(sensitivity), variance_decompositions=len(cases),
        response_parent_contributions=response_parent_contributions, sensitivity_parent_contributions=sensitivity_parent_contributions,
        two_by_two_eigendecompositions=eigendecompositions)
    if counts != CAPS:
        raise ArithmeticError('021 exact registered60/12/24 accounting differs')
    if not affinity_query(process, ctypes.byref(allowed_mask), ctypes.byref(system_mask)) or allowed_mask.value != selected_cpu_mask:
        raise OSError(ctypes.get_last_error(), '021 single-CPU process binding changed')
    memory_query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    memory_query.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
    memory_query.restype = wintypes.BOOL
    pointer_bytes = ctypes.sizeof(ctypes.c_size_t)
    memory = ctypes.create_string_buffer(8+9*pointer_bytes)
    ctypes.c_uint32.from_buffer(memory).value = ctypes.sizeof(memory)
    if not memory_query(process, ctypes.byref(memory), ctypes.sizeof(memory)):
        raise OSError(ctypes.get_last_error(), '021 native current/peak working-set query failed')
    resources = dict(elapsed_seconds=time.monotonic()-started,
        peak_working_set_bytes=ctypes.c_size_t.from_buffer(memory, 8).value,
        current_working_set_bytes=ctypes.c_size_t.from_buffer(memory, 8+pointer_bytes).value,
        initial_affinity_mask=initial_affinity_mask, selected_cpu_mask=selected_cpu_mask,
        actual_affinity_mask=allowed_mask.value, allowed_CPU_count=allowed_mask.value.bit_count(), BLAS_OMP_threads=1,
        Windows_process_working_set_used_for_RSS_cap=True,
        snapshot_stage='after_analysis_before_final_result_serialization')
    if resources['elapsed_seconds'] > 120 or resources['peak_working_set_bytes'] > 2*1024**3:
        raise RuntimeError('021 registered wall/RSS cap exceeded')
    result = dict(schema='n01/wp5-021-canonical-analysis/v1', run_id=RUN_ID, native_analysis_complete=True,
        question='finite population versus within-parent canonical signal-charge variance and local parameter-column compensation',
        coefficients=coefficients, variance_cases=cases, sensitivity_cases=sensitivity,
        exact_completed_counts=counts, closure_relative_tolerance=1e-12, closure_max=closure_max,
        preserved_parent_axes=dict(file='PRESERVED_PARENT_AXES.npz', sha256=p['inputs']['parent_axes']['sha256'],
            source_parents=6593, fit_parents=5274, heldout_parents=1319,
            original_physical_weights_shape=list(original_physical_weights.shape),
            original_physical_weights_dtype=str(original_physical_weights.dtype), original_physical_weights_retained=True,
            comparison_weights_do_not_replace_or_multiply_original_weights=True),
        resources=resources, scope='canonical signal charge before005 electronics and hard observation;1319 heldout only',
        original020_qualification=False, original_gates_unchanged=True, new_performance_gate=False,
        hard256_or_REAL_identifiability_inferred=False, optimizer_gradient_or_Hessian_inferred=False,
        WP5_full_confirmation=False, scientific_status='response_match_not_established')
    (output/'ANALYSIS.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if hashlib.sha256((output/'PRESERVED_PARENT_AXES.npz').read_bytes()).hexdigest() != p['inputs']['parent_axes']['sha256']:
        raise ArithmeticError('021 original axes/weights preservation differs')
    for path, digest in checked.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('021 frozen input/source changed during analysis: '+path)
    files = {path.name: dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(output.iterdir()) if path.is_file()}
    complete = dict(schema='n01/wp5-021-complete/v1', run_id=RUN_ID, native_status='COMPLETE',
        configuration_sha256=args.config_sha256, completed_counts=counts, artifacts=files,
        independent_numerical_readback_pending=True, original020_nonpass_preserved=True,
        further_computation_started=False, scientific_acceptance=False,
        elapsed_seconds=time.monotonic()-started, resources=resources)
    (output/'COMPLETE.json').write_text(json.dumps(complete, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    output_bytes = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    if output_bytes > 8*1024**2 or time.monotonic()-started > 120:
        raise RuntimeError('021 final output/wall cap exceeded')
    print(json.dumps(dict(run_id=RUN_ID, native_status='COMPLETE', variance_cases=len(cases),
        sensitivity_cases=len(sensitivity), eigendecompositions=eigendecompositions, output_bytes=output_bytes)), flush=True)
