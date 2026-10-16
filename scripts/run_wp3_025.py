"""Fixed024 plus2 versus zero: saved hard distributions against exposed REAL TRAIN."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

from wp1_ops import Store, read_json, require, sha, unchanged

RUN_ID = 'wp3_025_20261016'
REMOTE = '/home/rylan/projects/ej276d-n01/' + RUN_ID
HYPOTHESES = ('fast', 'slow', 'common_level')
POINTS = ('zero', 'plus2')
MARGINALS = ('Q_Vns', 'front_Vns', 'tail_Vns', 'PSD_raw', 'PSD_guarded',
             'peak_voltage_V', 'rise10_90_ns', 'FWHM_ns')
RATES = ('Q_nonpositive', 'PSD_undefined', 'PSD_guarded', 'rise_undefined',
         'FWHM_undefined', 'temporal_nonpositive_peak', 'rise_prepeak_censored',
         'FWHM_prepeak_censored', 'FWHM_postpeak_censored')
IDENTITY_METRICS = ('noise', 'baseline_mean_V', 'baseline_RMS_V')
METRICS = ('wave', 'QPSD', 'noise', *MARGINALS, 'baseline_mean_V', 'baseline_RMS_V',
           *(name+'_rate' for name in RATES))
PROJECTION_FINGERPRINT = '27ed58d90114de6bfa5bf30ff5ad251516cdc9bcadfb5eb10e2a6ec45ed11482'
SCIENCE_SOURCES025 = ('scripts/run_wp3_025.py', 'scripts/wp1_ops.py',
                      'scripts/stability011_metrics.py', 'src/n01/saved_array_kernels017.py')


def settings025():
    return dict(schema='n01/wp3-025-config/v1', run_id=RUN_ID, remote_root=REMOTE,
        status='draft', parents=6593, draws=4, banks=2, REAL_rows=21047,
        hypotheses=list(HYPOTHESES), points=list(POINTS), zero_index=0, plus2_index=5,
        plus2_theta=[0, 0, 1, 0, 0, 0], further_selection=False,
        projection_fingerprint=PROJECTION_FINGERPRINT, projection_seed=2026101430,
        stream_columns=16, minimum_conditional_support=16, CPU_threads=12,
        DATASET_numeric_member_whitelist=['real'], REAL_numeric_member_read_once=True,
        original_SOURCE_DEV_waveform_reads=0, REAL_baseline_read=False,
        uniform_original_row_measure=True, original_physical_weights_preserved_not_refitted=True,
        positive_T_checks=90, zero_null_T_checks=42,
        distance_change='D(plus2,REAL)-D(zero,REAL)', positive_T_delta_upper_multiplier=1.,
        all_six_bank_hypothesis_cells_required=True, bank_pooling=False,
        postselection_exposed_TRAIN_development=True, deterministic_A_gain_gate_used=False,
        training_enabled=False, scientific_training_updates=0, automatic_restart=False,
        one_active_scientific_job=True, overwrite_outputs=False, selected_collection_MiB=128,
        resource_limits=dict(CPU_threads=12, job_tree_RSS_GiB=8, address_space_GiB=8,
            wall_seconds=7200, CPU_soft_seconds=21590, CPU_hard_seconds=21600,
            output_MiB=1536, selected_collection_MiB=128),
        preflight=dict(minimum_free_disk_GiB=4, minimum_available_RAM_GiB=10),
        scientific_status='response_match_not_established')


def budget_caps025(p):
    n = p['parents']*p['draws']
    cells = p['banks']*len(p['hypotheses'])
    evaluations = cells*len(p['points'])
    wave = 6*32+2*256+2*4*(255+254+252+248)
    return dict(new_diagnostic_jobs=1, saved_hard_files=12,
        saved_hard_rows_decoded=evaluations*n, REAL_TRAIN_waveform_rows_read=p['REAL_rows'],
        DATASET_numeric_members_read=1, real_waveform_read_operations=1,
        SOURCE_DEV_original_waveform_rows_read=0, wave_and_Q_feature_rows=evaluations*n+p['REAL_rows'],
        point_to_REAL_evaluations=evaluations, candidate_zero_delta_cells=cells,
        wave_scalar_W1_columns_exact=evaluations*wave,
        global_QPSD_scalar_W1_columns_exact=evaluations*18,
        conditional_scalar_W1_columns_cap=evaluations*8,
        marginal_scalar_W1_columns_cap=evaluations*8,
        total_scalar_W1_columns_cap=evaluations*(wave+18+8+8),
        CDF_create_calls_cap=evaluations*(12+32+506+2+8+8),
        CDF_support_scalar_values_cap=evaluations*(wave+18+8+1)*(n+p['REAL_rows']),
        positive_T_noninferiority_deltas=90, zero_null_T_pending_deltas=42,
        euclidean_energy_pair_distances=0, NN_forward_rows=0, checkpoint_loads=0,
        AD_queries=0, G_updates=0, optimizer_steps=0, new_RNG_requests=0,
        new_hard_ADC_argmax_gather_rows=0, new_training_trials=0, parameter_fits=0)


def validate_settings025(p, *, frozen=True, driver_source=None):
    require(not frozen or p['status'] == 'frozen', '025 draft cannot execute')
    require(all(p.get(k) == v for k, v in settings025().items() if k != 'status'),
            '025 fixed settings differ')
    require(p['budget_caps'] == budget_caps025(p), '025 additive budget differs')
    if driver_source is not None:
        require(hashlib.sha256(driver_source).hexdigest() ==
                p['source_files']['scripts/run_wp3_025.py']['sha256'], '025 driver SHA differs')


def gate025(config, expected_sha, output):
    cp = Path(config).resolve()
    require(sha(cp) == expected_sha, '025 config SHA differs')
    p = read_json(cp)
    validate_settings025(p)
    require(Path(output).absolute() == Path(REMOTE)/'runs/primary/artifacts' and
            not Path(output).exists(), '025 exact fresh output required')
    require(set(SCIENCE_SOURCES025) <= set(p['source_files']), '025 actual source closure missing')
    require(all(Path(p['source_files'][name]['path']) == Path(REMOTE)/name for name in SCIENCE_SOURCES025),
            '025 imported source must be owned and bound')
    mf = cp.parent/p['saved_input_manifest']
    require(sha(mf) == p['saved_input_manifest_sha256'], '025 input manifest SHA differs')
    manifest = read_json(mf)
    keys = {f'hard_{point}_b{bank}_{hyp}' for point in POINTS for bank in range(2) for hyp in HYPOTHESES}
    keys.update(('baseline_b0', 'baseline_b1', 'axes024', 'profile024', 'dataset', 'statistics',
                 'tolerances013', 'real_flags013', 'projections011'))
    require(set(p['inputs']) == keys and manifest['run_id'] == RUN_ID and
            manifest['file_count'] == 21 and manifest['total_frozen_input_bytes'] == 198022817 and
            manifest['entries'] == p['inputs'], '025 exact input whitelist differs')
    records = {**p['source_files'], **{'input:'+k: v for k, v in p['inputs'].items()},
        **{'contract:'+k: v for k, v in p['contracts'].items()},
        **{'protected:'+k: v for k, v in p.get('protection_files', {}).items()}}
    roots = [Path(value).resolve() for value in p['allowed_input_roots']]
    checked = {str(cp): expected_sha, str(mf): p['saved_input_manifest_sha256']}
    for label, record in records.items():
        path = Path(record['path'])
        require(not path.is_symlink() and any(path.resolve().is_relative_to(root) for root in roots),
                '025 frozen path escaped: '+label)
        if str(path) not in checked:
            require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
                    '025 immutable bytes differ: '+label)
            checked[str(path)] = record['sha256']
        else:
            require(checked[str(path)] == record['sha256'], '025 conflicting frozen aliases')
    scope = read_json(p['contracts']['prospective_scope']['path'])
    require(scope['schema'] == 'n01/wp3-025-prospective-scope/v1' and scope['run_id'] == RUN_ID and
            scope['inputs'] == p['inputs'] and scope['selection']['plus2_index'] == 5,
            '025 registered scope differs')
    authority = read_json(p['contracts']['user_authorization']['path'])
    require(authority['necessary_additional_NN_AD_G_RNG_computation_authorized'] is True,
            '025 continuing registered-computation authorization absent')
    complete = read_json(p['contracts']['complete013']['path'])
    require(complete['all_REAL_TRAIN_rows'] == 21047 and complete['engineering_checks_passed'] is True and
            complete['immutable_files_sha256'][p['inputs']['dataset']['path']] == p['inputs']['dataset']['sha256'],
            '025 original role-validated REAL dataset chain differs')
    roles = read_json(p['contracts']['data_roles013']['path'])
    require(roles['original_roles_preserved'] is True and roles['REAL']['TRAIN_rows'] == 21047 and
            roles['REAL']['TRAIN_original_logical_groups'] == 422, '025 original REAL role contract differs')
    context = read_json(p['contracts']['projection_context011']['path'])
    require(context['native'] == context['actual'] and context['fingerprint_match'] is True and
            context['actual']['fingerprint'] == PROJECTION_FINGERPRINT and
            context['actual']['seed'] == 2026101430, '025 saved evaluation projection context differs')
    stats = read_json(p['inputs']['statistics']['path'])
    require(stats['statistics_split'] == stats['loss']['fit_split'] == 'TRAIN' and
            stats['prepared_dataset_sha256'] == p['inputs']['dataset']['sha256'] and
            stats['loss']['minimum_conditional_support'] == 16, '025 original frozen TRAIN statistics differ')
    tolerance = read_json(p['inputs']['tolerances013']['path'])
    require(all(set(tolerance['T_by_hypothesis'][h]) == set(METRICS) and
                sum(v is not None and v > 0 for v in tolerance['T_by_hypothesis'][h].values()) == 15
                for h in HYPOTHESES), '025 original013 calibrated metric labels differ')
    profile = read_json(p['inputs']['profile024']['path'])
    require(profile['run_id'] == 'wp3_024_20261016' and
            any(row['theta_index'] == 5 and row['theta_name'] == 'plus2' and row['finite_profile_witness']
                for row in profile['finite_profile_witnesses']), '025 fixed exposed024 selection lineage differs')
    return p, {key: Path(row['path']) for key, row in p['inputs'].items()}, checked, stats['loss'], tolerance


def features025(wave, stats):
    """The original013 observable definitions and011 frozen representations."""
    import numpy as np
    from stability011_metrics import wave_views, q_features, standard
    from saved_array_kernels017 import morphology
    y = np.asarray(wave, np.float64)
    q, tail = y.sum(1), y[:, 46:].sum(1)
    psd = np.full(len(y), np.nan)
    np.divide(tail, q, out=psd, where=q != 0)
    denominator = np.where(q < 0, -1., 1.)*np.maximum(np.abs(q), stats['q_floor'])
    observations = dict(Q_Vns=q, front_Vns=q-tail, tail_Vns=tail, PSD_raw=psd,
                        PSD_guarded=tail/denominator, **morphology(y))
    flags = dict(Q_nonpositive=q <= 0, PSD_undefined=q == 0, PSD_guarded=np.abs(q) < stats['q_floor'],
                 **{name: value for name, value in observations.items() if value.dtype.kind == 'b'})
    views = wave_views(y, stats['peak_floor'])
    for name in views:
        views[name] = standard(views[name], stats['views'][name])
    raw, _ = q_features(y, stats)
    return dict(views=views, q_coordinate=standard(raw, stats['qpsd']),
        q_bins=np.searchsorted(np.asarray(stats['charge_bin_boundaries'], np.float64), raw[:, 0], side='left'),
        observables=observations, flags=flags, rows=len(y))


def stream_score025(fake, real, banks, store, label):
    """Exact original grouped CDF terms, retaining only one group's sixteen-column cache."""
    import numpy as np
    from stability011_metrics import CDFCache, VIEWS, COORDINATE_VIEWS, LAGS
    descriptors = [('swd', name, None) for name in VIEWS]
    descriptors += [('coordinate', name, None) for name in COORDINATE_VIEWS]
    descriptors += [('short_lag', name, lag) for name in COORDINATE_VIEWS for lag in LAGS]
    descriptors += [('q_coordinate', '', None), ('q_projected', '', None)]
    descriptors += [('conditional', '', index) for index in range(8)]
    descriptors += [('marginal', name, None) for name in MARGINALS]
    groups, blocks, conditional, defined = [], [], [], {}
    for kind, name, index in descriptors:
        group = kind+(':'+name if name else '')+(':'+str(index) if index is not None else '')
        if kind == 'swd':
            f, r = fake['views'][name]@banks['wave:'+name], real['views'][name]@banks['wave:'+name]
        elif kind == 'coordinate':
            f, r = fake['views'][name], real['views'][name]
        elif kind == 'short_lag':
            direction = banks[f'lag:{name}:{index}']
            f = (fake['views'][name][:, :-index, None]*direction[0]+
                 fake['views'][name][:, index:, None]*direction[1]).reshape(fake['rows'], -1)
            r = (real['views'][name][:, :-index, None]*direction[0]+
                 real['views'][name][:, index:, None]*direction[1]).reshape(real['rows'], -1)
        elif kind == 'q_coordinate':
            f, r = fake['q_coordinate'], real['q_coordinate']
        elif kind == 'q_projected':
            f, r = fake['q_coordinate']@banks['qpsd'], real['q_coordinate']@banks['qpsd']
        elif kind == 'conditional':
            fm, rm = fake['q_bins'] == index, real['q_bins'] == index
            nf, nr = int(fm.sum()), int(rm.sum())
            supported = min(nf, nr) >= 16
            conditional.append(dict(bin=index, fake_count=nf, real_count=nr, supported=supported))
            if not supported:
                continue
            f, r = fake['q_coordinate'][fm, 1:2], real['q_coordinate'][rm, 1:2]
        else:
            x, y = fake['observables'][name], real['observables'][name]
            fm, rm = np.isfinite(x), np.isfinite(y)
            defined[name] = dict(fake_rows=len(x), real_rows=len(y), fake_defined_rows=int(fm.sum()),
                real_defined_rows=int(rm.sum()), domain='explicit_defined_subpopulation',
                undefined_rows_retained_in_flags=True)
            if not fm.any() or not rm.any():
                defined[name].update(value=None, reason='no_defined_population_on_one_side')
                continue
            f, r = x[fm, None], y[rm, None]
        category = ('wave_scalar_W1_columns_exact' if kind in ('swd', 'coordinate', 'short_lag') else
                    'global_QPSD_scalar_W1_columns_exact' if kind in ('q_coordinate', 'q_projected') else
                    'conditional_scalar_W1_columns_cap' if kind == 'conditional' else 'marginal_scalar_W1_columns_cap')
        fw, rw = np.ones(len(f), np.int64), np.ones(len(r), np.int64)
        weighted_sum = 0.
        for start in range(0, f.shape[1], 16):
            stop = min(start+16, f.shape[1])
            width = stop-start
            value = store.call('exact_saved_support_CDF',
                {category: width, 'total_scalar_W1_columns_cap': width, 'CDF_create_calls_cap': 1,
                 'CDF_support_scalar_values_cap': width*(len(f)+len(r))},
                lambda: CDFCache.create(f[:, start:stop], r[:, start:stop]).distance(fw, rw, column_chunk=16),
                cell=label, group=group, first_column=start, columns=width, fake_rows=len(f), real_rows=len(r))
            weighted_sum += value*width
            blocks.append(dict(group=group, first_column=start, columns=width, mean_W1=value,
                               fake_rows=len(f), real_rows=len(r)))
            if store.counts['CDF_create_calls_cap'] == 1:
                store.phase = 'first_actual_CDF_complete'
                store.json('FIRST_ACTUAL_CDF_PROOF.json', dict(store.base(), schema='n01/wp3-025-first-CDF/v1',
                    cell=label, group=group, actual_CDF_columns=width, actual_support_values=width*(len(f)+len(r)),
                    exact_float64_CDF_completed=True, first_block_mean_W1=value))
                store.status()
        mean = weighted_sum/f.shape[1]
        groups.append(dict(kind=kind, group=group, columns=f.shape[1], value=mean))
        if kind == 'marginal':
            defined[name]['value'] = mean
        del f, r, fw, rw
    parts = {kind: float(np.mean([row['value'] for row in groups if row['kind'] == kind]))
             for kind in ('swd', 'coordinate', 'short_lag')}
    global_q = .5*sum(row['value'] for row in groups if row['kind'] in ('q_coordinate', 'q_projected'))
    terms = [row['value'] for row in groups if row['kind'] == 'conditional']
    conditional_mean = float(np.mean(terms)) if terms else None
    values = dict(wave=.5*parts['swd']+.25*parts['coordinate']+.25*parts['short_lag'],
                  QPSD=.5*(global_q+conditional_mean) if terms else global_q,
                  **{name: defined[name]['value'] for name in MARGINALS})
    values.update({name: None for name in IDENTITY_METRICS})
    values.update({name+'_rate': abs(float(fake['flags'][name].mean())-float(real['flags'][name].mean()))
                   for name in RATES})
    return dict(values=values, wave_subcomponents=parts, QPSD_global=global_q,
        QPSD_conditional_mean=conditional_mean, conditional_counts=conditional,
        conditional_bins_used=len(terms), conditional_coverage_complete=len(terms) == 8,
        no_supported_bin_behavior='global_term_only_descriptive_full_QPSD_pending',
        defined_population=defined, groups=groups, blocks=blocks, fake_rows=fake['rows'], real_rows=real['rows'],
        all_rows_retained=True, uniform_original_row_measure=True,
        absolute_noise_and_baseline_distances='unknown_REAL_prefix_not_read',
        fake_flag_counts={name: dict(count=int(v.sum()), denominator=len(v), fraction=float(v.mean()))
                          for name, v in fake['flags'].items()},
        real_flag_counts={name: dict(count=int(v.sum()), denominator=len(v), fraction=float(v.mean()))
                          for name, v in real['flags'].items()})


def run025(config, config_sha256, output):
    p, paths, checked, stats, tolerance = gate025(config, config_sha256, output)
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (21590, 21600))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '12'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:12])
    require(len(os.sched_getaffinity(0)) == 12, '025 requires twelve owned CPUs')
    import numpy as np
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src/n01'))
    store = Store(output, p, checked)
    store.status()
    try:
        store.json('STARTED.json', dict(store.base(), schema='n01/wp3-025-started/v1',
            configuration_sha256=config_sha256, postselection_exposed_TRAIN_development=True,
            fixed_zero_index=0, fixed_plus2_index=5, no_response_acceptance=True))
        store.call('one_registered_distribution_question', dict(new_diagnostic_jobs=1), lambda: None)
        with np.load(paths['projections011'], allow_pickle=False) as z:
            original_banks = {name: z[name] for name in z.files}
        digest = hashlib.sha256()
        for name in sorted(original_banks):
            value = original_banks[name]
            require(value.dtype == np.float32 and np.isfinite(value).all(), '025 saved projection dtype differs')
            digest.update(name.encode('utf8'))
            digest.update(str(tuple(value.shape)).encode('ascii'))
            digest.update(b'torch.float32')
            digest.update(np.ascontiguousarray(value).tobytes())
        require(digest.hexdigest() == PROJECTION_FINGERPRINT, '025 numeric saved projection fingerprint differs')
        banks = {name: value.astype(np.float64) for name, value in original_banks.items()}
        del original_banks
        with np.load(paths['axes024'], allow_pickle=False) as z:
            axes = {name: z[name] for name in z.files}
        n = p['parents']*p['draws']
        require(axes['source_ids'].shape == axes['original_physical_weights'].shape == (6593,) and
                np.array_equal(axes['source_parent_row'], np.repeat(np.arange(6593), 4)) and
                np.array_equal(axes['draw_index'], np.tile(np.arange(4), 6593)) and
                axes['theta_names'][0] == 'zero' and axes['theta_names'][5] == 'plus2' and
                np.array_equal(axes['theta_points'][5], [0, 0, 1, 0, 0, 0]), '025 saved024 labels/axes differ')
        store.npz('SAVED024_AXES.npz', **axes)
        with np.load(paths['real_flags013'], allow_pickle=False) as z:
            real_metadata = {name: z[name] for name in z.files}
        require(all(v.shape == (21047,) for v in real_metadata.values()) and
                len(np.unique(real_metadata['real_ids'])) == 21047 and
                len(np.unique(real_metadata['real_groups'])) == 422, '025 original REAL identities/groups differ')
        store.phase = 'read_REAL_TRAIN_once'
        with np.load(paths['dataset'], allow_pickle=False) as z:
            real_wave = store.call('only_DATASET_real_numeric_member',
                dict(REAL_TRAIN_waveform_rows_read=21047, DATASET_numeric_members_read=1, real_waveform_read_operations=1),
                lambda: z['real'])
        require(real_wave.shape == (21047, 256) and real_wave.dtype.kind == 'f' and np.isfinite(real_wave).all(),
                '025 REAL member shape/type/finite differs')
        real = store.call('original_REAL_features_once', dict(wave_and_Q_feature_rows=21047),
                          lambda: features025(real_wave, stats))
        del real_wave
        store.npz('REAL_OBSERVABLES.npz', **real_metadata,
            **{'observable__'+k: v for k, v in real['observables'].items()},
            **{'derived_flag__'+k: v for k, v in real['flags'].items()},
            q_coordinate=real['q_coordinate'], output_Q_bin=real['q_bins'])
        del real_metadata
        store.json('INIT_PROOF.json', dict(store.base(), schema='n01/wp3-025-init-proof/v1',
            native_initialized=True, REAL_numeric_members_read=['real'], REAL_rows=21047,
            REAL_role='original_exposed_TRAIN', REAL_role_basis='raw_dataset_SHA+role_validated013_COMPLETE+DATA_ROLES',
            REAL_original_logical_groups=422, physical_independent_REAL_acquisitions=1,
            projection_fingerprint=digest.hexdigest(), projection_factory_called=False,
            saved024_zero_index=0, saved024_plus2_index=5, NN_CP_AD_G_RNG_ADC_increment=0))
        reports, baseline_means = {}, {}
        for bank in range(2):
            with np.load(paths[f'baseline_b{bank}'], allow_pickle=False) as z:
                prefix = z['baseline100']
            require(prefix.shape == (n, 100) and prefix.dtype == np.float64 and np.isfinite(prefix).all(),
                    '025 actual saved prefix100 differs')
            prefix_mean = prefix.mean(1)
            store.npz(f'bank{bank}_PREFIX_SUMMARY.npz', baseline_mean_V=prefix_mean,
                baseline_RMS_V=np.sqrt(np.square(prefix-prefix_mean[:, None]).mean(1)))
            del prefix, prefix_mean
            for hyp in HYPOTHESES:
                for point in POINTS:
                    label = f'{point}_b{bank}_{hyp}'
                    store.phase = 'decode_'+label
                    with np.load(paths['hard_'+label], allow_pickle=False) as z:
                        codes = z['ADC_crop_codes']
                        mu, peak = z['baseline_mean_V'], z['peak_index']
                        stored_flags = {name[6:]: z[name] for name in z.files if name.startswith('flag__')}
                        require(codes.dtype == np.uint16 and codes.shape == (n, 256) and
                                mu.shape == peak.shape == (n,) and mu.dtype == np.float64 and
                                peak.dtype == np.int64 and np.isfinite(mu).all() and
                                all(v.dtype == np.bool_ and v.shape == (n,) for v in stored_flags.values()),
                                '025 saved hard codec/flag boundary differs')
                        xor = z['IEEE754_reconstruction_XOR_uint64'] if 'IEEE754_reconstruction_XOR_uint64' in z.files else None
                        if xor is not None:
                            require(xor.dtype == np.uint64 and xor.shape == codes.shape, '025 codec XOR schema differs')
                        wave = store.call('lossless_saved024_hard_decode',
                            dict(saved_hard_files=1, saved_hard_rows_decoded=n),
                            lambda: ((codes.astype(np.float64)/16383-.5-mu[:, None]).view(np.uint64)^xor).view(np.float64)
                                if xor is not None else codes.astype(np.float64)/16383-.5-mu[:, None], cell=label)
                        del xor
                    if bank not in baseline_means:
                        baseline_means[bank] = mu.copy()
                    require(np.array_equal(mu.view(np.uint64), baseline_means[bank].view(np.uint64)),
                            '025 common024 bank baseline mean identity differs across points/hypotheses')
                    fake = store.call('original_hard_features_once', dict(wave_and_Q_feature_rows=n),
                                      lambda: features025(wave, stats), cell=label)
                    del codes, wave
                    store.phase = 'score_'+label
                    score = stream_score025(fake, real, banks, store, label)
                    store.call('complete_point_to_REAL_score', dict(point_to_REAL_evaluations=1), lambda: None, cell=label)
                    score.update(point=point, theta_index=0 if point == 'zero' else 5, bank=bank, hypothesis=hyp)
                    reports[(point, bank, hyp)] = score
                    store.json(label+'_DISTANCES.json', dict(schema='n01/wp3-025-distances/v1', run_id=RUN_ID, **score))
                    store.npz(label+'_OBSERVABLES.npz', saved_peak_index=peak, saved_baseline_mean_V=mu,
                        **{'observable__'+k: v for k, v in fake['observables'].items()},
                        **{'derived_flag__'+k: v for k, v in fake['flags'].items()},
                        **{'stored_flag__'+k: v for k, v in stored_flags.items()},
                        q_coordinate=fake['q_coordinate'], output_Q_bin=fake['q_bins'])
                    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 8*1024**3,
                            '025 peak RSS cap exceeded')
                    require(sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file()) <= 1536*1024**2,
                            '025 output cap exceeded')
                    del fake, stored_flags, peak, mu
        store.json('FULL_PREFIX_IDENTITY.json', dict(schema='n01/wp3-025-prefix-identity/v1', run_id=RUN_ID,
            identity_source='immutable024 same bank base+prefix-zero original015 adapter across all theta and hypotheses',
            one_actual_saved_full_prefix100_file_per_bank=True,
            all12_saved_hard_baseline_means_bit_identical_within_bank=True,
            new_independent_full_prefix_replay=False, metrics=list(IDENTITY_METRICS), delta=0.,
            absolute_noise_distance=None, REAL_prefix_read=False))
        del real, baseline_means
        deltas = []
        for bank in range(2):
            for hyp in HYPOTHESES:
                zero, candidate = reports[('zero', bank, hyp)], reports[('plus2', bank, hyp)]
                checks = []
                for metric in METRICS:
                    a, c = zero['values'][metric], candidate['values'][metric]
                    identity = metric in IDENTITY_METRICS
                    delta = 0. if identity else None if a is None or c is None else c-a
                    T = tolerance['T_by_hypothesis'][hyp][metric]
                    positive = T is not None and T > 0
                    complete = metric != 'QPSD' or (zero['conditional_coverage_complete'] and
                                                              candidate['conditional_coverage_complete'])
                    if not positive:
                        status = 'pending_not_calibrated'
                    elif delta is None:
                        status = 'pending_undefined_distance'
                    elif not complete:
                        status = 'pending_incomplete_QPSD_coverage'
                    else:
                        status = 'pass' if delta <= T else 'fail'
                    checks.append(dict(metric=metric, D_zero_REAL=a, D_plus2_REAL=c, delta=delta,
                        T=T, original013_status=tolerance['status'][hyp][metric], positive_calibrated_T=positive,
                        delta_over_T=None if not positive or delta is None else delta/T,
                        D_zero_over_T=None if not positive or a is None else a/T,
                        D_plus2_over_T=None if not positive or c is None else c/T,
                        status=status, relative_method='deterministic_identity' if identity else 'actual_same_REAL_distance_change',
                        coverage_complete=complete))
                positive_checks = [row for row in checks if row['positive_calibrated_T']]
                finite_status = ('fail' if any(row['status'] == 'fail' for row in positive_checks) else
                                 'pending' if any(row['status'] != 'pass' for row in positive_checks) else 'pass')
                deltas.append(dict(bank=bank, hypothesis=hyp, metrics=checks,
                    finite_positive_T_noninferiority_status=finite_status,
                    uncalibrated_metrics_remain_pending=True, response_acceptance=False))
                store.call('registered_candidate_zero_delta_cell', dict(candidate_zero_delta_cells=1,
                    positive_T_noninferiority_deltas=15, zero_null_T_pending_deltas=7), lambda: None,
                    bank=bank, hypothesis=hyp)
        bank_means = []
        for hyp in HYPOTHESES:
            for metric in METRICS:
                rows = [row for cell in deltas if cell['hypothesis'] == hyp for row in cell['metrics'] if row['metric'] == metric]
                bank_means.append(dict(hypothesis=hyp, metric=metric,
                    arithmetic_mean_of_two_bank_deltas=None if any(row['delta'] is None for row in rows) else
                        sum(row['delta'] for row in rows)/2,
                    pooled_bank_empirical_measure=False, decision_uses_each_bank=True))
        finite_status = ('fail' if any(cell['finite_positive_T_noninferiority_status'] == 'fail' for cell in deltas) else
                         'pending' if any(cell['finite_positive_T_noninferiority_status'] == 'pending' for cell in deltas) else 'pass')
        summary = dict(schema='n01/wp3-025-distribution-comparison/v1', run_id=RUN_ID,
            fixed_zero_index=0, fixed_plus2_index=5, delta_cells=deltas, separate_two_bank_means=bank_means,
            finite_positive_T_noninferiority_status=finite_status, positive_T_checks=90,
            uncalibrated_zero_null_T_pending_checks=42,
            pass_count=sum(row['status'] == 'pass' for cell in deltas for row in cell['metrics']),
            fail_count=sum(row['status'] == 'fail' for cell in deltas for row in cell['metrics']),
            pending_count=sum(row['status'].startswith('pending') for cell in deltas for row in cell['metrics']),
            noise_relative_delta_identity_only=True, absolute_noise_acceptance=False,
            exposed_REAL_TRAIN_postselection_development=True, independent_confirmation=False,
            original_deterministic_A_two_percent_gate_evaluated=False,
            all_original_rows_and_physical_weight_metadata_preserved=True,
            response_acceptance=False, WP5_WP6_complete=False, training_authorized_by_result=False)
        store.json('COMPARISON.json', summary)
        fixed = ('new_diagnostic_jobs', 'saved_hard_files', 'saved_hard_rows_decoded',
                 'REAL_TRAIN_waveform_rows_read', 'DATASET_numeric_members_read', 'real_waveform_read_operations',
                 'wave_and_Q_feature_rows', 'point_to_REAL_evaluations', 'candidate_zero_delta_cells',
                 'wave_scalar_W1_columns_exact', 'global_QPSD_scalar_W1_columns_exact',
                 'positive_T_noninferiority_deltas', 'zero_null_T_pending_deltas')
        require(all(store.counts[k] == p['budget_caps'][k] for k in fixed) and
                store.counts == store.attempted and not store.inflight, '025 exact completed counts differ')
        unchanged(checked)
        store.json('RESOURCE_USAGE.json', dict(run_id=RUN_ID,
            elapsed_seconds=time.monotonic()-store.started, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            output_bytes_before_final_records=sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file()),
            CPU_affinity=sorted(os.sched_getaffinity(0)), limits=p['resource_limits']))
        inventory = {str(q.relative_to(store.root)): dict(bytes=q.stat().st_size, sha256=sha(q))
            for q in sorted(store.root.rglob('*')) if q.is_file() and q.name not in ('status.json', 'operations.jsonl')}
        store.json('ARTIFACT_INVENTORY.json', dict(schema='n01/wp3-025-artifacts/v1', run_id=RUN_ID, files=inventory,
            selected_collection_MiB=128, all_new_outputs_are_small=True, large_saved024_hard_stays_remote=True))
        store.phase = 'complete_fixed_distribution_question'
        store.json('COMPLETE.json', dict(store.base(), schema='n01/wp3-025-complete/v1', native_driver_completed=True,
            artifact_inventory_sha256=sha(store.root/'ARTIFACT_INVENTORY.json'),
            finite_positive_T_noninferiority_status=finite_status, response_acceptance=False))
        store.status()
    except BaseException as error:
        store.error = repr(error)
        store.phase = 'failed'
        store.json('FAILED.json', dict(store.base(), schema='n01/wp3-025-failed/v1'))
        store.status()
        raise
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    return run025(args.config, args.config_sha256, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
