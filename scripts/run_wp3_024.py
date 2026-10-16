"""Fixed thirteen-point original015 mean profile over frozen017 M0 carriers."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from wp1_ops import Store, read_json, require, sha, unchanged
from run_wp4_017 import seed017

RUN_ID = 'wp3_024_20261016'
REMOTE = '/home/rylan/projects/ej276d-n01/' + RUN_ID
HYPOTHESES = ('fast', 'slow', 'common_level')
M0_CP_SHA = '27ad34f2f7fa0c2c78f881b649ecd5ad664c91ff4bbe2bb08c109f3536f0915c'
M0_TENSOR_SHA = '0b1bdf84f58d9d9d7fb5fe29d572955611bfc870a972c98fa196f78ed7b13fdf'
SCIENCE_SOURCES024 = tuple('scripts/' + n + '.py' for n in (
    'run_wp3_024', 'run_wp4_017', 'run_wp1_013', 'wp1_ops')) + tuple(
    'src/n01/' + n + '.py' for n in (
        '__init__', 'data', 'data008', 'diagnose_mean_random009',
        'electronics_v3', 'evaluate_window008', 'extension006',
        'losses', 'losses008', 'losses008_v2', 'model', 'model008', 'model008_fix01',
        'model010', 'model014', 'model015', 'model016', 'model017',
        'observation008', 'response008', 'train', 'train_shape010'))


def settings024():
    points = [[0.] * 6]
    names = ['zero']
    for axis in range(6):
        for sign in (1., -1.):
            point = [0.] * 6
            point[axis] = sign
            points.append(point)
            names.append(('plus' if sign > 0 else 'minus') + str(axis))
    return dict(schema='n01/wp3-024-config/v1', run_id=RUN_ID, remote_root=REMOTE,
        status='draft', parents=6593, banks=2, draws=4, chunk_rows=128,
        CPU_threads=12, hypotheses=list(HYPOTHESES), theta_points=points, theta_names=names,
        q_reference=3.7103998046755806, q_floor=0.0005319378913383611,
        mean_adapter=dict(parameters=6, Gaussian_sigma_ns=2., Gaussian_length=13,
            causal_tau_ns=16., causal_length=64, coefficient_limit=.1,
            RMS_trust_fraction=.1, epsilon_V=1e-12),
        seed_banks={str(bank): {stream: seed017(dict(run_id=RUN_ID,
            purpose='fixed_mean_axis_profile', bank=bank, stream=stream))
            for stream in ('shape', 'noise005', 'extension284')} for bank in (0, 1)},
        seed_namespace=dict(run_id=RUN_ID, purpose='fixed_mean_axis_profile',
            bank=[0, 1], stream=['shape', 'noise005', 'extension284']),
        M0_checkpoint_metadata_only=dict(bytes=485683, sha256=M0_CP_SHA,
            head_tensor_sha256=M0_TENSOR_SHA),
        original_SOURCE_REAL_DEV_waveform_reads=0, charge_enabled=False,
        new_A_M0_forward_rows=0, training_enabled=False, trainable_parameters=0,
        automatic_restart=False, one_active_scientific_job=True, overwrite_outputs=False,
        witness_raw_RMS_ratio_max=.90, witness_shape_RMS_strictly_improves=True,
        witness_absolute_Q_bias_does_not_increase=True, all_six_bank_hypothesis_cells_required=True,
        selected_collection_MiB=128,
        resource_limits=dict(GPU_GiB=10, job_tree_RSS_GiB=8, wall_seconds=7200,
            output_MiB=3072, poll_seconds=10, TERM_grace_seconds=20),
        preflight=dict(minimum_free_disk_GiB=40, minimum_available_RAM_GiB=10,
            GPU_must_be_empty=True),
        scientific_status='response_match_not_established')


def budget_caps024(p):
    n, b, r, h, t = p['parents'], p['banks'], p['draws'], len(p['hypotheses']), len(p['theta_points'])
    rows = n * b * r
    return dict(new_diagnostic_jobs=1, theta_points=t, population_mean_cells=t*b*h,
        baseline_paired_comparisons=(t-1)*b*h, training_trials=0,
        new_deployable_candidates=0, parameter_fits=0, G_updates=0, AD_queries=0,
        optimizer_steps=0, A_M0_checkpoint_loads=0, A_M0_original_NN_forward_rows=0,
        noise005_checkpoint_loads=1, checkpoint_loads_total=1, saved_A_B_Q_mmap_files=3,
        cached_M0_shape_assembly_rows=rows, noise005_NN_rows=rows,
        three_noise_extension_rows=h*rows, normal_shape_scalar_requests=8*rows,
        normal_noise005_scalar_requests=101*rows, normal_extension284_scalar_requests=284*rows,
        new_normal_scalar_requests_total=393*rows, new_sign_or_PIT_uniform_requests=0,
        stored_full384_base_carrier_rows=h*rows,
        stored_full384_base_carrier_raw_bytes=h*rows*384*8,
        mean_adapter_parent_theta_rows=n*t, hard_ADC_rows=t*h*rows,
        first_argmax_gather_rows=t*h*rows, SOURCE_REAL_DEV_original_waveform_reads=0,
        DEV_rows=0, new_energy_or_O19_pair_distances=0)


def validate_settings024(p, *, frozen=True, driver_source=None):
    reference = settings024()
    require(not frozen or p['status'] == 'frozen', '024 draft cannot execute')
    require(all(p.get(k) == v for k, v in reference.items() if k != 'status'), '024 fixed settings differ')
    require(p['budget_caps'] == budget_caps024(p), '024 exact additive budget differs')
    if driver_source is not None:
        import hashlib
        require(hashlib.sha256(driver_source).hexdigest() == p['source_files']['scripts/run_wp3_024.py']['sha256'],
            '024 driver source SHA differs')


def gate024(config, expected_sha, output):
    cp = Path(config).resolve()
    require(sha(cp) == expected_sha, '024 config SHA differs')
    p = read_json(cp)
    validate_settings024(p)
    require(Path(output).absolute() == Path(REMOTE)/'runs/primary/artifacts' and not Path(output).exists(),
        '024 exact fresh output required')
    require(set(SCIENCE_SOURCES024) <= set(p['source_files']), '024 scientific source closure missing')
    require(all(Path(p['source_files'][name]['path']) == Path(REMOTE)/name for name in SCIENCE_SOURCES024),
        '024 imported source not owned/bound')
    manifest_path = cp.parent/p['saved_input_manifest']
    require(sha(manifest_path) == p['saved_input_manifest_sha256'], '024 saved input manifest differs')
    manifest = read_json(manifest_path)
    require(manifest['file_count'] == 13 and manifest['entries'] == p['inputs'], '024 thirteen inputs differ')
    records = {**p['source_files'], **{'input:'+k: v for k, v in p['inputs'].items()},
        **{'contract:'+k: v for k, v in p['contracts'].items()},
        **{'protected:'+k: v for k, v in p['protection_files'].items()}}
    checked = {str(cp): expected_sha, str(manifest_path): sha(manifest_path)}
    roots = [Path(value).resolve() for value in p['allowed_input_roots']]
    for label, record in records.items():
        path = Path(record['path'])
        require(not path.is_symlink() and any(path.resolve().is_relative_to(root) for root in roots),
            '024 bound path escaped: '+label)
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
            '024 immutable bytes differ: '+label)
        checked[str(path)] = record['sha256']
    paths = {key: Path(row['path']) for key, row in p['inputs'].items()}
    paths.update({key: Path(p['contracts'][key]['path']) for key in ('init017', 'source_audit013')})
    old017, old015 = read_json(paths['config017']), read_json(paths['config015'])
    cache, init = read_json(paths['cache_manifest']), read_json(paths['init017'])
    head = old017['inputs']['M0_head_seed0']
    require(head['sha256'] == M0_CP_SHA and head['bytes'] == 485683 and
        head['path'].endswith('/wp3_015_20261015/runs/primary/artifacts/seed0_M0/final.pt'),
        '024 original017 M0 raw checkpoint binding differs')
    require(cache['M0_head_tensor_sha256'] == M0_TENSOR_SHA and cache['parents'] == 6593 and
        init['one_pass_A_decoded_M0_cache_complete'] is True and
        init['cache_manifest_sha256'] == p['inputs']['cache_manifest']['sha256'] and
        init['immutable_files_sha256'][str(paths['config017'])] == p['inputs']['config017']['sha256'] and
        init['immutable_files_sha256'][head['path']] == M0_CP_SHA,
        '024 saved M0/cache/config identity chain differs; no checkpoint fallback')
    for key, name in (('cache_A', 'A.npy'), ('cache_B', 'B.npy'), ('cache_Q', 'Q.npy'),
                      ('parent_split', 'PARENT_SPLIT.npz'), ('physical_weights', 'ORIGINAL_PHYSICAL_WEIGHTS.npz')):
        require(cache['cache_files'][name] == {k: p['inputs'][key][k] for k in ('bytes', 'sha256')},
            '024 original cache member binding differs: '+key)
    audit = read_json(paths['source_audit013'])
    require(audit['original_count_weights'] == 'uniform ones', '024 original count weights differ')
    for field, value in (('prefix_mean_V', 0.), ('peak_index', 126.), ('crop_start', 110.), ('crop_stop', 366.)):
        row = audit['measurements']['TRAIN'][field]
        require(row['rows'] == row['defined_rows'] == 6593 and row['minimum'] == row['maximum'] == value,
            '024 SOURCE control recovery audit differs: '+field)
    adapter = old015['mean_adapter']
    require(all(adapter[k] == v for k, v in p['mean_adapter'].items()) and
        adapter['control_Q_slice'] == [110, 366] and adapter['control_tail_slice'] == [156, 366],
        '024 original015 adapter definition differs')
    scope = read_json(p['contracts']['prospective_scope']['path'])
    require(scope['schema'] == 'n01/wp3-024-prospective-scope/v1' and scope['run_id'] == RUN_ID,
        '024 registered scope differs')
    authority = read_json(p['contracts']['user_authorization']['path'])
    require(authority['necessary_additional_NN_AD_G_RNG_computation_authorized'] is True,
        '024 persistent registered-computation authorization absent')
    return p, paths, checked, init['frozen_state_sha256']


def mean_adapter024(a, source_q, source_tail, theta, q_reference, q_floor):
    """Original015 arithmetic after exact SOURCE Q/tail recovery; no SOURCE template."""
    import torch
    import torch.nn.functional as F
    from n01.model008 import canonical_charge008, _onset
    a = a.double()
    rms_a = a.square().mean(1).sqrt()
    sign = torch.where(source_q < 0, -torch.ones_like(source_q), torch.ones_like(source_q))
    cq = torch.tanh(torch.log1p(source_q.abs()/q_reference)-math.log(2.))
    cs = torch.tanh(2*(source_tail/(sign*source_q.abs().clamp_min(q_floor))-.5))
    coefficients = .1*torch.tanh(torch.stack((theta[0]+theta[1]*cq,
        theta[2]+theta[3]*cs, theta[4]+theta[5]*cs), 1))
    t = torch.arange(-6, 7, device=a.device, dtype=torch.float64)
    gaussian = torch.exp(-.5*(t/2.)**2)
    gaussian = gaussian/gaussian.sum()
    causal = torch.exp(-torch.arange(64, device=a.device, dtype=torch.float64)/16.)
    causal = causal/causal.sum()
    smooth = F.conv1d(F.pad(a[:, None], (6, 6)), gaussian[None, None])[:, 0]
    tail = F.conv1d(F.pad(a[:, None], (63, 0)), causal.flip(0)[None, None])[:, 0]
    baseline_t = torch.arange(384, dtype=torch.float64)
    gate = _onset(baseline_t).reshape(1, 384)
    projection = gate*torch.exp(-.5*((baseline_t-180.)/65.)**2)
    projection = projection/canonical_charge008(projection)[:, None]
    gate, projection = gate.to(a.device), projection.to(a.device)
    bases = []
    for difference in (smooth-a, tail-a):
        raw = difference*gate
        projected = raw-canonical_charge008(raw)[:, None]*projection
        bases.append(projected*(rms_a/projected.square().mean(1).sqrt().clamp_min(1e-12))[:, None])
    raw = coefficients[:, 0, None]*a+coefficients[:, 1, None]*bases[0]+coefficients[:, 2, None]*bases[1]
    raw_rms = torch.linalg.vector_norm(raw, dim=1)/math.sqrt(384.)
    scale = torch.minimum(torch.ones_like(raw_rms), .1*rms_a/raw_rms.clamp_min(1e-12))
    delta = raw*scale[:, None]
    rms = torch.linalg.vector_norm(delta, dim=1)/math.sqrt(384.)
    relative = rms/rms_a.clamp_min(1e-12)
    require(bool(torch.isfinite(delta).all() and (delta[:, :100] == 0).all() and
        (rms <= .1*rms_a+1e-14).all()), '024 original015 prefix/finite/trust identity failed')
    return dict(delta_mean_pre_V=delta, mean_adapter_regular=relative.square().mean(),
        mean_RMS_V=rms, mean_RMS_relative_A=relative, mean_trust_scale=scale,
        mean_canonical_charge=canonical_charge008(delta), mean_coefficients=coefficients,
        mean_controls=torch.stack((cq, cs), 1),
        mean_zeroQ_basis_charge=torch.stack([canonical_charge008(b) for b in bases], 1))


def run024(config, config_sha256, output, device='cuda:0'):
    p, paths, checked, frozen = gate024(config, config_sha256, output)
    import numpy as np
    import torch
    import resource
    from n01.evaluate_window008 import _memory_limit
    from n01.model017 import assemble_cached017, deterministic_loading017
    from n01.model015 import _ExactZeroMeanAddition015
    from n01.observation008 import observe_generated_carrier
    from n01.response008 import fixed_noises, make_extension, state_digest
    from n01.train_shape010 import Streams010
    from run_wp4_017 import Runtime017
    from run_wp1_013 import lossless_window_codes

    torch.set_num_threads(p['CPU_threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _memory_limit(dict(gpu_memory_limit_GiB=p['resource_limits']['GPU_GiB']), device)
    store = Store(output, p, checked)
    store.status()
    try:
        store.json('STARTED.json', dict(store.base(), schema='n01/wp3-024-started/v1',
            configuration_sha256=config_sha256, control_only_original015_parameterization=True,
            old015_M1_replay=False, old_M1_protected_shape_head='C1', profile_shape_head='015 seed0 M0'))
        store.call('one_registered_fixed_profile', dict(new_diagnostic_jobs=1), lambda: None)
        cache = {}
        for name in ('A', 'B', 'Q'):
            cache[name] = store.call('read_only_cache_'+name, dict(saved_A_B_Q_mmap_files=1),
                lambda name=name: np.load(paths['cache_'+name], mmap_mode='r', allow_pickle=False))
        n, r = p['parents'], p['draws']
        require(cache['A'].shape == (n, 384) and cache['B'].shape == (n, 8, 384) and
            cache['Q'].shape == (n,) and all(v.dtype == np.float64 and not v.flags.writeable and
            np.isfinite(v).all() for v in cache.values()) and (cache['Q'] > 0).all() and
            (cache['A'][:, :100] == 0).all() and (cache['B'][:, :, :100] == 0).all(),
            '024 frozen cache schema/finite/prefix differs')
        with np.load(paths['parent_split'], allow_pickle=False) as z:
            ids, q_source_saved = z['source_ids'], z['q_source']
            source_peak, source_prefix = z['source_peak_index'], z['source_baseline_mean_V']
        with np.load(paths['physical_weights'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['source_ids'], ids)
            weights = z['weights']
        require(len(ids) == len(np.unique(ids)) == n and np.array_equal(weights, np.ones(n)) and
            (source_peak == 126).all() and (source_prefix == 0).all(), '024 original parent/count/control axes differ')
        with np.load(paths['source_information'], allow_pickle=False) as z:
            source_ids, roles = z['source_id'], z['role']
            source_q, source_tail = z['canonical_Q_Vns'], z['tail_clean_window256_Vns']
        lookup = {str(value): i for i, value in enumerate(source_ids)}
        require(len(lookup) == len(source_ids), '024 duplicate saved SOURCE information ID')
        ix = np.asarray([lookup[str(value)] for value in ids], np.int64)
        require(np.array_equal(source_ids[ix], ids) and (roles[ix] == 'TRAIN').all(), '024 SOURCE ID/TRAIN join differs')
        source_q, source_tail = source_q[ix], source_tail[ix]
        require(np.isfinite(source_q).all() and np.isfinite(source_tail).all(), '024 nonfinite SOURCE controls')
        np.testing.assert_array_equal(source_q, q_source_saved)
        with np.load(paths['REAL_mean_summary015'], allow_pickle=False) as z:
            real_mean = z['REAL_mean_wave256_V']
            real_variance = z['REAL_cluster_variance_of_mean_V2']
        provenance = read_json(paths['REAL_mean_provenance015'])
        require(real_mean.shape == real_variance.shape == (256,) and np.isfinite(real_mean).all() and
            np.isfinite(real_variance).all() and (real_variance >= 0).all() and
            provenance['REAL_rows'] == 21047 and provenance['REAL_logical_groups'] == 422 and
            provenance['group_is_independent_acquisition'] is False and provenance['conditional_mean_identified'] is False,
            '024 saved REAL TRAIN population mean/diagonal provenance differs')
        statistics = read_json(paths['statistics'])
        require(statistics['loss']['q_reference'] == p['q_reference'] and
            statistics['loss']['q_floor'] == p['q_floor'], '024 original015 control scales differ')
        noise = store.call('only_original005_checkpoint', dict(noise005_checkpoint_loads=1, checkpoint_loads_total=1),
            lambda: torch.load(paths['noise005'], map_location='cpu', weights_only=False))
        require(noise['schema_version'] == 5 and noise['update'] == 5000 and
            noise['model_type'] == 'ObservedBaselineNoiseV3' and noise['engineering'] is False, '024 original005 metadata differs')
        with deterministic_loading017():
            extension = make_extension(noise, device)
        del noise
        require(state_digest(extension.model005.state_dict()) == frozen['noise005'] and
            state_digest(extension.state_dict()) == frozen['extension'], '024 original005 frozen tensors differ')
        rt = Runtime017(p, cache, extension, statistics, frozen, store, device)
        row = np.repeat(np.arange(n), r)
        draw = np.tile(np.arange(r), n)
        store.npz('AXES_AND_TARGET.npz', source_ids=ids, original_physical_weights=weights,
            source_parent_row=row, draw_index=draw, source_q_Vns=source_q,
            source_tail_Vns=source_tail, cached_A_Q_Vns=cache['Q'],
            REAL_mean_wave256_V=real_mean, REAL_cluster_variance_of_mean_V2=real_variance,
            theta_points=np.asarray(p['theta_points']), theta_names=np.asarray(p['theta_names']))
        store.json('INIT_PROOF.json', dict(store.base(), schema='n01/wp3-024-init-proof/v1',
            native_initialized=True, noise005_checkpoint_loads=1, A_M0_checkpoint_loads=0,
            A_M0_original_NN_forward_rows=0, charge_enabled=False, original_M0_head_tensor_sha256=M0_TENSOR_SHA,
            original_M0_checkpoint_raw_sha256=M0_CP_SHA, cache_manifest_sha256=p['inputs']['cache_manifest']['sha256'],
            parent_count=n, legal_SOURCE_TRAIN_controls_from_saved013=True,
            SOURCE_control_Q_is_not_cached_A_Q=True, REAL_rows=21047, REAL_logical_groups=422,
            real_diagonal_only=True, conditional_REAL_truth=False, seed_banks=p['seed_banks'],
            protected_noise005_state_sha256=frozen['noise005'], protected_extension_sha256=frozen['extension'],
            control_only_original015_parameterization=True, exact_old015_M1_replay=False))
        states = {}
        store.phase = 'generate_two_fixed_full384_banks'
        with torch.no_grad():
            for bank in range(2):
                seeds = p['seed_banks'][str(bank)]
                streams = Streams010(seeds['shape'], seeds['noise005'], seeds['extension284'])
                states.update({f'bank{bank}_initial_{k}': v.numpy() for k, v in streams.state_dict().items()})
                bases = {h: np.lib.format.open_memmap(store.root/f'bank{bank}_{h}_BASE384.npy',
                    mode='w+', dtype=np.float64, shape=(n*r, 384)) for h in HYPOTHESES}
                for start in range(0, n*r, p['chunk_rows']):
                    stop = min(start+p['chunk_rows'], n*r)
                    size, selected = stop-start, row[start:stop]
                    z, nz, epsilon = rt.call('Streams010_three_explicit_float32_streams',
                        dict(normal_shape_scalar_requests=8*size, normal_noise005_scalar_requests=101*size,
                            normal_extension284_scalar_requests=284*size, new_normal_scalar_requests_total=393*size),
                        lambda: streams.draw(size, rt.device), bank=bank, row_start=start)
                    a, b, q = (torch.tensor(np.array(cache[k][selected], copy=True), device=rt.device,
                        dtype=torch.float64) for k in ('A', 'B', 'Q'))
                    signal = rt.call('original017_cached_M0_charge_off', dict(cached_M0_shape_assembly_rows=size),
                        lambda: assemble_cached017(a, b, q, z, None, None, None, enabled=False)['signal_pre'])
                    noises = rt.call('original005_and_shared_three_extensions',
                        dict(noise005_NN_rows=size, three_noise_extension_rows=3*size),
                        lambda: fixed_noises(extension, nz, epsilon))
                    values = {h: (signal+noises[h]).cpu().numpy() for h in HYPOTHESES}
                    store.call('persist_actual_full384_bases', dict(stored_full384_base_carrier_rows=3*size,
                        stored_full384_base_carrier_raw_bytes=3*size*384*8),
                        lambda: [bases[h].__setitem__(slice(start, stop), values[h]) for h in HYPOTHESES])
                for value in bases.values():
                    value.flush()
                states.update({f'bank{bank}_final_{k}': v.numpy() for k, v in streams.state_dict().items()})
                del bases, values, a, b, q, signal, noises, z, nz, epsilon
                if bank == 0:
                    store.json('FIRST_ACTUAL_BASE_PROOF.json', dict(store.base(), schema='n01/wp3-024-first-base-proof/v1',
                        bank=bank, actual005_forward_rows=n*r, actual_stored_base_rows=3*n*r,
                        all_three_full384_bases_flushed=True,
                        base_files={h: dict(bytes=(store.root/f'bank0_{h}_BASE384.npy').stat().st_size,
                            sha256=sha(store.root/f'bank0_{h}_BASE384.npy')) for h in HYPOTHESES},
                        G_updates=0, AD_queries=0, no_017_teacher_stream_reuse=True))
            store.npz('RANDOM_STATES.npz', **states)
            extension.assert_frozen()
            baseline_waves, baseline_peaks = {}, {}
            reports, means, trust = [], [], []
            for ti, (theta_name, point) in enumerate(zip(p['theta_names'], p['theta_points'])):
                store.phase = 'theta_'+theta_name
                theta = torch.tensor(point, dtype=torch.float64, device=rt.device)
                delta = np.empty((n, 384), np.float64)
                adapter_parts = []
                for start in range(0, n, p['chunk_rows']):
                    stop = min(start+p['chunk_rows'], n)
                    a = torch.tensor(np.array(cache['A'][start:stop], copy=True), dtype=torch.float64, device=rt.device)
                    qs = torch.tensor(source_q[start:stop], dtype=torch.float64, device=rt.device)
                    ts = torch.tensor(source_tail[start:stop], dtype=torch.float64, device=rt.device)
                    adapter = rt.call('original015_control_only_mean_adapter',
                        dict(mean_adapter_parent_theta_rows=stop-start),
                        lambda: mean_adapter024(a, qs, ts, theta, p['q_reference'], p['q_floor']), theta_index=ti)
                    delta[start:stop] = adapter['delta_mean_pre_V'].cpu().numpy()
                    adapter_parts.append({k: v.cpu().numpy() for k, v in adapter.items()
                        if k not in ('delta_mean_pre_V', 'mean_adapter_regular')})
                control = {k: np.concatenate([part[k] for part in adapter_parts]) for k in adapter_parts[0]}
                store.npz(f'theta{ti:02d}_TRUST.npz', **control)
                trust.append(dict(theta_index=ti, theta_name=theta_name, theta=point,
                    max_RMS_relative_A=float(control['mean_RMS_relative_A'].max()),
                    minimum_trust_scale=float(control['mean_trust_scale'].min()),
                    clipped_parents=int((control['mean_trust_scale'] < 1).sum()),
                    maximum_zeroQ_basis_charge_abs_Vns=float(np.abs(control['mean_zeroQ_basis_charge']).max())))
                for bank in range(2):
                    for h in HYPOTHESES:
                        base = np.load(store.root/f'bank{bank}_{h}_BASE384.npy', mmap_mode='r', allow_pickle=False)
                        waves = np.empty((n*r, 256), np.float64)
                        peak = np.empty(n*r, np.int64)
                        flags = {}
                        coded_parts = []
                        prefix_parts = []
                        for start in range(0, n*r, p['chunk_rows']):
                            stop = min(start+p['chunk_rows'], n*r)
                            carrier = torch.tensor(np.array(base[start:stop], copy=True), dtype=torch.float64, device=rt.device)
                            d = torch.tensor(delta[row[start:stop]], dtype=torch.float64, device=rt.device)
                            carrier = _ExactZeroMeanAddition015.apply(carrier, d)
                            obs = rt.call('original_hard_ADC_first_argmax_gather',
                                dict(hard_ADC_rows=stop-start, first_argmax_gather_rows=stop-start),
                                lambda: observe_generated_carrier(carrier, ste=False, q_floor=p['q_floor'],
                                    return_carrier=True, return_indices=True), theta_index=ti, bank=bank, hypothesis=h)
                            fields = {k: obs[k].cpu().numpy() for k in ('window_V', 'adc_carrier', 'baseline100', 'peak_index')}
                            fields['diagnostic__baseline_mean_V'] = obs['baseline_mean_V'].cpu().numpy()
                            waves[start:stop], peak[start:stop] = fields['window_V'], fields['peak_index']
                            coded_parts.append(lossless_window_codes(fields))
                            for key, value in obs['flags'].items():
                                if key not in flags:
                                    flags[key] = np.empty(n*r, bool)
                                flags[key][start:stop] = value.cpu().numpy()
                            if ti == 0 and h == HYPOTHESES[0]:
                                prefix_parts.append(fields['baseline100'])
                            if ti == 0 and bank == 0 and h == HYPOTHESES[0] and start == 0:
                                store.json('FIRST_ACTUAL_HARD_PROOF.json', dict(store.base(),
                                    schema='n01/wp3-024-first-hard-proof/v1', theta_index=0, bank=0, hypothesis=h,
                                    actual_hard_rows=stop-start, real_hard_ADC_argmax_gather_complete=True))
                        if prefix_parts:
                            store.npz(f'bank{bank}_BASELINE100.npz', baseline100=np.concatenate(prefix_parts))
                        coded = {k: np.concatenate([part[k] for part in coded_parts])
                            for k in ('ADC_crop_codes', 'baseline_mean_V', 'peak_index')}
                        if any('IEEE754_reconstruction_XOR_uint64' in part for part in coded_parts):
                            coded['IEEE754_reconstruction_XOR_uint64'] = np.concatenate([
                                part.get('IEEE754_reconstruction_XOR_uint64', np.zeros(part['ADC_crop_codes'].shape, np.uint64))
                                for part in coded_parts])
                        store.npz(f'theta{ti:02d}_bank{bank}_{h}_HARD_ROWS.npz', **coded,
                            **{'flag__'+key: value for key, value in flags.items()})
                        if ti == 0:
                            baseline_waves[(bank, h)] = waves.copy()
                            baseline_peaks[(bank, h)] = peak.copy()
                        parent_mean = waves.reshape(n, r, 256).mean(1)
                        mu = parent_mean.mean(0)
                        residual = mu-real_mean
                        integrals = np.column_stack((waves.sum(1), waves[:, :46].sum(1), waves[:, 46:].sum(1))).reshape(n, r, 3)
                        old_wave = baseline_waves[(bank, h)]
                        change_wave = waves-old_wave
                        integral_change = np.column_stack((change_wave.sum(1), change_wave[:, :46].sum(1), change_wave[:, 46:].sum(1))).reshape(n, r, 3)
                        peak_shift = peak-baseline_peaks[(bank, h)]
                        store.npz(f'theta{ti:02d}_bank{bank}_{h}_PARENT_STATS.npz',
                            mean_Q_front_tail=integrals.mean(1), draw_variance_Q_front_tail=integrals.var(1, ddof=1),
                            paired_mean_change_Q_front_tail=integral_change.mean(1),
                            paired_draw_variance_change_Q_front_tail=integral_change.var(1, ddof=1),
                            paired_parent_mean_wave_change_RMS_V=np.sqrt(np.square(change_wave.reshape(n, r, 256).mean(1)).mean(1)),
                            crop_change_draw_count=(peak_shift != 0).reshape(n, r).sum(1),
                            max_abs_peak_shift=np.abs(peak_shift).reshape(n, r).max(1),
                            **{key+'_draw_count': value.reshape(n, r).sum(1) for key, value in flags.items()})
                        report = dict(theta_index=ti, theta_name=theta_name, bank=bank, hypothesis=h,
                            parent_count=n, draws=r, population_mean_raw_RMS_V=float(np.sqrt(np.square(residual).mean())),
                            population_mean_DC_removed_shape_RMS_V=float(np.sqrt(np.square(residual-residual.mean()).mean())),
                            hard256_residual_DC_V=float(residual.mean()),
                            signed_mean_Q_bias_Vns=float(residual.sum()),
                            signed_mean_front_bias_Vns=float(residual[:46].sum()),
                            signed_mean_tail_bias_Vns=float(residual[46:].sum()),
                            crop_changed_rows=int((peak_shift != 0).sum()), maximum_abs_peak_shift=int(np.abs(peak_shift).max()),
                            flag_counts={key: int(value.sum()) for key, value in flags.items()},
                            fixed_library_scalar_MC_SE=np.sqrt(integrals.var(1, ddof=1).sum(0)/(r*n*n)).tolist(),
                            paired_change_scalar_MC_SE=np.sqrt(integral_change.var(1, ddof=1).sum(0)/(r*n*n)).tolist())
                        reports.append(report)
                        means.append(mu)
                        store.call('complete_one_population_mean_cell', dict(population_mean_cells=1), lambda: None)
                        require(sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file()) <= p['resource_limits']['output_MiB']*1024**2,
                            '024 output cap exceeded')
                        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= p['resource_limits']['job_tree_RSS_GiB']*1024**3,
                            '024 worker peak RSS cap exceeded')
                        del base, waves, coded, coded_parts, fields, obs, parent_mean, change_wave, integrals, integral_change
                store.call('complete_fixed_theta', dict(theta_points=1), lambda: None)
            witness_rows = []
            for ti in range(1, 13):
                rows = []
                for j in range(6):
                    initial, current = reports[j], reports[ti*6+j]
                    require(initial['population_mean_raw_RMS_V'] > 0 and initial['population_mean_DC_removed_shape_RMS_V'] > 0,
                        '024 baseline RMS normalization undefined')
                    ratio = current['population_mean_raw_RMS_V']/initial['population_mean_raw_RMS_V']
                    rows.append(dict(bank=current['bank'], hypothesis=current['hypothesis'], raw_RMS_ratio=ratio,
                        DC_removed_shape_strictly_improves=current['population_mean_DC_removed_shape_RMS_V'] < initial['population_mean_DC_removed_shape_RMS_V'],
                        absolute_Q_bias_does_not_increase=abs(current['signed_mean_Q_bias_Vns']) <= abs(initial['signed_mean_Q_bias_Vns']),
                        raw_ratio_at_most_090=ratio <= .90))
                witness_rows.append(dict(theta_index=ti, theta_name=p['theta_names'][ti], theta=p['theta_points'][ti],
                    finite_profile_witness=all(row['raw_ratio_at_most_090'] and row['DC_removed_shape_strictly_improves'] and
                        row['absolute_Q_bias_does_not_increase'] for row in rows), six_cells=rows))
            store.call('all_72_registered_baseline_paired_comparisons', dict(baseline_paired_comparisons=72), lambda: None)
        mean_array = np.stack(means).reshape(13, 2, 3, 256)
        store.npz('POPULATION_MEANS.npz', mean_wave256_V=mean_array,
            paired_mean_change_from_zero_V=mean_array-mean_array[:1],
            bank_difference_RMS_over_sqrt2_V=np.sqrt(np.square(mean_array[:, 0]-mean_array[:, 1]).mean(-1)/2),
            paired_change_bank_difference_RMS_over_sqrt2_V=np.sqrt(np.square(
                (mean_array-mean_array[:1])[:, 0]-(mean_array-mean_array[:1])[:, 1]).mean(-1)/2))
        store.json('PROFILE.json', dict(schema='n01/wp3-024-profile/v1', run_id=RUN_ID, cells=reports,
            trust=trust, finite_profile_witnesses=witness_rows,
            any_fixed_profile_witness=any(row['finite_profile_witness'] for row in witness_rows),
            REAL_mean_diagonal_SE_RMS_V=float(np.sqrt(real_variance.mean())),
            REAL_group_is_independent_acquisition=False, cross_time_REAL_covariance_available=False,
            conditional_REAL_truth=False, fixed_population_MC_not_parent_sampling=True,
            independent_time_coordinates_assumed=False, response_acceptance=False,
            no_witness_is_not_full_basis_unreachable=True, automatic_training_or_timewarp=False))
        extension.assert_frozen()
        require(state_digest(extension.model005.state_dict()) == frozen['noise005'] and
            state_digest(extension.state_dict()) == frozen['extension'], '024 protected005 changed')
        require(store.counts == p['budget_caps'] and store.attempted == store.counts and not store.inflight,
            '024 exact completed budget differs')
        unchanged(checked)
        inventory = {str(q.relative_to(store.root)): dict(bytes=q.stat().st_size, sha256=sha(q))
            for q in sorted(store.root.rglob('*')) if q.is_file() and q.name not in ('status.json', 'operations.jsonl')}
        store.json('ARTIFACT_INVENTORY.json', dict(schema='n01/wp3-024-artifacts/v1', run_id=RUN_ID, files=inventory,
            selected_collection_MiB=128, remote_large_files='*_BASE384.npy and *_HARD_ROWS.npz',
            selected_readback='JSON, AXES_AND_TARGET, RANDOM_STATES, POPULATION_MEANS, *_TRUST and *_PARENT_STATS'))
        store.phase = 'complete_fixed_profile'
        store.json('COMPLETE.json', dict(store.base(), schema='n01/wp3-024-complete/v1',
            native_driver_completed=True, population_mean_cells=78,
            any_fixed_profile_witness=any(row['finite_profile_witness'] for row in witness_rows),
            artifact_inventory_sha256=sha(store.root/'ARTIFACT_INVENTORY.json'),
            old015_M1_replay=False, old_M0_and005_unchanged=True, response_acceptance=False))
        store.status()
    except BaseException as error:
        store.error = repr(error)
        store.phase = 'failed'
        store.json('FAILED.json', dict(store.base(), schema='n01/wp3-024-failed/v1'))
        store.status()
        raise
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    return run024(args.config, args.config_sha256, args.output, args.device)


if __name__ == '__main__':
    raise SystemExit(main())
