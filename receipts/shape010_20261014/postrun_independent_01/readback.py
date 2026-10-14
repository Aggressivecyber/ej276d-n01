"""Read frozen shape010 artifacts and the expressly authorized TRAIN reference.

No response/noise NN creation, model forward, gradients, optimizer or DEV read.
Independent NumPy wave/QPSD statistics, CPU Torch only for serialized tensors and
the already registered evaluation projection generator. Original files untouched.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TEMP'] = os.environ['TMP'] = os.environ['TMPDIR'] = 'E:/GAN/runtime_tmp'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = os.environ['NUMEXPR_NUM_THREADS'] = '2'
import numpy as np
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ROOT = Path('E:/GAN/research/N01')
OWN = ROOT/'receipts/shape010_20261014/postrun_independent_01'
HYPOTHESES = ('fast', 'slow', 'common_level')
VIEWS = ('frozen_A', 'shape010_initial', 'shape010_final')
BOOTSTRAP_SEED = 2026101440
BOOTSTRAP_REPLICATES = 500
DISTANCE_TOLERANCE = 1e-7  # cross-CPU float32 normal/direction implementation


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def summary(value):
    x = np.asarray(value, np.float64)
    require(np.isfinite(x).all(), 'Nonfinite summary')
    return dict(mean=float(x.mean()), std=float(x.std(ddof=1)) if x.size > 1 else 0.,
                minimum=float(x.min()), median=float(np.median(x)),
                maximum=float(x.max()), p05=float(np.quantile(x, .05)), p95=float(np.quantile(x, .95)))


BOOTSTRAP_ROWS = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 512, (BOOTSTRAP_REPLICATES, 512))


def parent_scalar(value):
    x = np.asarray(value, np.float64).reshape(512, 4)
    parent = x.mean(1)
    return dict(mean=float(parent.mean()), parent_cluster_SE=float(parent.std(ddof=1)/np.sqrt(512)),
                parent_cluster_bootstrap_CI95=np.quantile(parent[BOOTSTRAP_ROWS].mean(1), [.025, .975]),
                mean_within_parent_MCSE=float((x.std(1, ddof=1)/2).mean()),
                parent_mean_summary=summary(parent), parent_clusters=512, draws_per_parent=4,
                uncertainty_scope='fixed TRAIN pool and saved four draws; not independent-acquisition uncertainty')


def repeated_difference(value):
    x = np.asarray(value, np.float64).reshape(512, 4, -1)
    mean = x.mean(1)
    variance = x.var(1, ddof=1)
    se2 = variance/4
    mean_rms = np.sqrt(np.mean(mean**2, 1))
    mcse_rms = np.sqrt(np.mean(se2, 1))
    draw_ms = np.mean(x*x, (1, 2))
    corrected_ms = np.mean(mean*mean-se2, 1)
    global_mcse = np.sqrt(se2.sum(0))/512
    return dict(mean_parent_mean_RMS_V=float(mean_rms.mean()),
                RMS_across_parent_means_V=float(np.sqrt(np.mean(mean*mean))),
                mean_parent_MCSE_RMS_V=float(mcse_rms.mean()),
                RMS_of_equal_parent_global_mean_V=float(np.sqrt(np.mean(mean.mean(0)**2))),
                global_mean_MCSE_RMS_V=float(np.sqrt(np.mean(global_mcse**2))),
                actual_all_draw_RMS_V=float(np.sqrt(draw_ms.mean())),
                mean_per_draw_RMS_V=float(np.sqrt(np.mean(x*x, 2)).mean()),
                parent_mean_RMS_summary_V=summary(mean_rms),
                average_conditional_mean_squared_MC_variance_corrected_V2=float(corrected_ms.mean()),
                corrected_parent_MS_cluster_bootstrap_CI95_V2=np.quantile(corrected_ms[BOOTSTRAP_ROWS].mean(1), [.025, .975]),
                parent_mean_RMS_cluster_bootstrap_CI95_V=np.quantile(mean_rms[BOOTSTRAP_ROWS].mean(1), [.025, .975]),
                MCSE_scope='four saved latent/noise draws per parent; estimated ddof1; not physical uncertainty')


def w1(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    require(x.shape[1] == y.shape[1] and min(len(x), len(y)) > 0, 'W1 shape')
    if len(x) == len(y):
        return float(np.abs(np.sort(x, axis=0)-np.sort(y, axis=0)).mean())
    values = []
    for column in range(x.shape[1]):
        a, b = np.sort(x[:, column]), np.sort(y[:, column])
        points = np.sort(np.concatenate((a, b)))
        cdf_a = np.searchsorted(a, points[:-1], side='right')/len(a)
        cdf_b = np.searchsorted(b, points[:-1], side='right')/len(b)
        values.append(np.sum(np.abs(cdf_a-cdf_b)*np.diff(points)))
    return float(np.mean(values))


def wave_views(y, peak_floor):
    shape = y/np.maximum(y.max(1, keepdims=True), peak_floor)
    values = dict(voltage=y, shape=shape, difference=np.diff(shape, axis=1), curvature=np.diff(shape, n=2, axis=1))
    for size in (5, 17):
        padded = np.pad(shape, ((0, 0), (size//2, size//2)), mode='edge')
        smooth = np.lib.stride_tricks.sliding_window_view(padded, size, axis=1).mean(2)
        values['difference_smooth_'+str(size)] = np.diff(smooth, axis=1)
    return values


def q_features(y, stats):
    q = y.sum(1)
    floor, reference = stats['q_floor'], stats['q_reference']
    log = np.where(q >= floor, np.log(np.maximum(q, floor)/reference),
                   np.log(floor/reference)+np.arcsinh((q-floor)/floor))
    psd = y[:, 46:].sum(1)/(np.where(q < 0, -1., 1.)*np.maximum(np.abs(q), floor))
    return np.stack((log, psd), 1)


def loss_statistics(fake, real, stats, banks):
    fv, rv = wave_views(fake, stats['peak_floor']), wave_views(real, stats['peak_floor'])
    f, r = {}, {}
    for name in fv:
        f[name] = (fv[name]-np.asarray(stats['views'][name]['center']))/np.asarray(stats['views'][name]['scale'])
        r[name] = (rv[name]-np.asarray(stats['views'][name]['center']))/np.asarray(stats['views'][name]['scale'])
    swd = np.mean([w1(f[name]@banks['wave:'+name], r[name]@banks['wave:'+name]) for name in fv])
    coordinate = np.mean([w1(f[name], r[name]) for name in ('voltage', 'shape')])
    joint = []
    for name in ('voltage', 'shape'):
        for lag in (1, 2, 4, 8):
            direction = banks[f'lag:{name}:{lag}']
            pair = lambda x: (x[:, :-lag, None]*direction[0]+x[:, lag:, None]*direction[1]).reshape(len(x), -1)
            joint.append(w1(pair(f[name]), pair(r[name])))
    short_lag = float(np.mean(joint))
    fq, rq = q_features(fake, stats), q_features(real, stats)
    center, scale = np.asarray(stats['qpsd']['center']), np.asarray(stats['qpsd']['scale'])
    fz, rz = (fq-center)/scale, (rq-center)/scale
    global_qpsd = .5*(w1(fz, rz)+w1(fz@banks['qpsd'], rz@banks['qpsd']))
    fb = np.searchsorted(stats['charge_bin_boundaries'], fq[:, 0], side='left')
    rb = np.searchsorted(stats['charge_bin_boundaries'], rq[:, 0], side='left')
    conditional, counts = [], []
    for index in range(8):
        fm, rm = fb == index, rb == index
        supported = min(int(fm.sum()), int(rm.sum())) >= stats['minimum_conditional_support']
        counts.append(dict(bin=index, fake_count=int(fm.sum()), real_count=int(rm.sum()), supported=supported))
        if supported:
            conditional.append(w1(fz[fm, 1], rz[rm, 1]))
    return dict(components=dict(wave=.5*swd+.25*coordinate+.25*short_lag,
                QPSD=.5*(global_qpsd+np.mean(conditional)) if conditional else global_qpsd),
                wave_subcomponents=dict(swd=float(swd), coordinate=float(coordinate), short_lag=short_lag),
                conditional_counts=counts, conditional_bins_used=len(conditional),
                fake_nonpositive_q_rows=int((fake.sum(1) <= 0).sum()),
                real_nonpositive_q_rows=int((real.sum(1) <= 0).sum()),
                fake_guarded_psd_rows=int((np.abs(fake.sum(1)) < stats['q_floor']).sum()),
                real_guarded_psd_rows=int((np.abs(real.sum(1)) < stats['q_floor']).sum()),
                implementation='independent NumPy W1/views/standardization/conditional reductions; registered Torch projection bank only')


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_sha(value.numpy()),
                elements=value.numel(), norm=float(value.double().norm()))


def main(collection):
    artifact_root = collection/'runs/primary/artifacts'
    supervisor = collection/'runs/primary/supervisor'
    require(artifact_root.is_dir() and supervisor.is_dir(), 'Collected primary layout missing')
    complete = load_json(artifact_root/'COMPLETE.json')
    proof = load_json(artifact_root/'FIRST_UPDATE_PROOF.json')
    status = load_json(artifact_root/'status.json')
    exit_receipt = load_json(supervisor/'exit_receipt.json')
    launch = load_json(supervisor/'launch.json')
    policy = load_json(ROOT/'shape010_policy.json')
    binding = load_json(ROOT/'SHAPE010_BINDINGS.json')
    manifest = load_json(ROOT/'SHAPE010_SOURCE_MANIFEST.json')
    freeze = load_json(ROOT/'receipts/shape010_20261014/FREEZE_RECEIPT.json')
    saved_binding = load_json(artifact_root/'BINDING.json')
    frozen_inputs = load_json(artifact_root/'FROZEN_INPUTS.json')
    source_hashes = {}
    for relative, expected in manifest['files'].items():
        actual = sha(ROOT/relative)
        require(actual == expected, 'Changed frozen010 source '+relative)
        source_hashes[relative] = actual
    protected = {}
    old_files = {}
    for relative, expected in manifest['protected_manifests'].items():
        require(sha(ROOT/relative) == expected, 'Changed protected manifest '+relative)
        old = load_json(ROOT/relative)
        for name, digest in old['files'].items():
            require(sha(ROOT/name) == digest, 'Changed old008/009 source '+name)
            require(name not in old_files or old_files[name] == digest, 'Conflicting old manifests')
            old_files[name] = digest
        protected[relative] = dict(sha256=expected, files=len(old['files']))
    require(sha(ROOT/'SHAPE010_SOURCE_MANIFEST.json') == freeze['source_manifest_sha256'] == binding['source_manifest']['sha256'], 'Source manifest chain')
    require(sha(ROOT/'SHAPE010_BINDINGS.json') == freeze['bindings_sha256'], 'Binding hash chain')
    require(sha(ROOT/'shape010_policy.json') == freeze['policy_sha256'] == binding['policy_sha256'], 'Policy chain')
    require(manifest['bound_inputs'] == {k:v['sha256'] for k,v in binding['files'].items()}, 'Input binding schema')
    expected_input_files = {v['path']:v['sha256'] for v in binding['files'].values()}
    expected_input_files.update({policy['remote_root']+'/shape010_policy.json':freeze['policy_sha256'],
        policy['remote_root']+'/SHAPE010_BINDINGS.json':freeze['bindings_sha256'],
        policy['remote_root']+'/SHAPE010_SOURCE_MANIFEST.json':freeze['source_manifest_sha256']})
    expected_source_files = {policy['remote_root']+'/'+name:digest for name,digest in manifest['files'].items()}
    require(saved_binding['files_sha256'] == expected_input_files and saved_binding['sources_sha256'] == expected_source_files,
            'Actual native binding differs from frozen inputs/source inventory')
    require({k:frozen_inputs['binding'][k] for k in saved_binding} == saved_binding and saved_binding['before_TRAIN_decode'] is True
            and saved_binding['before_checkpoint_load'] is True and saved_binding['DEV_numeric_read'] is False,
            'Native input binding order/receipt chain')
    require(set(frozen_inputs['binding'])-set(saved_binding) == {'CUDA_limit'}
            and frozen_inputs['binding']['CUDA_limit']['CUDA_allocator_limit_applied'] is True
            and frozen_inputs['binding']['CUDA_limit']['requested_GiB'] == 10.,
            'Runtime allocator guard missing')
    inventory = {}
    for relative, expected in complete['artifacts'].items():
        path = artifact_root/relative
        require(path.is_file() and path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], 'Artifact changed '+relative)
        inventory[relative] = expected
    require(set(p.name for p in artifact_root.iterdir() if p.is_file()) == set(inventory)|{'COMPLETE.json'}, 'Unexpected collected artifact file')
    require(exit_receipt['child_native_exit_code'] == exit_receipt['exit_code'] == 0 and exit_receipt['reason'] is None and exit_receipt['error'] is None, 'Native failed')
    require(launch['child_pid'] == exit_receipt['child_pid'] == status['pid'] == proof['pid'], 'PID chain')
    require(launch['supervisor_pid'] == exit_receipt['supervisor_pid'], 'Supervisor PID chain')
    require(complete['G_updates'] == 2000 and complete['counters']['attempted_G_updates'] == 2000, 'G budget')
    require(complete['counters'] == status['counters'] == status['attempted_operation_counters'], 'Completion accounting')
    require(status['in_flight'] is None and not (artifact_root/'FAILED.json').exists(), 'Incomplete operation')
    for counter in ('D_updates', 'noise_updates', 'population_updates', 'C2ST_updates'):
        require(complete['counters'][counter] == 0, 'Extra optimizer '+counter)
    frozen = complete['frozen_state_sha256']
    require(frozen == proof['frozen_state_sha256'] == frozen_inputs['frozen_state_sha256'], 'Frozen state chain')
    require(complete['DEV_numeric_read'] is False and complete['artificial_only'] is False, 'Wrong real-run split marker')
    require(complete['trainable_parameter_names'] == proof['trainable_parameter_names'] == ['basis_head.weight', 'basis_head.bias'], 'Wrong optimizer whitelist')
    initial_eval = load_json(artifact_root/'INITIAL_EVALUATION.json')
    final_eval = load_json(artifact_root/'FINAL_EVALUATION.json')
    require(initial_eval['distances'] == complete['initial_evaluation'] and final_eval['distances'] == complete['final_evaluation'], 'Saved metric receipts differ')
    require(initial_eval['projection_context'] == final_eval['projection_context'], 'Projection not paired')
    banksaved = dict(np.load(artifact_root/'EVALUATION_BANK.npz', allow_pickle=False))
    require(banksaved['shape_z'].shape == (2048, 8), 'Latent bank shape')
    require(len(np.unique(banksaved['selected_source_rows'])) == 512, 'Source selection duplicates')
    require(len(np.unique(banksaved['real_reference_rows'])) == len(np.unique(banksaved['real_control_rows'])) == 2048, 'Real selection repeats')
    require(not np.intersect1d(banksaved['real_reference_rows'], banksaved['real_control_rows']).size, 'REAL control overlap')
    require(np.array_equal(banksaved['selected_source_rows'], np.random.default_rng(policy['seeds']['source_selection']).permutation(6593)[:512]), 'Source selection seed')
    all_real_selection = np.random.default_rng(policy['seeds']['real_selection']).permutation(21047)[:4096]
    require(np.array_equal(all_real_selection[:2048], banksaved['real_reference_rows']) and np.array_equal(all_real_selection[2048:], banksaved['real_control_rows']), 'Real selection seed')
    expected_shape = torch.randn(2048, 8, generator=torch.Generator().manual_seed(policy['seeds']['evaluation_shape']), dtype=torch.float32).numpy()
    shape_recreation_error = float(np.max(np.abs(expected_shape-banksaved['shape_z'])))
    shape_recreation_byte_exact = bool(np.array_equal(expected_shape, banksaved['shape_z']))
    require(shape_recreation_error <= 5e-7, 'Evaluation shape stream seed outside cross-CPU numerical tolerance')
    dataset = ROOT/'data/window008/DATASET008.npz'
    require(sha(dataset) == binding['files']['dataset']['sha256'], 'TRAIN dataset hash BEFORE selective decode')
    stat_path = ROOT/'data/window008/FIT_STATISTICS008.json'
    require(sha(stat_path) == binding['files']['statistics']['sha256'], 'Statistics hash')
    statistics = load_json(stat_path)
    stats = statistics['loss']
    require(statistics['statistics_split'] == stats['fit_split'] == 'TRAIN' and statistics['prepared_dataset_sha256'] == binding['files']['dataset']['sha256'], 'TRAIN statistics provenance')
    permitted_members = ('real', 'real_baseline100', 'real_ids', 'source_ids', 'real_groups')
    with np.load(dataset, allow_pickle=False) as archive:
        train = {name: archive[name].copy() for name in permitted_members}
    require(train['real'].shape == (21047, 256) and train['real_baseline100'].shape == (21047, 100) and train['source_ids'].shape == (6593,), 'TRAIN field shapes')
    require(np.isfinite(train['real']).all() and np.isfinite(train['real_baseline100']).all(), 'TRAIN references nonfinite')
    require(np.array_equal(train['source_ids'][banksaved['selected_source_rows']], banksaved['selected_source_ids']), 'Source ID bank')
    real = train['real'][banksaved['real_reference_rows']].astype(np.float64)
    real_control = train['real'][banksaved['real_control_rows']].astype(np.float64)
    sys.path.insert(0, str(ROOT/'src'))
    from n01.losses008_v2 import create_evaluation_context
    context = create_evaluation_context(policy['seeds']['evaluation_projection'])
    local_projection_metadata = context.metadata()
    native_projection_metadata = initial_eval['projection_context']
    require(all(local_projection_metadata[k] == native_projection_metadata[k] for k in ('kind', 'seed', 'update')), 'Evaluation projection seed/kind/update')
    banks = {name:value.numpy().astype(np.float64) for name,value in context.banks.items()}
    arrays = {}
    structure = {}
    recomputed = {}
    max_loss_error = 0.
    expected_parent = np.repeat(np.arange(512), 4)
    expected_ids = np.repeat(banksaved['selected_source_ids'], 4)
    q_identity_max = 0.
    for view in VIEWS:
        arrays[view], structure[view], recomputed[view] = {}, {}, {}
        for h in HYPOTHESES:
            values = dict(np.load(artifact_root/f'{view}_{h}_TRAIN.npz', allow_pickle=False))
            arrays[view][h] = values
            require(values['waveform_V'].shape == (2048, 256) and values['pure_signal_V'].shape == (2048, 384), 'Saved shape '+view+h)
            require(all(v.dtype.kind != 'O' and np.isfinite(v).all() for v in values.values() if v.dtype.kind in 'bifuc'), 'Saved array nonfinite')
            require(np.array_equal(values['source_parent_row'], expected_parent) and np.array_equal(values['source_parent_ids'], expected_ids), 'Parent order '+view+h)
            require(not np.any(values['pure_signal_V'][:, :100]), 'Nonzero pure signal prefix')
            require(np.array_equal(values['crop_start'], values['peak_index']-16) and np.array_equal(values['crop_stop'], values['crop_start']+256), 'Crop index chain')
            require(np.all((values['peak_index'] >= 112) & (values['peak_index'] < 144)), 'Search bounds')
            y = values['waveform_V']
            q = y.sum(1)
            require(np.max(np.abs(q-values['Q'])) < 1e-10, 'Stored Q not window sum')
            indices = values['crop_start'][:, None]+np.arange(256)
            qsignal = np.take_along_axis(values['pure_signal_V'], indices, axis=1).sum(1)
            canonical = values['pure_signal_V'][:, 110:366].sum(1)
            error = max(float(np.max(np.abs(qsignal-values['Q_signal']))),
                        float(np.max(np.abs(q-values['Q_signal']-values['Q_noise']-values['Q_ADC_delta']))),
                        float(np.max(np.abs(values['Q_signal_crop_migration']-(qsignal-canonical)))))
            q_identity_max = max(q_identity_max, error)
            require(error < 1e-10, 'Same-crop Q identity')
            require(np.max(np.abs(values['Q_ADC_delta'])) <= 256/16383+1e-12, 'ADC residual bound')
            moments = dict(np.load(artifact_root/f'{view}_{h}_PARENT_MOMENTS.npz', allow_pickle=False))
            repeated = y.reshape(512, 4, 256)
            mean, variance = repeated.mean(1), repeated.var(1, ddof=1)
            errs = {name:float(np.max(np.abs(moments[name]-expected))) for name,expected in
                    (('mean', mean), ('sample_variance', variance), ('MonteCarlo_SE', np.sqrt(variance/4)))}
            require(max(errs.values()) < 1e-12 and np.all(moments['defined_draws'] == 4) and not np.any(moments['undefined_draws']), 'Parent moments')
            native = (initial_eval if view != 'shape010_final' else final_eval)['distances'][view][h]
            loss = loss_statistics(y, real, stats, banks)
            loss['absolute_errors'] = {name:abs(loss['components'][name]-native['components'][name]) for name in ('wave', 'QPSD')}
            loss['wave_subcomponent_absolute_errors'] = {name:abs(value-native['wave_subcomponents'][name]) for name,value in loss['wave_subcomponents'].items()}
            max_loss_error = max(max_loss_error, *loss['absolute_errors'].values(), *loss['wave_subcomponent_absolute_errors'].values())
            require(max(loss['absolute_errors'].values()) < DISTANCE_TOLERANCE and max(loss['wave_subcomponent_absolute_errors'].values()) < DISTANCE_TOLERANCE, 'Loss recomputation mismatch '+view+h)
            require(loss['conditional_counts'] == [{k:v for k,v in record.items() if k != 'status'} for record in native['conditional_diagnostics']['conditional_bins']], 'Conditional bin counts')
            recomputed[view][h] = loss
            psd = q_features(y, stats)[:, 1]
            structure[view][h] = dict(rows=2048, parent_clusters=512, repeats=4, parent_order_exact=True,
                prefix_exact_zero=True, moments_absolute_errors=errs, Q_identity_error=error,
                Q=parent_scalar(q), PSD=parent_scalar(psd), peak_index=summary(values['peak_index']),
                global_peak_index=summary(values['global_peak_index']),
                peak_boundary_rows=int(np.isin(values['peak_index'], [112, 143]).sum()),
                global_peak_outside_search_rows=int(((values['global_peak_index'] < 112)|(values['global_peak_index'] >= 144)).sum()),
                Q_signal_crop_migration=parent_scalar(values['Q_signal_crop_migration']),
                Q_noise=parent_scalar(values['Q_noise']), Q_ADC_delta=parent_scalar(values['Q_ADC_delta']))
        require(all(np.array_equal(arrays[view]['fast']['pure_signal_V'], arrays[view][h]['pure_signal_V']) for h in HYPOTHESES), 'Signal should not depend on observation hypothesis')
    rr = loss_statistics(real_control, real, stats, banks)
    rr_native = initial_eval['distances']['REAL_REAL']
    rr['absolute_errors'] = {name:abs(rr['components'][name]-rr_native['components'][name]) for name in ('wave', 'QPSD')}
    require(max(rr['absolute_errors'].values()) < DISTANCE_TOLERANCE, 'RR recomputation')
    pure_a = arrays['frozen_A']['fast']['pure_signal_V']
    require(np.array_equal(pure_a.reshape(512, 4, 384), np.repeat(pure_a.reshape(512, 4, 384)[:, :1], 4, axis=1)), 'Frozen A changes over repeats')
    shape_results, pairs = {}, {}
    for view in ('shape010_initial', 'shape010_final'):
        delta = arrays[view]['fast']['pure_signal_V']-pure_a
        shape_results[view] = repeated_difference(delta)
        shape_results[view]['canonical_random_Q_absmax_Vns'] = float(np.abs(delta[:, 110:366].sum(1)).max())
        shape_results[view]['canonical_signal_Q_difference_absmax_Vns'] = float(np.abs(arrays[view]['fast']['pure_signal_V'][:, 110:366].sum(1)-pure_a[:, 110:366].sum(1)).max())
        require(shape_results[view]['canonical_random_Q_absmax_Vns'] < 1e-10, 'Random shape not canonical zero Q')
        require(shape_results[view]['canonical_signal_Q_difference_absmax_Vns'] < 1e-10, 'Canonical total signal Q changed')
        pairs[view+'_minus_A'] = {}
        for h in HYPOTHESES:
            a, v = arrays['frozen_A'][h], arrays[view][h]
            changed = v['peak_index'] != a['peak_index']
            pairs[view+'_minus_A'][h] = dict(observed_wave_difference=repeated_difference(v['waveform_V']-a['waveform_V']),
                Q_difference=parent_scalar(v['Q']-a['Q']),
                PSD_difference=parent_scalar(q_features(v['waveform_V'], stats)[:, 1]-q_features(a['waveform_V'], stats)[:, 1]),
                Q_signal_difference=parent_scalar(v['Q_signal']-a['Q_signal']),
                Q_noise_difference=parent_scalar(v['Q_noise']-a['Q_noise']),
                Q_ADC_delta_difference=parent_scalar(v['Q_ADC_delta']-a['Q_ADC_delta']),
                Q_signal_migration_difference=parent_scalar(v['Q_signal_crop_migration']-a['Q_signal_crop_migration']),
                crop_changed_rows=int(changed.sum()), crop_changed_parents=int(changed.reshape(512, 4).any(1).sum()),
                peak_shift_samples=summary(v['peak_index']-a['peak_index']))
            same = ~changed
            require(np.max(np.abs(v['Q_noise'][same]-a['Q_noise'][same])) < 1e-12, 'Noise not shared for same crop')
    pairs['final_minus_initial'] = {}
    for h in HYPOTHESES:
        f, i = arrays['shape010_final'][h], arrays['shape010_initial'][h]
        pairs['final_minus_initial'][h] = dict(Q_difference=parent_scalar(f['Q']-i['Q']),
            PSD_difference=parent_scalar(q_features(f['waveform_V'], stats)[:, 1]-q_features(i['waveform_V'], stats)[:, 1]),
            observed_wave_difference=repeated_difference(f['waveform_V']-i['waveform_V']),
            crop_changed_rows=int((f['peak_index'] != i['peak_index']).sum()))
    checkpoints = {}
    head_weights = {}
    stream_digests = {}
    for name, update in [('initial.pt', 0), ('periodic_01000.pt', 1000), ('final.pt', 2000)]:
        checkpoint = torch.load(artifact_root/name, map_location='cpu', weights_only=True)
        require(checkpoint['schema'] == 'n01/shape010-checkpoint/v1' and checkpoint['G_updates'] == update and checkpoint['complete_step_boundary'], 'Thin checkpoint boundary '+name)
        require(checkpoint['frozen_state_sha256'] == frozen and checkpoint['policy'] == policy, 'Thin checkpoint frozen/policy')
        require(checkpoint['trainable_parameter_names'] == ['basis_head.weight', 'basis_head.bias'], 'Thin whitelist')
        require(set(checkpoint['new_head']) == {'weight', 'bias'}, 'Thin checkpoint carries old weights')
        require(checkpoint['new_head']['weight'].shape == (8, 32, 5) and checkpoint['new_head']['bias'].shape == (8,), 'Head shape')
        require(set(checkpoint['streams']) == {'shape', 'noise005', 'extension'}, 'Explicit stream slots')
        require(checkpoint['in_flight'] is None and not checkpoint['DEV_numeric_read'] and not checkpoint['artificial_only'], 'Thin boundary flags')
        samplers = {}
        for slot, n, draws in [('source_sampler', 6593, 512), ('real_sampler', 21047, 1024)]:
            state = checkpoint[slot]
            visits, order = state['visits'].numpy(), state['order'].numpy()
            total = draws*update
            epochs, position = divmod(total, n)
            require(state['n'] == n and visits.shape == (n,) and np.array_equal(np.sort(order), np.arange(n)), 'Sampler population/order')
            require(state['completed_epochs'] == epochs and state['position'] == position and int(visits.sum()) == total, 'Sampler accounting')
            expected_visits = np.full(n, epochs)
            expected_visits[order[:position]] += 1
            require(np.array_equal(visits, expected_visits), 'Sampler full cycles plus exact prefix')
            samplers[slot] = dict(n=n, total_visits=total, completed_epochs=epochs, position=position,
                unique_visited=int((visits > 0).sum()), visit_min=int(visits.min()), visit_max=int(visits.max()),
                order_sha256=array_sha(order), visits_sha256=array_sha(visits), rng=state['rng'])
        optimizer = checkpoint['optimizer']
        require(len(optimizer['param_groups']) == 1 and optimizer['param_groups'][0]['params'] == [0, 1], 'Optimizer whitelist indices')
        group = optimizer['param_groups'][0]
        require(group['lr'] == .0002 and group['weight_decay'] == .0001, 'Optimizer registration')
        require(set(optimizer['state']) == (set() if update == 0 else {0, 1}), 'Optimizer state population')
        step_record = {}
        for index, state in optimizer['state'].items():
            require(float(state['step']) == update, 'AdamW successful update count')
            target = checkpoint['new_head']['weight' if index == 0 else 'bias']
            require(state['exp_avg'].shape == state['exp_avg_sq'].shape == target.shape and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all(), 'AdamW state')
            step_record[index] = dict(step=int(state['step']), exp_avg=tensor_record(state['exp_avg']), exp_avg_sq=tensor_record(state['exp_avg_sq']))
        stream_digests[name] = {k:tensor_record(v) for k,v in checkpoint['streams'].items()}
        require(all(v['dtype'] == 'torch.uint8' and v['elements'] == 5056 for v in stream_digests[name].values()), 'CPU random state layout')
        require(checkpoint['counters']['G_updates'] == checkpoint['counters']['attempted_G_updates'] == update, 'CP G accounting')
        require(checkpoint['counters']['train_response_rows'] == checkpoint['counters']['train_noise005_rows'] == update*1024, 'CP response accounting')
        require(checkpoint['counters']['train_hard_observation_rows'] == update*3072, 'CP observation accounting')
        require(all(checkpoint['counters'][k] == 0 for k in ('D_updates', 'noise_updates', 'population_updates', 'C2ST_updates')), 'CP extra optimization')
        head_weights[name] = {k:v.numpy() for k,v in checkpoint['new_head'].items()}
        checkpoints[name] = dict(sha256=inventory[name]['sha256'], G_updates=update, purpose=checkpoint['purpose'],
            tensor_inventory={k:tensor_record(v) for k,v in checkpoint['new_head'].items()},
            optimizer_steps=step_record, counters=checkpoint['counters'], attempted=checkpoint['attempted'],
            samplers=samplers, streams=stream_digests[name], frozen_state_sha256=checkpoint['frozen_state_sha256'],
            only_new_head_and_optimizer_samplers_streams=True, full_baseline_or_noise_weights_saved=False)
    require(all(stream_digests['initial.pt'][slot]['sha256'] != stream_digests['periodic_01000.pt'][slot]['sha256'] != stream_digests['final.pt'][slot]['sha256'] for slot in ('shape', 'noise005', 'extension')), 'Train streams failed to advance')
    head_changes = {name:{k:float(np.linalg.norm(head_weights[name][k]-head_weights['initial.pt'][k])) for k in ('weight', 'bias')} for name in ('periodic_01000.pt', 'final.pt')}
    log = [json.loads(line) for line in (artifact_root/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    log_updates = [row['G_updates'] for row in log]
    require(log_updates == [1]+list(range(100, 2001, 100)), 'Training log update records')
    require(all(row['G_updates'] == row['attempted_G_updates'] for row in log), 'Logged failed attempted update')
    reported_changes = {}
    for h in HYPOTHESES:
        a, i, f = (recomputed[view][h]['components'] for view in VIEWS)
        reported_changes[h] = dict(A=a, initial=i, final=f,
            final_minus_A={k:f[k]-a[k] for k in a}, relative_final_minus_A_percent={k:100*(f[k]/a[k]-1) for k in a},
            final_minus_initial={k:f[k]-i[k] for k in a},
            wave_plus_2QPSD_proxy=dict(A=a['wave']+2*a['QPSD'], initial=i['wave']+2*i['QPSD'],
                final=f['wave']+2*f['QPSD'], final_minus_A=f['wave']+2*f['QPSD']-a['wave']-2*a['QPSD'],
                relative_final_minus_A_percent=100*((f['wave']+2*f['QPSD'])/(a['wave']+2*a['QPSD'])-1)),
            all_three_joint_components_improve_on_same_A_bank=bool(all(f[k] < a[k] for k in a)))
    result = dict(schema='n01/shape010-independent-numeric-readback/v1', UTC=datetime.now(timezone.utc).isoformat(),
        run_id=policy['run_id'], status='completed_saved_arrays_and_authorized_TRAIN_statistics_recomputed',
        analysis_scope=dict(new_response_or_noise_model_instances=0, new_model_forwards=0, gradients=0, optimizer_updates=0,
            DEV_or_validation_numeric_reads=0, original_checkpoint_reads=0, source_waveform_decodes=0,
            new_thin_checkpoints_read=3, whitelisted_TRAIN_members_decoded=list(permitted_members), CPU_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(), temp_root=os.environ['TEMP'], CUDA_used=False,
            bootstrap_scope='500 pure NumPy fixed saved-parent descriptive statistics only; no whole distribution-loss bootstrap or settings selection',
            bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=BOOTSTRAP_REPLICATES),
        collection=collection, artifacts=inventory, COMPLETE_sha256=sha(artifact_root/'COMPLETE.json'),
        inventory_count=len(inventory), inventoried_bytes=sum(v['bytes'] for v in inventory.values()),
        metadata_chain=dict(exit=exit_receipt, proof_pid=proof['pid'], proof_G=proof['actual_G_updates'],
            final_G=complete['G_updates'], counters=complete['counters'], coverage=complete['coverage'], in_flight=None,
            frozen_state_sha256=frozen, frozen_input_binding=saved_binding,
            freeze_receipt=freeze, source_file_count=len(source_hashes), protected_manifests=protected,
            original008_009_unique_protected_file_count=len(old_files), source_hashes=source_hashes),
        dataset_selective_read=dict(path=dataset, sha256=binding['files']['dataset']['sha256'],
            statistics_sha256=binding['files']['statistics']['sha256'], members=list(permitted_members),
            selected_real_ref_ids_sha256=array_sha(train['real_ids'][banksaved['real_reference_rows']]),
            selected_real_control_ids_sha256=array_sha(train['real_ids'][banksaved['real_control_rows']]),
            selected_real_ref_groups=np.unique(train['real_groups'][banksaved['real_reference_rows']], return_counts=True),
            selected_real_control_groups=np.unique(train['real_groups'][banksaved['real_control_rows']], return_counts=True),
            raw_originals_unchanged=True, source_or_DEV_waveforms_not_decoded=True),
        common_bank=dict(source_parents=512, draws_per_parent=4, rows=2048, parent_major_order_exact=True,
            selected_source_ids_sha256=array_sha(banksaved['selected_source_ids']), EVALUATION_BANK_sha256=sha(artifact_root/'EVALUATION_BANK.npz'),
            source_and_REAL_seed_selection_exact=True, evaluation_shape_stream_byte_exact=shape_recreation_byte_exact,
            shape_stream_recreation_max_error=shape_recreation_error, shape_stream_tolerance=5e-7, REAL_REAL_disjoint=True,
            native_projection_context=native_projection_metadata, local_projection_context=local_projection_metadata,
            projection_fingerprint_byte_exact=local_projection_metadata['fingerprint'] == native_projection_metadata['fingerprint'],
            distance_numeric_tolerance=DISTANCE_TOLERANCE, noise_extension_bank_shared_by_native_code=True),
        checkpoints=checkpoints, head_parameter_changes_from_initial=head_changes,
        log_record_count=len(log), logged_updates=log_updates,
        saved_array_readback=structure, maximum_same_crop_Q_identity_error_Vns=q_identity_max,
        recomputed_distances=recomputed, recomputed_REAL_REAL=rr, maximum_distance_recompute_absolute_error=max_loss_error,
        initial_final_direction=reported_changes, random_shape_pure_signal_readback=shape_results,
        paired_parent_statistics=pairs,
        explicit_limits=dict(fake_baseline100_not_saved=True, generated_noise_carriers_not_saved=True,
            fake_baseline_or_noise_full_replay=False,
            noise_component_independently_recomputed=False,
            native_noise_pairing_guard_and_frozen005_state_verified=True,
            complete_basis_tensor_not_saved=True,
            analytical_zero_conditional_mean_from_source_design_not_proven_by_four_draw_mean=True,
            independent_acquisition_uncertainty='unknown', TRAIN_response_acceptance=False,
            diagnostic009_or_old_D_distance_direct_comparison_valid=False,
            supervisor_last_poll_resource_peak_not_a_final_output_inventory_measure=True),
        scientific_status='response_match_not_established')
    write_json(OWN/'NUMERIC_READBACK.json', result)
    compact = dict(status=result['status'], file=str(OWN/'NUMERIC_READBACK.json'), sha256=sha(OWN/'NUMERIC_READBACK.json'),
        inventory_count=len(inventory), max_distance_error=max_loss_error, directions=reported_changes,
        random_shape=shape_results, final_paired_fast=pairs['shape010_final_minus_A']['fast'],
        G=complete['G_updates'], coverage=complete['coverage'])
    print(json.dumps(clean(compact), ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.collection)
    except BaseException as error:
        record = dict(error_type=type(error).__name__, error=str(error), UTC=datetime.now(timezone.utc).isoformat(),
            preserved_original_files=True, new_model_forwards=0, optimizer_updates=0, DEV_numeric_read=False)
        attempt = OWN/'READBACK_ATTEMPT01.json'
        if not attempt.exists():
            write_json(attempt, record)
        raise
