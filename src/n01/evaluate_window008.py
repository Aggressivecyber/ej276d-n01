"""Frozen Window008 endpoints; no optimizer, C2ST, fit or DEV selection.

All three final checkpoints and evaluation sources are bound before DEV values
are opened. Observation/units come from Window008, never old 480-point metrics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from .data import load_validation, sha256, write_json
from .data008 import window_spec
from .losses008 import _extended_log_charge
from .losses008_v2 import N01Loss008
from .model008 import N01Response008
from .observation008 import crop_existing_values, observe_existing_carrier, observe_generated_carrier
from .response008 import ARMS, HYPOTHESES, PairedStreams, fixed_noises, loss_tensor_statistics, make_extension, state_digest
from .train import utc

ROOT = Path(__file__).resolve().parents[2]
SCHEMES = ('output_Q_in_REAL_TRAIN_bins', 'fixed_SOURCE_Q_in_SOURCE_TRAIN_bins')
METRICS = ('Q', 'PSD')
COMPARATORS = ('deterministic', 'simple')


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def array_sha(value):
    value = np.ascontiguousarray(_array(value))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return _clean(_array(value).tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _json_sha(value):
    return hashlib.sha256(json.dumps(_clean(value), sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve frozen evaluation artifact: ' + str(path))
    write_json(path, _clean(value))


def save_npz(path, **values):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve frozen evaluation artifact: ' + str(path))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_json(value):
    return json.loads(_resolve(value).read_text(encoding='utf8')) if isinstance(value, (str, Path)) else copy.deepcopy(value)


def scalar_summary(value):
    value = np.asarray(value).reshape(-1)
    if value.dtype.kind == 'b':
        return dict(rows=len(value), true_rows=int(value.sum()), false_rows=int((~value).sum()))
    finite = np.isfinite(value)
    valid = value[finite]
    result = dict(rows=len(value), numeric_defined_rows=int(finite.sum()), undefined_rows=int((~finite).sum()))
    if len(valid):
        result.update(mean=float(valid.mean()), std=float(valid.std(ddof=0)),
                      quantile_levels=[0, .01, .05, .25, .5, .75, .95, .99, 1],
                      quantiles=np.quantile(valid, [0, .01, .05, .25, .5, .75, .95, .99, 1]))
    return result


def readouts256(waveform, q_floor):
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 256 or not np.isfinite(x).all():
        raise ValueError('Readout requires finite Bx256; no row removal')
    q, tail = x.sum(1), x[:, 46:].sum(1)
    if not np.isfinite(q).all() or not np.isfinite(tail).all():
        raise ValueError('Nonfinite Window008 charge')
    raw = np.full_like(q, np.nan)
    np.divide(tail, q, out=raw, where=q != 0)
    denominator = np.where(q < 0, -1., 1.) * np.maximum(np.abs(q), float(q_floor))
    return dict(Q=q, tail=tail, PSD=tail / denominator, PSD_raw=raw,
                Q_nonpositive=q <= 0, PSD_guarded=np.abs(q) < float(q_floor), PSD_undefined=q == 0)


def output_bins(q, statistics):
    stats = statistics['loss']
    q = torch.as_tensor(np.asarray(q), dtype=torch.float64)
    coordinate = _extended_log_charge(q, torch.tensor(float(stats['q_reference']), dtype=torch.float64),
                                     torch.tensor(float(stats['q_floor']), dtype=torch.float64))
    return torch.bucketize(coordinate.contiguous(), torch.as_tensor(stats['charge_bin_boundaries'], dtype=torch.float64)).numpy()


def source_bins(q, statistics):
    edges = np.asarray(statistics['domains']['source']['Q256_eight_quantile_bin_edges'], dtype=np.float64)
    if edges.shape != (9,) or not np.isfinite(edges).all() or (np.diff(edges) < 0).any():
        raise ValueError('Frozen SOURCE TRAIN Q8 edges are invalid')
    return np.searchsorted(edges[1:-1], np.asarray(q), side='left')


def balanced_real_domains(groups, count, seed):
    """Fixed disjoint A/B populations with equal per-group quotas, no outcomes."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if not len(unique):
        raise ValueError('REAL group identities are required')
    rng = np.random.default_rng(seed)
    quota = np.full(len(unique), count // len(unique), dtype=np.int64)
    quota[rng.permutation(len(unique))[:count % len(unique)]] += 1
    a, b = [], []
    for group, number in zip(unique, quota):
        rows = rng.permutation(np.flatnonzero(groups == group))
        if len(rows) < 2 * number:
            raise ValueError('Insufficient REAL group support for frozen matching quota')
        a.extend(rows[:number]); b.extend(rows[number:2 * number])
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    if len(a) != count or len(b) != count or np.intersect1d(a, b).size:
        raise RuntimeError('REAL A/B selection is not disjoint and complete')
    return a, b


class Moments:
    def __init__(self, length):
        self.n = 0
        self.mean = np.zeros(length, np.float64)
        self.m2 = np.zeros(length, np.float64)

    def add(self, x):
        x = np.asarray(x, np.float64)
        if not len(x):
            return
        n = len(x); mean = x.mean(0); m2 = ((x - mean) ** 2).sum(0)
        total = self.n + n; delta = mean - self.mean
        self.m2 += m2 + delta ** 2 * self.n * n / total
        self.mean += delta * n / total; self.n = total

    def values(self):
        if not self.n:
            return np.full_like(self.mean, np.nan), np.full_like(self.mean, np.nan)
        return self.mean.copy(), np.sqrt(np.maximum(self.m2 / self.n, 0))


def conditional_rows(observables, bins, parents=None):
    result = []
    for index in range(8):
        mask = bins == index
        result.append(dict(bin=index, rows=int(mask.sum()),
                           parents=None if parents is None else len(np.unique(parents[mask])),
                           Q=scalar_summary(observables['Q'][mask]), PSD=scalar_summary(observables['PSD'][mask]),
                           raw_PSD=scalar_summary(observables['PSD_raw'][mask]),
                           guarded_PSD_rows=int(observables['PSD_guarded'][mask].sum()),
                           nonpositive_Q_rows=int(observables['Q_nonpositive'][mask].sum())))
    return result


def _validate_split(data, split):
    ns, nr = (6593, 21047) if split == 'TRAIN' else (512, 599)
    required = {'source': (ns, 256), 'source_carrier384': (ns, 384), 'real': (nr, 256),
                'real_baseline100': (nr, 100), 'source_ids': (ns,), 'real_ids': (nr,), 'real_groups': (nr,)}
    for key, shape in required.items():
        if key not in data or np.asarray(data[key]).shape != shape:
            raise ValueError('Wrong frozen ' + split + ' field: ' + key)
        if np.asarray(data[key]).dtype.kind in 'fc' and not np.isfinite(data[key]).all():
            raise ValueError('Nonfinite ' + split + ' ' + key)
    if len(np.unique(data['source_ids'])) != ns or len(np.unique(data['real_ids'])) != nr:
        raise ValueError('Nonunique original parent identities')


def _load_dev(container, identities, spec, q_floor):
    """Only called after the complete endpoint/evaluator freeze receipt exists."""
    window_spec(spec)
    old = load_validation(container, identities)
    result = {key: value for key, value in old.items() if key not in ('source', 'real')}
    for name in ('source', 'real'):
        carrier = np.asarray(old[name], dtype=np.float64)[:, :384].copy()
        observed = crop_existing_values(carrier, q_floor=q_floor) if name == 'source' else observe_existing_carrier(carrier, q_floor=q_floor)
        result[name] = observed['waveform256']
        result[name + '_carrier384'] = carrier
        result[name + '_baseline100'] = observed['baseline100']
        result[name + '_peak_index480'] = observed['peak_index']
        result[name + '_peak_search_edge'] = observed['flags']['peak_at_search_boundary']
        result[name + '_global_peak_outside_search'] = observed['flags']['global_peak_outside_search']
    return result


def _fixed_settings(cfg):
    ev = cfg['evaluation']
    result = dict(draws_per_parent=int(ev['draws_per_parent']), chunk_rows=int(ev['chunk_rows']),
                  matched_TRAIN=int(ev['matched_TRAIN_per_class']), matched_DEV=int(ev.get('matched_DEV_per_class', 256)),
                  sample_waveform_rows=int(ev['sample_waveform_rows']), bootstrap_draws=int(ev['bootstrap_draws']),
                  minimum_parents_per_bin=int(ev['minimum_parents_per_bin']))
    expected = dict(draws_per_parent=4, chunk_rows=128, matched_TRAIN=2048, matched_DEV=256,
                    sample_waveform_rows=128, bootstrap_draws=1000, minimum_parents_per_bin=32)
    if result != expected or tuple(cfg['arms']) != ARMS or cfg['response']['updates_per_arm'] != 19000:
        raise ValueError('Frozen Window008 evaluation settings differ from the registered endpoint')
    if ev['C2ST_updates'] != 0 or ev['scientific_pass_threshold'] is not None:
        raise ValueError('No classifier optimization or scientific threshold is registered')
    window_spec(cfg['window'])
    return result


def _memory_limit(cfg, device):
    """Apply the registered CUDA allocator bound before creating GPU models."""
    requested = float(cfg['gpu_memory_limit_GiB'])
    if not np.isfinite(requested) or requested <= 0:
        raise ValueError('The registered GPU memory limit must be positive and finite')
    limit_bytes = int(requested * 1024 ** 3)
    target = torch.device(device)
    receipt = dict(requested_GiB=requested, requested_bytes=limit_bytes,
                   CUDA_allocator_limit_applied=False, physical_GPU_bytes=None, allocator_fraction=None)
    if target.type == 'cpu':
        return receipt
    if target.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Window008 evaluation requires an available CUDA device or a CPU fixture')
    physical = int(torch.cuda.get_device_properties(target).total_memory)
    if physical < limit_bytes:
        raise ValueError('The selected CUDA device has less memory than the registered GPU limit')
    fraction = limit_bytes / physical
    torch.cuda.set_per_process_memory_fraction(fraction, target)
    receipt.update(CUDA_allocator_limit_applied=True, physical_GPU_bytes=physical, allocator_fraction=fraction)
    return receipt


def training_completion_gate(run, cfg):
    """Require native success and the complete three-arm training hash chain.

    A copied local run may be evaluated, so the launch command is compared with
    the registered remote output path rather than the evaluator's local path.
    """
    native_path = run.parent / 'supervisor' / 'exit_receipt.json'
    launch_path = run.parent / 'supervisor' / 'launch.json'
    complete_path = run / 'TRAINING_COMPLETE.json'
    binding_path = run / 'BINDING.json'
    manifest_path = _resolve(cfg['source_manifest'])
    config_path = ROOT / 'window008_config.json'
    paths = (native_path, launch_path, complete_path, binding_path, manifest_path, config_path)
    hashes = {str(path): sha256(path) for path in paths}
    native, launch, complete, binding, manifest, registered_config = (
        json.loads(path.read_text(encoding='utf8')) for path in paths)
    if native.get('exit_code') != 0 or native.get('reason') is not None:
        raise ValueError('Native training exit0 without a bounded-stop reason is required')
    command = launch.get('command', [])
    expected_output = cfg['remote_root'].rstrip('/') + '/runs/science'
    def argument(flag):
        return command[command.index(flag) + 1] if flag in command and command.index(flag) + 1 < len(command) else None
    if (argument('-m') != 'n01.train_window008' or argument('--out') != expected_output
            or native.get('supervisor_pid') != launch.get('supervisor_pid')
            or native.get('child_pid') != launch.get('child_pid')
            or not isinstance(native.get('supervisor_pid'), int) or not isinstance(native.get('child_pid'), int)):
        raise ValueError('Native receipt is not bound to this registered Window008 training launch')
    config_hash, manifest_hash = hashes[str(config_path)], hashes[str(manifest_path)]
    if (registered_config != cfg or manifest.get('run_id') != cfg['run_id']
            or manifest.get('before_scientific_model_forward') is not True
            or manifest.get('config_sha256') != config_hash
            or complete.get('config_sha256') != config_hash or complete.get('source_manifest_sha256') != manifest_hash
            or binding.get('run_id') != cfg['run_id'] or binding.get('config_sha256') != config_hash
            or binding.get('source_manifest_sha256') != manifest_hash
            or binding.get('dataset_sha256') != cfg['dataset_sha256']
            or binding.get('statistics_sha256') != cfg['statistics_sha256']):
        raise ValueError('Training completion/config/source-manifest binding differs')
    if (complete.get('run_id') != cfg['run_id'] or complete.get('status') != 'fixed_window008_training_complete_pending_evaluation'
            or complete.get('G_updates') != 57000 or complete.get('D_updates') != 144000
            or any(complete.get(key) != 0 for key in ('C2ST_updates', 'noise_updates', 'population_updates'))
            or complete.get('DEV_waveforms_read') is not False or complete.get('evaluation_complete') is not False
            or cfg['response']['total_G_updates'] != 57000 or cfg['response']['total_D_updates'] != 144000):
        raise ValueError('Complete native three-arm training accounting is required')
    rows = complete.get('arms', [])
    if len(rows) != len(ARMS) or {row.get('arm') for row in rows} != set(ARMS):
        raise ValueError('Complete three-arm endpoint list is required')
    endpoints = {}
    for row in rows:
        arm = row['arm']; relative = 'train/' + arm + '/final.pt'; path = run / relative
        digest = sha256(path)
        if (row.get('G_updates') != 19000 or row.get('D_updates') != {h: 16000 for h in HYPOTHESES}
                or row.get('coverage', {}).get('source_unique') != 6593
                or row.get('coverage', {}).get('real_unique') != 21047
                or row.get('final_sha256') != digest or complete.get('files', {}).get(relative) != digest):
            raise ValueError('Completed arm/final checkpoint binding differs: ' + arm)
        endpoints[arm] = digest; hashes[str(path)] = digest
    if not isinstance(manifest.get('files'), dict) or not manifest['files']:
        raise ValueError('Training source manifest has no frozen files')
    for relative, digest in manifest['files'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256(path) != digest:
            raise ValueError('A frozen training source/input changed: ' + relative)
        hashes[str(path)] = digest
    return dict(files_sha256=hashes, endpoint_files_sha256=endpoints, native_exit_code=0,
                completion_G_updates=57000, completion_D_updates=144000,
                native_receipt=str(native_path), launch_receipt=str(launch_path),
                training_completion=str(complete_path), training_source_manifest=str(manifest_path))


def _freeze(run, cfg, statistics, dataset_path, statistics_path, raw_container, identities, device, destination):
    settings = _fixed_settings(cfg)
    completion = training_completion_gate(run, cfg)
    memory_limit = _memory_limit(cfg, device)
    if sha256(raw_container) != cfg['train_container_sha256'] or sha256(identities) != cfg['identities_sha256']:
        raise ValueError('Original evaluation input bindings changed')
    if statistics.get('statistics_split', statistics.get('split')) != 'TRAIN' or statistics['loss']['fit_split'] != 'TRAIN':
        raise ValueError('All evaluation scales and bins must be frozen TRAIN statistics')
    if statistics_path is not None and sha256(statistics_path) != cfg['statistics_sha256']:
        raise ValueError('Final statistics bytes differ from the registered training binding')
    if sha256(dataset_path) != statistics['prepared_dataset_sha256']:
        raise ValueError('Prepared TRAIN bytes differ from the final statistics binding')
    data_statistics_path = dataset_path.with_name('FIT_DATA_STATISTICS.json')
    if sha256(data_statistics_path) != statistics['data_statistics_sha256']:
        raise ValueError('Prepared SOURCE/REAL TRAIN statistics changed')
    data_statistics = json.loads(data_statistics_path.read_text(encoding='utf8'))
    if data_statistics.get('split') != 'TRAIN' or data_statistics.get('DEV_waveforms_decoded') is not False:
        raise ValueError('SOURCE Q8 edges require the original TRAIN-only preparation statistics')
    noise_path = _resolve(cfg['noise_reference']['path'])
    if sha256(noise_path) != cfg['noise_reference']['sha256']:
        raise ValueError('Frozen005 file changed')
    noise_checkpoint = torch.load(noise_path, map_location='cpu', weights_only=False)
    extension = make_extension(noise_checkpoint, device)
    noise_digest = state_digest(extension.model005.state_dict())
    models, files, state_hashes, common = {}, dict(completion['files_sha256']), {}, []
    for arm in ARMS:
        path = run / 'train' / arm / 'final.pt'
        before = sha256(path)
        if before != completion['endpoint_files_sha256'][arm]:
            raise ValueError('Endpoint changed after native completion verification: ' + arm)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if (checkpoint.get('schema') != 'response008/v1' or checkpoint['arm'] != arm or checkpoint['G_updates'] != 19000
                or checkpoint['D_updates'] != {h: 16000 for h in HYPOTHESES} or checkpoint['stage'] != 'low_learning_rate'
                or checkpoint['config'] != cfg or _json_sha(checkpoint['statistics']) != _json_sha(statistics)
                or checkpoint['noise005_state_sha256'] != noise_digest or checkpoint.get('noise_updates') != 0
                or checkpoint.get('population_learning') is not False or sha256(path) != before):
            raise ValueError('Wrong or changing fixed endpoint: ' + arm)
        model = N01Response008(statistics['model'], arm)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.set_stage('low_learning_rate'); model.requires_grad_(False); model.eval(); model.to(device)
        models[arm] = model; files[str(path)] = before; state_hashes[arm] = state_digest(model.state_dict())
        common.append(checkpoint['common_initial_state_sha256'])
    if len(set(common)) != 1:
        raise ValueError('Three endpoints did not share the registered common initializer')
    sources = {str(path): sha256(path) for path in sorted((ROOT / 'src' / 'n01').glob('*.py'))}
    manifest = _resolve(cfg['source_manifest'])
    sources[str(manifest)] = sha256(manifest)
    files.update({str(noise_path): sha256(noise_path), str(dataset_path): sha256(dataset_path),
                  str(data_statistics_path): sha256(data_statistics_path),
                  str(raw_container): cfg['train_container_sha256'], str(identities): cfg['identities_sha256']})
    if statistics_path is not None:
        files[str(statistics_path)] = sha256(statistics_path)
    receipt = dict(schema='n01/window008-evaluation-freeze/v1', UTC=utc(), status='all_endpoints_frozen_before_DEV_numeric_read',
                   checkpoint_state_sha256=state_hashes, noise_state_sha256=noise_digest, files_sha256=files,
                   sources_sha256=sources, config_sha256=_json_sha(cfg), statistics_sha256=_json_sha(statistics),
                   resolved_settings=settings, DEV_numeric_decoded=False, optimizer_updates=0, C2ST_updates=0,
                   common_initial_state_sha256=common[0], config=cfg, native_training_completion=completion,
                   runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                device=str(device), CUDA_build=torch.version.cuda,
                                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                cudnn_benchmark=torch.backends.cudnn.benchmark,
                                matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
                                cudnn_TF32=torch.backends.cudnn.allow_tf32,
                                CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                                cpu_threads=torch.get_num_threads(), GPU_memory_limit=memory_limit))
    save_json(destination / 'FROZEN_INPUTS.json', receipt)
    return models, extension, receipt, settings, data_statistics


def _verify_frozen(models, extension, receipt):
    extension.assert_frozen()
    if state_digest(extension.model005.state_dict()) != receipt['noise_state_sha256']:
        raise RuntimeError('Evaluation mutated005')
    for arm, model in models.items():
        if state_digest(model.state_dict()) != receipt['checkpoint_state_sha256'][arm]:
            raise RuntimeError('Evaluation mutated response: ' + arm)
    for path, expected in {**receipt['files_sha256'], **receipt['sources_sha256']}.items():
        if sha256(path) != expected:
            raise RuntimeError('A frozen evaluation input changed: ' + path)


@torch.no_grad()
def generate_split(split, data, models, extension, statistics, cfg, settings, destination, selected, device):
    n, draws, chunk = len(data['source']), settings['draws_per_parent'], settings['chunk_rows']
    parents = np.repeat(np.arange(n, dtype=np.int64), draws); draw = np.tile(np.arange(draws), n)
    qfloor = float(statistics['loss']['q_floor'])
    source_q = np.asarray(data['source'], np.float64).sum(1)
    sbins = source_bins(source_q, statistics); repeated_sbins = sbins[parents]
    streams = PairedStreams(*(cfg['seeds'][split + '_' + key] for key in ('response', 'noise005', 'extension')))
    initial = streams.state_dict()
    pieces = {a: {h: [] for h in HYPOTHESES} for a in ARMS}; signal_pieces = {a: [] for a in ARMS}
    moments = {a: {h: Moments(256) for h in HYPOTHESES} for a in ARMS}
    conditional_moments = {a: {h: [[Moments(256) for _ in range(8)] for _ in SCHEMES] for h in HYPOTHESES} for a in ARMS}
    baseline_moments = Moments(100)
    matched = {a: {h: np.empty((len(selected), 256), np.float64) for h in HYPOTHESES} for a in ARMS}
    matched_baseline = np.empty((len(selected), 100), np.float64)
    lookup = {int(row): position for position, row in enumerate(selected)}; written = np.zeros(len(selected), bool)
    samples, chunks = {}, []
    engineering = {a: dict(signal_prefix_absmax_V=0.,observed_Q_identity_absmax=0.,observed_tail_identity_absmax=0.,
                           canonical_charge_identity_absmax=0.,baseline_shared=True) for a in ARMS}
    for start in range(0, len(parents), chunk):
        stop = min(start + chunk, len(parents)); pr = parents[start:stop]
        response, noise_latents, epsilon = streams.draw(stop - start, device)
        noises = fixed_noises(extension, noise_latents, epsilon)
        source = torch.as_tensor(data['source_carrier384'][pr], dtype=torch.float64, device=device)
        record = dict(start=start, stop=stop, source_parent_rows_sha256=array_sha(pr),
                      response_sha256={k: array_sha(v) for k, v in response.items()},
                      noise_latents_sha256={k: array_sha(v) for k, v in noise_latents.items()},
                      extension_epsilon_sha256=array_sha(epsilon), outputs={})
        picked = [(j, lookup[int(parent)]) for j, parent in enumerate(pr) if draw[start + j] == 0 and int(parent) in lookup]
        common_baseline = None
        for arm, model in models.items():
            output = model.signal_forward(source, response); signal = output['signal_pre']
            eng = engineering[arm]
            eng['signal_prefix_absmax_V'] = max(eng['signal_prefix_absmax_V'], float(signal[:, :100].abs().max()))
            eng['canonical_charge_identity_absmax'] = max(eng['canonical_charge_identity_absmax'], float(output['canonical_charge_identity_residual'].abs().max()))
            if eng['signal_prefix_absmax_V'] > cfg['engineering']['signal_prefix_max_V'] or not bool(torch.isfinite(signal).all()):
                raise RuntimeError('Signal prefix or finite-output contract failed: ' + arm)
            common_signal = {k: _array(output[k]) for k in ('qsig', 'qbar', 'time_scale', 'canonical_charge',
                             'canonical_charge_identity_residual', 'canonical_local_charge', 'query_outside_carrier', 'beyond_extension')}
            signal_pieces[arm].append(common_signal); record['outputs'][arm] = {}
            for hypothesis in HYPOTHESES:
                noise = noises[hypothesis]; pre = signal + noise
                observation = observe_generated_carrier(pre, ste=False, q_floor=qfloor, return_carrier=True)
                baseline = _array(observation['baseline100'])
                if common_baseline is None:
                    common_baseline = baseline
                if not np.array_equal(baseline, common_baseline):
                    eng['baseline_shared'] = False
                    raise RuntimeError('Electronics/ADC baseline changed across arms or hypotheses')
                indices = observation['crop_indices']; adc_delta = observation['adc_carrier'] - pre
                terms = {}
                for name, values in (('signal', signal), ('noise', noise), ('ADC_delta', adc_delta)):
                    window = values.gather(1, indices) - values[:, :100].mean(1, keepdim=True)
                    terms['Q_' + name] = _array(window.sum(1)); terms['tail_' + name] = _array(window[:, 46:].sum(1))
                y = _array(observation['waveform256'])
                obs = {k: _array(observation[k]) for k in ('Q', 'tail', 'PSD', 'PSD_raw', 'peak_index', 'global_peak_index', 'crop_start', 'crop_stop', 'baseline_mean_V')}
                obs.update({k: _array(v) for k, v in observation['flags'].items()}); obs.update(terms)
                obs['Q_signal_crop_migration'] = obs['Q_signal'] - common_signal['qsig']
                obs['Q_identity_residual'] = obs['Q'] - obs['Q_signal'] - obs['Q_noise'] - obs['Q_ADC_delta']
                obs['tail_identity_residual'] = obs['tail'] - obs['tail_signal'] - obs['tail_noise'] - obs['tail_ADC_delta']
                eng['observed_Q_identity_absmax'] = max(eng['observed_Q_identity_absmax'], float(np.abs(obs['Q_identity_residual']).max()))
                eng['observed_tail_identity_absmax'] = max(eng['observed_tail_identity_absmax'], float(np.abs(obs['tail_identity_residual']).max()))
                if not np.isfinite(y).all() or any(not np.isfinite(obs[k]).all() for k in ('Q', 'tail', 'PSD')):
                    raise RuntimeError('Nonfinite observed endpoint; no row deletion')
                pieces[arm][hypothesis].append(obs); moments[arm][hypothesis].add(y)
                bins = (output_bins(obs['Q'], statistics), repeated_sbins[start:stop])
                for scheme, membership in enumerate(bins):
                    for b in range(8):
                        conditional_moments[arm][hypothesis][scheme][b].add(y[membership == b])
                record['outputs'][arm][hypothesis] = dict(waveform256_sha256=array_sha(y), observables_sha256={k: array_sha(v) for k, v in obs.items()})
                for local, target in picked:
                    matched[arm][hypothesis][target] = y[local]
                if start == 0:
                    samples[arm + '_' + hypothesis] = y[:settings['sample_waveform_rows']]
                    samples['noise384_' + hypothesis] = _array(noise)[:settings['sample_waveform_rows']]
            if start == 0:
                samples[arm + '_signal_pre384'] = _array(signal)[:settings['sample_waveform_rows']]
        baseline_moments.add(common_baseline)
        for local, target in picked:
            matched_baseline[target] = common_baseline[local]; written[target] = True
        if start == 0:
            samples['baseline100_ADC'] = common_baseline[:settings['sample_waveform_rows']]
            samples['SOURCE_carrier384'] = _array(source)[:settings['sample_waveform_rows']]
            samples['source_parent_ids'] = data['source_ids'][pr[:settings['sample_waveform_rows']]]
            samples['draw'] = draw[start:stop][:settings['sample_waveform_rows']]
        record['baseline100_ADC_sha256'] = array_sha(common_baseline); chunks.append(record)
    if not written.all():
        raise RuntimeError('Matched SOURCE sample is incomplete')
    allobs, curves, summary = {}, {}, {}
    flat = dict(source_parent_ids=data['source_ids'], source_parent_row=parents, draw=draw, source_Q=source_q,
                source_bins=sbins, real_original_ids=data['real_ids'], real_original_groups=data['real_groups'])
    if 'h' in data:
        flat['source_h_metadata_only'] = data['h']
    for arm in ARMS:
        allobs[arm] = {}; summary[arm] = {}
        for key in signal_pieces[arm][0]:
            flat[arm + '_' + key] = np.concatenate([piece[key] for piece in signal_pieces[arm]])
        for hypothesis in HYPOTHESES:
            obs = {key: np.concatenate([piece[key] for piece in pieces[arm][hypothesis]]) for key in pieces[arm][hypothesis][0]}
            allobs[arm][hypothesis] = obs
            label = arm + '_' + hypothesis
            flat.update({label + '_' + k: v for k, v in obs.items()})
            curves[label + '_mean'], curves[label + '_std'] = moments[arm][hypothesis].values()
            obins = output_bins(obs['Q'], statistics)
            summary[arm][hypothesis] = dict(population={k: (dict(true_rows=int(v.sum()), rows=len(v)) if v.dtype.kind == 'b' else scalar_summary(v)) for k, v in obs.items()},
                output_Q_in_REAL_TRAIN_bins=conditional_rows(obs, obins, parents),
                fixed_SOURCE_Q_in_SOURCE_TRAIN_bins=conditional_rows(obs, repeated_sbins, parents),
                SOURCE_to_output_bin_counts=np.bincount(repeated_sbins * 8 + obins, minlength=64).reshape(8, 8))
            for i, scheme in enumerate(SCHEMES):
                values = [m.values() for m in conditional_moments[arm][hypothesis][i]]
                curves[label + '_' + scheme + '_mean'] = np.asarray([v[0] for v in values])
                curves[label + '_' + scheme + '_std'] = np.asarray([v[1] for v in values])
                curves[label + '_' + scheme + '_rows'] = np.asarray([m.n for m in conditional_moments[arm][hypothesis][i]])
    real = readouts256(data['real'], qfloor)
    for key in ('peak_index480', 'peak_search_edge', 'global_peak_outside_search'):
        if 'real_' + key in data:
            real[key] = np.asarray(data['real_' + key])
    flat.update({'REAL_' + k: v for k, v in real.items()})
    curves['REAL_mean'], curves['REAL_std'] = data['real'].mean(0), data['real'].std(0, ddof=0)
    curves['generated_baseline100_mean'], curves['generated_baseline100_std'] = baseline_moments.values()
    curves['REAL_baseline100_mean'], curves['REAL_baseline100_std'] = data['real_baseline100'].mean(0), data['real_baseline100'].std(0, ddof=0)
    summary['REAL'] = dict(population={k: scalar_summary(v) for k, v in real.items()},
                           output_Q_in_REAL_TRAIN_bins=conditional_rows(real, output_bins(real['Q'], statistics)))
    save_npz(destination / (split + '_observables.npz'), **flat)
    save_npz(destination / (split + '_curves.npz'), **curves)
    save_npz(destination / (split + '_sample_waveforms.npz'), **samples)
    stream_values = {prefix + '_' + k: _array(v) for prefix, state in (('initial', initial), ('final', streams.state_dict())) for k, v in state.items()}
    save_npz(destination / (split + '_stream_states.npz'), **stream_values)
    save_json(destination / (split + '_generation.json'), dict(UTC=utc(), split=split, parents=n, draws_per_parent=draws,
        rows=len(parents), response_forward_rows_per_arm=len(parents), noise005_forward_rows=len(parents),
        rows_removed=0, chunks=chunks, summary=summary, engineering=engineering, matched_source_rows=selected,
        sample_rows=min(settings['sample_waveform_rows'], chunk), extra_sample_forwards=0, scientific_pass_threshold=None))
    return dict(observables=allobs, real=real, parents=parents, source_bins=repeated_sbins,
                matched=matched, matched_baseline=matched_baseline, engineering=engineering)


def _percentile_interval(samples):
    values = np.asarray(samples); shape = values.shape[1:]
    flat = values.reshape(len(values), -1); result = np.full((2, flat.shape[1]), np.nan)
    defined = np.isfinite(flat).sum(0)
    for index in np.flatnonzero(defined):
        result[:, index] = np.quantile(flat[np.isfinite(flat[:, index]), index], [.025, .975])
    return result.reshape((2,) + shape), defined.reshape(shape)


def parent_bootstrap(generated, statistics, cfg, settings, split):
    """Whole SOURCE parents, paired across arms/hypotheses; REAL held fixed.

    Membership is frozen from endpoint values. Output-Q bins may contain only
    some draws of a parent; the parent's single bootstrap weight covers them all.
    """
    parents = generated['parents']; n = int(parents.max()) + 1
    shape = (len(SCHEMES), len(ARMS), len(HYPOTHESES), 8)
    counts = np.zeros(shape + (n,), np.float64)
    parent_mean = np.zeros(shape + (2, n), np.float64)
    parent_m2 = np.zeros_like(parent_mean)
    parent_min = np.full_like(parent_mean, np.inf); parent_max = np.full_like(parent_mean, -np.inf)
    centers = np.zeros((len(ARMS), len(HYPOTHESES), 2), np.float64)
    support = np.zeros(shape, np.int64)
    for a, arm in enumerate(ARMS):
        for h, hypothesis in enumerate(HYPOTHESES):
            obs = generated['observables'][arm][hypothesis]
            memberships = (output_bins(obs['Q'], statistics), generated['source_bins'])
            for m, metric in enumerate(METRICS):
                centers[a, h, m] = obs[metric].mean()
            for s, bins in enumerate(memberships):
                for b in range(8):
                    mask = bins == b; pr = parents[mask]
                    counts[s, a, h, b] = np.bincount(pr, minlength=n)
                    support[s, a, h, b] = len(np.unique(pr))
                    for m, metric in enumerate(METRICS):
                        values = obs[metric][mask]
                        delta = values - centers[a, h, m]
                        count = counts[s, a, h, b]
                        mean = parent_mean[s, a, h, b, m]
                        np.divide(np.bincount(pr, weights=delta, minlength=n), count, out=mean, where=count > 0)
                        minimum, maximum = parent_min[s, a, h, b, m], parent_max[s, a, h, b, m]
                        np.minimum.at(minimum, pr, values); np.maximum.at(maximum, pr, values)
                        constant = (count > 0) & (minimum == maximum)
                        mean[constant] = minimum[constant] - centers[a, h, m]
                        parent_m2[s, a, h, b, m] = np.bincount(pr, weights=(delta - mean[pr]) ** 2, minlength=n)

    def calculate(weights):
        # Combine within-parent M2 and between-parent mean deviations. Direct
        # E[x^2] - E[x]^2 cancellation can invent a positive width in a sparse
        # bootstrap bin whose only observation was sampled multiple times.
        weighted_count = counts * weights
        denominator = weighted_count.sum(-1)[..., None]
        total = (parent_mean * weighted_count[..., None, :]).sum(-1)
        centered_mean = np.divide(total, denominator, out=np.zeros_like(total), where=denominator > 0)
        squared_deviation = (parent_mean - centered_mean[..., None]) ** 2
        m2 = ((parent_m2 + counts[..., None, :] * squared_deviation) * weights).sum(-1)
        variance = np.divide(m2, denominator, out=np.full_like(total, np.nan), where=denominator > 0)
        mean = centered_mean + centers[None, :, :, None, :]
        mean[denominator[..., 0] == 0] = np.nan
        selected = weighted_count[..., None, :] > 0
        minimum = np.min(np.where(selected, parent_min, np.inf), axis=-1)
        maximum = np.max(np.where(selected, parent_max, -np.inf), axis=-1)
        constant = (denominator > 0) & (minimum == maximum)
        mean[constant] = minimum[constant]; variance[constant] = 0.
        return mean, np.sqrt(variance)

    point_mean, point_std = calculate(np.ones(n))
    rng = np.random.default_rng(cfg['seeds']['bootstrap_' + split])
    means = np.empty((settings['bootstrap_draws'],) + point_mean.shape)
    stds = np.empty_like(means)
    for draw in range(settings['bootstrap_draws']):
        weights = np.bincount(rng.integers(0, n, n), minlength=n)
        means[draw], stds[draw] = calculate(weights)
    ci_mean, defined_mean = _percentile_interval(means); ci_std, defined_std = _percentile_interval(stds)
    full = ARMS.index('full'); comparisons = {}; payload = dict(mean=means, std=stds)
    for comparator in COMPARATORS:
        base = ARMS.index(comparator)
        mean_delta = means[:, :, full] - means[:, :, base]
        std_delta = stds[:, :, full] - stds[:, :, base]
        ratio = np.divide(stds[:, :, full], stds[:, :, base], out=np.full_like(std_delta, np.nan), where=stds[:, :, base] > 0)
        point_ratio = np.divide(point_std[:, full], point_std[:, base], out=np.full_like(point_std[:, full], np.nan), where=point_std[:, base] > 0)
        records = {}
        for name, values, point in (('mean_full_minus_reference', mean_delta, point_mean[:, full] - point_mean[:, base]),
                                    ('std_full_minus_reference', std_delta, point_std[:, full] - point_std[:, base]),
                                    ('std_full_over_reference', ratio, point_ratio)):
            interval, defined = _percentile_interval(values)
            records[name] = dict(point=point, percentile95=interval, defined_replicates=defined)
            payload[comparator + '_' + name] = values
        comparisons[comparator] = records
    real_scales = np.asarray([np.std(generated['real'][key], ddof=0) for key in METRICS])
    hypothesis_mean_range = np.ptp(means, axis=3)
    point_mean_range = np.ptp(point_mean, axis=2)
    hypothesis_width_ratio = np.divide(stds.max(3), stds.min(3),
        out=np.full_like(stds.max(3), np.nan), where=stds.min(3) > 0)
    point_width_ratio = np.divide(point_std.max(2), point_std.min(2),
        out=np.full_like(point_std.max(2), np.nan), where=point_std.min(2) > 0)
    normalized_mean_range = np.divide(hypothesis_mean_range, real_scales,
        out=np.full_like(hypothesis_mean_range, np.nan), where=real_scales > 0)
    normalized_point = np.divide(point_mean_range, real_scales,
        out=np.full_like(point_mean_range, np.nan), where=real_scales > 0)
    sensitivity = {}
    for name, values, point in (('hypothesis_mean_range', hypothesis_mean_range, point_mean_range),
                                ('hypothesis_mean_range_over_REAL_population_std', normalized_mean_range, normalized_point),
                                ('hypothesis_std_max_over_min', hypothesis_width_ratio, point_width_ratio)):
        interval, defined = _percentile_interval(values)
        sensitivity[name] = dict(point=point, percentile95=interval, defined_replicates=defined)
        payload[name] = values
    supported = support >= settings['minimum_parents_per_bin']
    report = dict(schema='n01/window008-parent-bootstrap/v1', split=split, schemes=SCHEMES, arms=ARMS,
        hypotheses=HYPOTHESES, metrics=METRICS, axes=['scheme', 'arm', 'hypothesis', 'Q_bin', 'metric'],
        point_mean=point_mean, point_std=point_std, mean_percentile95=ci_mean, std_percentile95=ci_std,
        mean_defined_replicates=defined_mean, std_defined_replicates=defined_std,
        parents_per_bin=support, rows_per_bin=counts.sum(-1), supported=supported,
        minimum_parents_per_bin=settings['minimum_parents_per_bin'], bootstrap_draws=settings['bootstrap_draws'],
        comparisons= comparisons, comparison_axes=['scheme', 'hypothesis', 'Q_bin', 'metric'],
        comparison_supported={c: supported[:, full] & supported[:, ARMS.index(c)] for c in COMPARATORS},
        hypothesis_sensitivity=sensitivity, hypothesis_sensitivity_axes=['scheme', 'arm', 'Q_bin', 'metric'],
        hypothesis_sensitivity_supported=supported.all(2), REAL_population_std=real_scales,
        method='SOURCE parent bootstrap paired across all arms and hypotheses; four draws stay together',
        bins_frozen_before_bootstrap=True, REAL_resampled=False, bootstrap_refits=0, rows_removed=0,
        unsupported_bins='Retained and marked; intervals are descriptive and not a support pass',
        interpretation='Width/mean differences are not response superiority or physical identification.',
        scientific_pass_threshold=None)
    return report, payload


@torch.no_grad()
def distance_receipt(objective, fake, real, fake_baseline, real_baseline, context):
    device = next(objective.buffers()).device
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    parts = objective.components(dict(waveform=tensor(fake), baseline100=tensor(fake_baseline)),
        tensor(real), stage='joint_stochastic', projection_context=context, real_baseline100=tensor(real_baseline))
    values = {key: float(parts[key]) for key in ('wave', 'QPSD', 'noise')}
    if not all(np.isfinite(value) for value in values.values()):
        raise RuntimeError('Nonfinite matched-population distance')
    return dict(components=values, wave_subcomponents={key: float(value) for key, value in objective.last_wave_components.items()},
                conditional_diagnostics=copy.deepcopy(objective.last_diagnostics), matched_rows=[len(fake), len(real)],
                explicit_baseline_samples=100, projection_context=context.metadata(), scientific_pass_threshold=None)


def evaluate_run(out, cfg, dataset, statistics, raw_container, identities, device='cuda:0', *, destination=None):
    """Evaluate out/train/<arm>/final.pt; every output goes to fresh out/evaluation.

    cfg/statistics accept dictionaries or JSON paths. Dataset must be the saved
    TRAIN-only DATASET008.npz, so its bytes can be frozen before DEV is accessed.
    """
    run = Path(out).resolve()
    if destination is None:
        destination = run / 'evaluation'
    else:
        destination = Path(destination).resolve()
        replay_root = run.parent / 'evaluation_jobs'
        if destination == replay_root or not destination.is_relative_to(replay_root) or destination.is_relative_to(run):
            raise ValueError('Alternate evaluation output must be a fresh child of run.parent/evaluation_jobs')
    if run.drive.lower() == 'c:' or destination.drive.lower() == 'c:':
        raise ValueError('Window008 artifacts must be on E/D or the registered remote workspace')
    cfg = _read_json(cfg); statistics_path = _resolve(statistics) if isinstance(statistics, (str, Path)) else None
    statistics = _read_json(statistics); dataset_path = _resolve(dataset)
    raw_container, identities = _resolve(raw_container), _resolve(identities)
    destination.mkdir(parents=True, exist_ok=False)
    dev_decoded = False; dev_decode_started = False
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(int(cfg['cpu_threads']))
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        models, extension, freeze, settings, data_statistics = _freeze(run, cfg, statistics, dataset_path, statistics_path,
                                                                     raw_container, identities, device, destination)
        # These already-fitted domain summaries are separately hash-bound; they
        # are not added to checkpoint statistics or refitted during evaluation.
        evaluation_statistics = copy.deepcopy(statistics)
        evaluation_statistics['domains'] = data_statistics['domains']
        with np.load(dataset_path, allow_pickle=False) as archive:
            if {'source_val', 'target_val', 'hval'} & set(archive.files):
                raise ValueError('Prepared dataset unexpectedly contains DEV numerical values')
            train = {key: archive[key].copy() for key in archive.files}
        _validate_split(train, 'TRAIN')
        # This is the first and only DEV waveform decode in the evaluator.
        dev_decode_started = True
        dev = _load_dev(raw_container, identities, cfg['window'], float(statistics['loss']['q_floor']))
        dev_decoded = True; _validate_split(dev, 'DEV')
        if (np.intersect1d(train['source_ids'], dev['source_ids']).size or np.intersect1d(train['real_ids'], dev['real_ids']).size
                or np.intersect1d(train['real_groups'], dev['real_groups']).size):
            raise ValueError('TRAIN/DEV parent or REAL-group overlap')
        if not np.array_equal(train['source_val_ids'], dev['source_ids']) or not np.array_equal(train['real_val_original_index'], dev['real_ids']):
            raise ValueError('DEV lineage differs from the prepared identity-only binding')
        objective = N01Loss008(loss_tensor_statistics(statistics['loss']), seed=cfg['seeds']['train_projection'],
                              evaluation_seed=cfg['seeds']['evaluation_projection']).to(device).eval().requires_grad_(False)
        context = objective.create_evaluation_context()
        save_npz(destination / 'evaluation_projection_bank.npz', **{key: _array(value) for key, value in context.banks.items()})
        selections, distances, comparisons, split_results = {}, {}, {}, {}
        for split, data in (('TRAIN', train), ('DEV', dev)):
            count = settings['matched_' + split]
            selected = np.random.default_rng(cfg['seeds']['selected_' + split]).permutation(len(data['source']))[:count]
            a, b = balanced_real_domains(data['real_groups'], count, cfg['seeds']['real_' + split])
            selections[split] = dict(SOURCE_rows=selected, SOURCE_parent_ids=data['source_ids'][selected],
                REAL_A_rows=a, REAL_B_rows=b, REAL_A_original_ids=data['real_ids'][a], REAL_B_original_ids=data['real_ids'][b],
                REAL_A_groups=data['real_groups'][a], REAL_B_groups=data['real_groups'][b])
            generated = generate_split(split, data, models, extension, evaluation_statistics, cfg, settings, destination, selected, device)
            bootstrap, replicates = parent_bootstrap(generated, evaluation_statistics, cfg, settings, split)
            save_json(destination / (split + '_PARENT_BOOTSTRAP.json'), bootstrap)
            save_npz(destination / (split + '_parent_bootstrap.npz'), **replicates)
            distances[split] = {arm: {h: distance_receipt(objective, generated['matched'][arm][h], data['real'][a],
                generated['matched_baseline'], data['real_baseline100'][a], context) for h in HYPOTHESES} for arm in ARMS}
            distances[split]['REAL_REAL'] = distance_receipt(objective, data['real'][b], data['real'][a],
                                                           data['real_baseline100'][b], data['real_baseline100'][a], context)
            comparisons[split] = {}
            for reference in COMPARATORS:
                comparisons[split][reference] = {h: {metric: dict(
                    full_minus_reference=distances[split]['full'][h]['components'][metric] - distances[split][reference][h]['components'][metric],
                    relative_distance_reduction=(1 - distances[split]['full'][h]['components'][metric] / distances[split][reference][h]['components'][metric])
                    if distances[split][reference][h]['components'][metric] > 0 else None)
                    for metric in ('wave', 'QPSD', 'noise')} for h in HYPOTHESES}
            matched = {arm + '_' + h: generated['matched'][arm][h] for arm in ARMS for h in HYPOTHESES}
            matched.update(generated_baseline100=generated['matched_baseline'], SOURCE_parent_ids=data['source_ids'][selected],
                REAL_A=data['real'][a], REAL_B=data['real'][b], REAL_A_baseline100=data['real_baseline100'][a],
                REAL_B_baseline100=data['real_baseline100'][b], REAL_A_original_ids=data['real_ids'][a], REAL_B_original_ids=data['real_ids'][b])
            save_npz(destination / (split + '_matched_waveforms.npz'), **matched)
            split_results[split] = dict(source_parents=len(data['source']), REAL_rows=len(data['real']),
                generated_rows_per_arm_hypothesis=len(generated['parents']), engineering=generated['engineering'],
                all_observation_rows_retained=True, bootstrap_draws=settings['bootstrap_draws'])
            context.validate()
            del generated, replicates
        save_json(destination / 'MATCHED_SELECTIONS.json', dict(selections=selections,
            REAL_rule='equal group quotas with randomized remainder; disjoint A/B within each group',
            SOURCE_rule='fixed seeded parents without replacement, draw0 only', original_population_summary_separate=True))
        save_json(destination / 'DISTRIBUTION_DISTANCES.json', dict(distances=distances, full_vs_reference=comparisons,
            projection_context=context.metadata(), hypothesis_selection=False, averaging_as_uncertainty=False))
        _verify_frozen(models, extension, freeze)
        artifacts = {path.name: sha256(path) for path in sorted(destination.iterdir()) if path.is_file()}
        summary = dict(schema='n01/window008-evaluation/v1', UTC=utc(), status='fixed_endpoint_evaluation_complete',
            fixed_training_G_updates=57000, fixed_training_D_updates=144000, evaluation_G_updates=0,
            evaluation_noise_updates=0, population_updates=0, C2ST_updates=0, DEV_numeric_decoded=dev_decoded,
            split_results=split_results, full_vs_reference=comparisons, distances=distances,
            scientific_status='response_match_not_established', scientific_pass_threshold=None,
            independent_acquisition_available=False, DEV_previously_exposed=cfg.get('validation_previously_exposed', True),
            paired_event_response_truth_available=False, hypotheses_are_sensitivity_assumptions_not_confidence_set=True,
            source_population_learning=False, all_original_rows_preserved=True, frozen_inputs_rechecked=True,
            units=dict(waveform='V', charge='V ns', sample_interval_ns=1), artifact_sha256=artifacts,
            interpretation='Unpaired single-acquisition distribution/width comparisons do not establish physical detector-response identity.')
        save_json(destination / 'SUMMARY.json', summary)
        return _clean(summary)
    except Exception as error:
        save_json(destination / 'FAILED.json', dict(UTC=utc(), status='evaluation_failed_preserved',
            error_type=type(error).__name__, error=str(error), DEV_numeric_decoded=dev_decoded,
            DEV_numeric_decode_started=dev_decode_started,
            evaluation_optimizer_updates=0, C2ST_updates=0, partial_artifacts_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True); parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True); parser.add_argument('--statistics', required=True)
    parser.add_argument('--raw-container', required=True); parser.add_argument('--identities', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--destination', help='Fresh artifact directory below out.parent/evaluation_jobs for exact replay')
    args = parser.parse_args()
    evaluate_run(args.out, args.config, args.dataset, args.statistics, args.raw_container, args.identities, args.device,
                 destination=args.destination)


if __name__ == '__main__':
    main()
