"""One registered CPU inference example; no training, parameter search or retry."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def write_record(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    raw_config = config_path.read_bytes()
    if hashlib.sha256(raw_config).hexdigest() != args.config_sha256:
        raise ValueError('028 config SHA differs')
    config = json.loads(raw_config)
    root = Path(config['remote_root'])
    output = Path(args.output).absolute()
    if (config['schema'] != 'n01/wp6-028-config/v1' or config['status'] != 'frozen'
            or config_path != root/'wp6_028_config.json' or Path.cwd() != root
            or output != root/'runs/primary/artifacts'):
        raise ValueError('028 exact frozen owned config/cwd/output required')
    limits = config['resource_limits']
    pre_limit_modules = [name for name in sys.modules if name.split('.')[0] in ('torch', 'numpy', 'n01')]
    resource.setrlimit(resource.RLIMIT_AS, (limits['address_space_GiB']*1024**3,)*2)
    resource.setrlimit(resource.RLIMIT_CPU, (limits['CPU_soft_seconds'], limits['CPU_hard_seconds']))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['TMPDIR'] = str(root/'tmp')
    os.environ['XDG_CACHE_HOME'] = str(root/'cache')
    sys.dont_write_bytecode = True
    output.mkdir(parents=True, exist_ok=False)
    accounting = dict(attempted={key: 0 for key in config['budget_caps']},
        completed={key: 0 for key in config['budget_caps']}, inflight=None)
    started = time.monotonic()
    start = dict(schema='n01/wp6-028-example-start/v1', run_id=config['run_id'],
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), config_sha256=args.config_sha256,
        pid=os.getpid(), parent_pid=os.getppid(), CPU_affinity=sorted(os.sched_getaffinity(0)),
        hard_RLIMIT_AS=list(resource.getrlimit(resource.RLIMIT_AS)),
        hard_RLIMIT_CPU=list(resource.getrlimit(resource.RLIMIT_CPU)),
        scientific_imports_before_limits=pre_limit_modules, automatic_restart=False)
    write_record(output/'START.json', start)
    try:
        import numpy as np
        import torch
        sys.path.insert(0, str(root/'src'))
        from n01.inference028 import load_reference028, verified_path028
        from n01.model010 import tensor_state_digest010
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        global_rng = torch.get_rng_state().clone()
        response = load_reference028(config, device=args.device, accounting=accounting)
        verified_path028(config['contracts']['supervisor'])
        historical = json.loads(verified_path028(config['provenance']['original016_manifest']).read_text(encoding='utf8'))
        if len(historical['files']) != 116 or len(historical['project_import_closure']) != 37:
            raise ValueError('028 original016 provenance closure differs')
        accounting['inflight'] = dict(operation='one_source_NPZ_three_members',
            completion='unknown until all three loads return')
        accounting['attempted']['NPZ_archive_opens'] = 1
        with np.load(verified_path028(config['inputs']['dataset']), allow_pickle=False) as archive:
            accounting['completed']['NPZ_archive_opens'] = 1
            accounting['attempted']['NPZ_member_value_loads'] += 1
            source = archive['source_carrier384']
            accounting['completed']['NPZ_member_value_loads'] += 1
            accounting['attempted']['NPZ_member_value_loads'] += 1
            parent_ids = archive['source_ids']
            accounting['completed']['NPZ_member_value_loads'] += 1
            accounting['attempted']['NPZ_member_value_loads'] += 1
            roles = archive['source_role']
            accounting['completed']['NPZ_member_value_loads'] += 1
        accounting['inflight'] = None
        if (source.shape != (6593, 384) or source.dtype != np.dtype('<f8')
                or parent_ids.shape != (6593,) or parent_ids.dtype != np.dtype('<U75')
                or roles.shape != (6593,) or roles.dtype != np.dtype('<U5')
                or not np.all(roles == 'TRAIN')):
            raise ValueError('028 fixed TRAIN member schema differs')
        selected = config['example']['source_rows']
        x = torch.from_numpy(source[selected].copy())
        ids = parent_ids[selected].copy()
        input_digest = tensor_state_digest010({'source': x})
        del source, parent_ids, roles
        saved = []
        summaries = []
        observations = ('window_V', 'baseline100', 'baseline_mean_V', 'peak_index', 'global_peak_index',
            'crop_start', 'crop_stop', 'Q', 'tail', 'PSD_raw', 'PSD', 'crop_indices', 'carrier_adc', 'pre_adc_carrier')
        contributions = ('signal_pre', 'M0_reference_signal_pre_V', 'base_signal_pre_V', 'random_shape_pre_V',
            'noise_pre', 'prefix_pre', 'scale', 'shape', 'baseline_gaussian', 'baseline_scale', 'extension', 'charge')
        checks = []
        with torch.no_grad():
            for model_seed in config['example']['model_seeds']:
                reference = None
                for index, call in enumerate(config['example']['calls_per_seed']):
                    result = response.sample(x, config['example']['draws'], config['example']['sampling_seed'],
                        response.detector_config, call['hypothesis'], model_seed=model_seed, parent_ids=ids,
                        shape_enabled=call['shape_enabled'], charge_enabled=call['charge_enabled'], return_contributions=True)
                    c = result['contributions']
                    centered = result['carrier_adc']-result['baseline_mean_V'][:, None]
                    peaks = centered[:, 112:144].argmax(1)+112
                    if (not torch.equal(result['baseline100'], result['carrier_adc'][:, :100])
                            or not torch.equal(result['baseline_mean_V'], result['baseline100'].mean(1))
                            or not torch.equal(result['peak_index'], peaks)
                            or not torch.equal(result['crop_start'], peaks-16)
                            or not torch.equal(result['crop_stop'], peaks+240)
                            or not torch.equal(result['crop_indices'], peaks[:, None]-16+torch.arange(256))
                            or not torch.equal(result['window_V'], centered.gather(1, result['crop_indices']))
                            or not torch.equal(result['Q'], result['window_V'].sum(1))
                            or not torch.equal(result['tail'], result['window_V'][:, 46:].sum(1))
                            or not torch.equal(result['row_parent'], torch.tensor([0, 0, 1, 1]))
                            or not torch.equal(result['draw'], torch.tensor([0, 1, 0, 1]))
                            or not np.array_equal(result['parent_ids'], ids)
                            or not result['ADC_applied'] or result['input_kind'] != 'new_ADC'):
                        raise ValueError('028 observation/lineage assembly inconsistent')
                    nonzero = result['Q'] != 0
                    raw_psd = torch.where(nonzero, result['tail']/torch.where(nonzero, result['Q'], torch.ones_like(result['Q'])),
                        torch.full_like(result['Q'], float('nan')))
                    guarded = result['Q'].abs() < response.q_floor
                    protected = torch.where(result['Q'] < 0, -torch.full_like(result['Q'], response.q_floor),
                        torch.full_like(result['Q'], response.q_floor))
                    psd = result['tail']/torch.where(guarded, protected, result['Q'])
                    if (tensor_state_digest010({'x': raw_psd}) != tensor_state_digest010({'x': result['PSD_raw']})
                            or not torch.equal(psd, result['PSD']) or not torch.equal(guarded, result['flags']['PSD_guarded'])
                            or any(value.requires_grad for value in (*[result[k] for k in observations], *c.values()))):
                        raise ValueError('028 PSD/no-grad result differs')
                    tensors = {key: result[key] for key in observations}
                    tensors.update({'c_'+key: c[key] for key in contributions})
                    tensors.update({'flag_'+key: value for key, value in result['flags'].items()})
                    if index == 0:
                        reference = (tensors, result['streams'])
                    elif index in (1, 2, 3, 4, 5):
                        if result['streams'] != reference[1]:
                            raise ValueError('028 same-seed random libraries differ')
                        for key in ('c_prefix_pre', 'c_scale', 'c_shape', 'c_baseline_gaussian', 'c_baseline_scale', 'c_extension', 'c_charge'):
                            if tensor_state_digest010({'x': tensors[key]}) != tensor_state_digest010({'x': reference[0][key]}):
                                raise ValueError('028 common prefix/latent bytes differ: '+key)
                    if index == 3 and tensor_state_digest010(tensors) != tensor_state_digest010(reference[0]):
                        raise ValueError('028 exact fast replay differs')
                    if index == 4 and (c['signal_pre'] is not c['M0_reference_signal_pre_V']
                            or tensor_state_digest010({'x': c['signal_pre']}) != tensor_state_digest010({'x': reference[0]['c_M0_reference_signal_pre_V']})):
                        raise ValueError('028 chargeoff lost original M0 identity')
                    if index == 5 and (tensor_state_digest010({'x': c['M0_reference_signal_pre_V']}) !=
                            tensor_state_digest010({'x': c['base_signal_pre_V']}) or not result['charge_enabled']):
                        raise ValueError('028 shapeoff failed to retain A/charge semantics')
                    saved.append({key: value.cpu().numpy() for key, value in tensors.items()})
                    summaries.append(dict(model_seed=model_seed, case=call['name'], hypothesis=call['hypothesis'],
                        shape_enabled=call['shape_enabled'], charge_enabled=call['charge_enabled'],
                        result_sha256=tensor_state_digest010(tensors), streams=result['streams'],
                        flags={key: int(value.sum()) for key, value in result['flags'].items()}))
                    checks.append(dict(model_seed=model_seed, case=call['name'], passed=True))
                    write_record(output/'PROGRESS.json', dict(schema='n01/wp6-028-example-progress/v1',
                        completed_calls=len(saved), accounting=accounting))
        if tensor_state_digest010({'source': x}) != input_digest or not torch.equal(global_rng, torch.get_rng_state()):
            raise ValueError('028 input bytes or global CPU RNG changed')
        after = dict(A=tensor_state_digest010(response.models[0].baseline), extension=tensor_state_digest010(response.extension),
            **{f'model{index}': tensor_state_digest010(model) for index, model in response.models.items()})
        if after != response.frozen_digests or any(module.training
                for model in (*response.models.values(), response.extension) for module in model.modules()):
            raise ValueError('028 frozen state or eval mode changed')
        if any(p.requires_grad or p.grad is not None
                for model in (*response.models.values(), response.extension) for p in model.parameters()):
            raise ValueError('028 inference parameters opened')
        if accounting['completed'] != config['budget_caps'] or accounting['attempted'] != config['budget_caps']:
            raise ValueError('028 exact completed/attempted budget differs')
        arrays = {key: np.stack([row[key] for row in saved]) for key in saved[0]}
        arrays.update(parent_ids=ids, source_rows=np.asarray(selected, dtype=np.int64),
            row_parent=np.asarray([0, 0, 1, 1], dtype=np.int64), draw=np.asarray([0, 1, 0, 1], dtype=np.int64))
        with (output/'EXAMPLE.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        array_schema = {key: dict(dtype=value.dtype.str, shape=list(value.shape), values=int(value.size), bytes=value.nbytes)
            for key, value in arrays.items()}
        write_record(output/'ARRAY_SCHEMA.json', dict(schema='n01/wp6-028-example-arrays/v1',
            file='EXAMPLE.npz', member_count=len(arrays), members=array_schema,
            total_values=sum(value.size for value in arrays.values()), total_payload_bytes=sum(value.nbytes for value in arrays.values()),
            private_arrays_not_for_Git=True))
        imported = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
            if (name == 'n01' or name.startswith('n01.'))}
        if set(imported) != set(config['runtime_n01_modules']):
            raise ValueError('028 actual n01 import closure differs')
        for name, path in imported.items():
            relative = 'src/n01/__init__.py' if name == 'n01' else 'src/'+name.replace('.', '/')+'.py'
            if Path(path) != Path(config['source_files'][relative]['path']).resolve():
                raise ValueError('028 runtime import escaped frozen path: '+name)
        elapsed = time.monotonic()-started
        peak_RSS = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if elapsed > limits['wall_seconds'] or peak_RSS > limits['job_tree_RSS_GiB']*1024**3:
            raise ValueError('028 terminal wall/RSS cap exceeded')
        complete = dict(schema='n01/wp6-028-example-complete/v1', run_id=config['run_id'],
            native_status='COMPLETE', native_exit_code_on_success=0, engineering_checks=checks,
            config_sha256=args.config_sha256, detector_config=response.detector_config,
            accounting=accounting, frozen_before=response.frozen_digests, frozen_after=after,
            versions=dict(python=sys.version, executable=sys.executable, torch=torch.__version__, numpy=np.__version__),
            runtime_project_imports=imported, driver=str(Path(__file__).resolve()),
            original016_provenance=dict(manifest_sha256=config['provenance']['original016_manifest']['sha256'],
                members=116, project_imports=historical['project_import_closure']),
            calls=summaries, input_sha256=input_digest, input_parent_count=2, rows_per_call=4,
            elapsed_seconds=elapsed, peak_RSS_bytes=peak_RSS, runtime_RSS_watchdog=False,
            peak_RSS_measurement_stage='after NPZ/schema and before COMPLETE serialization; supervisor records full child peak',
            scientific_status='response_match_not_established', scientific_acceptance=False,
            full_WP5_WP6_complete=False, weights_promoted=False, automatic_restart=False)
        output_bytes = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
        complete_bytes = len((json.dumps(complete, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf8'))
        if output_bytes+complete_bytes > limits['output_MiB']*1024**2:
            raise ValueError('028 terminal artifact cap exceeded')
        write_record(output/'COMPLETE.json', complete)
        print(json.dumps(dict(native_status='COMPLETE', calls=12, artifact_bytes=output_bytes+complete_bytes)), flush=True)
    except BaseException as error:
        write_record(output/'FAILED.json', dict(schema='n01/wp6-028-example-failed/v1', run_id=config['run_id'],
            error_type=type(error).__name__, error=error.reason if hasattr(error, 'reason') else str(error), accounting=accounting,
            elapsed_seconds=time.monotonic()-started, automatic_restart=False, scientific_acceptance=False))
        raise


if __name__ == '__main__':
    main()
