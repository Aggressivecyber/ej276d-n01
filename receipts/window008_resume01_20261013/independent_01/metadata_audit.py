"""Independent CPU-only inspection of immutable Window008 recovery metadata.

No scientific model is constructed; no forward, optimizer step or GPU call.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EXPECTED_SHA = '921ec4c58387a5ad53fdf55ade10169930536106141e9070f5e4a8df60ab6c58'
HYPOTHESES = ('fast', 'slow', 'common_level')
STOCHASTIC = ('physical_head.', 'basis_head.', 'envelope_head.', 'local_filter.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    def default(item):
        if isinstance(item, torch.Tensor):
            return item.cpu().tolist()
        if isinstance(item, (np.ndarray, np.generic)):
            return item.tolist()
        raise TypeError(type(item).__name__)
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, default=default)
    return hashlib.sha256(raw.encode('utf8')).hexdigest()


def tensor_record(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                finite=bool(torch.isfinite(value).all()),
                sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())


def inventory(value):
    if isinstance(value, torch.Tensor):
        return dict(type='tensor', **tensor_record(value))
    if isinstance(value, np.ndarray):
        return dict(type='ndarray', shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(key): inventory(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if all(type(item) in (str, int, float, bool, type(None)) for item in value):
            return list(value)
        return [inventory(item) for item in value]
    return value


def sampler(state, expected_n, expected_draws):
    order = state['order'].cpu().numpy()
    visits = state['visits'].cpu().numpy()
    epochs, position = divmod(expected_draws, expected_n)
    expected_visits = np.full(expected_n, epochs, dtype=np.int64)
    expected_visits[order[:position]] += 1
    rng = np.random.default_rng(0)
    rng.bit_generator.state = copy.deepcopy(state['rng'])
    checks = dict(keys=set(state) == {'n', 'rng', 'order', 'position', 'completed_epochs', 'visits'},
                  population=type(state['n']) is int and state['n'] == expected_n,
                  order_shape=order.shape == (expected_n,),
                  order_is_permutation=np.array_equal(np.sort(order), np.arange(expected_n)),
                  visits_dtype=state['visits'].dtype == torch.int64,
                  visits_shape=visits.shape == (expected_n,),
                  draws=int(visits.sum()) == expected_draws,
                  epoch_position=(state['completed_epochs'], state['position']) == (epochs, position),
                  visits_exact=np.array_equal(visits, expected_visits),
                  rng_PCG64=state['rng']['bit_generator'] == 'PCG64',
                  rng_roundtrip=rng.bit_generator.state == state['rng'])
    return dict(expected_draws=expected_draws, actual_draws=int(visits.sum()),
                expected_epochs=epochs, expected_position=position,
                actual_epochs=state['completed_epochs'], actual_position=state['position'],
                unique=int((visits > 0).sum()), checks=checks, passed=all(checks.values()))


def optimizer(saved, signature, expected_step):
    names = [row['name'] for row in signature]
    identifiers = saved['param_groups'][0]['params']
    checks = dict(keys=set(saved) == {'state', 'param_groups'},
                  one_group=len(saved['param_groups']) == 1,
                  exact_positional_ids=identifiers == list(range(len(names))),
                  every_parameter_has_slot=set(saved['state']) == set(identifiers))
    details = []
    for identifier, row in zip(identifiers, signature):
        slot = saved['state'].get(identifier)
        expected = expected_step(row['name'])
        valid = slot is not None and set(slot) == {'step', 'exp_avg', 'exp_avg_sq'}
        if valid:
            valid = (isinstance(slot['step'], torch.Tensor) and slot['step'].numel() == 1
                     and slot['step'].dtype == torch.float32 and float(slot['step']) == expected)
            valid = valid and all(isinstance(slot[key], torch.Tensor)
                                 and list(slot[key].shape) == row['shape']
                                 and str(slot[key].dtype) == row['dtype']
                                 and bool(torch.isfinite(slot[key]).all()) for key in ('exp_avg', 'exp_avg_sq'))
        details.append(dict(name=row['name'], expected_step=expected,
                            actual_step=float(slot['step']) if slot else None, passed=bool(valid)))
    checks['named_slots'] = all(row['passed'] for row in details)
    return dict(parameters=len(names), step_histogram=dict(Counter(str(row['actual_step']) for row in details)),
                param_group={key: value for key, value in saved['param_groups'][0].items() if key != 'params'},
                checks=checks, named_slots=details, passed=all(checks.values()))


def load(path):
    before = sha(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert sha(path) == before, 'Input changed during CPU load'
    return state, before


def main():
    torch.set_num_threads(2)
    latest = ROOT / 'runs/window008_fix01_20261013/runs/science/train/full/latest.pt'
    cp, digest = load(latest)
    assert digest == EXPECTED_SHA
    base = json.loads((ROOT / 'window008_config.json').read_text('utf8'))
    fix = json.loads((ROOT / 'window008_fix01_config.json').read_text('utf8'))
    pause = json.loads((ROOT / 'receipts/window008_fix01_20261013/USER_PAUSE_RECEIPT.json').read_text('utf8'))
    native_old = json.loads((ROOT / 'provenance/window008_fix01_reference/legacy_native_exit.json').read_text('utf8'))
    old_failure = json.loads((ROOT / 'provenance/window008_fix01_reference/legacy_failure.json').read_text('utf8'))
    native_fix = json.loads((ROOT / 'runs/window008_fix01_20261013/runs/supervisor/exit_receipt.json').read_text('utf8'))
    fix_failure = json.loads((ROOT / 'runs/window008_fix01_20261013/runs/science/FAILURE.json').read_text('utf8'))
    checks = dict(schema=cp['schema'] == 'response008_fix01/v1', arm=cp['arm'] == 'full',
                  config_exact=cp['config'] == base, purpose_periodic=cp['checkpoint_purpose'] == 'periodic',
                  not_midstep=cp['failure_may_be_mid_step'] is False,
                  engineering_false=cp['engineering_fixture'] is False,
                  automatic_resume_disallowed=cp['automatic_resume_allowed'] is False,
                  G_6000=cp['G_updates'] == 6000, D_each_3000=cp['D_updates'] == dict.fromkeys(HYPOTHESES, 3000),
                  joint_stage=cp['stage'] == 'joint_stochastic',
                  last_metrics_G=cp['last_metrics']['G_updates'] == 6000,
                  last_metrics_D=cp['last_metrics']['D_updates'] == dict.fromkeys(HYPOTHESES, 3000),
                  last_metrics_stage=cp['last_metrics']['stage'] == 'joint_stochastic',
                  statistics_hash=json_hash(cp['statistics']) == cp['statistics_sha256'],
                  new_G=cp['new_G_updates'] == 3000, new_D=cp['new_D_updates'] == dict.fromkeys(HYPOTHESES, 3000),
                  old_full_import=cp['lineage']['inherited_G_updates'] == 3000 and cp['lineage']['inherited_D_updates'] == dict.fromkeys(HYPOTHESES, 0),
                  noise_zero=cp['noise_updates'] == 0,
                  population_zero=cp['population_learning'] is False,
                  model_implementation=cp['model_implementation'] == 'n01.model008_fix01.N01Response008Fix01',
                  pause_hash_matches=pause['periodic_checkpoint']['sha256'] == digest)
    g = optimizer(cp['g_optimizer'], cp['parameter_signature'],
                  lambda name: 3000 if name.startswith(STOCHASTIC) else 6000)
    checks['signature_matches_model'] = all(row['name'] in cp['model']
        and row['shape'] == list(cp['model'][row['name']].shape)
        and row['dtype'] == str(cp['model'][row['name']].dtype)
        for row in cp['parameter_signature'])
    checks['G_hyperparameters'] = (g['param_group']['lr'] == base['loss']['G_lr']
        and g['param_group']['weight_decay'] == base['loss']['G_weight_decay']
        and g['param_group']['betas'] == (0.9, 0.999) and g['param_group']['eps'] == 1e-8
        and g['param_group']['amsgrad'] is False and g['param_group']['maximize'] is False)
    d = {}
    for hypothesis in HYPOTHESES:
        ds = []
        for name, tensor in cp['discriminators'].items():
            if name.startswith(hypothesis + '.') and name.split('.', 1)[1] not in ('wave_center', 'wave_scale', 'qpsd_center', 'qpsd_scale', 'noise_center', 'noise_scale'):
                ds.append(dict(name=name.split('.', 1)[1], shape=list(tensor.shape), dtype=str(tensor.dtype)))
        # PyTorch spectral_norm moves each weight to weight_orig; its bias is
        # registered before weight_orig. weight_u/weight_v remain buffers.
        ds = [row for row in ds if row['name'].endswith(('.weight_orig', '.bias'))]
        d[hypothesis] = optimizer(cp['d_optimizers'][hypothesis], ds, lambda name: 3000)
    checks['D_hyperparameters'] = all(row['param_group']['lr'] == base['loss']['D_lr']
        and row['param_group']['weight_decay'] == 0.0 and row['param_group']['betas'] == (0.9, 0.999)
        and row['param_group']['eps'] == 1e-8 and row['param_group']['amsgrad'] is False
        and row['param_group']['maximize'] is False for row in d.values())
    source = sampler(cp['source_sampler'], 6593,
                     6000 * base['distribution_microbatches'] * base['source_parents_per_microbatch'])
    real = sampler(cp['real_sampler'], 21047,
                   6000 * base['distribution_microbatches'] * base['real_rows_per_microbatch'])
    streams = {}
    for name, saved in cp['streams'].items():
        generator = torch.Generator(device='cpu')
        generator.set_state(saved)
        streams[name] = dict(**tensor_record(saved), roundtrip=torch.equal(generator.get_state(), saved))
    checks['all_model_tensors_finite'] = all(bool(torch.isfinite(value).all()) for value in cp['model'].values())
    checks['all_D_tensors_finite'] = all(bool(torch.isfinite(value).all()) for value in cp['discriminators'].values())
    checks['streams_exact_keys'] = set(streams) == {'response', 'noise005', 'extension'}
    checks['streams_dtype_shape_roundtrip'] = all(row['dtype'] == 'torch.uint8' and row['shape'] == [5056]
                                                and row['roundtrip'] for row in streams.values())
    failure_path = ROOT / 'runs/window008_fix01_20261013/runs/science/FAILURE_CHECKPOINT.pt'
    failed, failed_digest = load(failure_path)
    failure_sources = sampler(failed['source_sampler'], 6593,
                             failed['G_updates'] * base['distribution_microbatches'] * base['source_parents_per_microbatch'])
    failure_reals = sampler(failed['real_sampler'], 21047,
                           failed['G_updates'] * base['distribution_microbatches'] * base['real_rows_per_microbatch'])
    history = dict(original_G=old_failure['G_updates'], original_D=old_failure['D_updates'],
                   fix01_G=fix_failure['new_execution_G_updates'], fix01_D=fix_failure['new_execution_D_updates'],
                   resume01_new_G=32000, resume01_new_D=87000)
    history.update(after_success_G=history['original_G'] + history['fix01_G'] + 32000,
                   after_success_D=history['original_D'] + history['fix01_D'] + 87000,
                   effective_G=19000 + 6000 + 13000 + 19000,
                   effective_D=48000 + 9000 + 39000 + 48000,
                   discarded_original_G=269, discarded_original_D=807,
                   discarded_fix01_G=398, discarded_fix01_D=1194)
    checks['history_after_success'] = history['after_success_G'] == 57667 and history['after_success_D'] == 146001
    checks['effective_endpoint'] = history['effective_G'] == 57000 and history['effective_D'] == 144000
    review = dict(schema='n01/window008-resume01-independent-metadata/v1',
                  UTC=datetime.now(timezone.utc).isoformat(),
                  status='passed' if all(checks.values()) and g['passed'] and all(row['passed'] for row in d.values())
                       and source['passed'] and real['passed'] else 'failed',
                  checkpoint=dict(path=str(latest.relative_to(ROOT)).replace('\\', '/'), sha256=digest,
                                  keys=sorted(cp), bytes=latest.stat().st_size),
                  checks=checks, G_optimizer=g, D_optimizers=d, source_sampler=source, real_sampler=real,
                  streams=streams, projection_context=cp['last_metrics']['projection_context'],
                  recovery=cp['recovery'], lineage=cp['lineage'], history=history,
                  original_native_exit=native_old, fix01_native_exit=native_fix,
                  stop_checkpoint=dict(sha256=failed_digest, G_updates=failed['G_updates'], D_updates=failed['D_updates'],
                                       purpose=failed['checkpoint_purpose'], failure_may_be_mid_step=failed['failure_may_be_mid_step'],
                                       source_sampler=failure_sources, real_sampler=failure_reals,
                                       admissible=False),
                  no_scientific_model_construction=True, scientific_model_forwards=0, optimizer_updates=0,
                  DEV_numeric_decodes=0, GPU_calls=0, new_scientific_jobs=0)
    (OUT / 'CHECKPOINT_METADATA_TREE.json').write_text(json.dumps(inventory(cp), indent=2, ensure_ascii=False) + '\n', 'utf8')
    (OUT / 'CHECKPOINT_METADATA_REVIEW.json').write_text(json.dumps(review, indent=2, ensure_ascii=False) + '\n', 'utf8')
    print(json.dumps(dict(status=review['status'], failed_checks=[name for name, valid in checks.items() if not valid],
                          G_optimizer_steps=g['step_histogram'], D_optimizer_steps={name: row['step_histogram'] for name, row in d.items()},
                          source=source, real=real, history=history, stop_checkpoint=review['stop_checkpoint']), indent=2))


if __name__ == '__main__':
    main()
