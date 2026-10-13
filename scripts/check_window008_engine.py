"""Synthetic-only Window008 engine check with the fixed real noise005 weights.

Reads configuration, source code and the bound noise checkpoint only. It never
opens TRAIN/DEV containers, prepared scientific data or signal checkpoints.
Exactly one mean and one joint optimizer update per arm are engineering steps.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

# Must be set before importing torch / creating any CUDA context.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from n01.losses008 import fit_loss_statistics
from n01.model008 import canonical_charge008
from n01.observation008 import crop_existing_values, observe_generated_carrier
from n01.response008 import (ARMS, HYPOTHESES, PairedStreams, Response008Engine,
    candidates, common_initial_state, fixed_noises, replay_backward,
    saturation_penalty, state_digest)
from n01.train import utc


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def assert_tree_equal(first, second, label):
    if isinstance(first, torch.Tensor):
        if not isinstance(second, torch.Tensor) or first.dtype != second.dtype or not torch.equal(first.detach().cpu(), second.detach().cpu()):
            raise AssertionError('Tensor restore mismatch: ' + label)
    elif isinstance(first, np.ndarray):
        if not np.array_equal(first, second):
            raise AssertionError('Array restore mismatch: ' + label)
    elif isinstance(first, dict):
        if not isinstance(second, dict) or set(first) != set(second):
            raise AssertionError('Dictionary restore mismatch: ' + label)
        for key in first:
            assert_tree_equal(first[key], second[key], label + '/' + str(key))
    elif isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise AssertionError('Sequence restore mismatch: ' + label)
        for index, (a, b) in enumerate(zip(first, second)):
            assert_tree_equal(a, b, label + '/' + str(index))
    elif first != second:
        raise AssertionError('Value restore mismatch: ' + label)


def fixture(rows=128):
    """Analytic pulses and artificial baseline; no experimental data access."""
    generator = torch.Generator(device='cpu').manual_seed(80082026)
    t = torch.arange(384, dtype=torch.float64)[None]
    index = torch.arange(rows, dtype=torch.float64)[:, None]
    amplitude = .045 + .115 * index / (rows - 1)
    tau = 31. + 18. * (index.remainder(11) / 10.)
    rise = torch.exp(-((t - 126.) / (4.5 + .6 * index.remainder(3))) ** 2)
    source_shape = torch.where(t <= 126., rise, torch.exp(-(t - 126.) / tau))
    source_shape[:, :101] = 0.
    source = amplitude * source_shape + .003
    real_shape = torch.where(t <= 126., rise, .78 * torch.exp(-(t - 126.) / (tau * 1.04))
                             + .22 * torch.exp(-(t - 126.) / (tau * 1.8)))
    real_shape[:, :101] = 0.
    artificial_baseline = .0006 * torch.randn((rows, 384), generator=generator, dtype=torch.float64)
    artificial_baseline += .0003 * torch.sin(index * .7) + .0002 * (t - 191.5) / 383.
    observed = observe_generated_carrier(.96 * amplitude * real_shape + artificial_baseline)
    source_effective = crop_existing_values(source)['window_V']
    q = canonical_charge008(source - source[:, :100].mean(1, keepdim=True))
    statistics = dict(
        engineering_fixture=True,
        scientific_data_read=False,
        fit_split_note='TRAIN is an API role on artificial fixture arrays only; no actual TRAIN data was read.',
        model=dict(q_ref=float(q.median()), a_ref=float(source_effective.amax(1).median()),
                   log_q_min=math.log(float(q.min()) / 1.5), log_q_max=math.log(float(q.max()) * 1.5),
                   detector_condition=0., alignment_t0=126.),
        loss=serial(fit_loss_statistics(observed['window_V'], source_effective, observed['baseline100'])))
    data = dict(source_carrier384=source.numpy(), real=observed['window_V'].numpy(),
                real_baseline100=observed['baseline100'].numpy(),
                source_ids=np.arange(-20000, -20000 + rows), real_ids=np.arange(-10000, -10000 + rows))
    digest = hashlib.sha256()
    for key in sorted(data):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(data[key]).tobytes())
    return data, statistics, digest.hexdigest()


def probe_inputs(engine, count=4):
    streams = PairedStreams(81008001, 81008002, 81008003)
    response, noise005, extension = streams.draw(count, engine.device)
    return engine.source[:count], response, fixed_noises(engine.extension, noise005, extension)


@torch.no_grad()
def observation_probe(engine):
    source, response, noises = probe_inputs(engine)
    first = engine.model.signal_forward(source, response)
    repeat = engine.model.signal_forward(source, response)
    if not torch.equal(first['signal_pre'], repeat['signal_pre']):
        raise AssertionError('Explicit response replay changed')
    if not bool((first['signal_pre'][:, :100] == 0).all()):
        raise AssertionError('Signal leaked into baseline100')
    baseline = None
    result = {}
    for hypothesis in HYPOTHESES:
        signal, noise = first['signal_pre'], noises[hypothesis]
        pre = signal + noise
        observed = observe_generated_carrier(pre, return_indices=True, return_carrier=True)
        repeated = observe_generated_carrier(repeat['signal_pre'] + noise)
        if not torch.equal(observed['window_V'], repeated['window_V']):
            raise AssertionError('Observed explicit-latent replay changed')
        if observed['window_V'].shape != (len(source), 256):
            raise AssertionError('Observed output is not 256 samples')
        if baseline is None:
            baseline = observed['baseline100']
        elif not torch.equal(baseline, observed['baseline100']):
            raise AssertionError('Noise hypotheses changed their shared observed baseline')
        indices = observed['crop_indices']
        qs = signal.gather(1, indices).sum(1)
        qn = noise.gather(1, indices).sum(1)
        qa = (observed['carrier_adc'] - pre).gather(1, indices).sum(1)
        qb = -256. * observed['baseline_mean_V']
        error = float((observed['Q'] - qs - qn - qa - qb).abs().max())
        if error > 1e-11:
            raise AssertionError('Observed charge decomposition failed')
        result[hypothesis] = dict(output_shape=list(observed['window_V'].shape),
            explicit_replay_bitwise_equal=True, observed_peak_indices=observed['peak_index'].tolist(),
            Q_observed_decomposition_max_abs_error=error,
            signal_Q_crop_migration=(qs - first['qsig']).tolist(),
            raw_noise_Q_at_observed_crop=qn.tolist(),
            ADC_Q_correction=qa.tolist(), baseline_Q_correction=qb.tolist())
    return dict(hypotheses=result, signal_prefix100_exact_zero=True,
        common_hypothesis_baseline100_bitwise_equal=True,
        canonical_signal_Q_max_abs_error=float(first['canonical_charge_identity_residual'].abs().max()),
        canonical_local_Q_max_abs_error=float(first['canonical_local_charge'].abs().max()))


def vjp_probe(engine):
    """Compare engine two-pass replay with a direct graph; no optimizer step."""
    source, response, noises = probe_inputs(engine)
    direct = copy.deepcopy(engine.model)
    replay = copy.deepcopy(engine.model)
    direct.zero_grad(set_to_none=True)
    replay.zero_grad(set_to_none=True)
    generator = torch.Generator(device='cpu').manual_seed(81008004)
    gradients = {h: (.001 * torch.randn((len(source), 256), generator=generator, dtype=torch.float64)).to(engine.device)
                 for h in HYPOTHESES}
    batches, expected, terms = [], {h: [] for h in HYPOTHESES}, []
    groups = 2
    for start in (0, 2):
        section = slice(start, start + 2)
        sx = source[section]
        rz = {key: value[section] for key, value in response.items()}
        nz = {key: value[section] for key, value in noises.items()}
        batches.append((sx, rz, nz))
        output = direct.signal_forward(sx, rz)
        waves = candidates(output['signal_pre'], nz, ste=True)
        for h in HYPOTHESES:
            expected[h].append(waves[h].detach())
        regular = output['base_regular'] + sum(saturation_penalty(output['signal_pre'] + nz[h]) for h in HYPOTHESES) / 3
        terms.append(sum((waves[h] * gradients[h][section]).sum() for h in HYPOTHESES)
                     + (.01 * regular + .1 * output['trend']) / groups)
    sum(terms).backward()
    replay_backward(replay, batches, gradients, expected={h: torch.cat(values) for h, values in expected.items()})
    squared_error, squared_reference, max_error, compared = 0., 0., 0., 0
    for (name, left), (other_name, right) in zip(direct.named_parameters(), replay.named_parameters()):
        if name != other_name or ((left.grad is None) != (right.grad is None)):
            raise AssertionError('Replay gradient branch mismatch: ' + name)
        if left.grad is None:
            continue
        a, b = left.grad.detach().double(), right.grad.detach().double()
        if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
            raise AssertionError('Nonfinite replay/direct gradient')
        squared_error += float((a - b).square().sum())
        squared_reference += float(a.square().sum())
        max_error = max(max_error, float((a - b).abs().max()))
        compared += a.numel()
    relative_error = math.sqrt(squared_error / max(squared_reference, 1e-30))
    if compared == 0 or relative_error > 2e-5:
        raise AssertionError('Two-pass replay gradient differs from direct graph: ' + str(relative_error))
    return dict(parameter_gradient_elements=compared, relative_L2_error=relative_error,
                max_abs_error=max_error, optimizer_steps=0)


def verify_restore(first, restored):
    for name, left, right in (
        ('model', first.model.state_dict(), restored.model.state_dict()),
        ('discriminators', first.discriminators.state_dict(), restored.discriminators.state_dict()),
        ('G_optimizer', first.g_optimizer.state_dict(), restored.g_optimizer.state_dict()),
        ('source_sampler', first.source_sampler.state_dict(), restored.source_sampler.state_dict()),
        ('real_sampler', first.real_sampler.state_dict(), restored.real_sampler.state_dict()),
        ('explicit_streams', first.streams.state_dict(), restored.streams.state_dict())):
        assert_tree_equal(left, right, name)
    for hypothesis in HYPOTHESES:
        assert_tree_equal(first.d_optimizers[hypothesis].state_dict(), restored.d_optimizers[hypothesis].state_dict(), 'D_optimizer/' + hypothesis)
    if first.update != restored.update or first.d_updates != restored.d_updates or first.stage != restored.stage:
        raise AssertionError('Checkpoint counters or stage changed')


def run(args, output, report):
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; no CPU substitution performed')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    config['distribution_microbatches'] = 2
    config['source_parents_per_microbatch'] = 4
    config['responses_per_parent'] = 2
    config['real_rows_per_microbatch'] = 8
    config['engineering_fixture'] = True
    config['response'].update(mean_response_updates=1, joint_stochastic_updates=1,
        low_learning_rate_updates=0, updates_per_arm=2, total_G_updates=6, total_D_updates=9)
    noise_path = Path(config['noise_reference']['path'])
    if not noise_path.is_absolute():
        noise_path = config_path.parent / noise_path
    noise_hash = file_digest(noise_path)
    if noise_hash != config['noise_reference']['sha256']:
        raise ValueError('Frozen005 checkpoint hash mismatch')
    checkpoint = torch.load(noise_path, map_location='cpu', weights_only=False)
    data, statistics, synthetic_digest = fixture()
    common = common_initial_state(statistics['model'], config['seeds']['initialization'])
    report.update(device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        torch_version=torch.__version__, cuda_version=torch.version.cuda, deterministic_algorithms=True,
        config_path=str(config_path), config_sha256=file_digest(config_path),
        fixed_noise005_path=str(noise_path), fixed_noise005_sha256=noise_hash,
        fixture_sha256=synthetic_digest, artificial_source_rows=len(data['source_carrier384']),
        artificial_observed_rows=len(data['real']), engineering_batch=dict(microbatches=2,parents=4,responses_per_parent=2,real_rows_per_microbatch=8),
        scientific_updates=0, noise_updates=0, population_updates=0, optimizer_update_budget=dict(G=6,D_total=9),
        common_initial_state_sha256=state_digest(common))
    write_json(output / 'ENGINEERING_CONFIG.json', config)
    write_json(output / 'ARTIFICIAL_STATISTICS.json', statistics)
    mean_states = {}
    for arm in ARMS:
        row = report['arms'][arm] = dict(status='running', engineering_G_updates=0,engineering_D_updates={h:0 for h in HYPOTHESES})
        engine = Response008Engine(data, statistics, config, checkpoint, common, arm=arm, device=device)
        report['active_arm'] = arm
        try:
            initial_noise = state_digest(engine.extension.model005.state_dict())
            engine.set_stage('mean_response')
            mean = engine.step(audit=True, verify_replay=True)
            row['mean_step'] = mean
            row['engineering_G_updates'] = engine.update
            row['engineering_D_updates'] = dict(engine.d_updates)
            mean_states[arm] = {name: value.detach().cpu().clone() for name, value in engine.model.state_dict().items()}
            if any(engine.d_updates.values()) or mean['G_preclip_gradient_norm'] <= 0:
                raise AssertionError('Mean phase unexpectedly changed D or had no G gradient')
            saved = output / (arm + '_after_mean.engineering.pt')
            engine.checkpoint(saved)
            restored = Response008Engine(data, statistics, config, checkpoint, common, arm=arm, device=device)
            restored.restore(saved)
            verify_restore(engine, restored)
            row['checkpoint_restore'] = dict(path=str(saved), sha256=file_digest(saved),
                model_D_optimizers_samplers_streams_counters_exact=True)
            if arm == ARMS[0]:
                changed = copy.deepcopy(statistics)
                changed['loss']['q_reference'] *= 1.01
                rejected = Response008Engine(data, changed, config, checkpoint, common, arm=arm, device=device)
                try:
                    rejected.restore(saved)
                except (ValueError, RuntimeError) as error:
                    if 'statistics' not in str(error).lower():
                        raise AssertionError('Changed-statistics resume failed for an unrelated reason') from error
                    row['checkpoint_modified_loss_statistics_rejected'] = str(error)
                else:
                    raise AssertionError('Checkpoint accepted changed loss statistics')
                del rejected
            engine = restored
            engine.set_stage('joint_stochastic')
            joint = engine.step(audit=True, verify_replay=True)
            row['joint_step'] = joint
            if engine.update != 2 or any(count != 1 for count in engine.d_updates.values()):
                raise AssertionError('Engineering G/D counters differ from mean1+joint1')
            required = ['charge_head', 'mean_time', 'response_filter']
            if arm != 'simple':
                required.append('mean_head')
            if arm != 'deterministic':
                required.append('physical_head')
            if arm == 'full':
                required += ['basis_head', 'envelope_head', 'local_filter']
            norms = joint['G_branch_postclip_gradient_norms']
            for branch in required:
                if not math.isfinite(norms.get(branch, float('nan'))) or norms[branch] <= 0:
                    raise AssertionError('Missing finite nonzero branch gradient: ' + branch)
            for step in (mean, joint):
                if any(step['endpoint_gradients'][h]['norms']['noise'] != 0 for h in HYPOTHESES):
                    raise AssertionError('Frozen baseline noise loss acquired response gradient')
            row['observation_probe'] = observation_probe(engine)
            row['two_pass_VJP_vs_direct_graph'] = vjp_probe(engine)
            engine.extension.assert_frozen()
            if state_digest(engine.extension.model005.state_dict()) != initial_noise:
                raise AssertionError('Frozen005 changed during engineering steps')
            if any(parameter.grad is not None for parameter in engine.extension.parameters()):
                raise AssertionError('Frozen005 acquired parameter gradients')
            row.update(status='passed', fixed_noise005_state_sha256=initial_noise,
                final_model_state_sha256=state_digest(engine.model.state_dict()),
                initial_model_state_sha256=engine.initial_state_digest,initial_D_state_sha256=engine.initial_D_digest)
        finally:
            row['engineering_G_updates'] = engine.update
            row['engineering_D_updates'] = dict(engine.d_updates)
            report['engineering_G_updates_total'] = sum(item['engineering_G_updates'] for item in report['arms'].values())
            report['engineering_D_updates_total'] = sum(sum(item['engineering_D_updates'].values()) for item in report['arms'].values())
            write_json(output / 'ENGINEERING_RECEIPT.json', report)
        print(json.dumps(dict(arm=arm,status=row['status'],engineering_G_updates=engine.update,engineering_D_updates=engine.d_updates)),flush=True)
        del engine, restored
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    full, deterministic = mean_states['full'], mean_states['deterministic']
    for name, value in deterministic.items():
        if name not in full or not torch.equal(value, full[name]):
            raise AssertionError('Paired deterministic/full mean update diverged: ' + name)
    report['paired_mean_update_full_deterministic_shared_state_bitwise_equal'] = True
    report['status'] = 'passed_synthetic_engineering_only'
    report['active_arm'] = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0; CUDA is never silently replaced')
    parser.add_argument('--out', required=True, help='New output directory on E:/D: or the remote workspace')
    parser.add_argument('--config', default=str(ROOT / 'window008_config.json'))
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if os.name == 'nt' and output.drive.upper() not in ('E:', 'D:'):
        parser.error('Windows engineering outputs must be on E: or D:')
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    output.mkdir(parents=True, exist_ok=False)
    source_files = [ROOT/'src/n01'/name for name in ('model.py','model008.py','response008.py','observation008.py','losses008.py','losses008_v2.py','electronics_v3.py','extension006.py','data.py','train.py')]
    source_files.append(Path(__file__).resolve())
    report = dict(schema='window008/synthetic-engineering/v1',status='running',started_UTC=utc(),
        scope='Analytic pulse / artificial baseline engineering steps with fixed real005 weights; no scientific training or acceptance claim.',
        scientific_data_read=False, TRAIN_rows_read=0, DEV_rows_read=0, scientific_updates=0,
        source_sha256={str(path.relative_to(ROOT)):file_digest(path) for path in source_files}, arms={})
    started = time.monotonic()
    exit_code = 0
    try:
        run(args, output, report)
        after = {str(path.relative_to(ROOT)):file_digest(path) for path in source_files}
        if after != report['source_sha256']:
            raise RuntimeError('Source files changed while this engineering check ran; rerun on a fixed source snapshot')
    except Exception as error:
        report.update(status='failed_engineering_only',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        exit_code = 1
    finally:
        report.update(completed_UTC=utc(),elapsed_seconds=time.monotonic()-started)
        write_json(output/'ENGINEERING_RECEIPT.json', report)
    print(json.dumps(dict(status=report['status'],receipt=str(output/'ENGINEERING_RECEIPT.json'),
        engineering_G_updates=report.get('engineering_G_updates_total',0),engineering_D_updates=report.get('engineering_D_updates_total',0),
        scientific_updates=0,error=report.get('error'))),flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
