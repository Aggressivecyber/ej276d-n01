"""Zero-optimizer TRAIN marginal-risk audit of fixed B-only amplitudes.

Three response calls cache teacher/030 endpoint factors. Everything thereafter
is cached signal/ADC/loss algebra. Four independent full-parent banks permit an
energy comparator with genuine Euclidean distances and both self terms.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import run_response029 as old
from n01.losses008 import _wave_views
from n01.losses014 import integral_features014

torch = old.torch
REFERENCE_029_SHA256 = 'a39f5ddd3fcbd0d8cf58db487d7fb51d1b17646f69603047a1e739fece74e7d5'


def settings031():
    p = old.settings029()
    for unused in ('training', 'evaluation', 'learner_seeds', 'no_REAL_or_checkpoint_reads'):
        p.pop(unused)
    p.update(schema='n01/response031-cached-risk-audit/v1', run_id='response031_20261017',
        stage='cached_B_amplitude_risk_audit', endpoint_seeds=[2026101701, 2026101702],
        execution_kind='zero_optimizer_known_teacher_audit',
        no_REAL_reads=True, checkpoint_loads=2, no_optimizer=True,
        design_revision='One fixed B-only direction; A, charge_v and local law remain fixed.',
        reference_runner_sha256=REFERENCE_029_SHA256,
        audit=dict(blocks=32, parents_per_bank=128, bank_names=['Xa', 'Xb', 'Yt', 'Yu'],
            bank_seed_base=202610173100, bank_seed_block_stride=10, lambdas=[.9, 1., 1.1],
            native_projection_seed=2026102301, native_evaluation_seed=2026102401,
            integral_projection_seed=2026102501, carrier_projection_seed=2026102201,
            context_updates=list(range(1, 33)), descriptive_MCSE_multiplier=3.,
            regular_weight_reference=.01,
            NN_calls=3, NN_rows=384, scalar_AD_queries=64, optimizer_steps=0),
        checkpoint_inputs={
            'endpoint_seed2026101701': dict(path='/home/rylan/projects/ej276d-n01/response030_20261017/run_01/unpaired_seed2026101701/final.pt',
                bytes=6135427, sha256='aecd10e5477ac094525977817a336bcf1eedf5fbd0b729d85f5a65a58ad7c970'),
            'endpoint_seed2026101702': dict(path='/home/rylan/projects/ej276d-n01/response030_20261017/run_01/unpaired_seed2026101702/final.pt',
                bytes=6135427, sha256='5d9ccb7480e8d4e7f5d6bea8a012eb3ae030a3325cef83309ea1d9643403e9ab')})
    p['resources'].update(wall_seconds=900, output_MiB=64)
    return p


@torch.no_grad()
def cache_factors(model, source):
    noise = {key: source.new_zeros((len(source),)+shape) for key,shape in model.noise_shapes.items()}
    out = model(source, noise, enabled=False)
    return dict(A=out['base_signal_pre_V'], B=out['B'], v=out['charge_v'], envelope=out['signal_envelope'],
        taps=model.local_filter.taps().detach().double(), gate=model.mean_model.gate[0, 0],
        projection=model.mean_model.projection[0, 0], regular=out['regular'], regular_basis=out['regular_basis'])


def mean_euclidean(x, y):
    return torch.linalg.vector_norm(x[:, None, :]-y[None, :, :], dim=2).mean()/math.sqrt(x.shape[1])


def energy_four_banks(Xa, Xb, Yt, Yu):
    return .5*(mean_euclidean(Xa, Yt)+mean_euclidean(Xa, Yu)
        +mean_euclidean(Xb, Yt)+mean_euclidean(Xb, Yu))-mean_euclidean(Xa, Xb)-mean_euclidean(Yt, Yu)


def standardized_banks(observations, loss, carrier_scale):
    result = []
    for obs in observations:
        views = {name:loss._z(name, value) for name,value in _wave_views(obs['window_V'].double(), loss.peak_floor).items()}
        integrals = torch.asinh(integral_features014(obs['window_V'])/loss.integral_scale)
        carrier = (obs['carrier_adc']-obs['baseline_mean_V'][:, None])/carrier_scale
        result.append(dict(views=views, integrals=integrals, carrier=carrier))
    return result


def energy_data(observations, loss, carrier_scale):
    banks = standardized_banks(observations, loss, carrier_scale)
    wave = torch.stack([energy_four_banks(*(bank['views'][name] for bank in banks))
                        for name in banks[0]['views']]).mean()
    integral = energy_four_banks(*(bank['integrals'] for bank in banks))
    carrier = energy_four_banks(*(bank['carrier'] for bank in banks))
    return wave+2.*integral+.25*carrier


def native_data(observations, loss, context, directions, carrier_scale):
    values = []
    for x in observations[:2]:
        for y in observations[2:]:
            parts = loss.wave_components(x['window_V'], y['window_V'], None, context, mean_response=False)
            wave = sum(old.WAVE_WEIGHTS[name]*value for name,value in parts.items())
            integral = loss.integral_loss(x['window_V'], y['window_V'], None, context)
            f = (x['carrier_adc']-x['baseline_mean_V'][:, None])/carrier_scale
            r = (y['carrier_adc']-y['baseline_mean_V'][:, None])/carrier_scale
            carrier = .5*(old.empirical_wasserstein_1d(f, r)+old.empirical_wasserstein_1d(f@directions, r@directions))
            values.append(wave+2.*integral+.25*carrier)
    return torch.stack(values).mean()


def estimate(values):
    data = old.np.asarray(values, dtype=old.np.float64)
    mean = float(data.mean())
    mcse = float(data.std(ddof=1)/math.sqrt(len(data)))
    return dict(mean=mean, MCSE=mcse, blocks=len(data), descriptive_3MCSE_separation=abs(mean) > 3.*mcse)


def run(p, output, device, progress):
    started = time.monotonic()
    output.mkdir(parents=True)
    old.write_json(output/'START.json', dict(config=p, device=device, pid=os.getpid(), UTC=datetime.now(timezone.utc).isoformat(),
        no_optimizer=True, no_REAL=True, heldout_forwards=0, automatic_restart=False))
    train, unused_heldout, stats = old.make_sources(p)
    del unused_heldout
    source = train['source'].to(device)
    teacher = old.make_teacher(stats, p, device)
    states = {'teacher':cache_factors(teacher, source)}
    progress.update(NN_calls=1, NN_rows=128)
    for seed in p['endpoint_seeds']:
        key = f'endpoint_seed{seed}'
        record = p['checkpoint_inputs'][key]
        path = Path(record['path'])
        if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('031 pinned endpoint bytes differ: '+key)
        model = old.N01Response029(stats, seed=seed)
        state = torch.load(path, map_location='cpu', weights_only=True)
        progress['CP_loads'] += 1
        model.load_state_dict(state, strict=True)
        del state
        model.to(device).eval().requires_grad_(False)
        states[key] = cache_factors(model, source)
        progress['NN_calls'] += 1; progress['NN_rows'] += 128
        del model
    teacher_factors = states['teacher']
    # Identical 029 TRAIN-only scale fit; the 1024 outputs use cached algebra.
    rows = torch.arange(128, device=device)
    fit_rows = rows.repeat_interleave(8)
    fit_rng = torch.Generator(device=device).manual_seed(202610172901)
    reference = old.cached_sample(teacher_factors, fit_rows, teacher.sample_noise(1024, device=device, generator=fit_rng))
    observed = old.observe_generated_carrier(reference, q_floor=1e-8, return_carrier=True)
    source_window = old.crop_existing_values(train['source'].repeat_interleave(8, dim=0).to(device), q_floor=1e-8)['window_V']
    loss_stats = old.fit_loss_statistics(observed['window_V'].cpu(), source_window.cpu(), observed['baseline100'].cpu())
    integral_stats = old.fit_integral_statistics014(observed['window_V'].cpu())
    carrier_scale = (observed['carrier_adc']-observed['baseline_mean_V'][:, None]).square().mean(0).sqrt().clamp_min(old.ADC_STEP)
    old.write_json(output/'TRAIN_SCALES.json', dict(mean_MSE_scale=float(teacher_factors['A'][fit_rows].square().mean()),
        residual_MSE_scale=float((reference-teacher_factors['A'][fit_rows]).square().mean()), carrier_scale=carrier_scale.cpu().tolist(),
        carrier_scale_domain='same single hard ADC carrier384 minus its original prefix100 mean',
        fitted_parents=128, fitted_draws=8, heldout_used=False, real_or_checkpoint_reads=0))
    a = p['audit']
    loss = old.N01Loss014(loss_stats, integral_stats, seed=a['native_projection_seed'],
        evaluation_seed=a['native_evaluation_seed'], integral_seed=a['integral_projection_seed']).to(device)
    q_floor = float(loss_stats['q_floor'])
    projection_rng = torch.Generator(device='cpu').manual_seed(a['carrier_projection_seed'])
    del reference, observed, source_window
    blocks = []
    with (output/'BLOCKS.jsonl').open('x', encoding='utf8', buffering=1) as log:
        for block in range(32):
            if time.monotonic()-started > p['resources']['wall_seconds']:
                raise TimeoutError('031 wall cap exceeded')
            noise = [teacher.sample_noise(128, device=device, generator=torch.Generator(device=device).manual_seed(
                a['bank_seed_base']+a['bank_seed_block_stride']*block+bank)) for bank in range(4)]
            targets = [old.observe_generated_carrier(old.cached_sample(teacher_factors, rows, z),
                q_floor=q_floor, return_carrier=True) for z in noise[2:]]
            context = loss.create_projection_context(block+1)
            directions = torch.randn(384, 16, generator=projection_rng, dtype=torch.float64).to(device)
            directions = directions/directions.norm(dim=0, keepdim=True)
            record = dict(block=block, bank_seeds=[a['bank_seed_base']+a['bank_seed_block_stride']*block+j for j in range(4)],
                context_update=block+1, candidates={})
            candidates = [('teacher_lambda0.9', teacher_factors, .9), ('teacher_lambda1.0', teacher_factors, 1.),
                          ('teacher_lambda1.1', teacher_factors, 1.1)]+[(key,value,1.) for key,value in states.items() if key != 'teacher']
            for name, factor, amplitude in candidates:
                derivative = name == 'teacher_lambda1.0'
                lam = torch.tensor(amplitude, dtype=torch.float64, device=device, requires_grad=derivative)
                scaled = dict(factor, B=factor['B']*lam)
                X = [old.observe_generated_carrier(old.cached_sample(scaled, rows, z), ste=derivative,
                    q_floor=q_floor, return_carrier=True) for z in noise[:2]]
                observations = X+targets
                native = native_data(observations, loss, context, directions, carrier_scale)
                energy = energy_data(observations, loss, carrier_scale)
                regular = factor['regular']+(amplitude**2-1.)*factor['regular_basis']
                item = dict(native_data=float(native.detach()), energy_data=float(energy.detach()),
                    regular=float(regular), regular_basis=float(factor['regular_basis']))
                item.update(native_total=item['native_data']+.01*item['regular'],
                    energy_total=item['energy_data']+.01*item['regular'])
                if derivative:
                    item['native_STE_derivative'] = float(torch.autograd.grad(native, lam, retain_graph=True)[0])
                    progress['AD_queries'] += 1
                    item['energy_STE_derivative'] = float(torch.autograd.grad(energy, lam)[0])
                    progress['AD_queries'] += 1
                    item['regular_analytic_derivative'] = float(2.*factor['regular_basis'])
                record['candidates'][name] = item
            low, center, high = (record['candidates'][name] for name in ('teacher_lambda0.9','teacher_lambda1.0','teacher_lambda1.1'))
            record['B_direction'] = {metric:dict(hard_FD=(high[metric+'_data']-low[metric+'_data'])/.2,
                STE=center[metric+'_STE_derivative'], hard_FD_minus_STE=(high[metric+'_data']-low[metric+'_data'])/.2-center[metric+'_STE_derivative'])
                for metric in ('native','energy')}
            for metric in ('native','energy'):
                fd = (high[metric+'_total']-low[metric+'_total'])/.2
                ste = center[metric+'_STE_derivative']+.01*center['regular_analytic_derivative']
                record['B_direction'][metric+'_total'] = dict(hard_FD=fd, STE=ste, hard_FD_minus_STE=fd-ste)
            log.write(json.dumps(record, allow_nan=False)+'\n')
            blocks.append(record)
            progress['completed_blocks'] = block+1
            old.write_json(output/'STATUS.json', dict(progress, UTC=datetime.now(timezone.utc).isoformat()))
            print(json.dumps(dict(blocks=block+1, NN_calls=progress['NN_calls'], AD_queries=progress['AD_queries'])), flush=True)
            if sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) > 64*1024**2:
                raise OSError('031 output cap exceeded')
    summary = dict(candidates={}, paired_differences={}, B_direction={})
    for name in blocks[0]['candidates']:
        summary['candidates'][name] = {metric:estimate([b['candidates'][name][metric] for b in blocks])
            for metric in ('native_data','energy_data','regular','native_total','energy_total')}
        if name != 'teacher_lambda1.0':
            summary['paired_differences'][name+'_minus_teacher_lambda1.0'] = {
                metric:estimate([b['candidates'][name][metric]-b['candidates']['teacher_lambda1.0'][metric] for b in blocks])
                for metric in ('native_data','energy_data','regular','native_total','energy_total')}
    for metric in ('native','energy','native_total','energy_total'):
        summary['B_direction'][metric] = {kind:estimate([b['B_direction'][metric][kind] for b in blocks])
            for kind in ('hard_FD','STE','hard_FD_minus_STE')}
    summary['B_direction']['regular'] = dict(analytic_derivative=estimate([
        b['candidates']['teacher_lambda1.0']['regular_analytic_derivative'] for b in blocks]),
        hard_FD=estimate([(b['candidates']['teacher_lambda1.1']['regular']-b['candidates']['teacher_lambda0.9']['regular'])/.2 for b in blocks]))
    result = dict(schema='n01/response031-risk-audit-result/v1', native_status='COMPLETE', summary=summary,
        counts=progress, optimizer_steps=0, heldout_forwards=0, REAL_reads=0, elapsed_seconds=time.monotonic()-started,
        descriptive_MCSE_multiplier=3., scientific_acceptance=False, physical_detector_acceptance=False,
        native_definition='Original C1 data risk, mean of four 128-row X/Y crosspairs; regular reported separately.',
        energy_definition='Mean Euclidean distance/sqrt(d); symmetric four-bank energy on six full standardized wave views, six signed-asinh integrals and once-ADC centered384.',
        limitations='Fixed TRAIN source population and B-only amplitudes. Three-MCSE separation is descriptive; it does not grant response acceptance or select/fit a replacement network.')
    old.write_json(output/'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = Path(args.config).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('031 config SHA mismatch')
    if hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest() != REFERENCE_029_SHA256:
        raise ValueError('Closed029 source changed')
    p = json.loads(raw)
    if p['status'] != 'frozen' or any(p[key] != value for key,value in settings031().items() if key != 'status'):
        raise ValueError('031 requires the fixed frozen zero-optimizer audit')
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith('cuda'):
        target = torch.device(args.device)
        torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(target).total_memory, target)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve previous031 attempt: '+str(output))
    def terminate(signum, frame):
        raise InterruptedError('031 received signal '+str(signum))
    signal.signal(signal.SIGTERM, terminate)
    progress = dict(NN_calls=0, NN_rows=0, CP_loads=0, AD_queries=0, completed_blocks=0)
    try:
        result = run(p, output, args.device, progress)
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        old.write_json(output/'FAILED.json', dict(error_type=type(error).__name__, error=str(error), counts=progress, optimizer_steps=0))
        raise
    print(json.dumps(dict(native_status=result['native_status'], counts=progress)), flush=True)


if __name__ == '__main__':
    main()
