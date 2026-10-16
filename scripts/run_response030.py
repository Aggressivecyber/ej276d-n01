"""029 unpaired objective with its complete 128-parent marginal every update.

Only source-marginal sampling changes: one independent draw per parent instead
of two draws for independently selected 64-parent subsets. Teacher and learner
latents remain independent; losses see marginal sets, never conditional labels.
The closed 029 runner and model remain byte-identical and supply shared algebra.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import run_response029 as old

torch = old.torch
REFERENCE_029_SHA256 = 'a39f5ddd3fcbd0d8cf58db487d7fb51d1b17646f69603047a1e739fece74e7d5'


def settings030():
    p = old.settings029()
    p.update(schema='n01/response030-unpaired-allparents/v1', run_id='response030_20261017',
        stage='unpaired_all_parents', design_revision='029 closed; one controlled change to the source marginal sampler',
        controlled_change='all128parents_x1draw_per_update_for_both_independent_marginal_sets',
        reference_runner_sha256=REFERENCE_029_SHA256)
    p['training'].update(paired_updates=0, unpaired_updates=1000, parents_per_update=128, draws_per_parent=1)
    p['resources']['wall_seconds'] = 3600
    return p


def train030(model, teacher_train, train, fixed, p, seed, out, started, progress):
    out.mkdir()
    model.train()
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW([
        dict(params=groups['mean'], lr=p['training']['mean_lr']),
        dict(params=groups['shape']+groups['charge']+groups['local'], lr=p['training']['new_head_lr'])], weight_decay=0.)
    parameters = list(model.parameters())
    device = teacher_train['A'].device
    source = train['source'].to(device)
    parents = torch.arange(128, device=device)
    learner_rng = torch.Generator(device=device).manual_seed(seed+300)
    teacher_rng = torch.Generator(device=device).manual_seed(seed+400)
    projection_rng = torch.Generator(device='cpu').manual_seed(seed+500)
    progress.update(learner_seed=seed, current_run_completed_updates=0)
    old.write_json(out/'INITIAL.json', dict(seed=seed, mode='unpaired', updates=0,
        initial_state_sha256=old.state_digest(model.state_dict()), same_initialization_as029=True,
        source_parents=128, teacher_parents=128, draws_per_parent=1,
        learner_noise_seed=seed+300, independent_teacher_noise_seed=seed+400,
        conditional_teacher_labels_used_in_loss=False))
    with (out/'metrics.jsonl').open('x', encoding='utf8', buffering=1) as log:
        for step in range(1, p['training']['unpaired_updates']+1):
            if time.monotonic()-started > p['resources']['wall_seconds']:
                raise TimeoutError('Registered response030 wall limit exceeded')
            noise = model.sample_noise(128, device=device, generator=learner_rng)
            independent_noise = model.sample_noise(128, device=device, generator=teacher_rng)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(source[parents], noise)
            target = old.cached_sample(teacher_train, parents, independent_noise)
            observed = old.observe_generated_carrier(prediction['signal_pre'], ste=True, q_floor=fixed['q_floor'], return_carrier=True)
            truth = old.observe_generated_carrier(target, q_floor=fixed['q_floor'], return_carrier=True)
            loss = fixed['distribution_loss']
            context = loss.create_projection_context(step)
            parts = loss.wave_components(observed['window_V'], truth['window_V'], None, context, mean_response=False)
            wave = sum(old.WAVE_WEIGHTS[key]*value for key,value in parts.items())
            integral = loss.integral_loss(observed['window_V'], truth['window_V'], None, context)
            f = (observed['carrier_adc']-observed['baseline_mean_V'][:, None])/fixed['carrier_scale']
            r = (truth['carrier_adc']-truth['baseline_mean_V'][:, None])/fixed['carrier_scale']
            directions = torch.randn(384, 16, generator=projection_rng, dtype=torch.float64).to(device)
            directions = directions/directions.norm(dim=0, keepdim=True)
            carrier = .5*(old.empirical_wasserstein_1d(f, r)+old.empirical_wasserstein_1d(f@directions, r@directions))
            objective = wave+2.*integral+.25*carrier+.01*prediction['regular']
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError('Nonfinite response030 objective')
            objective.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            progress['total_completed_updates'] += 1
            progress['current_run_completed_updates'] = step
            record = dict(mode='unpaired', seed=seed, completed_updates=step, objective=float(objective.detach()),
                gradient_norm=float(gradient), gradient_clipped=bool(gradient > 5.), wave=float(wave.detach()),
                signed_integrals=float(integral.detach()), carrier384=float(carrier.detach()),
                regular=float(prediction['regular'].detach()), source_unique_parents=128, teacher_unique_parents=128)
            log.write(json.dumps(record, allow_nan=False)+'\n')
            if step == 1 or step % 100 == 0:
                old.write_json(out/'STATUS.json', dict(record, UTC=datetime.now(timezone.utc).isoformat()))
                print(json.dumps(record, allow_nan=False), flush=True)
                size = sum(path.stat().st_size for path in out.parent.rglob('*') if path.is_file())
                if size > p['resources']['output_MiB']*1024**2:
                    raise OSError('Registered response030 output limit exceeded')
    model.eval()
    torch.save({name:value.detach().cpu() for name,value in model.state_dict().items()}, out/'final.pt')


def run(p, output, device, progress):
    started = time.monotonic()
    output.mkdir(parents=True)
    old.write_json(output/'START.json', dict(UTC=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), config=p,
        device=device, reference_029_sha256=hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest(),
        scientific_array_reads=0, checkpoint_reads=0, paired_retraining=False, automatic_restart=False))
    train, heldout, stats = old.make_sources(p)
    teacher = old.make_teacher(stats, p, device)
    teacher_train = old.factors(teacher, train['source'].to(device))
    teacher_test = old.factors(teacher, heldout['source'].to(device))
    P, rawP, scales = old.teacher_projections(teacher_test)
    old.write_json(output/'TEACHER.json', dict(definition=p['teacher'], teacher_seed=p['teacher_seed'], stats=stats,
        same_teacher_and_projections_as029=True, local_law_change_not_tested=True))
    # Exact 029 TRAIN scale population, RNG address and observation algebra.
    generator = torch.Generator(device=device).manual_seed(202610172901)
    rows = torch.arange(128, device=device).repeat_interleave(8)
    noise = teacher.sample_noise(len(rows), device=device, generator=generator)
    reference = old.cached_sample(teacher_train, rows, noise)
    means = teacher_train['A'][rows]
    observation = old.observe_generated_carrier(reference, q_floor=1e-8, return_carrier=True)
    source_window = old.crop_existing_values(train['source'].repeat_interleave(8, dim=0).to(device), q_floor=1e-8)['window_V']
    loss_stats = old.fit_loss_statistics(observation['window_V'].cpu(), source_window.cpu(), observation['baseline100'].cpu())
    integral_stats = old.fit_integral_statistics014(observation['window_V'].cpu())
    fixed = dict(mean_MSE_scale=means.square().mean().detach(), residual_MSE_scale=(reference-means).square().mean().detach(),
        carrier_scale=(observation['carrier_adc']-observation['baseline_mean_V'][:, None]).square().mean(0).sqrt().clamp_min(old.ADC_STEP),
        q_floor=float(loss_stats['q_floor']))
    old.write_json(output/'TRAIN_SCALES.json', dict(mean_MSE_scale=float(fixed['mean_MSE_scale']),
        residual_MSE_scale=float(fixed['residual_MSE_scale']), carrier_scale=fixed['carrier_scale'].cpu().tolist(),
        carrier_scale_domain='same single hard ADC carrier384 minus its original prefix100 mean',
        fitted_parents=128, fitted_draws=8, heldout_used=False, real_or_checkpoint_reads=0))
    del reference, means, source_window, observation, noise
    results = []
    for seed in p['learner_seeds']:
        model = old.N01Response029(stats, seed=seed).to(device)
        fixed['distribution_loss'] = old.N01Loss014(loss_stats, integral_stats, seed=seed+600,
            evaluation_seed=seed+700, integral_seed=seed+800).to(device)
        destination = output/f'unpaired_seed{seed}'
        train030(model, teacher_train, train, fixed, p, seed, destination, started, progress)
        results.append(old.evaluate(model, teacher_test, heldout, P, rawP, scales, p, seed, destination, 'unpaired'))
        del model
    engineering = p['execution_kind'] == 'engineering_artificial_smoke'
    result = dict(schema='n01/response030-known-teacher-result/v1', native_status='COMPLETE',
        passed=all(value['passed'] for value in results), runs=results, elapsed_seconds=time.monotonic()-started,
        execution_kind=p['execution_kind'], total_completed_updates=progress['total_completed_updates'],
        engineering_updates=progress['total_completed_updates'] if engineering else 0,
        diagnostic_updates=0 if engineering else progress['total_completed_updates'],
        paired_retraining=False, controlled_change=p['controlled_change'],
        scientific_array_reads=0, checkpoint_reads=0, REAL_pilot_started=False,
        scientific_status='response_match_not_established', physical_detector_acceptance=False,
        limitation='Known teacher, unchanged two-template Q interpolation. The only controlled change from029 is marginal parent sampling; no conditional-label supervision or REAL response acceptance.')
    result['decision'] = ('engineering_path_exercised_not_scientific_acceptance' if engineering
        else 'all_parent_unpaired_conditional_recovery_passed' if result['passed']
        else 'all_parent_unpaired_conditional_recovery_failed')
    old.write_json(output/'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--engineering-one-update', action='store_true')
    args = parser.parse_args()
    raw = Path(args.config).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Frozen response030 config SHA mismatch')
    if hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest() != REFERENCE_029_SHA256:
        raise ValueError('Closed029 runner bytes changed')
    p = json.loads(raw)
    if p['status'] != 'frozen' or any(p[key] != value for key,value in settings030().items() if key != 'status'):
        raise ValueError('response030 requires its frozen single-factor configuration')
    if args.engineering_one_update:
        p['execution_kind'] = 'engineering_artificial_smoke'
        p['learner_seeds'] = [2026101799]
        p['training']['unpaired_updates'] = 1
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(p['resources']['CPU_threads'])
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith('cuda'):
        target = torch.device(args.device)
        total = torch.cuda.get_device_properties(target).total_memory
        torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/total, target)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve previous030 attempt: '+str(output))
    def terminate(signum, frame):
        raise InterruptedError('response030 received signal '+str(signum))
    signal.signal(signal.SIGTERM, terminate)
    progress = dict(total_completed_updates=0, current_run_completed_updates=0, learner_seed=None)
    try:
        result = run(p, output, args.device, progress)
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        old.write_json(output/'FAILED.json', dict(native_status='FAILED', error_type=type(error).__name__,
            error=str(error), progress=progress, execution_kind=p['execution_kind'],
            scientific_array_reads=0, checkpoint_reads=0, automatic_restart=False))
        raise
    print(json.dumps(dict(passed=result['passed'], completed_updates=result['total_completed_updates']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
