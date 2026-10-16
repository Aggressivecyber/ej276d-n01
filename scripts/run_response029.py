"""Known-response paired/unpaired diagnosis; never opens scientific data or CPs.

Paired latent supervision is an engineering positive control, not available REAL
supervision. Both learning modes start from the same seed-specific initial state.
Analytic pre-ADC conditional moments decide recovery; fresh common draws measure
Monte Carlo error and report the unchanged hard-ADC observation separately.
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

import numpy as np
import torch
import torch.nn.functional as F

from n01.losses008 import ADC_STEP, fit_loss_statistics
from n01.losses008_v2 import WAVE_WEIGHTS, empirical_wasserstein_1d
from n01.losses014 import N01Loss014, fit_integral_statistics014
from n01.model import causal_fir
from n01.model008 import canonical_charge008
from n01.model029 import N01Response029
from n01.observation008 import crop_existing_values, observe_generated_carrier


def settings029():
    return dict(schema='n01/response029-known-teacher/v1', run_id='response029_20261017',
        status='draft', stage='teacher_paired_then_unpaired', data_kind='artificial_known_response',
        learner_seeds=[2026101701, 2026101702], teacher_seed=2026101729,
        source=dict(TRAIN_per_template=64, heldout_per_template=32, Q_range_Vns=[.5, 8.],
            Q_stratum_edges_Vns=[.5, 1.5, 4., 8.], templates=[dict(rise_width=3., tail=30.), dict(rise_width=5., tail=60.)],
            peak=126, prefix_zero=100, carrier_samples=384),
        teacher=dict(mean_gain=1.12, time_scale=1.10, shape_weight_multiplier=10.,
            charge_v=[.08, .04, 0., 0., 0., 0., 0., 0., 0.], local_initialization_retained=True),
        training=dict(paired_updates=1000, unpaired_updates=1000, parents_per_update=64,
            draws_per_parent=2, mean_lr=1e-4, new_head_lr=3e-4, weight_decay=0., gradient_clip=5.,
            status_interval=100, statistics_draws_per_TRAIN_parent=8,
            wave_weight=1., integral_weight=2., carrier_weight=.25, regular_weight=.01),
        evaluation=dict(analytic_parents=64, MC_parent_rows=list(range(0, 32, 4))+list(range(32, 64, 4)),
            MC_draws_per_parent=256, projections=9, mean_NRMSE_max=.03,
            charge_variance_relative_RMS_max=.15, standardized_covariance_relative_Frobenius_max=.20,
            require_all_template_Q_strata=True, MC_is_gate=False),
        resources=dict(CPU_threads=4, RSS_GiB=12, GPU_GiB=10, wall_seconds=7200, output_MiB=256),
        execution_kind='known_teacher_diagnostic',
        design_revision='800/1200 was an unexecuted draft; both modes fixed at 1000 before training',
        no_REAL_or_checkpoint_reads=True, no_discriminator=True, no_noise005=True,
        automatic_restart=False, scientific_status='response_match_not_established')


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    tmp.replace(path)


def make_sources(p):
    """Disjoint log-Q grids; two analytic templates, never identity features."""
    t = torch.arange(384, dtype=torch.float64)
    grids = (torch.arange(64, dtype=torch.float64)/63., (2*torch.arange(32, dtype=torch.float64)+.5)/64.)
    result = []
    for phase in grids:
        waves, charges, labels = [], [], []
        q = torch.exp(math.log(.5)+phase*math.log(16.))
        for index, template in enumerate(p['source']['templates']):
            base = torch.where(t < 126, torch.exp(-.5*((t-126)/template['rise_width'])**2),
                               torch.exp(-(t-126)/template['tail']))
            base[:100] = 0
            waves.append(q[:, None]*base[None]/canonical_charge008(base))
            charges.append(q)
            labels.append(torch.full((len(q),), index, dtype=torch.int64))
        result.append(dict(source=torch.cat(waves), Q=torch.cat(charges), template=torch.cat(labels)))
    train, heldout = result
    stats = dict(q_ref=float(train['Q'].median()), a_ref=float(train['source'].amax(1).median()),
        log_q_min=math.log(.5)-.2, log_q_max=math.log(8.)+.2)
    return train, heldout, stats


def make_teacher(stats, p, device):
    model = N01Response029(stats, seed=p['teacher_seed']).to(device).eval().requires_grad_(False)
    with torch.no_grad():
        mean = model.mean_model
        mean.charge_head.parameters_from_shape.weight.zero_()
        mean.charge_head.parameters_from_shape.bias.zero_()
        mean.charge_head.parameters_from_shape.bias[17] = math.atanh(math.log(1.12)/2.)
        mean.mean_time[-1].weight.zero_()
        mean.mean_time[-1].bias.fill_(math.atanh(math.log(1.10)/math.log(2.)))
        model.basis_head.weight.mul_(10.)
        model.charge_head[-1].weight.zero_()
        model.charge_head[-1].bias.copy_(torch.tensor(p['teacher']['charge_v'], device=device))
    return model


@torch.no_grad()
def factors(model, source):
    noise = {key: source.new_zeros((len(source),)+shape) for key, shape in model.noise_shapes.items()}
    out = model(source, noise, enabled=False)
    return dict(A=out['base_signal_pre_V'], B=out['B'], v=out['charge_v'],
        envelope=out['signal_envelope'], taps=model.local_filter.taps().detach().double(),
        gate=model.mean_model.gate[0, 0], projection=model.mean_model.projection[0, 0])


def cached_sample(f, rows, noise):
    """The exact model029 output algebra on cached frozen teacher factors."""
    A, B, v, envelope = (f[key][rows] for key in ('A', 'B', 'v', 'envelope'))
    z = noise['joint'].double()
    alpha = ((v*z).sum(1)-.5*v.square().sum(1)).exp()
    local = causal_fir(F.pad(envelope[:, None], (16, 0))*noise['local_white'].double(), f['taps'], extended=True)
    local = local[:, 0]*f['gate']
    local = local-canonical_charge008(local)[:, None]*f['projection']
    return alpha[:, None]*A+(B*z[:, 1:, None]).sum(1)+local


def projected_moments(f, P):
    """Exact conditional covariance, including shared lognormal/shape terms."""
    A, B, v = f['A'], f['B'], f['v']
    a = torch.einsum('bt,btd->bd', A, P)
    b = torch.einsum('bkt,btd->bkd', B, P)
    charged = torch.expm1(v.square().sum(1))[:, None, None]*a[:, :, None]*a[:, None, :]
    shape = b.transpose(1, 2)@b
    shared = (b*v[:, 1:, None]).sum(1)
    cross = a[:, :, None]*shared[:, None, :]+shared[:, :, None]*a[:, None, :]
    qmask = P.new_zeros(384); qmask[110:366] = 1
    adjusted = P-qmask[None, :, None]*torch.einsum('t,btd->bd', f['projection'], P)[:, None]
    weighted = adjusted*f['gate'][None, :, None]
    coefficients = torch.zeros_like(weighted)
    for lag in range(17):
        coefficients[:, :384-lag] += f['taps'][lag]*weighted[:, lag:]
    coefficients = coefficients*f['envelope'][:, :, None]
    local = coefficients.transpose(1, 2)@coefficients
    covariance = charged+shape+cross+local
    Q = canonical_charge008(A)
    variance_Q = Q.square()*torch.expm1(v.square().sum(1))
    return dict(mean=a, covariance=covariance, charge=charged, shape=shape, cross=cross,
                local=local, Q=Q, variance_Q=variance_Q)


def teacher_projections(f):
    """One charge direction plus eight shape directions orthogonal to A."""
    unit = f['A']/f['A'].norm(dim=1, keepdim=True)
    columns = f['B'].transpose(1, 2)
    columns = columns-unit[:, :, None]*torch.einsum('bt,btk->bk', unit, columns)[:, None]
    shape, _ = torch.linalg.qr(columns, mode='reduced')
    charge = torch.zeros_like(unit); charge[:, 110:366] = 1/16.
    raw = torch.cat((charge[:, :, None], shape), 2)
    raw_moments = projected_moments(f, raw)
    scales = raw_moments['covariance'].diagonal(dim1=1, dim2=2).sqrt()
    return raw/scales[:, None, :], raw, scales


def strata(data):
    q = data['Q']
    levels = torch.bucketize(q, torch.tensor([1.5, 4.], dtype=q.dtype), right=True)
    return [('all', torch.ones(len(q), dtype=torch.bool))]+[
        (f'template{template}_{name}', (data['template'] == template)&(levels == level))
        for template in range(2) for level, name in enumerate(('low', 'middle', 'high'))]


def recovery_metrics(predicted, target, pred_moments, target_moments, data, p):
    cells = {}
    for name, mask_cpu in strata(data):
        mask = mask_cpu.to(predicted['A'].device)
        mean = (predicted['A'][mask]-target['A'][mask]).norm()/target['A'][mask].norm()
        variance = (pred_moments['variance_Q'][mask]-target_moments['variance_Q'][mask]).norm()/target_moments['variance_Q'][mask].norm()
        covariance = (pred_moments['covariance'][mask]-target_moments['covariance'][mask]).norm()/target_moments['covariance'][mask].norm()
        value = dict(parents=int(mask.sum()), mean_NRMSE=float(mean),
            charge_variance_relative_RMS=float(variance), standardized_covariance_relative_Frobenius=float(covariance))
        value['passed'] = bool(mean <= p['evaluation']['mean_NRMSE_max']
            and variance <= p['evaluation']['charge_variance_relative_RMS_max']
            and covariance <= p['evaluation']['standardized_covariance_relative_Frobenius_max'])
        cells[name] = value
    finite = all(bool(torch.isfinite(value).all()) for value in
        (predicted['A'], predicted['B'], predicted['v'], pred_moments['covariance'], pred_moments['variance_Q']))
    return dict(cells=cells, all_finite=finite, rows_retained=len(data['source']),
        passed=finite and all(value['passed'] for value in cells.values()))


@torch.no_grad()
def evaluate(model, target, data, P, rawP, scales, p, seed, out, mode):
    device = target['A'].device
    predicted = factors(model, data['source'].to(device))
    pm, tm = projected_moments(predicted, P), projected_moments(target, P)
    metrics = recovery_metrics(predicted, target, pm, tm, data, p)
    rows = torch.tensor(p['evaluation']['MC_parent_rows'], device=device)
    draws = p['evaluation']['MC_draws_per_parent']
    repeated = rows.repeat_interleave(draws)
    generator = torch.Generator(device=device).manual_seed(seed+9000)
    noise = model.sample_noise(len(repeated), device=device, generator=generator)
    fake, truth = cached_sample(predicted, repeated, noise), cached_sample(target, repeated, noise)
    score_fake = fake.reshape(len(rows), draws, 384)@P[rows]
    score_true = truth.reshape(len(rows), draws, 384)@P[rows]
    mc = {}
    score_arrays = {}
    hard_arrays = {}
    for name, wave, scores, moment in (('learner', fake, score_fake, pm), ('teacher', truth, score_true, tm)):
        q = canonical_charge008(wave).reshape(len(rows), draws)
        centered = scores-scores.mean(1, keepdim=True)
        covariance = centered.transpose(1, 2)@centered/(draws-1)
        variance = q.var(1, correction=1)
        qcenter = q-q.mean(1, keepdim=True)
        variance_SE = ((qcenter.pow(4).mean(1)-(draws-3)/(draws-1)*variance.square()).clamp_min(0)/draws).sqrt()
        outer = centered[:, :, :, None]*centered[:, :, None, :]
        covariance_SE = outer.var(1, correction=1).div(draws).sqrt()
        hard = observe_generated_carrier(wave, q_floor=1e-8)
        mc[name] = dict(Q_variance_relative_RMS=float((variance-moment['variance_Q'][rows]).norm()/moment['variance_Q'][rows].norm()),
            projected_covariance_relative_Frobenius=float((covariance-moment['covariance'][rows]).norm()/moment['covariance'][rows].norm()),
            variance_SE_relative_RMS=float(variance_SE.norm()/moment['variance_Q'][rows].norm()),
            covariance_SE_relative_Frobenius=float(covariance_SE.norm()/moment['covariance'][rows].norm()),
            hard256_Q_mean=float(hard['Q'].mean()), hard256_Q_std=float(hard['Q'].std()),
            hard256_PSD_mean=float(hard['PSD'].mean()), flags={key:int(value.sum()) for key,value in hard['flags'].items()})
        score_arrays[name+'_scores'] = scores.cpu().numpy()
        score_arrays[name+'_Q'] = q.cpu().numpy()
        hard_arrays[name+'_hard_Q'] = hard['Q'].cpu().numpy()
        hard_arrays[name+'_hard_PSD'] = hard['PSD'].cpu().numpy()
    metrics.update(mode=mode, learner_seed=seed, analytic_parents=len(data['source']),
        MC_parents=len(rows), MC_draws_per_parent=draws, MC_is_gate=False, MC_common_latents=True,
        MC=mc, conditional_moments='exact preADC; standardized fixed teacher linear projections',
        local_changed_law_recovery_claim=False, physical_detector_acceptance=False)
    write_json(out/'RESULT.json', metrics)
    np.savez_compressed(out/'SUMMARY.npz', teacher_mean=target['A'].cpu().numpy(),
        learner_mean=predicted['A'].cpu().numpy(), source_Q=data['Q'].numpy(), template=data['template'].numpy(),
        teacher_Q_variance=tm['variance_Q'].cpu().numpy(), learner_Q_variance=pm['variance_Q'].cpu().numpy(),
        teacher_covariance=tm['covariance'].cpu().numpy(), learner_covariance=pm['covariance'].cpu().numpy(),
        teacher_covariance_local=tm['local'].cpu().numpy(), teacher_covariance_shape=tm['shape'].cpu().numpy(),
        projections_raw=rawP.cpu().numpy(), projection_scales=scales.cpu().numpy(), MC_parent_rows=rows.cpu().numpy(),
        **score_arrays, **hard_arrays)
    return metrics


def train_mode(model, teacher_train, train, fixed, p, seed, mode, out, started, progress):
    out.mkdir()
    model.train()
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW([
        dict(params=groups['mean'], lr=p['training']['mean_lr']),
        dict(params=groups['shape']+groups['charge']+groups['local'], lr=p['training']['new_head_lr'])],
        weight_decay=0.)
    parameters = list(model.parameters())
    device = teacher_train['A'].device
    source = train['source'].to(device)
    source_generator = torch.Generator(device='cpu').manual_seed(seed+100)
    target_generator = torch.Generator(device='cpu').manual_seed(seed+200)
    learner_rng = torch.Generator(device=device).manual_seed(seed+300)
    teacher_rng = torch.Generator(device=device).manual_seed(seed+400)
    projection_rng = torch.Generator(device='cpu').manual_seed(seed+500)
    steps = p['training']['paired_updates' if mode == 'paired' else 'unpaired_updates']
    progress.update(mode=mode, learner_seed=seed, current_run_completed_updates=0)
    write_json(out/'INITIAL.json', dict(seed=seed, mode=mode, updates=0, same_initial_state_between_modes=True,
        paired_supervision_is_manufactured_only=True, initial_state_sha256=state_digest(model.state_dict())))
    with (out/'metrics.jsonl').open('x', encoding='utf8', buffering=1) as log:
        for step in range(1, steps+1):
            if time.monotonic()-started > p['resources']['wall_seconds']:
                raise TimeoutError('Registered response029 wall limit exceeded')
            parents = torch.randperm(len(source), generator=source_generator)[:64].repeat_interleave(2).to(device)
            noise = model.sample_noise(len(parents), device=device, generator=learner_rng)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(source[parents], noise)
            if mode == 'paired':
                target = cached_sample(teacher_train, parents, noise)
                teacher_mean = teacher_train['A'][parents]
                mean_loss = (prediction['base_signal_pre_V']-teacher_mean).square().mean()/fixed['mean_MSE_scale']
                residual_loss = ((prediction['signal_pre']-prediction['base_signal_pre_V'])-(target-teacher_mean)).square().mean()/fixed['residual_MSE_scale']
                objective = mean_loss+residual_loss
                components = dict(mean=float(mean_loss.detach()), residual=float(residual_loss.detach()))
            else:
                target_rows = torch.randperm(len(source), generator=target_generator)[:64].repeat_interleave(2).to(device)
                independent_noise = model.sample_noise(len(target_rows), device=device, generator=teacher_rng)
                target = cached_sample(teacher_train, target_rows, independent_noise)
                observed = observe_generated_carrier(prediction['signal_pre'], ste=True, q_floor=fixed['q_floor'], return_carrier=True)
                truth = observe_generated_carrier(target, q_floor=fixed['q_floor'], return_carrier=True)
                loss = fixed['distribution_loss']
                context = loss.create_projection_context(step)
                parts = loss.wave_components(observed['window_V'], truth['window_V'], None, context, mean_response=False)
                wave = sum(WAVE_WEIGHTS[key]*value for key,value in parts.items())
                integral = loss.integral_loss(observed['window_V'], truth['window_V'], None, context)
                f = observed['carrier_adc']-observed['baseline_mean_V'][:, None]
                r = truth['carrier_adc']-truth['baseline_mean_V'][:, None]
                f, r = f/fixed['carrier_scale'], r/fixed['carrier_scale']
                directions = torch.randn(384, 16, generator=projection_rng, dtype=torch.float64).to(device)
                directions = directions/directions.norm(dim=0, keepdim=True)
                carrier = .5*(empirical_wasserstein_1d(f, r)+empirical_wasserstein_1d(f@directions, r@directions))
                objective = wave+2.*integral+.25*carrier+.01*prediction['regular']
                components = dict(wave=float(wave.detach()), signed_integrals=float(integral.detach()),
                    carrier384=float(carrier.detach()), regular=float(prediction['regular'].detach()))
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError('Nonfinite response029 objective')
            objective.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            progress['total_completed_updates'] += 1
            progress['current_run_completed_updates'] = step
            record = dict(mode=mode, seed=seed, completed_updates=step, objective=float(objective.detach()),
                gradient_norm=float(gradient), **components)
            log.write(json.dumps(record, allow_nan=False)+'\n')
            if step == 1 or step % 100 == 0:
                write_json(out/'STATUS.json', dict(record, UTC=datetime.now(timezone.utc).isoformat()))
                print(json.dumps(record, allow_nan=False), flush=True)
                output_bytes = sum(path.stat().st_size for path in out.parent.rglob('*') if path.is_file())
                if output_bytes > p['resources']['output_MiB']*1024**2:
                    raise OSError('Registered response029 output limit exceeded')
    model.eval()
    torch.save({name:value.detach().cpu() for name,value in model.state_dict().items()}, out/'final.pt')


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode('utf8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(p, output, device, progress):
    started = time.monotonic()
    output.mkdir(parents=True)
    write_json(output/'START.json', dict(UTC=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        config=p, device=device, scientific_array_reads=0, checkpoint_reads=0, automatic_restart=False))
    train, heldout, stats = make_sources(p)
    teacher = make_teacher(stats, p, device)
    teacher_train = factors(teacher, train['source'].to(device))
    teacher_test = factors(teacher, heldout['source'].to(device))
    P, rawP, scales = teacher_projections(teacher_test)
    exact = projected_moments(teacher_test, P)
    oracle = {}
    for name, mask_cpu in strata(heldout):
        mask = mask_cpu.to(device)
        a = exact['mean'][mask]
        outer = a[:, :, None]*a[:, None, :]
        covariance = exact['covariance'][mask]
        true_kappa = torch.expm1(teacher_test['v'][mask].square().sum(1))
        optimum = (outer*covariance).sum((1, 2))/outer.square().sum((1, 2))
        kappa = torch.maximum(torch.minimum(optimum, 1.15*true_kappa), .85*true_kappa)
        fitted = kappa[:, None, None]*outer
        ratio = (fitted-covariance).norm()/covariance.norm()
        oracle[name] = dict(best_charge_only_covariance_relative_Frobenius=float(ratio),
            allowed_Q_variance_relative_interval=[.85, 1.15], fitted_kappa=kappa.cpu().tolist(),
            fails_covariance_gate=bool(ratio > p['evaluation']['standardized_covariance_relative_Frobenius_max']))
    write_json(output/'TEACHER.json', dict(definition=p['teacher'], stats=stats,
        projections='charge and eight A-orthogonal QR shape directions, each divided by exact teacher conditional SD',
        charge_only_oracle=oracle, local_law_change_not_tested=True,
        oracle_scope='Closed-form per-parent best kappa*A*A^T, Q variance within +/-15%, with fixed exact teacher A; does not cover arbitrary wrong conditional means.',
        analytic_covariance_includes_charge_shape_cross_terms=True))
    if not all(cell['fails_covariance_gate'] for cell in oracle.values()):
        raise ValueError('Fixed teacher charge-only oracle does not fail every covariance gate')
    generator = torch.Generator(device=device).manual_seed(202610172901)
    rows = torch.arange(128, device=device).repeat_interleave(8)
    noise = teacher.sample_noise(len(rows), device=device, generator=generator)
    reference = cached_sample(teacher_train, rows, noise)
    means = teacher_train['A'][rows]
    observation = observe_generated_carrier(reference, q_floor=1e-8, return_carrier=True)
    source_window = crop_existing_values(train['source'].repeat_interleave(8, dim=0).to(device), q_floor=1e-8)['window_V']
    loss_stats = fit_loss_statistics(observation['window_V'].cpu(), source_window.cpu(), observation['baseline100'].cpu())
    integral_stats = fit_integral_statistics014(observation['window_V'].cpu())
    fixed = dict(mean_MSE_scale=means.square().mean().detach(), residual_MSE_scale=(reference-means).square().mean().detach(),
        carrier_scale=(observation['carrier_adc']-observation['baseline_mean_V'][:, None]).square().mean(0).sqrt().clamp_min(ADC_STEP),
        q_floor=float(loss_stats['q_floor']), loss_stats=loss_stats, integral_stats=integral_stats)
    write_json(output/'TRAIN_SCALES.json', dict(mean_MSE_scale=float(fixed['mean_MSE_scale']),
        residual_MSE_scale=float(fixed['residual_MSE_scale']), carrier_scale=fixed['carrier_scale'].cpu().tolist(),
        carrier_scale_domain='same single hard ADC carrier384 minus its original prefix100 mean',
        fitted_parents=128, fitted_draws=8, heldout_used=False, real_or_checkpoint_reads=0))
    del reference, means, source_window, observation, noise
    results = []
    initial = {}
    for mode in ('paired', 'unpaired'):
        if mode == 'unpaired' and p['execution_kind'] != 'engineering_artificial_smoke' and not all(value['passed'] for value in results):
            break
        for seed in p['learner_seeds']:
            model = N01Response029(stats, seed=seed).to(device)
            digest = state_digest(model.state_dict())
            if mode == 'paired':
                initial[seed] = digest
            elif digest != initial[seed]:
                raise RuntimeError('Paired/unpaired initial model states differ')
            fixed['distribution_loss'] = N01Loss014(loss_stats, integral_stats, seed=seed+600,
                evaluation_seed=seed+700, integral_seed=seed+800).to(device)
            destination = output/f'{mode}_seed{seed}'
            train_mode(model, teacher_train, train, fixed, p, seed, mode, destination, started, progress)
            result = evaluate(model, teacher_test, heldout, P, rawP, scales, p, seed, destination, mode)
            results.append(result)
            del model
    paired = [value for value in results if value['mode'] == 'paired']
    unpaired = [value for value in results if value['mode'] == 'unpaired']
    result = dict(schema='n01/response029-known-teacher-result/v1', native_status='COMPLETE',
        paired_passed=len(paired) == len(p['learner_seeds']) and all(value['passed'] for value in paired),
        unpaired_started=bool(unpaired), unpaired_passed=len(unpaired) == len(p['learner_seeds']) and all(value['passed'] for value in unpaired),
        runs=results, elapsed_seconds=time.monotonic()-started,
        execution_kind=p['execution_kind'], total_completed_updates=progress['total_completed_updates'],
        engineering_updates=progress['total_completed_updates'] if p['execution_kind'] == 'engineering_artificial_smoke' else 0,
        diagnostic_updates=progress['total_completed_updates'] if p['execution_kind'] == 'known_teacher_diagnostic' else 0,
        scientific_array_reads=0, checkpoint_reads=0, REAL_pilot_started=False,
        scientific_status='response_match_not_established', physical_detector_acceptance=False,
        limitation='Manufactured known-latent paired control and unpaired conditional-recovery diagnostic; heldout tests Q interpolation for the same two shapes, not unseen-shape generalization. Existing REAL fitting remains a separate next run.')
    result['passed'] = result['paired_passed'] and result['unpaired_passed']
    result['decision'] = ('engineering_paths_exercised_not_scientific_acceptance' if p['execution_kind'] == 'engineering_artificial_smoke'
        else 'paired_recovery_failed_unpaired_not_started' if not result['paired_passed']
        else 'known_teacher_paired_and_unpaired_recovered' if result['unpaired_passed']
        else 'paired_recovered_unpaired_conditional_recovery_failed')
    write_json(output/'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--engineering-updates', type=int, choices=[0, 1])
    args = parser.parse_args()
    raw = Path(args.config).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Frozen response029 config SHA mismatch')
    p = json.loads(raw)
    expected = settings029()
    if p['status'] != 'frozen' or any(p[key] != value for key,value in expected.items() if key != 'status'):
        raise ValueError('response029 only executes its frozen artificial diagnostic settings')
    if args.engineering_updates is not None:
        p['execution_kind'] = 'engineering_artificial_smoke'
        p['learner_seeds'] = [2026101799]
        p['training']['paired_updates'] = p['training']['unpaired_updates'] = args.engineering_updates
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
        raise FileExistsError('Preserve prior response029 attempt: '+str(output))
    def terminate(signum, frame):
        raise InterruptedError('response029 received signal '+str(signum))
    signal.signal(signal.SIGTERM, terminate)
    progress = dict(total_completed_updates=0, current_run_completed_updates=0, mode='initialization', learner_seed=None)
    try:
        result = run(p, output, args.device, progress)
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'FAILED.json', dict(native_status='FAILED', error_type=type(error).__name__,
            error=str(error), progress=progress, execution_kind=p['execution_kind'],
            scientific_array_reads=0, checkpoint_reads=0, automatic_restart=False))
        raise
    print(json.dumps(dict(passed=result['passed'], paired_passed=result['paired_passed'],
        unpaired_passed=result['unpaired_passed']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
