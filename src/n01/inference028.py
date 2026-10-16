"""Frozen Q1/M0 research inference on the existing clean SOURCE carrier384."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .electronics_v3 import ObservedBaselineNoiseV3
from .model008 import N01Response008
from .model010 import tensor_state_digest010
from .model016 import FrozenM0ChargeResponse016
from .observation008 import FrozenObservedNoise008, observe_generated_carrier


def verified_path028(record):
    """Verify a pinned external file without decoding its contents."""
    path = Path(record['path'])
    if path.stat().st_size != record['bytes']:
        raise ValueError('028 file size mismatch: '+str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != record['sha256']:
        raise ValueError('028 file SHA mismatch: '+str(path))
    return path


class ResearchResponse028:
    """Stateless CPU sampling; repeated same-seed calls replay the same bank.

    Input is float64 [parent,384] in V on the bound 1 ns software grid.
    Output rows are parent-major/draw-minor. IDs never enter a neural module.
    This is a research reference, not an accepted physical response model.
    """

    def _run(self, name, charges, function):
        self.accounting['inflight'] = dict(operation=name, maximum_charges=charges,
            completion='unknown until function returns')
        for key, value in charges.items():
            self.accounting['attempted'][key] += value
        result = function()
        for key, value in charges.items():
            self.accounting['completed'][key] += value
        self.accounting['inflight'] = None
        return result

    @torch.no_grad()
    def sample(self, source_waveforms, n_draws, seed, detector_config, noise_hypothesis,
               *, model_seed, parent_ids, shape_enabled=True, charge_enabled=True,
               return_contributions=False):
        if (not isinstance(source_waveforms, torch.Tensor) or source_waveforms.ndim != 2
                or source_waveforms.shape[1] != 384 or not len(source_waveforms)
                or source_waveforms.dtype != torch.float64 or source_waveforms.device.type != 'cpu'):
            raise ValueError('028 needs a nonempty CPU float64 clean SOURCE [P,384] tensor')
        if type(n_draws) is not int or n_draws < 1 or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('028 requires positive integer draws and nonnegative 63-bit sampling seed')
        if (type(model_seed) is not int or model_seed not in (0, 1)
                or noise_hypothesis not in ('fast', 'slow', 'common_level')
                or detector_config != self.detector_config):
            raise ValueError('028 model seed, noise hypothesis, or detector binding differs')
        if any(type(v) is not bool for v in (shape_enabled, charge_enabled, return_contributions)):
            raise TypeError('028 module and diagnostic switches must be bool')
        if len(parent_ids) != len(source_waveforms):
            raise ValueError('028 parent references must match SOURCE rows')
        self.accounting['attempted']['sample_calls'] += 1
        rows = len(source_waveforms)*n_draws
        seeds = {name: int.from_bytes(hashlib.sha256(
            f'n01/wp6-028/v1/{seed}/{name}'.encode('utf8')).digest()[:8], 'little') & (2**63-1)
            for name in ('shape', 'noise005', 'extension', 'charge')}
        generators = self._run('four_sampling_generators', dict(sampling_generator_instances=4),
            lambda: {name: torch.Generator(device='cpu').manual_seed(value) for name, value in seeds.items()})
        shape = self._run('shape_latent', dict(sampling_randn_requests=1,
            sampling_normal_scalars=rows*8, shape_f32_scalars=rows*8),
            lambda: torch.randn(rows, 8, generator=generators['shape'], dtype=torch.float32))
        noise = {}
        for name, dims in ObservedBaselineNoiseV3.noise_shapes.items():
            count = rows*(100 if name == 'baseline_gaussian' else 1)
            key = 'noise005_gaussian_f32_scalars' if name == 'baseline_gaussian' else 'noise005_scale_f32_scalars'
            noise[name] = self._run(name, dict(sampling_randn_requests=1,
                sampling_normal_scalars=count, **{key: count}),
                lambda dims=dims: torch.randn((rows,)+dims, generator=generators['noise005'], dtype=torch.float32))
        epsilon = self._run('extension_latent', dict(sampling_randn_requests=1,
            sampling_normal_scalars=rows*284, extension_f32_scalars=rows*284),
            lambda: torch.randn(rows, 1, 284, generator=generators['extension'], dtype=torch.float32))
        charge = self._run('charge_latent', dict(sampling_randn_requests=1,
            sampling_normal_scalars=rows, charge_f64_scalars=rows),
            lambda: torch.randn(rows, generator=generators['charge'], dtype=torch.float64))
        latents = dict(shape=shape, baseline_gaussian=noise['baseline_gaussian'],
            baseline_scale=noise['baseline_scale'], extension=epsilon, charge=charge)
        model = self.models[model_seed]
        expanded = source_waveforms.repeat_interleave(n_draws, dim=0)
        parts = []
        for start in range(0, rows, 64):
            stop = min(start+64, rows)
            count = stop-start
            parts.append(self._run('frozen_A_and_M0_head_and_optional_charge',
                dict(response_rows=count, old_A_rows=count, head_actual_parameter_forward_rows=count,
                     charge_module_enabled_rows=count if charge_enabled else 0,
                     shape_enabled_rows=count if shape_enabled else 0),
                lambda start=start, stop=stop: model(expanded[start:stop], shape[start:stop], charge[start:stop],
                    enabled=shape_enabled, charge_enabled=charge_enabled)))
        # Keep the primitive's original tensor alias for the one-block/off path.
        names = ('signal_pre', 'M0_reference_signal_pre_V', 'base_signal_pre_V', 'random_shape_pre_V')
        signal = {name: parts[0][name] if len(parts) == 1 else torch.cat([p[name] for p in parts]) for name in names}
        if not charge_enabled and len(parts) > 1:
            if any(p['signal_pre'] is not p['M0_reference_signal_pre_V'] for p in parts):
                raise RuntimeError('028 original chargeoff tensor alias changed')
            signal['M0_reference_signal_pre_V'] = signal['signal_pre']
        common = self._run('frozen005_prefix', dict(noise005_rows=rows), lambda: self.extension.sample_prefix(noise))
        electronic = self._run('selected_extension_'+noise_hypothesis, dict(selected_extension_rows=rows),
            lambda: self.extension.extend_prefix(common['prefix_pre'], common['scale'], epsilon, noise_hypothesis))
        pre_adc = signal['signal_pre']+electronic['noise_pre']
        result = self._run('hard_ADC_firstargmax_gather',
            dict(hard_observation_rows=rows, ADC_carrier_scalar_samples=rows*384),
            lambda: observe_generated_carrier(pre_adc, ste=False, q_floor=self.q_floor,
                return_carrier=return_contributions, return_indices=True))
        result.update(parent_ids=tuple(parent_ids),
            row_parent=torch.arange(len(source_waveforms)).repeat_interleave(n_draws),
            draw=torch.arange(n_draws).repeat(len(source_waveforms)), model_seed=model_seed,
            noise_hypothesis=noise_hypothesis, detector_config=self.detector_config,
            shape_enabled=shape_enabled, charge_enabled=charge_enabled,
            streams=dict(sampling_seed=seed, CPU_seeds=seeds,
                latent_sha256={name: tensor_state_digest010({name: value}) for name, value in latents.items()},
                full_rows=rows, NN_chunk_rows=64, semantics='stateless complete-batch replay'),
            applicability=dict(input='prepared clean SOURCE carrier384 V', software_sample_interval_ns=1,
                physical_calibration='unknown', scientific_status='response_match_not_established',
                full_WP5_WP6_complete=False))
        if return_contributions:
            result['contributions'] = dict(**signal, noise_pre=electronic['noise_pre'],
                prefix_pre=common['prefix_pre'], scale=common['scale'], **latents)
        self.accounting['completed']['sample_calls'] += 1
        return result


@torch.no_grad()
def load_reference028(config, *, device='cpu', accounting=None):
    """Load the ROOT-frozen028 reference, not arbitrary CP combinations.

    Exact CP byte pins and strict loads bind the original A/005/M0/Q1 tensors.
    Historical frozen-state metadata retain their original hash encoding;
    tensor_state_digest010 separately verifies unchanged live inference state.
    No training loader or main is executed.
    """
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding='utf8'))
    if (config['schema'] != 'n01/wp6-028-config/v1' or config['run_id'] != 'wp6_028_20261017'
            or config['status'] != 'frozen' or device != 'cpu'):
        raise ValueError('028 requires its bound config and CPU device')
    response = ResearchResponse028()
    response.accounting = accounting if accounting is not None else dict(
        attempted={key: 0 for key in config['budget_caps']},
        completed={key: 0 for key in config['budget_caps']}, inflight=None)
    response.config = config
    for record in config['source_files'].values():
        verified_path028(record)
    values = {}
    for name in ('statistics', 'data_statistics', 'base_config'):
        values[name] = json.loads(verified_path028(config['inputs'][name]).read_text(encoding='utf8'))
    observation = json.loads(verified_path028(config['contracts']['observation']).read_text(encoding='utf8'))
    statistics = values['statistics']
    if (statistics['statistics_split'] != 'TRAIN' or values['data_statistics']['split'] != 'TRAIN'
            or statistics['prepared_dataset_sha256'] != config['inputs']['dataset']['sha256']
            or statistics['data_statistics_sha256'] != config['inputs']['data_statistics']['sha256']
            or statistics['DEV_waveforms_decoded'] is not False):
        raise ValueError('028 frozen TRAIN statistic provenance differs')
    response.q_floor = float(statistics['loss']['q_floor'])
    response.detector_config = hashlib.sha256(json.dumps(dict(
        checkpoints={name: config['inputs'][name]['sha256'] for name in config['checkpoint_keys']},
        statistics=config['inputs']['statistics']['sha256'], observation=config['contracts']['observation']['sha256'],
        sources={name: record['sha256'] for name, record in config['source_files'].items()},
        input_kind='clean_SOURCE_carrier384_float64_V', NN_chunk_rows=64),
        sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()
    response.models = {}
    checkpoints = {}
    for name in config['checkpoint_keys']:
        path = verified_path028(config['inputs'][name])
        checkpoints[name] = response._run('load_CP_'+name, dict(CP_loads=1),
            lambda path=path: torch.load(path, map_location='cpu', weights_only=False))
    a, noise_cp = checkpoints['deterministic'], checkpoints['noise005']
    if (a['schema'] != 'response008/v1' or a['arm'] != 'deterministic' or a['G_updates'] != 19000
            or a['stage'] != 'low_learning_rate' or a['config'] != values['base_config']
            or a['statistics'] != statistics or noise_cp['schema_version'] != 5
            or noise_cp['update'] != 5000 or noise_cp['model_type'] != 'ObservedBaselineNoiseV3'):
        raise ValueError('028 frozen A/005 metadata differs')
    with torch.random.fork_rng(devices=[]):
        baseline = response._run('A_constructor_discarded_initialization',
            dict(constructor_uniform_requests=114, constructor_uniform_scalars=1473092),
            lambda: N01Response008(statistics['model'], 'deterministic'))
        baseline.load_state_dict(a['model'], strict=True)
        baseline.set_stage(a['stage']); baseline.eval(); baseline.requires_grad_(False); baseline.to(device)
        noise005 = ObservedBaselineNoiseV3(noise_cp['initialization'])
        noise005.load_state_dict(noise_cp['model'], strict=True)
        noise005.eval(); noise005.requires_grad_(False); noise005.to(device)
        response.extension = FrozenObservedNoise008(noise005).eval().requires_grad_(False)
        for index, initialization in enumerate(config['model_initialization_seeds']):
            m0, q1 = checkpoints[f'M0_head_seed{index}'], checkpoints[f'Q1_final_seed{index}']
            if (m0['schema'] != 'n01/wp3-checkpoint/v1' or m0['arm'] != 'M0'
                    or q1['schema'] != 'n01/wp3-016-checkpoint/v1' or q1['candidate'] != 'Q1'
                    or m0['initial_seed'] != initialization or q1['initial_seed'] != initialization
                    or m0['G_updates'] != 2000 or q1['G_updates'] != 2000
                    or m0['purpose'] != 'final' or q1['purpose'] != 'final'
                    or not m0['complete_step_boundary'] or not q1['complete_step_boundary']
                    or not bool((m0['mean_theta'] == 0).all())
                    or q1['M0_reference']['sha256'] != config['inputs'][f'M0_head_seed{index}']['sha256']
                    or q1['M0_head_sha256'] != tensor_state_digest010(m0['new_head'])
                    or m0['frozen_state_sha256'] != q1['frozen_state_sha256']):
                raise ValueError('028 per-seed final M0/Q1 binding differs')
            model = response._run('M0_head_constructor_seed'+str(index),
                dict(constructor_uniform_requests=2, constructor_uniform_scalars=1288,
                     constructor_normal_requests=1, constructor_normal_scalars=1280),
                lambda: FrozenM0ChargeResponse016(baseline, m0['new_head'],
                    seed=initialization, calibration=q1['charge_initialization']))
            model.theta_a.copy_(q1['theta_a']); model.theta_b.copy_(q1['theta_b'])
            model.requires_grad_(False); model.eval()
            response.models[index] = model
    response.observation_contract = observation
    response.frozen_digests = dict(A=tensor_state_digest010(baseline),
        extension=tensor_state_digest010(response.extension),
        **{f'model{index}': tensor_state_digest010(model) for index, model in response.models.items()})
    return response
