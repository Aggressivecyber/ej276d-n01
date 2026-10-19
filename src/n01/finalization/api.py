"""Frozen research inference for legacy028 and the rejected S candidate.

Library use requires this project's src/ and scripts/ on PYTHONPATH.
The S path never fits statistics or opens a SOURCE/REAL population archive.
"""
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from n01.inference028 import load_reference028, verified_path028


ROOT = Path(__file__).resolve().parents[3]


class Legacy028Facade:
    """Adapt argument names; the original loader and sampler stay unchanged."""

    def __init__(self, reference):
        self.reference = reference
        self.accounting = reference.accounting
        self.detector_config = reference.detector_config

    def sample(self, source_waveforms, parent_ids, n_draws, seed, model_seed=0,
               noise_hypothesis='fast', return_contributions=False, *,
               shape_enabled=True, charge_enabled=True):
        result = self.reference.sample(source_waveforms, n_draws, seed,
            self.detector_config, noise_hypothesis, model_seed=model_seed,
            parent_ids=parent_ids, shape_enabled=shape_enabled,
            charge_enabled=charge_enabled, return_contributions=return_contributions)
        result['front'] = result['Q']-result['tail']
        return result


class ResearchResponseS:
    """One fixed S rule; same complete input batch and seed replay the bank.

    Sampling follows original039 FIT/0 streams in blocks of 128. Parent IDs
    are labels only. Changing batch order/size can change assigned random draws.
    S was rejected by the native protection gates; this is a research API.
    """

    @torch.no_grad()
    def __init__(self, config, device):
        from n01.model038 import N01Response038
        from n01.observation008 import FrozenObservedNoise008
        from .smooth_mean import SOLVER_SETTINGS

        self.config = config
        self.device = torch.device(device)
        for name, digest in config['source_pins'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Finalization source changed: '+name)
        run_path = ROOT/config['S_run']['path']
        if hashlib.sha256(run_path.read_bytes()).hexdigest() != config['S_run']['sha256']:
            raise ValueError('Frozen S run config changed')
        self.p = json.loads(run_path.read_text(encoding='utf8'))
        p = self.p
        if (importlib.metadata.version('osqp') != p['S']['solver_version']
                or SOLVER_SETTINGS != p['S']['solver_settings']):
            raise ValueError('Frozen S solver version/settings changed')
        setup = json.loads((ROOT/p['setup038']['path']).read_text(encoding='utf8'))
        reference = load_reference028(verified_path028(setup['reference028']))
        self.accounting = Counter(checkpoint_loads=reference.accounting['completed']['CP_loads'])
        PCA = setup['PCA013']
        with np.load(verified_path028(PCA), allow_pickle=False) as archive:
            condition = dict(mean=archive[PCA['mean_key']].copy(),
                axis=archive[PCA['components_key']][:, 0].copy(), scale=PCA['PC1_scale'])
        self.accounting['PCA_archives_decoded'] += 1
        baseline = copy.deepcopy(reference.models[0].baseline).to(self.device)
        baseline.eval().requires_grad_(False)
        self.extension = FrozenObservedNoise008(copy.deepcopy(
            reference.extension.model005).to(self.device)).eval().requires_grad_(False)
        self.models = {}
        with torch.random.fork_rng(devices=[]):
            for index in (0, 1):
                old = reference.models[index]
                model = N01Response038(baseline, old.basis_head.state_dict(),
                    seed=setup['seeds'][index], condition_state=condition,
                    charge_initialization=dict(theta_a_initial=float(old.theta_a),
                                               theta_b_initial=float(old.theta_b)))
                model.mean_adapter[-1].bias.copy_(torch.as_tensor(p['base_bias'],
                    device=self.device, dtype=model.mean_adapter[-1].bias.dtype))
                model.set_phase('frozen').eval()
                self.models[index] = model
        # The scalar Q1 branch is never enabled for S; its values do not enter S.
        self.q_floor = reference.q_floor
        self.template = json.loads((ROOT/p['template_file']).read_text(encoding='utf8'))
        self.detector_config = hashlib.sha256(json.dumps(dict(
            legacy_detector=reference.detector_config, S_run=config['S_run']['sha256'],
            source_pins=config['source_pins'], sampling='039/FIT/0/chunk128',
            input_kind='clean_SOURCE_carrier384_float64_V'),
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @torch.no_grad()
    def sample(self, source_waveforms, parent_ids, n_draws, seed, model_seed=0,
               noise_hypothesis='fast', return_contributions=False):
        from n01.model008 import canonical_charge008
        from n01.model015 import _ExactZeroMeanAddition015
        from n01.observation008 import observe_generated_carrier
        from n01.stats018 import morphology018
        from run_response036 import electronic_noise
        from run_response038 import encode_all
        from run_response039 import seed039
        from run_response042 import local_restitution042
        from run_response052 import conditional_sides052
        from run_response069 import level_warp069
        from .smooth_mean import constraint_space, fit_mean
        from .linear_random_transport import raw_warp_delta, project_random

        if (not isinstance(source_waveforms, torch.Tensor) or source_waveforms.ndim != 2
                or source_waveforms.shape[1] != 384 or not len(source_waveforms)
                or source_waveforms.dtype != torch.float64
                or not bool(torch.isfinite(source_waveforms).all())):
            raise ValueError('S needs nonempty finite float64 clean SOURCE [P,384] in V')
        if (type(n_draws) is not int or n_draws < 1 or type(seed) is not int
                or not 0 <= seed < 2**63 or type(model_seed) is not int
                or model_seed not in (0, 1) or noise_hypothesis not in ('fast','slow','common_level')
                or type(return_contributions) is not bool or len(parent_ids) != len(source_waveforms)):
            raise ValueError('S draw count, seed, model index, hypothesis, IDs or diagnostic switch differs')
        self.accounting['sample_calls'] += 1
        p, model = self.p, self.models[model_seed]
        source = source_waveforms.to(self.device)
        encoding = encode_all(model, source, p['chunk_rows'], self.accounting)
        A0 = encoding['original']['signal_pre']
        q = canonical_charge008(A0)
        adapter = model._mean_response(encoding, enabled=True)
        delta, _, _ = local_restitution042(adapter['delta_mean_pre_V'], p['base_restitution_fraction'])
        transport, _, _, _, _, _ = conditional_sides052(A0+delta, q, 1.,
            p['input_Q_reference_Vns'], p['side_coefficients_ns'])
        fixed_delta = delta+transport
        fixed = A0+fixed_delta
        target, _, warp, queries, _, _ = level_warp069(fixed, torch.zeros_like(fixed), q, self.template)
        row_parent = torch.arange(len(source), device=self.device).repeat_interleave(n_draws)
        rows = len(row_parent)
        seeds = {key: seed039(seed, 'FIT', 0, key) for key in ('shape','noise005','extension')}
        generators = {key: torch.Generator().manual_seed(value) for key, value in seeds.items()}
        signals, electronic = [], []
        for start in range(0, rows, p['chunk_rows']):
            indices = row_parent[start:start+p['chunk_rows']]
            cache = model.select_encoding(encoding, indices)
            z = torch.randn(len(indices), 8, generator=generators['shape'], dtype=torch.float32).to(self.device)
            self.accounting['response_calls'] += 1
            self.accounting['response_rows'] += len(indices)
            signals.append(model(cache['source'], z, cache=cache,
                mean_enabled=False, charge_enabled=False)['signal_pre'])
            self.accounting['noise005_calls'] += 1
            electronic.append(electronic_noise(self.extension, len(indices), generators['noise005'],
                generators['extension'], self.device, self.accounting)[noise_hypothesis])
        M0, noise = torch.cat(signals), torch.cat(electronic)
        random = M0-A0[row_parent]
        means, targets = fixed.cpu().numpy(), target[:,110:176].cpu().numpy()
        random_np, query_np = random[:,110:176].cpu().numpy(), queries.cpu().numpy()
        scales = p['S']['normalization_by_head'][str(model_seed)]['values']
        amplitude = warp['amplitude_scale'].cpu().numpy()
        mean_delta = np.zeros((len(source), 66))
        random_delta = np.zeros((rows, 66))
        qp = []
        for parent in range(len(source)):
            space = constraint_space(means[parent], scales, p['S']['radius'],
                p['S']['lambda_s'], p['S']['lambda_r'])
            try:
                mean_delta[parent], _, info = fit_mean(means[parent], targets[parent], space)
            except RuntimeError as error:
                raise RuntimeError(f'S QP parent row {parent}, ID {parent_ids[parent]}: {error}') from error
            self.accounting['mean_parent_rows'] += 1
            self.accounting['QP_solves' if info['status']=='solved' else 'zero_DOF_rows'] += 1
            qp.append(info)
            for row in range(parent*n_draws, (parent+1)*n_draws):
                raw = raw_warp_delta(random_np[row], query_np[parent], amplitude[parent], space['allowed'])
                random_delta[row] = project_random(raw, space)
                self.accounting['random_projection_rows'] += 1
        correction = torch.zeros_like(fixed)
        correction[:,110:176] = torch.as_tensor(mean_delta, device=self.device)
        dr = torch.zeros_like(random)
        dr[:,110:176] = torch.as_tensor(random_delta, device=self.device)
        mean = fixed+correction
        mean_signal = _ExactZeroMeanAddition015.apply(M0, (fixed_delta+correction)[row_parent])
        signal = _ExactZeroMeanAddition015.apply(mean_signal, dr)
        result = observe_generated_carrier(signal+noise, ste=False, q_floor=self.q_floor,
            return_carrier=True, return_indices=True)
        self.accounting['native_observation_rows'] += rows
        morphology = morphology018(result['window_V'].cpu().numpy())
        result['readout_flags'] = {key:torch.as_tensor(value, device=self.device)
            for key, value in morphology.items() if value.dtype == bool}
        result['readout_features'] = {key:torch.as_tensor(morphology[key], device=self.device)
            for key in ('peak_voltage_V','rise10_90_ns','FWHM_ns')}
        top = result['carrier_adc'][:,112:144].topk(2, dim=1).values
        result['diagnostic_flags'] = dict(
            native_peak_differs_from_mean=result['peak_index'] != mean[row_parent,112:144].argmax(1)+112,
            top_two_ADC_samples_tied=top[:,0]==top[:,1],
            **{key:warp[key][row_parent] for key in ('source_clamped_low','source_clamped_high',
                                                  'target_clamped_low','target_clamped_high')})
        result.update(front=result['Q']-result['tail'], parent_ids=tuple(parent_ids),
            row_parent=row_parent, draw=torch.arange(n_draws, device=self.device).repeat(len(source)),
            model_seed=model_seed, noise_hypothesis=noise_hypothesis, detector_config=self.detector_config,
            streams=dict(sampling_seed=seed, CPU_seeds=seeds, purpose='FIT', bank_id=0,
                full_rows=rows, NN_chunk_rows=128, semantics='stateless complete-batch replay; IDs are labels'),
            applicability=dict(input='prepared clean SOURCE carrier384 V', software_sample_interval_ns=1,
                physical_calibration='unknown', scientific_status='response_match_not_established',
                candidate_status='S_closed_negative_native_protection_failure', model_accepted=False,
                M3_triggered=False, new_MC_confirmation_opened=False),
            QP_parent_status=qp)
        if return_contributions:
            result['contributions'] = dict(A0_pre_V=A0[row_parent], BASE054_mean_pre_V=fixed[row_parent],
                mean_pre_V=mean[row_parent], S_mean_correction_pre_V=correction[row_parent],
                M0_signal_pre_V=M0, original_random_pre_V=random, S_random_correction_pre_V=dr,
                mean_signal_pre_V=mean_signal, signal_pre_V=signal, electronics_pre_V=noise)
        return result


def load_response(config_path, device='cpu'):
    """Load a pinned research config; 480-point inputs require explicit preparation."""
    config = json.loads(Path(config_path).read_text(encoding='utf8'))
    if config['schema'] == 'n01/wp6-028-config/v1':
        return Legacy028Facade(load_reference028(config_path, device=device))
    if (config['schema'] != 'n01/finalization-research-api/v1'
            or config['method'] != 'S'
            or config['scientific_status'] != 'response_match_not_established'
            or config['candidate_status'] != 'closed_negative'):
        raise ValueError('Unknown finalization method, detector binding or acceptance status')
    return ResearchResponseS(config, device)
