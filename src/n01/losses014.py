"""C0 is the native 010 objective; C1 substitutes one denominator-free target.

The .components interface stays the native 008 hard-evaluation objective.
Only .training_components selects the predeclared C0/C1 representation.
"""
from __future__ import annotations

import copy
import hashlib

import torch

from .losses008 import ADC_STEP, _directions, _waveform
from .losses008_v2 import N01Loss008, WAVE_WEIGHTS, empirical_wasserstein_1d, _require_context


SEGMENTS014 = ((0, 16), (16, 46), (46, 96), (96, 256))
FEATURE_NAMES014 = ('Q', 'T', 'segment_0_16', 'segment_16_46', 'segment_46_96', 'segment_96_256')


def integral_features014(wave):
    y = _waveform(wave).double()
    if len(y) == 0 or not bool(torch.isfinite(y).all()):
        raise ValueError('014 finite nonempty hard256 required; no row deletion')
    return torch.stack([y.sum(1), y[:, 46:].sum(1)]+[y[:, a:b].sum(1) for a, b in SEGMENTS014], 1)


@torch.no_grad()
def fit_integral_statistics014(real_train):
    """One fixed REAL TRAIN fit, before candidate outcomes; all signed rows stay."""
    raw = integral_features014(real_train).detach().cpu()
    if len(raw) < 2:
        raise ValueError('014 requires at least two REAL TRAIN rows')
    lengths = torch.tensor([256, 210]+[b-a for a, b in SEGMENTS014], dtype=torch.float64)
    floor = ADC_STEP*lengths.sqrt()
    scale = torch.maximum(raw.abs().quantile(.5, dim=0), floor)
    return dict(schema='n01/wp2-integral-statistics/v1', fit_split='TRAIN', fit_rows=len(raw),
        feature_names=list(FEATURE_NAMES014), segments_half_open=[list(v) for v in SEGMENTS014],
        scale_Vns=scale.tolist(), ADC_quadrature_floor_Vns=floor.tolist(),
        statistic='max(REAL_TRAIN median(abs(raw integral)),ADC_LSB*sqrt(segment_length))',
        transform='asinh(raw_integral/fixed_scale)', centering=False,
        nonpositive_Q_rows=int((raw[:, 0] <= 0).sum()), all_rows_retained=True,
        candidate_outputs_used=False, metadata_not_network_input=True)


def integral_directions014(context, seed):
    context.validate()
    namespace = f'N01-WP2-integral014|{context.kind}|{int(seed)}|{context.update}'.encode('ascii')
    mixed = int.from_bytes(hashlib.sha256(namespace).digest()[:8], 'little') & ((1 << 63)-1)
    return _directions(6, 16, torch.Generator(device='cpu').manual_seed(mixed))


class N01Loss014(N01Loss008):
    """Native hard metrics remain available independently of training target."""

    def __init__(self, stats, integral_statistics, *, seed, evaluation_seed, integral_seed):
        super().__init__(stats, seed=seed, evaluation_seed=evaluation_seed)
        p = integral_statistics
        if (p.get('schema') != 'n01/wp2-integral-statistics/v1' or p.get('fit_split') != 'TRAIN'
            or p.get('feature_names') != list(FEATURE_NAMES014)
            or p.get('segments_half_open') != [list(v) for v in SEGMENTS014]
            or p.get('all_rows_retained') is not True or p.get('candidate_outputs_used') is not False):
            raise ValueError('014 integral TRAIN scale contract changed')
        scale = torch.as_tensor(p['scale_Vns'], dtype=torch.float64)
        if scale.shape != (6,) or not bool((torch.isfinite(scale) & (scale > 0)).all()):
            raise ValueError('014 requires six finite positive frozen integral scales')
        self.register_buffer('integral_scale', scale.clone())
        self.integral_statistics_metadata = copy.deepcopy(p)
        if type(integral_seed) is not int or integral_seed in (seed, evaluation_seed):
            raise ValueError('014 requires an independent explicit integral projection seed')
        self.integral_projection_seed = integral_seed
        self.last_integral_diagnostics = {}

    def integral_loss(self, fake, real, weights, context):
        _require_context(context, self.training_projection_seed, self.evaluation_projection_seed)
        fraw, rraw = integral_features014(fake), integral_features014(real)
        f = torch.asinh(fraw/self.integral_scale.to(fraw))
        r = torch.asinh(rraw/self.integral_scale.to(rraw))
        directions = integral_directions014(context, self.integral_projection_seed).to(f)
        coordinate = empirical_wasserstein_1d(f, r, weights)
        joint = empirical_wasserstein_1d(f@directions, r@directions, weights)
        self.last_integral_diagnostics = dict(feature_names=list(FEATURE_NAMES014),
            fake_global_rows=len(fake), real_global_rows=len(real), all_rows_retained=True,
            fake_nonpositive_Q_rows=int((fraw[:, 0] <= 0).sum()),
            real_nonpositive_Q_rows=int((rraw[:, 0] <= 0).sum()),
            fake_guarded_PSD_rows=int((fraw[:, 0].abs() < self.q_floor).sum()),
            real_guarded_PSD_rows=int((rraw[:, 0].abs() < self.q_floor).sum()),
            QPSD_denominator_in_training_target=False, output_Q_conditioning_in_training_target=False,
            coordinate_W1=float(coordinate.detach()), joint_W1=float(joint.detach()),
            projection_context=context.metadata(), direction_namespace='N01-WP2-integral014',
            integral_projection_seed=self.integral_projection_seed)
        return .5*(coordinate+joint)

    def training_components(self, outputs, real, *, candidate, projection_context, real_baseline100):
        if candidate not in ('C0', 'C1'):
            raise ValueError('014 only builds C0 and C1; C2 requires separate authorization')
        context = _require_context(projection_context, self.training_projection_seed, self.evaluation_projection_seed)
        if candidate == 'C0':
            result = super().components(outputs, real, stage='joint_stochastic',
                        projection_context=context, real_baseline100=real_baseline100)
            return dict(result, distribution=result['QPSD'], QPSD_native=result['QPSD'],
                        training_representation='native010_QPSD', diagnostics=copy.deepcopy(self.last_diagnostics))
        fake, real = _waveform(outputs['waveform']), _waveform(real)
        parts = self.wave_components(fake, real, None, context, mean_response=False)
        self.last_wave_components = {k: v.detach() for k, v in parts.items()}
        wave = sum(WAVE_WEIGHTS[k]*v for k, v in parts.items())
        integral = self.integral_loss(fake, real, None, context)
        # Native ratio distance is reported in original units and never enters C1 AD.
        with torch.no_grad():
            native = self._qpsd_with_context(fake.detach(), real.detach(), None, context, False)
            native_diagnostics = copy.deepcopy(self.last_diagnostics)
        zero = fake.sum()*0
        regular, trend = outputs.get('regular', zero), outputs.get('trend', zero)
        return dict(wave=wave, distribution=integral, QPSD=integral, QPSD_native=native,
                    noise=self.noise_loss(outputs['baseline100'], real_baseline100, None, projection_context=context),
                    regular=regular.reshape(()), trend=trend.reshape(()),
                    training_representation='signed_asinh_joint_integrals_Q_T_segments',
                    diagnostics=dict(integrals=copy.deepcopy(self.last_integral_diagnostics), native_QPSD=native_diagnostics))
