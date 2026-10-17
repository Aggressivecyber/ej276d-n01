"""Original M0 plus a conditional, canonical-Q-preserving time adapter.

A0, its input encoding, and M0's voltage scale stay fixed. Electronics and
observation belong to the caller. Charge uses the existing independent scalar
016 latent; it never multiplies the M0 voltage fluctuation. There is no local
signal branch in the original M0, so none is introduced here.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn

from .model import InvalidWaveformError
from .model008 import canonical_charge008
from .model010 import tensor_state_digest010
from .model014 import FrozenDeterministicVoltageShape014
from .model015 import _ExactZeroMeanAddition015
from .model016 import Q_REFERENCE016, charge_coefficients016


TIME_KNOTS038 = (100., 106., 112., 119., 126., 142., 158., 190., 254., 318., 383.)
MINIMUM_Q_RATIO038 = .25


class MeanProposalOutsideDomain038(InvalidWaveformError):
    """A logged proposal rejection, never a clipped waveform or dropped row."""


class N01Response038(FrozenDeterministicVoltageShape014):
    """Preserve the complete014 M0 function, then adapt only its mean.

    Construct after moving the frozen original A0 to its final device:014's
    baseline layout locks tensor locations. ``condition_state`` contains the
    existing TRAIN Q-normalized384 PCA ``mean``, PC1 ``axis``, and PC1 ``scale``.
    No fit, random draw, checkpoint, data read or observation occurs here.
    """

    noise_shapes = {'shape': (8,), 'charge': ()}

    def __init__(self, baseline, initial_head: Mapping, *, seed: int,
                 condition_state: Mapping, charge_initialization: Mapping):
        super().__init__(baseline, seed=seed)
        self.basis_head.load_state_dict(initial_head, strict=True)
        self.reference_M0_head_sha256 = tensor_state_digest010(self.basis_head)
        ref = baseline.mean_head.weight
        device = ref.device
        mean = torch.as_tensor(condition_state['mean'], device=device, dtype=torch.float64)
        axis = torch.as_tensor(condition_state['axis'], device=device, dtype=torch.float64)
        scale = torch.as_tensor(condition_state['scale'], device=device, dtype=torch.float64)
        if (mean.shape != (384,) or axis.shape != (384,) or scale.ndim != 0
                or not bool(torch.isfinite(mean).all() and torch.isfinite(axis).all()
                            and torch.isfinite(scale) and scale > 0)):
            raise ValueError('038 requires finite fixed TRAIN PCA mean384/PC1-axis384/positive scalar scale')
        self.register_buffer('condition_mean', mean.detach().clone())
        self.register_buffer('condition_axis', axis.detach().clone())
        self.register_buffer('condition_scale', scale.detach().clone())
        self.register_buffer('time_knots', torch.tensor(TIME_KNOTS038, device=device, dtype=torch.float64))
        self.register_buffer('charge_q_reference', torch.tensor(Q_REFERENCE016, device=device, dtype=torch.float64))
        self.frozen_condition_sha256 = tensor_state_digest010(self._condition_state())
        with torch.random.fork_rng(devices=[]):
            adapter = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 8))
            generator = torch.Generator(device='cpu').manual_seed(seed)
            nn.init.kaiming_uniform_(adapter[0].weight, a=math.sqrt(5.), generator=generator)
            nn.init.uniform_(adapter[0].bias, -1/math.sqrt(2.), 1/math.sqrt(2.), generator=generator)
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
        self.mean_adapter = adapter.to(device=device, dtype=ref.dtype)
        self.theta_a = nn.Parameter(torch.tensor(charge_initialization['theta_a_initial'], device=device, dtype=torch.float64))
        self.theta_b = nn.Parameter(torch.tensor(charge_initialization['theta_b_initial'], device=device, dtype=torch.float64))
        charge_coefficients016(self.theta_a, self.theta_b, self.charge_q_reference)
        self._active_encoding = None
        self.set_phase('mean')
        self.assert_frozen_unchanged(deep=True)

    def _condition_state(self):
        return {name: getattr(self, name) for name in
                ('condition_mean', 'condition_axis', 'condition_scale', 'time_knots', 'charge_q_reference')}

    def assert_frozen_unchanged(self, *, deep=False):
        result = super().assert_frozen_unchanged(deep=deep)
        if deep and hasattr(self, 'frozen_condition_sha256'):
            if tensor_state_digest010(self._condition_state()) != self.frozen_condition_sha256:
                raise RuntimeError('038 fixed PCA/time/charge reference buffers changed')
        return result

    def parameter_groups(self):
        return dict(mean=tuple(self.mean_adapter.parameters()), shape=tuple(self.basis_head.parameters()),
                    charge=(self.theta_a, self.theta_b), local=())

    def set_phase(self, phase: str, *, train_charge=False):
        if phase not in ('mean', 'random', 'frozen') or type(train_charge) is not bool:
            raise ValueError('038 phase is mean/random/frozen with explicit boolean train_charge')
        if train_charge and phase != 'random':
            raise ValueError('038 charge parameters can open only in the random phase')
        self.phase, self.train_charge = phase, train_charge
        for name, parameters in self.parameter_groups().items():
            active = (phase == 'mean' and name == 'mean') or (phase == 'random' and
                     (name == 'shape' or (name == 'charge' and train_charge)))
            for parameter in parameters:
                parameter.requires_grad_(active)
                parameter.grad = None
        return self.train(self.training)

    def train(self, mode=True):
        if type(mode) is not bool:
            raise TypeError('038 train mode must be boolean')
        self.assert_frozen_unchanged()
        self.training = mode
        self.basis_head.train(mode and self.phase == 'random')
        self.mean_adapter.train(mode and self.phase == 'mean')
        return self

    def requires_grad_(self, requires_grad=True):
        if type(requires_grad) is not bool:
            raise TypeError('038 requires_grad flag must be boolean')
        for name, parameters in self.parameter_groups().items():
            active = requires_grad and ((self.phase == 'mean' and name == 'mean') or
                     (self.phase == 'random' and (name == 'shape' or (name == 'charge' and self.train_charge))))
            for parameter in parameters:
                parameter.requires_grad_(active)
        return self

    def trainable_parameter_names(self):
        return tuple(name for name, value in self.named_parameters() if value.requires_grad)

    def trainable_parameters(self):
        return tuple(value for value in self.parameters() if value.requires_grad)

    @torch.no_grad()
    def encode_source(self, source: Tensor):
        original, decoded = super()._baseline_once(source)
        x = source[:, 0] if source.ndim == 3 else source
        centered = x.double()-x.double()[:, :100].mean(1, keepdim=True)
        q = original['q_source']
        pc1 = ((centered/q[:, None]-self.condition_mean)*self.condition_axis).sum(1)/self.condition_scale
        condition = torch.stack(((q/self.baseline.q_ref).log(), pc1), 1).to(decoded)
        return dict(source=source.detach(), original=original, decoded=decoded.detach(), condition=condition.detach())

    @staticmethod
    def select_encoding(encoding: Mapping, indices: Tensor):
        """Retrieve frozen rows; the row indices never enter a neural module."""
        rows = len(encoding['source'])
        original = {name: value[indices] if isinstance(value, Tensor) and value.ndim and value.shape[0] == rows else value
                    for name, value in encoding['original'].items()}
        return dict(source=encoding['source'][indices], original=original,
                    decoded=encoding['decoded'][indices], condition=encoding['condition'][indices])

    def _baseline_once(self, source):
        # super014 still executes every original M0 head/voltage operation.
        encoding = self._active_encoding
        return encoding['original'], encoding['decoded']

    def _mean_response(self, encoding, *, enabled):
        A0 = encoding['original']['signal_pre']
        t = torch.arange(384, device=A0.device, dtype=torch.float64)
        free = self.mean_adapter(encoding['condition']).double()
        if not enabled:
            free = free*0.
        zero = free[:, :1]*0.
        logits = torch.cat((free[:, :3], zero, free[:, 3:], zero), 1)
        lengths = self.time_knots.diff()
        knots = [self.time_knots[:1].expand(len(A0), 1)]
        increments = []
        for start, stop in ((0, 4), (4, 10)):
            raw = lengths[start:stop]*torch.exp(logits[:, start:stop])
            increment = raw*((self.time_knots[stop]-self.time_knots[start])/raw.sum(1, keepdim=True))
            positions = self.time_knots[start]+increment.cumsum(1)[:, :-1]
            fixed_stop = self.time_knots[stop:stop+1].expand(len(A0), 1)
            knots.extend((positions, fixed_stop))
            increments.append(increment)
        mapped = torch.cat(knots, 1)
        segment = torch.searchsorted(self.time_knots, t, right=True).sub(1).clamp(0, 9)
        fraction = (t-self.time_knots[segment])/lengths[segment]
        displacement = mapped-self.time_knots
        queries = t+displacement[:, segment]+fraction*(displacement[:, segment+1]-displacement[:, segment])
        queries = torch.cat((t[:100].expand(len(A0), -1), queries[:, 100:]), 1)
        invalid = ~torch.isfinite(queries).all(1) | (queries.diff(dim=1) <= 0).any(1)
        if bool(invalid.any()):
            raise MeanProposalOutsideDomain038('038 time queries not finite strictly monotone', invalid.nonzero().flatten())
        left = queries.floor().long().clamp(0, 383)
        weight = queries-left
        warped = (1-weight)*A0.gather(1, left)+weight*A0.gather(1, (left+1).clamp_max(383))
        warped = torch.cat((A0[:, :100], warped[:, 100:]), 1)
        Q0 = canonical_charge008(A0)
        ratio = canonical_charge008(warped)/Q0
        invalid = ~torch.isfinite(ratio) | (ratio < MINIMUM_Q_RATIO038)
        if bool(invalid.any()):
            raise MeanProposalOutsideDomain038('038 Qwarp/Q_A0 below .25 or nonfinite; proposal rejected',
                                              invalid.nonzero().flatten(), ratio[invalid])
        gain = ratio.reciprocal()
        mean = torch.cat((A0[:, :100], (warped*gain[:, None])[:, 100:]), 1)
        delta = mean-A0
        regular_per_row = delta.square().mean(1)/A0.square().mean(1)
        return dict(mean_signal_pre_V=mean, delta_mean_pre_V=delta, time_free_log_increments=free,
                    time_queries=queries, time_interval_increments=torch.cat(increments, 1),
                    warp_Q_ratio=ratio, normalization_gain=gain, regular_mean=regular_per_row.mean(),
                    regular_mean_per_row=regular_per_row)

    def random_forward(self, source: Tensor, z_shape: Tensor, z_charge: Tensor | None = None, *,
                       cache: Mapping | None = None, shape_enabled=True, mean_enabled=True, charge_enabled=False):
        if any(type(value) is not bool for value in (shape_enabled, mean_enabled, charge_enabled)):
            raise TypeError('038 module switches must be boolean')
        if charge_enabled and self.phase == 'mean':
            raise ValueError('038 mean phase keeps charge disabled')
        encoding = self.encode_source(source) if cache is None else cache
        if source.data_ptr() != encoding['source'].data_ptr() or source.shape != encoding['source'].shape:
            raise ValueError('038 cache/source row binding differs')
        self._active_encoding = encoding
        try:
            out = super().random_forward(source, z_shape, enabled=shape_enabled)
        finally:
            self._active_encoding = None
        adapter = self._mean_response(encoding, enabled=mean_enabled)
        A0, mean = out['base_signal_pre_V'], adapter['mean_signal_pre_V']
        M0 = out['signal_pre']
        signal = _ExactZeroMeanAddition015.apply(M0, adapter['delta_mean_pre_V'])
        Q0 = canonical_charge008(A0)
        cv2 = torch.zeros_like(Q0)
        logvar, alpha = cv2, torch.ones_like(Q0)
        increment = torch.zeros_like(mean)
        if charge_enabled:
            if (not isinstance(z_charge, Tensor) or z_charge.shape != Q0.shape or z_charge.device != Q0.device
                    or not z_charge.is_floating_point() or not bool(torch.isfinite(z_charge).all())):
                raise ValueError('038 active charge needs an explicit finite same-device [B] latent')
            a, b = charge_coefficients016(self.theta_a, self.theta_b, self.charge_q_reference)
            cv2 = a/Q0+b
            logvar = torch.log1p(cv2)
            log_alpha = logvar.sqrt()*z_charge.double()-.5*logvar
            alpha = log_alpha.exp()
            increment = mean*torch.expm1(log_alpha)[:, None]
            signal = signal+increment
        if not bool(torch.isfinite(signal).all()):
            raise InvalidWaveformError('038 nonfinite signal; no rows clipped/dropped',
                                       (~torch.isfinite(signal).all(1)).nonzero().flatten())
        regular_shape = out['regular']
        random_shape = out['random_shape_pre_V']
        shape_per_row = (random_shape.square().mean(1)/self.baseline.a_ref.square()
                         +random_shape.diff(n=2, dim=1).square().mean(1)/self.baseline.a_ref.square())
        regular_charge = cv2.mean()
        out.update(adapter)
        out.update(signal_pre=signal, pure_signal_pre_V=signal, A0_signal_pre_V=A0,
                   M0_reference_signal_pre_V=M0, B=out['voltage_basis_V'], condition=encoding['condition'],
                   alpha=alpha, charge_cv2=cv2, charge_log_variance=logvar,
                   charge_increment_pre_V=increment, charge_Q_variance=Q0.square()*cv2,
                   moment_mean_pre_V=mean, moment_basis_V=out['voltage_basis_V'] if shape_enabled else out['voltage_basis_V']*0.,
                   moment_charge_cv2=cv2, moment_local_covariance_zero=True,
                   regular_shape=regular_shape, regular_charge=regular_charge, regular_local=regular_charge*0.,
                   regular_shape_per_row=shape_per_row, regular_charge_per_row=cv2,
                   regular_per_row=shape_per_row+adapter['regular_mean_per_row']+cv2,
                   regular=regular_shape+adapter['regular_mean']+regular_charge,
                   qbar=Q0, qsig=Q0*alpha, canonical_charge=canonical_charge008(signal),
                   canonical_charge_identity_residual=canonical_charge008(signal)-Q0*alpha,
                   canonical_mean_charge_residual=canonical_charge008(mean)-Q0,
                   mean_enabled=mean_enabled, charge_enabled=charge_enabled,
                   analytic_moment_scope='preADC independent charge/shape; covariance=BB^T+cv2*mean*mean^T; no029 shared-latent cross term',
                   phase038=self.phase, baseline_signal_forward_calls=int(cache is None))
        return out

    forward = random_forward
    signal_forward = random_forward


def _artificial_semantics038(output_path):
    """Bounded CPU engineering only; manufactured inputs, no CP/data/ADC."""
    from pathlib import Path
    import hashlib
    import json
    import numpy as np
    from .model008 import N01Response008

    torch.set_num_threads(2)
    t = torch.arange(384, dtype=torch.float64)
    amplitudes = torch.linspace(.04, .11, 8, dtype=torch.float64)[:, None]
    widths = torch.linspace(3., 6., 8, dtype=torch.float64)[:, None]
    tails = torch.linspace(30., 65., 8, dtype=torch.float64)[:, None]
    source = amplitudes*torch.where(t < 126, torch.exp(-.5*((t-126)/widths).square()), torch.exp(-(t-126)/tails))
    source[:, :100] = 0
    q = canonical_charge008(source)
    stats = dict(q_ref=float(q.median()), a_ref=float(amplitudes.median()),
                 log_q_min=float(q.min().log())-.2, log_q_max=float(q.max().log())+.2)
    with torch.random.fork_rng(devices=[]):
        baseline = N01Response008(stats, 'deterministic')
    baseline.set_stage('low_learning_rate'); baseline.eval(); baseline.requires_grad_(False)
    seed = 2026101738
    m0 = FrozenDeterministicVoltageShape014(baseline, seed=seed)
    model = N01Response038(baseline, m0.basis_head.state_dict(), seed=seed,
                          condition_state=dict(mean=torch.zeros(384), axis=torch.linspace(-1., 1., 384), scale=.1),
                          charge_initialization=dict(theta_a_initial=-7., theta_b_initial=-6.))
    z = torch.randn(8, 8, generator=torch.Generator().manual_seed(3801))
    zq = torch.randn(8, generator=torch.Generator().manual_seed(3802), dtype=torch.float64)
    counts = dict(A0_forward_calls=0, A0_forward_rows=0, combined_AD_queries=0, response_head_forward_calls=0)

    def count_head(module, arguments, result):
        counts['response_head_forward_calls'] += 1

    head_hooks = [head.register_forward_hook(count_head) for head in (m0.basis_head, model.basis_head)]
    counts['A0_forward_calls'] += 1; counts['A0_forward_rows'] += len(source)
    reference = m0(source, z)
    counts['A0_forward_calls'] += 1; counts['A0_forward_rows'] += len(source)
    encoding = model.encode_source(source)
    initial = model(source, z, cache=encoding)
    counts['A0_forward_calls'] += 1; counts['A0_forward_rows'] += len(source)
    uncached = model(source, z)
    assert torch.equal(reference['signal_pre'].view(torch.uint8), initial['signal_pre'].view(torch.uint8))
    assert torch.equal(initial['signal_pre'].view(torch.uint8), uncached['signal_pre'].view(torch.uint8))
    assert torch.equal(initial['time_queries'], t.expand(8, -1))
    assert bool((initial['delta_mean_pre_V'] == 0).all())
    probe = torch.sin(t/17.)[None]+torch.cos(t/31.)[None]
    (initial['signal_pre']*probe).sum().backward(); counts['combined_AD_queries'] += 1
    adapter_gradient = float(torch.linalg.vector_norm(model.mean_adapter[-1].bias.grad.double()))
    assert adapter_gradient > 0
    assert all(value.grad is None for value in baseline.parameters())
    with torch.no_grad():
        model.mean_adapter[-1].bias.copy_(torch.linspace(-.03, .04, 8))
    adapted = model(source, z, cache=encoding)
    assert torch.equal(adapted['B'], initial['B'])
    assert bool((adapted['time_queries'].diff(dim=1) > 0).all())
    assert torch.equal(adapted['time_queries'][:, [100, 126, 383]], t[[100, 126, 383]].expand(8, -1))
    assert bool((adapted['signal_pre'][:, :100] == 0).all())
    assert float(adapted['canonical_mean_charge_residual'].detach().abs().max()) < 1e-12
    selected = model.select_encoding(encoding, torch.tensor([1, 5]))
    subset = model(selected['source'], z[[1, 5]], cache=selected)
    assert torch.equal(subset['signal_pre'], adapted['signal_pre'][[1, 5]])
    model.set_phase('random', train_charge=True)
    charged = model(source, z, zq, cache=encoding, charge_enabled=True)
    (charged['signal_pre']*probe).sum().backward(); counts['combined_AD_queries'] += 1
    assert all(value.grad is None for value in model.mean_adapter.parameters())
    assert all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for value in model.basis_head.parameters())
    assert model.theta_a.grad is not None and model.theta_b.grad is not None
    replay = model(source, z, zq, cache=encoding, charge_enabled=True)
    assert torch.equal(charged['signal_pre'], replay['signal_pre'])
    off = model(source, z, cache=encoding, shape_enabled=False, mean_enabled=False, charge_enabled=False)
    assert torch.equal(off['signal_pre'].view(torch.uint8), initial['A0_signal_pre_V'].view(torch.uint8))
    assert bool((off['moment_basis_V'] == 0).all())
    # Gaussian quadrature over compact scalar factors, no neural re-evaluation.
    nodes, weights = np.polynomial.hermite.hermgauss(16)
    nodes = torch.from_numpy(nodes*math.sqrt(2.))
    weights = torch.from_numpy(weights/math.sqrt(math.pi))
    mean_score = (charged['mean_signal_pre_V'][0].detach()*probe[0]).sum()
    basis_scores = charged['B'][0].detach()@probe[0]
    shape_sd = basis_scores.square().sum().sqrt()
    s2 = charged['charge_log_variance'][0].detach()
    alpha = torch.exp(s2.sqrt()*nodes-.5*s2)
    scores = alpha[:, None]*mean_score+torch.tensor([-1., 1.], dtype=torch.float64)*shape_sd
    score_mean = (scores.mean(1)*weights).sum()
    score_variance = ((scores-mean_score).square().mean(1)*weights).sum()
    expected_variance = charged['charge_cv2'][0].detach()*mean_score.square()+shape_sd.square()
    assert torch.allclose(score_mean, mean_score, atol=1e-12, rtol=1e-12)
    assert torch.allclose(score_variance, expected_variance, atol=1e-12, rtol=1e-12)
    for hook in head_hooks:
        hook.remove()
    model.set_phase('frozen'); model.assert_frozen_unchanged(deep=True)
    assert not model.trainable_parameters() and model._active_encoding is None
    assert counts['A0_forward_calls'] <= 4 and counts['A0_forward_rows'] <= 32 and counts['combined_AD_queries'] <= 4
    receipt = dict(schema='n01/response038-artificial-semantics/v1', status='passed', artificial_only=True,
        actual_counts=counts, real_CP_loads=0, real_arrays=0, observations=0, optimizer_steps=0,
        manufactured_parents=8, fixed_knots=list(TIME_KNOTS038), stats=stats,
        model_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parameter_counts={key: sum(value.numel() for value in values) for key, values in model.parameter_groups().items()},
        identity_pure_M0_byte_equal=True, cached_uncached_byte_equal=True, subset_cache_replay=True,
        identity_last_bias_gradient_norm=adapter_gradient, fixed_B_after_mean_change=True,
        exact_anchors=True, strictly_monotone_queries=True, prefix100_zero=True,
        canonical_mean_charge_absmax=float(adapted['canonical_mean_charge_residual'].detach().abs().max()),
        canonical_full_charge_absmax=float(charged['canonical_charge_identity_residual'].detach().abs().max()),
        normalization_gain_range=[float(adapted['normalization_gain'].min()), float(adapted['normalization_gain'].max())],
        random_phase_mean_gradients_none=True, shape_charge_gradients_present=True, seeded_latent_replay=True,
        all_modules_off_A0_byte_equal=True, projected_mean_abs_error=float((score_mean-mean_score).abs()),
        projected_variance_abs_error=float((score_variance-expected_variance).abs()),
        moment_scope='independent scalar charge and original M0 Gaussian8, preADC only; no shared-latent029 cross term',
        scientific_acceptance=False)
    Path(output_path).write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf8', newline='\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Artificial038 semantics only; no checkpoint/data/observation')
    parser.add_argument('--artificial-semantic-receipt', required=True)
    _artificial_semantics038(parser.parse_args().artificial_semantic_receipt)
