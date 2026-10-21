"""Synthetic semantics only; no project data or model checkpoints."""
import json
from pathlib import Path
import argparse
import numpy as np
import torch

from n01.global_memory01 import (GlobalDistributionMemory01, MeanHeadRepair01,
                                 GlobalRepairResponse01, add_peak_selection_gradient,
                                 adc_margin_risk, joint_secant_weights)
from n01.losses008 import ADC_STEP
from n01.model import ADC_MIN, ADC_MAX
from n01.joint01_model import JointResponse01, response_cache
from n01.model008 import N01Response008
from n01.observation008 import observe_generated_carrier

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
torch.manual_seed(172)
t = torch.arange(384, dtype=torch.float64)
q = torch.linspace(.01, .13, 96, dtype=torch.float64)
carrier = q[:, None] * torch.exp(-.5 * ((t - 126) / 12).square())[None]
carrier[:, :100] = 0
electronics = .0002 * torch.randn(96, 384, dtype=torch.float64)
real = observe_generated_carrier(carrier + electronics, q_floor=.001, return_carrier=True)
memory = GlobalDistributionMemory01(real, q_floor=.001, levels=(4, 8, 16), thresholds=32)
za, zb = torch.randn(96, 1, dtype=torch.float64), torch.randn(96, 1, dtype=torch.float64)


def feature(theta, section, z):
    value = carrier[section] * (theta[0].exp() + theta[1] * z[section]) + electronics[section]
    return memory.sums(observe_generated_carrier(value, ste=True, q_floor=.001, return_carrier=True))


a = torch.tensor([.02, .01], dtype=torch.float64, requires_grad=True)
ma, mb = [feature(a, slice(None), z) / 96 for z in (za, zb)]
sum(memory.losses(ma, mb).values()).backward()
direct = a.grad.clone()
b = a.detach().clone().requires_grad_()
with torch.no_grad():
    ca, cb = memory.coefficients(*[feature(b, slice(None), z) / 96 for z in (za, zb)])
for coeff, z in ((ca, za), (cb, zb)):
    for start in range(0, 96, 19):
        (coeff * feature(b, slice(start, start + 19), z) / 96).sum().backward()
gradient_error = float((direct - b.grad).abs().max())
assert gradient_error < 2e-7
wave_parameter = a.detach().clone().requires_grad_()
wa, wb = [feature(wave_parameter, slice(None), z) / 96 for z in (za, zb)]
sel = memory.sections['native_joint']
ws = memory.joint.block_slices['wave']
ra, rb = [memory.joint.cholesky @ (m - memory.target)[sel] for m in (wa, wb)]
variance = memory.joint.cholesky.square().sum(1)
wave_loss = (ra[ws] * rb[ws] / variance[ws]).mean()
direct_wave = torch.autograd.grad(wave_loss, wave_parameter, retain_graph=True)[0]
wc_a, wc_b = memory.wave_coefficients(wa, wb)
coefficient_wave = torch.autograd.grad((wc_a * wa + wc_b * wb).sum(), wave_parameter, retain_graph=True)[0]
wave_gradient_error = float((direct_wave - coefficient_wave).abs().max())
assert wave_gradient_error < 1e-10
positive_wave_loss = (((ra[ws] + rb[ws]) / 2).square() / variance[ws]).mean()
direct_positive_wave = torch.autograd.grad(positive_wave_loss, wave_parameter, retain_graph=True)[0]
pc_a, pc_b = memory.positive_wave_coefficients(wa, wb)
coefficient_positive_wave = torch.autograd.grad((pc_a * wa + pc_b * wb).sum(), wave_parameter)[0]
positive_wave_gradient_error = float((direct_positive_wave - coefficient_positive_wave).abs().max())
assert positive_wave_gradient_error < 1e-10
old_memory = memory
memory = GlobalDistributionMemory01(real, q_floor=.001, levels=(4, 8, 16), thresholds=32,
                                    conditional_wave_min_count=8)
prefix = len(old_memory.target)
assert torch.equal(memory.target[:prefix], old_memory.target)
assert torch.equal(memory.weights[:prefix], old_memory.weights)
snapshot = memory.target.clone()
parameter = a.detach().clone().requires_grad_()
ma, mb = [feature(parameter, slice(None), z) / 96 for z in (za, zb)]
direct_local = torch.autograd.grad(sum(memory.losses(ma, mb).values()), parameter, retain_graph=True)[0]
coefficient_local = memory.conditional_wave_coefficients(ma, mb)
selection = memory.sections['Q_wave']
positive_local = (memory.weights[selection] * (((ma + mb) / 2 - memory.target)[selection]).square()).sum()
direct_positive_local = torch.autograd.grad(positive_local, parameter, retain_graph=True)[0]
from_coefficients = torch.autograd.grad((coefficient_local[0] * ma + coefficient_local[1] * mb).sum(), parameter)[0]
conditional_coefficient_error = float((direct_positive_local - from_coefficients).abs().max())
assert conditional_coefficient_error < 1e-10
stream = parameter.detach().clone().requires_grad_()
ca, cb = memory.coefficients(ma.detach(), mb.detach())
for coeff, z in ((ca, za), (cb, zb)):
    for start in range(0, 96, 19):
        (coeff * feature(stream, slice(start, start + 19), z) / 96).sum().backward()
conditional_stream_error = float((direct_local - stream.grad).abs().max())
assert conditional_stream_error < 2e-7
# Reassign Q while preserving every waveform. Global wave features stay the
# same; joint Q-wave memory must detect the changed association.
swapped = dict(real, Q=real['Q'].flip(0))
swapped_mean = memory.mean(swapped)
raw_delta = memory.joint.cholesky @ (swapped_mean - memory.target)[memory.sections['native_joint']]
assert float(raw_delta[:56].abs().max()) < 1e-12
assert float((swapped_mean - memory.target)[selection].abs().max()) > 1e-3
assert torch.equal(memory.target, snapshot)
assert bool(memory.wave_supported.any())
assert bool(torch.isfinite(memory.conditional_wave_risks(ma.detach(), mb.detach())).all())
memory = old_memory
low = observe_generated_carrier(torch.zeros(4, 384, dtype=torch.float64),
                                q_floor=.001, return_carrier=True)
assert bool(torch.isfinite(memory.sums(low)).all())
assert float(memory.sums(low)[memory.sections['undefined_mass']][1]) == 4
probe = (carrier + electronics).detach().requires_grad_()
hard = observe_generated_carrier(probe, ste=True, q_floor=.001, return_carrier=True)
native_values = {key: hard[key].detach().clone() for key in ('window_V', 'Q', 'tail', 'PSD', 'PSD_raw')}
native_gradient = torch.autograd.grad(hard['tail'].sum(), probe, retain_graph=True)[0]
selection = add_peak_selection_gradient(hard)
assert all(torch.equal(selection[key], value) for key, value in native_values.items())
selection_gradient = torch.autograd.grad(selection['tail'].sum(), probe)[0]
assert torch.isfinite(selection_gradient).all()
assert (selection_gradient - native_gradient).abs().max() > 0

# Neither endpoint obeys the Q/wave/rail constraints, but their mixture can.
secant_changes = np.array([[-.10, .20], [.02, -.08], [-.10, .01], [.01, -.10]])
allowances = np.array([.005, 0., 0., 0.])
objective = np.array([-.20, -.01])
mix = joint_secant_weights(objective, secant_changes, allowances)
assert (mix > 0).all() and mix.sum() <= 1. + 1e-10
assert (secant_changes @ mix <= allowances + 1e-10).all()
assert objective @ mix < 0.
assert (secant_changes[:, 0] > allowances).any() and (secant_changes[:, 1] > allowances).any()
no_step = joint_secant_weights([-1., -2.], [[1., 2.]], [0.])
assert np.array_equal(no_step, np.zeros(2))
rail = torch.zeros(3, 384, dtype=torch.float64)
rail[0, 120] = ADC_MAX - 2 * ADC_STEP
rail[1, 120] = ADC_MAX - .5 * ADC_STEP
rail[2, 120] = ADC_MAX + ADC_STEP
rail.requires_grad_()
risk = adc_margin_risk(rail)
assert torch.allclose(risk, torch.tensor([0., .25, 4.], dtype=torch.float64), atol=1e-10)
assert not bool(observe_generated_carrier(rail, q_floor=.001)['flags']['pre_ADC_out_of_range'][1])
assert bool(observe_generated_carrier(rail, q_floor=.001)['flags']['pre_ADC_out_of_range'][2])
risk.sum().backward()
assert rail.grad[1, 120] > 0 and rail.grad[2, 120] > rail.grad[1, 120]

base = N01Response008(dict(q_ref=3., a_ref=.1, log_q_min=-4., log_q_max=5.), 'deterministic').eval()
base.requires_grad_(False)
captured = []
hook = base.unet.register_forward_hook(lambda m, a, result: captured.append(result[0].detach()))
source = carrier[:12]
with torch.no_grad():
    original = base(source, {k: torch.zeros((12,) + s) for k, s in base.noise_shapes.items()})
hook.remove()
encoding = dict(source=source, decoded=captured[0], original=original)
with torch.no_grad():
    filtered = base.response_filter((source / source.amax(1)[:, None])[:, None])
head = MeanHeadRepair01(base, encoding, filtered)
rows = torch.arange(12)
identity_error = float((head(rows) - original['signal_pre']).detach().abs().max())
assert identity_error < 1e-12
cache = response_cache(original['signal_pre'], torch.zeros(12, 8, 384, dtype=torch.float64),
                       torch.zeros(12, 2))
theta = torch.full((54,), .01, dtype=torch.float64)
theta[-1] = 0
model = GlobalRepairResponse01(JointResponse01(cache), theta, head)
z = torch.randn(12, 8)
signal = model(rows, z)
fixed_peak = model.mean(rows)[:, 112:144].argmax(1).double() + 112.
fixed_peak_identity = float((model(rows, z, fixed_peak=fixed_peak) - signal).detach().abs().max())
assert fixed_peak_identity == 0
order = torch.randperm(12)
batch_error = float((model(rows[order], z[order]) - signal[order]).detach().abs().max())
assert batch_error < 1e-10
signal[:, 156:].sum().backward()
assert model.mean.mean_head.weight.grad.abs().max() > 0
assert model.mean.mean_time[-1].weight.grad.abs().max() > 0
assert all(p.grad is None and not p.requires_grad for p in base.parameters())
result = dict(status='PASS_SYNTHETIC_SEMANTICS', gradient_max_abs_error=gradient_error,
              conditional_wave_stream_gradient_max_abs_error=conditional_stream_error,
              conditional_wave_coefficient_max_abs_error=conditional_coefficient_error,
              existing_memory_prefix_bitwise_unchanged=True,
              Q_wave_association_detected_with_global_wave_unchanged=True,
              conditional_target_frozen_after_generated_observations=True,
              wave_coefficient_gradient_max_abs_error=wave_gradient_error,
              positive_wave_coefficient_gradient_max_abs_error=positive_wave_gradient_error,
              fixed_transform_peak_identity_max_V=fixed_peak_identity,
              selection_surrogate_hard_values_bitwise_equal=True,
              selection_surrogate_adds_finite_peak_sensitivity=True,
              joint_secant_mixture_feasible=True, joint_secant_no_descent_returns_zero=True,
              ADC_margin_penalty_precedes_actual_out_of_range=True,
              mean_identity_max_V=identity_error, batch_order_max_V=batch_error,
              undefined_Q_rows_retained=4, original_parameters_frozen=True,
              mean_head_and_time_head_receive_gradients=True,
              scope='STE two-pass algebra, not unbiased hard-ADC derivative or physical validation')
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps(result))
