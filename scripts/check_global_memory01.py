"""Synthetic semantics only; no project data or model checkpoints."""
import json
from pathlib import Path
import argparse
import torch

from n01.global_memory01 import GlobalDistributionMemory01, MeanHeadRepair01, GlobalRepairResponse01
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
coefficient_wave = torch.autograd.grad((wc_a * wa + wc_b * wb).sum(), wave_parameter)[0]
wave_gradient_error = float((direct_wave - coefficient_wave).abs().max())
assert wave_gradient_error < 1e-10
low = observe_generated_carrier(torch.zeros(4, 384, dtype=torch.float64),
                                q_floor=.001, return_carrier=True)
assert bool(torch.isfinite(memory.sums(low)).all())
assert float(memory.sums(low)[memory.sections['undefined_mass']][1]) == 4

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
order = torch.randperm(12)
batch_error = float((model(rows[order], z[order]) - signal[order]).detach().abs().max())
assert batch_error < 1e-10
signal[:, 156:].sum().backward()
assert model.mean.mean_head.weight.grad.abs().max() > 0
assert model.mean.mean_time[-1].weight.grad.abs().max() > 0
assert all(p.grad is None and not p.requires_grad for p in base.parameters())
result = dict(status='PASS_SYNTHETIC_SEMANTICS', gradient_max_abs_error=gradient_error,
              wave_coefficient_gradient_max_abs_error=wave_gradient_error,
              mean_identity_max_V=identity_error, batch_order_max_V=batch_error,
              undefined_Q_rows_retained=4, original_parameters_frozen=True,
              mean_head_and_time_head_receive_gradients=True,
              scope='STE two-pass algebra, not unbiased hard-ADC derivative or physical validation')
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps(result))
