"""Synthetic spline identity, shape dependence, Q smoothness and exact adjoints."""
import argparse
import json
import math
from pathlib import Path

import torch
from n01.model008 import N01Response008, canonical_charge008, relative_time_warp008
from n01.model import ResidualFiLM
from n01.global_memory01 import MeanHeadRepair01
from n01.mean_trend01 import MeanTrendGrid01
from n01.charge_spline01 import ChargeSplineMean01

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
torch.set_num_threads(2)
torch.manual_seed(613)
torch.set_default_device(args.device)
torch.use_deterministic_algorithms(True)
t = torch.arange(384, dtype=torch.float64)
shapes = [torch.where(t <= 126, torch.exp(-.5 * ((t - 126) / 5).square()), torch.exp(-(t - 126) / tau))
          for tau in (15., 100.)]
for shape in shapes:
    shape[:100] = 0
q = [x[110:366].sum() for x in shapes]
tail = [x[156:366].sum() for x in shapes]
groups = []
for target in (.207919414, .312575162):
    w = (tail[0] - target * q[0]) / (target * (q[1] - q[0]) - (tail[1] - tail[0]))
    groups.append(((1 - w) * shapes[0] + w * shapes[1])[None] * torch.linspace(.02, .12, 6)[:, None])
source = torch.cat(groups)
base = N01Response008(dict(q_ref=3., a_ref=.1, log_q_min=-4., log_q_max=5.), 'deterministic').eval()
with torch.no_grad():
    base.mean_head.weight.normal_(0, .005)
    for block in base.unet.modules():
        if isinstance(block, ResidualFiLM):
            block.film.weight.normal_(0, .003)
base.requires_grad_(False)
captured = []
hook = base.unet.register_forward_hook(lambda m, a, result: captured.append(result[0].detach()))
with torch.no_grad():
    original = base(source, {k: torch.zeros((12,) + s) for k, s in base.noise_shapes.items()})
    filtered = base.response_filter((source / source.amax(1)[:, None])[:, None])
hook.remove()
encoding = dict(source=source, decoded=captured[0], original=original)
inherited = MeanHeadRepair01(base, encoding, filtered).requires_grad_(False)
parents = torch.arange(12)
model = ChargeSplineMean01(base, encoding, filtered, inherited, parents)
model.prepare(parents, 3)
model.refresh(parents, 3)
with torch.no_grad():
    control = []
    for rows in parents.split(3):
        anchor = model.scalars[rows].clone()
        anchor[:, :2] = 0
        decoded, _ = base.unet(model.features[rows], anchor)
        residual = .20 * inherited.mean_head(decoded).tanh().double() * model.gate
        scale = torch.exp(math.log(2.) * inherited.mean_time(model.scalars[rows]).squeeze(1).tanh().double())
        warped, _ = relative_time_warp008(filtered[rows], scale)
        raw = warped * model.gate + residual
        control.append((original['qbar'][rows, None, None] * raw / canonical_charge008(raw)[..., None])[:, 0])
    identity = float((model(parents) - torch.cat(control)).abs().max())
    assert identity < 1e-7, f'Anchor identity error {identity} V'
    partition = float((model.parent_basis.sum(1) - 1).abs().max())
    assert partition < 1e-12 and bool((model.parent_basis >= 0).all())
    charge_error = float((canonical_charge008(model(parents)) - original['qbar']).abs().max())
    assert charge_error < 1e-12
    altered_scalars = model.scalars.clone()
    altered_scalars[:, :2] += torch.randn(12, 2)
    frozen_condition_error = float((model.anchored_decode(model.features, model.scalars) -
        model.anchored_decode(model.features, altered_scalars)).abs().max())
    assert frozen_condition_error == 0
    shape_effect = float((model(parents)[0] / original['qbar'][0] -
                          model(parents)[6] / original['qbar'][6]).abs().max())
    assert shape_effect > 1e-5
model.cached = None
parameters = [v for v in model.parameters() if v.requires_grad]
draw_rows = torch.tensor([1, 1, 5, 7, 8, 8, 2, 9, 11, 0])
weights = torch.randn(len(draw_rows), 384, dtype=torch.float64)
direct_waves = torch.cat([model.uncached(rows) for rows in parents.split(3)])
direct = torch.autograd.grad((direct_waves[draw_rows] * weights).sum(), parameters)
model.refresh(parents, 3, grad=True)
adjoint = torch.autograd.grad((model(draw_rows) * weights).sum(), model.cached)[0]
mapped = model.map_adjoints({'loss': adjoint}, parents, 3)['loss']
error = max(float((a - b).abs().max()) for a, b in zip(direct, mapped))
assert all(torch.allclose(a, b, rtol=3e-5, atol=3e-6) for a, b in zip(direct, mapped))
with torch.no_grad():
    model.mean_head.weight[2].add_(.003)
grid = MeanTrendGrid01(model, parents, points=4)
curve = grid.losses(model)
grad = torch.autograd.grad(curve.sum(), parameters)
gradient_norm = sum(float(g.square().sum()) for g in grad) ** .5
assert gradient_norm > 0 and bool((curve > 0).all())
assert all(not v.requires_grad for v in model.unet.parameters())
assert all(not v.requires_grad for v in model.mean_time.parameters())
assert all(not v.requires_grad for v in base.parameters())
assert [k for k, v in model.named_parameters() if v.requires_grad] == ['mean_head.weight', 'mean_head.bias']
receipt = dict(status='PASS_SYNTHETIC_SEMANTICS', device=args.device,
    anchor_identity_max_V=identity, spline_partition_max_error=partition,
    canonical_charge_max_error=charge_error, Q_amplitude_leak_into_UNet_max_error=frozen_condition_error,
    normalized_input_shape_effect=shape_effect, repeated_parent_streamed_adjoint_max_abs_error=error,
    spline_curvature_gradient_norm=gradient_norm, trainable_mean_parameters=sum(v.numel() for v in parameters),
    all_other_network_parameters_frozen=True, scientific_data_loaded=False, scientific_acceptance=False)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt))
