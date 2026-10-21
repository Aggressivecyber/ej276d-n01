"""Synthetic identity, adjoint, and conditional-curvature checks; no data loads."""
import argparse
import json
from pathlib import Path
import torch
from n01.model008 import N01Response008
from n01.model import ResidualFiLM
from n01.global_memory01 import MeanHeadRepair01
from n01.mean_trend01 import ConditionedMeanRepair01, MeanTrendGrid01

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
inherited = MeanHeadRepair01(base, encoding, filtered)
model = ConditionedMeanRepair01(base, encoding, filtered, inherited)
parents = torch.arange(12)
model.bind_reference(inherited, parents, 3)
model.refresh(parents, 3)
identity = float((model(parents) - inherited(parents)).detach().abs().max())
assert identity < 1e-12
model.cached = None
parameters = [v for v in model.parameters() if v.requires_grad]
draw_rows = torch.tensor([1, 1, 5, 7, 8, 8, 2, 9, 11, 0])
weights = torch.randn(len(draw_rows), 384, dtype=torch.float64)
# Compare to a retained graph with the same parent microbatches. CUDA kernels
# can round differently when the batch geometry itself changes.
direct_waves = torch.cat([model.uncached(rows) for rows in parents.split(3)])
direct = torch.autograd.grad((direct_waves[draw_rows] * weights).sum(), parameters)
model.refresh(parents, 3, grad=True)
adjoint = torch.autograd.grad((model(draw_rows) * weights).sum(), model.cached)[0]
mapped = model.map_adjoints({'loss': adjoint}, parents, 3)['loss']
error = max(float((a - b).abs().max()) for a, b in zip(direct, mapped))
assert all(torch.allclose(a, b, rtol=3e-5, atol=3e-6) for a, b in zip(direct, mapped))
grid = MeanTrendGrid01(model, parents, points=4)
curve = grid.losses(model)
grad = torch.autograd.grad(curve.sum(), parameters)
named = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
condition_norm = sum(float(g.square().sum()) for (k, _), g in zip(named, grad) if k.startswith('unet.')) ** .5
assert condition_norm > 0 and bool((curve > 0).all())
assert all(not v.requires_grad for v in model.mean_time.parameters())
assert all(not v.requires_grad for v in base.parameters())
assert all(k.startswith(('mean_head.', 'unet.condition.')) or '.film.' in k for k, _ in named)
receipt = dict(status='PASS_SYNTHETIC_SEMANTICS', inherited_mean_identity_max_V=identity,
    device=args.device,
    repeated_parent_streamed_adjoint_max_abs_error=error, curvature_conditioning_gradient_norm=condition_norm,
    waveform_curvature_nonzero=True, original_parameters_frozen=True, time_and_nonconditioning_convolutions_frozen=True,
    trainable_parameters=sum(v.numel() for v in parameters), scientific_acceptance=False)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt))
