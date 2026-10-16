"""One-off Response029 source semantics check; manufactured tensors only."""
from pathlib import Path
import hashlib
import json
import math

import torch

from n01.losses008 import fit_loss_statistics
from n01.losses008_v2 import N01Loss008
from n01.model008_fix01 import N01Response008Fix01

torch.set_num_threads(2)
torch.manual_seed(29000)
generator = torch.Generator().manual_seed(29001)
time = torch.arange(256, dtype=torch.float64)
prototype_a = torch.exp(-torch.relu(time-16)/32)*(time >= 16)
prototype_b = torch.exp(-torch.relu(time-16)/64)*(time >= 16)
prototype_a = prototype_a/prototype_a.sum()
prototype_b = prototype_b/prototype_b.sum()
# Each pair has exactly the same intended Q and two different input shapes.
# Swapping the two output shapes preserves Q monotonicity and all marginals.
truth = torch.stack((prototype_a, prototype_b)).repeat(128, 1)
truth = truth*torch.linspace(2., 6., 128).repeat_interleave(2)[:, None]
baseline = .0001*torch.randn((256, 100), generator=generator, dtype=torch.float64)
statistics = fit_loss_statistics(truth, truth, baseline)
loss = N01Loss008(statistics, seed=29002, evaluation_seed=29003)
context = loss.create_projection_context(1)
wrong = truth.reshape(128, 2, 256).flip(1).reshape(256, 256)
counterexample = {}
for stage in ('mean_response', 'joint_stochastic'):
    components = loss.components(
        dict(waveform=wrong, baseline100=baseline), truth, stage=stage,
        projection_context=context, real_baseline100=baseline)
    counterexample[stage] = {key: float(value.detach()) for key, value in components.items()}
    counterexample[stage]['supported_QPSD_bins'] = loss.last_diagnostics['conditional_bins_used']
counterexample['truth_map_per_parent_RMSE_V'] = float((wrong-truth).square().mean().sqrt())
counterexample['truth_map_per_parent_relative_RMSE'] = float(
    (wrong-truth).square().sum().sqrt()/truth.square().sum().sqrt())
counterexample['intended_Q_is_preserved'] = True
counterexample['actual_Q_pair_difference_absmax_Vns'] = float((wrong.sum(1)-truth.sum(1)).abs().max())

time = torch.arange(384, dtype=torch.float64)
pulse = torch.where(time < 126, torch.exp(-.5*((time-126)/4)**2), torch.exp(-(time-126)/42))
pulse[:100] = 0
source = .08*pulse[None].repeat(2, 1)
charge = float(source[:, 110:366].sum(1)[0])
model = N01Response008Fix01(dict(q_ref=charge, a_ref=.08,
    log_q_min=math.log(charge)-1, log_q_max=math.log(charge)+1), 'full')
model.set_stage('joint_stochastic')
noise = model.sample_noise(2, generator=torch.Generator().manual_seed(29004))
output = model.signal_forward(source, noise)
probe = torch.randn(output['signal_pre'].shape,
    generator=torch.Generator().manual_seed(29005), dtype=torch.float64)
(output['signal_pre']*probe).sum().backward()
branches = {}
for name, parameter in model.named_parameters():
    branch = name.split('.')[0]
    branches.setdefault(branch, dict(parameters=0, grad_tensors=0, squared_norm=0.))
    branches[branch]['parameters'] += parameter.numel()
    if parameter.grad is not None:
        branches[branch]['grad_tensors'] += 1
        branches[branch]['squared_norm'] += float(parameter.grad.double().square().sum())
for values in branches.values():
    values['norm'] = math.sqrt(values.pop('squared_norm'))

root = Path(__file__).resolve().parents[3]
files = ['src/n01/'+name+'.py' for name in
    ('model', 'model008', 'model008_fix01', 'model010', 'model014', 'model015',
     'model016', 'response008', 'response008_fix01', 'response008_resume01',
     'losses008', 'losses008_v2', 'observation008')]
result = dict(schema='n01/response029-model-audit-artificial/v1',
    scope='only manufactured tensors, current source imports, no checkpoint/data/optimizer',
    global_initialization_seed=29000,
    actual_loss_same_Q_shape_swap_counterexample=counterexample,
    full008Fix01_artificial_signal_gradient=branches,
    shape_encoder_initial_zero_gradient_reason='zero-initialized charge_head.parameters_from_shape weight; not detached',
    canonical_charge_identity_absmax=float(output['canonical_charge_identity_residual'].detach().abs().max()),
    basis_canonical_charge_absmax=float(output['basis_canonical_charge'].detach().abs().max()),
    source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
    artificial_model_forwards=1, artificial_loss_evaluations=2, artificial_backward_queries=1,
    scientific_forwards_on_real_data=0, checkpoint_reads=0, optimizer_updates=0)
destination = Path(__file__).with_name('ARTIFICIAL_SEMANTICS.json')
destination.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2))
