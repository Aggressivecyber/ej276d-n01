"""Response029 engineering semantics on manufactured inputs; no training/data."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch

from n01.model008 import canonical_charge008
from n01.model029 import N01Response029

torch.set_num_threads(2)
root = Path(__file__).resolve().parents[1]
time = torch.arange(384, dtype=torch.float64)
width = torch.tensor([3., 4., 5., 4.], dtype=torch.float64)[:, None]
tail = torch.tensor([30., 40., 50., 60.], dtype=torch.float64)[:, None]
amplitude = torch.tensor([.04, .06, .08, .10], dtype=torch.float64)[:, None]
source = amplitude*torch.where(time < 126, torch.exp(-.5*((time-126)/width)**2), torch.exp(-(time-126)/tail))
source[:, :100] = 0
charge = canonical_charge008(source)
stats = dict(q_ref=float(charge.median()), a_ref=float(amplitude.median()),
    log_q_min=float(charge.min().log())-.2, log_q_max=float(charge.max().log())+.2)
state_before = torch.get_rng_state().clone()
model = N01Response029(stats, seed=2026101729)
assert torch.equal(state_before, torch.get_rng_state())
groups = model.parameter_groups()
parameters = [parameter for values in groups.values() for parameter in values]
assert len({id(parameter) for parameter in parameters}) == len(parameters)
assert {id(parameter) for parameter in parameters} == {id(parameter) for parameter in model.parameters()}
assert all(parameter.requires_grad for parameter in parameters)
noise_a = model.sample_noise(4, generator=torch.Generator().manual_seed(290101))
noise_b = model.sample_noise(4, generator=torch.Generator().manual_seed(290101))
assert all(torch.equal(noise_a[key], noise_b[key]) for key in noise_a)
out = model(source, noise_a)
probe = torch.randn((4, 384), generator=torch.Generator().manual_seed(290102), dtype=torch.float64)
(out['signal_pre']*probe).sum().backward()
mean_gradients = {}
for name, parameter in model.mean_model.named_parameters():
    branch = name.split('.')[0]
    mean_gradients[branch] = mean_gradients.get(branch, 0.)+float(parameter.grad.double().square().sum())
mean_gradients = {key: math.sqrt(value) for key, value in mean_gradients.items()}
for branch in ('charge_head', 'response_filter', 'mean_time', 'unet', 'mean_head'):
    assert mean_gradients[branch] > 0, branch
assert float(out['basis_canonical_charge'].detach().abs().max()) < 1e-12
assert float(out['canonical_local_charge'].detach().abs().max()) < 1e-12
assert float(out['canonical_charge_identity_residual'].detach().abs().max()) < 1e-12
assert bool((out['signal_pre'][:, :100] == 0).all())
assert float(out['signal_envelope'].detach().max()) > 0
assert float(out['signal_noise'].detach().abs().max()) > 0
assert torch.equal(out['shape_coefficients'], noise_a['joint'][:, 1:])
assert torch.equal(out['random_shape_pre_V'], (out['B']*noise_a['joint'].double()[:, 1:, None]).sum(1))
with torch.no_grad():
    replay = model(source, noise_b)
    assert torch.equal(out['signal_pre'].detach(), replay['signal_pre'])
    off = model(source, noise_a, enabled=False)
    assert off['signal_pre'] is off['base_signal_pre_V']
    assert torch.equal(off['signal_pre'], replay['base_signal_pre_V'])
    assert bool((off['alpha'] == 1).all())
    assert bool((off['random_shape_pre_V'] == 0).all())
    assert bool((off['signal_noise'] == 0).all())
    assert torch.allclose(out['charge_v'].square().sum(1).sqrt(), torch.full((4,), .03, dtype=torch.float64))
    # Deliberately activate a shared coordinate; this changes no trained state.
    model.charge_head[-1].bias[1] = .02
    correlated = model(source[:1], {key: value[:1] for key, value in noise_a.items()})
    A = correlated['base_signal_pre_V'][0]
    B = correlated['B'][0, 0]
    v = correlated['charge_v'][0]
    assert v[1] != 0
    # Exact output algebra above permits deterministic quadrature over cached
    # A/B/v: no additional neural forwards or large response population.
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    nodes = torch.as_tensor(nodes*math.sqrt(2.), dtype=torch.float64)
    weights = torch.as_tensor(weights/math.sqrt(math.pi), dtype=torch.float64)
    z0, z1 = torch.meshgrid(nodes, nodes, indexing='ij')
    weights2 = (weights[:, None]*weights[None]).reshape(-1)
    alpha = torch.exp(v[0]*z0.reshape(-1)+v[1]*z1.reshape(-1)-.5*v.square().sum())
    score = z1.reshape(-1)
    Q = canonical_charge008(A)
    covariance = ((Q*alpha-Q)*score*weights2).sum()
    expected_covariance = Q*v[1]
    assert torch.allclose(covariance, expected_covariance, atol=1e-12, rtol=1e-12)
    assert expected_covariance != 0
    mean_signal = ((alpha[:, None]*A+score[:, None]*B)*weights2[:, None]).sum(0)
    assert torch.allclose(mean_signal, A, atol=1e-12, rtol=1e-12)
    assert not model.mean_model.unet._forward_hooks

receipt = dict(schema='n01/model029-artificial-engineering/v1', status='passed',
    model_source_sha256=hashlib.sha256((root/'src/n01/model029.py').read_bytes()).hexdigest(),
    check_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    fixture=dict(rows=4, length=384, peak=126, prefix100_zero=True, stats=stats),
    parameter_counts={key: sum(parameter.numel() for parameter in values) for key, values in groups.items()},
    mean_gradient_norms=mean_gradients,
    charge_initial_log_std=out['charge_v'].detach().square().sum(1).sqrt().tolist(),
    zero_Q_basis_absmax_Vns=float(out['basis_canonical_charge'].detach().abs().max()),
    zero_Q_local_absmax_Vns=float(out['canonical_local_charge'].detach().abs().max()),
    charge_identity_absmax_Vns=float(out['canonical_charge_identity_residual'].detach().abs().max()),
    local_envelope_max_V=float(out['signal_envelope'].detach().max()),
    local_signal_absmax_V=float(out['signal_noise'].detach().abs().max()),
    charge_shape_score_covariance_Vns=float(covariance),
    expected_charge_shape_score_covariance_Vns=float(expected_covariance),
    cached_quadrature_mean_waveform_absmax_error_V=float((mean_signal-A).abs().max()),
    same_seed_noise_and_output_exact=True, noiseoff_original_mean_tensor_exact=True,
    input_RNG_preserved_by_constructor=True, parameter_groups_disjoint_and_complete=True,
    actual_mean_forwards=4, actual_mean_forward_rows=13, artificial_backward_queries=1,
    cached_algebra_quadrature_nodes=576, checkpoint_reads=0,
    real_or_DEV_data_reads=0, optimizer_updates=0,
    scientific_status='not_a_detector_response_validation')
destination = root/'receipts/response029_20261017/model_audit/MODEL029_ENGINEERING.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))
