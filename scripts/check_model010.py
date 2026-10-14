"""Functional CPU checks of 010 with artificial, untrained A and analytic input.

No trained weights or SOURCE/REAL arrays are read. Exactly one artificial
AdamW update tests optimizer/freeze boundaries; it is not a scientific update.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import torch

from n01.model import InvalidWaveformError
from n01.model008 import N01Response008, canonical_charge008
from n01.model010 import FrozenDeterministicVoltageShape010, tensor_state_digest010


STATS = dict(q_ref=3., a_ref=.1, log_q_min=math.log(.1),
             log_q_max=math.log(100.), detector_condition=0.)
PROTECTED_IMPORTS = ('src/n01/model.py', 'src/n01/model008.py',
                     'src/n01/model008_fix01.py', 'src/n01/observation008.py',
                     'src/n01/ablation009.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar(value):
    return float(value.detach())


def bytes_equal(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape and
            a.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() ==
            b.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())


def fixture():
    t = torch.arange(384, dtype=torch.float64)
    rows = []
    for amplitude, rise, decay in ((.05, 3., 20.), (.1, 4., 40.), (.2, 5., 70.), (.4, 6., 100.)):
        pulse = torch.where(t < 126., torch.exp(-.5*((t-126.)/rise).square()), torch.exp(-(t-126.)/decay))
        rows.append(amplitude*pulse+.001)
    # A large negative tail lies beyond canonical Q, exercising max_abs(A)
    # rather than positive_peak(A) as the registered voltage amplitude.
    rows[0][366:] -= .2
    source = torch.stack(rows).repeat_interleave(2, dim=0)
    generator = torch.Generator(device='cpu').manual_seed(2026101440)
    shape = torch.randn((len(source), 8), generator=generator, dtype=torch.float32)
    return source, shape


def baseline_fixture():
    torch.manual_seed(2026101439)
    original = N01Response008(STATS, 'deterministic')
    original.set_stage('low_learning_rate')
    with torch.no_grad():
        original.mean_head.bias.fill_(.007)
        original.mean_time[-1].bias.fill_(.05)
        original.response_filter.raw[1] = .2
        original.charge_head.parameters_from_shape.bias[-1] = .07
    original.eval()
    original.requires_grad_(False)
    return original


def reject(action, words=None):
    try:
        action()
    except (ValueError, TypeError, RuntimeError, InvalidWaveformError, AttributeError) as error:
        if words is not None:
            assert words in str(error), str(error)
        return str(error)
    raise AssertionError('Expected rejection did not occur')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.drive.upper() not in ('E:', 'D:') or out.exists():
        raise ValueError('A fresh E/D output directory is required')
    out.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protected_before = {name: sha(ROOT/name) for name in PROTECTED_IMPORTS}
    source, z = fixture()
    source_before, z_before = source.clone(), z.clone()
    baseline = baseline_fixture()
    counter = {'UNet_forwards': 0}

    def count_unet(module, arguments, result):
        counter['UNet_forwards'] += 1

    instrumentation = baseline.unet.register_forward_hook(count_unet)
    baseline_sha = tensor_state_digest010(baseline)
    baseline_stage = baseline.stage
    cpu_rng = torch.get_rng_state().clone()
    model = FrozenDeterministicVoltageShape010(baseline)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert model.assert_frozen_unchanged(deep=True) == baseline_sha
    checks, forward_calls, forward_rows = [], 0, 0

    def passed(name, **detail):
        checks.append(dict(name=name, passed=True, **detail))

    def run(coefficients=z, enabled=True, values=source):
        nonlocal forward_calls, forward_rows
        before_rng = torch.get_rng_state().clone()
        before_unet = counter['UNet_forwards']
        hooks_before = tuple(baseline.unet._forward_hooks)
        forward_calls += 1
        forward_rows += len(values)
        result = model.random_forward(values, coefficients, enabled=enabled)
        assert counter['UNet_forwards']-before_unet == 1
        assert tuple(baseline.unet._forward_hooks) == hooks_before
        assert torch.equal(before_rng, torch.get_rng_state())
        assert result['baseline_signal_forward_calls'] == 1
        assert result['extra_baseline_signal_forward_calls'] == 0
        return result

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        second_rng = torch.get_rng_state().clone()
        second = FrozenDeterministicVoltageShape010(copy.deepcopy(baseline))
        assert torch.equal(second_rng, torch.get_rng_state())
    assert tensor_state_digest010(second.basis_head) == model.initial_head_state_sha256
    assert bool((model.basis_head.bias == 0).all())
    assert scalar(model.basis_head.weight.std()) > 0
    assert model.initialization_global_CPU_RNG_preserved
    passed('explicit_CPU_generator_initialization_reproducible_without_global_RNG_change',
           initialization_seed=model.initialization_seed,
           empirical_weight_std=scalar(model.basis_head.weight.std()),
           initial_head_state_sha256=model.initial_head_state_sha256)

    original_noise = {name: torch.zeros((len(source),)+shape, dtype=torch.float32)
                      for name, shape in baseline.noise_shapes.items()}
    with torch.no_grad():
        reference = baseline.signal_forward(source, original_noise)
    forward_calls += 1
    forward_rows += len(source)
    zero = run(torch.zeros_like(z))
    disabled = run(z, enabled=False)
    for candidate in (zero, disabled):
        for name in ('signal_pre', 'qbar', 'qsig', 'time_scale', 'signal_noise', 'unit_shape'):
            assert bytes_equal(candidate[name], reference[name]), name
        assert bytes_equal(candidate['base_signal_pre_V'], reference['signal_pre'])
        assert bool((candidate['random_shape_pre_V'] == 0).all())
        assert scalar(candidate['regular']) == 0
    passed('zero_latent_and_disabled_signal_are_exact_original_A_bytes')

    plus = run(z)
    minus = run(-z)
    assert torch.equal(plus['random_shape_pre_V'], -minus['random_shape_pre_V'])
    pair_error = ((plus['signal_pre']+minus['signal_pre'])/2-reference['signal_pre']).abs().max()
    assert scalar(pair_error) < 1e-14
    assert scalar(canonical_charge008(plus['voltage_basis_V']).abs().max()) < 1e-10
    assert scalar(plus['canonical_random_shape_charge'].abs().max()) < 1e-10
    assert scalar(plus['canonical_charge_identity_residual'].abs().max()) < 1e-10
    assert bool((plus['signal_pre'][:, :100] == 0).all())
    assert bool((plus['voltage_basis_V'][..., :100] == 0).all())
    passed('antithetic_shape_draws_preserve_preADC_mean_and_canonical_Q',
           antithetic_mean_absmax_V=scalar(pair_error),
           basis_canonical_Q_absmax=scalar(plus['basis_canonical_charge'].abs().max()),
           signal_charge_identity_absmax=scalar(plus['canonical_charge_identity_residual'].abs().max()))

    assert bytes_equal(plus['basis_amplitude_V'], .05*reference['signal_pre'].abs().amax(1))
    assert bool((reference['signal_pre'][:2].abs().amax(1) > reference['signal_pre'][:2].amax(1)).all())
    assert bool(torch.isfinite(plus['voltage_basis_V']).all())
    assert bool((plus['random_RMS_V'] > 0).all())
    ratio = plus['random_RMS_V']/reference['signal_pre'].abs().amax(1)
    assert scalar(ratio.max()) < .05
    passed('small_nonzero_voltage_bases_use_max_abs_frozen_A_only',
           initial_random_RMS_to_A_peak_abs_max=scalar(ratio.max()),
           initial_random_RMS_max_V=scalar(plus['random_RMS_V'].max()))

    assert bytes_equal(plus['regular_rms'], plus['random_shape_pre_V'].square().mean()/baseline.a_ref.square())
    assert bytes_equal(plus['regular_curvature'], torch.diff(plus['random_shape_pre_V'], n=2, dim=-1).square().mean()/baseline.a_ref.square())
    assert bytes_equal(plus['regular'], plus['regular_rms']+plus['regular_curvature'])
    assert bytes_equal(plus['random_RMS_V'], plus['random_shape_pre_V'].square().mean(1).sqrt())
    passed('regularization_uses_realized_voltage_energy_and_second_difference_in_fixed_units',
           energy=scalar(plus['regular_rms']), curvature=scalar(plus['regular_curvature']))

    for mode in (False, True, False, True):
        model.train(mode)
        assert model.training == mode and model.basis_head.training == mode
        assert baseline.stage == baseline_stage
        assert not any(module.training for module in baseline.modules())
        assert model.assert_frozen_unchanged(deep=True) == baseline_sha
    model.eval()
    assert not model.training and not model.basis_head.training
    model.requires_grad_(False)
    assert not any(value.requires_grad for value in model.trainable_parameters())
    model.requires_grad_(True)
    assert all(value.requires_grad for value in model.trainable_parameters())
    assert not any(value.requires_grad for value in baseline.parameters())
    assert not hasattr(model, 'set_stage')
    passed('train_eval_requires_grad_and_loss_stage_controls_never_reopen_original_A')

    actual_names = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    assert actual_names == model.trainable_parameter_names() == ('basis_head.weight', 'basis_head.bias')
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=.0002, weight_decay=.0001)
    model.assert_optimizer_membership(optimizer)
    bad_optimizer = torch.optim.AdamW(model.parameters(), lr=.0002)
    bad_membership = reject(lambda: model.assert_optimizer_membership(bad_optimizer), 'exactly')
    optimizer.zero_grad(set_to_none=True)
    learned = run(z)
    t = torch.arange(384, dtype=torch.float64)
    probe = torch.sin(t/19.)*baseline.gate[0, 0]
    probe = probe-canonical_charge008(probe)*baseline.projection[0, 0]
    target = .003*z[:, :1].double()*probe[None]
    objective = (learned['random_shape_pre_V']-target).square().mean()/baseline.a_ref.square()+.01*learned['regular']
    objective.backward()
    gradients = {}
    for name, value in model.basis_head.named_parameters():
        assert value.grad is not None and bool(torch.isfinite(value.grad).all())
        gradients['basis_head.'+name] = float(value.grad.double().square().sum().sqrt())
        assert gradients['basis_head.'+name] > 0, name
    assert all(value.grad is None for value in baseline.parameters())
    before_head = {name: value.detach().clone() for name, value in model.basis_head.named_parameters()}
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    changed = [name for name, value in model.basis_head.named_parameters() if not bytes_equal(value, before_head[name])]
    assert changed == ['weight', 'bias']
    assert model.assert_frozen_unchanged(deep=True) == baseline_sha
    assert tensor_state_digest010(model.basis_head) != model.initial_head_state_sha256
    model.assert_optimizer_membership(optimizer)
    passed('one_artificial_AdamW_update_has_nonzero_new_head_gradients_and_changes_only_whitelist',
           gradient_norms=gradients, changed_parameters=['basis_head.'+name for name in changed],
           trainable_parameter_count=sum(value.numel() for value in model.trainable_parameters()),
           frozen_parameters_with_grad=0, invalid_optimizer_rejection=bad_membership)

    rejects = {}
    hooks_before = tuple(baseline.unet._forward_hooks)
    rejects['bad_z_shape'] = reject(lambda: model.random_forward(source, z[:, :7]), 'Bx8')
    rejects['nonfinite_z'] = reject(lambda: model.random_forward(source, torch.full_like(z, float('nan'))), 'finite')
    rejects['nonbool_enabled'] = reject(lambda: model.random_forward(source, z, enabled=1), 'bool')
    rejects['bad_source_peak'] = reject(lambda: model.random_forward(source.roll(1, dims=1), z), 'registered126')
    assert tuple(baseline.unet._forward_hooks) == hooks_before
    model.assert_frozen_unchanged(deep=True)
    changed_stage = FrozenDeterministicVoltageShape010(copy.deepcopy(baseline))
    changed_stage.baseline.set_stage('joint_stochastic')
    rejects['old_set_stage_reopens_gradients'] = reject(lambda: changed_stage.assert_frozen_unchanged(), 'frozen')
    changed_bytes = FrozenDeterministicVoltageShape010(copy.deepcopy(baseline))
    changed_bytes.baseline.mean_time[-1].bias.data[0] += .125
    rejects['state_data_write_bypassing_version'] = reject(lambda: changed_bytes.assert_frozen_unchanged(deep=True), 'bytes changed')
    rejects['changed_gain'] = reject(lambda: FrozenDeterministicVoltageShape010(copy.deepcopy(baseline), amplitude_fraction=.1), 'registered')
    rejects['changed_seed'] = reject(lambda: FrozenDeterministicVoltageShape010(copy.deepcopy(baseline), seed=17), 'registered')
    passed('invalid_inputs_and_mutated_frozen_state_rejected_with_temporary_hook_removed', rejection_count=len(rejects))

    assert bytes_equal(source, source_before) and bytes_equal(z, z_before)
    assert not any(value.grad is not None for value in baseline.parameters())
    assert model.assert_frozen_unchanged(deep=True) == baseline_sha
    assert model.stage == baseline_stage
    protected_after = {name: sha(ROOT/name) for name in PROTECTED_IMPORTS}
    assert protected_before == protected_after
    passed('original_state_inputs_source_files_and_stage_preserved')
    instrumentation.remove()
    result = dict(
        schema='n01/shape010-model-cpu-engineering/v1', UTC=datetime.now(timezone.utc).isoformat(),
        status='passed_cpu_artificial_semantics_only', device='cpu', torch_version=torch.__version__,
        fixture='untrained deterministic A, analytic pulses, signed out-of-canonical tail, explicit seeded Bx8 coefficients',
        checks=checks, passed_checks=len(checks), rejection_details=rejects,
        artificial_valid_baseline_signal_forward_calls=forward_calls,
        artificial_valid_baseline_signal_forward_rows=forward_rows,
        measured_valid_original_UNet_forward_calls=counter['UNet_forwards'],
        artificial_optimizer_updates=1, scientific_optimizer_updates=0,
        scientific_SOURCE_forward_rows=0, scientific_REAL_forward_rows=0,
        scientific_arrays_read=False, trained_checkpoints_read=False, GPU_forwards=0,
        frozen_baseline_state_sha256=baseline_sha,
        initial_new_head_state_sha256=model.initial_head_state_sha256,
        final_artificial_new_head_state_sha256=tensor_state_digest010(model.basis_head),
        trainable_parameter_names=list(model.trainable_parameter_names()),
        protected_source_sha256=protected_after, protected_sources_unchanged=True,
        helper_sha256=sha(ROOT/'src/n01/model010.py'), script_sha256=sha(__file__),
        elapsed_seconds=time.monotonic()-started,
        limit='CPU artificial checks only. preADC mean/Q properties do not establish observed Q/PSD or detector-response acceptance.',
    )
    assert counter['UNet_forwards'] == forward_calls
    (out/'ENGINEERING_RECEIPT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({key: result[key] for key in ('status', 'passed_checks', 'artificial_valid_baseline_signal_forward_rows',
                                                'artificial_optimizer_updates', 'protected_sources_unchanged', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
