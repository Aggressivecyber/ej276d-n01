"""CPU analytic-input checks of frozen signal ablations; no scientific inputs.

Run with -B and a fresh E/D --out directory. No checkpoint, original SOURCE/REAL,
optimizer, device call or observation replacement is used.
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
sys.path.insert(0, str(ROOT / 'src'))

import torch

from n01.ablation009 import (
    ALL_SIGNAL_RANDOM_OFF009, FrozenSignalAblation009, SignalMask009,
    _intervened_signal_forward009, ablated_signal_forward,
    tensor_state_digest009,
)
from n01.model import InvalidWaveformError
from n01.model008 import N01Response008, canonical_charge008
from n01.model008_fix01 import N01Response008Fix01
from n01.observation008 import observe_generated_carrier


STATS = dict(q_ref=3., a_ref=.1, log_q_min=math.log(.1),
             log_q_max=math.log(100.), detector_condition=0.)
ORIGINAL_FILES = ('src/n01/model.py', 'src/n01/model008.py',
                  'src/n01/model008_fix01.py', 'src/n01/observation008.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bytes_equal(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape and
            a.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() ==
            b.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())


def equal_outputs(reference, candidate):
    for name, value in reference.items():
        assert name in candidate, name
        if isinstance(value, torch.Tensor):
            assert bytes_equal(value, candidate[name]), name
        else:
            assert value == candidate[name], name


def fixture():
    t = torch.arange(384, dtype=torch.float64)
    rows = []
    for amplitude, rise, decay in ((.05, 3., 20.), (.1, 4., 40.), (.2, 5., 70.), (.4, 6., 100.)):
        pulse = torch.where(t < 126., torch.exp(-.5*((t-126.)/rise).square()), torch.exp(-(t-126.)/decay))
        rows.append(amplitude*pulse+.001)
    source = torch.stack(rows).repeat_interleave(4, dim=0)
    generator = torch.Generator(device='cpu').manual_seed(2026101409)
    noise = {name: torch.randn((len(source),)+shape, generator=generator, dtype=torch.float32)
             for name, shape in N01Response008.noise_shapes.items()}
    return source, noise


def model_fixture():
    torch.manual_seed(2026101408)
    model = N01Response008Fix01(STATS, 'full')
    model.set_stage('low_learning_rate')
    with torch.no_grad():
        # Nonzero means and explicit nonzero off-diagonal entries test both
        # variance correction and the difference between row/column removal.
        model.mean_head.bias.fill_(.012)
        model.mean_time[-1].bias.fill_(.10)
        model.physical_head.refinement[-1].weight.zero_()
        model.physical_head.refinement[-1].bias.zero_()
        head = model.physical_head
        diagonal = head.rows == head.cols
        std = torch.tensor([.15, .07]+[.006]*8)
        head.base[-1].bias[diagonal] = torch.logit(std/head.limits)
        head.base[-1].bias[(head.cols == 0) & (head.rows > 0)] = .6
    model.eval()
    model.requires_grad_(False)
    return model


def reject(action, words=None):
    try:
        action()
    except (ValueError, TypeError, RuntimeError, InvalidWaveformError) as error:
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
    original_before = {name: sha(ROOT/name) for name in ORIGINAL_FILES}
    source, noise = fixture()
    source_before = source.clone()
    noise_before = {name: value.clone() for name, value in noise.items()}
    model = model_fixture()
    state_before = tensor_state_digest009(model)
    helper = FrozenSignalAblation009(model)
    calls, rows, checks = 0, 0, []

    def run(action, count=len(source)):
        nonlocal calls, rows
        calls += 1
        rows += count
        rng = torch.get_rng_state().clone()
        with torch.no_grad():
            value = action()
        assert torch.equal(rng, torch.get_rng_state()), 'Implicit RNG consumption'
        return value

    def passed(name, **details):
        checks.append(dict(name=name, passed=True, **details))

    original = run(lambda: model.signal_forward(source, noise))
    normal = run(lambda: helper.signal_forward(source, noise))
    equal_outputs(original, normal)
    assert normal['nominal_mean_signal_pre_V'] is None
    assert normal['intervention009']['extra_nominal_model_forwards'] == 0
    passed('all_on_original_outputs_bitwise_equal', original_output_keys=len(original))
    copied = run(lambda: _intervened_signal_forward009(model, source, noise, SignalMask009()))
    equal_outputs(original, copied)
    passed('copied_assembly_all_on_bitwise_equal')
    convenience = run(lambda: ablated_signal_forward(model, source, noise))
    equal_outputs(original, convenience)
    passed('convenience_all_on_original_outputs_bitwise_equal')

    mean_model = copy.deepcopy(model)
    mean_model.set_stage('mean_response')
    mean_model.eval()
    mean_model.requires_grad_(False)
    mean_reference = run(lambda: mean_model.signal_forward(source, noise))
    off = run(lambda: helper.signal_forward(source, noise, disable=ALL_SIGNAL_RANDOM_OFF009))
    for name in ('signal_pre', 'qbar', 'qsig', 'time_scale', 'unit_shape', 'mean_residual', 'canonical_charge'):
        assert bytes_equal(off[name], mean_reference[name]), name
    assert bytes_equal(off['qsig'], off['qbar'])
    assert bytes_equal(off['nominal_mean_signal_pre_V'], off['signal_pre'])
    assert bool((off['physical_cholesky'] == 0).all())
    assert bool((off['basis'] == 0).all()) and bool((off['signal_noise'] == 0).all())
    assert bool((off['stochastic_shape_voltage_RMS_V'] == 0).all())
    passed('all_signal_off_equals_same_full_own_mean_response_signal', original_model_stage=model.stage)

    q_off = run(lambda: helper.signal_forward(source, noise, disable=('Q',)))
    assert bytes_equal(q_off['qsig'], q_off['qbar'])
    assert bytes_equal(q_off['physical_eta'][:, 1:], original['physical_eta'][:, 1:])
    assert bytes_equal(q_off['physical_cholesky'][:, 1:], original['physical_cholesky'][:, 1:])
    assert bytes_equal(q_off['time_scale'], original['time_scale'])
    assert bool((q_off['sigma_q2'] == 0).all())
    assert bool((q_off['physical_cholesky'][:, 1:, 0] != 0).all())
    passed('charge_off_masks_output_row_and_removes_variance_correction')

    time_off = run(lambda: helper.signal_forward(source, noise, disable=('time',)))
    assert bytes_equal(time_off['time_scale'], mean_reference['time_scale'])
    assert bytes_equal(time_off['qsig'], original['qsig'])
    assert bytes_equal(time_off['physical_eta'][:, 0], original['physical_eta'][:, 0])
    assert bytes_equal(time_off['physical_eta'][:, 2:], original['physical_eta'][:, 2:])
    passed('time_off_preserves_all_other_physical_coordinates')

    basis_off = run(lambda: helper.signal_forward(source, noise, disable=('basis',)))
    assert bytes_equal(basis_off['qsig'], original['qsig'])
    assert bytes_equal(basis_off['time_scale'], original['time_scale'])
    assert bytes_equal(basis_off['signal_noise'], original['signal_noise'])
    assert bool((basis_off['basis'] == 0).all())
    assert bool((basis_off['physical_eta'][:, 2:] == 0).all())
    passed('all_eight_basis_rows_off_preserves_charge_time_and_local_draw')

    one_basis = run(lambda: helper.signal_forward(source, noise, disable=('basis3',)))
    keep = [i for i in range(10) if i != 5]
    assert bool((one_basis['physical_eta'][:, 5] == 0).all())
    assert bytes_equal(one_basis['physical_eta'][:, keep], original['physical_eta'][:, keep])
    assert bytes_equal(one_basis['basis'], original['basis'])
    passed('single_basis_output_row_off_preserves_other_correlated_coordinates')

    local_off = run(lambda: helper.signal_forward(source, noise, disable=('local',)))
    assert bytes_equal(local_off['physical_eta'], original['physical_eta'])
    assert bytes_equal(local_off['signal_pre'], original['qsig'][:, None]*original['unit_shape'])
    assert bool((local_off['signal_noise'] == 0).all())
    passed('local_noise_off_preserves_entire_physical_draw')

    wrong_noise = {name: value.clone() for name, value in noise.items()}
    wrong_noise['phys'][:, 0] = 0
    wrong = run(lambda: model.signal_forward(source, wrong_noise))
    assert not bytes_equal(wrong['physical_eta'][:, 1:], original['physical_eta'][:, 1:])
    assert bytes_equal(wrong['qsig'], wrong['qbar']*torch.exp(-.5*wrong['sigma_q2']))
    assert not bytes_equal(wrong['qsig'], wrong['qbar'])
    passed('column_zeroing_changes_retained_outputs_and_keeps_wrong_Q_half_variance',
           minimum_q_zero_latent_over_qbar=float((wrong['qsig']/wrong['qbar']).min()))

    expected_shape = original['qsig'][:, None]*((original['physical_eta'][:, 2:, None]*original['basis']).sum(1)/original['shape_Q'][:, None])
    assert bytes_equal(normal['stochastic_shape_pre_V'], expected_shape)
    assert bytes_equal(normal['stochastic_shape_voltage_RMS_V'], expected_shape.square().mean(1).sqrt())
    assert bytes_equal(normal['qsig_over_base_shape_Q'], original['qsig']/original['base_shape_Q'])
    assert bytes_equal(normal['local_RMS_V'], original['signal_noise'].square().mean(1).sqrt())
    assert float(normal['decomposition_roundoff_absmax_V'].max()) < 1e-14
    assert bool((normal['stochastic_shape_voltage_RMS_V'] > 0).all())
    assert bool((normal['local_RMS_V'] > 0).all())
    passed('engineering_components_are_derived_without_extra_forwards',
           decomposition_roundoff_absmax_V=float(normal['decomposition_roundoff_absmax_V'].max()),
           stochastic_voltage_RMS_max_V=float(normal['stochastic_shape_voltage_RMS_V'].max()))

    for value in (normal, off, q_off, time_off, basis_off, one_basis, local_off):
        assert bool((value['signal_pre'][:, :100] == 0).all())
        assert float(value['canonical_charge_identity_residual'].abs().max()) < 1e-10
        assert float(value['canonical_local_charge'].abs().max()) < 1e-10
        assert float(canonical_charge008(value['stochastic_shape_pre_V']).abs().max()) < 1e-10
    passed('prefix100_exact_zero_and_fixed_canonical_charge_identity_preserved')

    electronics = torch.linspace(-.001, .001, 384, dtype=torch.float64)[None].expand(len(source), -1)
    observed_original = observe_generated_carrier(original['signal_pre']+electronics, return_indices=True)
    observed_normal = observe_generated_carrier(normal['signal_pre']+electronics, return_indices=True)
    equal_outputs({key: value for key, value in observed_original.items() if isinstance(value, torch.Tensor)}, observed_normal)
    assert observed_original['flags'].keys() == observed_normal['flags'].keys()
    for name in observed_original['flags']:
        assert bytes_equal(observed_original['flags'][name], observed_normal['flags'][name]), name
    passed('existing_ADC_baseline_peak_crop_observation_unchanged')

    assert bytes_equal(source, source_before)
    assert all(bytes_equal(noise[name], noise_before[name]) for name in noise)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert helper.assert_unchanged(deep=True) == state_before
    passed('original_stage_eval_flags_parameters_buffers_stats_inputs_and_rng_unchanged', state_sha256=state_before)

    rejects = {}
    unfrozen = copy.deepcopy(model)
    unfrozen.requires_grad_(True)
    rejects['unfrozen'] = reject(lambda: FrozenSignalAblation009(unfrozen), 'freeze')
    training = copy.deepcopy(model)
    training.train()
    rejects['train_mode'] = reject(lambda: FrozenSignalAblation009(training), 'eval')
    stage_guard = FrozenSignalAblation009(copy.deepcopy(model))
    stage_guard.model.set_stage('joint_stochastic')
    rejects['stage_reenabled_parameters'] = reject(lambda: stage_guard.assert_unchanged(), 'freeze')
    byte_guard = FrozenSignalAblation009(copy.deepcopy(model))
    byte_guard.model.charge_head.parameters_from_shape.bias.data[0] += .125
    rejects['data_write_bypassing_version_counter'] = reject(lambda: byte_guard.assert_unchanged(deep=True), 'bytes changed')
    rejects['unknown_branch'] = reject(lambda: helper.signal_forward(source, noise, disable=('not_a_branch',)), 'Unknown')
    rejects['mask_and_disable'] = reject(lambda: helper.signal_forward(source, noise, disable=('Q',), mask=SignalMask009()), 'either')
    rejects['string_disable'] = reject(lambda: helper.signal_forward(source, noise, disable='Q'), 'sequence')
    rejects['missing_latent'] = reject(lambda: helper.signal_forward(source, {'phys': noise['phys']}, disable=('Q',)), 'explicit')
    rejects['malformed_latent'] = reject(lambda: helper.signal_forward(source, {**noise, 'phys': noise['phys'][:, :9]}, disable=('Q',)), 'latent')
    rejects['invalid_source_peak'] = reject(lambda: helper.signal_forward(source.roll(1, dims=1), noise, disable=('Q',)), 'registered126')
    helper.assert_unchanged(deep=True)
    passed('frozen_state_and_invalid_input_rejections', rejection_count=len(rejects))

    original_after = {name: sha(ROOT/name) for name in ORIGINAL_FILES}
    assert original_before == original_after
    result = dict(
        schema='n01/ablation009-cpu-engineering/v1', UTC=datetime.now(timezone.utc).isoformat(),
        status='passed_cpu_artificial_semantics_only', device='cpu', torch_version=torch.__version__,
        fixture='four analytic positive pulses repeated four times, explicit seeded Gaussian latents',
        checks=checks, passed_checks=len(checks), rejection_details=rejects,
        artificial_model_forward_calls=calls, artificial_model_forward_rows=rows,
        scientific_SOURCE_forward_rows=0, scientific_REAL_forward_rows=0,
        scientific_arrays_read=False, trained_checkpoints_read=False,
        optimizer_updates=0, scientific_updates=0, GPU_forwards=0,
        extra_nominal_model_forwards=0, old_observation_code_unchanged=True,
        original_imported_source_sha256=original_after, original_imported_source_unchanged=True,
        helper_sha256=sha(ROOT/'src/n01/ablation009.py'), script_sha256=sha(__file__),
        frozen_artificial_model_state_sha256=state_before, elapsed_seconds=time.monotonic()-started,
        limit='CPU analytic-input semantics only; not trained-weight, native CUDA or detector-response acceptance.',
    )
    (out/'ENGINEERING_RECEIPT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({key: result[key] for key in ('status', 'passed_checks', 'artificial_model_forward_rows',
                                                'original_imported_source_unchanged', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
