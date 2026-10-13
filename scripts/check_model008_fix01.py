"""CPU artificial-input semantic checks for the independent Window008 fix01.

No optimizer, scientific arrays, trained checkpoint, GPU call or DEV decode.
Run with -B and an unused E/D --out directory. All inputs are analytic pulses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from n01.model import InvalidWaveformError
from n01.model008 import N01Response008, canonical_charge008
from n01.model008_fix01 import N01Response008Fix01


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def scalar(value):
    return float(value.detach())


def frozen_hashes():
    manifest = json.loads((ROOT / "WINDOW008_SOURCE_MANIFEST.json").read_text(encoding="utf8"))
    got = {name: sha(ROOT / name) for name in manifest["files"]}
    if got != manifest["files"]:
        raise RuntimeError("Original frozen source mismatch")
    return got


STATS = dict(q_ref=3.0, a_ref=0.1, log_q_min=math.log(0.1),
             log_q_max=math.log(100.0), detector_condition=0.0)


def fixture():
    t = torch.arange(384, dtype=torch.float64)
    rows = []
    for amplitude, rise, decay in ((.05, 3., 20.), (.1, 4., 40.), (.2, 5., 70.), (.4, 6., 100.)):
        pulse = torch.where(t < 126., torch.exp(-.5*((t-126.)/rise).square()),
                            torch.exp(-(t-126.)/decay))
        rows.append(amplitude*pulse)
    return torch.stack(rows)


def noise_fixture():
    generator = torch.Generator(device="cpu").manual_seed(2026101301)
    return {key: torch.randn((4,)+shape, generator=generator, dtype=torch.float32)
            for key, shape in N01Response008.noise_shapes.items()}


def objective(output):
    t = torch.arange(384, dtype=torch.float64)
    probe = torch.sin(t/19.) + .3*torch.cos(t/37.) + .2*t/384.
    return (output['signal_pre']*probe).mean() + .003*output['base_regular']


def equal_old_keys(old, new):
    for key, value in old.items():
        if key not in new:
            raise AssertionError("Lost old output key: " + key)
        if torch.is_tensor(value):
            if not torch.equal(value, new[key]):
                raise AssertionError("Changed inactive-path output: " + key)
        elif value != new[key]:
            raise AssertionError("Changed inactive-path value: " + key)


def compare_gradients(old, new):
    for (name, op), (nn, np) in zip(old.named_parameters(), new.named_parameters(), strict=True):
        assert name == nn
        if op.grad is None or np.grad is None:
            assert op.grad is None and np.grad is None, name
        else:
            assert torch.equal(op.grad, np.grad), name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.drive.upper() not in ('E:', 'D:') or out.exists():
        raise ValueError('A fresh E/D output directory is required')
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    before = frozen_hashes()
    source, noises = fixture(), noise_fixture()
    forward_calls, forward_rows = 0, 0

    def run(model, values=source, latents=noises):
        nonlocal forward_calls, forward_rows
        forward_calls += 1
        forward_rows += len(values)
        return model.signal_forward(values, latents)

    equal_paths = []
    constructors = {}
    for arm in ('deterministic', 'simple', 'full'):
        torch.manual_seed(2026101380)
        old = N01Response008(STATS, arm)
        old_constructor_rng = torch.get_rng_state().clone()
        torch.manual_seed(2026101380)
        new = N01Response008Fix01(STATS, arm)
        assert torch.equal(old_constructor_rng, torch.get_rng_state())
        assert list(old.state_dict()) == list(new.state_dict())
        assert list(dict(old.named_parameters())) == list(dict(new.named_parameters()))
        assert list(dict(old.named_buffers())) == list(dict(new.named_buffers()))
        for key, value in old.state_dict().items():
            assert torch.equal(value, new.state_dict()[key]), key
        constructors[arm] = dict(state_keys=len(old.state_dict()),
                                 named_parameters=len(list(old.named_parameters())),
                                 exact_state_and_rng_and_order=True)
        stages = ('baseline', 'mean_response') if arm == 'full' else old.stages
        for stage in stages:
            old.set_stage(stage)
            new.set_stage(stage)
            old.zero_grad(set_to_none=True)
            new.zero_grad(set_to_none=True)
            rng_before = torch.get_rng_state().clone()
            a, b = run(old), run(new)
            equal_old_keys(a, b)
            assert torch.equal(rng_before, torch.get_rng_state())
            if stage != 'baseline':
                objective(a).backward()
                objective(b).backward()
                compare_gradients(old, new)
            equal_paths.append(dict(arm=arm, stage=stage, old_output_keys=len(a),
                                    outputs_bitwise_equal=True,
                                    gradients_bitwise_equal=True if stage != 'baseline' else None,
                                    baseline_has_no_trainable_parameters=stage == 'baseline',
                                    implicit_rng_draws=0))

    torch.manual_seed(2026101380)
    model = N01Response008Fix01(STATS, 'full')
    model.set_stage('joint_stochastic')
    output = run(model)
    repeat = run(model)
    for key, value in output.items():
        if torch.is_tensor(value):
            assert torch.equal(value, repeat[key]), key
    assert set(N01Response008.noise_shapes) == set(model.noise_shapes)
    assert output['basis_canonical_charge'].shape == (4, 8)
    assert scalar(output['basis_canonical_charge'].abs().max()) < 1e-10
    basis_rms_error = scalar((output['basis'].square().mean(-1).sqrt()-1.).abs().max())
    assert basis_rms_error < 1e-12
    assert bool((output['signal_pre'][:, :100] == 0).all())
    assert scalar(output['canonical_charge_identity_residual'].abs().max()) < 1e-10
    assert scalar(output['canonical_local_charge'].abs().max()) < 1e-10
    assert torch.allclose(output['base_shape_Q'], output['shape_Q'], atol=1e-10, rtol=1e-12)
    residual = output['mean_residual'][:, None]
    basis = output['basis']
    gram = basis@basis.transpose(1, 2)/384
    offdiag = gram-torch.diag_embed(gram.diagonal(dim1=1, dim2=2))
    expected_regular = residual.square().mean() + torch.diff(residual, dim=-1).square().mean()
    expected_regular = expected_regular + .01*offdiag.square().mean() + torch.diff(basis, n=2, dim=-1).square().mean() + output['signal_envelope'].square().mean()/model.a_ref.square()
    assert torch.equal(output['regular'], expected_regular)
    model.zero_grad(set_to_none=True)
    objective(output).backward()
    gradient_square_sums = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all()), name
            branch = name.split('.')[0]
            gradient_square_sums[branch] = gradient_square_sums.get(branch, 0.) + float(parameter.grad.double().square().sum())
    gradients = {name: value**.5 for name, value in gradient_square_sums.items()}
    required_nonzero_branches = ('charge_head', 'mean_time', 'response_filter', 'mean_head',
                                'physical_head', 'basis_head', 'envelope_head', 'local_filter')
    for name in required_nonzero_branches:
        assert gradients[name] > 0.0, name

    stress = []
    with torch.no_grad():
        for factor in (100., 10000.):
            fixed = {key: value.clone() for key, value in noises.items()}
            fixed['phys'][:, 2:] *= factor
            changed = run(model, latents=fixed)
            assert torch.equal(output['qsig'], changed['qsig'])
            assert torch.equal(output['time_scale'], changed['time_scale'])
            assert torch.equal(output['base_shape_Q'], changed['base_shape_Q'])
            assert float(changed['canonical_charge_identity_residual'].abs().max()) < 1e-8
            assert float((changed['base_shape_Q']-changed['shape_Q']).abs().max()) < 1e-8
            assert bool((changed['signal_pre'][:, :100] == 0).all())
            stress.append(dict(shape_latent_factor=factor,
                               shape_q=float(changed['shape_Q'].min()),
                               charge_identity_absmax=float(changed['canonical_charge_identity_residual'].abs().max()),
                               base_shape_q_change=0.0, charge_time_latents_unchanged=True))
    guard_results = {}
    # These are only artificial parameter states, never trained checkpoints.
    torch.manual_seed(2026101380)
    guard_model = N01Response008Fix01(STATS, 'full')
    guard_model.set_stage('joint_stochastic')
    with torch.no_grad():
        guard_model.mean_head.weight.zero_()
        guard_model.mean_head.bias.fill_(-8.)
    try:
        run(guard_model, values=source[:1], latents={k: v[:1] for k, v in noises.items()})
    except InvalidWaveformError as error:
        assert 'base shape canonical charge' in str(error)
        guard_results['nonpositive_base_charge'] = str(error)
    else:
        raise AssertionError('Negative base Q was hidden')
    with torch.no_grad():
        guard_model.mean_head.bias.zero_()
        guard_model.basis_head.weight.zero_()
        guard_model.basis_head.bias.zero_()
    try:
        run(guard_model, values=source[:1], latents={k: v[:1] for k, v in noises.items()})
    except ValueError as error:
        assert 'degenerate' in str(error)
        guard_results['degenerate_basis'] = str(error)
    else:
        raise AssertionError('Degenerate basis was replaced or clipped')
    with torch.no_grad():
        guard_model.basis_head.bias[0] = float('nan')
    try:
        run(guard_model, values=source[:1], latents={k: v[:1] for k, v in noises.items()})
    except ValueError as error:
        assert 'Nonfinite' in str(error)
        guard_results['nonfinite_basis'] = str(error)
    else:
        raise AssertionError('Nonfinite basis was hidden')
    after = frozen_hashes()
    assert before == after
    result = dict(
        status='passed_cpu_artificial_semantics_only', torch_version=torch.__version__,
        device='cpu', fixture='four analytic pulses, no scientific arrays or weights',
        scientific_arrays_read=False, trained_checkpoints_read=False,
        DEV_numeric_decodes=0, actual_SOURCE_forward_rows=0, actual_REAL_forward_rows=0,
        artificial_model_forward_calls_including_expected_failures=forward_calls,
        artificial_model_forward_rows_including_expected_failures=forward_rows,
        optimizer_updates=0, scientific_updates=0, GPU_calls=0,
        original_68_files_verified_and_unchanged=len(before) == 68,
        source_manifest_sha256=sha(ROOT / 'WINDOW008_SOURCE_MANIFEST.json'),
        model008_sha256=sha(ROOT / 'src/n01/model008.py'),
        model008_fix01_sha256=sha(ROOT / 'src/n01/model008_fix01.py'),
        script_sha256=sha(__file__), constructors=constructors,
        equal_inactive_paths=equal_paths,
        full_joint=dict(basis_canonical_absmax=scalar(output['basis_canonical_charge'].abs().max()),
                        basis_rms_error=basis_rms_error,
                        canonical_charge_identity_absmax=scalar(output['canonical_charge_identity_residual'].abs().max()),
                        canonical_local_charge_absmax=scalar(output['canonical_local_charge'].abs().max()),
                        base_vs_final_shape_q_absmax=scalar((output['base_shape_Q']-output['shape_Q']).abs().max()),
                        prefix100_exact_zero=True, explicit_latent_replay_bitwise=True,
                        final_projected_basis_regularization_verified=True,
                        branch_gradient_norms=gradients,
                        required_finite_nonzero_branches=list(required_nonzero_branches),
                        shape_encoder_zero_gradient_at_identity_charge_initialization=True),
        finite_shape_latent_stress=stress, guards=guard_results,
        limit='CPU artificial semantics only; no native CUDA, failed-event or scientific response acceptance.'
    )
    (out / 'ENGINEERING_RECEIPT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
