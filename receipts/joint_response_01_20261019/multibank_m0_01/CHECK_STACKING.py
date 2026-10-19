from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time

import numpy as np
import torch

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from n01.joint01_native_step import native_joint_step
from n01.joint01_features import BLOCKS

plan = json.loads((HERE / 'STACKING_CHECK_PLAN.json').read_text())
torch.set_num_threads(plan['CPU_threads'])
started = time.monotonic()
cases = []
for seed in range(2):
    with np.load(ROOT / f'receipts/joint_response_01_20261019/wake_20261019_1059_01/collected/run_01/seed{seed}_SECANT_TRUST.npz', allow_pickle=False) as archive:
        a = {key: torch.as_tensor(archive[key]) for key in ('theta', 'lower', 'upper',
            'FIT_baseline_residual', 'FIT_baseline_native', 'FIT_cholesky', 'original_jacobian', 'original_native_jacobian')}
    r, n, J, N = (a[key] for key in ('FIT_baseline_residual', 'FIT_baseline_native', 'original_jacobian', 'original_native_jacobian'))
    fmap = SimpleNamespace(cholesky=a['FIT_cholesky'],
        block_slices={name: slice(i * 56, (i + 1) * 56) for i, name in enumerate(BLOCKS)})
    single_maps = {h: fmap for h in ('fast', 'slow', 'common_level')}
    triple_maps = {str(bank) + '/' + h: fmap for bank in range(3) for h in single_maps}
    rr, JJ = torch.cat([r] * 3) / np.sqrt(3), torch.cat([J] * 3) / np.sqrt(3)
    nn, NN = torch.cat([n] * 3), torch.cat([N] * 3)
    np.testing.assert_allclose((JJ.T @ JJ).numpy(), (J.T @ J).numpy(), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose((JJ.T @ rr).numpy(), (J.T @ r).numpy(), rtol=1e-11, atol=1e-11)
    radius = plan['radius']
    outputs = []
    for label, residual, jacobian, native, native_jacobian, maps in (
            ('single', r, J, n, N, single_maps), ('triplicated', rr, JJ, nn, NN, triple_maps)):
        step, geometry = native_joint_step(residual, jacobian * radius, maps, native, native_jacobian * radius,
            a['theta'] / radius, a['lower'] / radius, a['upper'] / radius,
            ridge_fraction=.02, radius=1., macro_factor=1.15, sample_count=4096,
            max_iterations=plan['maxiter'], ftol=plan['ftol'])
        step *= radius
        assert geometry['normalized_minimum_slack'] >= -1e-6
        remaining = float((r + J @ step).square().sum() / r.square().sum())
        np.testing.assert_allclose(remaining, geometry['predicted_joint_remaining_fraction'], rtol=1e-10)
        outputs.append(dict(label=label, iterations=geometry['iterations'], remaining=remaining,
            physical_step_norm=float(step.norm()), ridge=geometry['ridge'] / radius**2))
    np.testing.assert_allclose(outputs[0]['remaining'], outputs[1]['remaining'], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(outputs[0]['ridge'], outputs[1]['ridge'], rtol=1e-12)
    cases.append(dict(seed=seed, results=outputs))
review = dict(status='PASS_REPEATED_BANK_OBJECTIVE_AND_CONSTRAINT_NORMALIZATION', cases=cases,
    solver_calls=4, elapsed_seconds=time.monotonic() - started, new_generator_calls=0,
    committed_parameter_updates=0, scientific_acceptance=False,
    scope='Identical-bank duplication validates algebra and solver dimensions; it is not an independent-bank scientific result.')
(HERE / 'STACKING_CHECK_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps(review))
