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

plan = json.loads((HERE / 'SOLVER_REPARAMETERIZATION_PLAN.json').read_text())
torch.set_num_threads(plan['CPU_threads'])
started = time.monotonic()
results, arrays = [], {}
for seed in range(2):
    source = ROOT / f'receipts/joint_response_01_20261019/wake_20261019_1010_01/collected/run_01/seed{seed}_READOUT_ATTRIBUTION.npz'
    with np.load(source, allow_pickle=False) as archive:
        a = {key: torch.as_tensor(archive[key]) for key in ('theta', 'cholesky',
            'original_baseline_residual', 'original_baseline_native', 'case0_original_jacobian',
            'case0_original_native_jacobian', 'case0_probe1_proposal',
            'case0_probe1_original_actual_residual', 'case0_probe1_original_actual_native')}
    theta, chol = a['theta'], a['cholesky']
    r, n = a['original_baseline_residual'], a['original_baseline_native']
    lower, upper = torch.full_like(theta, -1.), torch.ones_like(theta)
    lower[-1] = 0.
    maps = {h: SimpleNamespace(cholesky=chol,
        block_slices={name: slice(i * 56, (i + 1) * 56) for i, name in enumerate(BLOCKS)})
        for h in ('fast', 'slow', 'common_level')}
    for corrected in (False, True):
        J, N = a['case0_original_jacobian'].clone(), a['case0_original_native_jacobian'].clone()
        if corrected:
            s = a['case0_probe1_proposal'] - theta
            J += torch.outer(a['case0_probe1_original_actual_residual'] - r - J @ s, s) / s.square().sum()
            N += torch.outer(a['case0_probe1_original_actual_native'] - n - N @ s, s) / s.square().sum()
        ridge = float(.02 * (J.T @ J).diag().mean())
        for radius in plan['radii']:
            d = torch.linspace(-1., 1., len(theta), dtype=theta.dtype)
            d *= radius * .25 / d.norm()
            u = d / radius
            np.testing.assert_allclose((J * radius) @ u, J @ d, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose((N * radius) @ u, N @ d, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose((lower / radius - theta / radius) * radius,
                lower - theta, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(ridge * float(d @ d), ridge * radius**2 * float(u @ u), rtol=1e-12)
            for scaled in (False, True):
                assert time.monotonic() - started < plan['internal_seconds']
                factor = radius if scaled else 1.
                record = dict(seed=seed, corrected=corrected, radius=radius, scaled=scaled)
                try:
                    delta, geometry = native_joint_step(r, J * factor, maps, n, N * factor,
                        theta / factor, lower / factor, upper / factor, ridge_fraction=.02,
                        radius=radius / factor, macro_factor=1.15, sample_count=4096,
                        max_iterations=200, ftol=1e-9)
                except RuntimeError as error:
                    record.update(success=False, error=str(error))
                    results.append(record)
                    continue
                delta *= factor
                joint = float((r + J @ delta).square().sum() / r.square().sum())
                np.testing.assert_allclose(geometry['predicted_joint_remaining_fraction'], joint, rtol=1e-12)
                np.testing.assert_allclose(geometry['ridge'] / factor**2, ridge, rtol=1e-12)
                slack = [float(1. - delta.square().sum() / radius**2)]
                slack.extend((-N @ delta / n).tolist())
                for ih, h in enumerate(maps):
                    take = slice(ih * len(chol), (ih + 1) * len(chol))
                    scale = chol.square().sum(1).sqrt()
                    raw0 = chol @ (r[take] * np.sqrt(3)) / scale
                    raw1 = chol @ ((r + J @ delta)[take] * np.sqrt(3)) / scale
                    for block in BLOCKS:
                        part = maps[h].block_slices[block]
                        limit = 1.15 * float(raw0[part].square().mean()) + 2. / 4096
                        value = float(raw1[part].square().mean())
                        slack.append(1. - value / limit)
                        np.testing.assert_allclose(geometry['predicted_macro_risks'][h + '/' + block], value, rtol=1e-11, atol=1e-13)
                assert min(slack) >= -1e-6
                assert bool(((theta + delta >= lower - 1e-12) & (theta + delta <= upper + 1e-12)).all())
                record.update(success=True, iterations=geometry['iterations'], step_norm=float(delta.norm()),
                    minimum_original_constraint_slack=min(slack), predicted_joint_remaining_fraction=joint)
                arrays[f'seed{seed}_corrected{int(corrected)}_radius{radius}_scaled{int(scaled)}'] = delta.numpy()
                results.append(record)
assert len(results) == plan['maximum_solver_calls']
assert all(record['success'] for record in results if record['scaled'])
np.savez(HERE / 'SOLVER_SCALE_DIRECTIONS.npz', **arrays)
review = dict(status='PASS_EQUIVALENT_VARIABLE_REPARAMETERIZATION_AND_SCALED_FEASIBILITY',
    cases=results, solver_calls=len(results), elapsed_seconds=time.monotonic() - started,
    unscaled_failures=sum(not x['success'] for x in results if not x['scaled']),
    scaled_failures=sum(not x['success'] for x in results if x['scaled']),
    new_generator_calls=0, parameter_updates=0, scientific_acceptance=False,
    exact_failed_fourth_matrix_not_saved=True)
(HERE / 'SOLVER_SCALE_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps(review))
