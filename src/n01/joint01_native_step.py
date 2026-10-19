"""Joint quadratic direction with macro and native metric constraints."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
import torch

from .joint01_features import BLOCKS


def native_joint_step(residual, jacobian, maps, native_values, native_jacobian,
                      theta, lower, upper, *, ridge_fraction, radius,
                      macro_factor, sample_count, max_iterations, ftol):
    r, j = (value.detach().cpu().numpy() for value in (residual, jacobian))
    values, derivatives = (value.detach().cpu().numpy() for value in (native_values, native_jacobian))
    # Each metric is divided by its current positive distance. This changes
    # constraint units only; the original joint objective is unchanged.
    native_operator = derivatives / values[:, None]
    gram = j.T @ j
    ridge = ridge_fraction * np.diag(gram).mean()
    hessian = gram + ridge * np.eye(len(theta))
    gradient, objective_scale = j.T @ r, float(r @ r)
    operators, offsets, names, limits = [], [], [], []
    feature_count = maps[next(iter(maps))].cholesky.shape[0]
    for index, (hypothesis, fmap) in enumerate(maps.items()):
        selection = slice(index * feature_count, (index + 1) * feature_count)
        cholesky = fmap.cholesky.detach().cpu().numpy()
        scale = np.sqrt(np.square(cholesky).sum(1))
        raw_r = cholesky @ (r[selection] * np.sqrt(len(maps))) / scale
        raw_j = (cholesky @ (j[selection] * np.sqrt(len(maps)))) / scale[:, None]
        for block in BLOCKS:
            take = fmap.block_slices[block]
            limit = macro_factor * float(np.square(raw_r[take]).mean()) + 2. / sample_count
            normalization = np.sqrt(len(raw_r[take]) * limit)
            offsets.append(raw_r[take] / normalization)
            operators.append(raw_j[take] / normalization)
            names.append(hypothesis + '/' + block)
            limits.append(limit)
    a, b = np.stack(operators), np.stack(offsets)

    def objective(delta):
        return float((.5 * delta @ hessian @ delta + gradient @ delta) / objective_scale)

    def objective_gradient(delta):
        return (hessian @ delta + gradient) / objective_scale

    def constraints(delta):
        predicted = b + a @ delta
        return np.r_[1. - np.square(predicted).sum(1), -native_operator @ delta,
                     1. - delta @ delta / radius**2]

    def constraint_jacobian(delta):
        predicted = b + a @ delta
        return np.vstack((-2. * np.einsum('mni,mn->mi', a, predicted),
                          -native_operator, -2. * delta / radius**2))

    bounds = list(zip((lower - theta).cpu().numpy(), (upper - theta).cpu().numpy()))
    solution = minimize(objective, np.zeros(len(theta)), jac=objective_gradient, method='SLSQP',
        bounds=bounds, constraints={'type': 'ineq', 'fun': constraints, 'jac': constraint_jacobian},
        options={'maxiter': max_iterations, 'ftol': ftol, 'disp': False})
    if not solution.success:
        raise RuntimeError('Native-constrained joint solver failed: ' + solution.message)
    slack = constraints(solution.x)
    if float(slack.min()) < -1e-6:
        raise RuntimeError('Native-constrained joint solver returned an infeasible direction')
    predicted = b + a @ solution.x
    return torch.as_tensor(solution.x, dtype=theta.dtype, device=theta.device), dict(
        solver='SLSQP_full_joint_quadratic_native_nonincrease', iterations=int(solution.nit),
        objective_evaluations=int(solution.nfev), ridge=float(ridge),
        step_norm=float(np.linalg.norm(solution.x)), normalized_minimum_slack=float(slack.min()),
        macro_limits=dict(zip(names, limits)),
        predicted_macro_risks=dict(zip(names, (np.square(predicted).sum(1) * np.asarray(limits)).tolist())),
        original_native_values=values.tolist(),
        predicted_native_values=(values + derivatives @ solution.x).tolist(),
        predicted_joint_remaining_fraction=float(np.square(r + j @ solution.x).sum() / objective_scale),
        joint_objective_unchanged=True, constraints_do_not_replace_hard_observation=True)
