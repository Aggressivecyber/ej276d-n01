"""Full joint quadratic direction with the existing macro bounds imposed."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
import torch

from .joint01_features import BLOCKS


def constrained_joint_step(residual, jacobian, maps, theta, lower, upper, *, ridge_fraction,
                           radius, macro_factor, sample_count, max_iterations, ftol):
    r = residual.detach().cpu().numpy()
    j = jacobian.detach().cpu().numpy()
    gram = j.T @ j
    ridge = ridge_fraction * np.diag(gram).mean()
    hessian = gram + ridge * np.eye(len(theta))
    gradient = j.T @ r
    objective_scale = float(r @ r)
    operators, offsets, names, original_risks, limits = [], [], [], [], []
    feature_count = maps[next(iter(maps))].cholesky.shape[0]
    for index, (hypothesis, fmap) in enumerate(maps.items()):
        selection = slice(index * feature_count, (index + 1) * feature_count)
        cholesky = fmap.cholesky.detach().cpu().numpy()
        scale = np.sqrt(np.square(cholesky).sum(1))
        raw_r = cholesky @ (r[selection] * np.sqrt(len(maps))) / scale
        raw_j = (cholesky @ (j[selection] * np.sqrt(len(maps)))) / scale[:, None]
        for block in BLOCKS:
            take = fmap.block_slices[block]
            current = float(np.square(raw_r[take]).mean())
            limit = macro_factor * current + 2. / sample_count
            normalization = np.sqrt(len(raw_r[take]) * limit)
            offsets.append(raw_r[take] / normalization)
            operators.append(raw_j[take] / normalization)
            names.append(hypothesis + '/' + block)
            original_risks.append(current)
            limits.append(limit)
    a, b = np.stack(operators), np.stack(offsets)

    def objective(delta):
        return float((.5 * delta @ hessian @ delta + gradient @ delta) / objective_scale)

    def objective_gradient(delta):
        return (hessian @ delta + gradient) / objective_scale

    def constraints(delta):
        predicted = b + a @ delta
        return np.r_[1. - np.square(predicted).sum(1), 1. - delta @ delta / radius**2]

    def constraint_jacobian(delta):
        predicted = b + a @ delta
        return np.vstack((-2. * np.einsum('mni,mn->mi', a, predicted), -2. * delta / radius**2))

    bounds = list(zip((lower - theta).cpu().numpy(), (upper - theta).cpu().numpy()))
    solution = minimize(objective, np.zeros(len(theta)), jac=objective_gradient, method='SLSQP',
        bounds=bounds, constraints={'type': 'ineq', 'fun': constraints, 'jac': constraint_jacobian},
        options={'maxiter': max_iterations, 'ftol': ftol, 'disp': False})
    if not solution.success:
        raise RuntimeError('Joint constrained solver failed: ' + solution.message)
    slack = constraints(solution.x)
    if float(slack.min()) < -1e-6:
        raise RuntimeError('Joint constrained solver returned an infeasible direction')
    delta = torch.as_tensor(solution.x, dtype=theta.dtype, device=theta.device)
    predicted = b + a @ solution.x
    return delta, dict(solver='SLSQP_full_joint_quadratic',iterations=int(solution.nit),
        objective_evaluations=int(solution.nfev), analytic_objective_and_constraint_J=True,
        ridge=float(ridge), step_norm=float(np.linalg.norm(solution.x)),
        normalized_minimum_slack=float(slack.min()),
        original_macro_risks=dict(zip(names, original_risks)),
        macro_limits=dict(zip(names, limits)),
        predicted_macro_risks=dict(zip(names, (np.square(predicted).sum(1) * np.asarray(limits)).tolist())),
        predicted_joint_remaining_fraction=float(np.square(r + j @ solution.x).sum() / objective_scale),
        constraints_do_not_replace_original_hard_CHECK=True)
