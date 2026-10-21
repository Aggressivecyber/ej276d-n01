"""Common first-order descent in the declared parameter learning-rate units."""
import numpy as np
from scipy.optimize import minimize


def common_descent_direction(gradients, rates):
    scaled = gradients * rates[None]
    norms = np.linalg.norm(scaled, axis=1)
    active = norms > 0
    a = scaled[active] / norms[active, None]
    if not len(a):
        return np.zeros_like(rates), dict(status='ALL_GRADIENTS_ZERO', common_norm=0., active_rows=[])
    gram = a @ a.T
    solution = minimize(lambda w: .5 * w @ gram @ w,
        np.full(len(a), 1. / len(a)), jac=lambda w: gram @ w, method='SLSQP',
        bounds=[(0., 1.)] * len(a),
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1., 'jac': lambda w: np.ones_like(w)},
        options={'maxiter': 128, 'ftol': 1e-12, 'disp': False})
    if not solution.success:
        raise RuntimeError('Common-descent simplex solve failed: ' + solution.message)
    combined = solution.x @ a
    norm = float(np.linalg.norm(combined))
    directional = a @ combined
    # The minimum-norm convex combination must satisfy this projection KKT inequality.
    gap = float(directional.min() - norm**2)
    if (abs(solution.x.sum() - 1.) > 1e-8 or solution.x.min() < -1e-8 or gap < -2e-6
            or (norm > 1e-8 and directional.min() <= 0.)):
        raise RuntimeError('Common-descent simplex KKT check failed')
    stalled = norm <= 1e-8
    delta = np.zeros_like(rates) if stalled else -rates * combined / np.abs(combined).max()
    return delta, dict(status='NO_COMMON_FIRST_ORDER_DIRECTION' if stalled else 'COMMON_DIRECTION',
        common_norm=norm, weights=solution.x.tolist(), active_rows=np.flatnonzero(active).tolist(),
        normalized_projection_gap=gap, iterations=int(solution.nit),
        predicted_changes=(gradients @ delta).tolist(),
        maximum_step_in_learning_rate_units=float(np.abs(delta / rates).max()))
