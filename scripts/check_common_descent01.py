"""Analytic geometry checks; no scientific data or model calls."""
import json
import numpy as np
from n01.common_descent01 import common_descent_direction

cases = []
for name, gradients, rates, expected in [
    ('orthogonal', [[1., 0.], [0., 1.]], [1., 1.], [-1., -1.]),
    ('opposite', [[1., 0.], [-1., 0.]], [1., 1.], [0., 0.]),
    ('anisotropic_units', [[2., 0.], [0., 1.]], [.5, 2.], [-.5, -2.]),
    ('inactive_zero', [[0., 0.], [1., 1.]], [1., 1.], [-1., -1.]),
    ('stationary', [[0., 0.], [0., 0.]], [1., 1.], [0., 0.]),
]:
    gradient, rate = np.asarray(gradients), np.asarray(rates)
    step, receipt = common_descent_direction(gradient, rate)
    np.testing.assert_allclose(step, expected, rtol=1e-7, atol=1e-10)
    assert (gradient @ step <= 1e-10).all()
    cases.append(dict(name=name, passed=True, receipt=receipt))
gradient = np.array([[1., -2., .5], [-1., 1., 2.], [.4, .3, 1.]])
step, receipt = common_descent_direction(gradient, np.array([.002, .0002, .0002]))
assert (gradient @ step < 0).all()
assert receipt['maximum_step_in_learning_rate_units'] <= 1. + 1e-10
cases.append(dict(name='conflicting_but_feasible', passed=True, receipt=receipt))
print(json.dumps(dict(status='PASS', cases=cases, scientific_model_calls=0)))
