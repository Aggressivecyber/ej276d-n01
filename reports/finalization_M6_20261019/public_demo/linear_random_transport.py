"""The random correction is linear for a fixed input and deterministic mean."""
import numpy as np


def raw_warp_delta(random_local, query_absolute, scale, allowed):
    """Original069 single-area-scale interpolation before its leakage removal."""
    query = query_absolute-110
    lower = np.clip(np.floor(query).astype(np.int64), 0, 64)
    weight = query-lower
    active = np.where(allowed, random_local, 0.)
    mapped = np.where(allowed, (1.-weight)*active[lower]+weight*active[lower+1], 0.)
    return scale*mapped-active


def project_random(raw_delta, space):
    """Metric projection onto the same equality-constrained spline space."""
    basis = space['basis']
    if basis.shape[1] == 0:
        return np.zeros(66)
    coefficients = np.linalg.solve(space['hessian'], basis.T@space['metric']@raw_delta)
    return basis@coefficients
