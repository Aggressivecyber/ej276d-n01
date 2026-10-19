"""One continuous spline correction in an event's conserved subspace."""
import numpy as np
import osqp
from scipy import sparse
from scipy.interpolate import BSpline


POSITIONS = np.arange(110, 176)
KNOTS = np.r_[np.zeros(4), 65*np.arange(1, 21)/21, np.full(4, 65.)]
PHI = BSpline.design_matrix(np.arange(66), KNOTS, 3).toarray()
SECOND = np.diff(np.eye(66), n=2, axis=0)
SOLVER_SETTINGS = dict(eps_abs=1e-12, eps_rel=1e-12, max_iter=10000,
                       adaptive_rho=True, adaptive_rho_interval=50,
                       polishing=True, warm_starting=False,
                       verbose=False)


def constraint_space(mean, scales, radius=4, lambda_s=.01, lambda_r=1.):
    """Build constraints and a metric using only the deterministic mean."""
    peak = int(mean[112:144].argmax()+112)
    local = mean[110:176]
    allowed = (local > 0.) & (POSITIONS >= peak-14)
    structural = ~allowed | np.isin(POSITIONS, [110, 111, 174, 175])
    front = (POSITIONS >= peak-16) & (POSITIONS < peak+30)
    tail = (POSITIONS >= peak+30) & (POSITIONS < peak+240)
    equality = np.vstack((PHI[structural], PHI[front].sum(0), PHI[tail].sum(0)))
    norms = np.linalg.norm(equality, axis=1)
    scaled = equality[norms > 0.]/norms[norms > 0., None]
    _, singular, vh = np.linalg.svd(scaled, full_matrices=True)
    rank = int(np.count_nonzero(singular > 1e-12*singular[0]))
    null = vh[rank:].T
    basis = PHI@null
    basis[structural] = 0.  # Structural zeros are exact in the represented space.
    a_ref, q_ref, t_ref = scales
    neighbors = np.arange(max(112, peak-radius), min(144, peak+radius+1))
    readout = np.vstack([row for q in neighbors for row in (
        ((POSITIONS >= q-16) & (POSITIONS < q+240))/q_ref,
        ((POSITIONS >= q+30) & (POSITIONS < q+240))/t_ref)])
    metric = (np.eye(66)/(66*a_ref*a_ref)
              + 2*lambda_s*(SECOND.T@SECOND)/(64*a_ref*a_ref)
              + 2*lambda_r*(readout.T@readout)/len(neighbors))
    hessian = basis.T@metric@basis
    hessian = (hessian+hessian.T)/2
    other = np.arange(112, 144)
    other = other[other != peak]
    gaps = mean[peak]-mean[other]
    margins = np.minimum(gaps, np.where(other < peak, 1e-8*a_ref, 1e-10*a_ref))
    inequality = np.vstack((basis[allowed], basis[peak-110]-basis[other-110]))
    positive_floor = np.minimum(local[allowed], 1e-10*a_ref)
    lower = np.r_[positive_floor-local[allowed], margins-gaps]
    return dict(peak=peak, allowed=allowed, structural=structural, front=front, tail=tail,
                equality_rank=rank, null=null, basis=basis, metric=metric,
                hessian=hessian, inequality=inequality, lower=lower, scales=scales)


def fit_mean(mean, target_delta, space):
    """Solve the registered convex QP; failures never replace or drop an event."""
    basis = space['basis']
    dimensions = basis.shape[1]
    if dimensions == 0:
        return np.zeros(66), np.zeros(0), dict(status='NO_FREE_DOF_RETAIN_PARENT',
            free_dimensions=0, iterations=0, primal_residual=0., dual_residual=0.,
            objective_delta=0., equality_rank=space['equality_rank'])
    linear = -(basis.T@target_delta)/(66*space['scales'][0]**2)
    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(np.triu(space['hessian'])), q=linear,
                 A=sparse.csc_matrix(space['inequality']), l=space['lower'],
                 u=np.full(len(space['lower']), np.inf), **SOLVER_SETTINGS)
    result = solver.solve(raise_error=False)
    if result.info.status != 'solved':
        raise RuntimeError(f'S mean QP {result.info.status}; iterations={result.info.iter}')
    delta = basis@result.x
    violation = float(np.max(space['lower']-space['inequality']@result.x))
    adapted = mean.copy()
    adapted[110:176] += delta
    residual = max(abs(delta[space['front']].sum()), abs(delta[space['tail']].sum()))
    if residual > 1e-12 or violation > 1e-12 or not np.isfinite(delta).all():
        raise RuntimeError(f'S mean equality/nonfinite failure: {residual}')
    if np.any(adapted[110:176][space['allowed']] < 0.):
        raise RuntimeError('S mean positive support became negative')
    if int(adapted[112:144].argmax()+112) != space['peak']:
        raise RuntimeError('S mean first argmax changed after float64 assembly')
    return delta, result.x, dict(status='solved', free_dimensions=dimensions,
        iterations=int(result.info.iter), primal_residual=float(result.info.prim_res),
        dual_residual=float(result.info.dual_res), objective_delta=float(result.info.obj_val),
        equality_rank=space['equality_rank'], polishing_status=int(result.info.status_polish),
        maximum_linear_constraint_violation_V=violation)
