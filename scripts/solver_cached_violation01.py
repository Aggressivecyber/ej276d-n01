"""Process-local repair of the pinned SciPy1.16.3 cached-violation interface."""
import hashlib
import inspect
import numpy as np
from scipy._lib.cobyqa.problem import NonlinearConstraints

original_violation = NonlinearConstraints.violation
assert hashlib.sha256(inspect.getsource(original_violation).encode()).hexdigest() == '7f3433f3730fac39a233bfde55ca19686747b414cc7f01467151e6dd989a1eb4'

def cached_violation(self, x, cub_val=None, ceq_val=None):
    if cub_val is not None and ceq_val is not None:
        return np.r_[np.maximum(cub_val, 0.), np.abs(ceq_val)]
    return original_violation(self, x, cub_val, ceq_val)

NonlinearConstraints.violation = cached_violation
