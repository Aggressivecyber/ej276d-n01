"""Exact uniform 8:1 empirical W1 used by the 32-draw numerical repair."""
from .losses008_v2 import empirical_wasserstein_1d as original_empirical_w1


def empirical_wasserstein_1d(x, y, weights=None, real_weights=None):
    if weights is None and real_weights is None and len(x) == 8 * len(y):
        y = y.to(x)
        xs, ys = x.sort(dim=0).values, y.sort(dim=0).values
        return (xs.reshape(len(y), 8, *x.shape[1:]) - ys.unsqueeze(1)).abs().mean()
    return original_empirical_w1(x, y, weights, real_weights)
