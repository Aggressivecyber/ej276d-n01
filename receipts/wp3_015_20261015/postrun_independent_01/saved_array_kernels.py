"""Mature NumPy-only morphology and pairwise-finite covariance kernels. No model imports."""

import numpy as np

def require(value, message):
    if not value:
        raise ValueError(message)

def _crossing(y, fraction, side):
    peak = float(y[16])
    if peak <= 0:
        return np.nan
    target = fraction*peak
    pairs = ((j-1, j) for j in range(16, 0, -1)) if side == 'left' else ((j, j+1) for j in range(16, 255))
    for lo, hi in pairs:
        a, b = float(y[lo]), float(y[hi])
        if (side == 'left' and a <= target <= b) or (side == 'right' and b <= target <= a):
            return float(hi if side == 'left' else lo) if a == b else lo+(target-a)/(b-a)
    return np.nan

def morphology(values):
    y = np.asarray(values, np.float64)
    require(y.ndim == 2 and y.shape[1] == 256 and np.isfinite(y).all(), 'Invalid hard observed wave')
    left = np.asarray([[_crossing(row, f, 'left') for f in (.1, .5, .9)] for row in y])
    right = np.asarray([[_crossing(row, f, 'right') for f in (.1, .5, .9)] for row in y])
    peak = y[:, 16]
    nonpositive = peak <= 0
    rise, width = left[:, 2]-left[:, 0], right[:, 1]-left[:, 1]
    rise_missing, width_missing = ~np.isfinite(rise), ~np.isfinite(width)
    pre_rise = (~nonpositive & rise_missing &
        ((~np.isfinite(left[:, 0]) & (y[:, 0] > .1*peak)) |
         (~np.isfinite(left[:, 2]) & (y[:, 0] > .9*peak))))
    pre_width = ~nonpositive & width_missing & ~np.isfinite(left[:, 1]) & (y[:, 0] > .5*peak)
    post_width = ~nonpositive & width_missing & ~np.isfinite(right[:, 1]) & (y[:, -1] > .5*peak)
    result = dict(peak_voltage_V=peak, rise10_90_ns=rise, FWHM_ns=width,
        left_crossing10_ns=left[:, 0], left_crossing50_ns=left[:, 1], left_crossing90_ns=left[:, 2],
        right_crossing50_ns=right[:, 1], rise_undefined=rise_missing, FWHM_undefined=width_missing,
        temporal_nonpositive_peak=nonpositive, rise_prepeak_censored=pre_rise,
        rise_other_undefined=rise_missing & ~nonpositive & ~pre_rise,
        FWHM_prepeak_censored=pre_width, FWHM_postpeak_censored=post_width,
        FWHM_other_undefined=width_missing & ~nonpositive & ~pre_width & ~post_width,
        selected_peak_below_global_crop_max=peak < y.max(1))
    require(np.array_equal(rise_missing, nonpositive | pre_rise | result['rise_other_undefined']),
            'Rise missingness causes do not cover all rows')
    require(np.array_equal(width_missing, nonpositive | pre_width | post_width | result['FWHM_other_undefined']),
            'FWHM missingness causes do not cover all rows')
    return result

def scalar_moments(x):
    valid=np.isfinite(x);count=valid.sum(1);s=np.where(valid,x,0.).sum(1)
    mean=np.full_like(s,np.nan);np.divide(s,count,out=mean,where=count>0)
    ss=np.where(valid,(x-mean[:,None])**2,0.).sum(1)
    var=np.full_like(s,np.nan);np.divide(ss,count-1,out=var,where=count>1)
    se=np.sqrt(var/np.maximum(count,1))
    p,_,k=x.shape;cov=np.full((p,k,k),np.nan);joint=np.zeros((p,k,k),np.int64)
    for i in range(k):
        for j in range(i,k):
            pair=valid[:,:,i]&valid[:,:,j];n=pair.sum(1)
            mi=np.zeros(p);mj=np.zeros(p)
            np.divide(np.where(pair,x[:,:,i],0.).sum(1),n,out=mi,where=n>0)
            np.divide(np.where(pair,x[:,:,j],0.).sum(1),n,out=mj,where=n>0)
            q=np.where(pair,(x[:,:,i]-mi[:,None])*(x[:,:,j]-mj[:,None]),0.).sum(1)
            v=np.full(p,np.nan);np.divide(q,n-1,out=v,where=n>1)
            cov[:,i,j]=cov[:,j,i]=v;joint[:,i,j]=joint[:,j,i]=n
    return dict(mean=mean,sample_variance=var,MonteCarlo_SE=se,defined_draws=count,undefined_draws=x.shape[1]-count,
        sample_covariance=cov,jointly_defined_draws=joint)
