"""Independent algebra checks for the N01 review.

Uses artificial arrays ONLY. Does not load N01 weights, fit experimental data,
train a detector model, access the network, or verify the repository's metrics.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch


def preadc_moments(A: np.ndarray, B: np.ndarray, v: np.ndarray,
                   local_cov: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Moments of exp(v'z - ||v||^2/2) A + B z[1:] + independent noise.

    A: (T,), B: (T, K), v: (K+1,). local_cov: (T,T), zero-mean noise.
    Does NOT describe post-ADC or peak-selected/cropped moments.
    """
    A, B, v = [np.asarray(x, dtype=np.float64) for x in (A, B, v)]
    if A.ndim != 1 or B.ndim != 2 or B.shape[0] != len(A) or v.shape != (B.shape[1]+1,):
        raise ValueError('Expected A[T], B[T,K], v[K+1].')
    if not all(np.isfinite(x).all() for x in (A, B, v)):
        raise ValueError('Inputs must be finite.')
    cross = B @ v[1:]
    cov = (np.expm1(v @ v) * np.outer(A, A) + B @ B.T
           + np.outer(A, cross) + np.outer(cross, A))
    if local_cov is not None:
        local_cov = np.asarray(local_cov, dtype=np.float64)
        if local_cov.shape != (len(A), len(A)) or not np.isfinite(local_cov).all():
            raise ValueError('Invalid local covariance.')
        cov += local_cov
    return A.copy(), cov


def adam_clip_check() -> dict:
    g = torch.tensor([100., 10., -.1], dtype=torch.float64)
    outputs = []
    for do_clip in (False, True):
        p = torch.nn.Parameter(torch.zeros_like(g))
        opt = torch.optim.AdamW([p], lr=1e-4, weight_decay=0., eps=1e-8)
        p.grad = g.clone()
        if do_clip:
            torch.nn.utils.clip_grad_norm_([p], 5.)
        grad_norm = float(p.grad.norm())
        opt.step()
        outputs.append({'clipped': do_clip, 'gradient_norm': grad_norm,
                        'parameter_step_norm': float(p.detach().norm()),
                        'parameter_step': p.detach().tolist()})
    ratio = outputs[1]['parameter_step_norm']/outputs[0]['parameter_step_norm']
    assert ratio > .99999, ratio
    return {'case': 'Artificial fresh AdamW first update, not N01 replay',
            'runs': outputs, 'clipped_to_unclipped_step_ratio': ratio,
            'interpretation': 'Gradient norm reduction need not reduce Adam step by the same factor.'}


def moment_check() -> dict:
    rng = np.random.default_rng(20261017)
    T, K, N = 24, 8, 300000
    t = np.arange(T)
    A = .15*np.exp(-.5*((t-7)/3.5)**2)
    B = rng.normal(0, .002, size=(T, K)); B -= B.mean(axis=0, keepdims=True)
    v = np.array([.08, .10, -.03, .04, .02, -.02, .015, .01, -.025])
    R = rng.normal(0, .0002, size=(T, 4)); R -= R.mean(axis=0, keepdims=True)
    mu, cov = preadc_moments(A, B, v, R@R.T)
    total = np.zeros(T); outer_sum = np.zeros((T,T)); count = 0
    for _ in range(N//10000):
        z = rng.normal(size=(10000,K+1)); w = rng.normal(size=(10000,4))
        alpha = np.exp(z@v-.5*(v@v))
        samples = alpha[:,None]*A + z[:,1:]@B.T + w@R.T
        total += samples.sum(0); outer_sum += samples.T@samples; count += len(samples)
    mean_emp = total/count
    cov_emp = (outer_sum-count*np.outer(mean_emp,mean_emp))/(count-1)
    rel = float(np.linalg.norm(cov_emp-cov)/np.linalg.norm(cov))
    mean_zmax = float(np.max(np.abs(mean_emp-mu)/np.sqrt(np.diag(cov)/count)))
    cov_without_cross = np.expm1(v@v)*np.outer(A,A)+B@B.T+R@R.T
    omission_error = float(np.linalg.norm(cov_without_cross-cov)/np.linalg.norm(cov))
    qvar = float(A.sum()**2*np.expm1(v@v))
    assert rel < .02, rel
    assert mean_zmax < 6., mean_zmax
    assert np.allclose(np.ones(T)@cov@np.ones(T),qvar,rtol=1e-11,atol=1e-14)
    return {'case':'Artificial correlated lognormal/linear-Gaussian preADC signal',
            'sample_count':count, 'coordinates':T,
            'covariance_relative_MC_error':rel,
            'mean_max_abs_standard_errors':mean_zmax,
            'relative_covariance_error_when_cross_terms_omitted':omission_error,
            'canonical_charge_variance_analytic':qvar,
            'scope':'No inference about the fitted N01 coefficients or hard observation covariance.'}


def reported_number_arithmetic() -> dict:
    lsb = 1/16383
    return {'scope':'Arithmetic of rounded published numbers, not an independent metric replay',
            'source_commit':'20a8c45ddb13ff6b6c4279a02a8a1eaeb7018ddf',
            'initial_G0_wave_relative_worsening_percent':100*(.225999/.153839-1),
            'final036_wave_ratio_to_A':.846729/.153839,
            'first_step_signal_RMS_mV':72.652*lsb*1000,
            'first_step_max_signal_change_mV':1521.545*lsb*1000,
            'clipped_step_fraction':1998/2000,
            'first_step_peak_change_fraction':32/128}

if __name__ == '__main__':
    torch.set_num_threads(2)
    results = {'original_data_used':False, 'N01_checkpoints_loaded':False,
               'detector_training_performed':False,
               'adam':adam_clip_check(), 'moments':moment_check(),
               'published_numbers':reported_number_arithmetic()}
    target=Path(__file__).with_name('verification_results.json')
    target.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(results,ensure_ascii=False,indent=2))
