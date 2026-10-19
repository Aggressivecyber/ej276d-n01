"""Synthetic coupling demonstrations, not an EJ-276D model replay."""
import json, math
from pathlib import Path
import numpy as np
from scipy.special import ndtr

rng = np.random.default_rng(20261019)
n = 400_000
z0, z1 = rng.standard_normal((2, n))
rows=[]
# Index 0 is the earlier sample, index 1 is the higher noiseless peak.
# Voltage expressed in units of one ADC LSB. Same per-coordinate RMS in all rows.
for rho in (0.0, 0.9, -0.5):
    early=z0
    late=rho*z0 + math.sqrt(1-rho*rho)*z1
    y0=99.25+early
    y1=100.25+late
    quant0=np.floor(y0+0.5)
    quant1=np.floor(y1+0.5)
    rows.append(dict(rho=rho,noise_RMS_each_LSB=1.0,mean_gap_LSB=1.0,
        unquantized_early_win=float(np.mean(y0>y1)),
        unquantized_theory=float(ndtr(-1.0/math.sqrt(2-2*rho))),
        quantized_first_argmax_early_win=float(np.mean(quant0>=quant1)),
        code_tie_rate=float(np.mean(quant0==quant1))))
# Noncommutation of taking the peak and taking the expectation.
base=np.array([.8,1.0,.8])
noise=.2*rng.standard_normal((n,3))
a=base+noise
selected=a.argmax(1)
selection=dict(mean_of_selected_peak=float(a.max(1).mean()),
               selected_peak_of_mean=float(a.mean(0).max()),
               mean_noise_at_selected_peak=float(noise[np.arange(n),selected].mean()))
a_ref=.15554293597289215
lsb=1/16383
result=dict(synthetic_only=True,seed=20261019,draws=n,peak_competition=rows,
            selection_bias=selection,S_margin_pre_V=1e-8*a_ref,
            ADC_step_V=lsb,S_margin_pre_over_LSB=1e-8*a_ref/lsb,
            S_margin_post_over_LSB=1e-10*a_ref/lsb)
Path(__file__).with_name('RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
