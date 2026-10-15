"""Arithmetic and synthetic checks, not a rerun of N01 or private waveforms.
Source constants are from commit 45a8a15f2a3ac899e14e9fb2db6521ccdd55b2e3.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

N = 435
threshold = math.ceil(0.95*N)
teacher_support = {'fast':407,'slow':407,'common_level':408}
student_support = {'fast':404,'slow':405,'common_level':405}
common_support = {'fast':402,'slow':403,'common_level':404}

A = 0.15432927983704245
M0 = 0.15012505860962605
Q1 = 0.15011008209012994
lsb = 1/16383
Tdc = 4.506386193005153e-7

# Independent implementation of the same nearest left/right crossing convention.
def crossing(row: np.ndarray, frac: float, side: str) -> float:
    threshold = frac*row[16]
    pairs = [(j-1,j) for j in range(16,0,-1)] if side=='left' else [(j,j+1) for j in range(16,255)]
    for i,j in pairs:
        lo,hi = row[i],row[j]
        brackets = lo <= threshold <= hi if side=='left' else hi <= threshold <= lo
        if brackets:
            return float(i + (threshold-lo)/(hi-lo)) if hi != lo else float(j if side=='left' else i)
    return float('nan')

t=np.arange(256,dtype=float)
y=np.exp(-0.5*((t-16)/10)**2)
t10,t90=crossing(y,.1,'left'),crossing(y,.9,'left')
assert not np.isfinite(t10) and np.isfinite(t90)
assert y[0] > .1*y[16]
# The same perfectly matched synthetic population would still fail raw support.
assert max(teacher_support.values()) < threshold

# A true common DC shift cancels before ADC under prefix-mean subtraction.
tc=np.arange(384,dtype=float)
carrier=.13*np.exp(-.5*((tc-126)/8)**2)+.0007*np.sin(tc*.4)
shift=15e-6
center=lambda x:x-x[:100].mean()
res=float(np.max(np.abs(center(carrier+shift)-center(carrier))))
assert res < 1e-15

result={
 'scope':'Derived arithmetic and synthetic counterexamples only; no NN forward, no private data, no retraining.',
 'commit':'45a8a15f2a3ac899e14e9fb2db6521ccdd55b2e3',
 'support':{
   'original_parents':N,'minimum_required':threshold,
   'teacher_count':teacher_support,'student_count':student_support,'common_count':common_support,
   'teacher_fraction':{k:v/N for k,v in teacher_support.items()},
   'common_fraction':{k:v/N for k,v in common_support.items()},
   'teacher_support_ceiling_fails':all(v<threshold for v in teacher_support.values())},
 'effect_sizes':{
   'M0_wave':M0,'Q1_wave':Q1,'absolute_improvement':M0-Q1,
   'relative_improvement_percent':100*(M0-Q1)/M0,
   'Q1_over_A_improvement_percent':100*(A-Q1)/A,
   'real_fit_a_seed_ratio':5.860415918e-5/1.601902908e-5,
   'real_fit_b_seed_ratio':4.934797363e-6/3.214281573e-6},
 'approximate_units_from_rounded_D_over_T':{
   'adc_LSB_uV':lsb*1e6,
   'baseline_T_uV':Tdc*1e6,
   'baseline_distance_uV_approx':33.2674*Tdc*1e6,
   'baseline_distance_LSB_approx':33.2674*Tdc/lsb,
   'peak_distance_mV_approx':6.69*.005499862543299458*1e3,
   'rise_distance_software_ns_approx':7.4*.21159366879623287,
   'FWHM_distance_software_ns_approx':4.8*.5129398878764326},
 'synthetic_censoring':{
   'waveform':'positive Gaussian centered at sample16, sigma10 samples',
   't10_defined_in_window':False,
   't90_in_window':t90,
   't10_exact_on_extended_domain':16-10*math.sqrt(2*math.log(10)),
   'true_rise_on_extended_domain':10*(math.sqrt(2*math.log(10))-math.sqrt(2*math.log(1/.9))),
   'prepeak_censored':True,
   'interpretation':'Missing 10% crossing is not itself an invalid waveform; it is a coarsened observation.'},
 'common_DC_preADC_invariance_absmax_V':res,
 'caveats':[
   'D/T-derived values above are distribution distances, not necessarily signed mean biases.',
   'Sub-LSB population differences can be statistically meaningful. Quantization is not a hard lower bound on estimating averages.',
   'ADC phase, clipping, time-varying drift and prefix-only changes do not share exact common-DC invariance.',
   'A single finite synthetic check is not a scientific evaluation of N01.'
 ]
}
path=Path(__file__).with_name('review_calculations.json')
path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
