"""Bounded NumPy readback of job252 saved endpoint; no model execution."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from datetime import datetime, timezone
import ast, hashlib, json, resource, sys, time
import numpy as np
from stability011_metrics import CDFCache, VIEWS, COORDINATE_VIEWS, LAGS, wave_views, q_features

root=Path('/home/rylan/projects/ej276d-n01/finite_native_goal01_recovery01_20261025')
out=root/'closure_readback_20261025'
began=time.monotonic()
p=json.loads((root/'source/configs/finite_native_goal01_recovery01_P1_20261025.json').read_bytes())
n=json.loads((root/'P1/RESULT.json').read_bytes())
reference=json.loads(Path(p['source_native_result']).read_bytes())
counts=Counter()
args=SimpleNamespace(out=out)

def progress(stage, **extra):
    assert time.monotonic()-began<240
    (out/'PROGRESS.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),stage=stage,seconds=time.monotonic()-began,execution_counts=dict(counts),**extra))+'\n')

with np.load(root/'P1/FINITE_NATIVE_POINT.npz',allow_pickle=False) as z:
    point={k:z[k].copy() for k in z.files}
with np.load(root/'P1/FIT_POINT_HARD_READOUT.npz',allow_pickle=False) as z:
    arrays={k:z[k].copy() for k in z.files}
with np.load(root/'P1/FIT_POINT_MEMORY.npz',allow_pickle=False) as z:
    means=z['candidate'].copy()
with np.load(p['source_memory'],allow_pickle=False) as z:
    saved={k:z[k].copy() for k in z.files}
with np.load(p['mean_tangent'],allow_pickle=False) as z:
    mean={k:z[k].copy() for k in ['decoded','raw_mean','logits','qbar','gate','source_FIT_rows','REAL_FIT_rows','original_mean','initial_head']}
with np.load(p['additive_head'],allow_pickle=False) as z:
    fields=z['direction_fields'].copy();radii=z['mean_radii'].copy();physical=z['coordinate_steps'].copy()
with np.load(p['metric_state'],allow_pickle=False) as z:
    state={k:z[k].copy() for k in z.files}
with np.load('/home/rylan/projects/ej276d-n01/full_population_hard_readout01_20261023/run_01/REAL_HARD_READOUT.npz',allow_pickle=False) as z:
    fit=point['REAL_FIT_rows'];real=z['window_adc_codes'][fit].astype(np.float64)/16383-.5-z['baseline_mean_V'][fit,None]
indices=arrays['global_generated_indices'];parents=(indices%52744)//8
np.testing.assert_array_equal(point['source_FIT_rows'],mean['source_FIT_rows'])
np.testing.assert_array_equal(point['REAL_FIT_rows'],mean['REAL_FIT_rows'])
np.testing.assert_array_equal(point['original_head'],mean['initial_head'])
np.testing.assert_array_equal(point['head_delta'],point['primal'][:161]*point['coordinate_steps'][:161])
np.testing.assert_array_equal(point['NN_coefficients'],point['primal'][161:]*point['coordinate_steps'][161:])
np.testing.assert_array_equal(point['theta'],point['original_theta'])
assert len(indices)==len(np.unique(indices))==103440 and len(fit)==18948
assert set(parents)==set(point['source_FIT_rows'])
for bank in range(2):
    select=indices//52744==bank
    np.testing.assert_array_equal(np.bincount(parents[select],minlength=6593)[point['source_FIT_rows']],np.full(6465,8))
with np.load(p['original_readout'],allow_pickle=False) as z:
    np.testing.assert_array_equal(arrays['prefix_adc_codes'],z['prefix_adc_codes'][indices])
    np.testing.assert_array_equal(arrays['baseline_mean_V'],z['baseline_mean_V'][indices])
    crop_changed=int(np.count_nonzero(arrays['crop_start']!=z['crop_start'][indices]))
    adc_changed=int(np.count_nonzero(arrays['window_adc_codes']!=z['window_adc_codes'][indices]))
flag_checks=[]
for bank in range(2):
    for name in p['flag_names']:
        value=int(arrays['flag_'+name][indices//52744==bank].sum())
        assert value==n['final_record']['flags'][bank][name]
        flag_checks.append(value<=reference['records']['original']['flags'][bank][name])
assert all(flag_checks) and len(flag_checks)==14

def risks(value,bank):
    residual=value-saved['target'];weights=saved['weights'];target=saved['target']
    values=[float(np.sum(weights[first:last]*residual[first:last]**2)) for first,last in [(0,112),(112,14448),(14448,28784),(28787,29059),(29059,35331)]]
    raw=saved['joint_cholesky']@residual[28787:29059];variance=np.sum(saved['joint_cholesky']**2,axis=1)
    values.extend(float(np.mean(raw[first:last]**2/variance[first:last])) for first,last in [(0,56),(112,168)])
    conditional=target[:112,None]*(value[112:14448].reshape(112,128)/value[:112,None]-target[112:14448].reshape(112,128)/target[:112,None])
    values.append(float(np.sum(weights[112:14448].reshape(112,128)*conditional**2)))
    local=np.mean((residual[29059:].reshape(112,56)/saved['wave_denominator'][:,None])**2,axis=1)
    names=[f'Q_wave_L{k}_{j:02d}' for k in [16,32,64] for j in range(k)]
    inherited=reference['records']['inherited']['metrics_by_bank'][bank]
    support=saved['wave_supported'];baseline=np.array([inherited[key] for key in names])
    values.append(float(np.mean(np.maximum(local[support]/(1.02*baseline[support])-1,0)**2)))
    return np.array(values)

assert means.shape==(2,35331)
risk_error=0.;constraints=[]
for bank in range(2):
    values=risks(means[bank],bank)
    old=reference['records']['original']['metrics_by_bank'][bank]
    for name,value in zip(p['risk_names'],values):
        risk_error=max(risk_error,abs(value-n['final_record']['metrics_by_bank'][bank][name]))
        constraints.append((value-old[name])/max(1.,abs(old[name])))
    constraints.extend(n['final_record']['flags'][bank][name]-reference['records']['original']['flags'][bank][name] for name in p['flag_names'])
assert risk_error<1e-9
tree=ast.parse((root/'source/scripts/finite_native_goal01.py').read_bytes())
method=next(x for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='FiniteNativeGoal01' for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='mean_delta')
exec(compile(ast.Module(body=[method],type_ignores=[]),'<frozen mean_delta arithmetic>','exec'))
obj=SimpleNamespace(mean=mean,steps=point['coordinate_steps'],fields=fields,radii=radii,additive_physical_steps=physical)
delta,rms=mean_delta(obj,point['primal'])
assert abs(rms-n['final_record']['mean_relative_RMS_max'])<1e-12
assert float(abs(point['primal']).sum())<=1+1e-12
del obj,mean,fields,radii,delta
wave=arrays['window_adc_codes'].astype(np.float64)/16383-.5-arrays['baseline_mean_V'][:,None]
tree=ast.parse((out/'evaluate_finite_native_goal01.py').read_bytes())
function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='independent_metrics')
exec(compile(ast.Module(body=[function],type_ignores=[]),'<independent NumPy CDF only>','exec'))
cdf=independent_metrics(wave,real,state,'returned252_saved')
assert counts['independent_wave_CDF_columns']==8776 and counts['independent_QPSD_CDF_columns']==26
errors={k:abs(cdf[k]-n['final_record']['values'][k]) for k in ['wave','QPSD']}
assert max(errors.values())<1e-9
constraints.extend([cdf['QPSD']/reference['native']['original']['values']['QPSD']-1,rms/.01-1])
assert np.max(abs(np.array(constraints)-n['actual_constraints']))<1e-9
assert 'torch' not in sys.modules
result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='PASS_SAVED_252_NUMERICAL_READING_GATE_NOT_MET',job_number=252,independent_CDF=cdf,CDF_abs_error=errors,max_risk_arithmetic_error=risk_error,actual_constraints=constraints,all18risk_and14flag_checks_pass=True,mean_relative_RMS_max=rms,mapped_L1=float(abs(point['primal']).sum()),prefix_and_baseline_exact=True,SOURCE_FIT_parents=6465,REAL_FIT_rows=18948,each_parent_per_bank_draws=8,REPORT_rows=0,ADC_cells_changed_vs_M0=adc_changed,crops_changed_vs_M0=crop_changed,seconds=time.monotonic()-began,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,execution_counts=dict(counts),parameter_fits=0,candidate_selections=0,N01_model_calls=0,new_stochastic_draws=0,P2_native_generation_executed=False,original036_native_guards_recomputed=False,scientific_acceptance=False,restricted_average_response_acceptance=False,NAS_access=False)
(out/'SAVED_ARRAY_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
