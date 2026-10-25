"""Audit saved job248 trace only; no model execution or endpoint selection."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import sys

q = Path(__file__).resolve().parent
root = Path(r'E:\GAN\research\N01')
out = Path(sys.argv[1]) if len(sys.argv)>1 else root/'deliverables/finite_native_goal01_248_20261024'
assert out.is_absolute() and out.drive.upper() in ['E:', 'D:']
out.mkdir(exist_ok=False)
os.environ['MPLCONFIGDIR'] = str(q/'mpl_cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

state = json.loads((q/'REMOTE_STATE.json').read_bytes())
source_audit = json.loads((q/'REMOTE_HASH_AND_SOURCE_AUDIT.json').read_bytes())
assert source_audit['frozen_source_mismatches'] == []
assert source_audit['N01_project_processes'] == []
copied = []
for record in source_audit['remote_terminal_files']:
    path = q/'terminal'/record['path']
    data = path.read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    copied.append(dict(path=str(path),bytes=len(data),sha256=record['sha256']))
failure = json.loads((q/'terminal/P1/FAILED.json').read_bytes())
exit_record = json.loads((q/'terminal/control_P1/exit_receipt.json').read_bytes())
counts = failure['execution_counts']
assert exit_record['exit_code'] == 1 and not state['science_PID_exists']
assert failure['error_type'] == 'AssertionError'
for name in ['RESULT.json','COMPLETE.json','POINT_FROZEN.json','FINITE_NATIVE_POINT.npz','FIT_POINT_HARD_READOUT.npz']:
    assert not (q/'terminal/P1'/name).exists()
config = root/'configs/finite_native_goal01_P1_20261024.json'
assert hashlib.sha256(config.read_bytes()).hexdigest() == state['config_sha256']
p = json.loads(config.read_bytes())
for name,record in p['new_source_pins'].items():
    local = root/'receipts/finite_native_goal01_20261024/P1/source'/Path(name).name
    assert hashlib.sha256(local.read_bytes()).hexdigest() == record['sha256']

rows = [json.loads(line) for line in (q/'terminal/P1/POINTS.jsonl').read_text().splitlines()]
assert len(rows) == 768
np.testing.assert_array_equal([row['point_index'] for row in rows],np.arange(1,769))
coefficients = np.array([row['coefficients'] for row in rows])
assert coefficients.shape == (768,163) and np.isfinite(coefficients).all()
l1 = abs(coefficients).sum(1)
assert float(l1.max()) <= 1+1e-12
wave = np.array([row['values']['wave'] for row in rows])
qpsd = np.array([row['values']['QPSD'] for row in rows])
violation = np.array([row['max_constraint'] for row in rows])
rms = np.array([row['mean_relative_RMS_max'] for row in rows])
assert np.isfinite(np.r_[wave,qpsd,violation,rms]).all()
m0 = source_audit['original_reference_values']
gain = 1-wave/m0['wave']
qchange = qpsd/m0['QPSD']-1
recorded_feasible = (violation<=1e-10)&(qchange<=0)&(rms<=.01+1e-12)
assert recorded_feasible.any()
assert counts['new_finite_point_evaluations'] == counts['completed_finite_point_evaluations'] == counts['distinct_solver_points'] == len(rows)
assert counts['cached_observation_rows'] == 103440*len(rows)
assert counts['mean_head_parent_rows'] == 6465*len(rows)
assert counts['parameter_fits'] == 1 and counts['candidate_selections'] == 0
assert counts['objective_interface_calls']+counts['constraint_interface_calls'] == counts['memo_reuses']+len(rows)+1
assert counts['inflight_cached_observation_rows_upper'] == counts['inflight_mean_head_parent_rows_upper'] == 0
for key in ['original_backbone_calls','new_stochastic_draws','optimizer_updates','REPORT_rows_used_in_metrics']:
    assert counts[key] == 0

rounded = np.round(coefficients,14)
rounded[rounded==0] = 0
groups = {}
near_differences = []
metric_differences = []
for index,value in enumerate(rounded):
    key = value.tobytes()
    if key in groups:
        previous = groups[key]
        near_differences.append(float(abs(coefficients[index]-coefficients[previous]).max()))
        metric_differences.append(float(max(abs(wave[index]-wave[previous]),abs(qpsd[index]-qpsd[previous]))))
    else:
        groups[key] = index

metrics = []
for index,row in enumerate(rows):
    metrics.append(dict(point_index=row['point_index'],UTC=row['UTC'],elapsed_seconds=row['elapsed_seconds'],point_seconds=row['point_seconds'],
        wave=row['values']['wave'],wave_improvement_percent=float(100*gain[index]),QPSD=row['values']['QPSD'],QPSD_change_percent=float(100*qchange[index]),
        mean_RMS_percent=float(100*rms[index]),max_recorded_constraint=float(violation[index]),recorded_FIT_constraints_pass=bool(recorded_feasible[index]),
        mapped_L1=float(l1[index]),objective_calls=row['execution_counts']['objective_interface_calls'],constraint_calls=row['execution_counts']['constraint_interface_calls'],
        cached_observation_rows=row['execution_counts']['cached_observation_rows']))
with (out/'ALL_768_TRIAL_METRICS.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    writer = csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)

audit = dict(UTC=datetime.now(timezone.utc).isoformat(),status='CLOSED_INCOMPLETE_DISTINCT_POINT_BUDGET_BEFORE_SOLVER_RETURN',job_number=248,
    native_exit_code=1,elapsed_seconds=failure['elapsed_seconds'],peak_RSS_bytes=exit_record['peak_rss_bytes'],execution_counts=counts,
    original_M0=dict(wave=m0['wave'],QPSD=m0['QPSD']),verified_trial_rows=len(rows),fit_stage_feasible_from_recorded_scalars=int(recorded_feasible.sum()),
    wave_2percent_points=int((gain>=.02).sum()),wave2percent_and_recorded_FIT_feasible_points=int(((gain>=.02)&recorded_feasible).sum()),
    max_observed_wave_improvement_percent=float(100*gain.max()),max_recorded_feasible_wave_improvement_percent=float(100*gain[recorded_feasible].max()),
    target_percent=2.,remaining_gap_at_recorded_feasible_max_percentage_points=float(2-100*gain[recorded_feasible].max()),
    last_trace_point_wave_improvement_percent=float(100*gain[-1]),max_mean_RMS_percent=float(100*rms.max()),max_mapped_L1=float(l1.max()),
    near_equal_diagnostic=dict(round_decimal_places=14,diagnostic_groups=len(groups),repeated_points=len(near_differences),max_coefficient_difference=float(max(near_differences)),max_wave_QPSD_difference=float(max(metric_differences)),counts_deducted=0,used_for_endpoint_selection=False),
    frozen_source_files_verified=source_audit['frozen_source_count'],config_sha256=state['config_sha256'],terminal_files_exact=copied,
    solver_returned_endpoint=False,candidate_selected=False,P2_executed=False,P3_executed=False,scientific_acceptance=False,restricted_average_response_acceptance=False,
    new_N01_model_executions_in_audit=0,new_scientific_parameter_fits_in_audit=0,
    independent_readback_scope='Exact artifact collection, frozen source/config verification, all saved coefficient/metric/counter rows. Individual18risk/14flag and full8776/26CDF values for each trial are not independently regenerated in this audit.',
    failure_meaning='Implementation stopped before solver return. No sampled point reached2percent; this is not a global infeasibility result.',
    N01_project_processes=source_audit['N01_project_processes'],GPU_after_termination=state['GPU'],NAS_access=False)
(out/'TRACE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axes = plt.subplots(2,2,figsize=(13.3,8.2),constrained_layout=True)
index = np.arange(1,769)
colors = np.where(recorded_feasible,'#146b85','#b0b7bd')
axes[0,0].scatter(index,100*gain,c=colors,s=9,alpha=.8,rasterized=True)
axes[0,0].axhline(2.,color='#bc353a',ls='--',lw=1.4,label='2% entry requirement')
axes[0,0].set(xlabel='Actual joint evaluation index',ylabel='Wave distance improvement (%)',title='A. No trial reached the entry requirement',ylim=(100*gain.min()-.03,2.05))
axes[0,0].legend(loc='lower right',frameon=False)
axes[0,1].scatter(index,100*qchange,c=colors,s=9,alpha=.8,rasterized=True)
axes[0,1].axhline(0,color='#bc353a',ls='--',lw=1.4)
axes[0,1].set(xlabel='Actual joint evaluation index',ylabel='QPSD distance change (%)',title='B. Positive values fail the QPSD condition')
axes[1,0].semilogy(index,index,color='#146b85',label='Actual joint evaluations')
axes[1,0].semilogy(index,[row['objective_calls'] for row in metrics],color='#ca7b1a',label='Objective interface calls')
axes[1,0].semilogy(index,np.maximum(1,[row['constraint_calls'] for row in metrics]),color='#56783f',label='Constraint interface calls')
axes[1,0].scatter([768],[counts['constraint_interface_calls']],color='#56783f',s=25)
axes[1,0].set(xlabel='Actual joint evaluation index',ylabel='Cumulative count (log scale)',title='C. Joint budget filled before solver return')
axes[1,0].legend(frameon=False,loc='lower right',fontsize=9)
axes[1,1].plot(index,100*rms,color='#146b85',lw=.9)
axes[1,1].axhline(1,color='#bc353a',ls='--',lw=1.4,label='1% actual mean RMS limit')
axes[1,1].set(xlabel='Actual joint evaluation index',ylabel='Maximum parent mean RMS (%)',title='D. Mean correction stayed within its bound',ylim=(0,1.05))
axes[1,1].legend(frameon=False,loc='upper right',bbox_to_anchor=(1.,.88),fontsize=9)
for ax in axes.flat:ax.grid(axis='y',alpha=.18)
fig.suptitle('N01 job248 | 768 completed FIT evaluations, no returned endpoint\nBlue points satisfy recorded FIT constraints; gray points do not. No point is adopted.',fontsize=13)
fig.savefig(out/'FIT_TRACE.png',dpi=170)
fig.savefig(out/'FIT_TRACE.svg')
plt.close(fig)
print(json.dumps({key:audit[key] for key in ['status','verified_trial_rows','fit_stage_feasible_from_recorded_scalars','max_recorded_feasible_wave_improvement_percent','wave_2percent_points','near_equal_diagnostic']},ensure_ascii=False))
