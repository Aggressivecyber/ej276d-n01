"""Close job252 from saved records, without choosing any search trial."""
from pathlib import Path
import csv, datetime, json, os, statistics
import numpy as np

r=Path('E:/GAN/research/N01')
q=Path(__file__).parent
d=r/'deliverables/finite_native_goal01_252_20261025'
d.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(q/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

state=json.loads((q/'REMOTE_STATE.json').read_text())
result=state['json']['P1/RESULT.json'];receipt=state['json']['control_P1/exit_receipt.json']
p=json.loads((r/'configs/finite_native_goal01_recovery01_P1_20261025.json').read_text())
base=state['reference_result']['native']['original']['values']
rows=[json.loads(s) for s in (q/'terminal/P1/POINTS.jsonl').read_text().splitlines()]
assert len(rows)==768 and [x['point_index'] for x in rows]==list(range(1,769))
assert receipt['exit_code']==0 and result['job_number']==252 and result['solver_status']==5 and result['solver_nfev']==768
assert state['verified_pin_count']==492 and not state['pin_mismatches'] and not state['project_processes'] and not state['GPU_processes'].strip()
assert 'P1/FAILED.json' not in state['files'] and state['json']['P1/COMPLETE.json']['status']=='COMPLETE'
for key,limit in p['operation_budget'].items():assert result['execution_counts'][key]<=limit,key
assert result['execution_counts']['new_finite_point_evaluations']==result['execution_counts']['completed_finite_point_evaluations']==769
with np.load(q/'terminal/P1/FINITE_NATIVE_POINT.npz',allow_pickle=False) as z:point=z['primal'].copy()
matrix=np.array([x['coefficients'] for x in rows])
assert matrix.shape==(768,163) and max(np.abs(matrix).sum(1))<=1+1e-12
assert abs(np.abs(point).sum()-result['mapped_L1'])<1e-14
metrics=[dict(point_index=x['point_index'],UTC=x['UTC'],wave=x['values']['wave'],wave_improvement_percent=100*(1-x['values']['wave']/base['wave']),QPSD=x['values']['QPSD'],QPSD_relative_change_percent=100*(x['values']['QPSD']/base['QPSD']-1),mean_RMS_percent=100*x['mean_relative_RMS_max'],max_constraint=x['max_constraint'],recorded_fit_constraints_pass=x['max_constraint']<=1e-10,mapped_L1=float(np.abs(x['coefficients']).sum()),point_seconds=x['point_seconds'],elapsed_seconds=x['elapsed_seconds']) for x in rows]
with (d/'ALL_768_SEARCH_TRIALS.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
feasible=[x for x in metrics if x['recorded_fit_constraints_pass']]
endpoint=result['final_record']['values'];changes={k:endpoint[k]/base[k]-1 for k in ['wave','QPSD']}
for key in changes:assert abs(changes[key]-result['actual_relative_changes'][key])<1e-14
assert not result['eligible_for_one_native_readback'] and changes['wave']>-.02 and changes['QPSD']<0
constraint_rows=[];origin=state['reference_result']['records']['original']
for bank in range(2):
    for kind,names,key in [('risk',p['risk_names'],'metrics_by_bank'),('flag',p['flag_names'],'flags')]:
        for name in names:
            old=origin[key][bank][name];new=result['final_record'][key][bank][name]
            difference=(new-old)/max(1.,abs(old)) if kind=='risk' else new-old
            constraint_rows.append(dict(bank=bank,kind=kind,name=name,M0=old,returned_point=new,normalized_difference=difference,pass_fit_guard=difference<=1e-10))
with (d/'RETURNED_POINT_18_RISKS_14_FLAGS.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(constraint_rows[0]));w.writeheader();w.writerows(constraint_rows)
audit=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='COMPLETE_FINITE_FIT_NO_2PERCENT_GAIN',native_exit_code=0,solver_status=5,solver_converged=False,solver_nfev=768,search_points=768,final_replays=1,execution_counts=result['execution_counts'],elapsed_seconds=result['elapsed_seconds'],peak_RSS_bytes=result['peak_RSS_bytes'],source_pin_count=492,source_or_config_mismatches=[],M0_values={k:base[k] for k in ['wave','QPSD']},returned_values=endpoint,returned_wave_improvement_percent=-100*changes['wave'],returned_QPSD_improvement_percent=-100*changes['QPSD'],target_wave=base['wave']*.98,remaining_absolute_wave_gap=endpoint['wave']-base['wave']*.98,remaining_percentage_point_gap=2+100*changes['wave'],returned_mean_RMS_percent=result['final_record']['mean_relative_RMS_max']*100,mapped_L1=result['mapped_L1'],all18risks_and14flags_pass=all(x['pass_fit_guard'] for x in constraint_rows),max_trial_wave_improvement_percent=max(x['wave_improvement_percent'] for x in metrics),max_recorded_feasible_wave_improvement_percent=max(x['wave_improvement_percent'] for x in feasible),feasible_trace_points=len(feasible),wave2percent_trial_count=sum(x['wave_improvement_percent']>=2 for x in metrics),exact_returned_point_trace_matches=np.flatnonzero((matrix==point).all(1)).tolist(),point_seconds_mean=statistics.mean(x['point_seconds'] for x in metrics),point_seconds_max=max(x['point_seconds'] for x in metrics),tail_last100=dict(feasible=sum(x['recorded_fit_constraints_pass'] for x in metrics[-100:]),max_wave_improvement_percent=max(x['wave_improvement_percent'] for x in metrics[-100:])),interpretation='Original2percentwave gate failed; all recorded FIT guards pass at returned point. Signed L1 boundary active; mean RMS and QPSD have slack. Solver exhausted768 evaluations without convergence. Neither proved optimum nor global infeasibility. Historical trial extrema are diagnostics only.',P2_executed=False,P3_executed=False,scientific_acceptance=False,restricted_average_response_acceptance=False,NAS_access=False)
(d/'TRACE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
x=np.array([m['point_index'] for m in metrics]);gain=np.array([m['wave_improvement_percent'] for m in metrics]);qc=np.array([m['QPSD_relative_change_percent'] for m in metrics]);ok=np.array([m['recorded_fit_constraints_pass'] for m in metrics])
fig,axs=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
axs[0,0].scatter(x[~ok],gain[~ok],s=8,c='#c5c9cc',label='Recorded FIT guard fail');axs[0,0].scatter(x[ok],gain[ok],s=8,c='#1864ab',label='Recorded FIT guard pass');axs[0,0].axhline(2,color='#c92a2a',ls='--',lw=1,label='2% entry');axs[0,0].scatter([769],[-100*changes['wave']],s=130,marker='*',c='#e67700',label='Sole returned point');axs[0,0].set(ylabel='Full wave improvement vs M0 (%)',title='Complete hard objective | 8,776 CDF columns');axs[0,0].legend(fontsize=8)
axs[0,1].scatter(gain,qc,s=9,c=np.where(ok,'#1864ab','#c5c9cc'));axs[0,1].axvline(2,c='#c92a2a',ls='--',lw=1);axs[0,1].axhline(0,c='#333',lw=.8);axs[0,1].scatter([-100*changes['wave']],[100*changes['QPSD']],marker='*',s=130,c='#e67700');axs[0,1].set(xlabel='Full wave improvement (%)',ylabel='QPSD distance change (%)',title='Wave and QPSD tradeoff')
axs[1,0].plot(x,[m['mean_RMS_percent'] for m in metrics],lw=.7,c='#1864ab');axs[1,0].axhline(1,c='#c92a2a',ls='--',lw=1);axs[1,0].set(ylabel='Max parent mean RMS (%)',title='Actual mean-change guard')
axs[1,1].plot(x,[m['max_constraint'] for m in metrics],lw=.7,c='#1864ab');axs[1,1].axhline(0,c='#333',lw=.8);axs[1,1].set(yscale='symlog',ylabel='Maximum normalized violation',title='34 recorded FIT constraints')
for ax in [axs[0,0],axs[1,0],axs[1,1]]:ax.set_xlabel('Search trial index')
axs[0,0].set_xlabel('Search trial index; returned-point replay at 769')
axs[1,1].set_yscale('symlog',linthresh=1e-5)
fig.suptitle('N01 job252 | Complete bounded fit; original 2% gate not met')
fig.savefig(d/'FIT_TRACE.png',dpi=160);fig.savefig(d/'FIT_TRACE.svg');plt.close(fig)
print(json.dumps({k:audit[k] for k in ['returned_wave_improvement_percent','returned_QPSD_improvement_percent','remaining_percentage_point_gap','returned_mean_RMS_percent','max_trial_wave_improvement_percent','max_recorded_feasible_wave_improvement_percent','feasible_trace_points','wave2percent_trial_count','exact_returned_point_trace_matches','tail_last100']},indent=2))
