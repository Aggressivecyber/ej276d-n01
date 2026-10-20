from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,shutil
import numpy as np

root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
out=here/'terminal_01';run=out/'collected/run_01'
p=json.loads((root/'configs/joint_response_01_original_draw_precision_20261020.json').read_bytes())
manifest=json.loads((here/'SOURCE_MANIFEST.json').read_bytes())
remote=json.loads((out/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
stop=json.loads((here/'RUNTIME_STOP_RECEIPT.json').read_bytes())
assert remote['native_exit']==stop['exit_receipt'] and remote['native_exit']['exit_code']==-15
assert not remote['processes'] and not remote['complete_present'] and not remote['result_present']
assert stop['identity_verified'] and stop['signal']=='SIGTERM'
local_sources=0
for name,record in manifest['files'].items():
    if (root/name).is_file():
        raw=(root/name).read_bytes()
        assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],name
        local_sources+=1
assert remote['source_files_verified']==len(manifest['files'])==805
for name,digest in p['source_pins'].items():assert manifest['files'][name]['sha256']==digest,name
for name,record in remote['outputs'].items():
    raw=(out/'collected'/name).read_bytes()
    assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],name
raw=(out/'terminal.tar').read_bytes()
assert len(raw)==remote['archive_bytes'] and hashlib.sha256(raw).hexdigest()==remote['archive_sha256']
old=root/'receipts/joint_response_01_20261020/multibank_m0_01/terminal_01/run_01'
zero,old_zero=np.load(run/'M0_ZERO_IDENTITY.npz'),np.load(old/'M0_ZERO_IDENTITY.npz')
for key in zero.files:np.testing.assert_array_equal(zero[key],old_zero[key])
a=np.load(run/'seed0_MULTIBANK_M0.npz',allow_pickle=False)
low=np.load(old/'seed0_MULTIBANK_M0.npz',allow_pickle=False)
report=json.loads((run/'seed0_MULTIBANK_M0.json').read_bytes())
progress=json.loads((run/'PROGRESS.json').read_bytes())
k=report['FD_columns_completed']
assert k==progress['column']==progress['execution_counts']['finite_difference_columns']==9
assert not report['trials'] and report['current_solve'] is None and report['selected_trial'] is None
assert not (run/'seed1_MULTIBANK_M0.npz').exists()
assert all(np.isfinite(a[key]).all() for key in a.files)
assert (a['theta']==0).all()
for label in p['role_seed_bases']:
    for suffix in ('source_parents','real_rows','cholesky'):
        np.testing.assert_array_equal(a[label+'_'+suffix],low[label+'_'+suffix])
    np.testing.assert_allclose(a[label+'_replayed_low_residual'],low[label+'_baseline_residual'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a[label+'_replayed_low_native'],low[label+'_baseline_native'],rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(a[label+'_baseline_native'],report['baseline_native'][label])
    vector=a[label+'_baseline_residual'];L=a[label+'_cholesky'];variance=np.square(L).sum(1)
    for ih,h in enumerate(('fast','slow','common_level')):
        white=vector[ih*272:(ih+1)*272]*np.sqrt(3);raw=L@white
        risk=report['baseline_risks'][label][h]
        np.testing.assert_allclose(risk['joint_whitened'],np.mean(white**2),rtol=1e-11,atol=1e-12)
        for ib,block in enumerate(('wave','charge_morphology','peak_codes','baseline','joint')):
            take=slice(56*ib,56*(ib+1)) if ib<4 else slice(224,272)
            np.testing.assert_allclose(risk[block],np.mean(raw[take]**2/variance[take]),rtol=1e-11,atol=1e-12)
theta=a['theta'];plus=np.minimum(a['upper'][:,None],theta[:,None]+p['finite_difference_step']*np.eye(54))[:,:k]
minus=np.maximum(a['lower'][:,None],theta[:,None]-p['finite_difference_step']*np.eye(54))[:,:k]
np.testing.assert_array_equal(a['FD_plus_theta'],plus);np.testing.assert_array_equal(a['FD_minus_theta'],minus)
assert a['FD_plus_residual'].shape==a['FD_minus_residual'].shape==(2448,9)
assert a['FD_plus_native'].shape==a['FD_minus_native'].shape==(27,9)
spans=np.diag(plus)-np.diag(minus)
assert np.isfinite((a['FD_plus_residual']-a['FD_minus_residual'])/spans).all()
c=progress['execution_counts'];assert c['joint_response_calls']==2+10+9*6==66
assert c['joint_response_rows']==256+5*4096+59*32768==1954048
intervals={key:dict(lower=value,upper=value) for key,value in c.items()}
for key,additional in dict(finite_difference_columns=9,FIT_bank_finite_difference_columns=27,
    joint_response_calls=54,joint_response_rows=54*32768,generated_observation_calls=162,
    generated_observation_rows=162*32768,native_component_rows=162*36864).items():intervals[key]['upper']+=additional
now=datetime.now(timezone.utc).isoformat()
review=dict(status='PASS_STOPPED_ORIGINAL_PRECISION_PREFIX_AUDIT',UTC=now,
    native_exit=remote['native_exit'],source_files_verified=805,source_pins=len(p['source_pins']),local_source_hashes=local_sources,
    outputs_verified=len(remote['outputs']),output_bytes=sum(v['bytes'] for v in remote['outputs'].values()),
    zero_signal_rows_reverified=256,low_draw_centers_reverified=5,high_center_risk_views_recomputed=15,
    complete_FD_columns_saved=9,FD_parameter_arrays_verified=True,partial_source_parent_indices_preserved=True,
    execution_counts_saved=c,actual_execution_intervals=intervals,committed_parameter_updates=0,
    stopped_for_runtime_feasibility=True,original_budget_extended=False,scientific_failure_not_assessed=True,scientific_acceptance=False,
    uncertainty='After the nine-column snapshot and before SIGTERM, zero to nine additional columns may have completed; a final checkpoint boundary can be interrupted before persistence. Full waveform/native replay was not performed locally.')
(out/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8',newline='\n')
(out/'before_closure').mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(root/name,out/'before_closure'/name)
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((root/name).read_bytes())
    if name=='jobs.json':
        assert len(d['jobs'])==131 and d['jobs'][-1]['job_id']==p['job_id']
        d['jobs'][-1].update(status='stopped_runtime_feasibility_prefix_audited',native_exit_code=-15,
            ended_UTC=remote['native_exit']['ended_utc'],science_pid=None,supervisor_pid=None,
            complete_FD_columns_saved=9,actual_execution_intervals=intervals,
            terminal_review=(out/'TERMINAL_REVIEW.json').relative_to(root).as_posix(),scientific_failure_not_assessed=True)
    d.update(status='JOINT_RESPONSE_01_original_draw_precision_runtime_repair',active_scientific_jobs=0,
        active_job_id=None,active_run_id=None,training_pid=None,supervisor_pid=None,pending_job_id=None,
        current_stage='4_original_generator_numerical_runtime_repair',active_work_package='4_original_generator_numerical_runtime_repair',
        updated_UTC=now,execution_state_asof_UTC=remote['UTC'],scientific_acceptance=False,
        current_action='Preserved nine-column prefix after measured runtime exceeded the registered feasibility estimate.',
        next_action='Register exact 8:1 native W1 numerical repair with frozen same data, random streams, model and scientific gates.',
        followup_status='NO_PENDING_WAIT',followup_automation_id=None,followup_scheduled_local=None,followup_next_run_local=None,next_wake_local=None)
    d['joint_response_01'].update(original_draw_precision_status='stopped_runtime_feasibility_prefix_audited',
        original_draw_precision_terminal_review=(out/'TERMINAL_REVIEW.json').relative_to(root).as_posix(),
        original_draw_precision_actual_execution_intervals=intervals)
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
print(json.dumps({k:v for k,v in review.items() if k not in ('execution_counts_saved','actual_execution_intervals')}))
