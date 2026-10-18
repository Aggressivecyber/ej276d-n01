from pathlib import Path
from datetime import datetime,timezone
import base64,copy,hashlib,json,sys
import numpy as np
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;sys.path[:0]=[str(r/'scripts'),str(r/'src')]
from verify_response071_diagnostics import verify_diagnostics071
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=read(w/'STARTUP_RAW.json');files=v['files'];p=read(r/'configs/response071_20261018.json');dispatch=read(w/'DISPATCH_RECEIPT.json')
assert v['boot_id']==dispatch['boot_id'] and 'run_01/FAILED.json' not in files and 'control_01/exit_receipt.json' not in files
python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
native=[x for x in v['processes'] if x['argv'][:3]==[python,'-B',dispatch['remote_root']+'/scripts/run_response071.py']];assert len(native)==1
assert files['run_01/STARTED.json']['pid']==native[0]['pid'] and files['run_01/STARTED.json']['config_sha256']==dispatch['config_sha256']==sha(r/'configs/response071_20261018.json')
count=files['run_01/PROGRESS.json']['execution_counts'];assert count['frozen_endpoint_head_replays']>=1 and count['full_native_hypothesis_evaluations']>=9 and count['diagnostic_decomposition_records']>=3
assert all(value<=p['operation_bounds'][key] for key,value in count.items())
assert count['new_candidate_settings']==count['AD_queries']==count['optimizer_steps']==count['DEV_archives_decoded']==0
assert files['control_01/resource_status.json']['rss_bytes']<12*1024**3 and v['current_run_bytes']<64*1024**2
expected=read(r/p['replay_file'])['reports']['0']
for case in ('M0','LEVEL069','SPLITWARP'):assert files['run_01/seed0_'+case+'.json']==expected[case],case
arrays={}
for name,entry in v['binary_files'].items():
    data=base64.b64decode(entry['base64']);assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
    path=w/Path(name).name
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
    arrays[path.name]=dict(np.load(path,allow_pickle=False))
assert len(arrays)==40
prior=r/'receipts/response070_20261018/wake_20261018_2043_01/collected/run_01'
old_names=[name for name in arrays if not name.endswith('_DECOMPOSITION_ROWS.npz')];assert len(old_names)==37
for name in old_names:
    before=dict(np.load(prior/name,allow_pickle=False));new=arrays[name];assert set(before)==set(new),name
    for key in new:assert np.array_equal(new[key],before[key],equal_nan=True),(name,key)
for name,value in files.items():
    if name.endswith('_PROFILE_DEFINITION.json') or name=='run_01/DEFINITIONS.json':assert value==read(prior/Path(name).name),name
diagnostic=verify_diagnostics071(r,arrays,files,0)
for name,digest in p['source_pins'].items():assert sha(r/name)==digest,name
manifest=read(w/'SOURCE_MANIFEST.json')
for name,meta in manifest['files'].items():assert (r/name).stat().st_size==meta['bytes'] and sha(r/name)==meta['sha256'],name
record=dict(status='CORRECTLY_STARTED_FROZEN070_NATIVE_PEAK_TAIL_DIAGNOSTIC',verified_UTC=v['UTC'],job_id=p['run_id'],boot_id=v['boot_id'],native_identity=native[0],
    supervisor_identity=next(x for x in v['processes'] if x['argv'][:3]==[python,'-B',dispatch['remote_root']+'/scripts/supervise.py']),
    dispatcher_identity=next(x for x in v['processes'] if x['pid']==dispatch['dispatcher_pid']),timeout_identity=next(x for x in v['processes'] if x['argv'][0]=='timeout'),
    config_sha256=dispatch['config_sha256'],source_pins_verified=len(p['source_pins']),manifest_files_verified=len(manifest['files']),first_head_9H_complete=True,original070_first_head_all_case_records_exact=True,original070_37_scientific_arrays_exact=True,
    diagnostic=diagnostic,diagnostic_BASE054_rows_exact060=True,original070_negative_result_retained=True,new_response_candidates=0,
    latest_progress=files['run_01/PROGRESS.json'],RSS_bytes=files['control_01/resource_status.json']['rss_bytes'],current_run_bytes=v['current_run_bytes'],
    continuous_polling_stopped=True,later_progress_unqueried=True,scientific_goal_achieved=False,raw_receipt_sha256=sha(w/'STARTUP_RAW.json'))
assert record['dispatcher_identity']['start_ticks']==dispatch['dispatcher_start_ticks']
(w/'STARTUP_ASSESSMENT.json').write_text(json.dumps(record,indent=2)+'\n')
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    path=r/name;d=read(path);job=d['response071'];assert job['stage']=='STARTED_UNVERIFIED' and job['active']
    job.update(stage='RUNNING_STARTUP_VERIFIED',native_identity=record['native_identity'],supervisor_identity=record['supervisor_identity'],last_verified_UTC=v['UTC'],execution_counts_at_startup=count,
        execution_counts_scope='First head9H and37 original scientific arrays exact070; three diagnostic tables19779 rows and39 fast examples verified.',
        startup_assessment=(w/'STARTUP_ASSESSMENT.json').relative_to(r).as_posix(),startup_assessment_sha256=sha(w/'STARTUP_ASSESSMENT.json'),later_progress_unqueried=True)
    d.update(status='response071_running_startup_verified',current_stage='response071_native_peak_tail_decomposition',latest_diagnostic_run_id=p['run_id'],
        current_action='Frozen070 replay correct; all first-head BASE054 features/flags/peaks match060; paired Q/tail and peak decompositions close.',
        next_action='At heartbeat collect071 within64MiB; verify both heads, original07018H/72NPZ replay and six diagnostic tables39558 rows plus78 examples; analyze allH, close and advance the next justified minimal step.',updated_UTC=v['UTC'],last_verified_UTC=v['UTC'])
    d['response_goal_20261018'].update(stage_status='running_startup_verified',current_stage=d['current_stage'],next_action=d['next_action'],updated_UTC=v['UTC'])
    if name=='jobs.json':assert len(d['jobs'])==90;d['jobs'][-1]=copy.deepcopy(job)
    path.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:record[k] for k in ('status','verified_UTC','diagnostic','RSS_bytes','current_run_bytes')}))
