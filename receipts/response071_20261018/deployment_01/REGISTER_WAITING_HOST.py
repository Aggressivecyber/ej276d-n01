from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();now=datetime.now(timezone.utc).isoformat()
status=read(w/'transfer_attempt02/NETWORK_STATUS.json');assert status['backend_state']=='Running' and status['self']['Online'] and len(status['target'])==1 and status['target'][0]['Online'] is False
assert not (w/'DISPATCH_RECEIPT.json').exists() and not (w/'DEPLOYMENT_VERIFICATION.json').exists()
config=read(r/'configs/response071_20261018.json');pkg=read(w/'PACKAGE_RECEIPT.json');assert sha(r/'configs/response071_20261018.json')==pkg['config_sha256'] and sha(w/'response071_source.tar.gz')==pkg['archive_sha256']
for name,digest in config['source_pins'].items():assert sha(r/name)==digest,name
before=w/'before_host_wait';before.mkdir()
record=dict(UTC=now,status='REGISTERED_NOT_STARTED_AWAITING_REMOTE_HOST',job_id=config['run_id'],native_started=False,active_scientific_jobs=0,source_pins_verified=len(config['source_pins']),config_sha256=pkg['config_sha256'],archive_sha256=pkg['archive_sha256'],host='rylan@100.82.231.102',host_online=False,host_last_seen_UTC=status['target'][0]['LastSeen'],network_receipt=(w/'transfer_attempt02/NETWORK_STATUS.json').relative_to(r).as_posix(),scientific_code_or_budget_changed=False,partial_remote_archive_possible=True,remote_root_created_by_installer=False,dispatch_not_attempted=True,automatic_restart=False,next_action='At next heartbeat check target once; when online retry exact frozen071 package installation, verify CUDA and source, then dispatch and verify first9H. Otherwise remain quiet and reschedule a bounded availability check. No replacement candidate or extra budget.')
(w/'DEPLOYMENT_WAIT_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n')
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 path=r/name;d=read(path);assert d['active_scientific_jobs']==0 and d['response071']['stage']=='REGISTERED_NOT_STARTED' and not d['response071']['active'];(before/name).write_bytes(path.read_bytes());job=d['response071']
 job.update(deployment_status='waiting_for_remote_host',deployment_wait_receipt=(w/'DEPLOYMENT_WAIT_RECEIPT.json').relative_to(r).as_posix(),deployment_wait_receipt_sha256=sha(w/'DEPLOYMENT_WAIT_RECEIPT.json'),native_started=False)
 d.update(status='response071_registered_awaiting_remote_host',updated_UTC=now,current_action='071 implementation, CPU semantic checks and frozen package complete. Two SCP attempts failed; Tailscale confirms target host offline. No native dispatch attempted.',next_action=record['next_action'])
 d['response_goal_20261018'].update(stage_status='registered_awaiting_remote_host',next_action=d['next_action'],updated_UTC=now)
 if name=='jobs.json':assert len(d['jobs'])==90;d['jobs'][-1]=copy.deepcopy(job)
 path.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
assert read(before/'jobs.json')['jobs'][:-1]==read(r/'jobs.json')['jobs'][:-1]
print(json.dumps(record))
