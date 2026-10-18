from pathlib import Path
import json,subprocess,copy
r=Path('E:/GAN/research/N01');w=r/'receipts/response070_20261018/deployment_01';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(w/'DEPLOYMENT_VERIFICATION.json')['engineering_result']['status']=='PASS'
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 d=read(r/name);assert d['active_scientific_jobs']==0 and d['response070']['stage']=='REGISTERED_NOT_STARTED'
c=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','rylan@100.82.231.102','python3 -'],input=(w/'REMOTE_DISPATCH_REQUEST.py').read_text(),capture_output=True,text=True,timeout=45)
(w/'DISPATCH_STDOUT.txt').write_text(c.stdout);(w/'DISPATCH_STDERR.txt').write_text(c.stderr);assert c.returncode==0,c.stderr
receipt=json.loads(c.stdout);(w/'DISPATCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 f=r/name;d=read(f);job=d['response070'];job.update(stage='STARTED_UNVERIFIED',active=True,native_started=True,dispatch_receipt=(w/'DISPATCH_RECEIPT.json').relative_to(r).as_posix(),started_UTC=receipt['UTC'],dispatcher_pid=receipt['dispatcher_pid'],dispatcher_start_ticks=receipt['dispatcher_start_ticks'],boot_id=receipt['boot_id'])
 d.update(status='response070_started_unverified',active_scientific_jobs=1,active_job_id=job['job_id'],updated_UTC=receipt['UTC'])
 d['response_goal_20261018'].update(stage_status='started_unverified',updated_UTC=receipt['UTC'])
 if name=='jobs.json':assert len(d['jobs'])==89;d['jobs'][-1]=copy.deepcopy(job)
 f.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(receipt))
