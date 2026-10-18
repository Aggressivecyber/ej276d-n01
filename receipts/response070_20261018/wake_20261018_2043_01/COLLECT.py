from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,subprocess,tarfile
r=Path('E:/GAN/research/N01');w=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
startup=read(r/'receipts/response070_20261018/deployment_01/STARTUP_ASSESSMENT.json')
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=read(r/name);assert d['active_scientific_jobs']==1 and d['response070']['stage']=='RUNNING_STARTUP_VERIFIED'
identities=tuple((startup[k]['pid'],startup[k]['start_ticks']) for k in ('dispatcher_identity','supervisor_identity','timeout_identity','native_identity'))
code=(r/'receipts/response069_20261018/wake_20261018_2019_01/REMOTE_STATUS_AND_COLLECTION_REQUEST.py').read_text()
previous_root=json.loads((r/'receipts/response069_20261018/deployment_01/DISPATCH_RECEIPT.json').read_text())['remote_root']
current_root=json.loads((r/'receipts/response070_20261018/deployment_01/DISPATCH_RECEIPT.json').read_text())['remote_root']
code=code.replace(previous_root,current_root).replace('response069_outputs','response070_outputs').replace('20261018_2019_01','20261018_2043_01').replace("'069 ","'070 ")
old=next(line for line in code.splitlines() if line.startswith('for pid,expected in '))
code=code.replace(old,'for pid,expected in '+repr(identities)+':');ast.parse(code)
(w/'REMOTE_STATUS_AND_COLLECTION_REQUEST.py').write_text(code)
z=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','rylan@100.82.231.102','python3 -'],input=code,capture_output=True,text=True,timeout=60)
(w/'REMOTE_STATUS_STDOUT.txt').write_text(z.stdout);(w/'REMOTE_STATUS_STDERR.txt').write_text(z.stderr);assert z.returncode==0,z.stderr
v=json.loads(z.stdout);(w/'REMOTE_STATUS.json').write_text(json.dumps(v,indent=2)+'\n')
assert v['boot_id']==startup['boot_id']
print(json.dumps({k:v[k] for k in ('UTC','identities','files','console_tail') if k!='files'}))
print(json.dumps({k:z for k,z in v['files'].items() if k in ('control_01/exit_receipt.json','run_01/COMPLETE.json')}))
if 'collection' not in v:
    print('No terminal collection: inspect saved bounded identity snapshot and registered deadline');raise SystemExit(0)
c=v['collection'];assert c['uncompressed_bytes']<=64*1024**2 and c['archive_bytes']<=64*1024**2
assert shutil.disk_usage(w).free>3*64*1024**2
archive=w/'response070_outputs.tar.gz'
z=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=12','rylan@100.82.231.102:'+c['archive_path'],str(archive)],capture_output=True,text=True,timeout=60)
(w/'SCP_STDOUT.txt').write_text(z.stdout);(w/'SCP_STDERR.txt').write_text(z.stderr);assert z.returncode==0,z.stderr
assert archive.stat().st_size==c['archive_bytes'] and sha(archive)==c['archive_sha256']
dest=w/'collected';dest.mkdir()
with tarfile.open(archive) as tf:
    assert tf.getnames()==list(c['manifest'])
    for m in tf.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
    tf.extractall(dest,filter='data')
for name,meta in c['manifest'].items():
    path=dest/name;assert path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256'],name
assert len([x for x in dest.rglob('*') if x.is_file()])==c['file_count']
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),file_count=c['file_count'],uncompressed_bytes=c['uncompressed_bytes'],archive_bytes=c['archive_bytes'],archive_sha256=c['archive_sha256'],
    all_file_lengths_and_sha256_verified=True,terminal_source_verification=v['terminal_source_verification'],
    run_bytes=sum(p.stat().st_size for p in (dest/'run_01').rglob('*') if p.is_file()),no_new_scientific_evaluations=True)
(w/'COLLECTION_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
