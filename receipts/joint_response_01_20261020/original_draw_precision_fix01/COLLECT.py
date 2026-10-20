from pathlib import Path
import hashlib,json,subprocess,tarfile
f=Path(__file__).resolve().parent;t=f/'terminal_01';t.mkdir(exist_ok=False)
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','rylan@100.82.231.102','python3 -'],input=(f/'REMOTE_COLLECT.py').read_bytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
(t/'COLLECT_STDERR.txt').write_bytes(r.stderr);r.check_returncode();record=json.loads(r.stdout)
(t/'REMOTE_TERMINAL_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
r=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','rylan@100.82.231.102:'+record['archive'],str(t/'terminal.tar')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
(t/'SCP_STDERR.txt').write_bytes(r.stderr);r.check_returncode()
raw=(t/'terminal.tar').read_bytes();assert len(raw)==record['archive_bytes'] and hashlib.sha256(raw).hexdigest()==record['archive_sha256']
with tarfile.open(t/'terminal.tar') as tar:
    members=tar.getmembers();assert set(m.name for m in members)==set(record['outputs'])
    for m in members:
        assert m.isfile() and (t/'collected'/m.name).resolve().is_relative_to((t/'collected').resolve())
    tar.extractall(t/'collected',filter='data')
for name,r in record['outputs'].items():
    raw=(t/'collected'/name).read_bytes();assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
print(json.dumps({'status':'COLLECTED_AND_HASH_VERIFIED','files':len(record['outputs']),'bytes':sum(r['bytes'] for r in record['outputs'].values()),'native_exit':record['native_exit']}))
