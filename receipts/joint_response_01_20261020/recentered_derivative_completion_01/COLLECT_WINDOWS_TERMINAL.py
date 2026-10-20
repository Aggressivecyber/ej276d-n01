"""Verify completed scientific files in place through the existing Z mapping."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
T = HERE/'terminal_01'
ROOT = Path('E:/GAN/research/N01')
p = json.loads((ROOT/'configs/joint_response_01_recentered_derivative_completion_thin_20261020.json').read_bytes())
NAS = Path(p['storage']['windows_root'])
manifest = json.loads((HERE/'SOURCE_MANIFEST.json').read_bytes())
assert not (T/'REMOTE_TERMINAL_VERIFICATION.json').exists()
request = """
from pathlib import Path
from datetime import datetime,timezone
import json
root='/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_completion_thin_20261020'
d=dict(UTC=datetime.now(timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),processes=[])
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
        argv=[v.decode(errors='replace') for v in (path/'cmdline').read_bytes().split(b'\\0') if v]
        if any(root in v for v in argv):
            d['processes'].append(dict(pid=int(path.name),argv=argv,start_ticks=(path/'stat').read_text().rsplit(')',1)[1].split()[19]))
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
print(json.dumps(d))
"""
r = subprocess.run(['ssh','-x','-o','BatchMode=yes','-o','ConnectTimeout=10',
    'rylan@100.82.231.102','/usr/bin/python3 -B -'], input=request.encode(),capture_output=True,timeout=30)
(T/'PROCESS_EXIT_VERIFICATION_STDOUT.json').write_bytes(r.stdout)
(T/'PROCESS_EXIT_VERIFICATION_STDERR.txt').write_bytes(r.stderr)
r.check_returncode()
d = json.loads(r.stdout)
assert d['boot_id']=='83c9f761-c124-42e1-89b8-5f45f0b3acc4' and not d['processes']
d['files'] = {}
for name in ('run_01/PROGRESS.json','run_01/COMPLETE.json','run_01/FAILED.json','run_01/RESULT.json',
    'control_01/launch.json','control_01/resource_status.json','control_01/exit_receipt.json'):
    path = NAS/name
    if path.exists():d['files'][name]=json.loads(path.read_bytes())
assert d['files']['control_01/exit_receipt.json']['exit_code']==0
assert 'run_01/COMPLETE.json' in d['files'] and 'run_01/FAILED.json' not in d['files']
with ThreadPoolExecutor(max_workers=4) as pool:
    pending=[(name,entry,pool.submit((NAS/name.replace('/mnt/MiNas/','Z:/')).read_bytes)) for name,entry in manifest['files'].items()]
    for name,entry,future in pending:
        raw=future.result()
        assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
outputs={}
for sub in ('run_01','control_01'):
    for path in (NAS/sub).rglob('*'):
        if path.is_file():
            raw=path.read_bytes()
            outputs[path.relative_to(NAS).as_posix()]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
d.update(outputs=outputs,native_exit=d['files']['control_01/exit_receipt.json'],
    complete_present=True,failed_present=False,source_files_verified=len(manifest['files']),
    config_sha256=manifest['config_sha256'],scientific_payloads_remain_on_NAS=True,
    remote_root=p['storage']['remote_root'],existing_windows_root=str(NAS),
    collection_completed_UTC=datetime.now(timezone.utc).isoformat(),
    access='Existing Windows Z mapping for scientific file reads; SSH only reads process state.',
    NAS_connection_mount_transfer_or_migration_changes=False)
(T/'REMOTE_TERMINAL_VERIFICATION.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')

original=(HERE/'AUDIT_COMPLETE.py').read_text(encoding='utf-8')
adapted=original
replacements=[
    ('NAS/name).read_bytes','NAS/name.replace(\'/mnt/MiNas/\',\'Z:/\')).read_bytes'),
    ('Path(name).read_bytes','Path(name.replace(\'/mnt/MiNas/\',\'Z:/\')).read_bytes'),
    ("NAS/p['roles_file']","NAS/p['roles_file'].replace('/mnt/MiNas/','Z:/')"),
    ("NAS/p['saved_arrays'][seed]","NAS/p['saved_arrays'][seed].replace('/mnt/MiNas/','Z:/')"),
    ("Path(manifest['prior_NAS_root'])","Path(manifest['prior_NAS_root'].replace('/mnt/MiNas/','Z:/'))")]
counts=[]
for before,after in replacements:
    count=adapted.count(before)
    assert count>0,before
    counts.append(count)
    adapted=adapted.replace(before,after)
restored=adapted
for before,after in reversed(replacements):restored=restored.replace(after,before)
assert restored==original
ast.parse(adapted)
(HERE/'AUDIT_COMPLETE_WINDOWS.py').write_text(adapted,encoding='utf-8',newline='\n')
receipt=dict(status='PASS_EXISTING_PATH_TERMINAL_COLLECTION_AND_PATH_ONLY_AUDIT_ADAPTATION',
    UTC=datetime.now(timezone.utc).isoformat(),original_auditor_sha256=hashlib.sha256((HERE/'AUDIT_COMPLETE.py').read_bytes()).hexdigest(),
    windows_auditor_sha256=hashlib.sha256((HERE/'AUDIT_COMPLETE_WINDOWS.py').read_bytes()).hexdigest(),
    path_replacements=[dict(before=b,after=a,occurrences=n) for (b,a),n in zip(replacements,counts)],
    reverse_replacements_restore_exact_original=True,numerical_formulas_changed=False,
    local_numpy=__import__('numpy').__version__,local_scipy=__import__('scipy').__version__,
    sources_verified=len(manifest['files']),outputs_verified=len(outputs),native_exit=d['native_exit'],
    previous_SSH_scientific_file_collection='240 second timeout; stdout and stderr retained; no scientific restart.',
    large_payload_copies=False,storage_changes=False)
(T/'WINDOWS_TERMINAL_COLLECTION_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(receipt))
