from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
root=Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_diagnostic_20261020')
record=dict(UTC=datetime.now(timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),files={},processes=[])
for name in ('run_01/PROGRESS.json','run_01/COMPLETE.json','run_01/FAILED.json','run_01/RESULT.json',
    'control_01/launch.json','control_01/resource_status.json','control_01/exit_receipt.json'):
    if (root/name).exists():record['files'][name]=json.loads((root/name).read_bytes())
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
        argv=[v.decode(errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v]
        if any(str(root) in v for v in argv):
            record['processes'].append(dict(pid=int(path.name),argv=argv,start_ticks=(path/'stat').read_text().rsplit(')',1)[1].split()[19]))
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
m=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
for name,entry in m['files'].items():
    raw=(root/name).read_bytes();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
record.update(source_files_verified=len(m['files']),config_sha256=m['config_sha256'])
if 'control_01/exit_receipt.json' in record['files']:
    assert not record['processes']
    outputs={}
    for sub in ('run_01','control_01'):
        for path in (root/sub).rglob('*'):
            if path.is_file():
                raw=path.read_bytes();outputs[path.relative_to(root).as_posix()]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    record.update(outputs=outputs,native_exit=record['files']['control_01/exit_receipt.json'],
        complete_present=(root/'run_01/COMPLETE.json').exists(),failed_present=(root/'run_01/FAILED.json').exists(),
        scientific_payloads_remain_on_NAS=True,remote_root=str(root))
print(json.dumps(record))
