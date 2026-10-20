from datetime import datetime,timezone
from pathlib import Path
import json,subprocess
HERE=Path(__file__).resolve().parent;T=HERE/'terminal_01';T.mkdir(exist_ok=True)
NAS=Path('Z:/GAN/research/N01/server_runs/joint_response01_full_energy_psd_20261021')
remote="""from pathlib import Path
from datetime import datetime,timezone
import json
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(bytes([0])) if v]
        if any('joint_response01_full_energy_psd_20261021/scripts/' in v for v in argv):
            processes.append(dict(pid=int(p.name),argv=argv,start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19]))
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),processes=processes)))
"""
r=subprocess.run(['ssh','-x','-o','BatchMode=yes','-o','ConnectTimeout=10','rylan@100.82.231.102','/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],input=remote.encode(),capture_output=True,timeout=30)
(T/'PROCESS_STDERR.txt').write_bytes(r.stderr);r.check_returncode();proc=json.loads(r.stdout)
artifacts={}
for relative in ('run_01/PROGRESS.json','run_01/RESULT.json','run_01/COMPLETE.json','run_01/FAILED.json','control_01/launch.json','control_01/resource_status.json','control_01/exit_receipt.json'):
    try:value=json.loads((NAS/relative).read_bytes())
    except FileNotFoundError:value=None
    artifacts[relative]=value
    if value is not None:(T/Path(relative).name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),boot_id=proc['boot_id'],processes=proc['processes'],native_exit=artifacts['control_01/exit_receipt.json'],complete_present=artifacts['run_01/COMPLETE.json'] is not None,failed_present=artifacts['run_01/FAILED.json'] is not None,NAS_file_IO_checks=False,NAS_file_hash_verification=False,actual_progress=artifacts['run_01/PROGRESS.json'])
(T/'TERMINAL_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='COMPLETE' if receipt['native_exit'] is not None else 'RUNNING_OR_UNOBSERVED',process_PIDs=[p['pid'] for p in proc['processes']],**{k:receipt[k] for k in ('native_exit','complete_present','failed_present','actual_progress')})))
