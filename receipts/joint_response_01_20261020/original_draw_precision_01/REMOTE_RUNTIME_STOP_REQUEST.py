from pathlib import Path
import datetime,hashlib,json,os,signal,time
root=Path('/home/rylan/projects/ej276d-n01/joint_response01_original_draw_precision_20261020')
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='83c9f761-c124-42e1-89b8-5f45f0b3acc4'
pid=145187
p=Path('/proc')/str(pid)
assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]=='3031495'
args=[v.decode() for v in p.joinpath('cmdline').read_bytes().split(bytes([0])) if v]
assert args[2]==str(root/'scripts/diagnose_joint01_original_draw_precision.py')
assert args[args.index('--config-sha256')+1]=='c3b72d59de88a990c77d4438951d51fd9ff2f7279fca0ead4e925aea0fa0e011'
assert not (root/'control_01/exit_receipt.json').exists()
record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=pid,start_ticks='3031495',signal='SIGTERM',identity_verified=True,reason='Measured 108-FD projection exceeds registered internal budget; separate exact-W1 numerical repair follows preserved prefix.')
(root/'RUNTIME_STOP_REQUEST.json').write_text(json.dumps(record,indent=2)+'\n')
os.kill(pid,signal.SIGTERM)
time.sleep(2)
record['exit_receipt']=json.loads((root/'control_01/exit_receipt.json').read_text())
print(json.dumps(record))
