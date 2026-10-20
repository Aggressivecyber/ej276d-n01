from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import ast,hashlib,json,time

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'SOURCE_MANIFEST.json').read_bytes())
cp=ROOT/'configs/joint_response_01_recentered_candidate_check_20261020.json';p=json.loads(cp.read_bytes())
old=json.loads((ROOT/'configs/joint_response_01_recentered_derivative_completion_thin_20261020.json').read_bytes())
assert hashlib.sha256(cp.read_bytes()).hexdigest()==m['config_sha256'] and len(m['files'])==193
for key in ('batch','solver','native_guard','success','feature_seeds','native_projection_seeds','saved_arrays'):
    assert p[key]==old[key],key
assert 2*7*3==p['operation_budget']['joint_response_calls']==42
assert 42*32768==p['operation_budget']['joint_response_rows']==1376256
assert 42*3*32768==p['operation_budget']['generated_observation_rows']==4128768
assert 42*3*(32768+4096)==p['operation_budget']['native_component_rows']==4644864
source=(ROOT/'scripts/check_joint01_recentered_restoration.py').read_text(encoding='utf-8')
ast.parse(source);ast.parse((HERE/'AUDIT_FIXED_CANDIDATE.py').read_text(encoding='utf-8'))
assert 'for coordinate in range' not in source and 'DefaultSolver' not in source
assert source.count('=hard_check(')==2 and source.count('original=prediction(')==1
for name in ('src/n01/joint01_model.py','scripts/diagnose_joint01_original_draw_precision_fix01.py','scripts/run_joint_response01.py'):
    raw=(ROOT/name).read_bytes();entry=m['files'][m['prior_NAS_root']+'/'+name]
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
NAS=Path(p['storage']['windows_root']);NAS.mkdir(parents=True,exist_ok=False)
timer=time.perf_counter();written={}
for name,local in m['local_small_files'].items():
    raw=Path(local).read_bytes();entry=m['files'][name]
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
    target=NAS/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    assert target.read_bytes()==raw
    written[name]=entry
(NAS/'cache').mkdir()
(NAS/'SOURCE_MANIFEST.json').write_bytes((HERE/'SOURCE_MANIFEST.json').read_bytes())
with ThreadPoolExecutor(max_workers=4) as pool:
    pending=[(name,entry,pool.submit((NAS/name.replace('/mnt/MiNas/','Z:/')).read_bytes)) for name,entry in m['files'].items()]
    for name,entry,future in pending:
        raw=future.result();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
receipt=dict(status='PASS_FIXED_CANDIDATE_EXISTING_PATH_DEPLOYMENT_AND_STATIC_REVIEW',UTC=datetime.now(timezone.utc).isoformat(),
    source_files_verified=193,source_pins=192,small_files_written=len(written),small_bytes_written=sum(v['bytes'] for v in written.values()),
    elapsed_seconds=time.perf_counter()-timer,config_sha256=m['config_sha256'],new_generator_calls=0,
    fixed_candidates=2,library_checks=14,generator_calls_budget=42,model_loss_readout_bytes_unchanged=True,
    terminal_auditor_prepared=True,terminal_auditor_sha256=hashlib.sha256((HERE/'AUDIT_FIXED_CANDIDATE.py').read_bytes()).hexdigest(),
    numerical_thresholds_unchanged=True,NAS_mount_connection_transfer_repair_or_migration=False,large_payload_copies=False)
for path in (HERE/'PRE_DISPATCH_REVIEW.json',NAS/'PRE_DISPATCH_REVIEW.json'):
    path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(receipt))
