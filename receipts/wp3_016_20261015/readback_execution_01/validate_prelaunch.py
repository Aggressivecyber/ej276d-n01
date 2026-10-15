"""Read-only local source/pin validation before one registered CPU start."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import subprocess

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;STAGE=HERE.parent;REG=STAGE/'readback_registration_01'
read=lambda p:json.loads(p.read_bytes())
def sha(p):
    assert p.suffix.lower() not in ('.npz','.pt','.pth','.npy')
    return hashlib.sha256(p.read_bytes()).hexdigest()
config=REG/'READBACK_EXECUTION.json';expected='afa1b6a340e9e71eb5517ca68929c15bb05c9c5a1d990da50ebe350cce0ba8a8'
assert sha(config)==expected;p=read(config)
path=STAGE/'readback_engineering_01/readback_control.py';assert sha(path)=='90931be0d929404d093e3de532cd50fc22b3035a1dc0d61c1c7196febffe5048'
spec=importlib.util.spec_from_file_location('CPU016_control_local',path);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.validate(p)
for row in p['files'].values():
    local=Path(row['local_path']);assert local.drive.upper() in ('E:','D:') and local.stat().st_size==row['bytes'] and sha(local)==row['sha256']
source=read(REG/'SOURCE_COMMIT_READBACK.json');commit=source['source_commit']
assert commit=='3c1b0e218b9f54d91d6dc713d75e5e2ec2ddbf60' and source['source_controls']==6 and source['raw_Git_blobs_byte_exact']
for name,row in source['files'].items():
    data=subprocess.check_output(['git','cat-file','blob',commit+':'+name],cwd=ROOT)
    assert data==(ROOT/name).read_bytes() and hashlib.sha256(data).hexdigest()==row['sha256']
old=read(STAGE/'SOURCE_COMMIT_READBACK_FIX02.json');assert len(old['files'])==118
assert all(sha(ROOT/name)==row['sha256'] for name,row in old['files'].items())
backup=read(REG/'D_SOURCE_BACKUP_RECEIPT.json');assert backup['files']==45 and backup['source_commit']==commit
assert sha(Path(backup['destination'])/'MANIFEST.json')==backup['manifest_sha256']
review=STAGE/'postrun_independent_01/PRELAUNCH_REVIEW.json';assert sha(review)=='4987f4f8fd38243aaad7197f651996653882bb76bdd932a3b4d144b0b08b6bff' and read(review)['passed']
jobs=read(ROOT/'jobs.json');program=read(ROOT/'PROGRAM.json');assert jobs['active_scientific_jobs']==0 and program['new_training_updates']==179667
assert all(v==0 for v in p['scientific_budget'].values()) and p['expected_operations']==200
for name in ('readback_prepare_01','readback_start_01','readback_start_verification_01'):assert not (STAGE/name).exists(),'preserve existing action '+name
result=dict(schema='n01/wp3-016-CPU-local-prelaunch/v1',UTC=datetime.now(timezone.utc).isoformat(),passed=True,source_commit=commit,
    config_sha256=expected,controller_sha256=sha(path),files=p['files'],Git_controls_verified=6,original_training_controls_verified=118,
    D_backup_files=45,active_scientific_jobs=0,cumulative_training_updates=179667,expected_operations=200,
    scientific_budget=p['scientific_budget'],actual_private_NPZ_CP_NN_AD_RNG_training_GPU_SSH=0)
with (HERE/'LOCAL_PRELAUNCH_VALIDATION.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,source_commit=commit,config_sha256=expected,controls=6,original_controls=118)))
