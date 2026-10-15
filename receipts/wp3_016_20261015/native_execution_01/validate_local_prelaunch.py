"""Read-only exact-byte local prelaunch checks; no scientific binary files."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import wp3_016_controller as c

def read(path):return json.loads(path.read_bytes())
def sha(path):
    assert path.suffix.lower() not in {'.pt','.pth','.npy','.npz'}
    return hashlib.sha256(path.read_bytes()).hexdigest()
pins=dict(config_sha256='20884dc1a3916c2ac23820fa1ff7c47da21861a465a3f08fa5988cb71220281d',
    source_manifest_sha256='9151ca1a9503cde85786c0623effe431e10fa706ec5b92895bc448714ed93789',
    source_package_sha256='9d054aeb744cf35c0f50080ddebc0397574a66ac404dd2f68a5d4e9221cf6fb9',
    controller_sha256='f1d5eceadb145d6a921c41055dd7a7e4d9fb05c36a150e99689e50e21358d02a')
stage=ROOT/'receipts/wp3_016_20261015';freeze=stage/'runtime_package_01'
fr=read(freeze/'FREEZE_RECEIPT.json');assert all(fr[k]==v for k,v in pins.items())
assert sha(ROOT/c.CFG)==sha(freeze/c.CFG)==pins['config_sha256']
assert sha(ROOT/c.MANIFEST)==sha(freeze/c.MANIFEST)==pins['source_manifest_sha256']
assert sha(ROOT/c.CONTROLLER)==pins['controller_sha256']
assert sha(freeze/'wp3_016_source.tar.gz')==pins['source_package_sha256']
manifest=read(ROOT/c.MANIFEST);assert len(manifest['files'])==115
assert all(sha(ROOT/n)==r['sha256'] and (ROOT/n).stat().st_size==r['bytes'] for n,r in manifest['files'].items())
old=read(ROOT/'WP3_SOURCE_MANIFEST.json');assert len(old['files'])==105
assert all(sha(ROOT/n)==r['sha256'] for n,r in old['files'].items())
assert sha(ROOT/'wp3_015_config.json')==old['config_sha256']
source=read(stage/'SOURCE_COMMIT_READBACK.json');assert source['source_controls']==117 and source['all_raw_Git_blobs_match_frozen_bytes'] is True
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert commit==source['source_commit']=='450ebb086a465e0dcbd1c1081e71edcf51c405b9'
for name,row in source['files'].items():
    value=subprocess.check_output(['git','cat-file','blob',commit+':'+name],cwd=ROOT)
    assert hashlib.sha256(value).hexdigest()==row['sha256'] and value==(ROOT/name).read_bytes()
backup=read(stage/'D_SOURCE_BACKUP_RECEIPT.json');assert backup['files']==123 and backup['status']=='all_copied_files_SHA256_verified'
assert sha(Path(backup['destination'])/'MANIFEST.json')==backup['manifest_sha256']
review=stage/'independent_01/FROZEN_PACKAGE_REVIEW.json'
assert sha(review)=='a315e8fdaae7724cb25e7a81759bf01b52bdb7ec32daa59602f817a9449d156f' and read(review)['passed']
jobs=read(ROOT/'jobs.json');program=read(ROOT/'PROGRAM.json')
assert jobs['active_scientific_jobs']==0 and program['new_training_updates']==175667
for name in ['deploy_native_01','start_native_01','start_verification_01']:
    assert not (stage/name).exists(),'preserve earlier native attempt '+name
p=read(ROOT/c.CFG);c.validate_config(p)
assert p['budget_caps']['G_updates']==p['budget_caps']['new_AD_queries']==4000 and p['budget_caps']['checkpoint_loads']==4
result=dict(schema='n01/wp3-016-local-prelaunch/v1',UTC=datetime.now(timezone.utc).isoformat(),passed=True,
    source_commit=commit,pins=pins,source_files_verified=115,Git_blob_controls_verified=117,original015_sources_verified=105,
    original015_controls_unchanged=True,D_source_backup_files=123,active_scientific_jobs=0,cumulative_training_before=175667,
    actual_CP_array_reads_NN_AD_training_random_SSH=0,science_started=False,
    native_authorization='One deploy, one start, one bounded first-event existence wait, one verify-start; root directs metadata and timer.')
with (HERE/'LOCAL_PRELAUNCH_VALIDATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,source_commit=commit,source_files=115,controls=117,active_jobs=0,training_before=175667)))
