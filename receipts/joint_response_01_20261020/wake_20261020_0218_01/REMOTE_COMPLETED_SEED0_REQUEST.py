from pathlib import Path
import datetime,hashlib,json,tarfile
root=Path('/home/rylan/projects/ej276d-n01/joint_response01_multibank_real_20261020')
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
for name,item in manifest['files'].items():
 raw=(root/name).read_bytes()
 assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
for name in ['seed0_joint','seed0_physical']:
 assert len(json.loads((root/'run_01'/f'{name}_TRACE.json').read_text())['iterations'])==12
assert (root/'run_01/seed0_KERNEL.json').exists()
files=sorted((root/'run_01').glob('seed0_*'))+[root/'run_01/REAL_CALIBRATION_ROLES.npz']
outputs={}
for f in files:
 raw=f.read_bytes();outputs[str(f.relative_to(root))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime_UTC=datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat())
archive=root/'completed_seed0_0218.tar'
with tarfile.open(archive,'w') as tar:
 for f in files:tar.add(f,arcname=str(f.relative_to(root)))
raw=archive.read_bytes()
print(json.dumps(dict(status='PASS_COMPLETED_SEED0_SOURCE_AND_OUTPUT_FREEZE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files_verified=len(manifest['files']),outputs=outputs,archive=str(archive),archive_bytes=len(raw),archive_sha256=hashlib.sha256(raw).hexdigest(),scientific_job_terminal=False)))
