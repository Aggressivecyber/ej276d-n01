"""Freeze reviewed code and all inputs for the zero-training 006 run."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ID='integration_006_20261012';REC=ROOT/'receipts'/ID
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def main():
    manifestp=ROOT/'INTEGRATION006_SOURCE_MANIFEST.json'
    if manifestp.exists():raise FileExistsError('Already frozen')
    cfg=read(ROOT/'integration006_config.json');reg=read(REC/'PREREGISTRATION.json')
    assert sha(ROOT/'integration006_config.json')==reg['config_sha256']
    assert sha(ROOT/'docs/INTEGRATION_006_PROTOCOL.md')==reg['protocol_sha256']
    for name,digest in read(ROOT/'NOISE_ACF005_SOURCE_MANIFEST.json')['files'].items():assert sha(ROOT/name)==digest,name
    tests=read(REC/'TEST_SUMMARY.json');review=read(REC/'INDEPENDENT_REVIEW.json')
    assert tests['status']=='passed' and review['status']=='passed_for_registered_execution'
    for receipt in (tests,review):
        for name,digest in receipt['source_hashes'].items():assert sha(ROOT/name)==digest,name
    paths=set(ROOT.glob('src/n01/*.py'))
    paths.update(ROOT/'scripts'/name for name in ('launch_integration006.sh','preflight_noise_improvement_004.py',
        'supervise.py','register_integration006.py','freeze_integration006.py','verify_integration006.py'))
    paths.update(ROOT/'tests'/name for name in ('test_extension006.py','test_evaluate_integration006.py','test_verify_integration006.py'))
    paths.update(ROOT/name for name in ('integration006_config.json','docs/INTEGRATION_006_PROTOCOL.md','data/PARENT_IDENTITIES.npz'))
    paths.update(REC/name for name in ('PREREGISTRATION.json','PROTOCOL_REVIEW.json','INDEPENDENT_REVIEW.json','TEST_SUMMARY.json','PREFLIGHT.json','OUTPUT_SCHEMA.json'))
    for item in cfg['references'].values():
        p=ROOT/item['path'];assert sha(p)==item['sha256'];paths.add(p)
    assert all(p.is_file() for p in paths)
    manifest=dict(run_id=ID,UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        config_path='integration006_config.json',config_sha256=reg['config_sha256'],
        protocol_path='docs/INTEGRATION_006_PROTOCOL.md',protocol_sha256=reg['protocol_sha256'],
        preregistration_path=f'receipts/{ID}/PREREGISTRATION.json',preregistration_sha256=sha(REC/'PREREGISTRATION.json'),
        files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)},
        before_new_scientific_array_decode=True,before_new_scientific_forward=True,
        scientific_update_budget=0,legacy_sources_unchanged=True)
    write(manifestp,manifest)
    binding=REC/'IMPLEMENTATION_BINDING.json'
    write(binding,dict(UTC=manifest['UTC'],manifest_sha256=sha(manifestp),files=len(paths),
        protocol_unchanged=True,configuration_unchanged=True,science_started=False))
    paths|={manifestp,binding}
    output=ROOT/'runs'/ID;output.mkdir(parents=True,exist_ok=True)
    package=output/'EXECUTION_SOURCE.tar.gz'
    if package.exists():raise FileExistsError(package)
    with tarfile.open(package,'w:gz') as archive:
        for p in sorted(paths):archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    write(REC/'EXECUTION_PACKAGE.json',dict(path=package.relative_to(ROOT).as_posix(),bytes=package.stat().st_size,
        sha256=sha(package),members=len(paths)))
    print(json.dumps(dict(status='frozen',sha256=sha(package),files=len(paths),package=str(package))))
if __name__=='__main__':main()
