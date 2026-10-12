"""Freeze the independently reviewed exact metadata representation repair."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1];ID='integration_006_20261012'
REC=ROOT/'receipts'/ID;FIX=REC/'binding_fix01'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def main():
    out=ROOT/'INTEGRATION006_BINDING_FIX01_SOURCE_MANIFEST.json';assert not out.exists()
    old=read(ROOT/'INTEGRATION006_SOURCE_MANIFEST.json');reg=read(FIX/'PREREGISTRATION.json')
    assert sha(ROOT/'INTEGRATION006_SOURCE_MANIFEST.json')==reg['initial_source_manifest_sha256']
    assert sha(ROOT/'integration006_config.json')==reg['unchanged_configuration_sha256']==old['config_sha256']
    assert sha(ROOT/'docs/INTEGRATION_006_PROTOCOL.md')==reg['unchanged_protocol_sha256']==old['protocol_sha256']
    assert sha(ROOT/reg['amendment'])==reg['amendment_sha256']
    changed={'src/n01/evaluate_integration006.py','tests/test_evaluate_integration006.py'}
    for name,digest in old['files'].items():
        if name not in changed:assert sha(ROOT/name)==digest,name
    for name,digest in read(ROOT/'NOISE_ACF005_SOURCE_MANIFEST.json')['files'].items():assert sha(ROOT/name)==digest,name
    review=read(FIX/'INDEPENDENT_REVIEW.json');tests=read(FIX/'TEST_SUMMARY.json')
    assert review['status']=='passed_for_registered_execution' and tests['status']=='passed'
    for r in [review,tests]:
        for name,digest in r['source_hashes'].items():assert sha(ROOT/name)==digest,name
    paths={ROOT/name for name in old['files']}
    paths.update(ROOT/name for name in ['INTEGRATION006_SOURCE_MANIFEST.json','docs/INTEGRATION_006_BINDING_FIX01.md',
        'scripts/register_integration006_binding_fix01.py','scripts/freeze_integration006_binding_fix01.py',
        'scripts/launch_integration006_binding_fix01.sh'])
    paths.update(FIX/name for name in ['PREREGISTRATION.json','INDEPENDENT_REVIEW.json','METADATA_INDEPENDENT_REVIEW.json','TEST_SUMMARY.json'])
    manifest={k:v for k,v in old.items() if k not in ['files','UTC','preregistration_path','preregistration_sha256']}
    manifest.update(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),attempt_id='binding_fix01',
        historical_initial_source_manifest_sha256=reg['initial_source_manifest_sha256'],
        preregistration_path=(FIX/'PREREGISTRATION.json').relative_to(ROOT).as_posix(),preregistration_sha256=sha(FIX/'PREREGISTRATION.json'),
        amendment_path=reg['amendment'],amendment_sha256=reg['amendment_sha256'],
        checkpoint_metadata_previously_decoded=True,SOURCE_REAL_datasets_previously_decoded=False,
        scientific_forward_rows_before_freeze=0,
        files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
    write(out,manifest)
    binding=FIX/'IMPLEMENTATION_BINDING.json'
    write(binding,dict(UTC=manifest['UTC'],manifest_sha256=sha(out),files=len(paths),
        protocol_unchanged=True,configuration_unchanged=True,science_started=False,initial_failure_preserved=True))
    paths|={out,binding}
    run=ROOT/'runs'/ID/'binding_fix01';run.mkdir(exist_ok=True)
    package=run/'EXECUTION_SOURCE.tar.gz';assert not package.exists()
    with tarfile.open(package,'w:gz') as t:
        for p in sorted(paths):t.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    value=dict(path=package.relative_to(ROOT).as_posix(),bytes=package.stat().st_size,sha256=sha(package),members=len(paths),
        source_manifest_sha256=sha(out))
    write(FIX/'EXECUTION_PACKAGE.json',value);print(json.dumps(value))
if __name__=='__main__':main()
