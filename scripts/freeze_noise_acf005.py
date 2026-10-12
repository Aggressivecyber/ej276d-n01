"""Freeze reviewed 005 code and references before any new scientific decode."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ID='noise_acf005_20261012'
REC=ROOT/'receipts'/ID
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')


def main():
    manifestp=ROOT/'NOISE_ACF005_SOURCE_MANIFEST.json'
    if manifestp.exists():raise FileExistsError('Preserve frozen execution source')
    config_path='noise_acf005_config.json';protocol_path='docs/NOISE_ACF005_PROTOCOL.md'
    prereg_path='receipts/'+ID+'/PREREGISTRATION.json'
    config=read(ROOT/config_path);reg=read(ROOT/prereg_path)
    assert sha(ROOT/config_path)==reg['config_sha256'] and sha(ROOT/protocol_path)==reg['protocol_sha256']
    tests=read(REC/'TEST_SUMMARY.json');review=read(REC/'INDEPENDENT_REVIEW.json')
    assert tests['status']=='passed' and review['status']=='passed_for_registered_execution'
    for name,digest in tests['source_hashes'].items():assert sha(ROOT/name)==digest,name
    for name,digest in review['source_hashes'].items():assert sha(ROOT/name)==digest,name
    previous=read(ROOT/'runs/noise_improvement_004_20261012/runs/science/COMPLETE.json')
    for name,digest in previous['source_files'].items():assert sha(ROOT/name)==digest,name
    paths=set(ROOT.glob('src/n01/*.py'))
    paths.update(ROOT/'scripts'/name for name in [
        'launch_noise_acf005.sh','preflight_noise_improvement_004.py','supervise.py',
        'verify_noise_acf005.py','verify_noise_acf005_details.py','calibrate_real_real_null.py','freeze_noise_acf005.py',
        'register_noise_acf005.py'])
    paths.update(ROOT/'tests'/name for name in ['test_acf20.py','test_train_noise_acf005.py','test_noise_validation_acf005.py'])
    paths.update(ROOT/name for name in [config_path,protocol_path,prereg_path,'data/PARENT_IDENTITIES.npz'])
    paths.update(REC/name for name in ['TEST_SUMMARY.json','INDEPENDENT_REVIEW.json','PROTOCOL_REVIEW.json','PREFLIGHT.json','SOURCE_CLAIM_CHECK.json'])
    paths.update(ROOT/item['path'] for item in config['references'].values())
    for item in config['references'].values():assert sha(ROOT/item['path'])==item['sha256']
    assert all(p.is_file() for p in paths)
    manifest=dict(run_id=ID,UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)},
        config_path=config_path,config_sha256=sha(ROOT/config_path),
        protocol_path=protocol_path,protocol_sha256=sha(ROOT/protocol_path),
        preregistration_path=prereg_path,preregistration_sha256=sha(ROOT/prereg_path),
        before_new_scientific_array_decode=True,before_new_initialization=True,before_new_fit=True,
        old_executed_sources_unchanged=True,diagnostic_rows=8192,diagnostic_seed=config['diagnostics']['seed'],
        explicit_replay_rows=1024,explicit_replay_seed=config['noise']['generation_seed']+10000)
    write(manifestp,manifest)
    write(REC/'IMPLEMENTATION_BINDING.json',dict(UTC=manifest['UTC'],manifest_sha256=sha(manifestp),
        files=len(paths),configuration_unchanged=True,before_new_scientific_array_decode=True))
    paths|={manifestp,REC/'IMPLEMENTATION_BINDING.json'}
    folder=ROOT/'runs'/ID;folder.mkdir(parents=True,exist_ok=True)
    archive=folder/'EXECUTION_SOURCE.tar.gz'
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(paths):tar.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    write(REC/'EXECUTION_PACKAGE.json',dict(path=archive.relative_to(ROOT).as_posix(),
        bytes=archive.stat().st_size,sha256=sha(archive),members=len(paths)))
    print(json.dumps(dict(package=str(archive),sha256=sha(archive),members=len(paths))))


if __name__=='__main__':main()
