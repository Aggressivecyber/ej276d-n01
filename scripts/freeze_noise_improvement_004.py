"""Bind exact reviewed source/config bytes and create one execution package."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ID='noise_improvement_004_20261012'
RECEIPTS=ROOT/'receipts'/ID
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def main():
    manifestp=ROOT/'EXECUTION_SOURCE_MANIFEST.json'
    if manifestp.exists():raise FileExistsError('Preserve execution source binding')
    config='noise_improvement_004_config.json';protocol='docs/NOISE_IMPROVEMENT_004_PROTOCOL.md'
    prereg='receipts/'+ID+'/PREREGISTRATION.json'
    reg=json.loads((ROOT/prereg).read_text(encoding='utf8'))
    assert sha(ROOT/config)==reg['config_sha256'] and sha(ROOT/protocol)==reg['protocol_sha256']
    prior=json.loads((ROOT/'receipts/repair_002_20261012/FINAL_RECEIPT.json').read_text(encoding='utf8'))
    for name,digest in prior['executed_source_hashes_unchanged'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Prior executed source changed: '+name)
    paths=set(ROOT.glob('src/n01/*.py'))
    paths.update(ROOT/'scripts'/name for name in ['launch_noise_improvement_004.sh','supervise.py','verify_noise_improvement_004.py',
        'preflight_noise_improvement_004.py','freeze_noise_improvement_004.py','register_noise_improvement_004.py'])
    paths.update(ROOT/'tests'/name for name in ['test_electronics_v3.py','test_train_noise_v3.py'])
    paths.update(ROOT/name for name in [config,protocol,prereg,'data/PARENT_IDENTITIES.npz'])
    paths.update(RECEIPTS/name for name in ['TEST_SUMMARY.json','INDEPENDENT_REVIEW.json','PREFLIGHT.json'])
    assert all(p.is_file() for p in paths)
    files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    manifest=dict(run_id=ID,UTC=dt.datetime.now(dt.timezone.utc).isoformat(),files=files,
        config_path=config,config_sha256=sha(ROOT/config),protocol_path=protocol,protocol_sha256=sha(ROOT/protocol),
        preregistration_path=prereg,preregistration_sha256=sha(ROOT/prereg),
        before_scientific_initialization=True,prior_executed_source_unchanged=True,
        explicit_replay_seed_rule='noise.generation_seed + 10000',replay_rows=1024,replay_chunk_size=256)
    write(manifestp,manifest)
    write(RECEIPTS/'IMPLEMENTATION_BINDING.json',dict(manifest_sha256=sha(manifestp),UTC=manifest['UTC'],files=len(files),
        configuration_unchanged=True,before_scientific_initialization=True))
    out=ROOT/'runs'/ID;out.mkdir(parents=True,exist_ok=True)
    pkg=out/'EXECUTION_SOURCE.tar.gz'
    if pkg.exists():raise FileExistsError(pkg)
    paths|={manifestp,RECEIPTS/'IMPLEMENTATION_BINDING.json'}
    with tarfile.open(pkg,'w:gz') as z:
        for p in sorted(paths):z.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    write(RECEIPTS/'EXECUTION_PACKAGE.json',dict(path=pkg.relative_to(ROOT).as_posix(),sha256=sha(pkg),bytes=pkg.stat().st_size,members=len(paths)))
    print(json.dumps(dict(package=str(pkg),sha256=sha(pkg),members=len(paths))))
if __name__=='__main__':main()
