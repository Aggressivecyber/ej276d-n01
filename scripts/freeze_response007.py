"""Freeze reviewed007 files and an isolated execution archive before science."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1];ID='response_007_20261013';REC=ROOT/'receipts'/ID
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def main():
    manifest_path=ROOT/'RESPONSE007_SOURCE_MANIFEST.json';assert not manifest_path.exists()
    cfg=read(ROOT/'response007_config.json');reg=read(REC/'PREREGISTRATION.json')
    review=read(REC/'SELF_REVIEW.json');tests=read(REC/'SELF_TEST.json')
    assert review['status']=='passed_primary_agent_self_review_for_bounded_execution' and tests['status']=='passed'
    for receipt in [review,tests]:
        for name,digest in receipt['source_hashes'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'response007_config.json')==reg['config_sha256']
    assert sha(ROOT/'docs/RESPONSE_007_PROTOCOL.md')==reg['protocol_sha256']
    for name,digest in read(ROOT/'INTEGRATION006_ID_FIX02_SOURCE_MANIFEST.json')['files'].items():assert sha(ROOT/name)==digest,name
    paths=set((ROOT/'src').rglob('*.py'))
    paths.update(ROOT/name for name in ['response007_config.json','docs/RESPONSE_007_PROTOCOL.md','docs/RESPONSE_007_IMPLEMENTATION.md',
        'scripts/register_response007.py','scripts/self_review_response007.py','scripts/freeze_response007.py',
        'scripts/launch_response007.sh','scripts/start_response007_remote.py','scripts/supervise.py','scripts/benchmark_response007.py','scripts/prepare_response007_benchmark.py',
        'tests/test_response007.py','data/PARENT_IDENTITIES.npz','AGENTS.md','USER_PROPOSAL.md'])
    paths.update(ROOT/item['path'] for item in cfg['references'].values())
    paths.update(REC/name for name in ['PREREGISTRATION.json','SELF_REVIEW.json','SELF_TEST.json','SELF_TEST_OUTPUT.txt','ENGINEERING_TEST_HISTORY.json'])
    manifest=dict(run_id=ID,UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        config_sha256=sha(ROOT/'response007_config.json'),protocol_path='docs/RESPONSE_007_PROTOCOL.md',protocol_sha256=sha(ROOT/'docs/RESPONSE_007_PROTOCOL.md'),
        before_new_scientific_array_decode=True,before_new_scientific_forward=True,
        planned_G_updates=38000,planned_D_updates=96000,planned_C2ST_updates=4200,noise_updates=0,
        training_container_sha256=cfg['train_container_sha256'],identities_sha256=cfg['identities_sha256'],
        review_mode='primary_agent_self_review_per_user_no_subagents',
        historical006_source_manifest_sha256=sha(ROOT/'INTEGRATION006_ID_FIX02_SOURCE_MANIFEST.json'),
        files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
    write(manifest_path,manifest);paths.add(manifest_path)
    folder=ROOT/'runs'/ID;folder.mkdir(parents=True,exist_ok=True)
    package=folder/'EXECUTION_SOURCE.tar.gz';assert not package.exists()
    with tarfile.open(package,'w:gz') as archive:
        for p in sorted(paths):archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    # Reopen every archive member before delivery.
    with tarfile.open(package,'r:gz') as archive:
        for member in archive.getmembers():
            expected=sha(ROOT/member.name)
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==expected,member.name
    receipt=dict(UTC=manifest['UTC'],package=package.relative_to(ROOT).as_posix(),bytes=package.stat().st_size,
        sha256=sha(package),source_manifest_sha256=sha(manifest_path),members=len(paths),every_member_reopened_and_hashed=True,
        raw_waveform_datasets_in_package=False,private_parent_identity_index_in_package=True,frozen_noise_reference_in_package=True)
    write(REC/'EXECUTION_PACKAGE.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
