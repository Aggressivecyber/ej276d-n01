"""Commit explicit017 controls/evidence only, then verify raw control blobs."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(args):return subprocess.check_output(['git',*args],cwd=ROOT)

def main():
    config=ROOT/'wp4_017_config.json';manifest=ROOT/'WP4_017_SOURCE_MANIFEST.json'
    p,mf=read(config),read(manifest)
    assert p['status']=='frozen' and read(ROOT/'jobs.json')['active_scientific_jobs']==0
    ready=read(OUT/'EXECUTION_READY.json');assert ready['actual017_G_AD']==0
    review=read(OUT/'independent_01/SOURCE_REVIEW.json');assert review['passed'] and not review['blockers']
    controls={**mf['files'],config.name:dict(sha256=sha(config),bytes=config.stat().st_size),
        manifest.name:dict(sha256=sha(manifest),bytes=manifest.stat().st_size)}
    for name,row in controls.items():
        path=ROOT/name;assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],name
    protected=read(OUT/'engineering_01/PROTECTED_BEFORE.json')['files'];assert len(protected)==124
    for name,row in protected.items():
        path=ROOT/name;assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],name
    assert not run(['diff','--cached','--name-only']).strip()
    files=[ROOT/n for n in controls]+[ROOT/n for n in ('.gitattributes','MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')]
    for directory in (OUT,OUT/'planning_01',OUT/'engineering_01',OUT/'independent_01'):
        files += [f for f in directory.iterdir() if f.is_file() and f.suffix in ('.py','.json','.txt','.md','.log')]
    files += list((OUT/'engineering_01').rglob('ENGINEERING_RECEIPT.json'))
    files += list((OUT/'engineering_01').rglob('*.py'))
    files += [f for f in (OUT/'engineering_01/source_review_fix01').rglob('*')
        if f.is_file() and f.suffix in ('.json','.md','.txt')
        and 'artifacts' not in f.relative_to(OUT/'engineering_01/source_review_fix01').parts]
    files.append(OUT/'runtime_package_01/FREEZE_RECEIPT.json')
    files=list(dict.fromkeys(files))
    assert all(f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(ROOT.resolve()) for f in files)
    assert all(f.stat().st_size<8*1024**2 for f in files)
    assert sum(f.stat().st_size for f in files)<32*1024**2
    names=[f.relative_to(ROOT).as_posix() for f in files]
    run(['add','-f','--',*names]);run(['diff','--cached','--check'])
    staged=run(['diff','--cached','--name-only']).decode().splitlines()
    assert set(staged)<=set(names) and staged
    commit_output=run(['commit','-m','Complete local WP4 known-kernel implementation and fixed stage gate']).decode(errors='replace')
    commit=run(['rev-parse','HEAD']).decode().strip()
    for name,row in controls.items():
        blob=run(['cat-file','blob',commit+':'+name])
        assert len(blob)==row['bytes'] and hashlib.sha256(blob).hexdigest()==row['sha256'],name
    path=OUT/'SOURCE_COMMIT_ASSEMBLY.json'
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(dict(schema='n01/wp4-017-explicit-source-commit/v1',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
            source_commit=commit,source_controls=len(controls),all_raw_control_blobs_byte_exact=True,
            original124_controls_unchanged=True,explicit_staged_paths=staged,listed_evidence_paths=names,
            private_numeric_payloads_committed=False,public_push_performed=False,scientific_operations=0,
            commit_output=commit_output),stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(status='reviewed017_source_committed_raw_verified',source_commit=commit,
        controls=len(controls),changed_paths=len(staged),private_numeric_payloads=False)))

if __name__=='__main__':main()
