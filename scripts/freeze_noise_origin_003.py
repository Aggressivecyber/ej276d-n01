"""Bind reviewed source and preregistration before any new scientific execution."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ID='noise_origin_audit_003_20261012'
RECEIPTS=ROOT/'receipts'/ID

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')

def main():
    binding=RECEIPTS/'IMPLEMENTATION_BINDING.json'
    if binding.exists():
        raise FileExistsError('Preserve the existing scientific source binding')
    prior=json.loads((RECEIPTS/'PREREGISTRATION.json').read_text(encoding='utf8'))
    for name,key in [('noise_origin_003_config.json','config_sha256'),('docs/NOISE_ORIGIN_003_PROTOCOL.md','protocol_sha256')]:
        if sha(ROOT/name)!=prior[key]:
            raise RuntimeError('Prospective registration changed: '+name)
    addendum=json.loads((RECEIPTS/'PAIRED_ADDENDUM_REGISTRATION.json').read_text(encoding='utf8'))
    if sha(ROOT/addendum['addendum'])!=addendum['addendum_sha256']:
        raise RuntimeError('Prospective addendum changed')
    old=json.loads((ROOT/'receipts/repair_002_20261012/FINAL_RECEIPT.json').read_text(encoding='utf8'))
    for name,digest in old['executed_source_hashes_unchanged'].items():
        if sha(ROOT/name)!=digest:
            raise RuntimeError('Frozen previous source/data changed: '+name)
    paths=set(ROOT.glob('src/n01/*.py'))
    paths.update(ROOT/'scripts'/name for name in ['audit_noise_origin.py','run_noise_origin_003.py','launch_noise_origin_003.sh','register_noise_origin_003.py','freeze_noise_origin_003.py'])
    paths.update([ROOT/'scripts/calibrate_real_real_null.py', ROOT/'scripts/supervise.py',
                  ROOT/'tests/test_noise_origin.py',ROOT/'tests/test_real_real_null.py',
                  ROOT/'noise_origin_003_config.json',ROOT/'data/PARENT_IDENTITIES.npz'])
    for name in ['NOISE_ORIGIN_003_PROTOCOL.md','NOISE_ORIGIN_003_PAIRED_ADDENDUM.md','NOISE_ORIGIN_SOURCE_CHAIN.md','NOISE_ORIGIN_METHOD.md','REAL_REAL_NULL_METHOD.md','NOISE_ORIGIN_QA_PLAN.md']:
        paths.add(ROOT/'docs'/name)
    paths.update((ROOT/'provenance/noise_origin_003_reference').glob('*.py'))
    paths.update([RECEIPTS/'PREREGISTRATION.json',RECEIPTS/'PAIRED_ADDENDUM_REGISTRATION.json',RECEIPTS/'TEST_SUMMARY.json',RECEIPTS/'INDEPENDENT_REVIEW.json'])
    files=[]
    for p in sorted(paths):
        if not p.is_file():
            raise FileNotFoundError(p)
        files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    write(binding,{'experiment_id':ID,'frozen_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),
        'before_new_scientific_execution':True,'preregistration_unchanged':True,'old_executed_source_and_identity_unchanged':True,
        'files':files,'noise_training_updates':0,'response_training_updates':0,'classifier_updates_planned':25600})
    out=ROOT/'runs'/ID
    out.mkdir(parents=True,exist_ok=True)
    package=out/'EXECUTION_SOURCE.tar.gz'
    if package.exists():
        raise FileExistsError(package)
    with tarfile.open(package,'w:gz') as archive:
        for p in sorted(paths|{binding}):
            archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    write(RECEIPTS/'EXECUTION_PACKAGE.json',{'path':package.relative_to(ROOT).as_posix(),'bytes':package.stat().st_size,
        'sha256':sha(package),'members':len(paths)+1,'implementation_binding_sha256':sha(binding)})
    print(json.dumps({'package':str(package),'sha256':sha(package),'members':len(paths)+1}))

if __name__=='__main__':
    main()
