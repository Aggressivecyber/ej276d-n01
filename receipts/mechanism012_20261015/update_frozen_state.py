from datetime import datetime,timezone
from pathlib import Path
import importlib.util,json,hashlib

ROOT=Path('E:/GAN/research/N01')
R=ROOT/'receipts/mechanism012_20261015'
spec=importlib.util.spec_from_file_location('register_012',R/'register_012.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
freeze=helper.read(R/'freeze_01/FREEZE_RECEIPT.json')
review_path=R/'independent_01/PRELAUNCH_REVIEW.json'
review=helper.read(review_path)
assert review.get('status')=='PASS'
assert helper.sha(review_path)=='def26aef53f63b9b18635c909e795af20cf329a34845310d1159b2e95c4b78da'
assert freeze['source_file_count']==69 and freeze['input_count']==36
assert freeze['D_backup_SHA256_verified'] is True and not freeze['scientific_started']
assert helper.sha(ROOT/'MECHANISM012_SOURCE_MANIFEST.json')==freeze['source_manifest_sha256']
assert helper.sha(ROOT/'MECHANISM012_BINDINGS.json')==freeze['bindings_sha256']
utc=datetime.now(timezone.utc).isoformat()
links=dict(mechanism012_source_manifest='MECHANISM012_SOURCE_MANIFEST.json',mechanism012_bindings='MECHANISM012_BINDINGS.json',
    mechanism012_freeze_receipt='receipts/mechanism012_20261015/freeze_01/FREEZE_RECEIPT.json',
    mechanism012_independent_review='receipts/mechanism012_20261015/independent_01/PRELAUNCH_REVIEW.json')
for path in (ROOT/'PROGRAM.json',Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')):
    value=helper.read(path)
    value.update(status='mechanism012_frozen_not_started',current_stage='mechanism012_frozen_deployment',
        updated_UTC=utc,active_scientific_jobs=0,mechanism012_source_freeze_pending=False,
        mechanism012_source_manifest_sha256=freeze['source_manifest_sha256'],mechanism012_bindings_sha256=freeze['bindings_sha256'],
        mechanism012_source_package_sha256=freeze['package_sha256'],
        next_action='Deploy the frozen012 source and verify immutable native inputs; launch once after frozen package bridge PASS, verify first formal AD proof then schedule one wake and rest.')
    for key,relative in links.items():value[key]=str(ROOT/relative) if path.parent!=ROOT else relative
    helper.write(path,value)
jobs=helper.read(ROOT/'jobs.json')
target=[job for job in jobs['jobs'] if job.get('run_id')=='mechanism012_20261015']
assert len(target)==1
target[0].update(status='frozen_not_started',source_freeze_pending=False,source_manifest_sha256=freeze['source_manifest_sha256'],
    bindings_sha256=freeze['bindings_sha256'],source_package_sha256=freeze['package_sha256'],
    freeze_receipt=links['mechanism012_freeze_receipt'],independent_review=links['mechanism012_independent_review'])
jobs['updated_UTC']=utc;helper.write(ROOT/'jobs.json',jobs)
print(json.dumps(dict(status='frozen_not_started',source69=69,inputs36=36,actual_NN=0,actual_AD=0)))
