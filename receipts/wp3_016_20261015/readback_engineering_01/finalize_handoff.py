"""Final local source/delta identities for independent review; no execution."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import difflib
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;POST=HERE.parent/'postrun_independent_01';OLD=ROOT/'receipts/wp3_015_20261015'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def ref(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
before=HERE/'before_canonical_residual_fix01'
old_delta=HERE/'WORKER_DELTA.patch'
with (before/'WORKER_DELTA.patch').open('xb') as f:f.write(old_delta.read_bytes())
for source,target,delta in [(OLD/'postrun_independent_01/readback_saved_arrays.py',POST/'readback_saved_arrays.py',HERE/'WORKER_DELTA.patch'),
    (OLD/'readback_engineering_01/readback_control.py',HERE/'readback_control.py',HERE/'CONTROLLER_DELTA.patch')]:
    with delta.open('w',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(source.read_text('utf8').splitlines(True),target.read_text('utf8').splitlines(True),fromfile=str(source),tofile=str(target))))
worker=read(POST/'READBACK_WORKER_READY.json');controller=read(HERE/'CONTROLLER_READY.json')
for key in ('worker','kernel','plan','controller','inventory'):
    assert worker[key]==controller[key] and sha(Path(worker[key]['path']))==worker[key]['sha256']
test=read(HERE/'artificial_charge_02/ENGINEERING_RECEIPT.json');std=read(HERE/'artificial_stdlib_02/ENGINEERING_RECEIPT.json')
assert test['passed'] and std['passed']
manifest=read(ROOT/'WP3_016_SOURCE_MANIFEST.json');assert len(manifest['files'])==116
assert all(sha(ROOT/name)==row['sha256'] for name,row in manifest['files'].items())
raw=read(HERE.parent/'collection_01/REMOTE_FULL_INVENTORY.json')['omitted_remote_inventory'];pairs={}
for name in raw:
    if name.endswith('_HARD_ROWS.npz'):
        parts=name.split('/');stem=parts[-1];bank=int(stem[4]);key='/'.join(parts[:-1])+'/'+stem[6:]
        pairs.setdefault(key,[]).append(bank)
assert len(pairs)==24 and all(sorted(v)==[0,1] for v in pairs.values())
result=dict(schema='n01/wp3-016-readback-source-review-handoff/v1',UTC=datetime.now(timezone.utc).isoformat(),status='READY_FOR_INDEPENDENT_SOURCE_REVIEW_NO_EXECUTION',
    worker_ready=ref(POST/'READBACK_WORKER_READY.json'),controller_ready=ref(HERE/'CONTROLLER_READY.json'),
    sources={k:worker[k] for k in ('worker','kernel','controller','plan','inventory')},
    deltas=[ref(HERE/'WORKER_DELTA.patch'),ref(HERE/'CONTROLLER_DELTA.patch')],
    artificial_charge=ref(HERE/'artificial_charge_02/ENGINEERING_RECEIPT.json'),artificial_stdlib=ref(HERE/'artificial_stdlib_02/ENGINEERING_RECEIPT.json'),
    artificial_charge_groups=len(test['checks']),artificial_charge_negative_cases=test['negative_count'],stdlib_groups=std['passed_count'],
    new_charge_independent_NumPy_formulas=True,producer_numeric_functions_imported=False,kernel_original_bytes=True,
    canonical_residual_predeclared_bridge='Independently reduced CPU sum and saved CPU sum both checked against derived double-reduction bound; stored residual checked algebraically against saved GPU mu minus saved CPU sum; bound formula64eps/atol0; no relative-only comparison of cancellation residuals from different legal reductions.',
    canonical_bridge_artificial_1ULP_positive_and_two_negative_cases_passed=True,
    expected_operations=200,expected_complete_counters=worker['expected_complete_counters'],exact24hard_bank_pairs_verified_from_paths=True,
    prior_015_sources_unmodified=True,current116_frozen_sources_unchanged=True,
    helper_fragment_is_local_assembly_only=True,deployed_external_helpers=[],remote_file_slots=5,execution_control_sixth_slot=True,
    actual_private_raw_CP_NN_AD_RNG_GPU_SSH=0,root_registration_or_start_performed=False,scientific_acceptance=False)
with (HERE/'SOURCE_REVIEW_HANDOFF.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(handoff_sha=sha(HERE/'SOURCE_REVIEW_HANDOFF.json'),worker_ready_sha=sha(POST/'READBACK_WORKER_READY.json'),controller_ready_sha=sha(HERE/'CONTROLLER_READY.json'),exact_pairs=len(pairs))))
