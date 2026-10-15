"""Rebind local READY after the reviewed single-line bit-algebra correction."""
from pathlib import Path
from datetime import datetime,timezone
import difflib
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;POST=HERE.parent/'postrun_independent_01'
BEFORE=HERE/'before_saved_residual_bits_fix01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def ref(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def replace_preserved(path,original,value):
    assert path.read_bytes()==original.read_bytes(),'preserve unexpected READY revision'
    with path.open('w',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

focused_path=HERE/'artificial_residual_bits_01/ENGINEERING_RECEIPT.json';focused=read(focused_path)
assert focused['passed'] and focused['exact_one_line_bytes_changed'] and focused['full_suites_rerun'] is False
source=ref(POST/'readback_saved_arrays.py');assert source==focused['worker']
oldworker=read(BEFORE/'READBACK_WORKER_READY.json');assert oldworker['worker']['sha256']=='bc8b6b5f15080049fe6d9da275392f08b33a6fc5b484975c8724ac47d96e5d09'
for key in ('controller','kernel','plan','inventory'):assert sha(Path(oldworker[key]['path']))==oldworker[key]['sha256']
worker=dict(oldworker,UTC=datetime.now(timezone.utc).isoformat(),worker=source,source_revision='saved_residual_bits_fix01',
    prior_READY=ref(BEFORE/'READBACK_WORKER_READY.json'),focused_delta_engineering=ref(focused_path),
    current_source_evidence='Prior full suites plus byte-identical remainder and focused one-line regression; full suites not rerun.',
    saved_residual_algebra='Exact uint64 bit equality of float64(mu-saved_CPU_sum) and saved residual; no tolerance.',
    focused_positive_cases=focused['positive_cases'],focused_negative_cases=focused['negative_cases'])
replace_preserved(POST/'READBACK_WORKER_READY.json',BEFORE/'READBACK_WORKER_READY.json',worker)
controller=read(BEFORE/'CONTROLLER_READY.json');controller.update(UTC=datetime.now(timezone.utc).isoformat(),worker=source,
    worker_ready=ref(POST/'READBACK_WORKER_READY.json'),source_revision='saved_residual_bits_fix01',
    prior_READY=ref(BEFORE/'CONTROLLER_READY.json'),focused_worker_delta_engineering=ref(focused_path),
    controller_source_and_prior_stdlib_suite_unchanged=True,full_suites_rerun=False)
replace_preserved(HERE/'CONTROLLER_READY.json',BEFORE/'CONTROLLER_READY.json',controller)
old=ROOT/'receipts/wp3_015_20261015/postrun_independent_01/readback_saved_arrays.py';new=POST/'readback_saved_arrays.py'
with (HERE/'WORKER_DELTA.patch').open('w',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(old.read_text('utf8').splitlines(True),new.read_text('utf8').splitlines(True),fromfile=str(old),tofile=str(new))))
handoff=read(BEFORE/'SOURCE_REVIEW_HANDOFF.json');handoff.update(UTC=datetime.now(timezone.utc).isoformat(),
    worker_ready=ref(POST/'READBACK_WORKER_READY.json'),controller_ready=ref(HERE/'CONTROLLER_READY.json'),
    source_revision='saved_residual_bits_fix01',focused_worker_delta=ref(focused_path),full_suites_rerun=False,
    prior_handoff=ref(BEFORE/'SOURCE_REVIEW_HANDOFF.json'))
handoff['sources']['worker']=source
handoff['deltas']=[ref(HERE/'WORKER_DELTA.patch'),ref(HERE/'CONTROLLER_DELTA.patch')]
replace_preserved(HERE/'SOURCE_REVIEW_HANDOFF.json',BEFORE/'SOURCE_REVIEW_HANDOFF.json',handoff)
print(json.dumps(dict(worker=source,controller=worker['controller'],worker_READY_sha256=sha(POST/'READBACK_WORKER_READY.json'),
    controller_READY_sha256=sha(HERE/'CONTROLLER_READY.json'),handoff_sha256=sha(HERE/'SOURCE_REVIEW_HANDOFF.json'),
    focused_receipt_sha256=sha(focused_path)),ensure_ascii=False))
