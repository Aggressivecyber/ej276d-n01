"""Close the inherited provisional saved-row flag; preserve its prior bytes."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
root = here.parents[2]
current = root.parents[1] / 'CURRENT_SIM2REAL_PROGRAM.json'
snap = here / 'selection_status_fix01'
snap.mkdir(exist_ok=False)
selection = here / 'FINAL_DEVELOPMENT_SELECTION.json'
metadata = [root / n for n in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json')] + [current]
for p in [selection] + metadata:
    shutil.copy2(p, snap / p.name)
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
write = lambda p,d: p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old_sha = digest(selection)
d = read(selection)
assert d['independent_raw_readback_complete'] is False and not d['weights_promoted']
d.update(independent_raw_readback_complete=True, bounded_saved_row_readback_complete=True,
    independent_raw_readback_scope='Registered36-pair/176-saved-NPZ row/moment/covariance/MCSE/gradient audit completed; not complete loss replay.',
    full_distance_replay_from_saved_rows=False, C1_median_fit_independently_recomputed=False)
write(selection, d)
new_sha, size = digest(selection), selection.stat().st_size

def update_refs(node):
    if isinstance(node, dict):
        if node.get('sha256') == old_sha and str(node.get('path','')).replace('\\','/').endswith('postrun_root_02/FINAL_DEVELOPMENT_SELECTION.json'):
            node['sha256'], node['bytes'] = new_sha, size
        for v in node.values(): update_refs(v)
    elif isinstance(node, list):
        for v in node: update_refs(v)

for p in metadata:
    value=read(p);update_refs(value);write(p,value)
write(snap/'FIX_RECEIPT.json',dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
    reason='An inherited provisional saved-row completion flag remained false; decision/gates/costs unchanged.',
    before_sha256=old_sha, after_sha256=new_sha, bounded_saved_row_readback_complete=True,
    full_distribution_loss_replay_complete=False, C1_scale_refit=False, new_G_AD_NN_CP=0,
    cumulative_training_updates=167667, weights_promoted=False, scientific_acceptance=False,
    updated_metadata={str(p):digest(p) for p in metadata}))
print(json.dumps(dict(selection_sha256=new_sha, bytes=size, bounded_saved_row_readback_complete=True,
    full_loss_replay=False, new_G_AD_NN_CP=0)))
