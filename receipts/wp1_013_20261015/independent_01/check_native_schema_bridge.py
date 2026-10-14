"""Tiny artificial check of the two corrected012-to-WP1 interfaces only."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT/'receipts/wp1_013_20261015/independent_01/native_schema_bridge_01'
sys.path.insert(0,str(ROOT/'scripts'))
import wp1_boundary_saved as boundary


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if OUT.exists():raise RuntimeError('Preserve previous independent fixture')
    policy = json.loads((ROOT/'mechanism012_policy.json').read_bytes())
    rows = dict(source_parent_row=np.repeat(np.arange(3,dtype=np.int64),2),
                source_parent_rows=np.array([8,137,6550],dtype=np.int64),
                source_parent_ids=np.array([424,1337,6542],dtype=np.int64))
    ids = boundary.row_identities(rows,2)
    checks = [dict(name='third_path_uses_actual_native012_contract',passed=tuple(policy['probe']['paths'])==boundary.PROBE_PATHS),
              dict(name='localordinal_maps_to_originalTRAIN_row_and_source_ID',
                   passed=np.array_equal(ids['source_parent_ordinal'],np.repeat(np.arange(3),2))
                       and np.array_equal(ids['source_original_TRAIN_row'],np.repeat(rows['source_parent_rows'],2))
                       and np.array_equal(ids['source_id'],np.repeat(rows['source_parent_ids'],2))),
              dict(name='six_saved_integer_and_fractional_step_key_names',passed=[boundary.saved_prefix(s,t) for s in (.1,1,10) for t in (-1,1)]==
                   ['probe_0p1_minus','probe_0p1_plus','probe_1_minus','probe_1_plus','probe_10_minus','probe_10_plus'])]
    bad = {**rows,'source_parent_row':np.repeat(rows['source_parent_rows'],2)}
    rejected = False
    try:boundary.row_identities(bad,2)
    except ValueError:rejected=True
    checks.append(dict(name='originalTRAIN_rows_cannot_masquerade_as_localordinal',passed=rejected))
    receipt = dict(schema='n01/wp1-013-independent-native-schema-artificial/v1',passed=all(x['passed'] for x in checks),
                   status='PASS' if all(x['passed'] for x in checks) else 'FAIL',artificial_only=True,
                   artificial_rows=6,artificial_parent_count=3,checks=checks,
                   scripts_sha256={p:sha(ROOT/p) for p in ('scripts/wp1_input_information.py','scripts/wp1_boundary_saved.py')},
                   actual_scientific_array_decodes=0,checkpoint_reads=0,NN_forward_rows=0,AD_queries=0,
                   updates=0,DEV_values_read=0,SSH_calls=0,old_pipeline_replays=0)
    OUT.mkdir(parents=True,exist_ok=False)
    target=OUT/'ENGINEERING_RECEIPT.json'
    with target.open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    assert json.loads(target.read_bytes())['passed'] is True
    print(json.dumps(dict(path=str(target),sha256=sha(target),status=receipt['status'],checks=len(checks))))


if __name__=='__main__':main()
