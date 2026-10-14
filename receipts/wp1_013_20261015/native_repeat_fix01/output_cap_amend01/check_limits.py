"""Targeted stdlib-only output/collection boundary, all sizes are mocks."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest import mock

ROOT=Path(__file__).resolve().parents[4]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve()
    if out.drive.upper() not in ('E:','D:') or out.exists():raise ValueError('Fresh E/D fixture required')
    out.mkdir(parents=True)
    path=ROOT/'scripts/wp1_controller.py';spec=importlib.util.spec_from_file_location('wp1_output_amend_check',path)
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    p=json.loads((ROOT/'wp1_013_config.json').read_text('utf8'));p['resource_limits']['output_MiB']=1536
    assert c.SOURCE_LIMIT==16*1024**2 and c.PRIVATE_LIMIT==64*1024**2 and c.COLLECTION_LIMIT==1552*1024**2
    assert c.COLLECTION_LIMIT-p['resource_limits']['output_MiB']*1024**2==16*1024**2
    c.validate_config(p,frozen=False);checks=[dict(name='1536_output_1552_collection_and_unchanged_source_private_caps',passed=True)]
    for amount in (1024,1535,1537):
        bad=copy.deepcopy(p);bad['resource_limits']['output_MiB']=amount
        try:c.validate_config(bad,frozen=False)
        except ValueError:pass
        else:raise AssertionError('Wrong exact output cap accepted')
    checks.append(dict(name='old1024_and_adjacent1535_1537_config_values_rejected',passed=True))
    bad=copy.deepcopy(p);bad['resource_limits']['CPU_threads']=13
    try:c.validate_config(bad,frozen=False)
    except ValueError:pass
    else:raise AssertionError('Unrelated CPU cap changed')
    checks.append(dict(name='other_registered_resource_limits_still_exact',passed=True))
    tree=out/'virtual_collection';(tree/'runs/primary').mkdir(parents=True);(tree/'runs/primary/toy.bin').write_bytes(b'x')
    state={'START_RECEIPT.json':{'boot_id':'old_mock_boot'},'current_boot_id':'new_mock_boot',
        'durable_child_progress':{},'protection_errors':[]}
    def call(amount):
        with mock.patch.object(c,'REMOTE',str(tree)),mock.patch.object(c,'named_record',return_value={'bytes':amount,'sha256':'0'*64}):
            return c.archive_inventory(state,p,partial=True)
    inventory,_=call(c.COLLECTION_LIMIT);assert sum(r['bytes'] for r in inventory.values())==1552*1024**2
    checks.append(dict(name='actual_archive_inventory_accepts_exact_collection_bound_with_virtual_size',passed=True))
    try:call(c.COLLECTION_LIMIT+1)
    except ValueError as e:assert 'collection output cap exceeded' in str(e)
    else:raise AssertionError('Collection bound plus one accepted')
    checks.append(dict(name='actual_archive_inventory_rejects_collection_bound_plus_one_with_virtual_size',passed=True))
    assert 'numpy' not in sys.modules and 'torch' not in sys.modules
    value=dict(schema='n01/wp1-output-cap-amend-stdlib/v1',status='PASS',artificial_only=True,checks=checks,passed_checks=len(checks),
        controller_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),output_MiB=1536,collection_MiB=1552,control_margin_MiB=16,
        source_MiB=16,private_MiB=64,CPU_threads=12,GPU_GiB=10,RSS_GiB=8,wall_seconds=7200,
        virtual_byte_sizes_not_actual_gigabyte_allocation=True,no_scientific_retests=True,raw_data_decodes=0,CP_loads=0,
        Torch_imported=False,NumPy_imported=False,NN_rows=0,ADC_calls=0,AD_queries=0,training_updates=0,SSH_calls=0)
    (out/'ENGINEERING_RECEIPT.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',passed_checks=len(checks),controller_sha256=value['controller_sha256'])))

if __name__=='__main__':main()
