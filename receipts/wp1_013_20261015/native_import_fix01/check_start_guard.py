"""Same-snapshot lifecycle regression, stdlib only and SSH always mocked."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[3]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    output=args.out.resolve()
    if output.drive.upper() not in ('E:','D:') or output.exists():raise ValueError('Fresh E/D artifact directory required')
    output.mkdir(parents=True)
    path=ROOT/'scripts/wp1_controller.py';spec=importlib.util.spec_from_file_location('wp1_repair_controller_guard',path)
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    original=json.loads((ROOT/'receipts/wp1_013_20261015/start_verification_01/REMOTE_STATE.json').read_text('utf8'))
    config=json.loads((ROOT/'wp1_013_config.json').read_text('utf8'))
    # These deliberate in-memory control injections test the lifecycle rule;
    # they are never saved as a real execution/start success or frozen source.
    receipt={k:'0'*64 for k in ('source_package_sha256','private_package_sha256','config_sha256','source_manifest_sha256','controller_sha256','legacy_ops_sha256')}
    receipt['artificial_only']=False # mock boundary; no native action can occur
    healthy=copy.deepcopy(original);healthy['runs/primary/artifacts/FAILED.json']=None
    healthy['runs/primary/artifacts/status.json']['phase']='full_TRAIN_reference';healthy['runs/primary/artifacts/status.json']['error']=None
    healthy['runs/primary/supervisor/exit_receipt.json']=None
    cases=[('healthy_first',healthy,True)]
    completed=copy.deepcopy(healthy);completed['runs/primary/artifacts/status.json']['phase']='complete'
    completed['runs/primary/supervisor/exit_receipt.json']=dict(child_native_exit_code=0,exit_code=0,reason=None,error=None)
    cases.append(('native_zero_compatible',completed,True))
    failed=copy.deepcopy(healthy);failed['runs/primary/artifacts/FAILED.json']=original['runs/primary/artifacts/FAILED.json']
    cases.append(('same_FIRST_plus_FAILED',failed,False))
    phase=copy.deepcopy(healthy);phase['runs/primary/artifacts/status.json']['phase']='failed'
    cases.append(('same_FIRST_plus_failed_status',phase,False))
    child=copy.deepcopy(healthy);child['runs/primary/supervisor/exit_receipt.json']=dict(child_native_exit_code=1,exit_code=1)
    cases.append(('same_FIRST_plus_nonzero_native',child,False))
    supervisor=copy.deepcopy(healthy);supervisor['runs/primary/supervisor/exit_receipt.json']=dict(child_native_exit_code=0,exit_code=1)
    cases.append(('same_FIRST_plus_supervisor_nonzero',supervisor,False))
    checks=[]
    for name,state,accepted in cases:
        target=output/name
        with mock.patch.object(controller,'load_freeze',return_value=(receipt,(config,None,None,None))),mock.patch.object(controller,'ssh_action',return_value=state) as remote,mock.patch.object(controller,'proof_valid',wraps=controller.proof_valid) as proof:
            try:result=controller.controller_action('verify-start',output/'unused_mock_freeze',target)
            except ValueError as error:
                assert not accepted and 'same snapshot already failed' in str(error);assert not proof.called
                assert not (target/'START_VALIDATION.json').exists()
            else:
                assert accepted and proof.call_count==1 and result['status']=='actual_first128_family_verified'
                assert (target/'START_VALIDATION.json').is_file()
            assert remote.call_count==1 and json.loads((target/'REMOTE_STATE.json').read_text('utf8'))==state
            # All values are metadata only; immutable saved native snapshot is
            # preserved unchanged, including rejection cases.
            checks.append(dict(name=name,passed=True,control_fields_injected_for_artificial_contract=True,
                snapshot_preserved=True,proof_evaluated=accepted,actual_SSH_calls=0))
    # Read-only status remains available even if its snapshot already failed.
    with mock.patch.object(controller,'load_freeze',return_value=(receipt,(config,None,None,None))),mock.patch.object(controller,'ssh_action',return_value=failed):
        result=controller.controller_action('status',output/'unused_mock_freeze',output/'status_after_failed')
    assert result['status']=='one_snapshot_saved'
    checks.append(dict(name='failed_snapshot_readonly_status_still_available',passed=True))
    value=dict(schema='n01/wp1-start-lifecycle-artificial/v1',status='PASS',artificial_only=True,passed_checks=len(checks),checks=checks,
        source_sha256=sha(path),saved_snapshot_sha256=sha(ROOT/'receipts/wp1_013_20261015/start_verification_01/REMOTE_STATE.json'),
        actual_scientific_data_decodes=0,actual_scientific_CP_loads=0,actual_scientific_NN_rows=0,actual_AD_queries=0,actual_training_updates=0,
        actual_SSH_calls=0,Torch_imported=False,NumPy_imported=False,scientific_execution_claim=False)
    (output/'ENGINEERING_RECEIPT.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',passed_checks=len(checks),controller_sha256=value['source_sha256'])))

if __name__=='__main__':main()
