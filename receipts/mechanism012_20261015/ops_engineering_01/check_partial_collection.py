"""Stdlib-only failed-output archive check; injected native headers, zero science."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('ops_partial_fixture',ROOT/'scripts/mechanism012_ops.py')
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=ops.fresh_directory(args.output,local=True);owned=ops.fresh_directory(output/'artificial_owned',local=True)
    artifacts=owned/'runs/primary/artifacts';artifacts.mkdir(parents=True)
    source=Path(__file__).parent/'cpu_02/artificial_failed'
    for name in ('operations.jsonl','status.json'):
        with (source/name).open('rb') as src,(artifacts/name).open('xb') as dst:shutil.copyfileobj(src,dst)
    (owned/'START_RECEIPT.json').write_text('{"artificial_only":true,"supervisor_pid":12345}',encoding='utf8')
    native=dict(exit_code=1,child_native_exit_code=-9,reason='injected_SIGKILL',error=None,artificial_only=True)
    supervisor=owned/'runs/primary/supervisor';supervisor.mkdir();ops.write_new(supervisor/'exit_receipt.json',native)
    progress=ops.operation_progress(artifacts)
    state={'runs/primary/supervisor/exit_receipt.json':native,'durable_child_progress':progress,'protection_errors':['injected readonly hash mismatch'],
        'current_boot_id':'artificial_current_boot', 'START_RECEIPT.json':dict(boot_id='different_mock_boot',supervisor_pid=12345)}
    inventory,verified=ops.collection_inventory(owned,state,partial=True)
    ops.require(verified['status']=='partial_evidence_only' and verified['protection_errors'] and verified['actual_inflight_completion']=='unknown',
        'Partial collection concealed failure or unknown state')
    ops.require(progress['completed_counts_durable_lower_bound']['old_A_rows']==128
        and progress['completed_counts_durable_lower_bound']['head_final_rows']==0 and progress['unknown_inflight_operation_ids']==[1],
        'Partial counter evidence changed')
    archive=output/'artificial_partial.tar'
    with tarfile.open(archive,'x:') as tar:
        for name in sorted(inventory):tar.add(owned/name,arcname=name,recursive=False)
    target=ops.exclusive_extract(archive,output/'artificial_collected',inventory,144*1024**2)
    restored=ops.operation_progress(target/'runs/primary/artifacts')
    ops.require(restored==progress,'Archived partial evidence changed')
    # No Python FAILED is required after supervisor KILL or a reboot.
    state['runs/primary/supervisor/exit_receipt.json']=None
    ops.require(ops.collection_inventory(owned,state,partial=True)[1]['status']=='partial_evidence_only','Reboot no-FAILED evidence refused')
    try:ops.collection_inventory(owned,{},partial=False)
    except ValueError:pass
    else:raise AssertionError('Partial output accepted as complete')
    result=dict(schema='n01/mechanism012-ops-partial-artificial/v1',status='PASS',artificial_only=True,passed_checks=5,
        native_headers_injected=True,raw_partial_evidence_preserved=True,hash_error_preserved=True,reboot_without_FAILED_supported=True,
        unknown_inflight_not_completed=True,archive_SHA256=ops.sha(archive),inventory=inventory,
        ops_SHA256=ops.sha(ROOT/'scripts/mechanism012_ops.py'),actual_SSH_SCP_calls=0,numeric_decodes=0,checkpoint_loads=0,NN_rows=0,AD_queries=0,optimizer_steps=0)
    ops.write_new(output/'ENGINEERING_RECEIPT.json',result);print(json.dumps(dict(status='PASS',passed_checks=5,output=str(output))))

if __name__=='__main__':main()
