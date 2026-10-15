"""Pure stdlib sidecar controls and actual Audit writer bridge; no arrays."""
from __future__ import annotations
import argparse
import base64
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


def require(test,message):
    if not test:raise AssertionError(message)


def rejects(function,message):
    try:function()
    except (ValueError,AssertionError):return
    raise AssertionError(message)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    repo=args.repo.resolve();out=args.out.resolve();require(not out.exists() and out.drive.upper() in ('E:','D:'),'fresh E/D fixture')
    out.mkdir(parents=True);(out/'tmp').mkdir();os.environ.update(TEMP=str(out/'tmp'),TMP=str(out/'tmp'),PYTHONDONTWRITEBYTECODE='1')
    source=repo/'receipts/wp2_014_20261015/readback_engineering_01/readback_control.py'
    spec=importlib.util.spec_from_file_location('readback_control_artificial',source);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    directory=repo/'receipts/wp2_014_20261015/postrun_independent_01'
    locals={'readback_control.py':source,'readback_saved_arrays.py':directory/'readback_saved_arrays.py','saved_array_kernels.py':directory/'saved_array_kernels.py',
        'READBACK_PLAN.json':directory/'READBACK_PLAN.json','BOUND_ORIGINAL_INVENTORY.json':repo/'receipts/wp2_014_20261015/collection_01/REMOTE_FULL_INVENTORY.json'}
    p=dict(schema='n01/wp2-independent-readback-execution/v1',run_id=c.RUN_ID,remote_root=c.SIDE,original_root=c.ORIGINAL,
        resource_limits=c.LIMITS,automatic_retry=False,original_files_read_only=True,
        files={n:dict(path=c.SIDE+'/'+n,local_path=str(path),**c.record(path)) for n,path in locals.items()})
    c.validate(p);bad=copy.deepcopy(p);bad['resource_limits']=dict(c.LIMITS,CPU_threads=3)
    rejects(lambda:c.validate(bad),'changed readback scope accepted')
    bad=copy.deepcopy(p);bad['remote_root']=c.ORIGINAL+'/independent_readback_01'
    rejects(lambda:c.validate(bad),'output inside original tree accepted')
    passed=['exact_5_files_sidecar_outside_original_and_CPU_resource_contract']

    # Bootstrap has its SHA in the stdin payload, not a second required CLI flag.
    previous_argv=sys.argv;previous_native=c.native;seen=[]
    try:
        payload=dict(mode='prepare',config_sha256='a'*64,controller_sha256=c.sha(source),token='0'*32)
        sys.argv=[str(source),'--bootstrap',base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()]
        c.native=lambda value:(seen.append(value) or {'status':'synthetic_bootstrap_parsed'})
        with contextlib.redirect_stdout(io.StringIO()):require(c.main()==0,'bootstrap parser failed')
        require(seen==[payload],'bootstrap payload changed')
    finally:sys.argv=previous_argv;c.native=previous_native
    passed.append('real_main_bootstrap_parser_accepts_SHA_in_payload_without_remote_action')

    # Actual final producer Audit constructor/base/op/state, artificial-only data.
    worker=locals['readback_saved_arrays.py'];spec=importlib.util.spec_from_file_location('readback_actual_writer_artificial',worker)
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    original=out/'manufactured_empty_original';original.mkdir();side=out/'manufactured_side';side.mkdir()
    audit=w.Audit(SimpleNamespace(run_root=original,out=side/'artifacts',artificial_only=True))
    inventory=json.loads(locals['BOUND_ORIGINAL_INVENTORY.json'].read_text());audit.inventory_sha=c.sha(locals['BOUND_ORIGINAL_INVENTORY.json'])
    for name,row in inventory['omitted_remote_inventory'].items():
        audit.op('SYNTHETIC_RAW_SHA_COUNTER_NO_IO',dict(raw_files_SHA_checked=1,raw_bytes_SHA_read=row['bytes']),lambda:None,not_IO=True)
    for _ in range(2):audit.op('SYNTHETIC_BASELINE_COUNTER_NO_ARRAY',dict(baseline_npz_files_decoded=1),lambda:None)
    def fake_pair():
        for _ in range(2):audit.op('SYNTHETIC_BANK_COUNTER_NO_ARRAY',dict(hard_rows_files_decoded=1,report_npz_files_decoded=1,waveform_rows_decoded=26372),lambda:None)
    audit.op('SYNTHETIC_PAIR_COUNTER_NO_ARRAY',dict(bank_pairs_verified=1),fake_pair)
    proof=dict(audit.base(),schema='n01/wp2-readback-first-formal-pair/v1',actual_saved_array_pair_completed=True,no_extra_tiny_actual_input=True,actual_new_CP_NN_AD=0)
    w.write_new(audit.out/'FIRST_PAIR_PROOF.json',proof)
    state={'START.json':dict(boot_id='synthetic',supervisor_pid=900000),'supervisor/launch.json':dict(supervisor_pid=900000,child_pid=os.getpid()),
        'current_boot_id':'synthetic','supervisor/exit_receipt.json':None,'artifacts/FIRST_PAIR_PROOF.json':copy.deepcopy(proof),'artifacts/status.json':audit.base(),'artifacts/FAILED.json':None}
    rejects(lambda:c.first_valid(state,p),'artificial first pair blessed')
    state['artifacts/FIRST_PAIR_PROOF.json']['artificial_only']=False
    c.first_valid(state,p)
    for kind in ('failed_file','status_failed','native_nonzero','unknown_pair'):
        bad=copy.deepcopy(state)
        if kind=='failed_file':bad['artifacts/FAILED.json']={'error':'manufactured'}
        if kind=='status_failed':bad['artifacts/status.json']['native_status']='FAILED'
        if kind=='native_nonzero':bad['supervisor/exit_receipt.json']={'child_native_exit_code':1,'exit_code':1}
        if kind=='unknown_pair':bad['artifacts/FIRST_PAIR_PROOF.json']['inflight']={'x':{'name':'unknown'}}
        rejects(lambda bad=bad:c.first_valid(bad,p),'historical/unknown first pair accepted')
    passed.append('actual_Audit_writer_firstpair_identity_and_same_snapshot_failure_guard')

    for _ in range(14):audit.op('SYNTHETIC_BASELINE_COUNTER_NO_ARRAY',dict(baseline_npz_files_decoded=1),lambda:None)
    for _ in range(35):audit.op('SYNTHETIC_PAIR_COUNTER_NO_ARRAY',dict(bank_pairs_verified=1),fake_pair)
    for _ in range(8):audit.op('SYNTHETIC_PROBE_COUNTER_NO_ARRAY',dict(probe_raw_npz_files_decoded=1,probe_gradient_npz_files_decoded=1,probe_snapshots_verified=1),lambda:None)
    require(audit.entered==audit.completed==308,'synthetic scope annotations not308')
    w.write_new(audit.out/'ARRAY_READBACK.json',dict(artificial_only=True,synthetic_counter_annotations_not_IO=True,actual_arrays=0))
    complete=dict(audit.base(),native_status='COMPLETE',passed=True,ARRAY_READBACK_sha256=c.sha(audit.out/'ARRAY_READBACK.json'),scientific_acceptance=False)
    w.write_new(audit.out/'COMPLETE.json',complete)
    shutil.copyfile(locals['BOUND_ORIGINAL_INVENTORY.json'],side/'BOUND_ORIGINAL_INVENTORY.json')
    state.update({'artifacts/COMPLETE.json':copy.deepcopy(complete),'supervisor/exit_receipt.json':dict(child_native_exit_code=0,exit_code=0,reason=None,error=None),
        'durable_progress':c.legacy().journal_tools().durable_progress(audit.out)})
    old_side=c.SIDE;c.SIDE=str(side)
    try:
        rejects(lambda:c.complete_valid(state,p),'artificial complete blessed')
        state['artifacts/COMPLETE.json']['artificial_only']=False;c.complete_valid(state,p)
        bad=copy.deepcopy(state);bad['artifacts/COMPLETE.json']['counters']['bank_pairs_verified']=35
        rejects(lambda:c.complete_valid(bad,p),'incomplete pair scope blessed')
        bad=copy.deepcopy(state);bad['durable_progress']['unknown_inflight_ids']=[999]
        rejects(lambda:c.complete_valid(bad,p),'unknown durable operation blessed')
    finally:c.SIDE=old_side
    passed.append('actual_Audit_308_operations_terminal_counter_and_durable_unknown_bridge')
    require(not {'torch','numpy','n01'}&set(sys.modules),'numerical package imported by operational fixtures')
    receipt=dict(schema='n01/wp2-readback-controller-engineering/v1',artificial_only=True,passed_count=len(passed),passed_checks=passed,
        controller_sha256=c.sha(source),worker_sha256=c.sha(worker),kernel_sha256=c.sha(locals['saved_array_kernels.py']),plan_sha256=c.sha(locals['READBACK_PLAN.json']),
        fullinventory_sha256=c.sha(locals['BOUND_ORIGINAL_INVENTORY.json']),producer_writer_imported=True,numerical_packages_imported=False,
        synthetic_counter_annotations_not_IO=True,positive_native_markers_in_memory_only=True,
        actual_operations=dict(original_CP_reads=0,raw_NPZ_reads=0,array_decodes=0,NN_rows=0,AD_queries=0,new_random_draws=0,training_updates=0,SSH_calls=0),old_frozen_sources_modified=False)
    c.core().write_new(out/'ENGINEERING_RECEIPT.json',receipt)
    print(json.dumps(dict(status='readback_controller_synthetic_interfaces_passed',checks=len(passed),controller_sha256=receipt['controller_sha256'],receipt_sha256=c.sha(out/'ENGINEERING_RECEIPT.json'))))


if __name__=='__main__':main()
