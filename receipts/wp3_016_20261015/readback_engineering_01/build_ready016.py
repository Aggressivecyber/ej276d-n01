"""Create local worker/controller handoff receipts; never register or execute."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;POST=HERE.parent/'postrun_independent_01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def ref(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def save(p,obj):
    mode='x'
    if p.name=='READBACK_WORKER_READY.json' and p.exists():
        assert p.read_bytes()==(HERE/'before_canonical_residual_fix01/READBACK_WORKER_READY.json').read_bytes(),'preserve unrecognized READY revision'
        mode='w'
    with p.open(mode,encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

plan=read(POST/'READBACK_PLAN.json');assert sha(POST/'READBACK_PLAN.json')=='9fd6ed29156c5567ef9738835e00fb06f3c2a214bb5694a267b22f06b34e938e'
paths=dict(worker=POST/'readback_saved_arrays.py',kernel=POST/'saved_array_kernels.py',plan=POST/'READBACK_PLAN.json',
    controller=HERE/'readback_control.py',inventory=HERE.parent/'collection_01/REMOTE_FULL_INVENTORY.json',scope_note=POST/'SCOPE_NOTE.json')
references={k:ref(v) for k,v in paths.items()}
assert references['kernel']['sha256']=='fe1366f35f36e97974f1a4037a36ad858c03dd8453957c1b45b65080f0fee3ac'
assert references['inventory']['sha256']==plan['inventory']['sha256']
engineering=read(HERE/'artificial_charge_02/ENGINEERING_RECEIPT.json');assert engineering['passed']
assert engineering['worker']=={k:references['worker'][k] for k in ('bytes','sha256')}
if sys.argv[1]=='worker':
    result=dict(schema='n01/wp3-independent-readback-worker-ready/v1',status='ready_for_root_registration_not_execution',
        UTC=datetime.now(timezone.utc).isoformat(),**references,source116_frozen_config_sha256=plan['frozen_configuration']['sha256'],
        manifest_sha256=plan['frozen_manifest']['sha256'],expected_complete_counters=plan['expected_complete_counters'],expected_operations=200,
        actual_schema='n01/wp3-readback-state/v1',first_proof_schema='n01/wp3-readback-first-formal-pair/v1',result_schema='n01/wp3-independent-raw-saved-array-readback/v1',
        CLI='python -B readback_saved_arrays.py --run-root '+plan['original_root']+' --inventory <bound inventory JSON> --out '+plan['sidecar_root']+'/artifacts',
        result_files=['status.json','operations.jsonl','FIRST_PAIR_PROOF.json','ARRAY_READBACK.json','COMPLETE.json','FAILED.json'],
        scientific_budget=plan['scientific_budget'],resource_budget=plan['finite_resource_budget'],source_before_import=True,
        new_charge_checks_independent_of_producer_functions=True,producer_functions_imported=False,
        kernel_byte_exact_015=True,mature_numeric_and_durable_methods_AST_unchanged=engineering['unchanged_mature_methods'],
        artificial_engineering=ref(HERE/'artificial_charge_02/ENGINEERING_RECEIPT.json'),artificial_groups=len(engineering['checks']),artificial_negative_cases=engineering['negative_count'],
        original_real_raw_CP_NN_AD_RNG_GPU_SSH=0,old015_numeric_batch_rerun=False,
        source_digest_scope='Saved digest repetition/crossbank/ID consistency only; no independent SOURCE waveform read/re-hash.',
        charge_r8_added=False,full_stochastic_preADC_or_Z_replayed=False,
        new_external_import_helpers=[],remote_file_slots=['readback_control.py','readback_saved_arrays.py','saved_array_kernels.py','READBACK_PLAN.json','BOUND_ORIGINAL_INVENTORY.json'],
        execution_control_sixth_slot='READBACK_EXECUTION.json',scientific_acceptance=False)
    save(POST/'READBACK_WORKER_READY.json',result)
    print(json.dumps(dict(worker_ready_sha=sha(POST/'READBACK_WORKER_READY.json'),sources=references),ensure_ascii=False))
elif sys.argv[1]=='controller':
    check=read(HERE/'artificial_stdlib_02/ENGINEERING_RECEIPT.json');assert check['passed']
    assert check['controller']=={k:references['controller'][k] for k in ('bytes','sha256')}
    result=dict(schema='n01/wp3-readback-controller-ready/v1',status='ready_for_root_registration_not_execution',
        UTC=datetime.now(timezone.utc).isoformat(),**references,worker_ready=ref(POST/'READBACK_WORKER_READY.json'),
        controller_artificial_engineering=ref(HERE/'artificial_stdlib_02/ENGINEERING_RECEIPT.json'),
        numeric_artificial_engineering=ref(HERE/'artificial_charge_02/ENGINEERING_RECEIPT.json'),
        expected_operations=200,expected_complete_counters=plan['expected_complete_counters'],
        raw_file_SHA_passes=1,raw_decode_passes=1,resource_limits=dict(CPU_threads=2,RSS_MiB=1536,virtual_memory_MiB=3072,wall_seconds=1800,output_MiB=64),
        remote_root=plan['sidecar_root'],original_root=plan['original_root'],old015_read_only=True,
        slots=5,execution_control_slots_total=6,new_remote_helper_dependencies=[],
        scientific_budget=plan['scientific_budget'],source_review_pending=True,registered_or_remote_actions_performed=False,
        original_raw_NPZ_or_CP_tensor_reads=0,SSH=0,scientific_acceptance=False)
    save(HERE/'CONTROLLER_READY.json',result)
    print(json.dumps(dict(controller_ready_sha=sha(HERE/'CONTROLLER_READY.json'),source_references=references),ensure_ascii=False))
else:raise ValueError('worker/controller only')
