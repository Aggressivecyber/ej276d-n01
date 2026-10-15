"""Create reviewable014 bindings and caps; drafts cannot dispatch science."""
from pathlib import Path, PurePosixPath
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import sys

ROOT=Path('E:/GAN/research/N01')
REMOTE='/home/rylan/projects/ej276d-n01/wp2_014_20261015'
OLD013='/home/rylan/projects/ej276d-n01/wp1_013_repeat_fix01_20261015'
OUT=ROOT/'receipts/wp2_014_20261015'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('root014_factory',ROOT/'scripts/run_wp2_014.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
old=read(ROOT/'wp1_013_config.json')
p=module.settings014()
p.update(status='draft_registered_before_any_scientific_operation',registered_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
    remote_root=REMOTE,source_files={},protection_files={},
    user_authority=dict(path='USER_MASTER_PLAN_20261015.txt',sha256=sha(ROOT/'USER_MASTER_PLAN_20261015.txt')),
    master_plan='MASTER_EXECUTION_PLAN.json',
    source_classification='010 architecture with newly calibrated initial head; frozen A and005; noGAN',
    previous_wp1_result=dict(path='receipts/wp1_013_20261015/postrun_root_01/FINAL_RESULT_RECEIPT.json',sha256=sha(ROOT/'receipts/wp1_013_20261015/postrun_root_01/FINAL_RESULT_RECEIPT.json'),method_choice='adequate'),
    cumulative_training_before_run=159667,
    preflight=dict(minimum_free_disk_GiB=8,minimum_available_RAM_GiB=10,GPU_must_be_empty=True),
    candidate_trial_order=['seed0_C0','seed0_C1','seed1_C0','seed1_C1'],
    C2_enabled=False,
    long_job_policy='Verify actual firstG1 and same-snapshot health, save one future wake, then rest without agent monitoring.',
    collection_policy=dict(mode='inventory_verified_selective',selected_archive_MiB=512,
        include_head_checkpoints=True,head_checkpoint_maximum_MiB=64,
        raw_NPZ_retained_remote=True,all_raw_collected_claim_allowed=False,
        independent_raw_saved_array_readback_required_before_final_numeric_acceptance=True))
p['evaluation']['bank_seeds']=copy.deepcopy(old['reference']['bank_seeds'])
p['evaluation']['projection_seed']=old['reference']['projection']['seed']
p['evaluation']['projection_fingerprint']=old['reference']['projection']['fingerprint']
p['inputs']={k:copy.deepcopy(old['inputs'][k]) for k in ['dataset','statistics','data_statistics','base_config','deterministic','noise005','training_complete']}
source_info=ROOT/'runs/wp1_013_20261015/runtime_fix01_collected_03/runs/primary/information/PARENT_INFORMATION.npz'
p['inputs']['source_information']=dict(path=OLD013+'/runs/primary/information/PARENT_INFORMATION.npz',
    local_path=str(source_info),sha256=sha(source_info),bytes=source_info.stat().st_size)
def contract(rel):
    local=ROOT/rel
    return dict(path=REMOTE+'/'+rel,local_path=str(local),sha256=sha(local),bytes=local.stat().st_size)
p['contracts']={k:contract(rel) for k,rel in {
    'observation':'OBSERVATION_CONTRACT.json',
    'data_roles':'WP2_DATA_ROLES.json',
    'acceptance':'docs/WP2_ACCEPTANCE_CONTRACT.md',
    'tolerance_table':'CALIBRATED_TOLERANCES_WP1_013.json',
    'configuration':'docs/WP2_014_CONFIGURATION.md',
    'input_information_audit':'INPUT_INFORMATION_AUDIT.json',
    'preceding_wp1_result':'receipts/wp1_013_20261015/postrun_root_01/FINAL_RESULT_RECEIPT.json',
}.items()}
roots={REMOTE,OLD013}
prefix=PurePosixPath('/home/rylan/projects/ej276d-n01')
for row in list(p['inputs'].values())+list(old['protection_files'].values()):
    native=PurePosixPath(row['path']);assert native.is_relative_to(prefix)
    roots.add(str(prefix/native.relative_to(prefix).parts[0]))
p['allowed_input_roots']=sorted(roots)
p['budget_caps']=module.budget_caps014(p)
for row in list(p['inputs'].values())+list(p['contracts'].values()):
    local=Path(row['local_path']);assert local.drive.upper() in ('E:','D:')
    assert local.stat().st_size==row['bytes'] and sha(local)==row['sha256']
assert p['budget_caps']['head_total_kernel_rows']==8653376
assert p['budget_caps']['G_updates']==8000 and p['budget_caps']['new_AD_queries']==9872
assert p['budget_caps']['hard_ADC_rows']==26480928 and p['budget_caps']['checkpoint_loads']==2
target=ROOT/'wp2_014_config.json';assert not target.exists()
with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(p,f,ensure_ascii=False,indent=2);f.write('\n')
receipt=dict(schema='n01/wp2-controls-preparation/v1',UTC=p['registered_UTC'],status='draft_ready_for_review',
    config_sha256=sha(target),budget_caps=p['budget_caps'],source_freeze_pending=True,
    input_file_count=len(p['inputs']),contract_file_count=len(p['contracts']),
    numpy_imported=False,torch_imported=False,checkpoint_loads=0,model_forwards=0,AD_queries=0,training_updates=0,
    scientific_dispatch_authorized_only_after_final_frozen_source_and_native_preflight=True)
with (OUT/'CONTROL_PREPARATION_RECEIPT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(receipt))
