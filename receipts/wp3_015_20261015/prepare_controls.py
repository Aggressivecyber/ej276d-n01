"""Root registration of 015 inputs/roles/caps; no scientific numeric decode."""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
REMOTE='/home/rylan/projects/ej276d-n01/wp3_015_20261015'
OLD014='/home/rylan/projects/ej276d-n01/wp2_014_20261015'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
stamp=dt.datetime.now(dt.timezone.utc).isoformat()

def write(path,value):
    assert not path.exists(),str(path)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')

def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)

spec=importlib.util.spec_from_file_location('root015_factory',ROOT/'scripts/wp3_controller.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
p,budget=module.factory_contract((ROOT/'scripts/run_wp3_015.py').read_bytes())
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
old=read(ROOT/'wp2_014_config.json')
ready_path=OUT/'implementation_01/READY.json';ready=read(ready_path)
review_path=OUT/'independent_01/SOURCE_REVIEW.json';review=read(review_path)
for name,row in ready['sources'].items():assert sha(ROOT/name)==row['sha256']
assert review.get('passed') is True or review.get('status')=='PASS'
assert read(ROOT/'PROGRAM.json')['new_training_updates']==167667
assert read(ROOT/'jobs.json')['active_scientific_jobs']==0
design_path=ROOT/'wp3_015_design.json'
shutil.copy2(design_path,OUT/'design_root_01/before_final_registration.json')
design=read(design_path)
design.update(status='implementation_ready_pending_final_configuration_freeze',updated_UTC=stamp,
    exact_NN_AD_CP_caps_pending_implementation=False,exact_budget_caps=budget(p),
    implementation=ref(ready_path),independent_source_review=ref(review_path),computation_started=False)
with design_path.open('w',encoding='utf-8',newline='\n') as stream:
    json.dump(design,stream,ensure_ascii=False,indent=2);stream.write('\n')

p.update(status='draft_registered_before_any_scientific_operation',registered_UTC=stamp,remote_root=REMOTE,
    source_files={},protection_files={},
    user_authority=dict(path='USER_MASTER_PLAN_20261015.txt',sha256=sha(ROOT/'USER_MASTER_PLAN_20261015.txt')),
    master_plan='MASTER_EXECUTION_PLAN.json',
    source_classification='frozen originalA/005 and014C1 research reference; paired shape continuation or6parameter deterministic mean adapter; noGAN',
    previous_wp2_result=ref(ROOT/'receipts/wp2_014_20261015/FINAL_RESULT_RECEIPT.json'),
    cumulative_training_before_run=167667,global_completed_short_trials_before=4,
    WP3_reserved_short_trials_this_block=4,WP3_remaining_unreserved_maximum=2,
    preflight=dict(minimum_free_disk_GiB=8,minimum_available_RAM_GiB=10,GPU_must_be_empty=True),
    candidate_trial_order=['seed0_M0','seed0_M1','seed1_M0','seed1_M1'],
    long_job_policy='Verify first realG1 and same-snapshot health; save one future wake in the same n01 then rest. A correctly completed negative mean gate is collected immediately.',
    collection_policy=dict(mode='inventory_verified_selective',selected_archive_MiB=512,
        include_head_checkpoints=True,head_checkpoint_maximum_MiB=64,raw_NPZ_retained_remote=True,
        all_raw_collected_claim_allowed=False,independent_raw_saved_array_readback_required_before_final_numeric_acceptance=True))
p['evaluation']=copy.deepcopy(old['evaluation'])
p['inputs']=copy.deepcopy(old['inputs'])
inventory_path=ROOT/'receipts/wp2_014_20261015/collection_01/REMOTE_FULL_INVENTORY.json'
assert sha(inventory_path)=='4cf27f6c771294561a474ef2de9574ab711d9322c588da488f0d9ce2e49792d5'
inventory=read(inventory_path)['whole_remote_inventory']
selected=ROOT/'runs/wp2_014_20261015/selected_collected_01'

def old_input(key,name,local):
    row=dict(path=OLD014+'/'+name,**inventory[name])
    if local:
        path=selected/name;assert path.is_file() and sha(path)==row['sha256']
        assert path.stat().st_size==row['bytes'];row['local_path']=str(path)
    else:row['remote_only_saved014_REPORT']=True
    p['inputs'][key]=row

old_input('integral_statistics','runs/primary/artifacts/INTEGRAL_TRAIN_STATISTICS.json',True)
for seed in range(2):
    old_input('C1_head_seed%d'%seed,'runs/primary/artifacts/seed%d_C1/final.pt'%seed,True)
    for bank in range(2):
        for h in ('fast','slow','common_level'):
            old_input(f'mean_seed{seed}_bank{bank}_{h}',
                f'runs/primary/artifacts/seed{seed}_C1/final_evaluation/bank{bank}_final_{h}_REPORT.npz',False)
assert len(p['inputs'])==23
target_inventory=ROOT/'WP2_REMOTE_FULL_INVENTORY.json'
assert not target_inventory.exists();shutil.copy2(inventory_path,target_inventory)
p['remote_inventory_evidence']=dict(path=REMOTE+'/WP2_REMOTE_FULL_INVENTORY.json',
    local_path=str(inventory_path),sha256=sha(inventory_path),bytes=inventory_path.stat().st_size)

def contract(rel):
    local=ROOT/rel
    return dict(path=REMOTE+'/'+rel,local_path=str(local),sha256=sha(local),bytes=local.stat().st_size)

p['contracts']={key:contract(rel) for key,rel in {
    'observation':'OBSERVATION_CONTRACT.json',
    'data_roles':'WP3_DATA_ROLES.json',
    'acceptance':'docs/WP3_ACCEPTANCE_CONTRACT.md',
    'tolerance_table':'CALIBRATED_TOLERANCES_WP1_013.json',
    'configuration':'docs/WP3_015_CONFIGURATION.md',
    'input_information_audit':'INPUT_INFORMATION_AUDIT.json',
    'preceding_wp1_result':'receipts/wp1_013_20261015/postrun_root_01/FINAL_RESULT_RECEIPT.json',
    'preceding_wp2_result':'receipts/wp2_014_20261015/FINAL_RESULT_RECEIPT.json',
    'stage_design':'wp3_015_design.json',
    'inherited_general_roles':'DATA_ROLES.json',
    'inherited_full_acceptance':'docs/ACCEPTANCE_CONTRACT.md',
}.items()}
p['allowed_input_roots']=sorted(set(old['allowed_input_roots'])|{REMOTE})
p['budget_caps']=budget(p)
p['development_selection']=dict(
    bank_distance_aggregation='mean_of2_bank_distances_not_pooled8',
    noninferiority_cells='ALL6_same_candidate_2seeds_x3hypotheses',
    noninferiority_components='each_positive_defined_T_component',
    noninferiority_references=['paired_A','common_initial_C1_reference'],
    noninferiority_bound='mean2bank(final_distance-reference_distance)<=fixed_T',
    required_calibrated_metric_missing_or_undefined='pending_not_eligible_not_pass',
    uncalibrated_component='pending_never_automatic_PASS',
    score='arithmetic_mean12_final_hard_wave_distances',tie_absolute_epsilon=1e-12,tie_preference='M0',
    replace_research_reference_only_if_score_below_common_initial_by_more_than=1e-12,
    otherwise='retain014_C1_reference_mean_disabled',
    relative_wave_usefulness='ALL6_mean2bank_improvement_vsA>=.02_report_separately',
    absolute_T_and_full_scientific_acceptance='separate_not_implied_by_research_reference_selection',
    reference_selection_implies_weight_promotion=False)
for row in p['inputs'].values():
    if row.get('local_path'):
        local=Path(row['local_path']);assert local.drive.upper() in ('E:','D:')
        assert sha(local)==row['sha256'] and local.stat().st_size==row['bytes']
    else:module.remote_input_binding015(row,read(inventory_path))
module.validate_config(p,frozen=False)
target=ROOT/'wp3_015_config.json';write(target,p)
write(OUT/'CONTROL_PREPARATION_RECEIPT.json',dict(schema='n01/wp3-controls-preparation/v1',UTC=stamp,
    status='draft_ready_for_final_review_freeze',config_sha256=sha(target),budget_caps=p['budget_caps'],
    source_freeze_pending=True,input_file_count=23,local_hash_verified_input_files=11,remote_only_REPORTs=12,
    remote_only_binding='Original318 inventory metadata only; native will still SHA each physical input before decode.',
    remote_inventory_evidence=p['remote_inventory_evidence'],contract_file_count=len(p['contracts']),
    original014_eight_inputs_preserved=True,numpy_imported=False,torch_imported=False,
    checkpoint_loads=0,model_forwards=0,AD_queries=0,training_updates=0,
    cumulative_training_updates=167667,scientific_acceptance=False))
print(json.dumps(dict(status='draft_ready_for_final_review_freeze',config_sha256=sha(target),inputs=len(p['inputs']),
    remote_only_REPORTs=12,contracts=len(p['contracts']),G_AD_caps=8000,CP_cap=4,numerical_decodes=0)))
