"""Bind accepted017 literals and existing input hashes without numeric decoding."""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
REMOTE='/home/rylan/projects/ej276d-n01/wp4_017_20261015'
PROPOSAL_SHA='b2db64dfd3549f6cd22cb5253a1e374738248012e22554308ec26885d2d42e8e'
CONTRACT_REVIEW_SHA='31a0a106bf033ec7d363abc13811a7f0e6dd87e79d139ea7ad6c867a67927fd7'
OWNED=('scripts/run_wp4_017.py','scripts/wp4_017_controller.py','src/n01/model017.py',
       'src/n01/stats017.py','src/n01/saved_array_kernels017.py')

def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)
def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
def checked(path,row):
    assert path.is_file() and sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],str(path)
def flatten_budget(value):
    result=copy.deepcopy(value);normals=result.pop('scalar_normal_requests')
    result.update({'normal_'+key+'_scalar_requests':v for key,v in normals.items()})
    return result

def main():
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    boundary_path=OUT/'REMOTE_EXECUTION_DEFERRED_BY_USER.json';boundary=read(boundary_path)
    assert boundary['remote_execution_authorized'] is False
    assert read(ROOT/'PROGRAM.json')['new_training_updates']==179667
    assert read(ROOT/'jobs.json')['active_scientific_jobs']==0
    assert read(ROOT/'receipts/wp3_016_20261015/FINAL_RESULT_RECEIPT.json')['status']=='016_bounded_training_array_audit_and_report_complete'
    proposal_path=OUT/'planning_01/ENGINEERING_PROPOSAL.json'
    contract_review_path=OUT/'planning_01/INDEPENDENT_CONTRACT_REVIEW.json'
    assert sha(proposal_path)==PROPOSAL_SHA and sha(contract_review_path)==CONTRACT_REVIEW_SHA
    proposal=read(proposal_path);contract_review=read(contract_review_path)
    assert contract_review['passed'] and not contract_review['blockers']
    review_path=OUT/'independent_01/SOURCE_REVIEW.json';review=read(review_path)
    assert review['passed'] and not review['blockers']
    records=review['sources']
    assert set(OWNED)<=set(records)
    for name,row in records.items():checked(ROOT/name,row)
    protected=read(OUT/'engineering_01/PROTECTED_BEFORE.json')['files'];assert len(protected)==124
    for name,row in protected.items():checked(ROOT/name,row)
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('root017_factories',ROOT/'scripts/run_wp4_017.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    p=module.settings017();full=module.budget_caps017(p);first=module.budget_caps017(p,stage1=True)
    assert full==flatten_budget(proposal['exact_planned_budgets']['full_eight_trial_envelope'])
    assert first==flatten_budget(proposal['exact_planned_budgets']['stage1_gate'])
    assert p['trial_order'][:2]==['T1_paired_seed0','T1_paired_seed1'] and len(p['trial_order'])==8
    assert p['qref']==3.7103998046755806 and p['trainable_parameters']==2
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    p.update(registered_UTC=stamp,source_files={},protection_files={},budget_caps=full,stage1_budget_caps=first,
        execution_scope='local_package_only',native_launch_authorized=False,
        local_user_execution_boundary=ref(boundary_path),
        user_authority=dict(path='USER_MASTER_PLAN_20261015.txt',sha256=sha(ROOT/'USER_MASTER_PLAN_20261015.txt')),
        master_plan='MASTER_EXECUTION_PLAN.json',cumulative_training_before_run=179667,
        source_classification='known fixed A/005/seed0 M0; exactly2charge scalars; teacher recovery; no adversarial training',
        teacher_trial_budget_separate_from_WP3_short_candidates=True,
        global_completed_short_trials_before=10,WP3_completed_short_trials_before=6,
        planned_teacher_trials=8,first_stage_teacher_trials=2,remaining_teacher_trials_after_all_required_stage1_pass=6,
        stage1_pending_policy='Do not enter stage2; preserve MC_unresolved and support reasons.',
        preflight=dict(minimum_free_disk_GiB=40,minimum_available_RAM_GiB=10,GPU_must_be_empty=True),
        previous016_result=ref(ROOT/'receipts/wp3_016_20261015/FINAL_RESULT_RECEIPT.json'),
        previous016_final_development_selection=ref(ROOT/'receipts/wp3_016_20261015/postrun_independent_01/FINAL_DEVELOPMENT_SELECTION.json'),
        latest_model_policy=ref(ROOT/'receipts/wp3_015_20261015/postrun_root_02/AGENT_MODEL_POLICY_SOLMAX.json'),
        long_job_policy='Verify first real update of both theta and current same-snapshot owned process/resource health; set one future n01 wake and rest.',
        no_automatic_scientific_retry=True,
        collection_policy=dict(mode='inventory_verified_selective',selected_archive_MiB=256,
            include_head_checkpoints=True,raw_NPZ_retained_remote=True,
            saved_array_independent_readback_required_before_final_numeric_acceptance=True))
    previous=read(ROOT/'wp3_016_config.json')
    names=('dataset','statistics','data_statistics','base_config','deterministic','noise005','M0_head_seed0')
    p['inputs']={name:copy.deepcopy(previous['inputs'][name]) for name in names}
    for name in ('deterministic','noise005','M0_head_seed0'):
        assert p['inputs'][name]==proposal['fixed_bases'][name]
    for row in p['inputs'].values():
        path=Path(row['local_path']);assert path.drive.upper() in ('E:','D:');checked(path,row)
    contracts={
        'execution_boundary':'receipts/wp4_017_20261015/REMOTE_EXECUTION_DEFERRED_BY_USER.json',
        'implementation_clarifications':'receipts/wp4_017_20261015/planning_01/ROOT_IMPLEMENTATION_CLARIFICATIONS.json',
        'proposal':'receipts/wp4_017_20261015/planning_01/ENGINEERING_PROPOSAL.json',
        'contract_review':'receipts/wp4_017_20261015/planning_01/INDEPENDENT_CONTRACT_REVIEW.json',
        'root_contract_decision':'receipts/wp4_017_20261015/planning_01/ROOT_CONTRACT_DECISION.json',
        'configuration':'docs/WP4_017_CONFIGURATION.md','acceptance':'docs/WP4_017_CONFIGURATION.md',
        'data_roles':'WP4_017_DATA_ROLES.json','observation':'OBSERVATION_CONTRACT.json',
        'tolerance_table':'CALIBRATED_TOLERANCES_WP1_013.json','input_information_audit':'INPUT_INFORMATION_AUDIT.json',
        'inherited_general_roles':'DATA_ROLES.json','inherited_full_acceptance':'docs/ACCEPTANCE_CONTRACT.md',
        'preceding016_result':'receipts/wp3_016_20261015/FINAL_RESULT_RECEIPT.json',
        'preceding016_selection':'receipts/wp3_016_20261015/postrun_independent_01/FINAL_DEVELOPMENT_SELECTION.json',
        'agent_model_policy':'receipts/wp3_015_20261015/postrun_root_02/AGENT_MODEL_POLICY_SOLMAX.json'}
    p['contracts']={key:dict(path=REMOTE+'/'+name,local_path=str(ROOT/name),
        sha256=sha(ROOT/name),bytes=(ROOT/name).stat().st_size) for key,name in contracts.items()}
    p['allowed_input_roots']=sorted(set(previous['allowed_input_roots'])|{REMOTE})
    module.validate_settings017(p,frozen=False)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    target=ROOT/'wp4_017_config.json';write(target,p)
    write(OUT/'CONTROL_PREPARATION_RECEIPT.json',dict(schema='n01/wp4-017-control-preparation/v1',UTC=stamp,
        status='017_local_only_draft_inputs_roles_budget_bound_before_numeric_decode',configuration_sha256=sha(target),
        source_review=ref(review_path),contract_review=ref(contract_review_path),
        accepted_proposal=ref(proposal_path),bound_original_inputs=len(names),local_input_SHA_verified=len(names),
        contracts=len(contracts),original124_controls_unchanged=True,
        stage1_budget_caps=first,budget_caps=full,scientific_operations=0,
        new_numpy_torch_imports=False,source_freeze_pending=True,scientific_acceptance=False,
        remote_execution_authorized=False,user_execution_boundary=ref(boundary_path)))
    print(json.dumps(dict(status='017_draft_ready_for_freeze',configuration_sha256=sha(target),
        inputs=len(names),stage1_G_AD_cap=4000,full_G_AD_cap=16000,scientific_operations=0)))

if __name__=='__main__':main()
