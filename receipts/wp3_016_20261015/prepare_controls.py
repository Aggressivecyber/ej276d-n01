"""Bind016's reviewed design and existing input bytes without numeric decoding."""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REMOTE = '/home/rylan/projects/ej276d-n01/wp3_016_20261015'
OLD015 = '/home/rylan/projects/ej276d-n01/wp3_015_20261015'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), bytes=path.stat().st_size)


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    assert read(ROOT/'PROGRAM.json')['new_training_updates'] == 175667
    assert read(ROOT/'jobs.json')['active_scientific_jobs'] == 0
    assert read(ROOT/'receipts/wp3_015_20261015/FINAL_RESULT_RECEIPT.json')['status'] == '015_engineering_experiment_bounded_audit_and_report_complete'
    design = read(ROOT/'wp3_016_design.json')
    review_path = OUT/'independent_01/SOURCE_REVIEW.json'
    design_review_path = OUT/'independent_01/DESIGN_REVIEW.json'
    review, design_review = read(review_path), read(design_review_path)
    assert review['passed'] and not review['blockers'] and design_review['passed']
    for name, row in review['sources'].items():
        path = ROOT/name
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], name
    spec = importlib.util.spec_from_file_location('root016_factory', ROOT/'scripts/wp3_016_controller.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p, budget = module.factory_contract((ROOT/'scripts/run_wp3_016.py').read_bytes())
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    assert budget(p) == design['budget_caps_design'] == review['budget_caps']
    old = read(ROOT/'wp3_015_config.json')
    assert p['evaluation']['bank_seeds'] == old['evaluation']['bank_seeds']
    assert p['evaluation']['projection_seed'] == old['evaluation']['projection_seed']
    for new_seed, old_seed in zip(p['seed_sets'], old['seed_sets']):
        assert all(new_seed[k] == v for k,v in old_seed.items())
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    p.update(status='draft_registered_before_any_scientific_operation', registered_UTC=stamp, remote_root=REMOTE,
        source_files={}, protection_files={},
        user_authority=dict(path='USER_MASTER_PLAN_20261015.txt',sha256=sha(ROOT/'USER_MASTER_PLAN_20261015.txt')),
        master_plan='MASTER_EXECUTION_PLAN.json',
        source_classification='frozen originalA/005/015M0; two positive mean-corrected charge coefficients only; noGAN',
        cumulative_training_before_run=175667, global_completed_short_trials_before=8,
        WP3_completed_short_trials_before=4, WP3_reserved_short_trials_this_block=2,
        WP3_remaining_unreserved_maximum=0, candidate_trial_order=['seed0_Q1','seed1_Q1'],
        preflight=dict(minimum_free_disk_GiB=8,minimum_available_RAM_GiB=10,GPU_must_be_empty=True),
        previous015_result=ref(ROOT/'receipts/wp3_015_20261015/FINAL_RESULT_RECEIPT.json'),
        previous015_final_development_selection=ref(ROOT/'receipts/wp3_015_20261015/postrun_independent_02/FINAL_DEVELOPMENT_SELECTION.json'),
        latest_model_policy=ref(ROOT/'receipts/wp3_015_20261015/postrun_root_02/AGENT_MODEL_POLICY_SOLMAX.json'),
        long_job_policy='Verify first real two-parameter optimizer update and current same-snapshot health, set one future n01 wake, then rest.',
        collection_policy=dict(mode='inventory_verified_selective',selected_archive_MiB=256,include_head_checkpoints=True,
            head_checkpoint_maximum_MiB=64,raw_NPZ_retained_remote=True,all_raw_collected_claim_allowed=False,
            independent_raw_saved_array_readback_required_before_final_numeric_acceptance=True))
    base_keys=('dataset','statistics','data_statistics','base_config','deterministic','noise005','training_complete','source_information','integral_statistics')
    p['inputs']={k:copy.deepcopy(old['inputs'][k]) for k in base_keys}
    inventory_path=ROOT/'receipts/wp3_015_20261015/collection_01/REMOTE_FULL_INVENTORY.json'
    assert sha(inventory_path)=='9d253a57bb79d9f25b112b4f5e0009a239c21360766c89cef7357d040adebe42'
    inventory=read(inventory_path)['whole_remote_inventory']
    selected=ROOT/'runs/wp3_015_20261015/selected_collected_01'
    for seed in range(2):
        name=f'runs/primary/artifacts/seed{seed}_M0/final.pt'
        row=dict(path=OLD015+'/'+name, **inventory[name], local_path=str(selected/name))
        p['inputs'][f'M0_head_seed{seed}']=row
    assert len(p['inputs'])==11
    def contract(rel):
        path=ROOT/rel
        return dict(path=REMOTE+'/'+rel,local_path=str(path),sha256=sha(path),bytes=path.stat().st_size)
    p['contracts']={key:contract(name) for key,name in {
        'observation':'OBSERVATION_CONTRACT.json', 'data_roles':'WP3_016_DATA_ROLES.json',
        'acceptance':'docs/WP3_016_CONFIGURATION.md', 'configuration':'docs/WP3_016_CONFIGURATION.md',
        'tolerance_table':'CALIBRATED_TOLERANCES_WP1_013.json', 'stage_design':'wp3_016_design.json',
        'input_information_audit':'INPUT_INFORMATION_AUDIT.json',
        'inherited_general_roles':'DATA_ROLES.json', 'inherited_full_acceptance':'docs/ACCEPTANCE_CONTRACT.md',
        'preceding015_result':'receipts/wp3_015_20261015/FINAL_RESULT_RECEIPT.json',
        'preceding015_selection':'receipts/wp3_015_20261015/postrun_independent_02/FINAL_DEVELOPMENT_SELECTION.json',
        'agent_model_policy':'receipts/wp3_015_20261015/postrun_root_02/AGENT_MODEL_POLICY_SOLMAX.json',
    }.items()}
    p['allowed_input_roots']=sorted(set(old['allowed_input_roots'])|{REMOTE})
    p['budget_caps']=budget(p)
    p['development_selection']=dict(bank_distance_aggregation='mean2bank_distances_not_pooled8',
        noninferiority_cells='ALL6_same_candidate_2seeds_x3hypotheses',
        noninferiority_components='each_required_positive_defined_T',
        noninferiority_references=['paired_A','paired015_M0_charge_off'],
        noninferiority_bound='mean2bank(final_distance-reference_distance)<=fixed_T',
        required_calibrated_metric_missing_or_undefined='pending_not_eligible_not_pass',
        zero_or_undefined_T='unresolved_never_automatic_PASS',
        score='arithmetic_mean12_final_hard_wave_distances',reference_score='arithmetic_mean12_paired_M0_hard_wave_distances',
        replace_M0_only_if_eligible_and_score_below_reference_by_more_than=1e-12,
        ties='retain_M0_charge_disabled',initial_charge_on_reference='reported_separately',
        relative_wave_usefulness='ALL6_mean2bank_improvement_vsA>=.02_report_separately',
        absolute_T_and_scientific_acceptance='separate_not_implied',weights_promoted=False)
    for row in p['inputs'].values():
        path=Path(row['local_path'])
        assert path.drive.upper() in ('E:','D:') and path.is_file()
        assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes']
    module.validate_config(p,frozen=False)
    target=ROOT/'wp3_016_config.json'
    write(target,p)
    write(OUT/'CONTROL_PREPARATION_RECEIPT.json',dict(schema='n01/wp3-016-control-preparation/v1',UTC=stamp,
        status='draft_inputs_roles_budget_bound_before_numeric_decode',config_sha256=sha(target),
        inputs=11,local_input_SHA_verified=11,remote_only_inputs=0,contracts=len(p['contracts']),
        source_review=ref(review_path),design_review=ref(design_review_path),budget_caps=p['budget_caps'],
        source_freeze_pending=True,new_numpy_torch_imports=False,new_NPZ_CP_NN_AD_G_random=0,
        cumulative_training_updates=175667,scientific_acceptance=False))
    print(json.dumps(dict(status='draft016_ready_for_freeze',configuration_sha256=sha(target),inputs=11,
        G_AD_cap=4000,CP_cap=4,scientific_operations=0)))


if __name__=='__main__':
    main()
