"""Register one reviewed storage/numerical repair before any new native run."""
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=HERE.parent
NEW_REMOTE='/home/rylan/projects/ej276d-n01/wp3_016_runtime_fix02_20261015'


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)


def write(path,value,*,new=False):
    with path.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def main():
    failed_path=OUT/'NATIVE_FAILURE_RECONCILIATION.json'
    failure=read(failed_path)
    ready_path=HERE/'ENGINEERING_FIX_READY.json'
    review_path=HERE/'INDEPENDENT_FIX_REVIEW.json'
    ready,review=read(ready_path),read(review_path)
    assert review['passed'] and not review['blockers']
    for name,row in review['sources'].items():
        assert sha(ROOT/name)==row['sha256'] and (ROOT/name).stat().st_size==row['bytes'],name
    program=read(ROOT/'PROGRAM.json');jobs=read(ROOT/'jobs.json')
    assert program['new_training_updates']==175667 and program['active_scientific_jobs']==jobs['active_scientific_jobs']==0
    job=next(j for j in jobs['jobs'] if j['job_id']=='N01_WP3_016_20261015')
    assert job['status']=='terminal_native1_before_first_G_update'
    assert failure['actual_completed_counts']['G_updates']==failure['actual_completed_counts']['new_AD_queries']==0
    assert failure['active_scientific_jobs']==0 and failure['unknown_inflight_ids']==[]
    p=read(ROOT/'wp3_016_config.json');old=copy.deepcopy(p)
    previous_source=read(OUT/'SOURCE_COMMIT_READBACK_FIX01.json')
    assert p['status']=='frozen' and sha(ROOT/'wp3_016_config.json')==previous_source['configuration_sha256']
    assert p['remote_root']!=NEW_REMOTE
    before=HERE/'before_canonical_revision';before.mkdir(exist_ok=False)
    for name in ('wp3_016_config.json','WP3_016_SOURCE_MANIFEST.json'):
        shutil.copy2(ROOT/name,before/name)
    spec=importlib.util.spec_from_file_location('root016_fix02_factory',ROOT/'scripts/wp3_016_controller.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    settings,budget_factory=module.factory_contract((ROOT/'scripts/run_wp3_016.py').read_bytes())
    # Scientific settings stay fixed; a new explicit numerical/storage contract may be added.
    for key in ('run_id','candidates','seed_sets','G_updates','TRAIN_counts','training','charge','calibration',
                'evaluation','resource_limits','run_resource_limits','development_selection'):
        if key in settings:
            assert settings[key]==old[key],key
    assert sha(ROOT/'src/n01/model016.py')==previous_source['files']['src/n01/model016.py']['sha256']
    assert budget_factory({**p,**settings})==old['budget_caps']
    prior=failure['actual_completed_counts'];caps=old['budget_caps']
    aggregate={key:caps[key]+prior[key] for key in caps}
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    amendment=dict(schema='n01/wp3-016-runtime-numerical-storage-amendment/v1',UTC=stamp,
        user_authority='USER_MASTER_PLAN_20261015.txt',launch_revision='fix02',
        engineering_ready=ref(ready_path),independent_review=ref(review_path),native_failure=ref(failed_path),
        preserved_failure_backup=ref(OUT/'D_FAILURE_BACKUP_RECEIPT.json'),
        preserved_failure_local_commit=ref(OUT/'FAILURE_COMMIT_RECEIPT.json'),
        reason='Retain actual frozen-A numerical repeats and per-draw charge moments; replace an unsupported exact-equality assertion with the existing013 voltage replay gate.',
        original_failure_difference_magnitude='unknown_not_saved',
        baseline_repeat_gate=dict(source='scripts/run_wp1_013.py:A_numerical_residual',max_abs_V=1e-7,
            same_frozen_A=True,actual_A384_and_Q_mu_per_draw_retained=True,
            diagnostics_written_before_gate=True,no_broadcast_or_mean_substitution=True,
            exceeds_gate='save_evidence_then_fail_without_training_retry'),
        charge_law_model_objective_optimizer_seed_inputs_observation_T_unchanged=True,
        per_draw_theory='E[q_d]=mu_d; Var(q_d)=a*mu_d+b*mu_d**2',
        sample_variance_expectation='mean(per_draw_theory_variance)+sample_variance(per_draw_mu)',
        numerical_mu_spread_is_not_physical_randomness=True,
        A384_is_frozen_average_response_only_not_full_stochastic_preADC_carrier=True,
        original_remote_root=old['remote_root'],replacement_remote_root=NEW_REMOTE,
        prior_actual_cost=prior,replacement_attempt_budget_caps=caps,aggregate_budget_caps=aggregate,
        scientific_trial_slots=2,additional_candidate_slots=0,completed_prior_candidates=0,
        maximum_replacement_native_attempts=1,automatic_restart=False,
        another_native_failure='Preserve partial evidence; do not automatically submit another native attempt.',
        resource_limits=old['resource_limits'],run_resource_limits=old['run_resource_limits'],
        prior_native_elapsed_seconds=30.187189445001422,
        aggregate_wall_cap_seconds=old['resource_limits']['wall_seconds']+30.187189445001422,
        raw_arrays_retained_remote=True,selective_collection_cap_MiB=256,
        newly_started_operations_by_this_registration=0,scientific_status='response_match_not_established')
    amendment_path=ROOT/'docs/WP3_016_RUNTIME_FIX02.json'
    write(amendment_path,amendment,new=True)
    p.update(settings)
    p.update(status='draft_runtime_numerical_storage_fix02',source_files={},protection_files={},
        remote_root=NEW_REMOTE,launch_revision='fix02',
        previous_launch_configuration_sha256=sha(before/'wp3_016_config.json'),
        previous_launch_fix_reason=old.get('launch_fix_reason'),launch_fix_reason=amendment['reason'],
        runtime_repair=amendment,
        prior_failed_execution=dict(source_commit=previous_source['source_commit'],
            frozen_configuration=ref(before/'wp3_016_config.json'),
            frozen_manifest=ref(before/'WP3_016_SOURCE_MANIFEST.json'),
            failure=ref(failed_path),collection=ref(OUT/'failure_collection_01/COLLECTION_RECEIPT.json')))
    for row in p['contracts'].values():
        prefix=old['remote_root']+'/'
        if row['path'].startswith(prefix):row['path']=NEW_REMOTE+'/'+row['path'][len(prefix):]
    p['contracts']['runtime_numerical_storage_amendment']=dict(path=NEW_REMOTE+'/docs/WP3_016_RUNTIME_FIX02.json',
        local_path=str(amendment_path),sha256=sha(amendment_path),bytes=amendment_path.stat().st_size)
    p['allowed_input_roots']=list(dict.fromkeys(p['allowed_input_roots']+[NEW_REMOTE]))
    module.validate_config(p,frozen=False)
    assert p['inputs']==old['inputs'] and p['budget_caps']==old['budget_caps']
    write(ROOT/'wp3_016_config.json',p)
    receipt=dict(schema='n01/wp3-016-runtime-fix-control-preparation/v1',UTC=stamp,
        status='one_reviewed_replacement_ready_for_refreeze_not_started',
        config_draft_sha256=sha(ROOT/'wp3_016_config.json'),amendment=ref(amendment_path),
        independent_review=ref(review_path),engineering_ready=ref(ready_path),
        prior_actual_cost=prior,aggregate_budget_caps=aggregate,
        source_model016_unchanged=True,scientific_setting_change=False,
        new_candidate_trial_slots=0,new_scientific_operations=0,cumulative_training_updates=175667)
    write(OUT/'CONTROL_PREPARATION_RECEIPT_FIX02.json',receipt,new=True)
    print(json.dumps(dict(status=receipt['status'],draft_sha256=receipt['config_draft_sha256'],
        aggregate_G_cap=aggregate['G_updates'],aggregate_CP_cap=aggregate['checkpoint_loads'],
        aggregate_A_row_cap=aggregate['old_A_rows'])))


if __name__=='__main__':main()
