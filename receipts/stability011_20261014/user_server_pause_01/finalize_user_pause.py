"""Complete the local pause checkpoint after independent saved-output readback."""
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import json

ROOT = Path('E:/GAN/research/N01')
PAUSE = ROOT/'receipts/stability011_20261014/user_server_pause_01'
REL = PAUSE.relative_to(ROOT).as_posix()
COLLECTED = ROOT/'runs/stability011_20261014/collected_01'
spec = importlib.util.spec_from_file_location('record_user_pause', PAUSE/'record_user_pause.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
read, digest, write = helper.read, helper.digest, helper.write

def main():
    timestamp = datetime.now(timezone.utc).isoformat()
    review_path = PAUSE/'independent_readback_01/READBACK.json'
    review = read(review_path)
    assert review.get('passed') is True or review.get('status') in ('PASS', 'pass', 'passed')
    result = read(COLLECTED/'runs/primary/artifacts/STABILITY_RESULT.json')
    assert result['classification'] == 'saved_TRAIN_bank_gain_stable_under_paired_parent_group_reweighting'
    assert result['all_replicate_conditional_coverage_complete'] is True
    assert result['coverage_limited_replicates'] == 0
    assert result['response_match_established'] is False
    assert result['formal_significance_test'] is False and result['formal_frequentist_coverage_established'] is False
    assert result['primary']['Delta']['percentile_resampling_interval95'][1] < 0
    assert result['bootstrap_replicates_completed'] == result['bootstrap_replicates_attempted'] == result['bootstrap_replicates_computed_successfully'] == 500
    assert not result['DEV_numeric_read'] and not result['SOURCE_waveform_decode']
    assert all(value == 0 for value in result['counters'].values())
    manifest = read(ROOT/'STABILITY011_SOURCE_MANIFEST.json')
    verified = {}
    for name, expected in manifest['files'].items():
        actual = digest(ROOT/name)
        assert actual == expected, f'Frozen source changed: {name}'
        verified[name] = actual
    for name, expected in manifest['protected_manifests'].items():
        if isinstance(expected, dict):
            expected = expected['sha256']
        actual = digest(ROOT/name)
        assert actual == expected, f'Protected manifest changed: {name}'
        verified[name] = actual
    freeze = read(ROOT/'receipts/stability011_20261014/freeze_01/FREEZE_RECEIPT.json')
    assert digest(ROOT/'STABILITY011_SOURCE_MANIFEST.json') == freeze['source_manifest_sha256']
    assert digest(ROOT/'STABILITY011_BINDINGS.json') == freeze['bindings_sha256']
    assert digest(ROOT/'stability011_policy.json') == freeze['policy_sha256']
    assert digest(Path(freeze['package'])) == digest(Path(freeze['D_source_backup'])) == freeze['package_sha256']
    preservation = dict(schema='n01/stability011-local-pause-preservation/v1', UTC=timestamp,
        source_files_verified=len(manifest['files']), old_source_files=manifest['old_shape010_source_file_count'],
        protected_manifests_verified=len(manifest['protected_manifests']),
        E_package_SHA256=freeze['package_sha256'], D_source_backup=freeze['D_source_backup'],
        D_backup_SHA256_verified=True, sources=verified,
        new_model_forwards=0, new_gradient_queries=0, new_optimizer_updates=0)
    write(PAUSE/'LOCAL_SOURCE_PRESERVATION.json', preservation, new=True)
    checkpoint = dict(schema='n01/stability011-user-pause-final/v1', UTC=timestamp,
        status='completed_closeout_user_paused_waiting_server_restart_notice', run_id='stability011_20261014',
        user_pause_receipt=REL+'/USER_PAUSE_RECEIPT.json',
        active_project_jobs_at_last_native_readback=0, signal_sent=False,
        native_exit_code=0, bootstrap=dict(attempted=500, computed=500, saved=500),
        collected_root=str(COLLECTED),
        collection_receipt=REL+'/collection_01/COLLECTION_RECEIPT.json',
        independent_readback=REL+'/independent_readback_01/READBACK.json',
        independent_readback_SHA256=digest(review_path),
        independent_scope='saved artifact hashes, native and journal identities, plans, metric algebra and percentile summaries; no new500 full-proxy reductions',
        classification=result['classification'], main_definition='mean_h(wave+2*QPSD);Delta=final-A',
        point=result['point'], primary_resampling_summary=result['primary'],
        coverage_limited_replicates=0, all500_conditional_coverage_complete=True,
        interval_scope=result['interval_scope'], original_group_description=result['original_group_description'],
        formal_significance_test=False, formal_frequentist_coverage_established=False,
        scientific_status='response_match_not_established', response_match_established=False,
        counters=result['counters'], DEV_numeric_read=False, SOURCE_waveform_decode=False,
        followup_automation_id='n01', followup_automation_status='PAUSED',
        resume_requires_user_server_started_notice=True, automatic_restart=False,
        next_diagnostic_012_registered=False, next_diagnostic_012_started=False,
        server_power_state='unknown; user reported planned temporary shutdown',
        next_action='Wait for the user server-start notice. Verify server and project state once on resume;011 is complete and must not be rerun. Any012 model diagnosis requires its separate registration and frozen budget before execution.',
        local_source_preservation=REL+'/LOCAL_SOURCE_PRESERVATION.json',
        E_metadata_snapshot=REL+'/metadata_before_pause', new_training_updates_in_this_closeout=0)
    write(PAUSE/'FINAL_SERVER_PAUSE_RECEIPT.json', checkpoint, new=True)
    for metadata_path in (ROOT/'PROGRAM.json', Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')):
        value = read(metadata_path)
        absolute_links = metadata_path.parent != ROOT
        def link(relative):
            return str(ROOT/relative) if absolute_links else relative
        value.update(updated_UTC=timestamp,
            current_stage='stability011_completed_independently_read_back_user_paused',
            next_action=checkpoint['next_action'],
            stability011_independent_readback_complete=True,
            stability011_independent_readback_scope=checkpoint['independent_scope'],
            stability011_independent_readback=link(REL+'/independent_readback_01/READBACK.json'),
            stability011_result_receipt=link(REL+'/FINAL_SERVER_PAUSE_RECEIPT.json'),
            stability011_classification=result['classification'], stability011_primary_summary=result['primary'],
            stability011_conditional_coverage_complete=True, stability011_source_freeze_pending=False,
            stability011_closeout_complete=True, followup_pause_receipt=link(REL+'/FINAL_SERVER_PAUSE_RECEIPT.json'))
        assert value['status'] == 'paused_user_request_waiting_server_restart'
        assert value['active_scientific_jobs'] == 0 and value['followup_automation_status'] == 'PAUSED'
        assert value['new_training_updates'] == 159667
        write(metadata_path, value)
    jobs_path = ROOT/'jobs.json'
    jobs = read(jobs_path)
    targets = [job for job in jobs['jobs'] if job.get('run_id') == 'stability011_20261014']
    assert len(targets) == 1
    targets[0].update(status='completed_native0_collected_independently_read_back_user_paused',
        independent_readback_complete=True, independent_readback=REL+'/independent_readback_01/READBACK.json',
        independent_readback_scope=checkpoint['independent_scope'], result_receipt=REL+'/FINAL_SERVER_PAUSE_RECEIPT.json',
        classification=result['classification'], conditional_coverage_complete=True, bounded_budget_closed=True,
        bootstrap_replicates_remaining=0, additional_optimizer_updates=0)
    jobs['updated_UTC'] = timestamp
    write(jobs_path, jobs)
    reopened = read(PAUSE/'FINAL_SERVER_PAUSE_RECEIPT.json')
    assert reopened == checkpoint
    for metadata_path in (ROOT/'PROGRAM.json', Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')):
        value = read(metadata_path)
        assert value['resume_requires_user_server_started_notice'] is True
        assert value['active_scientific_jobs'] == 0 and value['stability011_analysis_pid'] is None
        assert value['stability011_closeout_complete'] and value['followup_scheduled_local'] is None
    assert read(jobs_path)['active_scientific_jobs'] == 0
    print(json.dumps(dict(status=checkpoint['status'], receipt=str(PAUSE/'FINAL_SERVER_PAUSE_RECEIPT.json'),
        completed=500, active_jobs=0, followup='PAUSED', sources_verified=len(manifest['files']),
        protected_manifests_verified=len(manifest['protected_manifests']),
        Delta95=result['primary']['Delta']['percentile_resampling_interval95']), ensure_ascii=False))

if __name__ == '__main__':
    main()
