"""Independent JSON/counter/schema checks without any scientific execution."""
from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from metadata_audit import ROOT, OUT, HYPOTHESES, sha

sys.path.insert(0, str(ROOT / 'src'))
from n01 import train_window008_resume01 as driver


def main():
    source = ROOT / 'src/n01/train_window008_resume01.py'
    digest = sha(source)
    base = driver.read_json(ROOT / 'window008_config.json')
    fix = driver.read_json(ROOT / 'window008_fix01_config.json')
    resume = driver.read_json(ROOT / 'window008_resume01_config.json')
    documents = {name: driver.read_json(ROOT / record['path'])
                 for name, record in resume['reference_inputs'].items() if record['path'].endswith('.json')}
    original_chain = driver.validate_legacy_receipts(base, fix, documents)
    driver.validate_plan(base, fix, resume)
    native_chain = driver.validate_stop_chain(base, fix, resume, documents, original_chain)
    results = [dict(case='actual_resume_configuration_and_stop_native_chain', passed=True, fixture=False)]

    def reject_plan(name, alter):
        candidate = copy.deepcopy(resume)
        alter(candidate)
        try:
            driver.validate_plan(base, fix, candidate)
        except (ValueError, KeyError, TypeError) as error:
            results.append(dict(case=name, passed=True, rejected=True, fixture=True, rejection=str(error)))
        else:
            results.append(dict(case=name, passed=False, rejected=False, fixture=True))

    reject_plan('plan_wrong_new_G', lambda s: s.update(new_execution_G_updates=35000))
    reject_plan('plan_wrong_D_budget', lambda s: s.update(new_execution_D_updates=96000))
    reject_plan('plan_wrong_historical_G', lambda s: s.update(historical_actual_G_updates_after_success=57269))
    reject_plan('plan_wrong_historical_D', lambda s: s.update(historical_actual_D_updates_after_success=144807))
    reject_plan('plan_wrong_effective_G', lambda s: s.update(effective_comparison_G_updates=57667))
    reject_plan('plan_wrong_inherited_D', lambda s: s.update(inherited_effective_D_updates=48000))
    reject_plan('plan_resume_from_userstop6398', lambda s: s['resume'].update(G_updates=6398, next_G_update=6399))
    reject_plan('plan_missing_local_raw_reference', lambda s: s['reference_inputs'].pop('fix01_user_stop_verification_local'))
    reject_plan('plan_model_changed', lambda s: s.update(model_loss_data_seed_readout_changes=True))
    reject_plan('plan_DEV_training', lambda s: s.update(DEV_waveforms_used_for_training=True))
    reject_plan('plan_noise_training', lambda s: s.update(noise_updates=1))
    reject_plan('plan_retrain_deterministic', lambda s: s.update(execution_arms=['deterministic', 'full', 'simple']))
    reject_plan('plan_simple_not_fresh', lambda s: s['fresh_simple'].update(G_updates=1))
    reject_plan('plan_first_update_replay_disabled', lambda s: s['resume'].update(first_update_same_version_replay=False))

    def reject_chain(name, alter):
        candidate = copy.deepcopy(documents)
        alter(candidate)
        try:
            driver.validate_stop_chain(base, fix, resume, candidate, original_chain)
        except (ValueError, KeyError, TypeError) as error:
            results.append(dict(case=name, passed=True, rejected=True, fixture=True, rejection=str(error)))
        else:
            results.append(dict(case=name, passed=False, rejected=False, fixture=True))

    reject_chain('chain_old_native1_mislabeled_success', lambda s: s['fix01_native_exit'].update(exit_code=0))
    reject_chain('chain_wrong_parent_pid', lambda s: s['fix01_binding'].update(parent_pid=1))
    reject_chain('chain_wrong_launch_module', lambda s: s['fix01_launch']['command'].__setitem__(7, 'n01.train_window008_resume01'))
    reject_chain('chain_wrong_native_time', lambda s: s['fix01_native_exit'].update(started_utc='2026-10-13T11:00:00+00:00'))
    reject_chain('chain_first_import_proof_missing', lambda s: s['fix01_resume_validation'].update(optimizer_sampler_streams_exact=False))
    reject_chain('chain_non_SIGTERM_failure', lambda s: s['fix01_failure'].update(error='Numerical failure'))
    reject_chain('chain_stop_G_counter_wrong', lambda s: s['fix01_failure'].update(current_G_updates=6000))
    reject_chain('chain_stopped_checkpoint_auto_allowed', lambda s: s['fix01_failure'].update(automatic_resume_allowed=True))
    reject_chain('chain_stop_signal_not_owned', lambda s: s['fix01_user_stop_verification'].update(signal_sent='SIGTERM_all_python'))
    reject_chain('chain_pause_hashes_mix_serializations', lambda s: s['fix01_user_pause'].update(stop_receipt_sha256=resume['reference_inputs']['fix01_user_stop_verification']['sha256']))
    reject_chain('chain_local_remote_JSON_changed', lambda s: s['fix01_user_stop_verification_local'].update(trainer_absent=False))
    reject_chain('chain_archive_periodic_bytes_changed', lambda s: s['fix01_user_stop_collection']['files']['runs/science/train/full/latest.pt'].update(bytes=1))
    reject_chain('chain_archive_failed_hash_verification', lambda s: s['fix01_user_stop_collection'].update(all_transferred_member_hashes_verified=False))
    reject_chain('chain_periodic_pause_G_mismatch', lambda s: s['fix01_user_pause']['periodic_checkpoint'].update(G_updates=6398))

    duplicates = OUT / 'duplicate_keys_fixture.json'
    duplicates.write_text('{"G_updates":6000,"G_updates":6398}\n', 'utf8')
    try:
        driver.read_json(duplicates)
    except ValueError as error:
        results.append(dict(case='duplicate_JSON_key_rejected', passed=True, rejected=True, fixture=True, rejection=str(error)))
    else:
        results.append(dict(case='duplicate_JSON_key_rejected', passed=False, fixture=True))
    for path in ['../outside.txt', 'C:\\outside.txt', '/outside.txt', '', 'data/../../outside.txt']:
        try:
            driver.bound(ROOT, path)
        except ValueError as error:
            results.append(dict(case='bound_path_reject_' + path, passed=True, rejected=True, fixture=True, rejection=str(error)))
        else:
            results.append(dict(case='bound_path_reject_' + path, passed=False, fixture=True))

    retained = driver.retained_endpoint(fix, documents, documents['fix01_legacy_validation'])
    def engine(arm, g, d):
        return SimpleNamespace(arm=arm, new_G_updates=g, new_D_updates=dict(zip(HYPOTHESES, d)))
    accounting_cases = [
        ('before_import', [retained], None, 0, 0),
        ('import6000_does_not_count_new', [retained], engine('full', 0, [0, 0, 0]), 0, 0),
        ('first6001_full_update', [retained], engine('full', 1, [1, 1, 1]), 1, 3),
        ('partial_D_without_G', [retained], engine('full', 0, [1, 0, 0]), 0, 1),
        ('first_simple_after_full', [retained, dict(arm='full', new_G_updates=13000, new_D_updates=dict.fromkeys(HYPOTHESES, 13000))], engine('simple', 1, [0, 0, 0]), 13001, 39000),
        ('simple_partial_D_after_full', [retained, dict(arm='full', new_G_updates=13000, new_D_updates=dict.fromkeys(HYPOTHESES, 13000))], engine('simple', 3000, [1, 0, 0]), 16000, 39001),
        ('current_engine_completed_row_not_double_counted', [retained, dict(arm='full', new_G_updates=13000, new_D_updates=dict.fromkeys(HYPOTHESES, 13000))], engine('full', 13000, [13000]*3), 13000, 39000),
        ('final32000_87000', [retained, dict(arm='full', new_G_updates=13000, new_D_updates=dict.fromkeys(HYPOTHESES, 13000)), dict(arm='simple', new_G_updates=19000, new_D_updates=dict.fromkeys(HYPOTHESES, 16000))], None, 32000, 87000),
    ]
    for name, completed, active, expected_g, expected_d in accounting_cases:
        actual = driver.update_accounting(completed, active)
        passed = (actual['new_execution_G_updates'] == expected_g and actual['new_execution_D_updates'] == expected_d
                  and actual['effective_comparison_G_updates'] == 25000 + expected_g
                  and actual['effective_comparison_D_updates'] == 57000 + expected_d
                  and actual['historical_actual_G_updates'] == 25667 + expected_g
                  and actual['historical_actual_D_updates'] == 59001 + expected_d
                  and actual['global_N01_actual_G_updates'] == 125667 + expected_g)
        results.append(dict(case='accounting_' + name, passed=passed, fixture=True, accounting=actual))

    endpoints = [copy.deepcopy(retained)]
    identity = dict(run_id=driver.RUN_ID, config_sha256=sha(ROOT / 'window008_resume01_config.json'),
                    source_manifest_sha256='f'*64, base_configuration_sha256=resume['base_configuration_sha256'],
                    base_source_manifest_sha256=resume['legacy_source_manifest_sha256'],
                    fix01_configuration_sha256=resume['fix01_configuration_sha256'],
                    fix01_source_manifest_sha256=resume['fix01_source_manifest_sha256'])
    for arm, inherited, new_g, new_d, origin in [('full', 6000, 13000, 13000, 'fix01_periodic6000_resume'),
                                               ('simple', 0, 19000, 16000, 'fresh_common_initialization')]:
        endpoints.append(dict(arm=arm, G_updates=19000, D_updates=dict.fromkeys(HYPOTHESES, 16000), stage='low_learning_rate',
                              coverage=dict(source_unique=6593, real_unique=21047), checkpoint_schema='response008_resume01/v1',
                              origin=origin, model_implementation=driver.MODEL_IMPLEMENTATION,
                              inherited_G_updates=inherited, new_G_updates=new_g, new_D_updates=dict.fromkeys(HYPOTHESES, new_d),
                              common_initial_state_sha256=retained['common_initial_state_sha256'], noise005_state_sha256=retained['noise005_state_sha256'],
                              final_sha256='a'*64, recovery=identity, endpoint_path='runs/science/train/' + arm + '/final.pt',
                              lineage=dict(source_checkpoint_sha256=driver.FULL6000_SHA256 if arm == 'full' else None)))
    payload = driver.completion_payload(base, resume, identity, endpoints, {}, native_chain, 'd'*64, 1.0)
    results.append(dict(case='completion_never_self_certifies_native_or_evaluation',
                        passed=payload['awaiting_own_supervisor_native_exit'] is True
                          and payload['native_training_success'] is False and payload['evaluation_complete'] is False
                          and payload['scientific_status'] == 'response_match_not_established', fixture=True))
    results.append(dict(case='completion_mixed_endpoint_schema_and_origins',
                        passed=payload['endpoint_map']['deterministic']['schema'] == 'response008/v1'
                          and payload['endpoint_map']['full']['schema'] == payload['endpoint_map']['simple']['schema'] == 'response008_resume01/v1', fixture=True))
    for name, mutate in [('missing_simple_endpoint', lambda s: s.pop()),
                         ('wrong_full_schema', lambda s: s[1].update(checkpoint_schema='response008_fix01/v1')),
                         ('wrong_full_inherited_updates', lambda s: s[1].update(inherited_G_updates=3000)),
                         ('wrong_noise_reference', lambda s: s[2].update(noise005_state_sha256='b'*64)),
                         ('incomplete_REAL_coverage', lambda s: s[2]['coverage'].update(real_unique=1))]:
        candidate = copy.deepcopy(endpoints)
        mutate(candidate)
        try:
            driver.validate_endpoints(candidate)
        except (ValueError, KeyError, TypeError) as error:
            results.append(dict(case='endpoint_' + name, passed=True, rejected=True, fixture=True, rejection=str(error)))
        else:
            results.append(dict(case='endpoint_' + name, passed=False, fixture=True))

    assert sha(source) == digest, 'Driver changed during independent metadata review'
    report = dict(schema='n01/window008-resume01-independent-driver-counterexamples/v1',
                  UTC=datetime.now(timezone.utc).isoformat(), source_sha256=digest,
                  config_sha256=sha(ROOT / 'window008_resume01_config.json'),
                  status='passed' if all(row['passed'] for row in results) else 'failed', cases=results,
                  native_chain=native_chain, scientific_model_construction=False, scientific_model_forwards=0,
                  optimizer_updates=0, GPU_calls=0, DEV_numeric_decodes=0, new_scientific_jobs=0,
                  scope='JSON plan and stop chain, pure accounting and endpoint metadata; production main/bind not executed.')
    (OUT / 'DRIVER_METADATA_COUNTEREXAMPLES.json').write_text(json.dumps(report, indent=2) + '\n', 'utf8')
    print(json.dumps(dict(status=report['status'], cases=len(results), source_sha256=digest,
                          failed=[row for row in results if not row['passed']]), indent=2))


if __name__ == '__main__':
    main()
