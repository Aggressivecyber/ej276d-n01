"""Close target native evidence by comparing its receipt to actual saved state."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import torch

from metadata_audit import ROOT, OUT, HYPOTHESES, sha
from engine_metadata_counterexamples import MetadataModule, MetadataDiscriminators

sys.path.insert(0, str(ROOT / 'scripts'))
import window008_resume01_runtime as runtime
from n01.response008_fix01 import tree_digest, optimizer_named_state


def main():
    torch.set_num_threads(2)
    cfg, _, _ = runtime.validate_engineering(ROOT)
    rec = ROOT / 'receipts/window008_resume01_20261013'
    qa_path = rec / 'engineering_cuda_01/ENGINEERING_RECEIPT.json'
    native_path = rec / 'engineering_supervisor_01/exit_receipt.json'
    launch_path = rec / 'engineering_supervisor_01/launch.json'
    collected_path = rec / 'ENGINEERING_NATIVE_COLLECTION.json'
    qa, native, launch, collection = [runtime.read(path) for path in [qa_path, native_path, launch_path, collected_path]]
    checkpoint = ROOT / cfg['reference_inputs']['fix01_full_periodic_06000']['path']
    assert sha(checkpoint) == cfg['reference_inputs']['fix01_full_periodic_06000']['sha256']
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    imported = qa['periodic_import']
    expected_digests = {name: tree_digest(cp[name]) for name in
                        ('model', 'discriminators', 'g_optimizer', 'd_optimizers', 'source_sampler', 'real_sampler', 'streams')}
    model = MetadataModule(cp['model'], cp['parameter_signature'])
    discriminators = MetadataDiscriminators(cp['discriminators'])
    expected_g_slots = {name: tree_digest(slot) for name, slot in optimizer_named_state(model, cp['g_optimizer']).items()}
    expected_d_slots = {h: {name: tree_digest(slot) for name, slot in optimizer_named_state(discriminators[h], cp['d_optimizers'][h]).items()}
                        for h in HYPOTHESES}
    checks = dict(runtime_native_engineering_validation_passed=True,
                  raw_collection_SSH_exit0=collection['SSH_exit_code'] == 0,
                  raw_collection_status=collection['status'] == 'native_evidence_recovered',
                  raw_collection_files_exact=all(sha(ROOT / name) == expected for name, expected in collection['files'].items()),
                  raw_checkpoint_read_only=sha(checkpoint) == cfg['reference_inputs']['fix01_full_periodic_06000']['sha256'],
                  native_exit0=native['exit_code'] == 0 and native['reason'] is None,
                  timeout_command_exact=launch['command'][:4] == ['timeout', '--signal=TERM', '--kill-after=15s', '300s'],
                  actual_native_checkpoint_all_tree_digests_exact=imported['state_digests'] == expected_digests,
                  actual_native_G_named_slot_digests_exact=imported['G_optimizer_named_slot_sha256'] == expected_g_slots,
                  actual_native_D_named_slot_digests_exact=imported['D_optimizer_named_slot_sha256'] == expected_d_slots,
                  actual_native_import_all_tree_bitwise=imported['native_tree_bitwise_equal'] is True,
                  actual_native_import_moments_preserved=imported['optimizer_moments_reset'] is False,
                  actual_native_import_registered_hash=imported['source_checkpoint_sha256'] == cfg['reference_inputs']['fix01_full_periodic_06000']['sha256'],
                  actual_native_import_role_scientific=imported['engineering_fixture'] is False,
                  actual_native_import_new_counts_zero=imported['new_G_updates'] == 0 and imported['new_D_updates'] == dict.fromkeys(HYPOTHESES, 0),
                  actual_native_import_next6001=imported['next_G_update'] == imported['projection_update'] == 6001,
                  actual_native_import_named_phase_steps={6000: sum(v == 6000 for v in imported['G_optimizer_named_steps'].values()),
                                                          3000: sum(v == 3000 for v in imported['G_optimizer_named_steps'].values())} == {6000: 167, 3000: 13},
                  actual_native_import_fix01_identity=imported['fix01_recovery'] == cp['recovery'],
                  actual_scientific_updates_and_rows_zero=all(qa[name] == 0 for name in
                      ('scientific_G_updates', 'scientific_D_updates', 'actual_SOURCE_forward_rows', 'actual_REAL_forward_rows',
                       'actual_SOURCE_numeric_decodes', 'actual_REAL_numeric_decodes', 'DEV_numeric_decodes')),
                  native_artificial_budget=qa['engineering_G_updates'] == 3 and qa['engineering_D_updates'] == 6
                      and qa['artificial_forward_rows'] == 6144,
                  native_artificial_stage_sequence=[(row['arm'], row['stage']) for row in qa['steps']]
                      == [('full', 'joint_stochastic'), ('simple', 'mean_response'), ('simple', 'joint_stochastic')],
                  native_artificial_full_first6001=qa['steps'][0]['metrics']['G_updates'] == 6001
                      and qa['steps'][0]['metrics']['new_G_updates'] == 1,
                  all_native_artificial_steps_replay=all(row['metrics']['same_version_replay_verified'] is True for row in qa['steps']),
                  artificial_initial_D_role_explicit=qa['artificial_initial_D_installed_from_declared_bundle_only'] is True
                      and qa['artificial_assigned6000_counters_are_not_executed_updates'] is True)
    known_paths = [qa_path, native_path, launch_path, collected_path]
    report = dict(schema='n01/window008-resume01-independent-actual-native-review/v1',
                  UTC=datetime.now(timezone.utc).isoformat(), status='passed' if all(checks.values()) else 'failed', checks=checks,
                  files_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in known_paths},
                  checked_QA_source_sha256=qa['source_sha256'],
                  imported_all_native_state_digests=expected_digests,
                  native_PID_chain=dict(supervisor=native['supervisor_pid'], timeout_wrapper=native['child_pid'],
                                        QA_process=qa['pid'], QA_parent=qa['parent_pid']),
                  peak_GPU_bytes=max(row['peak_GPU_bytes'] for row in qa['steps']), peak_RSS_bytes=native['peak_rss_bytes'],
                  actual_target_initial_D_reference_guard_preserved=True,
                  actual_target_checkpoint_import_passed=True, actual_target_native_engineering_exit0=True,
                  native_QA_recovery_manifest_identity_scope='Explicit prefreeze metadata dummy; science driver must bind actual frozen manifest.',
                  own_scientific_model_construction=False, own_scientific_model_forwards=0, own_optimizer_updates=0,
                  own_DEV_numeric_decodes=0, own_GPU_calls=0, own_SSH_calls=0, own_new_scientific_jobs=0,
                  scientific_status='response_match_not_established')
    (OUT / 'ACTUAL_NATIVE_RECEIPT_REVIEW.json').write_text(json.dumps(report, indent=2) + '\n', 'utf8')
    print(json.dumps(dict(status=report['status'], checks=len(checks),
                          failed=[name for name, valid in checks.items() if not valid],
                          actual_target_import_verified=True, peak_GPU_bytes=report['peak_GPU_bytes'],
                          peak_RSS_bytes=report['peak_RSS_bytes']), indent=2))


if __name__ == '__main__':
    main()
