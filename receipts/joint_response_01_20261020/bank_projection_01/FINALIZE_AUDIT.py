from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
out = HERE / 'collected/run_01'
registration = json.loads((HERE / 'REGISTRATION.json').read_text())
p = json.loads((ROOT / registration['config']).read_bytes())
assert hashlib.sha256((ROOT / registration['config']).read_bytes()).hexdigest() == registration['config_sha256']
terminal = json.loads((HERE / 'REMOTE_TERMINAL_VERIFICATION.json').read_text())
manifest = json.loads((HERE / 'SOURCE_MANIFEST.json').read_text())
assert terminal['native_exit']['exit_code'] == 0 and terminal['processes'] == []
assert terminal['source_files_verified'] == len(manifest['files']) == 682
for name, digest in p['source_pins'].items():
    assert manifest['files'][name]['sha256'] == digest
for name, record in terminal['outputs'].items():
    raw = (HERE / 'collected' / name).read_bytes()
    assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
result = json.loads((out / 'RESULT.json').read_text())
assert result['config_sha256'] == registration['config_sha256']
assert set(result['execution_counts']) <= set(p['operation_budget'])
for name, limit in p['operation_budget'].items():
    assert result['execution_counts'].get(name, 0) == limit, name
assert len(result['original_replay']) == 18
assert all(x['residual_max_abs_difference'] == x['native_max_abs_difference'] == 0 for x in result['original_replay'])
audit_receipts = []
for seed in (0, 1):
    folder = HERE / 'hard_output_audit_fix01'
    audit = json.loads((folder / f'seed{seed}_HARD_OUTPUT_AUDIT.json').read_text())
    execution = json.loads((folder / f'seed{seed}_EXECUTION.json').read_text())
    assert execution['exit_code'] == 0 and execution['reason'] is None
    assert audit['status'] == 'PASS_INDEPENDENT_NUMPY_SCIPY_HARD_OUTPUT_REPLAY'
    assert audit['scalar_metric_checks'] == 243 and audit['fixed_factor_decompositions'] == 18
    assert audit['source_sha256'] == hashlib.sha256((folder / 'AUDIT_HARD_OUTPUTS.py').read_bytes()).hexdigest()
    assert audit['result_sha256'] == terminal['result_sha256']
    audit_receipts.append(dict(seed=seed, maximum_absolute_difference=audit['maximum_absolute_difference'],
        elapsed_seconds=audit['elapsed_seconds'], execution=execution,
        receipt=str((folder / f'seed{seed}_HARD_OUTPUT_AUDIT.json').relative_to(ROOT)).replace('\\', '/')))
    case = p['cases'][seed]
    with np.load(ROOT / case['arrays'], allow_pickle=False) as original:
        for bi, bank in enumerate(('FIT0', 'FIT1', 'FIT2')):
            with np.load(out / f'seed{seed}_{bank}.npz', allow_pickle=False) as saved:
                assert np.array_equal(saved['source_parents'], original[bank + '_source_parents'])
                assert np.array_equal(saved['real_rows'], original[bank + '_real_rows'])
                assert int(saved['bank_seed']) == p['training_seed'] + seed * 100000 + 1000 + bi * 10
                for point in ('M0', 'trial0', 'trial1'):
                    expected = original['theta'] if point == 'M0' else original[point + '_proposal']
                    assert np.array_equal(saved[point + '_theta'], expected)
                    for hi in range(3):
                        for key, width in (('window_V', 256), ('baseline100', 100)):
                            values = saved[point + f'_h{hi}_' + key]
                            assert values.shape == (4096, width) and np.isfinite(values).all()
summary = {}
for metric in ('wave', 'QPSD'):
    contrasts = [item for value in result['seeds'].values() for item in value['contrasts'] if item['metric'] == metric]
    shares = np.array([item['sum_squares_fractions'] for item in contrasts])
    accepted = [item for item in contrasts if item['point'] == 'trial1']
    summary[metric] = dict(fixed_candidate_hypothesis_cases=len(contrasts),
        bank_main_effect_variance_fraction_range=[float(shares[:, 0].min()), float(shares[:, 0].max())],
        projection_main_effect_variance_fraction_range=[float(shares[:, 1].min()), float(shares[:, 1].max())],
        interaction_variance_fraction_range=[float(shares[:, 2].min()), float(shares[:, 2].max())],
        original_accepted_trial_risk_ratio_range=[min(x['minimum'] for x in accepted), max(x['maximum'] for x in accepted)])
review = dict(status='PASS_FIXED_CANDIDATE_CROSSING_AND_INDEPENDENT_HARD_OUTPUT_AUDIT',
    UTC=datetime.now(timezone.utc).isoformat(), config_sha256=registration['config_sha256'],
    source_files_verified=682, source_pins_verified=27, output_files_verified=len(terminal['outputs']),
    output_bytes=sum(item['bytes'] for item in terminal['outputs'].values()),
    result_sha256=terminal['result_sha256'], native_exit=terminal['native_exit'],
    elapsed_seconds=result['elapsed_seconds'], execution_counts=result['execution_counts'],
    original_points_replayed=18, original_diagonal_native_and_joint_residual_max_difference=0.,
    fixed_row_selection_entries_checked=30720, raw_hard_observation_views=54, raw_REAL_target_banks=6,
    independent_hard_output_replay_pass=True, independent_scalar_metric_checks=486,
    independent_maximum_absolute_difference=max(x['maximum_absolute_difference'] for x in audit_receipts),
    independent_audit_receipts=audit_receipts, exact_rational_variance_decompositions=36,
    prior_audit_failure_retained=True, prior_audit_failure_review=str((HERE / 'AUDIT_ROUNDOFF_REVIEW.json').relative_to(ROOT)).replace('\\', '/'),
    findings=summary, new_directions=0, parameter_updates=0, heldout_rows_evaluated=0, scientific_acceptance=False,
    interpretation='In these fixed 3x3 crossings wave risk changes have substantial projection and bank effects; QPSD changes are dominated by the bank factor. Bank includes SOURCE/REAL sampling and latent/noise draws. This is a descriptive fixed-candidate risk decomposition, not attribution of all previous finite-difference gradient instability.',
    next_action='Register the prepared zero-update C1 interpolation engineering contrast, motivated jointly by the earlier full-signal attribution and the current fixed-risk crossing. Preserve original model/results and require renewed coupled-teacher validation before any REAL use if engineering passes.')
(HERE / 'TERMINAL_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n', encoding='utf-8', newline='\n')
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    path = ROOT / name
    ledger = json.loads(path.read_bytes())
    assert ledger['active_scientific_jobs'] == 0
    ledger.update(status='JOINT_RESPONSE_01_bank_projection_closed_audited', updated_UTC=review['UTC'],
        current_action='Fixed bank/projection crossing and independent hard-output replay complete.',
        next_action=review['next_action'], current_run_training_updates=0,
        current_job_native_exit_code=0, current_job_ended_UTC=terminal['native_exit']['ended_utc'])
    ledger['joint_response_01'].update(bank_projection_terminal_review=str((HERE / 'TERMINAL_REVIEW.json').relative_to(ROOT)).replace('\\', '/'),
        bank_projection_native_exit_code=0, bank_projection_independent_replay_pass=True, bank_projection_parameter_updates=0)
    if name == 'jobs.json':
        job = ledger['jobs'][-1]
        assert job['job_id'] == p['job_id']
        job.update(status='closed_complete_audited', terminal_review=ledger['joint_response_01']['bank_projection_terminal_review'],
            independent_scalar_metric_checks=486, original_replay_max_difference=0.)
    path.write_bytes((json.dumps(ledger, indent=2, ensure_ascii=False) + '\n').replace('\n', '\r\n').encode('utf-8'))
print(json.dumps({key: value for key, value in review.items() if key not in ('execution_counts', 'independent_audit_receipts')}))
