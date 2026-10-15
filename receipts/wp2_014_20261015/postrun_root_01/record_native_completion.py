"""Register one saved WP2 terminal snapshot and account actual training once."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT / 'receipts/wp2_014_20261015/postrun_root_01'
STATE = ROOT / 'receipts/wp2_014_20261015/wake_status_01/REMOTE_STATE.json'

def read(p):
    return json.loads(p.read_bytes())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value, exclusive=False):
    with p.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

state = read(STATE)
native = state['runs/primary/supervisor/exit_receipt.json']
complete = state['runs/primary/artifacts/COMPLETE.json']
cfg = read(ROOT / 'wp2_014_config.json')
program = read(ROOT / 'PROGRAM.json')
source = read(ROOT / 'receipts/wp2_014_20261015/SOURCE_COMMIT_READBACK.json')
assert native['child_native_exit_code'] == native['exit_code'] == 0
assert complete['all_four_trials_completed'] and len(complete['trials']) == 4
assert not state['runs/primary/artifacts/FAILED.json']
assert not state['protection_errors']
assert not native['unknown_inflight_ids'] and not native['failed_ids'] and not native['truncated_tail']
assert native['entered_counts'] == native['completed_counts']
counts = {k: native['completed_counts'].get(k, 0) for k in cfg['budget_caps']}
assert counts == cfg['budget_caps']
assert counts['new_AD_queries'] == counts['training_backward_queries'] + counts['probe_AD_queries']
assert counts['head_total_kernel_rows'] == counts['head_actual_parameter_forward_rows'] + counts['calibration_map_rows']
assert all(x['G_updates'] == 2000 for x in complete['trials'])
assert [x['directory'] for x in complete['trials']] == cfg['candidate_trial_order']
for rel, record in source['files'].items():
    assert sha(ROOT / rel) == record['sha256'], rel
before = cfg['cumulative_training_before_run']
assert before == 159667
assert program['new_training_updates'] == before + program['wp2_G_updates'] == 159864
assert program['wp2_G_updates'] == 197
total = before + counts['G_updates']
now = dt.datetime.now(dt.timezone.utc).isoformat()
receipt = dict(schema='n01/wp2-native-terminal-readback/v1', UTC=now,
    state=str(STATE.relative_to(ROOT)), state_sha256=sha(STATE),
    scope='one saved terminal snapshot and frozen hashes; selective collection and independent numeric readback pending',
    source_commit=source['source_commit'], configuration_sha256=sha(ROOT/'wp2_014_config.json'),
    source_manifest_sha256=sha(ROOT/'WP2_SOURCE_MANIFEST.json'),
    native_exit=native, trials=complete['trials'], actual_counts=counts,
    historical_training_before_014=before, startup_lower_bound_already_accounted=197,
    remaining_new_G_updates_accounted_now=counts['G_updates']-197,
    cumulative_training_updates=total, short_candidate_runs_completed=4,
    active_scientific_jobs=0, collection_complete=False, independent_readback_complete=False,
    raw_NPZ_retained_remote=True, all_raw_collected=False,
    formal_response_acceptance=False, scientific_status='response_match_not_established')
write(OUT/'NATIVE_COMPLETION_READBACK.json', receipt, True)
ref = dict(path=str((OUT/'NATIVE_COMPLETION_READBACK.json').relative_to(ROOT)),
    sha256=sha(OUT/'NATIVE_COMPLETION_READBACK.json'),
    native_exit_code=0, supervisor_exit_code=0, ended_UTC=native['UTC'],
    actual_counts=counts, source_commit=source['source_commit'],
    collection_complete=False, independent_readback_complete=False,
    new_training_updates=counts['G_updates'], cumulative_training_updates=total,
    scientific_status='response_match_not_established', formal_response_acceptance=False)
paths = [ROOT/'MASTER_EXECUTION_PLAN.json', ROOT/'PROGRAM.json', ROOT/'jobs.json',
    Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
backup = OUT/'metadata_before_native_closeout'
backup.mkdir()
for p in paths:
    shutil.copy2(p, backup/p.name)
for p in paths:
    d = read(p)
    d['updated_UTC'] = now
    d['active_scientific_jobs'] = 0
    d['scientific_status'] = 'response_match_not_established'
    if p.name == 'jobs.json':
        job = next(v for v in d['jobs'] if v.get('run_id') == cfg['run_id'])
        job.update(status='completed_native0_collection_and_numeric_readback_pending',
            native_exit_code=0, supervisor_native_exit_code=0, completed_UTC=native['UTC'],
            analysis_pid=None, supervisor_pid=None, historical_analysis_pid=native['child_pid'],
            historical_supervisor_pid=native['supervisor_pid'], G_updates=counts['G_updates'],
            new_AD_queries=counts['new_AD_queries'], actual_counters=counts,
            actual_counters_last_verified=counts, completed_short_candidate_runs=4,
            collection_complete=False, independent_readback_complete=False, terminal_readback=ref)
        job['execution'].update(status=job['status'], new_training_updates=counts['G_updates'],
            new_AD_queries=counts['new_AD_queries'], actual_costs_are_durable_lower_bounds_while_running=False,
            analysis_pid=None, supervisor_pid=None, last_verified_completed_counts=counts,
            native_exit_code=0, supervisor_exit_code=0)
    else:
        d['status'] = 'WP2_014_native0_collection_and_numeric_readback'
        d['wp2_terminal_readback'] = ref
        d['current_work_package'] = 'WP2'
        d['current_stage'] = 'WP2_014_saved_results_independent_readback'
        d['active_research'] = cfg['run_id']
        d['next_action'] = 'Finish selective collection and independent saved-array numeric readback; then select the fixed development path and advance bounded WP3.'
        d['last_verified_UTC'] = state['UTC']
        if 'wp2_execution' in d:
            d['wp2_execution'].update(status='completed_native0_collection_and_numeric_readback_pending',
                new_training_updates=counts['G_updates'], new_AD_queries=counts['new_AD_queries'],
                last_verified_completed_counts=counts, actual_costs_are_durable_lower_bounds_while_running=False,
                analysis_pid=None, supervisor_pid=None, historical_analysis_pid=native['child_pid'],
                historical_supervisor_pid=native['supervisor_pid'], native_exit_code=0, supervisor_exit_code=0)
        if p.name == 'PROGRAM.json':
            d.update(new_training_updates=total, current_run_training_updates=counts['G_updates'],
                wp2_G_updates=counts['G_updates'], wp2_AD_queries=counts['new_AD_queries'],
                active_scientific_jobs_status_basis='Single terminal snapshot: child and supervisor native0; no later dispatch.',
                training_update_accounting='Historical159667 plus actual014 G8000 once; startup197 is included, never added again.')
        if p.name == 'MASTER_EXECUTION_PLAN.json':
            d['active_work_package'] = 'WP2'
            wp = next(v for v in d['work_packages'] if v['id'] == 'WP2')
            wp['status'] = 'completed_native0_collection_and_numeric_readback_pending'
            wp['execution'].update(status=wp['status'], new_training_updates=counts['G_updates'],
                new_AD_queries=counts['new_AD_queries'], last_verified_completed_counts=counts,
                actual_costs_are_durable_lower_bounds_while_running=False, analysis_pid=None, supervisor_pid=None,
                native_exit_code=0, supervisor_exit_code=0)
            wp['terminal_readback'] = ref
            d['budgets']['short_candidate_runs']['completed'] = 4
            d['current_action'] = d['next_action']
        if p.name == 'CURRENT_SIM2REAL_PROGRAM.json':
            d['latest_run_id'] = cfg['run_id']
            d['cumulative_training_updates'] = total
            d['current_run_training_updates'] = counts['G_updates']
    write(p, d)
for p in paths:
    d = read(p)
    assert d['active_scientific_jobs'] == 0
assert read(ROOT/'PROGRAM.json')['new_training_updates'] == 167667
print(json.dumps(dict(status='native0_costs_registered_readback_pending', cumulative_training_updates=total,
    new_G_updates=8000, new_AD_queries=9872, active_scientific_jobs=0, short_candidate_runs_completed=4)))
