"""Register saved native terminal facts without rerunning scientific work."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT / 'receipts/wp1_013_20261015/postrun_root_01'
STATE = ROOT / 'receipts/wp1_013_20261015/native_repeat_fix01/wake_status_01/REMOTE_STATE.json'

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, obj, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

state = read(STATE)
native = state['runs/primary/supervisor/exit_receipt.json']
complete = state['runs/primary/artifacts/COMPLETE.json']
cfg = read(ROOT / 'wp1_013_config.json')
assert native['child_native_exit_code'] == native['exit_code'] == 0
assert complete and not state['runs/primary/artifacts/FAILED.json']
assert not state['protection_errors']
assert not native['unknown_inflight_ids'] and not native['failed_ids'] and not native['truncated_tail']
counts = {k: native['completed_counts'].get(k, 0) for k in cfg['budget_caps']}
assert counts == cfg['budget_caps']
assert native['entered_counts'] == native['completed_counts']
program = read(ROOT / 'PROGRAM.json')
assert program['new_training_updates'] == 159667
prior = program['wp1_execution']['prior_attempt_actual_cost']
total = {k: counts[k] + prior.get(k, 0) for k in counts}
assert total == program['wp1_execution']['aggregate_budget_caps']
now = dt.datetime.now(dt.timezone.utc).isoformat()
receipt = dict(schema='n01/wp1-native-terminal-readback/v1', UTC=now,
    basis='single saved remote state; collection and independent numerical verification still pending',
    state=str(STATE.relative_to(ROOT)), state_sha256=sha(STATE),
    native_exit=native, attempt=3, attempt3_actual_counts=counts,
    prior_two_failed_attempt_actual_counts=prior, aggregate_actual_counts=total,
    formal_response_acceptance=False, scientific_status='response_match_not_established',
    cumulative_training_updates=159667, new_training_updates=0, new_AD_queries=0,
    producer_method_choice=complete['method_choice'], independent_method_readback_pending=True,
    collection_complete=False, active_scientific_jobs=0)
write(OUT / 'NATIVE_COMPLETION_READBACK.json', receipt, True)
ref = dict(receipt=str((OUT / 'NATIVE_COMPLETION_READBACK.json').relative_to(ROOT)),
    receipt_sha256=sha(OUT / 'NATIVE_COMPLETION_READBACK.json'),
    native_exit_code=0, supervisor_exit_code=0, ended_UTC=native['UTC'],
    attempt3_actual_counts=counts, aggregate_actual_counts=total,
    source_commit=program['wp1_execution']['source_commit'],
    collection_complete=False, independent_readback_complete=False,
    producer_method_choice=complete['method_choice'], formal_response_acceptance=False)
files = [ROOT/'MASTER_EXECUTION_PLAN.json', ROOT/'PROGRAM.json', ROOT/'jobs.json', Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
backup = OUT / 'metadata_before_native_closeout'
backup.mkdir()
for f in files:
    shutil.copy2(f, backup / f.name)
for f in files:
    d = read(f)
    d['updated_UTC'] = now
    d['active_scientific_jobs'] = 0
    if f.name == 'jobs.json':
        j = next(j for j in d['jobs'] if j['job_id'] == 'N01_WP1_013_20261015')
        j.update(status='completed_native0_collection_pending', native_exit_code=0,
            supervisor_native_exit_code=0, analysis_pid=None, supervisor_pid=None,
            historical_analysis_pid=native['child_pid'], historical_supervisor_pid=native['supervisor_pid'],
            completed_UTC=native['UTC'], actual_counters=counts,
            total_actual_counters=total, terminal_readback=ref,
            collection_complete=False, independent_readback_complete=False)
    else:
        d['status'] = 'WP1_013_native0_postrun_readback'
        d['wp1_terminal_readback'] = ref
        if f.name == 'PROGRAM.json':
            d['current_stage'] = 'WP1_013_collection_and_independent_readback'
            d['active_scientific_jobs_status_basis'] = 'WP1 attempt3 child and supervisor native0; saved single terminal snapshot; no later dispatch'
        if f.name == 'MASTER_EXECUTION_PLAN.json':
            next(w for w in d['work_packages'] if w['id'] == 'WP1')['status'] = 'native0_collection_pending'
            d['current_action'] = 'Collect saved013; independently read back numeric tolerances and method choice before WP2.'
    write(f, d)
print(json.dumps(dict(status='native_terminal_facts_registered', active_scientific_jobs=0,
    cumulative_training_updates=159667, new_training_updates=0, aggregate_counts=total)))
