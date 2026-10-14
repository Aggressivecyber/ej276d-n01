"""Record the user's server pause from saved receipts; no SSH or compute."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

ROOT = Path('E:/GAN/research/N01')
PAUSE = ROOT / 'receipts/stability011_20261014/user_server_pause_01'
REL = PAUSE.relative_to(ROOT).as_posix()
COLLECTED = ROOT / 'runs/stability011_20261014/collected_01'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value, *, new=False):
    path = Path(path)
    if new and path.exists():
        raise RuntimeError(f'Preserve existing receipt: {path}')
    temporary = path.with_name(path.name + '.pause_tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def main():
    timestamp = datetime.now(timezone.utc).isoformat()
    processes = read(PAUSE / 'REMOTE_PROCESS_READBACK.json')
    collection = read(PAUSE / 'collection_01/COLLECTION_RECEIPT.json')
    terminal = read(COLLECTED / 'runs/primary/supervisor/exit_receipt.json')
    complete = read(COLLECTED / 'runs/primary/artifacts/COMPLETE.json')
    assert processes['ssh_exit_code'] == 0 and not processes['matched_project_python_processes']
    assert terminal['exit_code'] == terminal['child_native_exit_code'] == 0
    assert complete['bootstrap_replicates_completed'] == complete['bootstrap_replicates_attempted'] == complete['bootstrap_replicates_computed_successfully'] == 500
    assert all(number == 0 for number in complete['counters'].values())
    assert '\nstatus = "PAUSED"\n' in (PAUSE / 'automation_after_pause.toml').read_text(encoding='utf-8')
    assert collection['job']['bootstrap_replicates_completed'] == 500
    receipt = dict(schema='n01/user-server-pause/v1', UTC=timestamp,
        status='user_pause_registered_remote_quiescent_and_results_collected',
        user_instruction=processes['authorization'], run_id='stability011_20261014',
        active_project_jobs_at_last_native_readback=0, native_process_readback_UTC=processes['UTC'],
        signal_sent=False, signal_reason='011 had already completed normally; no matching project Python processes',
        native_child_exit_code=terminal['child_native_exit_code'], native_supervisor_exit_code=terminal['exit_code'],
        completed_UTC=terminal['finished_UTC'], attempted=500, computed=500, saved=500,
        counters=complete['counters'], collected_root=str(COLLECTED),
        collection_archive_sha256=collection['archive_sha256'], collection_file_count=collection['file_count'],
        scientific_status='response_match_not_established', independent_local_readback='pending',
        followup_automation_id='n01', followup_automation_status='PAUSED',
        resume_requires_user_server_started_notice=True, automatic_restart=False,
        next_diagnostic_012_registered=False, next_diagnostic_012_started=False,
        server_power_state='unknown; user reported planned temporary shutdown',
        evidence={str(path.relative_to(ROOT)):digest(path) for path in (
            PAUSE/'REMOTE_PROCESS_READBACK.json', PAUSE/'automation_before_pause.toml',
            PAUSE/'automation_after_pause.toml', PAUSE/'AUTOMATION_PAUSE_UPDATE_RESULT.json',
            PAUSE/'collection_01/COLLECTION_RECEIPT.json', COLLECTED/'runs/primary/supervisor/exit_receipt.json',
            COLLECTED/'runs/primary/artifacts/COMPLETE.json')})
    write(PAUSE/'USER_PAUSE_RECEIPT.json', receipt, new=True)
    for metadata_path in (ROOT/'PROGRAM.json', Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')):
        value = read(metadata_path)
        assert value['program_id'] == 'N01'
        absolute_links = metadata_path.parent != ROOT
        def link(relative):
            return str(ROOT/relative) if absolute_links else relative
        value.update(status='paused_user_request_waiting_server_restart', updated_UTC=timestamp,
            current_stage='stability011_complete_collected_local_readback_pending', active_scientific_jobs=0,
            active_scientific_jobs_status_basis='one native process readback was empty;011 child/supervisor native0 and500saved',
            active_research=None, last_verified_UTC=processes['UTC'],
            next_action='Finish local receipt/readback closeout, then wait for the user to report the server has restarted. Do not launch012 or reactivate followup automatically.',
            followup_automation_status='PAUSED', followup_scheduled_local=None, followup_next_run_local=None,
            followup_pause_receipt=link(REL+'/USER_PAUSE_RECEIPT.json'),
            user_requested_pause=True, pause_reason='temporary server shutdown requested by user',
            resume_requires_user_server_started_notice=True, automatic_restart=False,
            stability011_analysis_pid=None, stability011_supervisor_pid=None,
            stability011_completed_bootstrap_replicates=500, stability011_completed_bootstrap_replicates_last_verified=500,
            stability011_attempted_bootstrap_replicates_last_verified=500, stability011_computed_bootstrap_replicates_last_verified=500,
            stability011_unobserved_bootstrap_progress='full500 terminal receipts collected and verified',
            stability011_native_exit_code=0, stability011_completed_UTC=terminal['finished_UTC'],
            stability011_collection_complete=True, stability011_independent_readback_complete=False,
            stability011_collection_receipt=link(REL+'/collection_01/COLLECTION_RECEIPT.json'),
            stability011_pause_receipt=link(REL+'/USER_PAUSE_RECEIPT.json'),
            stability011_historical_analysis_pid=terminal['child_pid'], stability011_historical_supervisor_pid=terminal['supervisor_pid'])
        assert value['new_training_updates'] == 159667 and value['scientific_status'] == 'response_match_not_established'
        write(metadata_path, value)
    jobs_path = ROOT/'jobs.json'
    jobs = read(jobs_path)
    targets = [job for job in jobs['jobs'] if job.get('run_id') == 'stability011_20261014']
    assert len(targets) == 1
    targets[0].update(status='completed_native0_collected_user_paused_pending_local_readback',
        completed_bootstrap_replicates=500, attempted_bootstrap_replicates=500, computed_bootstrap_replicates=500,
        last_verified_UTC=processes['UTC'], completed_UTC=terminal['finished_UTC'], native_exit_code=0,
        supervisor_pid=None, analysis_pid=None, historical_supervisor_pid=terminal['supervisor_pid'],
        historical_analysis_pid=terminal['child_pid'], collection_complete=True,
        actual_unobserved_bootstrap_progress='full500 terminal receipts collected and verified',
        collection_receipt=REL+'/collection_01/COLLECTION_RECEIPT.json',
        pause_receipt=REL+'/USER_PAUSE_RECEIPT.json', user_requested_pause=True,
        followup_automation_status='PAUSED', automatic_restart=False, resume_requires_user_server_started_notice=True)
    jobs.update(active_scientific_jobs=0, updated_UTC=timestamp, user_requested_pause=True,
        resume_requires_user_server_started_notice=True)
    write(jobs_path, jobs)
    print(json.dumps(dict(status='pause_recorded', active_scientific_jobs=0, saved=500, followup='PAUSED',
        receipt=str(PAUSE/'USER_PAUSE_RECEIPT.json')), ensure_ascii=False))

if __name__ == '__main__':
    main()
