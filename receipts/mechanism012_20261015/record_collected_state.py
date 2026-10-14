"""Record terminal native evidence without executing a model or numeric array."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('E:/GAN/research/N01')
RECEIPTS = ROOT / 'receipts/mechanism012_20261015'
COLLECTED = ROOT / 'runs/mechanism012_20261015/collected_01/runs/primary'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    temp = path.with_suffix(path.suffix + '.record012.tmp')
    assert not temp.exists(), temp
    temp.write_bytes(data)
    temp.replace(path)


def main():
    out = RECEIPTS / 'terminal_state_01'
    out.mkdir(exist_ok=False)
    before = out / 'metadata_before'
    before.mkdir()
    complete = read(COLLECTED / 'artifacts/COMPLETE.json')
    native = read(COLLECTED / 'supervisor/exit_receipt.json')
    collection = read(RECEIPTS / 'collection_01/COLLECTION_RECEIPT.json')
    freeze = read(RECEIPTS / 'freeze_01/FREEZE_RECEIPT.json')
    assert native['exit_code'] == 0 and native.get('reason') is None
    assert complete['point_checks_passed'] and complete['engineering_checks_passed']
    assert complete['head_tensors_unchanged'] and complete['frozen_state_unchanged']
    assert complete['finite_formal_AD'] and complete['inputs_unchanged']
    assert collection['status'] == 'native_output_bytes_exclusively_verified'
    assert not collection['partial_evidence_only']
    assert not complete['formal_response_acceptance']
    counters = collection['job']['counters']
    assert (counters['old_A_rows'], counters['noise005_rows'], counters['head_final_rows']) == (2048, 2048, 2048)
    assert (counters['scalar_AD_queries'], counters['masked_endpoint_VJP_queries']) == (9, 2)
    assert counters['ADC_hard_or_STE_rows'] == 49152
    for key in ['optimizer_created', 'optimizer_steps', 'G_updates', 'D_updates', 'noise_updates', 'population_updates', 'C2ST_updates', 'DEV_rows', 'sampler_created', 'new_latent_draws', 'new_noise_draws']:
        assert counters[key] == 0, key
    utc = datetime.now(timezone.utc).isoformat()
    state = 'mechanism012_native_complete_collected_pending_independent_readback'
    files = [ROOT / 'PROGRAM.json', ROOT / 'jobs.json', Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
    snapshots = []
    for path in files:
        snapshot = before / path.name
        snapshot.write_bytes(path.read_bytes())
        snapshots.append({'path': str(path), 'snapshot': str(snapshot), 'sha256_before': digest(snapshot)})
    shared = dict(status=state, current_stage=state, updated_UTC=utc,
        active_scientific_jobs=0, active_research='mechanism012_20261015',
        active_scientific_jobs_status_basis='012 native terminal exit0, COMPLETE and byte-verified collection; no later N01 dispatch',
        latest_run_id='mechanism012_20261015', latest_training_run_id='shape010_20261014',
        scientific_status='response_match_not_established', current_run_training_updates=0,
        user_requested_pause=False, resume_requires_user_server_started_notice=False,
        automatic_restart=False, pause_reason=None, followup_automation_status='PAUSED',
        followup_next_run_local=None, followup_scheduled_local=None,
        mechanism012_completed=True, mechanism012_native_exit_code=0,
        mechanism012_model_forward_rows=sum(counters[k] for k in ['old_A_rows','noise005_rows','head_final_rows']),
        mechanism012_gradient_evaluations=11, mechanism012_actual_counters=counters,
        mechanism012_analysis_pid=None, mechanism012_supervisor_pid=None,
        mechanism012_historical_analysis_pid=collection['job']['child_pid'],
        mechanism012_historical_supervisor_pid=collection['job']['supervisor_pid'],
        mechanism012_collection_complete=True, mechanism012_independent_readback_complete=False,
        mechanism012_source_package_sha256=freeze['package_sha256'],
        mechanism012_source_freeze_pending=False, mechanism012_head_tensors_unchanged=True,
        mechanism012_started_UTC=read(RECEIPTS/'start_native_01/START_RECEIPT.json').get('UTC'),
        mechanism012_completed_UTC=complete['UTC'],
        mechanism012_start_receipt='receipts/mechanism012_20261015/start_native_01/START_RECEIPT.json',
        mechanism012_start_verification='receipts/mechanism012_20261015/start_verification_01/START_VALIDATION.json',
        mechanism012_collection_receipt='receipts/mechanism012_20261015/collection_01/COLLECTION_RECEIPT.json',
        mechanism012_completion_receipt='runs/mechanism012_20261015/collected_01/runs/primary/artifacts/COMPLETE.json',
        mechanism012_formal_response_acceptance=False,
        next_action='Independently read saved012 gradients, row mechanisms and six probes; write bounded interpretation and next registered design before any new computation.')
    for path in [files[0], files[2]]:
        value = read(path)
        assert value['new_training_updates'] == 159667
        value.pop('mechanismism012_source_package_sha256', None)
        value.update(shared)
        write(path, value)
    jobs = read(files[1])
    matched = [j for j in jobs['jobs'] if j.get('run_id') == 'mechanism012_20261015']
    assert len(matched) == 1
    job = matched[0]
    job.update(status='completed_native0_collected_pending_independent_readback',
        actual_counters=counters, native_exit_code=0, started_UTC=shared['mechanism012_started_UTC'],
        completed_UTC=complete['UTC'], collection_complete=True, independent_readback_complete=False,
        collection_receipt=shared['mechanism012_collection_receipt'],
        completion_receipt=shared['mechanism012_completion_receipt'],
        supervisor_pid=None, analysis_pid=None, historical_supervisor_pid=collection['job']['supervisor_pid'],
        historical_analysis_pid=collection['job']['child_pid'], bounded_budget_closed=True,
        additional_optimizer_updates=0, followup_automation_status='PAUSED')
    job.update(counters)
    jobs['updated_UTC'] = utc
    jobs['active_scientific_jobs'] = 0
    write(files[1], jobs)
    for item in snapshots:
        item['sha256_after'] = digest(Path(item['path']))
    receipt = dict(schema='n01/mechanism012-terminal-metadata/v1', UTC=utc, status=state,
        native_exit_code=0, COMPLETE_sha256=digest(COLLECTED/'artifacts/COMPLETE.json'),
        native_exit_sha256=digest(COLLECTED/'supervisor/exit_receipt.json'),
        collection_receipt_sha256=digest(RECEIPTS/'collection_01/COLLECTION_RECEIPT.json'),
        files=snapshots, counters=counters, active_scientific_jobs=0,
        new_training_updates=0, NN_rows=0, AD_queries=0, optimizer_steps=0,
        scientific_status='response_match_not_established')
    write(out/'TERMINAL_STATE_RECEIPT.json', receipt)
    for path in files:
        assert read(path)['active_scientific_jobs'] == 0
    print(json.dumps({'status':state,'receipt':str(out/'TERMINAL_STATE_RECEIPT.json')},ensure_ascii=False))


if __name__ == '__main__':
    main()
