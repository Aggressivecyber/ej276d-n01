"""Close012 from persisted receipts, preserve metadata, and back up small evidence."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('E:/GAN/research/N01')
R = ROOT/'receipts/mechanism012_20261015'
A = ROOT/'runs/mechanism012_20261015/collected_01/runs/primary/artifacts'
REPORT = ROOT/'reports/mechanism012_20261015/N01_MECHANISM012_REPORT.md'
POINTER = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
BACKUP = Path('D:/GAN_backups/N01/mechanism012_20261015/closeout_01')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    data = (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    temp = path.with_suffix(path.suffix+'.close012.tmp')
    assert not temp.exists()
    temp.write_bytes(data)
    temp.replace(path)


def main():
    independent_path = R/'postrun_independent_01/NUMERIC_READBACK.json'
    independent = read(independent_path)
    assert independent['status'] == 'PASS' and independent['passed'] and not independent['failed_checks']
    assert len(independent['checks']) == 2163
    scope = independent['scientific_execution_in_this_readback']
    for key in ['original_dataset_reads','checkpoint_reads','NN_instances','NN_forward_rows','AD_queries','optimizer_created','sampler_created','DEV_reads','SSH_SCP_calls','pipeline_replays','new010_or011_evaluations']:
        assert scope[key] == 0, (key,scope[key])
    qa_path = R/'postrun_independent_01/REPORT_QA_FINAL.json'
    qa = read(qa_path)
    assert qa['status'] == 'PASS'
    assert qa['report_sha256'] == sha(REPORT)
    assert sha(REPORT) == 'b7087e5fccc2bf17980e21e1f372f453a6be67fd5cf64e4d193f678d82a08846'
    collection_path = R/'collection_01/COLLECTION_RECEIPT.json'
    collection = read(collection_path)
    assert collection['status'] == 'native_output_bytes_exclusively_verified'
    assert not collection['partial_evidence_only']
    complete = read(A/'COMPLETE.json')
    native = read(A.parent/'supervisor/exit_receipt.json')
    assert native['exit_code'] == 0 and native['reason'] is None
    assert complete['engineering_checks_passed'] and complete['inputs_unchanged']
    assert complete['frozen_state_unchanged'] and complete['head_tensors_unchanged']
    assert not complete['formal_response_acceptance']
    assert independent['gradient_sum']['relative_error'] < 1e-5
    for manifest_name,count in [('MECHANISM012_SOURCE_MANIFEST.json',69),('STABILITY011_SOURCE_MANIFEST.json',63),('SHAPE010_SOURCE_MANIFEST.json',53)]:
        manifest = read(ROOT/manifest_name)
        assert len(manifest['files']) == count
        for relative,expected in manifest['files'].items():
            expected_hash = expected if isinstance(expected,str) else expected['sha256']
            assert sha(ROOT/relative) == expected_hash, relative
    followup = read(R/'FOLLOWUP_STATE_RECEIPT.json')
    assert followup['automation']['status'] == 'PAUSED'
    assert sha(Path(followup['source'])) == followup['source_sha256']
    out = R/'final_state_01'
    out.mkdir(exist_ok=False)
    before = out/'metadata_before'
    before.mkdir()
    paths = [ROOT/'PROGRAM.json',ROOT/'jobs.json',POINTER]
    snapshots = []
    for path in paths:
        target = before/path.name
        target.write_bytes(path.read_bytes())
        snapshots.append({'path':str(path),'snapshot':str(target),'sha256_before':sha(target)})
    utc = datetime.now(timezone.utc).isoformat()
    result_path = R/'FINAL_RESULT_RECEIPT.json'
    shared = dict(status='mechanism012_closed_independently_read_back',
        current_stage='mechanism012_gradient_and_crop_diagnosis_closed',
        updated_UTC=utc,last_verified_UTC=utc,active_scientific_jobs=0,active_research=None,
        active_scientific_jobs_status_basis='012 native exit0 and full collection independently read back; no later N01 dispatch',
        latest_run_id='mechanism012_20261015',latest_training_run_id='shape010_20261014',
        current_run_training_updates=0,scientific_status='response_match_not_established',
        mechanism012_completed=True,mechanism012_collection_complete=True,
        mechanism012_independent_readback_complete=True,mechanism012_closeout_complete=True,
        mechanism012_independent_readback='receipts/mechanism012_20261015/postrun_independent_01/NUMERIC_READBACK.json',
        mechanism012_independent_readback_sha256=sha(independent_path),
        mechanism012_independent_readback_scope='saved arrays algebra, statistics, identities and persisted receipts; no complete CDF loss or original carrier/noise replay',
        mechanism012_report='reports/mechanism012_20261015/N01_MECHANISM012_REPORT.md',
        mechanism012_report_sha256=sha(REPORT),
        mechanism012_report_QA='receipts/mechanism012_20261015/postrun_independent_01/REPORT_QA_FINAL.json',
        mechanism012_result_receipt='receipts/mechanism012_20261015/FINAL_RESULT_RECEIPT.json',
        mechanism012_native_exit_code=0,mechanism012_bounded_budget_closed=True,
        mechanism012_additional_optimizer_updates=0,mechanism012_G_updates=0,mechanism012_D_updates=0,
        mechanism012_head_tensors_unchanged=True,mechanism012_formal_response_acceptance=False,
        mechanism012_analysis_pid=None,mechanism012_supervisor_pid=None,
        mechanism012_classification='QPSD_dominates_local_gradient_and_moving_crop_step_dependence_detected',
        mechanism012_next_design='saved_output_crop_boundary_read_only',mechanism012_next_training_budget=0,
        mechanism012_D_closeout_backup=str(BACKUP),mechanism012_publication_status='local_only_not_published',
        user_requested_pause=False,resume_requires_user_server_started_notice=False,automatic_restart=False,
        followup_automation_status='PAUSED',followup_scheduled_local=None,followup_next_run_local=None,
        next_action='Register a bounded saved-output crop-boundary method diagnosis; training remains0 until an evidence-based matched change is specified and independently reviewed.')
    for path in [paths[0],paths[2]]:
        value = read(path)
        assert value['new_training_updates'] == 159667
        value.pop('mechanismism012_source_package_sha256',None)
        value.update(shared)
        write(path,value)
    jobs = read(paths[1])
    matches = [j for j in jobs['jobs'] if j.get('run_id') == 'mechanism012_20261015']
    assert len(matches) == 1
    matches[0].update(status='completed_native0_collected_independently_read_back',
        independent_readback_complete=True,independent_readback=shared['mechanism012_independent_readback'],
        report=shared['mechanism012_report'],report_sha256=sha(REPORT),
        final_report_QA=shared['mechanism012_report_QA'],completion_receipt=shared['mechanism012_result_receipt'],
        bounded_budget_closed=True,additional_optimizer_updates=0,scientific_status='response_match_not_established',
        last_verified_UTC=utc,analysis_pid=None,supervisor_pid=None,followup_automation_status='PAUSED')
    jobs.update(updated_UTC=utc,active_scientific_jobs=0)
    write(paths[1],jobs)
    for item in snapshots:
        item['sha256_after'] = sha(Path(item['path']))
    refs = [collection_path,A/'COMPLETE.json',A.parent/'supervisor/exit_receipt.json',
        R/'LOCAL_SOURCE_PRESERVATION.json',R/'FOLLOWUP_STATE_RECEIPT.json',R/'REPORT_BUILD_RECEIPT.json',
        independent_path,qa_path,REPORT,ROOT/'MECHANISM012_SOURCE_MANIFEST.json',ROOT/'MECHANISM012_BINDINGS.json']
    result = dict(schema='n01/mechanism012-final-result/v1',UTC=utc,
        status='completed_native0_collected_independently_read_back',run_id='mechanism012_20261015',
        native_exit_code=0,active_scientific_jobs=0,actual_native_counters=collection['job']['counters'],
        cumulative_scientific_training_updates=159667,new_training_updates=0,
        independent_checks=len(independent['checks']),independent_failed_checks=0,
        independent_readback_execution=scope,report_sha256=sha(REPORT),
        scientific_status='response_match_not_established',formal_response_acceptance=False,
        head_tensors_unchanged=True,frozen_inputs_and_sources_unchanged=True,
        local_source_guard_manifests_and_file_counts={'012':69,'011':63,'010':53},
        files=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in refs],
        metadata_updates=snapshots,followup_status='PAUSED',new_wake_created=False,
        followup_reason='native diagnostic already completed at first start verification',
        automatic_restart=False,external_publication_performed=False,D_closeout_backup=str(BACKUP),
        decisive_results={'QPSD_L2':independent['gradient_vectors']['QPSD']['L2'],
            'total_L2':independent['gradient_vectors']['total']['L2'],
            'gradient_sum_relative_error':independent['gradient_sum']['relative_error'],
            'changed_parent_union':independent['crop_changed_union_parents'],'parent_denominator':512,
            'AD_loss_per_V':independent['direction_probe']['d_AD_loss_per_V'],
            'next_design':shared['mechanism012_next_design'],'next_training_budget':0},
        limitations=independent['limitations'],NN_rows_in_this_finalization=0,AD_queries_in_this_finalization=0,
        checkpoint_loads_in_this_finalization=0,optimizer_steps_in_this_finalization=0)
    write(result_path,result)
    backup_files = set(refs+[result_path,ROOT/'PROGRAM.json',ROOT/'jobs.json'])
    for rel in ['mechanism012_policy.json','docs/MECHANISM012_PROTOCOL.md',
                'scripts/mechanism012_ops.py','scripts/supervise_mechanism012.py',
                'scripts/run_mechanism012.py','scripts/mechanism012_losses.py']:
        backup_files.add(ROOT/rel)
    backup_files.update(p for p in (R/'native_result_json').iterdir() if p.is_file())
    needed = sum(p.stat().st_size for p in backup_files)+POINTER.stat().st_size
    assert shutil.disk_usage('D:/').free > needed+256*2**20
    assert not BACKUP.exists()
    BACKUP.mkdir(parents=True)
    inventory = []
    for source in sorted(backup_files):
        target = BACKUP/'N01'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
        assert sha(source) == sha(target)
        inventory.append({'source':str(source),'copy':str(target),'sha256':sha(target),'bytes':target.stat().st_size})
    pointer_copy = BACKUP/'CURRENT_SIM2REAL_PROGRAM.json'
    pointer_copy.write_bytes(POINTER.read_bytes())
    assert sha(pointer_copy) == sha(POINTER)
    inventory.append({'source':str(POINTER),'copy':str(pointer_copy),'sha256':sha(pointer_copy),'bytes':pointer_copy.stat().st_size})
    sums = '\n'.join(f"{item['sha256']}  {Path(item['copy']).relative_to(BACKUP).as_posix()}" for item in inventory)+'\n'
    (BACKUP/'SHA256SUMS.txt').write_text(sums,encoding='utf-8')
    backup_receipt = dict(schema='n01/mechanism012-closeout-backup/v1',UTC=utc,status='all_small_evidence_copies_SHA256_verified',
        root=str(BACKUP),files=inventory,total_bytes=sum(item['bytes'] for item in inventory),
        SHA256SUMS_sha256=sha(BACKUP/'SHA256SUMS.txt'),arrays_and_result_tar_backup=False,
        full_result_tar_preserved_E=str(R/'collection_01/mechanism012_complete_01.tar'),
        full_result_tar_sha256=collection['archive_sha256'],source_package_backup_separately_verified=True)
    write(out/'CLOSEOUT_BACKUP_RECEIPT.json',backup_receipt)
    for path in paths:
        assert read(path)['active_scientific_jobs'] == 0
    assert read(ROOT/'PROGRAM.json')['mechanism012_independent_readback_complete']
    assert sha(REPORT) == qa['report_sha256']
    print(json.dumps({'status':shared['status'],'final_result_receipt':str(result_path),'sha256':sha(result_path),
        'backup':str(BACKUP),'backup_files':len(inventory),'backup_bytes':backup_receipt['total_bytes']},ensure_ascii=False))


if __name__ == '__main__':
    main()
