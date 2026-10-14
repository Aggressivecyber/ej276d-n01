"""Commit only owned012 sources and small evidence; retain scientific payloads on E."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('E:/GAN/research/N01')
R = ROOT/'receipts/mechanism012_20261015'
POINTER = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
ALLOWED = {'.py','.json','.jsonl','.md','.txt'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments):
    result = subprocess.run(['git','-c','core.autocrlf=false',*arguments],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'git {arguments[0]} failed: {result.stderr}')
    return result.stdout.strip()


def select():
    source = read(ROOT/'MECHANISM012_SOURCE_MANIFEST.json')
    paths = {ROOT/relative for relative in source['files']}
    for relative in ['PROGRAM.json','jobs.json','MECHANISM012_SOURCE_MANIFEST.json','MECHANISM012_BINDINGS.json',
            'reports/mechanism012_20261015/N01_MECHANISM012_REPORT.md']:
        paths.add(ROOT/relative)
    paths.update(p for p in R.iterdir() if p.is_file() and p.suffix in ALLOWED)
    stages = ['design_01','ops_engineering_01','independent_01','restart_preflight_01','registration_01',
        'freeze_01','deployment_01','start_native_01','start_verification_01','collection_01',
        'terminal_state_01','postrun_independent_01','final_state_01','native_result_json','report_versions',
        'finalization_attempt01_before_QA_schema_fix']
    for stage in stages:
        directory = R/stage
        if not directory.exists():
            continue
        paths.update(p for p in directory.iterdir() if p.is_file() and p.suffix in ALLOWED)
        for snapshot_directory in ['metadata_before','metadata_before_registration','before','originals']:
            snapshots = directory/snapshot_directory
            if snapshots.exists():
                paths.update(p for p in snapshots.iterdir() if p.is_file() and p.suffix in ALLOWED)
    paths.add(R/'design_01/cpu_pipeline_01/ENGINEERING_RECEIPT.json')
    assert all(p.is_file() and p.suffix in ALLOWED and p.is_relative_to(ROOT) for p in paths)
    assert all('LOCAL_FINAL_DELIVERY_RECEIPT' not in p.name for p in paths)
    return sorted(paths,key=lambda p:p.relative_to(ROOT).as_posix())


def main():
    result_path = R/'FINAL_RESULT_RECEIPT.json'
    final = read(result_path)
    assert final['status'] == 'completed_native0_collected_independently_read_back'
    assert final['active_scientific_jobs'] == 0 and final['new_training_updates'] == 0
    assert not final['formal_response_acceptance']
    report = ROOT/'reports/mechanism012_20261015/N01_MECHANISM012_REPORT.md'
    qa = read(R/'postrun_independent_01/REPORT_QA_FINAL.json')
    assert qa['status'] == 'PASS' and qa['report']['sha256'] == sha(report)
    assert not git('diff','--cached','--name-only'), 'existing staged changes; keep intact and inspect'
    tracked_changes = set(git('diff','--name-only').splitlines())
    assert tracked_changes <= {'PROGRAM.json','jobs.json'}, tracked_changes
    paths = select()
    parent = git('rev-parse','HEAD')
    assert parent == '0b486810d5570306bf917c67a24712d507350844'
    selection_path = R/'SELECTED_COMMIT_FILES.json'
    assert not selection_path.exists()
    selection = dict(schema='n01/mechanism012-explicit-local-commit-selection/v1',
        UTC=datetime.now(timezone.utc).isoformat(),parent_commit=parent,
        files=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths],
        scientific_payloads_staged=False,old_untracked_preserved=True,
        scope='owned012 sources, controls, small receipts and current PROGRAM/jobs; no NPZ/PT/tar/log')
    selection_path.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    message = R/'COMMIT_MESSAGE.txt'
    message.write_text('Close N01 mechanism012 gradient and crop diagnosis\n\nRecord native0 completion and independent saved-output readback, with all eleven AD queries and zero training updates. Preserve the frozen inputs and bound the next step to crop-boundary method diagnosis.\n',encoding='utf-8')
    paths += [selection_path,message]
    relative = [p.relative_to(ROOT).as_posix() for p in paths]
    git('add','--',*relative)
    staged = set(git('diff','--cached','--name-only').splitlines())
    assert staged <= set(relative)
    assert not any(Path(p).suffix in {'.npz','.pt','.pth','.tar','.gz','.log','.npy'} for p in staged)
    print(json.dumps({'phase':'selected_paths_staged','staged_files':len(staged),'bytes_in_selection':sum(x['bytes'] for x in selection['files'])},ensure_ascii=False),flush=True)
    git('commit','--only','--file',str(message),'--',*relative)
    commit = git('rev-parse','HEAD')
    committed_paths = set(git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines())
    assert committed_paths == staged
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')
    for item in selection['files']:
        assert sha(ROOT/item['path']) == item['sha256']
    program = read(ROOT/'PROGRAM.json')
    jobs = read(ROOT/'jobs.json')
    pointer = read(POINTER)
    assert program['status'] == pointer['status'] == 'mechanism012_closed_independently_read_back'
    assert program['new_training_updates'] == pointer['new_training_updates'] == 159667
    assert program['latest_training_run_id'] == pointer['latest_training_run_id'] == 'shape010_20261014'
    assert program['active_scientific_jobs'] == jobs['active_scientific_jobs'] == pointer['active_scientific_jobs'] == 0
    source = read(ROOT/'MECHANISM012_SOURCE_MANIFEST.json')
    for relative_path,expected in source['files'].items():
        assert sha(ROOT/relative_path) == expected
    package = R/'freeze_01/mechanism012_source.tar.gz'
    freeze = read(R/'freeze_01/FREEZE_RECEIPT.json')
    assert sha(package) == sha(Path(freeze['source_backup'])) == freeze['package_sha256']
    archive = R/'collection_01/mechanism012_complete_01.tar'
    assert sha(archive) == final['files'][0].get('archive_sha256', read(R/'collection_01/COLLECTION_RECEIPT.json')['archive_sha256'])
    refs = [result_path,report,R/'postrun_independent_01/NUMERIC_READBACK.json',R/'postrun_independent_01/REPORT_QA_FINAL.json',
        R/'final_state_01/CLOSEOUT_BACKUP_RECEIPT.json',ROOT/'PROGRAM.json',ROOT/'jobs.json',POINTER,selection_path]
    delivery = dict(schema='n01/mechanism012-local-final-delivery/v1',UTC=datetime.now(timezone.utc).isoformat(),
        status='local_commit_and_final_receipts_reopened_SHA256_verified',local_commit=commit,parent_commit=parent,
        committed_changed_files=len(committed_paths),tracked_worktree_clean=True,old_untracked_preserved=True,
        active_scientific_jobs=0,new_training_updates=0,cumulative_scientific_training_updates=159667,
        native_exit_code=0,independent_readback_PASS=True,final_report_QA_PASS=True,
        followup_status='PAUSED',automatic_restart=False,external_publication_performed=False,
        scientific_status='response_match_not_established',formal_response_acceptance=False,
        files=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in refs],
        result_archive={'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size},
        source_package={'path':str(package),'sha256':sha(package)},
        D_backup=str(Path(freeze['source_backup']).parent),
        D_closeout_backup=str(Path(final['D_closeout_backup'])),
        metadata_and_receipts_changed_after_commit=False)
    delivery_path = R/'LOCAL_FINAL_DELIVERY_RECEIPT.json'
    assert not delivery_path.exists()
    delivery_path.write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    supplemental = Path(final['D_closeout_backup'])/'supplements/final_delivery_01'
    supplemental.mkdir(parents=True,exist_ok=False)
    copy = supplemental/delivery_path.name
    copy.write_bytes(delivery_path.read_bytes())
    assert sha(copy) == sha(delivery_path)
    (supplemental/'SHA256SUMS.txt').write_text(f'{sha(copy)}  {copy.name}\n',encoding='utf-8')
    assert read(delivery_path)['local_commit'] == commit
    print(json.dumps({'status':delivery['status'],'commit':commit,'changed_files':len(committed_paths),
        'receipt':str(delivery_path),'sha256':sha(delivery_path)},ensure_ascii=False))


if __name__ == '__main__':
    main()
