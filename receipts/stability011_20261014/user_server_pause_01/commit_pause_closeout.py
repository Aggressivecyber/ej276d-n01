"""Commit explicitly selected011 source/receipts, preserving unrelated files."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path('E:/GAN/research/N01')
RECEIPTS = ROOT/'receipts/stability011_20261014'
PAUSE = RECEIPTS/'user_server_pause_01'

def git(*arguments):
    result = subprocess.run(['git', '-C', str(ROOT), *arguments], capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    return result.stdout.strip()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')

def main():
    final_path = PAUSE/'FINAL_SERVER_PAUSE_RECEIPT.json'
    final = json.loads(final_path.read_text(encoding='utf-8'))
    assert final['resume_requires_user_server_started_notice'] is True
    assert final['followup_automation_status'] == 'PAUSED'
    assert not git('diff', '--cached', '--name-only'), 'Existing staged files must be preserved'
    names = ['PROGRAM.json', 'jobs.json', 'STABILITY011_SOURCE_MANIFEST.json', 'STABILITY011_BINDINGS.json',
        'stability011_policy.json', 'docs/STABILITY011_PROTOCOL.md', 'scripts/run_stability011.py',
        'scripts/stability011_metrics.py', 'scripts/stability011_ops.py', 'scripts/supervise_stability011.py']
    directories = [RECEIPTS] + [RECEIPTS/name for name in (
        'registration_01', 'resource_preflight_01', 'method_review_01', 'freeze_01', 'deployment_01',
        'start_native_01', 'start_verification_01', 'gradient_design_01', 'independent_01',
        'design_01', 'design_01/actual_scripts_cpu_03', 'ops_engineering_01/check02',
        'independent_01/metrics_synthetic_01', 'independent_01/schema_archive_cpu_02',
        'independent_01/supervisor_mock_03', 'user_server_pause_01',
        'user_server_pause_01/collection_01', 'user_server_pause_01/native_before',
        'user_server_pause_01/native_result_json', 'user_server_pause_01/metadata_before_pause',
        'user_server_pause_01/independent_readback_01')]
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix in ('.json', '.py', '.toml', '.md', '.txt') and path.stat().st_size < 1200000:
                names.append(path.relative_to(ROOT).as_posix())
    names = sorted(set(names))
    assert all((ROOT/name).is_file() for name in names)
    assert all('/fake_output/' not in name and '/artificial_contract_root/' not in name for name in names)
    assert all(Path(name).suffix not in ('.npz', '.pt', '.tar', '.gz', '.jsonl') for name in names)
    before_commit = git('rev-parse', 'HEAD')
    selected = dict(schema='n01/stability011-selected-local-commit/v1', UTC=datetime.now(timezone.utc).isoformat(),
        previous_commit=before_commit, files={name:dict(bytes=(ROOT/name).stat().st_size, sha256=sha(ROOT/name)) for name in names},
        unrelated_files_staged=False, raw_array_or_archive_staged=False, github_push=False)
    stage_manifest = PAUSE/'SELECTED_COMMIT_FILES.json'
    save_new(stage_manifest, selected)
    names.append(stage_manifest.relative_to(ROOT).as_posix())
    git('add', '--', *names)
    staged_names = git('diff', '--cached', '--name-only').splitlines()
    assert set(staged_names) == set(names), 'Actual staged set differs from selected files'
    git('diff', '--cached', '--check')
    git('commit', '-m', 'Complete STABILITY011 saved-bank analysis and pause for server shutdown')
    commit = git('rev-parse', 'HEAD')
    assert commit != before_commit
    assert not git('status', '--porcelain', '--untracked-files=no'), 'Tracked changes remain after own commit'
    for name, row in selected['files'].items():
        assert sha(ROOT/name) == row['sha256'], f'Working file changed during local commit: {name}'
    manifest = json.loads((ROOT/'STABILITY011_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/name) == expected for name, expected in manifest['files'].items())
    current_path = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
    current = json.loads(current_path.read_text(encoding='utf-8'))
    assert current['status'] == 'paused_user_request_waiting_server_restart'
    assert current['stability011_closeout_complete'] is True and current['active_scientific_jobs'] == 0
    output = dict(schema='n01/stability011-local-pause-delivery/v1', UTC=datetime.now(timezone.utc).isoformat(),
        status='local_closeout_saved_and_user_paused_waiting_server_notice', local_commit=commit, previous_commit=before_commit,
        committed_file_count=len(staged_names), tracked_working_tree_clean=True, unrelated_untracked_files_preserved=True,
        github_push=False, final_pause_receipt=str(final_path), final_pause_receipt_sha256=sha(final_path),
        current_pointer=str(current_path), current_pointer_sha256=sha(current_path),
        source_files_verified=len(manifest['files']), active_jobs_at_last_native_readback=0,
        native500_already_complete=True, followup_automation_status='PAUSED', automatic_restart=False,
        resume_requires_user_server_started_notice=True, next_diagnostic_012_started=False,
        final_delivery_receipt_is_postcommit_and_not_in_its_own_commit=True)
    receipt_path = PAUSE/'LOCAL_FINAL_DELIVERY_RECEIPT.json'
    save_new(receipt_path, output)
    assert json.loads(receipt_path.read_text(encoding='utf-8')) == output
    print(json.dumps(output, ensure_ascii=False))

if __name__ == '__main__':
    main()
