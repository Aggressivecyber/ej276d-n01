"""Commit the verified startup and one-time followup metadata only."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path('E:/GAN/research/N01')
prefix = 'receipts/wp1_013_20261015/native_repeat_fix01/'
files = ['README.md', 'MASTER_EXECUTION_PLAN.json', 'PROGRAM.json', 'jobs.json'] + [prefix + name for name in (
    'LOCAL_SOURCE_COMMIT_RECEIPT.json', 'SOURCE_COMMIT_READBACK.json', 'EXECUTION_STARTED.json',
    'FOLLOWUP_PLAN.json', 'FOLLOWUP_PROMPT.txt', 'FOLLOWUP_RECEIPT.json', 'D_STARTUP_BACKUP_RECEIPT.json',
    'deploy_native_03/REMOTE_INSTALL_RECEIPT.json', 'start_native_03/START_RECEIPT.json',
    'start_verification_03/START_VALIDATION.json', 'start_verification_03/REMOTE_STATE.json',
    'save_started_metadata.py')]
assert all((root / name).is_file() and (root / name).stat().st_size < 2 * 1024 ** 2 for name in files)
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root, text=True).strip()
assert not subprocess.check_output(['git', 'diff', '--name-only', '--', 'scripts', 'src', 'wp1_013_config.json', 'WP1_SOURCE_MANIFEST.json',
    'OBSERVATION_CONTRACT.json', 'DATA_ROLES.json', 'docs/ACCEPTANCE_CONTRACT.md'], cwd=root, text=True).strip()
followup = json.loads((root / prefix / 'FOLLOWUP_RECEIPT.json').read_text('utf-8'))
assert followup['actual_saved_settings_verified'] and followup['correct_start_verified'] and followup['count'] == 1
subprocess.run(['git', 'add', '--'] + files, cwd=root, check=True)
assert set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root, text=True).splitlines()) == set(files)
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=root, check=True)
subprocess.run(['git', 'commit', '-m', 'Record verified WP1 startup and one-time scheduled collection'], cwd=root, check=True)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
result = dict(schema='n01/wp1-startup-metadata-commit/v1', UTC=datetime.now(timezone.utc).isoformat(), metadata_commit=head,
    executed_source_commit='7be67b8d3809246a58bcb6f03cd531e8f7539ad3', files=files,
    public_push_performed=False, no_new_remote_calls=True, no_scientific_source_or_inputs_changed=True)
with (root / prefix / 'STARTUP_METADATA_COMMIT_RECEIPT.json').open('xb') as handle:
    handle.write((json.dumps(result, indent=2) + '\n').encode('utf-8'))
print(json.dumps(dict(metadata_commit=head, executed_source_commit=result['executed_source_commit'], files=len(files),
    public_push_performed=False, no_new_remote_calls=True)))
