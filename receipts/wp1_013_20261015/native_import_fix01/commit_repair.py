"""Commit only the declared repair controls and small engineering evidence."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path('E:/GAN/research/N01')
files = ['.gitattributes', 'README.md', 'PROGRAM.json', 'jobs.json', 'MASTER_EXECUTION_PLAN.json',
    'wp1_013_config.json', 'WP1_SOURCE_MANIFEST.json', 'scripts/wp1_ops.py', 'scripts/wp1_controller.py',
    'receipts/wp1_013_20261015/prepare_runtime_fix.py',
    'receipts/wp1_013_20261015/LOCAL_SOURCE_COMMIT_RECEIPT.json',
    'receipts/wp1_013_20261015/SOURCE_COMMIT_READBACK.json',
    'receipts/wp1_013_20261015/start_native_01/START_RECEIPT.json',
    'receipts/wp1_013_20261015/start_verification_01/START_VALIDATION.json',
    'receipts/wp1_013_20261015/collection_failed_01/COLLECTION_RECEIPT.json',
    'receipts/wp1_013_20261015/independent_01/ENGINEERING_RESTART_REVIEW.json',
    'receipts/wp1_013_20261015/independent_01/check_isolated_runtime_fix.py',
    'receipts/wp1_013_20261015/independent_01/native_import_fix01/PRELAUNCH_RUNTIME_FIX_REVIEW.json',
    'receipts/wp1_013_20261015/native_import_fix01/RUNTIME_REPAIR_REGISTRATION.json',
    'receipts/wp1_013_20261015/native_import_fix01/ROOT_CAUSE.json',
    'receipts/wp1_013_20261015/native_import_fix01/NATIVE_TRACEBACK.txt',
    'receipts/wp1_013_20261015/native_import_fix01/FINAL.patch',
    'receipts/wp1_013_20261015/native_import_fix01/READY.json',
    'receipts/wp1_013_20261015/native_import_fix01/check_start_guard.py',
    'receipts/wp1_013_20261015/native_import_fix01/check_isolated_package.py',
    'receipts/wp1_013_20261015/native_import_fix01/controller_cpu_01/ENGINEERING_RECEIPT.json',
    'receipts/wp1_013_20261015/native_import_fix01/isolated_cpu_01/ENGINEERING_RECEIPT.json',
    'receipts/wp1_013_20261015/native_import_fix01/record_attempt.py',
    'receipts/wp1_013_20261015/native_import_fix01/record_followup.py',
    'receipts/wp1_013_20261015/native_import_fix01/backup_attempt.py',
    'receipts/wp1_013_20261015/native_import_fix01/commit_repair.py',
    'receipts/wp1_013_20261015/native_import_fix01/runtime_package_02/FREEZE_RECEIPT.json',
    'receipts/wp1_013_20261015/native_import_fix01/EXECUTION_ADOPT.json',
    'receipts/wp1_013_20261015/native_import_fix01/D_SOURCE_BACKUP_RECEIPT.json']
assert len(files) == len(set(files))
assert all((root / name).is_file() for name in files)
assert all((root / name).stat().st_size < 2 * 1024 ** 2 for name in files)
assert all(Path(name).suffix.lower() not in ('.npz', '.pt', '.tar', '.gz') for name in files)
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root, text=True).strip()
subprocess.run(['git', 'add', '--'] + files, cwd=root, check=True)
indexed = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root, text=True).splitlines()
assert set(indexed) == set(files)
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=root, check=True)
before = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
subprocess.run(['git', 'commit', '-m', 'Include the frozen CDF dependency and reject failed startup snapshots'], cwd=root, check=True)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
result = dict(schema='n01/wp1-repair-local-source-commit/v1', UTC=datetime.now(timezone.utc).isoformat(), attempt=2,
    before_commit=before, commit=head, explicit_files=files, public_push_performed=False, raw_waveforms_or_weights_committed=False)
out = root / 'receipts/wp1_013_20261015/native_import_fix01/LOCAL_SOURCE_COMMIT_RECEIPT.json'
with out.open('xb') as handle:
    handle.write((json.dumps(result, indent=2) + '\n').encode('utf-8'))
print(json.dumps(dict(commit=head, explicit_files=len(files), public_push_performed=False)))
