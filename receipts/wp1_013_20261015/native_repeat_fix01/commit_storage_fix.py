"""Save only the declared numerical-storage repair and small evidence in Git."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path('E:/GAN/research/N01')
files = ['.gitattributes', 'README.md', 'PROGRAM.json', 'jobs.json', 'MASTER_EXECUTION_PLAN.json',
    'wp1_013_config.json', 'WP1_SOURCE_MANIFEST.json', 'scripts/run_wp1_013.py', 'scripts/wp1_ops.py', 'scripts/wp1_controller.py',
    'receipts/wp1_013_20261015/prepare_repeat_storage_fix.py',
    'receipts/wp1_013_20261015/native_import_fix01/LOCAL_SOURCE_COMMIT_RECEIPT.json',
    'receipts/wp1_013_20261015/native_import_fix01/SOURCE_COMMIT_READBACK.json',
    'receipts/wp1_013_20261015/native_import_fix01/start_native_02/START_RECEIPT.json',
    'receipts/wp1_013_20261015/native_import_fix01/collection_failed_02/COLLECTION_RECEIPT.json',
    'receipts/wp1_013_20261015/native_import_fix01/SECOND_NATIVE_FAILURE.json',
    'receipts/wp1_013_20261015/native_import_fix01/record_second_failure.py',
    'receipts/wp1_013_20261015/native_repeat_readback01/read_saved_variation.py',
    'receipts/wp1_013_20261015/native_repeat_readback01/saved_01/SAVED_REPEAT_READBACK.json',
    'receipts/wp1_013_20261015/native_repeat_readback01/DIAGNOSIS.md',
    'receipts/wp1_013_20261015/native_repeat_readback01/FINAL.patch',
    'receipts/wp1_013_20261015/native_repeat_readback01/check_actual_storage.py',
    'receipts/wp1_013_20261015/native_repeat_readback01/storage_cpu_02/ENGINEERING_RECEIPT.json',
    'receipts/wp1_013_20261015/native_repeat_readback01/READY.json',
    'receipts/wp1_013_20261015/independent_01/review_numerical_storage_repair.py',
    'receipts/wp1_013_20261015/independent_01/native_repeat_fix01/NUMERICAL_STORAGE_REPAIR_REVIEW.json',
    'receipts/wp1_013_20261015/independent_01/native_repeat_fix01/PRELAUNCH_REPEAT_FIX_REVIEW.json',
    'receipts/wp1_013_20261015/independent_01/native_repeat_fix01/OUTPUT_CAP_AMENDMENT_REVIEW.json',
    'receipts/wp1_013_20261015/native_repeat_fix01/NUMERICAL_STORAGE_REPAIR_REGISTRATION.json',
    'receipts/wp1_013_20261015/native_repeat_fix01/STORAGE_CAP_AMENDMENT.json',
    'receipts/wp1_013_20261015/native_repeat_fix01/amend_storage_cap.py',
    'receipts/wp1_013_20261015/native_repeat_fix01/output_cap_amend01/check_limits.py',
    'receipts/wp1_013_20261015/native_repeat_fix01/output_cap_amend01/stdlib_01/ENGINEERING_RECEIPT.json',
    'receipts/wp1_013_20261015/native_repeat_fix01/reuse_lifecycle_helpers.py',
    'receipts/wp1_013_20261015/native_repeat_fix01/reuse_followup_helper.py',
    'receipts/wp1_013_20261015/native_repeat_fix01/record_attempt.py',
    'receipts/wp1_013_20261015/native_repeat_fix01/record_followup.py',
    'receipts/wp1_013_20261015/native_repeat_fix01/backup_attempt.py',
    'receipts/wp1_013_20261015/native_repeat_fix01/commit_storage_fix.py',
    'receipts/wp1_013_20261015/native_repeat_fix01/runtime_package_03/FREEZE_RECEIPT.json',
    'receipts/wp1_013_20261015/native_repeat_fix01/EXECUTION_ADOPT.json',
    'receipts/wp1_013_20261015/native_repeat_fix01/D_SOURCE_BACKUP_RECEIPT.json']
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
subprocess.run(['git', 'commit', '-m', 'Preserve actual A repeats under the registered numerical replay limit'], cwd=root, check=True)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
result = dict(schema='n01/wp1-storage-repair-local-commit/v1', UTC=datetime.now(timezone.utc).isoformat(), attempt=3,
    before_commit=before, commit=head, explicit_files=files, public_push_performed=False, raw_waveforms_or_weights_committed=False)
out = root / 'receipts/wp1_013_20261015/native_repeat_fix01/LOCAL_SOURCE_COMMIT_RECEIPT.json'
with out.open('xb') as handle:
    handle.write((json.dumps(result, indent=2) + '\n').encode('utf-8'))
print(json.dumps(dict(commit=head, explicit_files=len(files), public_push_performed=False)))
