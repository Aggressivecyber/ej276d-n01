"""Verify small source/startup backups for the packaging repair on D:."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

root = Path('E:/GAN/research/N01')
base = root / 'receipts/wp1_013_20261015/native_import_fix01'
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=('source', 'startup'), required=True)
parser.add_argument('--freeze', required=True, type=Path)
parser.add_argument('--verification', type=Path)
args = parser.parse_args()
freeze = args.freeze.resolve(strict=True)
assert freeze.is_relative_to(base)
target = Path('D:/GAN_backups/N01/wp1_013_20261015/native_import_fix01') / (args.stage + '_02')
assert target.resolve().is_relative_to(Path('D:/GAN_backups/N01/wp1_013_20261015').resolve())
assert target.drive.upper() == 'D:' and not target.exists()
frozen = json.loads((freeze / 'FREEZE_RECEIPT.json').read_text('utf-8'))
sources = {name: root / name for name in ('wp1_013_config.json', 'WP1_SOURCE_MANIFEST.json',
    'MASTER_EXECUTION_PLAN.json', 'PROGRAM.json', 'jobs.json', 'README.md', 'OBSERVATION_CONTRACT.json',
    'DATA_ROLES.json', 'docs/ACCEPTANCE_CONTRACT.md', 'USER_MASTER_PLAN_20261015.txt')}
sources['CURRENT_SIM2REAL_PROGRAM.json'] = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
sources['RUNTIME_REPAIR_REGISTRATION.json'] = base / 'RUNTIME_REPAIR_REGISTRATION.json'
if args.stage == 'source':
    sources.update({'wp1_source.tar.gz': freeze / 'wp1_source.tar.gz', 'FREEZE_RECEIPT.json': freeze / 'FREEZE_RECEIPT.json',
        'EXECUTION_ADOPT.json': base / 'EXECUTION_ADOPT.json',
        'PRELAUNCH_RUNTIME_FIX_REVIEW.json': root / 'receipts/wp1_013_20261015/independent_01/native_import_fix01/PRELAUNCH_RUNTIME_FIX_REVIEW.json'})
    assert hashlib.sha256(sources['wp1_source.tar.gz'].read_bytes()).hexdigest() == frozen['source_package_sha256']
else:
    assert args.verification and args.verification.resolve().is_relative_to(base)
    sources.update({name: args.verification / name for name in ('REMOTE_STATE.json', 'START_VALIDATION.json')})
    sources.update({name: base / name for name in ('EXECUTION_STARTED.json', 'SOURCE_COMMIT_READBACK.json', 'FOLLOWUP_RECEIPT.json')})
assert all(path.is_file() for path in sources.values())
assert sum(path.stat().st_size for path in sources.values()) < 32 * 1024 ** 2
target.mkdir(parents=True, exist_ok=False)
records = {}
for name, src in sources.items():
    dest = target / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    with src.open('rb') as handle:
        expected = hashlib.file_digest(handle, 'sha256').hexdigest()
    with dest.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert actual == expected
    records[name] = dict(sha256=actual, bytes=dest.stat().st_size)
(target / 'SHA256SUMS.txt').write_bytes(''.join(row['sha256'] + '  ' + name + '\n' for name, row in sorted(records.items())).encode('utf-8'))
result = dict(schema='n01/wp1-repair-small-backup/v1', attempt=2, stage=args.stage, root=str(target), files=records,
    byte_verified=True, private_input_or_weight_payload_copied=False, bytes=sum(row['bytes'] for row in records.values()))
with (base / ('D_' + args.stage.upper() + '_BACKUP_RECEIPT.json')).open('xb') as handle:
    handle.write((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(json.dumps(dict(stage=args.stage, root=str(target), files=len(records), byte_verified=True, bytes=result['bytes'])))
