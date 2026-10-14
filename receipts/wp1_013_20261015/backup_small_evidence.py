"""Copy frozen source or verified startup evidence to fresh D-drive backups."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

root = Path('E:/GAN/research/N01')
base = Path('D:/GAN_backups/N01/wp1_013_20261015')
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=('source', 'startup'), required=True)
parser.add_argument('--freeze', required=True, type=Path)
parser.add_argument('--verification-dir', type=Path)
args = parser.parse_args()
freeze = args.freeze.resolve(strict=True)
receipt = json.loads((freeze / 'FREEZE_RECEIPT.json').read_text('utf-8'))
target = base / (args.stage + '_01')
assert target.resolve().is_relative_to(base.resolve()) and target.drive.upper() == 'D:' and not target.exists()
sources = {}
for relative in ('wp1_013_config.json', 'WP1_SOURCE_MANIFEST.json', 'MASTER_EXECUTION_PLAN.json', 'PROGRAM.json',
    'jobs.json', 'README.md', '.gitattributes', 'USER_MASTER_PLAN_20261015.txt', 'OBSERVATION_CONTRACT.json',
    'DATA_ROLES.json', 'docs/ACCEPTANCE_CONTRACT.md'):
    sources[relative] = root / relative
sources['CURRENT_SIM2REAL_PROGRAM.json'] = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
if args.stage == 'source':
    archive = freeze / 'wp1_source.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['source_package_sha256']
    sources['wp1_source.tar.gz'] = archive
    sources['FREEZE_RECEIPT.json'] = freeze / 'FREEZE_RECEIPT.json'
    for relative in ('EXECUTION_READY.json', 'FROZEN_CONTROL_ADOPTION.json', 'EXTERNAL_ACQUISITION_METADATA_AUDIT.json',
        'independent_01/PRELAUNCH_REVIEW.json'):
        sources[relative] = root / ('receipts/wp1_013_20261015/' + relative)
else:
    assert args.verification_dir and args.verification_dir.resolve().is_relative_to(root)
    for name in ('REMOTE_STATE.json', 'START_VALIDATION.json'):
        sources[name] = args.verification_dir / name
    for relative in ('EXECUTION_STARTED.json', 'FOLLOWUP_RECEIPT.json', 'SOURCE_COMMIT_READBACK.json'):
        path = root / ('receipts/wp1_013_20261015/' + relative)
        assert path.is_file()
        sources[relative] = path
assert all(path.is_file() for path in sources.values())
assert sum(path.stat().st_size for path in sources.values()) < 32 * 1024 ** 2
target.mkdir(parents=True, exist_ok=False)
records = {}
for relative, src in sources.items():
    dest = target / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()
    shutil.copyfile(src, dest)
    with src.open('rb') as handle:
        expected = hashlib.file_digest(handle, 'sha256').hexdigest()
    with dest.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert actual == expected
    records[relative] = {'sha256': actual, 'bytes': dest.stat().st_size}
checksums = ''.join(record['sha256'] + '  ' + name + '\n' for name, record in sorted(records.items()))
(target / 'SHA256SUMS.txt').write_bytes(checksums.encode('utf-8'))
result = {'schema': 'n01/wp1-small-backup/v1', 'run_id': 'wp1_013_20261015', 'stage': args.stage,
    'root': str(target), 'files': records, 'copied_bytes': sum(x['bytes'] for x in records.values()),
    'byte_verified': True, 'private_input_or_weight_payload_copied': False, 'scientific_status_changed': False}
out = root / ('receipts/wp1_013_20261015/D_' + args.stage.upper() + '_BACKUP_RECEIPT.json')
with out.open('xb') as handle:
    handle.write((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(json.dumps({'stage': args.stage, 'root': str(target), 'files': len(records),
    'copied_bytes': result['copied_bytes'], 'byte_verified': True}))
