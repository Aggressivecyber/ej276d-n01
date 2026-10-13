"""Bounded stdlib-only metadata and synthetic tar check; never calls full()."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
import os
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'scripts/archive_window008_resume01.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
started = time.monotonic()
preservation = json.loads((HERE / 'PRESERVATION.json').read_text(encoding='utf8'))
old = (HERE / 'before/archive_window008_resume01.py').read_bytes()
assert hashlib.sha256(old).hexdigest() == preservation['previous_helper_sha256']
assert SOURCE.read_bytes() == old.replace(
    b's = child.stat(follow_symlinks=False); reason = excluded(relative)',
    b's = path.lstat(); reason = excluded(relative)', 1)
ast.parse(SOURCE.read_text(encoding='utf8'))
spec = importlib.util.spec_from_file_location('bounded_archive_helper', SOURCE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

native = []
for relative in ('.git/COMMIT_EDITMSG', '.git/HEAD', '.git/index',
                 'scripts/archive_window008_resume01.py',
                 'receipts/window008_resume01_20261013/REPORT_VISUAL_QA.json'):
    path = ROOT / relative
    with os.scandir(path.parent) as iterator:
        child = next(item for item in iterator if item.name == path.name)
        entry = helper.fingerprint(child.stat(follow_symlinks=False))
    direct = helper.fingerprint(path.lstat())
    with path.open('rb') as handle:
        opened = helper.fingerprint(os.fstat(handle.fileno()))
    assert direct == opened
    assert direct['device'] != 0 and direct['inode'] != 0
    native.append(dict(path=relative, DirEntry_no_follow=entry,
                       Path_lstat=direct, handle_fstat=opened,
                       Path_lstat_equals_handle_fstat=True))

fixture = HERE / 'synthetic_workspace'
fixture.mkdir()
(fixture / 'empty').mkdir()
(fixture / '\u4eba\u5de5').mkdir()
(fixture / 'binary.bin').write_bytes(bytes(range(256)) * 16)
(fixture / '\u4eba\u5de5' / '\u8bf4\u660e.txt').write_text('synthetic only: \u4eba\u5de5\u6d4b\u8bd5\n', encoding='utf8')
(fixture / 'zero.dat').write_bytes(b'')
(fixture / '__pycache__').mkdir()
(fixture / '__pycache__' / 'omitted.pyc').write_bytes(b'transient synthetic')
(fixture / '.git').mkdir()
(fixture / '.git' / 'index.lock').write_bytes(b'transient synthetic lock')
(fixture / '.git' / 'COMMIT_EDITMSG').write_bytes(b'synthetic commit metadata\n')
link_check = {'native_symlink_created': False}
outside = HERE / 'synthetic_outside_root.txt'
outside.write_bytes(b'synthetic link target; no traversal into this file')
try:
    os.symlink(outside, fixture / 'explicit_link')
except OSError as error:
    link_check.update(status='not_exercised_native_privilege_or_platform',
                      error_type=type(error).__name__, error=str(error))
else:
    link_check.update(native_symlink_created=True,
                      status='native_link_target_only_preserved_without_traversal')

entries, exclusions = helper.scan(fixture)
assert {row['path'] for row in exclusions} == {'__pycache__', '.git/index.lock'}
for row in entries.values():
    assert row['fingerprint'] == helper.fingerprint(Path(row['source']).lstat())
    if row['kind'] == 'file':
        helper.hash_row(row)
        assert row['sha256'] == sha(Path(row['source']))
if link_check['native_symlink_created']:
    assert entries['explicit_link']['kind'] == 'symlink_or_junction'
    assert entries['explicit_link']['link_target'] == os.readlink(fixture / 'explicit_link')
    assert len([x for x in entries if x.startswith('explicit_link')]) == 1

guard_checks = []
for key in ('device', 'inode'):
    bad = dict(entries['binary.bin'])
    bad['fingerprint'] = dict(bad['fingerprint'])
    bad['fingerprint'][key] += 1
    try:
        helper.hash_row(bad)
    except ValueError as error:
        assert 'File changed before hash' in str(error)
        guard_checks.append(dict(injected_mismatch=key, rejected=True, message=str(error)))
    else:
        raise AssertionError(key + ' guard did not reject mismatch')

tar = HERE / 'synthetic_only.tar'
metadata = {'META/artificial.json': b'{"artificial_only":true}\n'}
expected = helper.write_tar(tar, entries, metadata)
verification = helper.verify_tar(tar, expected)
assert verification['members_verified'] == len(entries) + 1
assert verification['archive_sha256'] == sha(tar)
helper.stable_tree(entries, fixture)
try:
    helper.write_tar(tar, entries, metadata)
except ValueError as error:
    assert str(error) == 'Archive exists'
else:
    raise AssertionError('Existing artificial archive could be overwritten')
assert verification['archive_sha256'] == sha(tar)

failed = Path(preservation['delivery01_path'])
after = {p.relative_to(failed).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
         for p in sorted(failed.rglob('*')) if p.is_file()}
assert after == preservation['delivery01_before_inventory']
assert sha(HERE / 'before/archive_window008_resume01.py') == preservation['previous_helper_sha256']
frozen_review = json.loads((ROOT / 'receipts/window008_resume01_20261013/eval_independent_final01/REVIEW.json').read_text(encoding='utf8'))
frozen_hashes = frozen_review['reviewed_source_sha256']
assert len(frozen_hashes) == 25
assert all(sha(ROOT / relative) == expected_hash for relative, expected_hash in frozen_hashes.items())
receipt = dict(schema='n01/window008-resume01-archive-helper-repair/v1', status='passed',
               UTC=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
               pid=os.getpid(), reported_parent_pid=os.getppid(), standard_library_only=True,
               source_sha256=sha(SOURCE), previous_source_sha256=preservation['previous_helper_sha256'],
               exact_one_line_metadata_source_change=True, no_follow_Path_lstat=True,
               native_metadata_files=len(native), native_metadata=native,
               retained_device_inode_negative_checks=guard_checks,
               synthetic_symlink=link_check, synthetic_exclusions=exclusions,
               tiny_tar_reopen=verification, existing_archive_overwrite_rejected=True,
               delivery01_inventory_unchanged=after, previous_source_backup_unchanged=True,
               frozen_24_plus_numeric_core_sha256_unchanged=True,
               frozen_reference_sha256=sha(ROOT / 'receipts/window008_resume01_20261013/eval_independent_final01/REVIEW.json'),
               check_script_sha256=sha(Path(__file__)),
               real_full_archive_executed=False, real_workspace_scan_executed=False,
               data_decode_model_GPU_executed=False, git_commands_executed=False, D_backup_executed=False)
with (HERE / 'REPAIR_NATIVE.json').open('x', encoding='utf8') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(status='passed', source_sha256=sha(SOURCE),
                      native_metadata_files=len(native), synthetic_tar_bytes=tar.stat().st_size,
                      members_verified=verification['members_verified'],
                      receipt=str(HERE / 'REPAIR_NATIVE.json'),
                      receipt_sha256=sha(HERE / 'REPAIR_NATIVE.json'))))
