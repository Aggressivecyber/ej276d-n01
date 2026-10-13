"""Native metadata-only coverage plus a small artificial tar round trip."""
from pathlib import Path
import datetime
import importlib.util
import json
import os
import time

ROOT = Path('E:/GAN/research/N01')
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('archive_helper', ROOT/'scripts/archive_window008_resume01.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
started = time.monotonic()
entries, exclusions = h.scan(ROOT)
checked = 0
mode_differences = []
for name, row in entries.items():
    if row['kind'] != 'file':
        continue
    with Path(row['source']).open('rb') as stream:
        observed = h.fingerprint(os.fstat(stream.fileno()))
        assert h.handle_matches(stream, row['fingerprint']), (name, row['fingerprint'], observed)
        if observed['mode'] != row['fingerprint']['mode']:
            mode_differences.append(dict(path=name, path_mode=row['fingerprint']['mode'], handle_mode=observed['mode']))
    assert h.fingerprint(Path(row['source']).lstat()) == row['fingerprint'], name
    checked += 1

fixture = OUT/'synthetic_workspace'
fixture.mkdir(exist_ok=False)
(fixture/'example.exe').write_bytes(b'Artificial bytes, never executed.\n')
(fixture/'plain.txt').write_text('Artificial text\n', encoding='utf8')
(fixture/'empty').write_bytes(b'')
rows, excluded = h.scan(fixture)
for row in rows.values():
    if row['kind'] == 'file':
        h.hash_row(row)
negative = []
sample = rows['example.exe']
with Path(sample['source']).open('rb') as stream:
    for field, delta in [('bytes',1),('mtime_ns',1),('device',1),('inode',1),('mode',0o200),('mode',0o100000)]:
        corrupt = dict(sample['fingerprint'])
        corrupt[field] ^= delta
        assert not h.handle_matches(stream, corrupt), (field,delta)
        negative.append(dict(field=field, xor_delta=delta, rejected=True))
tar = OUT/'synthetic_only.tar'
expected = h.write_tar(tar, rows, {'fixture_metadata.json': b'{"artificial":true}\n'})
verification = h.verify_tar(tar, expected)
h.stable_tree(rows, fixture)
result = dict(schema='n01/windows-archive-handle-mode-repair/v1', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='passed', native_workspace_metadata_files=checked, mode_differences=mode_differences,
    path_mode_unchanged_and_exact=True, device_inode_size_mtime_write_bits_exact=True,
    negative_controls=negative, synthetic_tar=verification,
    helper_sha256=h.sha(ROOT/'scripts/archive_window008_resume01.py'),
    previous_helper_sha256=h.sha(OUT/'archive_window008_resume01_before.py'),
    wall_seconds=time.monotonic()-started, scientific_array_decodes=0, model_calls=0, optimizer_updates=0,
    compatibility_rule='Normalize only Windows inferred execute bits for cross-API handle comparison; full pathname mode remains checked exactly.')
h.write_new(OUT/'REPAIR_NATIVE.json', result)
print(json.dumps(dict(status=result['status'], native_workspace_metadata_files=checked,
    inferred_mode_differences=len(mode_differences), synthetic_members=verification['members_verified'],
    helper_sha256=result['helper_sha256'], wall_seconds=result['wall_seconds'])))
