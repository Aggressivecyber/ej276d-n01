"""Archive the completed bounded run on E and a small source/evidence backup on D."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ID = 'response_007_20261013'
RUN = ROOT / 'runs' / ID
REC = ROOT / 'receipts' / ID / 'deadline01'
REPORT = ROOT / 'reports' / ID
DEST = Path('E:/GAN/archives/N01/response_007_20261013_deadline01')
BACKUP = Path('D:/GAN_backups/research/N01/response_007_20261013_deadline01')


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def digest(stream):
    h = hashlib.sha256()
    for b in iter(lambda: stream.read(1024 ** 2), b''):
        h.update(b)
    return h.hexdigest()


def sha(p):
    with p.open('rb') as f:
        return digest(f)


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')


def items(folder, prefix):
    return [(p, prefix + '/' + p.relative_to(folder).as_posix()) for p in sorted(folder.rglob('*')) if p.is_file()]


def archive(path, members):
    assert not path.exists()
    inventory = {n: dict(bytes=p.stat().st_size, sha256=sha(p)) for p, n in members}
    assert len(inventory) == len(members)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, n in members:
            z.write(p, n)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for n, expected in inventory.items():
            with z.open(n) as f:
                assert digest(f) == expected['sha256'], n
    return inventory


def main():
    assert not DEST.exists() and not BACKUP.exists(), 'Preserve existing archives'
    final = read(REC / 'FINAL_RECEIPT.json')
    assert final['status'] == 'bounded_stage_comparison_complete_reviewed'
    assert read(REC / 'VISUAL_QA.json')['status'] == 'passed'
    assert sha(ROOT / final['report_PDF']) == final['verification']['final_PDF_sha256']
    assert shutil.disk_usage('E:/').free > 2 * sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()) + 1024 ** 3
    assert shutil.disk_usage('D:/').free >= 320 * 1024 ** 2
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    DEST.mkdir(parents=True)
    source = DEST / 'n01-response007-deadline01-source-and-evidence.zip'
    subprocess.run(['git', 'archive', '--format=zip', '--output=' + str(source), commit], cwd=ROOT, check=True)
    source_inventory = {}
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        for member in z.infolist():
            if member.is_dir():
                continue
            with z.open(member) as f:
                member_sha = digest(f)
            assert (ROOT / member.filename).is_file()
            assert sha(ROOT / member.filename) == member_sha, 'Committed source differs from working tree: ' + member.filename
            source_inventory[member.filename] = dict(bytes=member.file_size, sha256=member_sha)
    bundle = DEST / 'n01-response007-deadline01-source.bundle'
    subprocess.run(['git', 'bundle', 'create', str(bundle), 'HEAD'], cwd=ROOT, check=True)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == commit
    subprocess.run(['git', 'bundle', 'verify', str(bundle)], cwd=ROOT, check=True, capture_output=True)
    pointer_copy = DEST / 'CURRENT_SIM2REAL_PROGRAM.snapshot.json'
    shutil.copy2(ROOT.parents[1] / 'CURRENT_SIM2REAL_PROGRAM.json', pointer_copy)
    members = items(RUN, 'runs/' + ID) + items(ROOT / 'receipts' / ID, 'receipts/' + ID) + items(REPORT, 'reports/' + ID)
    members += [(source, 'source/' + source.name), (bundle, 'source/' + bundle.name), (pointer_copy, pointer_copy.name)]
    full = DEST / 'n01-response007-deadline01-private-run.zip'
    full_inventory = archive(full, members)
    inventory_path = DEST / 'ARCHIVE_INVENTORY.json'
    write(inventory_path, dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(), commit=commit,
        scope='All local response007 run artifacts through bounded-stage closeout, committed source/history, reports and receipts',
        original_remote_raw_archives_copied=False,
        input_reproduction_requires_access_to_hash_bound_original_SOURCE_REAL_data=True,
        private_archive_members=full_inventory, source_archive_members=source_inventory))
    copied = [source, bundle, inventory_path, pointer_copy]
    small_total = sum(p.stat().st_size for p in copied)
    assert small_total <= 64 * 1024 ** 2, 'Do not consume scarce D space with full scientific payloads'
    assert shutil.disk_usage('D:/').free - small_total >= 256 * 1024 ** 2
    BACKUP.mkdir(parents=True)
    backup_entries = []
    for p in copied:
        target = BACKUP / p.name
        shutil.copy2(p, target)
        assert sha(p) == sha(target)
        backup_entries.append(dict(path=str(target), bytes=target.stat().st_size, sha256=sha(target)))
    with zipfile.ZipFile(BACKUP / source.name) as z:
        assert z.testzip() is None
        for n, expected in source_inventory.items():
            with z.open(n) as f:
                assert digest(f) == expected['sha256'], n
    subprocess.run(['git', 'bundle', 'verify', str(BACKUP / bundle.name)], cwd=ROOT, check=True, capture_output=True)
    receipt = dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='E_full_run_and_D_small_source_evidence_reopened_SHA256_verified', source_commit=commit,
        E_private_archive=dict(path=str(full), bytes=full.stat().st_size, sha256=sha(full), members=len(full_inventory)),
        source_archive_members=len(source_inventory), D_small_backup=backup_entries,
        D_backup_bytes=small_total, D_free_bytes_after=shutil.disk_usage('D:/').free,
        minimum_D_free_reserve_bytes=256 * 1024 ** 2, raw_remote_inputs_duplicated=False,
        scientific_weights_and_logs_preserved_on=['E', 'original remote execution root'],
        FINAL_RECEIPT_sha256=sha(REC / 'FINAL_RECEIPT.json'), PDF_sha256=final['verification']['final_PDF_sha256'],
        public_release_created=False, shared_upload_created=False, implementation_sha256=sha(Path(__file__)))
    write(REC / 'BACKUP_RECEIPT.json', receipt)
    write(DEST / 'BACKUP_RECEIPT.json', receipt)
    write(BACKUP / 'BACKUP_RECEIPT.json', receipt)
    sums = ''.join(sha(p) + '  ' + p.name + '\n' for p in [source, bundle, full, inventory_path, pointer_copy, DEST / 'BACKUP_RECEIPT.json'])
    (DEST / 'SHA256SUMS.txt').write_text(sums, encoding='ascii')
    (BACKUP / 'SHA256SUMS.txt').write_text(''.join(sha(p) + '  ' + p.name + '\n' for p in [BACKUP / p.name for p in copied] + [BACKUP / 'BACKUP_RECEIPT.json']), encoding='ascii')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
