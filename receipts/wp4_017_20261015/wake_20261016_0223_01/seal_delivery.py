"""Commit and back up this completed017 evidence; no remote operations."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

root = Path('E:/GAN/research/N01')
out = Path(__file__).resolve().parent
collected = root / 'runs/wp4_017_20261015/selected_collected_01'
resume = root / 'receipts/wp4_017_20261015/resume_20261016_01'
current = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
result_path = root / 'receipts/wp4_017_20261015/FINAL_RESULT_RECEIPT.json'
report = root / 'reports/wp4_017_20261015/REPORT.md'
result = json.loads(result_path.read_bytes())
program = json.loads((root / 'PROGRAM.json').read_bytes())
settings = tomllib.loads(Path('C:/Users/10629/.codex/automations/n01/automation.toml').read_text(encoding='utf-8'))
assert settings['status'] == 'PAUSED'
for path in (report, out / 'saved_support_review_01/REVIEW.md'):
    text = path.read_text(encoding='utf-8')
    assert '\ufffd' not in text
    for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
        if not link.startswith(('https://', 'http://')):
            assert (path.parent / link).resolve().is_file(), link

files = [root / n for n in ('MASTER_EXECUTION_PLAN.json', 'PROGRAM.json', 'jobs.json', 'README.md')]
files += [result_path, report]
for directory in (out, collected):
    files += [p for p in directory.rglob('*') if p.is_file()
              and p.suffix in ('.py', '.json', '.md', '.txt', '.log', '.toml')
              and p.stat().st_size < 8 * 1024**2]
files = list(dict.fromkeys(files))
names = [p.relative_to(root).as_posix() for p in files]
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root).strip()
subprocess.run(['git', 'add', '-f', '--', *names], cwd=root, check=True)
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=root, check=True)
commit_output = subprocess.check_output(['git', 'commit', '-m',
    'Close bounded WP4 stage1 with verified support-pending evidence'], cwd=root).decode()
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
for path in files:
    blob = subprocess.check_output(['git', 'cat-file', 'blob',
        commit + ':' + path.relative_to(root).as_posix()], cwd=root)
    assert blob == path.read_bytes(), path
unchanged = subprocess.check_output(['git', 'diff', '--name-only', result['source_commit'], commit,
    '--', 'scripts/run_wp4_017.py', 'scripts/wp4_017_controller.py', 'src/n01/model017.py',
    'src/n01/stats017.py', 'src/n01/saved_array_kernels017.py'], cwd=root)
assert not unchanged.strip()

heads = sorted(collected.rglob('*.pt'))
copy_files = list(dict.fromkeys(files + heads + [root / 'wp4_017_config.json',
    root / 'WP4_017_SOURCE_MANIFEST.json', resume / 'runtime_package_01/FREEZE_RECEIPT.json',
    resume / 'runtime_package_01/wp4_017_source.tar.gz', current]))
payload_bytes = sum(p.stat().st_size for p in copy_files)
assert shutil.disk_usage('D:/').free > payload_bytes + 64 * 1024**2
dest = Path('D:/GAN_backups/N01/wp4_017_20261015/postrun_20261016_01')
dest.mkdir(parents=True)
inventory = {}
for path in copy_files:
    name = path.relative_to(root).as_posix() if path != current else 'workspace_entry/CURRENT_SIM2REAL_PROGRAM.json'
    target = dest / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    row = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert target.stat().st_size == row['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest() == row['sha256']
    inventory[name] = row
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest_path = dest / 'MANIFEST.json'
with manifest_path.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(dict(UTC=stamp, source_commit=result['source_commit'], postrun_commit=commit,
                   files=inventory), stream, ensure_ascii=False, indent=2)
    stream.write('\n')
final = dict(schema='n01/wp4-017-postrun-delivery/v1', UTC=stamp,
    status='bounded_stage1_support_pending_reviewed_and_archived', source_commit=result['source_commit'],
    postrun_commit=commit, result_receipt=result_path.relative_to(root).as_posix(),
    result_receipt_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
    actual017_G_AD=4000, cumulative_training_updates=program['new_training_updates'],
    active_scientific_jobs=0, stage2_entered=False, stage2_allowed=False,
    producer_pass_cells=2616, independently_verified_support_pending_cells=72,
    bounded_saved_support_review_complete=True, full_primary_metrics_independently_recomputed=False,
    owned5_runtime_sources_unchanged=True, postrun_evidence_raw_Git_blobs_verified=len(files),
    collected_new_head_checkpoints_preserved=len(heads),
    large_NPZ_arrays_retained_remote=True, large_archive_and_operation_log_retained_E=True,
    D_evidence_backup=dict(destination=str(dest), payload_files=len(inventory),
        payload_bytes=payload_bytes, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        all_copy_SHA256_verified=True),
    automation_status='PAUSED', no_future_training_wake=True, automatic_017_retry=False,
    scientific_status='response_match_not_established', scientific_acceptance=False,
    all_six_work_packages_complete=False, public_push=False, commit_output=commit_output)
final_path = out / 'FINAL_POSTRUN_DELIVERY_RECEIPT.json'
with final_path.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(final, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
shutil.copy2(final_path, dest / final_path.name)
receipt_sha = hashlib.sha256(final_path.read_bytes()).hexdigest()
assert hashlib.sha256((dest / final_path.name).read_bytes()).hexdigest() == receipt_sha
with (dest / 'SHA256SUMS.txt').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(''.join(v['sha256'] + '  ' + n + '\n' for n, v in sorted(inventory.items())))
    stream.write(receipt_sha + '  ' + final_path.name + '\n')
with (dest / 'FINAL_DELIVERY_MANIFEST.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        final_receipt_sha256=receipt_sha,
        SHA256SUMS_sha256=hashlib.sha256((dest / 'SHA256SUMS.txt').read_bytes()).hexdigest()), stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(status=final['status'], postrun_commit=commit,
    training_total=program['new_training_updates'], D_payload_files=len(inventory),
    D_payload_bytes=payload_bytes, receipt_sha256=receipt_sha)))
