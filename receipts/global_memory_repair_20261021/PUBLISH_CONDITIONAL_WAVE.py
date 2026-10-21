from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import tarfile

root = Path('E:/GAN/research/N01')
here = Path(__file__).parent
run = here / 'conditional_wave_01'
figure = root / 'deliverables/global_memory_repair_20261021/conditional_wave_01'
out = here / 'github_06'
out.mkdir(exist_ok=False)
stamp = datetime.now(timezone.utc).isoformat()
qa = dict(UTC=stamp, status='VISUALLY_REVIEWED', figures=[
    'GLOBAL_MEMORY_PSD_LINEAR.png', 'GLOBAL_MEMORY_PSD_FULL_RANGE.png',
    'CONDITIONAL_PSD_64_COMPARISON.png', 'CONDITIONAL_WAVE_RISK.png'],
    axes_legends_and_population_labels_readable=True,
    inherited_and_new_endpoint_curves_overlap=True, comb_structure_remains=True,
    full_range_all_rows_visible=True, new_updates=0, scientific_acceptance=False,
    interpretation='Conditional memory was implemented and the probe ran. No update passed; the endpoint is the inherited candidate, not a new repair.')
(figure / 'VISUAL_QA.json').write_bytes((json.dumps(qa, indent=2) + '\n').encode())
storage = json.loads((run / 'ANALYSIS_STORAGE_RECEIPT.json').read_bytes())
storage.update(visual_QA_pending=False, visual_QA=str((figure / 'VISUAL_QA.json').relative_to(root)))
(run / 'ANALYSIS_STORAGE_RECEIPT.json').write_bytes((json.dumps(storage, indent=2) + '\n').encode())
assert not (run / 'ENDPOINT_SCALARS_ANALYSIS.npz').exists()
with tarfile.open(run / 'runtime.tar') as archive:
    for name in ('src/n01/global_memory01.py', 'scripts/run_global_memory01.py',
                 'scripts/check_global_memory01.py', 'configs/global_memory_conditional_wave_20261021.json'):
        assert archive.extractfile(name).read() == (root / name).read_bytes(), name
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((root / name).read_bytes())
    assert d['active_scientific_jobs'] == 0
    d['global_memory_repair'].update(latest_probe_job='GLOBAL_MEMORY_CONDITIONAL_WAVE_PROBE_20261021',
        latest_probe_full_Q_figure=str((figure / 'GLOBAL_MEMORY_PSD_LINEAR.png').relative_to(root)),
        conditional_wave_risk_figure=str((figure / 'CONDITIONAL_WAVE_RISK.png').relative_to(root)),
        conditional_wave_visual_QA=str((figure / 'VISUAL_QA.json').relative_to(root)))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
files = ['PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json', 'JOINT_RESPONSE_01_PLAN.md',
    'src/n01/global_memory01.py', 'scripts/run_global_memory01.py', 'scripts/check_global_memory01.py',
    'scripts/plot_global_memory01.py', 'scripts/plot_global_memory_comparison01.py', 'scripts/plot_conditional_wave01.py',
    'configs/global_memory_conditional_wave_20261021.json']
files += [(here / name).relative_to(root).as_posix() for name in [
    'PREPARE_CONDITIONAL_WAVE.py', 'RECORD_CONDITIONAL_START.py', 'COLLECT_CONDITIONAL_WAVE.py',
    'ANALYZE_CONDITIONAL_WAVE.py', 'CLOSE_CONDITIONAL_WAVE.py', 'PLOT_CONDITIONAL_WAVE.py',
    'PUBLISH_CONDITIONAL_WAVE.py', 'CONDITIONAL_WAVE_ACTIVE_CELL_NEXT_DESIGN.json']]
files += [(run / name).relative_to(root).as_posix() for name in [
    'SYNTHETIC_CHECKS.json', 'REGISTRATION_RECEIPT.json', 'DISPATCH_RECEIPT.json', 'STARTUP_SCIENTIFIC_RECEIPT.json',
    'CONDITIONAL_WAVE_MEMORY.json', 'BINDINGS.json', 'INITIAL.json', 'TRAINING_RESULT.json',
    'TRAINING_STEPS.jsonl', 'RESULT.json', 'exit_receipt.json', 'CLOSURE_RECEIPT.json',
    'DIRECTION_ANALYSIS.json', 'ANALYSIS_STORAGE_RECEIPT.json']]
files += [p.relative_to(root).as_posix() for p in sorted(figure.iterdir()) if p.suffix in ('.png', '.svg', '.json')]
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root)
old_jobs = json.loads(subprocess.check_output(['git', 'show', 'HEAD:jobs.json'], cwd=root))['jobs']
assert len(old_jobs) == 161
assert json.loads((root / 'jobs.json').read_bytes())['jobs'][:161] == old_jobs
records = {}
for name in files:
    path = root / name
    assert path.suffix in ('.py', '.json', '.jsonl', '.md', '.png', '.svg')
    if path.suffix != '.png':
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    raw = path.read_bytes()
    records[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
        git_blob=hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest())
(out / 'PUBLICATION_MANIFEST.json').write_bytes((json.dumps(records, indent=2) + '\n').encode())
subprocess.run(['git', 'add', '--', *files], cwd=root, check=True, capture_output=True)
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=root, check=True, capture_output=True)
for name in files:
    assert subprocess.check_output(['git', 'show', ':' + name], cwd=root) == (root / name).read_bytes()
r = subprocess.run(['git', 'commit', '-m', 'Add conditional waveform memory and record bounded probe rejection'],
    cwd=root, check=True, capture_output=True)
(out / 'COMMIT_STDOUT.txt').write_bytes(r.stdout)
r = subprocess.run(['git', 'push', 'origin', 'HEAD:main'], cwd=root, check=True, capture_output=True)
(out / 'PUSH_STDOUT.txt').write_bytes(r.stdout)
(out / 'PUSH_STDERR.txt').write_bytes(r.stderr)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
assert subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=root).decode().split()[0] == head
assert not subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'], cwd=root)
receipt = dict(status='COMMITTED_PUSHED_REMOTE_MAIN_VERIFIED', UTC=datetime.now(timezone.utc).isoformat(),
    commit=head, files=len(files), public_bytes=sum(v['bytes'] for v in records.values()),
    first161_jobs_unchanged=True, native_runtime_scientific_sources_match=True,
    new_optimizer_updates=0, active_scientific_jobs=0, no_pending_wait_or_N01_timer=True,
    scientific_acceptance=False, private_arrays_and_checkpoints_not_published=True,
    local_analysis_arrays_removed=True, NAS_file_IO_checks=False, NAS_file_hash_verification=False)
(out / 'DELIVERY_RECEIPT.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
print(json.dumps(receipt))
