from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import tarfile

root = Path('E:/GAN/research/N01')
here = Path(__file__).parent
run = here / 'comb_components_01'
figure = root / 'deliverables/global_memory_repair_20261021/comb_components_01'
out = here / 'github_07'
out.mkdir(exist_ok=False)
assert not (run / 'COMPONENT_SCALARS_ANALYSIS.npz').exists()
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root)
old_jobs = json.loads(subprocess.check_output(['git', 'show', 'HEAD:jobs.json'], cwd=root))['jobs']
assert len(old_jobs) == 162
assert json.loads((root / 'jobs.json').read_bytes())['jobs'][:162] == old_jobs
with tarfile.open(run / 'runtime.tar') as archive:
    for name in ('scripts/diagnose_joint01_comb_components.py', 'configs/comb_components_20261021.json'):
        assert archive.extractfile(name).read() == (root / name).read_bytes(), name
files = ['PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json', 'JOINT_RESPONSE_01_PLAN.md',
    'configs/comb_components_20261021.json', 'scripts/diagnose_joint01_comb_components.py',
    'scripts/analyze_comb_components01.py']
files += [(here / name).relative_to(root).as_posix() for name in [
    'PREPARE_COMB_COMPONENTS.py', 'COLLECT_COMB_COMPONENTS.py',
    'CLOSE_COMB_COMPONENTS.py', 'PUBLISH_COMB_COMPONENTS.py']]
files += [(run / name).relative_to(root).as_posix() for name in [
    'REGISTRATION_RECEIPT.json', 'DISPATCH_RECEIPT.json', 'BINDINGS.json', 'RESULT.json',
    'exit_receipt.json', 'CLOSURE_RECEIPT.json', 'ANALYSIS_STORAGE_RECEIPT.json']]
files += [p.relative_to(root).as_posix() for p in sorted(figure.iterdir()) if p.suffix in ('.png', '.svg', '.json', '.md')]
records = {}
for name in files:
    path = root / name
    assert path.suffix in ('.py', '.json', '.md', '.png', '.svg')
    if path.suffix == '.svg':
        path.write_bytes(b'\n'.join(line.rstrip() for line in path.read_bytes().splitlines()) + b'\n')
    raw = path.read_bytes()
    records[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
(out / 'PUBLICATION_MANIFEST.json').write_bytes((json.dumps(records, indent=2) + '\n').encode())
subprocess.run(['git', 'add', '--', *files], cwd=root, check=True, capture_output=True)
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=root, check=True, capture_output=True)
for name in files:
    assert subprocess.check_output(['git', 'show', ':' + name], cwd=root) == (root / name).read_bytes(), name
r = subprocess.run(['git', 'commit', '-m', 'Locate PSD comb in conditioned mean residual with frozen component controls'],
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
    first162_jobs_unchanged=True, native_runtime_scientific_sources_match=True,
    optimizer_updates=0, active_scientific_jobs=0, no_pending_wait_or_N01_timer=True,
    diagnostic_success=True, scientific_acceptance=False,
    private_arrays_and_checkpoints_not_published=True, local_analysis_arrays_removed=True,
    NAS_file_IO_checks=False, NAS_file_hash_verification=False)
(out / 'DELIVERY_RECEIPT.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
print(json.dumps(receipt))
