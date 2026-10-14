"""Byte-bind the isolated 010 trial before any scientific input decode."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OLD_REMOTE = '/home/rylan/projects/ej276d-n01/window008_resume01_20261013'
REMOTE = '/home/rylan/projects/ej276d-n01/shape010_20261014'
INPUTS = {
    'dataset': ('data/window008/DATASET008.npz', 'data/window008/DATASET008.npz'),
    'statistics': ('data/window008/FIT_STATISTICS008.json', 'data/window008/FIT_STATISTICS008.json'),
    'data_statistics': ('data/window008/FIT_DATA_STATISTICS.json', 'data/window008/FIT_DATA_STATISTICS.json'),
    'noise005': ('provenance/integration006_reference/noise005.pt', 'provenance/integration006_reference/noise005.pt'),
    'deterministic': ('provenance/window008_fix01_reference/deterministic_final.pt', 'provenance/window008_fix01_reference/deterministic_final.pt'),
    'base_config': ('window008_config.json', 'window008_config.json'),
    'training_complete': ('runs/window008_resume01_20261013/runs/science/TRAINING_COMPLETE.json', 'runs/science/TRAINING_COMPLETE.json'),
}
EXTRA = ('src/n01/model010.py', 'src/n01/train_shape010.py', 'shape010_policy.json',
    'docs/SHAPE010_PROTOCOL.md', 'scripts/check_model010.py', 'scripts/check_shape010.py',
    'scripts/check_mean_random009.py',
    'scripts/register_shape010.py', 'scripts/freeze_shape010.py',
    'scripts/start_shape010_remote.py', 'scripts/supervise_shape010.py', 'scripts/snapshot_shape010.py',
    'scripts/deploy_shape010.py')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    protected = {}
    for name in ('WINDOW008_RESUME01_SOURCE_MANIFEST.json', 'WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json',
                 'WINDOW008_RESUME01_DIAGNOSTICS_SOURCE_MANIFEST.json', 'MEAN_RANDOM009_SOURCE_MANIFEST.json'):
        protected[name] = sha(ROOT / name)
        for rel, digest in json.loads((ROOT / name).read_text(encoding='utf8'))['files'].items():
            if sha(ROOT / rel) != digest:
                raise ValueError('Original frozen bytes changed: ' + rel)
    tracked = subprocess.check_output(['git','ls-files','-z','src/n01'], cwd=ROOT).decode('utf8').split('\0')
    sources = sorted({v for v in tracked if v.endswith('.py')} | set(EXTRA))
    files = {v: sha(ROOT / v) for v in sources}
    inputs = {key: sha(ROOT / local) for key, (local, _) in INPUTS.items()}
    policy = json.loads((ROOT / 'shape010_policy.json').read_text(encoding='utf8'))
    if policy['remote_root'] != REMOTE or policy['G_updates'] != 2000 or policy['D_updates'] != 0:
        raise ValueError('010 identity/budget changed')
    manifest_path = ROOT / 'SHAPE010_SOURCE_MANIFEST.json'
    manifest = dict(schema='n01/shape010-source/v1', run_id=policy['run_id'],
        frozen_UTC=datetime.now(timezone.utc).isoformat(), before_scientific_numeric_decode=True,
        G_update_budget=2000, D_update_budget=0, policy_sha256=files['shape010_policy.json'],
        files=files, bound_inputs=inputs, protected_manifests=protected,
        commit_before_source_freeze=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        baseline_class='n01.model008.N01Response008', candidate_class=policy['shape']['class_name'])
    write_new(manifest_path, manifest)
    bindings = dict(schema='n01/shape010-bindings/v1', run_id=policy['run_id'], roots=[REMOTE,OLD_REMOTE],
        policy_sha256=files['shape010_policy.json'],
        source_manifest=dict(path=REMOTE + '/SHAPE010_SOURCE_MANIFEST.json',sha256=sha(manifest_path)),
        files={key:dict(path=OLD_REMOTE + '/' + remote,sha256=inputs[key]) for key,(_,remote) in INPUTS.items()})
    binding_path = ROOT / 'SHAPE010_BINDINGS.json'
    write_new(binding_path, bindings)
    out = ROOT / 'receipts/shape010_20261014/runtime_package_01'
    out.mkdir(parents=True,exist_ok=False)
    package = out / 'shape010_source.tar.gz'
    with tarfile.open(package,'x:gz') as archive:
        for rel in sources + ['SHAPE010_SOURCE_MANIFEST.json','SHAPE010_BINDINGS.json']:
            archive.add(ROOT/rel,arcname=rel,recursive=False)
    with tarfile.open(package,'r:gz') as archive:
        for rel,digest in files.items():
            if hashlib.sha256(archive.extractfile(rel).read()).hexdigest() != digest:
                raise ValueError('Packaged bytes changed: ' + rel)
    backup = Path('D:/GAN_backups/N01/shape010_20261014/source_frozen_01')
    if shutil.disk_usage('D:/').free < package.stat().st_size + 256*1024**2:
        raise ValueError('D source backup would cross the preserved free-space margin')
    backup.mkdir(parents=True,exist_ok=False)
    shutil.copy2(package,backup/package.name)
    if sha(backup/package.name) != sha(package):
        raise ValueError('Source backup mismatch')
    (backup/'SHA256SUMS.txt').write_text(sha(package)+'  '+package.name+'\n',encoding='utf8')
    receipt = dict(status='additive_shape010_sources_and_inputs_frozen_before_science',
        UTC=datetime.now(timezone.utc).isoformat(), source_manifest_sha256=sha(manifest_path),
        bindings_sha256=sha(binding_path),policy_sha256=files['shape010_policy.json'],
        package_sha256=sha(package),package_bytes=package.stat().st_size,source_file_count=len(files),
        input_count=len(inputs),protected_manifest_count=len(protected),
        D_source_backup=str(backup/package.name),D_backup_SHA256_verified=True,
        new_scientific_forwards=0,new_G_updates=0,registered_G_budget=2000)
    write_new(ROOT/'receipts/shape010_20261014/FREEZE_RECEIPT.json',receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
