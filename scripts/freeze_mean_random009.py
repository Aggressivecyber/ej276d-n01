"""Freeze additive code and byte-bound original TRAIN inputs before dispatch."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OLD_REMOTE = '/home/rylan/projects/ej276d-n01/window008_resume01_20261013'
REMOTE = '/home/rylan/projects/ej276d-n01/mean_random009_20261014'
INPUTS = {
    'dataset': ('data/window008/DATASET008.npz', 'data/window008/DATASET008.npz'),
    'statistics': ('data/window008/FIT_STATISTICS008.json', 'data/window008/FIT_STATISTICS008.json'),
    'data_statistics': ('data/window008/FIT_DATA_STATISTICS.json', 'data/window008/FIT_DATA_STATISTICS.json'),
    'noise005': ('provenance/integration006_reference/noise005.pt', 'provenance/integration006_reference/noise005.pt'),
    'deterministic': ('provenance/window008_fix01_reference/deterministic_final.pt', 'provenance/window008_fix01_reference/deterministic_final.pt'),
    'full': ('runs/window008_resume01_20261013/runs/science/train/full/final.pt', 'runs/science/train/full/final.pt'),
    'base_config': ('window008_config.json', 'window008_config.json'),
    'training_complete': ('runs/window008_resume01_20261013/runs/science/TRAINING_COMPLETE.json', 'runs/science/TRAINING_COMPLETE.json'),
}
OPERATIONAL = ('mean_random009_policy.json', 'docs/MEAN_RANDOM009_PROTOCOL.md',
    'scripts/check_ablation009.py', 'scripts/check_mean_random009.py',
    'scripts/freeze_mean_random009.py', 'scripts/start_mean_random009_remote.py',
    'scripts/supervise_mean_random009.py', 'scripts/snapshot_mean_random009.py')


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
                 'WINDOW008_RESUME01_DIAGNOSTICS_SOURCE_MANIFEST.json'):
        protected[name] = sha(ROOT / name)
        for rel, digest in json.loads((ROOT / name).read_text(encoding='utf8'))['files'].items():
            if sha(ROOT / rel) != digest:
                raise ValueError('Original frozen input/source changed: ' + rel)
    tracked = subprocess.check_output(['git', 'ls-files', '-z', 'src/n01'], cwd=ROOT).decode('utf8').split('\0')
    sources = sorted({rel for rel in tracked if rel.endswith('.py')} |
                     {'src/n01/ablation009.py', 'src/n01/diagnose_mean_random009.py'} | set(OPERATIONAL))
    files = {rel: sha(ROOT / rel) for rel in sources}
    inputs = {key: sha(ROOT / local) for key, (local, _) in INPUTS.items()}
    policy = json.loads((ROOT / 'mean_random009_policy.json').read_text(encoding='utf8'))
    if policy['remote_root'] != REMOTE or policy['optimizer_updates'] != 0:
        raise ValueError('Diagnostic identity/budget changed')
    manifest_path = ROOT / 'MEAN_RANDOM009_SOURCE_MANIFEST.json'
    manifest = dict(schema='n01/mean-random009-source/v1', diagnostic_id=policy['diagnostic_id'],
        frozen_UTC=datetime.now(timezone.utc).isoformat(), before_scientific_numeric_decode=True,
        optimizer_updates=0, policy_sha256=files['mean_random009_policy.json'], files=files,
        bound_inputs=inputs, protected_008_manifests=protected,
        commit_before_source_freeze=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        endpoint_classes=dict(deterministic='n01.model008.N01Response008', full='n01.model008_fix01.N01Response008Fix01'))
    write_new(manifest_path, manifest)
    bindings = dict(schema='n01/mean-random009-bindings/v1', diagnostic_id=policy['diagnostic_id'],
        roots=[REMOTE, OLD_REMOTE], policy_sha256=files['mean_random009_policy.json'],
        source_manifest=dict(path=REMOTE + '/MEAN_RANDOM009_SOURCE_MANIFEST.json', sha256=sha(manifest_path)),
        files={key: dict(path=OLD_REMOTE + '/' + remote, sha256=inputs[key])
               for key, (_, remote) in INPUTS.items()})
    bindings_path = ROOT / 'MEAN_RANDOM009_BINDINGS.json'
    write_new(bindings_path, bindings)
    out = ROOT / 'receipts/mean_random009_20261014/runtime_package_01'
    out.mkdir(parents=True, exist_ok=False)
    package = out / 'mean_random009_source.tar.gz'
    with tarfile.open(package, 'x:gz') as tar:
        for rel in sources + ['MEAN_RANDOM009_SOURCE_MANIFEST.json', 'MEAN_RANDOM009_BINDINGS.json']:
            tar.add(ROOT / rel, arcname=rel, recursive=False)
    with tarfile.open(package, 'r:gz') as tar:
        for rel, digest in files.items():
            if hashlib.sha256(tar.extractfile(rel).read()).hexdigest() != digest:
                raise ValueError('Packaged source changed: ' + rel)
    receipt = dict(status='additive_sources_and_inputs_frozen_before_scientific_forward',
        source_manifest_sha256=sha(manifest_path), bindings_sha256=sha(bindings_path),
        package_sha256=sha(package), package_bytes=package.stat().st_size,
        source_file_count=len(files), protected_manifest_count=len(protected), input_count=len(inputs),
        new_scientific_forwards=0, optimizer_updates=0)
    write_new(ROOT / 'receipts/mean_random009_20261014/FREEZE_RECEIPT.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
