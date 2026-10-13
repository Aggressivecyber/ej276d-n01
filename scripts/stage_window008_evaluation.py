"""Package/install only the additive frozen evaluation payload; no job dispatch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from window008_evaluation_common import (
    ROOT, RUN_ID, EVAL_MANIFEST, TRAIN_MANIFEST, POLICY, SOURCE_FILES,
    allow_output, bound, evaluation_binding, require, sha, training_binding, write_new,
)


def package(output, root=ROOT):
    root = Path(root).resolve()
    cfg, training, policy, manifest = evaluation_binding(root)
    output = allow_output(output)
    output.mkdir(parents=True, exist_ok=False)
    target = output / 'window008_evaluation.tar.gz'
    with tarfile.open(target, 'x:gz') as archive:
        for name in sorted(set(manifest['files']) | {EVAL_MANIFEST}):
            path = bound(root, name)
            require(path == root / name and path.is_file() and not (root / name).is_symlink(),
                    'Payload must contain unlinked regular files: ' + name)
            archive.add(path, arcname=name, recursive=False)
    result = dict(schema='n01/window008-evaluation-bundle/v1', UTC=datetime.now(timezone.utc).isoformat(),
        archive=str(target), archive_sha256=sha(target),
        evaluation_source_manifest_sha256=sha(root / EVAL_MANIFEST),
        training_source_manifest_sha256=sha(root / TRAIN_MANIFEST),
        files=len(manifest['files']) + 1, protected_training_files=len(training['files']),
        optimizer_updates=0, evaluation_dispatched=False)
    write_new(output / 'BUNDLE.json', result)
    return result


def archive_payload(archive_path):
    values = {}
    with tarfile.open(archive_path, 'r:gz') as archive:
        total = 0
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            require(member.isfile() and not member.issym() and not member.islnk(), 'Only regular payload files are allowed')
            require(not name.is_absolute() and '..' not in name.parts and '\\' not in member.name
                    and ':' not in member.name and str(name) == member.name and member.name not in values,
                    'Unsafe/duplicate archive path: ' + member.name)
            total += member.size
            require(total <= 32 * 1024 ** 2, 'Evaluation source payload exceeds 32 MiB')
            stream = archive.extractfile(member)
            require(stream is not None, 'Missing archive body')
            values[member.name] = stream.read()
    require(EVAL_MANIFEST in values, 'Missing evaluation manifest')
    manifest = json.loads(values[EVAL_MANIFEST])
    require(set(values) == set(manifest['files']) | {EVAL_MANIFEST}, 'Archive membership differs from the frozen payload')
    for name, digest in manifest['files'].items():
        require(hashlib.sha256(values[name]).hexdigest() == digest, 'Archive payload hash differs: ' + name)
    return manifest, values


def install(archive_path, root=ROOT):
    root = Path(root).resolve()
    cfg, training = training_binding(root)
    require(os.name == 'posix' and str(root) == cfg['remote_root'], 'Install only in the registered Linux workspace')
    import fcntl
    (root / 'runs').mkdir(exist_ok=True)
    with (root / 'runs/.evaluation_dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return install_locked(archive_path, root, cfg, training)


def install_locked(archive_path, root, cfg, training):
    manifest, values = archive_payload(archive_path)
    require(manifest['schema'] == 'n01/window008-evaluation-source/v1' and manifest['run_id'] == RUN_ID,
            'Wrong evaluation source version/run')
    require(manifest['training_source_manifest_sha256'] == sha(root / TRAIN_MANIFEST)
            and manifest['training_config_sha256'] == sha(root / 'window008_config.json'), 'Training freeze differs')
    require(manifest['before_evaluation_numeric_read'] is True and manifest['optimizer_updates'] == 0,
            'Wrong evaluation source scope')
    protected = set(training['files']) | {TRAIN_MANIFEST}
    require(not set(values) & protected and set(SOURCE_FILES) <= set(manifest['files']), 'Incomplete or training-modifying payload')
    policy = json.loads(values[POLICY])
    require(manifest['policy_sha256'] == hashlib.sha256(values[POLICY]).hexdigest(), 'Policy hash differs')
    require(policy['job_root'] == 'runs/evaluation_jobs' and policy['jobs'] == ['primary', 'replay01'], 'Wrong job layout')
    for kind in policy['jobs']:
        require(not (root / policy['job_root'] / kind).exists(), 'Preserve already-attempted evaluation: ' + kind)
    existing_manifest = root / EVAL_MANIFEST
    require(not existing_manifest.exists() or existing_manifest.read_bytes() == values[EVAL_MANIFEST],
            'Refuse replacement of an existing evaluation freeze')
    modules = {item.relative_to(root).as_posix() for item in (root / 'src/n01').glob('*.py')}
    added_modules = {name for name in values if name.startswith('src/n01/') and name.endswith('.py')}
    require(modules | added_modules == set(manifest['module_inventory']), 'Remote module inventory differs')
    targets = {name: bound(root, name) for name in values}
    for name, path in targets.items():
        require(path == root / name and not (root / name).is_symlink(),
                'Refuse a linked or redirected payload target: ' + name)
        require(not path.exists() or path.is_file(), 'Existing payload target is not a file: ' + name)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    receipt = root / 'receipts' / RUN_ID / ('evaluation_stage_' + stamp)
    receipt.mkdir(parents=True, exist_ok=False)
    write_new(receipt / 'INSTALL_ATTEMPT.json', dict(UTC=stamp, archive_sha256=sha(Path(archive_path)),
        evaluation_source_manifest_sha256=hashlib.sha256(values[EVAL_MANIFEST]).hexdigest(),
        training_source_manifest_sha256=sha(root / TRAIN_MANIFEST), automatic_retry_allowed=False))
    changed, identical = [], []
    try:
        for name in sorted(values, key=lambda item: (item == EVAL_MANIFEST, item)):
            path, content = targets[name], values[name]
            if path.exists() and path.read_bytes() == content:
                identical.append(name)
                continue
            if path.exists():
                backup = receipt / 'before' / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.evaluation-stage-' + stamp)
            with temporary.open('xb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            changed.append(name)
        evaluation_binding(root)
        result = dict(status='evaluation_payload_installed', archive_sha256=sha(Path(archive_path)),
            evaluation_source_manifest_sha256=sha(root / EVAL_MANIFEST), changed=changed, already_identical=identical,
            training_files_unchanged=len(training['files']), scientific_job_dispatched=False, DEV_numeric_read=False)
        write_new(receipt / 'INSTALL_COMPLETE.json', result)
        return result
    except BaseException as error:
        write_new(receipt / 'INSTALL_FAILED.json', dict(error=repr(error), installed_paths=changed,
            preserved_before=str(receipt / 'before'), automatic_retry_allowed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--package', metavar='NEW_DIRECTORY')
    action.add_argument('--install', metavar='ARCHIVE')
    parser.add_argument('--root', default=str(ROOT))
    args = parser.parse_args()
    result = package(args.package, args.root) if args.package else install(args.install, args.root)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
