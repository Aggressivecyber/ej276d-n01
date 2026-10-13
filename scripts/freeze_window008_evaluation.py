"""Freeze a separate, additive evaluation version without changing training bytes."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from window008_evaluation_common import (
    ROOT, RUN_ID, TRAIN_MANIFEST, EVAL_MANIFEST, POLICY, SOURCE_FILES,
    evaluation_binding, policy_binding, require, sha, write_new,
)


def freeze(root=ROOT):
    root = Path(root).resolve()
    cfg, training, policy = policy_binding(root)
    require(not (root / EVAL_MANIFEST).exists(), 'Preserve the existing evaluation source freeze')
    for kind in policy['jobs']:
        require(not (root / policy['job_root'] / kind).exists(), 'Evaluation already attempted: ' + kind)
    modules = {item.relative_to(root).as_posix() for item in (root / 'src/n01').glob('*.py')}
    protected = set(training['files']) | {TRAIN_MANIFEST}
    paths = set(SOURCE_FILES) | (modules - protected)
    require(not paths & protected, 'The evaluation payload may not modify a frozen training file')
    files = {name: sha(root / name) for name in sorted(paths)}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    payload = dict(schema='n01/window008-evaluation-source/v1', run_id=RUN_ID,
        UTC=datetime.now(timezone.utc).isoformat(), source_commit=commit,
        training_source_manifest_sha256=sha(root / TRAIN_MANIFEST),
        training_config_sha256=sha(root / 'window008_config.json'), policy_sha256=sha(root / POLICY),
        before_evaluation_numeric_read=True, optimizer_updates=0,
        module_inventory=sorted(modules), files=files,
        scientific_status='response_match_not_established',
        policy_scope='Independent frozen evaluation and same-stream exact replay only')
    receipt = root / 'receipts' / RUN_ID / 'EVALUATION_FREEZE_ATTEMPT.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write_new(receipt, dict(UTC=payload['UTC'], source_files=len(files),
        training_source_manifest_sha256=payload['training_source_manifest_sha256'], automatic_retry_allowed=False))
    write_new(root / EVAL_MANIFEST, payload)
    evaluation_binding(root)
    result = dict(status='evaluation_sources_frozen', manifest=str(root / EVAL_MANIFEST),
        sha256=sha(root / EVAL_MANIFEST), source_files=len(files), training_files_unchanged=len(training['files']),
        evaluation_executed=False, DEV_numeric_read=False, optimizer_updates=0)
    write_new(root / 'receipts' / RUN_ID / 'EVALUATION_FREEZE_RECEIPT.json', result)
    return result


if __name__ == '__main__':
    import json
    print(json.dumps(freeze()))
