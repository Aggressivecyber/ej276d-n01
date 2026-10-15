"""Commit only015 closeout artifacts; preserve unrelated work and016 sources."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent


def call(args):
    return subprocess.check_output(['git']+args, cwd=ROOT)


def main():
    assert not call(['diff','--cached','--name-only']).strip(), 'Preserve any existing staged work'
    manifest=json.loads((ROOT/'reports/wp3_015_20261015/report_manifest.json').read_bytes())
    paths=['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']
    paths += ['reports/wp3_015_20261015/'+n for n in manifest['outputs']]
    paths += ['reports/wp3_015_20261015/'+n for n in ('report_manifest.json','VALIDATION.json')]
    base='receipts/wp3_015_20261015/'
    paths += [base+n for n in ('FINAL_RESULT_RECEIPT.json','D_CLOSEOUT_BACKUP_RECEIPT.json',
        'readback_wake_status_01/REMOTE_STATE.json','readback_collection_01/COLLECTION_RECEIPT.json',
        'readback_collection_01/REMOTE_ARCHIVE.json','readback_execution_01/WAKE_COMPLETION_SUMMARY.json',
        'readback_execution_01/WAKE_COMPLETION_SUMMARY.md','readback_registration_01/FINAL_STARTUP_DELIVERY_RECEIPT.json')]
    for folder in ('postrun_independent_02','postrun_report_impl_01','postrun_root_02'):
        paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/base/folder).iterdir() if p.is_file() and p.suffix in ('.json','.py','.md')]
    for folder in ('postrun_root_02/html_qa_01',):
        paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/base/folder).iterdir() if p.is_file() and p.suffix in ('.png','.html','.json')]
    paths=list(dict.fromkeys(paths))
    assert all((ROOT/p).is_file() for p in paths)
    assert all(not p.endswith(('.pt','.npz','.tar','.gz','.log')) for p in paths)
    subprocess.run(['git','add','--']+paths,cwd=ROOT,check=True,capture_output=True)
    staged=call(['diff','--cached','--name-only','-z']).decode().strip('\0').split('\0')
    assert set(staged) <= set(paths)
    subprocess.run(['git','commit','-m','Close WP3 015 with bounded independent audit and rendered report'],cwd=ROOT,check=True,capture_output=True)
    commit=call(['rev-parse','HEAD']).decode().strip()
    for path in paths:
        blob, local = call(['cat-file','blob',commit+':'+path]), (ROOT/path).read_bytes()
        if path == 'README.md':
            assert blob.replace(b'\r\n', b'\n') == local.replace(b'\r\n', b'\n'), path
        else:
            assert blob == local, path
    receipt=dict(status='local015_closeout_commit_exact_blobs_verified',commit=commit,
        changed_paths=len(staged),verified_artifact_paths=len(paths),
        committed_artifact_paths_sha256=hashlib.sha256('\n'.join(sorted(paths)).encode()).hexdigest(),
        README_Git_newline_normalization_verified=True,all_other_artifacts_exact_raw_blobs=True,
        no_original_raw_or_checkpoints=True,public_push=False,source016_committed_by_this_action=False)
    destination=HERE/'CLOSEOUT_COMMIT_RECEIPT.json'
    with destination.open('x',encoding='utf-8') as f:
        json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
