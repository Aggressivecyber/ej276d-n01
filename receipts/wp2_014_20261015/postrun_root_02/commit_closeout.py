"""Explicit-path local Git closeout; no private numeric payloads or publication."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[3]
here=Path(__file__).resolve().parent
call=lambda args:subprocess.check_output(['git']+args,cwd=root)
assert call(['diff','--cached','--name-only']).strip()==b'', 'Preserve an existing staged change'
manifest=json.loads((root/'reports/wp2_014_20261015/report_manifest.json').read_text(encoding='utf-8'))
paths=['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']
paths += ['reports/wp2_014_20261015/'+n for n in manifest['outputs']]+['reports/wp2_014_20261015/report_manifest.json']
base='receipts/wp2_014_20261015/'
paths += [base+n for n in (
    'FINAL_RESULT_RECEIPT.json','D_CLOSEOUT_BACKUP_RECEIPT.json',
    'readback_wake_status_01/REMOTE_STATE.json',
    'readback_collection_01/COLLECTION_RECEIPT.json','readback_collection_01/REMOTE_ARCHIVE.json',
    'readback_execution_01/WAKE_COMPLETION_SUMMARY.json','readback_execution_01/WAKE_COMPLETION_SUMMARY.md',
    'postrun_independent_02/CLOSEOUT_REVIEW.json','postrun_independent_02/closeout_review.py',
    'postrun_report_impl_01/build_report.py','postrun_report_impl_01/validate_report.py',
    'postrun_report_impl_01/REPORT_BUILD_RECEIPT.json','postrun_report_impl_01/REPORT_QA.json')]
paths += [p.relative_to(root).as_posix() for p in here.glob('*.py')]
paths += [p.relative_to(root).as_posix() for p in here.glob('*.json')]
paths += [base+'postrun_root_02/selection_status_fix01/FIX_RECEIPT.json',base+'postrun_root_02/html_qa_01/top.png']
paths += [base+'postrun_root_02/html_qa_02/'+n for n in ('wave.html','conditional.html','readback.html','wave.png','conditional.png','readback.png','PREVIEW_BINDINGS.json')]
paths=list(dict.fromkeys(paths))
assert all((root/p).is_file() for p in paths), [p for p in paths if not (root/p).is_file()]
assert all(not p.endswith(('.pt','.npz','.tar','.gz','.log')) for p in paths)
subprocess.run(['git','add','--']+paths,cwd=root,check=True)
staged=call(['diff','--cached','--name-only','-z']).decode().strip('\0').split('\0')
assert set(staged)<=set(paths)
subprocess.run(['git','commit','-m','Close WP2 with bounded independent saved-array audit and report'],cwd=root,check=True)
commit=call(['rev-parse','HEAD']).decode().strip()
for name in [p for p in paths if p.startswith(('reports/','receipts/'))]:
    assert call(['cat-file','blob',commit+':'+name])==(root/name).read_bytes(),name
receipt=dict(status='local_closeout_commit_and_exact_artifact_blobs_verified',commit=commit,
    committed_changed_paths=len(staged),verified_artifact_paths=len([p for p in paths if p.startswith(('reports/','receipts/'))]),
    no_raw_numeric_payload=True,public_push=False,source015_not_committed_by_this_action=True)
(here/'CLOSEOUT_COMMIT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
