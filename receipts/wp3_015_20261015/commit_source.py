"""Commit only the reviewed015 source and explicit registration evidence."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
call=lambda args:subprocess.check_output(['git']+args,cwd=root)
assert not call(['diff','--cached','--name-only']).strip()
assert read(out/'independent_01/PRELAUNCH_REVIEW.json')['passed'] is True
master_sha=sha(root/'MASTER_EXECUTION_PLAN.json')
for path in (root/'PROGRAM.json',root.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'):
    value=read(path)
    if 'master_plan_sha256' in value:
        value['master_plan_sha256']=master_sha
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
manifest=read(root/'WP3_SOURCE_MANIFEST.json')
paths=list(manifest['files'])+['wp3_015_config.json','WP3_SOURCE_MANIFEST.json','.gitattributes',
    'MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']
paths += [p.relative_to(root).as_posix() for p in out.glob('*.py')]
paths += [p.relative_to(root).as_posix() for p in out.glob('*.json')]
for directory in ('implementation_01','independent_01'):
    paths += [p.relative_to(root).as_posix() for p in (out/directory).glob('*') if p.is_file() and p.suffix in ('.py','.json','.md')]
paths += [p.relative_to(root).as_posix() for p in (out/'engineering_01').glob('*.py')]
paths += [p.relative_to(root).as_posix() for p in (out/'engineering_01').glob('*/ENGINEERING_RECEIPT.json')]
paths += [(out/'runtime_package_01/FREEZE_RECEIPT.json').relative_to(root).as_posix()]
paths=list(dict.fromkeys(paths));assert all((root/p).is_file() for p in paths)
assert all(not p.endswith(('.pt','.npz','.tar','.gz','.log')) for p in paths)
subprocess.run(['git','add','--']+paths,cwd=root,check=True)
staged=call(['diff','--cached','--name-only','-z']).decode().strip('\0').split('\0')
assert set(staged)<=set(paths)
subprocess.run(['git','commit','-m','Add bounded WP3 mean-adapter paired trials with a pre-model evidence gate'],cwd=root,check=True)
commit=call(['rev-parse','HEAD']).decode().strip()
print(json.dumps(dict(status='reviewed015_source_committed',commit=commit,changed_paths=len(staged),
    private_numeric_payloads_included=False,public_push=False)))
