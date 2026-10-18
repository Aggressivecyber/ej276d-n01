from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,re,subprocess,collections
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert subprocess.run(['git','branch','--show-current'],cwd=r,capture_output=True,text=True,check=True).stdout.strip()=='main'
config=read(r/'configs/response071_20261018.json');selected=set();local_payloads=[]
allowed={'.py','.md','.json','.txt','.png','.svg','.pdf','.html','.diff','.patch'}
for name,digest in config['source_pins'].items():
    path=r/name;assert sha(path)==digest,name
    if path.suffix.lower() in allowed:selected.add(name)
    else:local_payloads.append(dict(path=name,bytes=path.stat().st_size,sha256=digest))
selected.update(['.gitattributes','AGENTS.md','PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','docs/RESPONSE_GOAL_20261018.md'])
for folder in ('configs','docs','scripts','reports'):
    for path in (r/folder).rglob('*'):
        if not path.is_file() or path.suffix.lower() not in allowed:continue
        name=path.relative_to(r).as_posix();m=re.search(r'(?:response|RESPONSE)(\d{3})',name)
        if m and 39<=int(m.group(1))<=71:selected.add(name)
receipts={'ARTIFACT_VERIFICATION.json','CLOSURE_RECEIPT.json','TERMINAL_REVIEW.json','ROOT_DESIGN_AND_REVIEW.json','COLLECTION_VERIFICATION.json','STARTUP_ASSESSMENT.json','DISPATCH_RECEIPT.json','DEPLOYMENT_VERIFICATION.json','PACKAGE_RECEIPT.json','SOURCE_MANIFEST.json','VISUAL_QA.json','FOLLOWUP_RECEIPT.json','ACTIVE_HANDOFF.json','LOCAL_ENGINEERING_TEST.json','LOCAL039_TEST.json','REPAIRED_CAP_REGISTRATION.json','DEPLOYMENT_WAIT_RECEIPT.json','NETWORK_STATUS.json'}
for folder in (r/'receipts').glob('response0*'):
    m=re.match(r'response(\d{3})_',folder.name)
    if not m or not 39<=int(m.group(1))<=71:continue
    for path in folder.rglob('*'):
        if path.is_file() and path.name in receipts and not any(z in path.parts for z in ('before_state','before_closure','before_followup','startup_incomplete_01')):selected.add(path.relative_to(r).as_posix())
# Include concise terminal evidence and code used by the currently frozen job, not duplicate raw startup payloads or source archives.
for n in (70,71):
    for path in (r/f'receipts/response{n:03d}_20261018').rglob('*'):
        if not path.is_file():continue
        if path.suffix=='.py' or path.name in {'STARTED.json','RESULT.json','COMPLETE.json','exit_receipt.json','RECEIPT.json'}:selected.add(path.relative_to(r).as_posix())
selected.add((w/'SELECT_PUBLICATION.py').relative_to(r).as_posix())
assert all(not name.endswith(('.npz','.npy','.pt','.pth','.tar.gz','.zip','.log','.toml')) for name in selected)
assert not any(Path(name).name in ('STARTUP_RAW.json','STARTUP_STDOUT.txt','REMOTE_STATUS_STDOUT.txt','FOLLOWUP_PROMPT_EFFECTIVE.txt') for name in selected)
patterns=[re.compile(rb'gh[pousr]_[A-Za-z0-9]{25,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{25,}'),re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{30,}'),re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(rb'https?://[^\s/:]+:[^\s/@]+@')]
suspect=[];parsed=0
for name in sorted(selected):
    path=r/name;assert path.is_file(),name;data=path.read_bytes();assert len(data)<20*1024**2,(name,len(data))
    if path.suffix.lower() not in {'.png','.pdf'}:
        if any(p.search(data) for p in patterns):suspect.append(name)
    if name.startswith('scripts/') and path.suffix=='.py':ast.parse(data.decode('utf-8-sig'));parsed+=1
assert not suspect,{'credential_pattern_candidate_paths':suspect}
state=read(r/'PROGRAM.json');record=dict(UTC=datetime.now(timezone.utc).isoformat(),authorization='User explicitly requested GitHub submission in this chat.',repository='Aggressivecyber/ej276d-n01',branch='main',scope='Response039 through071 code, fixed contracts, reports, verification receipts and current ledgers. Historical negative outcomes retained. Raw scientific arrays, checkpoints, compressed runtime/source archives, duplicate base64 snapshots and unrelated legacy untracked files stay local.',current_status=state['status'],scientific_goal_achieved=False,selected_files=len(selected),selected_bytes=sum((r/name).stat().st_size for name in selected),current_script_AST_checks=parsed,credential_pattern_scan_passed=True,local_only_pinned_payloads=local_payloads,files={name:dict(bytes=(r/name).stat().st_size,sha256=sha(r/name)) for name in sorted(selected)})
(w/'PUBLICATION_MANIFEST.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
selected.add((w/'PUBLICATION_MANIFEST.json').relative_to(r).as_posix())
(w/'GIT_PATHS.nul').write_bytes(b'\0'.join(n.encode('utf-8') for n in sorted(selected))+b'\0')
print(json.dumps({k:v for k,v in record.items() if k not in ('files','local_only_pinned_payloads')},ensure_ascii=False));print('local_only_pinned_payload_count',len(local_payloads))
