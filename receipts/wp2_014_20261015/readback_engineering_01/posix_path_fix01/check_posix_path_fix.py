"""Targeted stdlib check of the actual local incoming expression and guard."""
import ast
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

base=Path(__file__).resolve().parents[1]
source=base/'readback_control.py'
before=base/'before_posix_path_fix01/readback_control.py'
old=before.read_text(encoding='utf-8');new=source.read_text(encoding='utf-8')
old_line="        expected=str(Path(SIDE).parent/('.readback.incoming_'+payload['token']))"
new_line="        expected=SIDE.rsplit('/',1)[0]+'/.readback.incoming_'+payload['token']"
assert hashlib.sha256(before.read_bytes()).hexdigest()=='c275c74777478d63ea3f31045ff9b5969ee32ec47d1bb97297127167055332f5'
assert old.count(old_line)==1 and new==old.replace(old_line,new_line)
tree=ast.parse(new);action=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='action')
assignment=next(n for n in ast.walk(action) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in n.targets))
guard=next(n for n in ast.walk(action) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and any(isinstance(a,ast.Constant) and a.value=='sidecar incoming path differs' for a in n.value.args))
code=compile(ast.fix_missing_locations(ast.Module(body=[assignment,guard],type_ignores=[])),str(source),'exec')
native=json.loads(json.loads((base.parent/'readback_prepare_01/prepare_NATIVE.json').read_text(encoding='utf-8'))['stdout'])
side='/home/rylan/projects/ej276d-n01/wp2_014_saved_readback_20261015';token=native['incoming'].rsplit('_',1)[1]
def require(ok,message):
    if not ok:raise ValueError(message)
def run(path_type,incoming):
    env=dict(SIDE=side,payload={'token':token},incoming=incoming,Path=path_type,require=require)
    exec(code,env)
    return env['expected']
assert str(PureWindowsPath(side).parent/('.readback.incoming_'+token))!=native['incoming']
assert run(PureWindowsPath,dict(native))==run(PurePosixPath,dict(native))==native['incoming']
for changed in (dict(native,incoming=native['incoming']+'_wrong'),dict(native,archive=native['archive']+'_wrong')):
    try:run(PureWindowsPath,changed)
    except ValueError as error:assert str(error)=='sidecar incoming path differs'
    else:raise AssertionError('changed native incoming/archive accepted')
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Path' and any(isinstance(a,ast.Name) and a.id in ('SIDE','ORIGINAL') for a in n.args) for n in ast.walk(action))
receipt=dict(schema='n01/readback-posix-path-fix-engineering/v1',recorded_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    artificial_only=True,passed=4,native_exit_code=0,checks=['before_SHA_and_exact_one_line_patch','actual_expression_native_reply_Windows_and_POSIX','actual_guard_rejects_changed_incoming_and_archive','other_local_action_remote_paths_stay_literal_or_opaque'],
    controller={'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
    before={'bytes':before.stat().st_size,'sha256':hashlib.sha256(before.read_bytes()).hexdigest()},
    checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    prior_failure_receipt_sha256=hashlib.sha256((base.parent/'readback_execution_01/PREPARE_PATH_FAILURE.json').read_bytes()).hexdigest(),
    source_patch='Only action.expected uses literal POSIX parent/token concatenation.',
    local_action_path_review='No other Path(SIDE)/Path(ORIGINAL) construction remains in action; SCP remote strings use unmodified native incoming/archive. Native path constructors run only on POSIX.',
    SSH_calls=0,raw_SHA_passes=0,array_decodes=0,CP_NN_AD_training=0,worker_or_old95_sources_modified=False)
out=Path(__file__).with_name('ENGINEERING_RECEIPT.json')
with out.open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'passed':4,'controller':receipt['controller'],'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
