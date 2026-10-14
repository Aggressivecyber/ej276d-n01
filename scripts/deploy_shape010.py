"""Deploy the frozen small 010 source bundle; never install a Python environment.

--self-check is entirely local and artificial. Deployment creates one private
UUID incoming directory, uploads the archive, verifies every regular member and
seven immutable input hashes, and creates the registered new root once. --start
only invokes the separately frozen native start_shape010_remote.py; it is never
automatic after installation. Remote Python code is sent on stdin, not -c.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid


ROOT = Path(__file__).resolve().parents[1]
HOST = 'rylan@100.82.231.102'
PYTHON = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
REMOTE_ROOT = '/home/rylan/projects/ej276d-n01/shape010_20261014'
MANIFEST_NAME = 'SHAPE010_SOURCE_MANIFEST.json'
BINDINGS_NAME = 'SHAPE010_BINDINGS.json'


# These same functions are executed locally for package QA and on the remote
# interpreter for installation. No code from an unverified tar member is run.
REMOTE_SCRIPT = r'''
from datetime import datetime,timezone
import hashlib,json,os,re,shutil,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath

REMOTE_ROOT='/home/rylan/projects/ej276d-n01/shape010_20261014'
OLD_ROOT='/home/rylan/projects/ej276d-n01/window008_resume01_20261013'
PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
MANIFEST_NAME='SHAPE010_SOURCE_MANIFEST.json'
BINDINGS_NAME='SHAPE010_BINDINGS.json'
RUN_ID='shape010_20261014'
INPUTS={
 'dataset':'data/window008/DATASET008.npz',
 'statistics':'data/window008/FIT_STATISTICS008.json',
 'data_statistics':'data/window008/FIT_DATA_STATISTICS.json',
 'noise005':'provenance/integration006_reference/noise005.pt',
 'deterministic':'provenance/window008_fix01_reference/deterministic_final.pt',
 'base_config':'window008_config.json',
 'training_complete':'runs/science/TRAINING_COMPLETE.json',
}
REQUIRED_SOURCES={'shape010_policy.json','src/n01/model010.py','src/n01/train_shape010.py',
 'scripts/deploy_shape010.py','scripts/start_shape010_remote.py','scripts/supervise_shape010.py'}

def require(condition,message):
 if not condition:raise ValueError(message)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()

def valid_sha(value):
 return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def member_name(name):
 require(isinstance(name,str) and bool(name),'Empty/non-string source path')
 require(not name.startswith('/') and '\\' not in name and ':' not in name
         and all(ord(c)>=32 and ord(c)!=127 for c in name),'Absolute/control/Windows tar path')
 parts=name.split('/')
 require(all(part not in ('','.', '..') for part in parts),'Noncanonical tar path')
 require(PurePosixPath(name).as_posix()==name,'Noncanonical tar spelling')
 return name

def validate_documents(manifest,bindings,policy,manifest_sha):
 require(manifest.get('schema')=='n01/shape010-source/v1' and manifest.get('run_id')==RUN_ID
         and manifest.get('before_scientific_numeric_decode') is True,'Wrong010 source identity')
 require(manifest.get('G_update_budget')==2000 and manifest.get('D_update_budget')==0
         and manifest.get('baseline_class')=='n01.model008.N01Response008'
         and manifest.get('candidate_class')=='n01.model010.FrozenDeterministicVoltageShape010','Wrong010 budget/model')
 files=manifest.get('files')
 require(isinstance(files,dict) and REQUIRED_SOURCES<=set(files),'Missing010 source inventory')
 for name,digest in files.items():
  member_name(name);require(valid_sha(digest),'Invalid source SHA256')
 require(MANIFEST_NAME not in files and BINDINGS_NAME not in files,'Recursive source inventory')
 require(bindings.get('schema')=='n01/shape010-bindings/v1' and bindings.get('run_id')==RUN_ID
         and isinstance(bindings.get('roots'),list) and len(bindings['roots'])==2
         and set(bindings['roots'])=={REMOTE_ROOT,OLD_ROOT},'Wrong010 binding roots')
 require(bindings.get('source_manifest')==dict(path=REMOTE_ROOT+'/'+MANIFEST_NAME,sha256=manifest_sha),
         'Source manifest binding differs')
 require(isinstance(bindings.get('files'),dict) and set(bindings['files'])==set(INPUTS)
         and isinstance(manifest.get('bound_inputs'),dict) and set(manifest['bound_inputs'])==set(INPUTS),
         'Exactly seven immutable input bindings required')
 for key,relative in INPUTS.items():
  row=bindings['files'][key]
  require(isinstance(row,dict) and set(row)=={'path','sha256'}
          and row['path']==OLD_ROOT+'/'+relative and valid_sha(row['sha256'])
          and row['sha256']==manifest['bound_inputs'][key],'Immutable input path/hash differs: '+key)
 require(manifest.get('policy_sha256')==files['shape010_policy.json']
         and bindings.get('policy_sha256')==files['shape010_policy.json'],'Policy freeze differs')
 require(policy.get('schema')=='n01/shape010-policy/v1' and policy.get('run_id')==RUN_ID
         and policy.get('remote_root')==REMOTE_ROOT and policy.get('immutable_endpoint_root')==OLD_ROOT
         and policy.get('G_updates')==2000 and policy.get('D_updates')==0
         and policy.get('automatic_restart') is False,'Wrong010 policy identity/budget')
 return files

def validate_package(path,archive_sha,manifest_sha,bindings_sha):
 require(all(valid_sha(v) for v in (archive_sha,manifest_sha,bindings_sha)),'Invalid pinned SHA256')
 require(sha(path)==archive_sha,'Archive SHA256 differs')
 require(Path(path).stat().st_size<=64*1024**2,'Source archive exceeds small-package bound')
 with tarfile.open(path,'r:gz') as archive:
  members=archive.getmembers();names=[];contents={};total=0
  for entry in members:
   name=member_name(entry.name)
   require(name not in names,'Duplicate tar member')
   require(entry.isfile() and not entry.issym() and not entry.islnk(),'Nonregular/link tar member')
   require(0<=entry.size<=8*1024**2,'Source member exceeds bounded size')
   total+=entry.size;require(total<=128*1024**2,'Expanded source inventory exceeds bound')
   stream=archive.extractfile(entry);require(stream is not None,'Missing tar content')
   data=stream.read();require(len(data)==entry.size,'Truncated tar member')
   names.append(name);contents[name]=data
 require(MANIFEST_NAME in contents and BINDINGS_NAME in contents,'Missing manifest/bindings members')
 require(hashlib.sha256(contents[MANIFEST_NAME]).hexdigest()==manifest_sha,'Packaged manifest SHA differs')
 require(hashlib.sha256(contents[BINDINGS_NAME]).hexdigest()==bindings_sha,'Packaged bindings SHA differs')
 manifest=json.loads(contents[MANIFEST_NAME]);bindings=json.loads(contents[BINDINGS_NAME])
 require('shape010_policy.json' in contents,'Missing packaged policy')
 policy=json.loads(contents['shape010_policy.json'])
 files=validate_documents(manifest,bindings,policy,manifest_sha)
 require(set(names)==set(files)|{MANIFEST_NAME,BINDINGS_NAME},'Tar members differ from exact frozen inventory')
 for name,digest in files.items():
  require(hashlib.sha256(contents[name]).hexdigest()==digest,'Source member SHA differs: '+name)
 return dict(manifest=manifest,bindings=bindings,policy=policy,contents=contents,
             archive_sha256=archive_sha,manifest_sha256=manifest_sha,bindings_sha256=bindings_sha)

def unredirected(path):
 path=Path(path).absolute()
 for part in (path,)+tuple(path.parents):
  require(not part.is_symlink(),'Refuse redirected path component: '+str(part))
 require(path.resolve()==path,'Resolved path differs: '+str(path))
 return path

def verify_inputs(bindings):
 result={}
 for key,relative in INPUTS.items():
  expected=OLD_ROOT+'/'+relative
  require(bindings['files'][key]['path']==expected,'Input binding escaped immutable root')
  path=unredirected(Path(expected))
  require(path.is_file(),'Missing immutable input: '+key)
  digest=sha(path);require(digest==bindings['files'][key]['sha256'],'Immutable input SHA differs: '+key)
  result[key]=dict(path=str(path),sha256=digest,bytes=path.stat().st_size)
 return result

def prepare(token):
 require(re.fullmatch('[0-9a-f]{32}',token) is not None,'Invalid incoming token')
 target=Path(REMOTE_ROOT);parent=unredirected(target.parent)
 require(parent.is_dir(),'Registered parent is missing')
 require(not os.path.lexists(target),'Preserve existing010 root')
 incoming=parent/('.shape010_incoming_'+token)
 require(not os.path.lexists(incoming),'Preserve existing incoming directory')
 incoming.mkdir(mode=0o700)
 return dict(status='private_incoming_prepared',incoming=str(incoming),archive=str(incoming/'source.tar.gz'),
             target=str(target),target_created=False)

def install_payload(payload):
 target=Path(REMOTE_ROOT);unredirected(target.parent)
 require(not os.path.lexists(target),'Preserve existing010 root')
 target.mkdir(mode=0o700)
 # No extractall: write only already-hashed regular bytes with exclusive opens.
 for name,data in payload['contents'].items():
  member_name(name);destination=target/name
  destination.parent.mkdir(parents=True,exist_ok=True)
  resolved=unredirected(destination.parent)
  require(resolved==target or target in resolved.parents,'Installation escaped new root')
  flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)
  fd=os.open(destination,flags,0o600)
  with os.fdopen(fd,'wb') as f:f.write(data)
 for name,digest in payload['manifest']['files'].items():
  require(sha(target/name)==digest,'Installed source SHA differs: '+name)
 require(sha(target/MANIFEST_NAME)==payload['manifest_sha256'] and sha(target/BINDINGS_NAME)==payload['bindings_sha256'],
         'Installed freeze bytes differ')
 return target

def runtime_metadata(policy,incoming):
 for name in ('tmp','cache'):(incoming/name).mkdir(exist_ok=False)
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(incoming/'tmp'),XDG_CACHE_HOME=str(incoming/'cache'))
 import torch
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                               '--format=csv,noheader'],text=True).strip()
 markers=('-m n01.','scripts/supervise_window','scripts/supervise_shape010.py',
          'scripts/supervise_mean_random009.py','scripts/run_window','scripts/supervise.py')
 active=[line for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
         if any(marker in line for marker in markers)]
 gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu',
                              '--format=csv,noheader'],text=True).strip()
 available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')))*1024
 disk=shutil.disk_usage(Path(REMOTE_ROOT).parent).free
 require(not apps and not active,'Another GPU/project scientific job is active')
 require(available>=policy['preflight']['minimum_available_RAM_GiB']*1024**3
         and disk>=policy['preflight']['minimum_free_disk_GiB']*1024**3,'Insufficient free RAM/disk')
 require(torch.cuda.is_available(),'Existing runtime has no available CUDA')
 return dict(Python_executable=sys.executable,Python_version=sys.version,Torch_version=str(torch.__version__),
             Torch_CUDA_version=torch.version.cuda,CUDA_available=bool(torch.cuda.is_available()),
             GPU=gpu,GPU_compute_processes=apps,active_project_scientific_jobs=active,
             RAM_available_bytes=available,disk_free_bytes=disk,environment_packages_modified=False)

def start_native(manifest_sha,bindings_sha):
 target=unredirected(Path(REMOTE_ROOT));require(target.is_dir(),'010 source root not installed')
 require(sha(target/MANIFEST_NAME)==manifest_sha and sha(target/BINDINGS_NAME)==bindings_sha,'Remote start freeze differs')
 manifest=json.loads((target/MANIFEST_NAME).read_text());bindings=json.loads((target/BINDINGS_NAME).read_text())
 policy=json.loads((target/'shape010_policy.json').read_text());validate_documents(manifest,bindings,policy,manifest_sha)
 for name,digest in manifest['files'].items():require(sha(unredirected(target/name))==digest,'Changed installed source: '+name)
 verify_inputs(bindings)
 result=subprocess.run([PYTHON,'-B',str(target/'scripts/start_shape010_remote.py')],cwd=target,
                       text=True,capture_output=True,timeout=60)
 require(result.returncode==0,'Native010 start rejected/failed: '+result.stderr+result.stdout)
 return dict(status='native_start_cli_completed_awaiting_first_update_proof',native_start_exit_code=result.returncode,
             native_start_receipt=json.loads(result.stdout),scientific_first_update_verified=False)

def main():
 require(os.name=='posix' and sys.executable==PYTHON,'Use the registered existing Linux Python')
 require(len(sys.argv)==6,'Expected mode/token/archiveSHA/manifestSHA/bindingsSHA')
 mode,token,archive_sha,manifest_sha,bindings_sha=sys.argv[1:]
 require(mode in ('prepare','install','start') and re.fullmatch('[0-9a-f]{32}',token) is not None
         and all(valid_sha(v) for v in (archive_sha,manifest_sha,bindings_sha)),'Invalid safe remote arguments')
 if mode=='prepare':
  print(json.dumps(prepare(token)));return
 if mode=='start':
  print(json.dumps(start_native(manifest_sha,bindings_sha)));return
 incoming=unredirected(Path(REMOTE_ROOT).parent/('.shape010_incoming_'+token))
 require(incoming.is_dir(),'Prepared incoming directory missing')
 archive=unredirected(incoming/'source.tar.gz');require(archive.is_file(),'Incoming archive missing')
 require(not os.path.lexists(REMOTE_ROOT),'Preserve existing010 root before verification')
 payload=validate_package(archive,archive_sha,manifest_sha,bindings_sha)
 before=verify_inputs(payload['bindings'])
 metadata=runtime_metadata(payload['policy'],incoming)
 target=install_payload(payload)
 after=verify_inputs(payload['bindings']);require(before==after,'Immutable inputs changed during source installation')
 receipt=dict(status='new_shape010_source_root_installed',UTC=datetime.now(timezone.utc).isoformat(),
              remote_root=str(target),incoming_archive=str(archive),package_sha256=archive_sha,
              source_manifest_sha256=manifest_sha,bindings_sha256=bindings_sha,
              policy_sha256=payload['manifest']['policy_sha256'],source_file_count=len(payload['manifest']['files']),
              immutable_input_count=len(after),immutable_inputs_verified=after,immutable_inputs_unchanged=True,
              source_overwrites=0,metadata=metadata,scientific_numeric_decode=False,checkpoint_loads=0,
              scientific_model_forwards=0,optimizer_updates=0,automatic_restart=False)
 with (target/'REMOTE_SOURCE_INSTALL_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
 print(json.dumps(receipt))

if __name__=='__main__':
 try:main()
 except Exception as error:
  print(json.dumps(dict(status='failed_or_unknown_preserved',error_type=type(error).__name__,error=str(error),
                       remote_root=REMOTE_ROOT,automatic_retry_allowed=False)))
  sys.exit(1)
'''


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def namespace():
    value = {'__name__': 'shape010_artificial_or_local_package_validation'}
    exec(compile(REMOTE_SCRIPT, 'shape010_verified_stdin_installer', 'exec'), value)
    return value


def fresh_output(path):
    value = Path(path).resolve()
    if value.drive.upper() not in ('E:', 'D:') or value.exists():
        raise ValueError('A fresh E/D output directory is required')
    value.mkdir(parents=True, exist_ok=False)
    return value


def ssh_phase(mode, token, digests):
    command = ['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST,
               PYTHON, '-B', '-', mode, token, digests['package'], digests['manifest'], digests['bindings']]
    process = subprocess.run(command, input=REMOTE_SCRIPT, text=True, encoding='utf8',
                             capture_output=True, timeout=240 if mode == 'install' else 80)
    record = dict(command=command, native_ssh_exit_code=process.returncode,
                  stdout=process.stdout, stderr=process.stderr, remote_mode=mode)
    if process.returncode != 0:
        return record, None
    return record, json.loads(process.stdout)


def local_freeze(package=None):
    manifest_path, binding_path = ROOT/MANIFEST_NAME, ROOT/BINDINGS_NAME
    freeze = json.loads((ROOT/'receipts/shape010_20261014/FREEZE_RECEIPT.json').read_text(encoding='utf8'))
    digests = dict(package=freeze['package_sha256'], manifest=sha(manifest_path), bindings=sha(binding_path))
    if digests['manifest'] != freeze['source_manifest_sha256'] or digests['bindings'] != freeze['bindings_sha256']:
        raise ValueError('Local source freeze differs from its receipt')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    bindings = json.loads(binding_path.read_text(encoding='utf8'))
    policy = json.loads((ROOT/'shape010_policy.json').read_text(encoding='utf8'))
    validators = namespace()
    validators['validate_documents'](manifest, bindings, policy, digests['manifest'])
    for name, digest in manifest['files'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Local frozen source differs: '+name)
    if freeze['policy_sha256'] != manifest['policy_sha256'] or manifest['files']['scripts/deploy_shape010.py'] != sha(__file__):
        raise ValueError('Policy/deployer is not the frozen source')
    if package is not None:
        validators['validate_package'](package, digests['package'], digests['manifest'], digests['bindings'])
    return digests


def native_action(args, output):
    digests = local_freeze(None if args.start else args.package)
    token = uuid.uuid4().hex
    stages = []
    result = None
    try:
        if args.start:
            if args.install_receipt is None:
                raise ValueError('--start requires the successful local install receipt')
            installed = json.loads(Path(args.install_receipt).read_text(encoding='utf8'))
            if (installed.get('status') != 'remote_install_verified' or installed.get('digests') != digests
                    or installed.get('remote_root') != REMOTE_ROOT):
                raise ValueError('Start receipt is not this verified frozen installation')
            record, result = ssh_phase('start', token, digests)
            stages.append(record)
        else:
            record, prepared = ssh_phase('prepare', token, digests)
            stages.append(record)
            if prepared is None or prepared.get('status') != 'private_incoming_prepared':
                raise RuntimeError('Remote incoming preparation failed/unknown')
            expected = REMOTE_ROOT.rsplit('/', 1)[0]+'/.shape010_incoming_'+token+'/source.tar.gz'
            if prepared.get('archive') != expected or prepared.get('target') != REMOTE_ROOT or prepared.get('target_created') is not False:
                raise ValueError('Remote preparation returned an unexpected target')
            command = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                       str(Path(args.package).resolve()), HOST+':'+expected]
            transfer = subprocess.run(command, text=True, encoding='utf8', capture_output=True, timeout=80)
            stages.append(dict(command=command, native_scp_exit_code=transfer.returncode,
                               stdout=transfer.stdout, stderr=transfer.stderr))
            if transfer.returncode != 0:
                raise RuntimeError('Source transfer failed/unknown')
            record, result = ssh_phase('install', token, digests)
            stages.append(record)
        expected_status = ('native_start_cli_completed_awaiting_first_update_proof' if args.start else
                           'new_shape010_source_root_installed')
        if result is None or result.get('status') != expected_status:
            raise RuntimeError('Remote action failed/unknown; preserve its directory and receipts')
        receipt = dict(UTC=datetime.now(timezone.utc).isoformat(),
                       status='remote_start_cli_verified' if args.start else 'remote_install_verified',
                       remote_root=REMOTE_ROOT, digests=digests, stages=stages, remote_result=result,
                       scientific_first_update_verified=False, automatic_retry_allowed=False)
    except Exception as error:
        receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), status='failed_or_unknown_preserved',
                       error_type=type(error).__name__, error=str(error), remote_root=REMOTE_ROOT,
                       digests=digests, stages=stages, remote_result=result, automatic_retry_allowed=False)
        write_new(output/('REMOTE_START_RECEIPT.json' if args.start else 'REMOTE_INSTALL_RECEIPT.json'), receipt)
        raise
    path = output/('REMOTE_START_RECEIPT.json' if args.start else 'REMOTE_INSTALL_RECEIPT.json')
    write_new(path, receipt)
    print(json.dumps(dict(status=receipt['status'], receipt=str(path), remote_result=result)))


def self_check(output):
    """Exercise the actual stdin validator/installer against opaque fake files."""
    ns = namespace()
    fake_parent = output/'artificial_remote_parent'
    fake_parent.mkdir()
    old, target = fake_parent/'old_endpoint', fake_parent/'new010'
    old.mkdir()
    ns['OLD_ROOT'], ns['REMOTE_ROOT'] = old.as_posix(), target.as_posix()
    input_hashes = {}
    for key, relative in ns['INPUTS'].items():
        path = old/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('ARTIFICIAL OPAQUE FILE '+key+'\n').encode())
        input_hashes[key] = sha(path)
    old_before = {name: sha(old/relative) for name, relative in ns['INPUTS'].items()}
    policy = dict(schema='n01/shape010-policy/v1', run_id='shape010_20261014',
                  remote_root=target.as_posix(), immutable_endpoint_root=old.as_posix(),
                  G_updates=2000, D_updates=0, automatic_restart=False)
    encode = lambda value: (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
    sources = {name: ('ARTIFICIAL SOURCE '+name+'\n').encode() for name in ns['REQUIRED_SOURCES']}
    sources['shape010_policy.json'] = encode(policy)
    manifest = dict(schema='n01/shape010-source/v1', run_id='shape010_20261014', before_scientific_numeric_decode=True,
                    G_update_budget=2000, D_update_budget=0, baseline_class='n01.model008.N01Response008',
                    candidate_class='n01.model010.FrozenDeterministicVoltageShape010',
                    files={name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
                    bound_inputs=input_hashes, policy_sha256=hashlib.sha256(sources['shape010_policy.json']).hexdigest())
    manifest_bytes = encode(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    bindings = dict(schema='n01/shape010-bindings/v1', run_id='shape010_20261014',
                    roots=[target.as_posix(), old.as_posix()], policy_sha256=manifest['policy_sha256'],
                    source_manifest=dict(path=target.as_posix()+'/'+MANIFEST_NAME, sha256=manifest_sha),
                    files={key: dict(path=old.as_posix()+'/'+relative, sha256=input_hashes[key])
                           for key, relative in ns['INPUTS'].items()})
    binding_bytes = encode(bindings)
    binding_sha = hashlib.sha256(binding_bytes).hexdigest()
    contents = dict(sources, **{MANIFEST_NAME: manifest_bytes, BINDINGS_NAME: binding_bytes})
    checks = []

    def archive(name, entries):
        path = output/(name+'.tar.gz')
        with tarfile.open(path, 'x:gz') as tar:
            for member, data in entries:
                if isinstance(member, str):
                    member = tarfile.TarInfo(member)
                    member.size = len(data)
                tar.addfile(member, io.BytesIO(data) if member.isfile() else None)
        return path

    def validate(path, archive_digest=None):
        return ns['validate_package'](path, sha(path) if archive_digest is None else archive_digest, manifest_sha, binding_sha)

    def reject(name, action):
        try:
            action()
        except (ValueError, FileExistsError) as error:
            checks.append(dict(name=name, passed=True, rejection=str(error)))
            return
        raise AssertionError('Unsafe fixture accepted: '+name)

    valid = archive('valid', list(contents.items()))
    payload = validate(valid)
    before = ns['verify_inputs'](bindings)
    prepared = ns['prepare']('0'*32)
    assert prepared['target_created'] is False and not target.exists()
    installed = ns['install_payload'](payload)
    assert installed == target and all(sha(target/name) == hashlib.sha256(data).hexdigest() for name, data in contents.items())
    assert ns['verify_inputs'](bindings) == before
    checks.append(dict(name='valid_exact_inventory_install_and_seven_opaque_SHA_inputs', passed=True,
                       source_files=len(sources), immutable_inputs=len(before), private_numeric_decode=False))
    reject('existing_root_never_overwritten', lambda: ns['install_payload'](payload))
    reject('prepare_existing_root_never_overwritten', lambda: ns['prepare']('1'*32))
    reject('wrong_archive_digest', lambda: validate(valid, 'f'*64))
    for name in ('../escape.py', '/absolute.py', 'src//alias.py', 'src/./alias.py', 'src/../alias.py', 'C:/windows.py', 'src\\windows.py'):
        path = archive('unsafe_'+str(len(checks)), list(contents.items())+[(name, b'unsafe')])
        reject('unsafe_tar_path_'+name, lambda path=path: validate(path))
    duplicate = archive('duplicate', list(contents.items())+[(next(iter(sources)), b'duplicate')])
    reject('duplicate_member', lambda: validate(duplicate))
    extra = archive('extra', list(contents.items())+[('unlisted.py', b'extra')])
    reject('unlisted_member', lambda: validate(extra))
    missing = archive('missing', [(name, data) for name, data in contents.items() if name != 'src/n01/model010.py'])
    reject('missing_member', lambda: validate(missing))
    changed = archive('changed', [(name, b'changed' if name == 'src/n01/model010.py' else data) for name, data in contents.items()])
    reject('changed_source_digest', lambda: validate(changed))
    for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE):
        member = tarfile.TarInfo('src/n01/model010.py');member.type = kind;member.linkname = '/unsafe_target'
        path = archive('wrong_type_'+str(len(checks)), [(name, data) for name, data in contents.items() if name != member.name]+[(member, b'')])
        reject('symlink_hardlink_directory_type_'+repr(kind), lambda path=path: validate(path))
    redirected = json.loads(json.dumps(bindings))
    redirected['files']['deterministic']['path'] = (output/'outside.pt').as_posix()
    reject('immutable_input_redirected', lambda: ns['verify_inputs'](redirected))
    changed_binding = json.loads(json.dumps(bindings))
    changed_binding['files']['dataset']['sha256'] = 'a'*64
    reject('immutable_input_digest_changed', lambda: ns['verify_inputs'](changed_binding))
    assert old_before == {name: sha(old/relative) for name, relative in ns['INPUTS'].items()}
    result = dict(schema='n01/shape010-deployment-artificial-qa/v1', UTC=datetime.now(timezone.utc).isoformat(),
                  status='passed_artificial_tar_and_fresh_root_checks', checks=checks, passed_checks=len(checks),
                  native_SSH_calls=0, SCP_calls=0, actual_remote_installations=0, scientific_numeric_decodes=0,
                  actual_trained_checkpoint_loads=0, scientific_model_forwards=0, optimizer_updates=0,
                  protected_artificial_immutable_files_unchanged=True, deployer_sha256=sha(__file__),
                  receipt_semantics='Opaque fake .npz/.pt files are hash fixtures; no numeric loader or remote metadata function ran.')
    write_new(output/'ENGINEERING_RECEIPT.json', result)
    print(json.dumps(dict(status=result['status'], passed_checks=result['passed_checks'], receipt=str(output/'ENGINEERING_RECEIPT.json'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--self-check', action='store_true')
    modes.add_argument('--start', action='store_true')
    parser.add_argument('--package', help='Frozen small source tar.gz; required for installation')
    parser.add_argument('--install-receipt', help='Successful local REMOTE_INSTALL_RECEIPT.json, required for --start')
    parser.add_argument('--out', required=True, help='Fresh E/D receipt directory')
    args = parser.parse_args()
    if not args.self_check and not args.start and args.package is None:
        parser.error('--package is required for installation')
    output = fresh_output(args.out)
    if args.self_check:
        self_check(output)
    else:
        native_action(args, output)


if __name__ == '__main__':
    main()
