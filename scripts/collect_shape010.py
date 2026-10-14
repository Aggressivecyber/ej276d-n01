"""Recover one completed 010 job by SHA/JSON verification, never numeric replay.

The completed archive is exclusive and regular-file-only. The collector does
not import torch/numpy, load a checkpoint, decode arrays, train or restart.
--inspect-only reads an existing remote archive after an uncertain SSH result;
it never silently recreates an archive or overwrites local recovery files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

from deploy_shape010 import HOST, PYTHON, REMOTE_ROOT, REMOTE_SCRIPT as INSTALL_HELPERS, local_freeze


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = 'shape010_complete_01.tar'
CONTROLS = ('START_RECEIPT.json', 'REMOTE_SOURCE_INSTALL_RECEIPT.json', 'shape010_policy.json',
            'SHAPE010_SOURCE_MANIFEST.json', 'SHAPE010_BINDINGS.json')


COMMON = r'''
def required(condition,message):
 if not condition:raise ValueError(message)

def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()

def safe_name(name):
 required(isinstance(name,str) and bool(name) and not name.startswith('/')
          and '\\' not in name and ':' not in name
          and all(ord(c)>=32 and ord(c)!=127 for c in name),'Unsafe archive path')
 required(all(part not in ('','.', '..') for part in name.split('/'))
          and PurePosixPath(name).as_posix()==name,'Noncanonical archive path')
 return name

def verify_job(documents):
 start=documents['START_RECEIPT.json'];launch=documents['runs/primary/supervisor/launch.json']
 native=documents['runs/primary/supervisor/exit_receipt.json']
 complete=documents['runs/primary/artifacts/COMPLETE.json']
 status=documents['runs/primary/artifacts/status.json']
 proof=documents['runs/primary/artifacts/FIRST_UPDATE_PROOF.json']
 frozen=documents['runs/primary/artifacts/FROZEN_INPUTS.json']
 required(native.get('exit_code')==0 and native.get('child_native_exit_code')==0
          and 'reason' in native and native['reason'] is None
          and 'error' in native and native['error'] is None
          and native.get('registered_G_budget')==2000,'Native job is not a clean zero exit')
 required(complete.get('schema')=='n01/shape010-complete/v1' and complete.get('run_id')=='shape010_20261014'
          and complete.get('status')=='shape010_training_and_TRAIN_evaluation_complete'
          and complete.get('G_updates')==2000 and complete.get('artificial_only') is False
          and complete.get('DEV_numeric_read') is False,'Wrong completed010 identity/budget')
 counters=complete.get('counters',{})
 required(counters.get('G_updates')==counters.get('attempted_G_updates')==2000
          and all(counters.get(key)==0 for key in ('D_updates','noise_updates','population_updates','C2ST_updates')),
          'Completed/attempted G or other optimizer counts differ')
 required(complete.get('frozen_A_005_unchanged') is True
          and complete.get('trainable_parameter_names')==['basis_head.weight','basis_head.bias'],
          'Frozen/parameter-whitelist proof differs')
 required(status.get('status')==complete['status'] and status.get('counters')==counters
          and status.get('attempted_operation_counters')==counters and status.get('in_flight') is None
          and status.get('DEV_numeric_read') is False,'Terminal status is not the complete boundary')
 supervisor=start.get('supervisor_pid');child=native.get('child_pid')
 required(type(supervisor) is int and supervisor>0 and type(child) is int and child>0
          and supervisor==launch.get('supervisor_pid')==native.get('supervisor_pid')
          and child==launch.get('child_pid')==status.get('pid')==proof.get('pid'),'PID receipt chain differs')
 required(launch.get('command')==native.get('command') and '-m' in native['command']
          and 'n01.train_shape010' in native['command'] and start.get('lock_inherited_by_supervisor') is True,
          'Native command/scientific lock differs')
 required(proof.get('actual_G_updates')==1 and proof.get('new_head_gradient_nonzero') is True
          and proof.get('new_head_parameters_changed') is True and proof.get('no_discriminator_created') is True
          and proof.get('artificial_only') is False and proof.get('DEV_numeric_read') is False,
          'Real first-update proof differs')
 required(complete.get('frozen_state_sha256')==proof.get('frozen_state_sha256')==frozen.get('frozen_state_sha256')
          and frozen.get('DEV_numeric_read') is False and frozen.get('artificial_only') is False,
          'Saved frozen-state consistency differs')
 coverage=complete.get('coverage',{})
 required(coverage.get('source_unique')==coverage.get('source_total')==6593
          and coverage.get('real_unique')==coverage.get('real_total')==21047,'Full legal TRAIN coverage differs')
 artifacts=complete.get('artifacts')
 required(isinstance(artifacts,dict) and bool(artifacts),'Missing completed artifact inventory')
 for name,row in artifacts.items():
  safe_name(name);required('/' not in name and isinstance(row,dict) and set(row)=={'sha256','bytes'}
      and isinstance(row['sha256'],str) and re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None
      and type(row['bytes']) is int and row['bytes']>=0,'Invalid artifact SHA/size record')
 required('COMPLETE.json' not in artifacts and not any('FAIL' in name.upper() or name.lower()=='failure.pt' for name in artifacts),
          'Failure/recursive inventory appears in completed job')
 return dict(supervisor_pid=supervisor,child_pid=child,child_native_exit_code=0,supervisor_native_exit_code=0,
             G_updates=2000,attempted_G_updates=2000,other_optimizer_updates=0,
             artifact_count=len(artifacts),counters=counters,frozen_state_sha256=complete['frozen_state_sha256'],
             complete_status=complete['status'],finished_UTC=native['finished_UTC'],elapsed_seconds=native['elapsed_seconds'])

def index_tar(path):
 result={}
 with tarfile.open(path,'r:') as archive:
  for member in archive.getmembers():
   name=safe_name(member.name)
   required(name not in result,'Duplicate archive member')
   required(member.isfile() and not member.issym() and not member.islnk(),'Nonregular/link archive member')
   stream=archive.extractfile(member);required(stream is not None,'Missing archive payload')
   digest=hashlib.sha256();size=0
   for block in iter(lambda:stream.read(1024**2),b''):digest.update(block);size+=len(block)
   required(size==member.size,'Truncated archive payload')
   result[name]=dict(sha256=digest.hexdigest(),bytes=size)
 return result

def tar_json(path,names):
 result={}
 with tarfile.open(path,'r:') as archive:
  for name in names:
   member=archive.getmember(name);required(member.isfile(),'Nonregular JSON control')
   required(member.size<=4*1024**2,'JSON control exceeds metadata bound')
   result[name]=json.loads(archive.extractfile(member).read())
 return result

def verify_archive_inventory(path,expected):
 actual=index_tar(path)
 required(actual==expected,'Archive members/SHA/sizes differ from completed source inventory')
 return actual
'''


REMOTE_CODE = r'''
from datetime import datetime,timezone
import hashlib,json,os,re,sys,tarfile
from pathlib import Path,PurePosixPath
REMOTE_ROOT='/home/rylan/projects/ej276d-n01/shape010_20261014'
ARCHIVE_NAME='shape010_complete_01.tar'
CONTROLS=('START_RECEIPT.json','REMOTE_SOURCE_INSTALL_RECEIPT.json','shape010_policy.json',
          'SHAPE010_SOURCE_MANIFEST.json','SHAPE010_BINDINGS.json')
JSON_JOB=('START_RECEIPT.json','runs/primary/supervisor/launch.json','runs/primary/supervisor/exit_receipt.json',
          'runs/primary/artifacts/COMPLETE.json','runs/primary/artifacts/status.json',
          'runs/primary/artifacts/FIRST_UPDATE_PROOF.json','runs/primary/artifacts/FROZEN_INPUTS.json')
SHARED_INSTALLERS
SHARED_VALIDATORS

def main():
 required(os.name=='posix' and len(sys.argv)==5,'Expected native collector mode and frozen hashes')
 mode,manifest_sha,bindings_sha,policy_sha=sys.argv[1:]
 required(mode in ('create','inspect') and all(re.fullmatch('[0-9a-f]{64}',v) for v in (manifest_sha,bindings_sha,policy_sha)),
          'Invalid safe collector arguments')
 root=installers['unredirected'](Path(REMOTE_ROOT));required(root.is_dir(),'Missing completed remote root')
 required(sha_file(root/'SHAPE010_SOURCE_MANIFEST.json')==manifest_sha
          and sha_file(root/'SHAPE010_BINDINGS.json')==bindings_sha
          and sha_file(root/'shape010_policy.json')==policy_sha,'Completed control freeze differs')
 read=lambda name:json.loads((root/name).read_text(encoding='utf8'))
 manifest=read('SHAPE010_SOURCE_MANIFEST.json');bindings=read('SHAPE010_BINDINGS.json');policy=read('shape010_policy.json')
 installers['validate_documents'](manifest,bindings,policy,manifest_sha)
 source_digests={}
 for name,digest in manifest['files'].items():
  path=installers['unredirected'](root/name)
  required(sha_file(path)==digest,'Frozen source differs: '+name);source_digests[name]=digest
 inputs=installers['verify_inputs'](bindings)
 documents={name:read(name) for name in JSON_JOB};summary=verify_job(documents)
 complete=documents['runs/primary/artifacts/COMPLETE.json'];artifact_root=root/'runs/primary/artifacts'
 actual_artifacts={path.name for path in artifact_root.iterdir() if path.is_file() and not path.is_symlink()}
 required(actual_artifacts==set(complete['artifacts'])|{'COMPLETE.json'},'Actual completed artifacts differ')
 required(not any(path.name.upper().startswith('FAIL') for path in artifact_root.iterdir()),'Failure artifact remains')
 for name,row in complete['artifacts'].items():
  path=installers['unredirected'](artifact_root/name)
  required(path.is_file() and path.stat().st_size==row['bytes'] and sha_file(path)==row['sha256'],
           'Completed artifact SHA/size differs: '+name)
 expected_files={row['path']:row['sha256'] for row in bindings['files'].values()}
 expected_files.update({str(root/'shape010_policy.json'):policy_sha,str(root/'SHAPE010_SOURCE_MANIFEST.json'):manifest_sha,
                        str(root/'SHAPE010_BINDINGS.json'):bindings_sha})
 expected_sources={str(root/name):digest for name,digest in source_digests.items()}
 for freeze in (read('runs/primary/artifacts/BINDING.json'),documents['runs/primary/artifacts/FROZEN_INPUTS.json']['binding']):
  required(freeze['files_sha256']==expected_files and freeze['sources_sha256']==expected_sources,
           'Saved binding does not match verified frozen files')
 files=[]
 for path in sorted((root/'runs/primary').rglob('*')):
  required(not path.is_symlink(),'Link in completed job tree')
  if path.is_dir():continue
  required(path.is_file(),'Nonregular file in completed job tree')
  installers['unredirected'](path);files.append(path)
 files += [installers['unredirected'](root/name) for name in CONTROLS]
 expected={path.relative_to(root).as_posix():dict(sha256=sha_file(path),bytes=path.stat().st_size) for path in files}
 required(len(expected)==len(files),'Duplicate completed file inventory')
 archive=root/ARCHIVE_NAME
 if mode=='create':
  required(not os.path.lexists(archive),'Preserve existing complete archive; inspect it explicitly')
  with tarfile.open(archive,'x:',format=tarfile.PAX_FORMAT) as tar:
   for path in files:tar.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
 else:required(archive.is_file() and not archive.is_symlink(),'Completed archive unavailable for inspection')
 verify_archive_inventory(archive,expected)
 required(installers['verify_inputs'](bindings)==inputs,'Immutable inputs changed during collection')
 for name,digest in source_digests.items():required(sha_file(root/name)==digest,'Frozen source changed during collection')
 result=dict(status='completed_archive_created_verified' if mode=='create' else 'existing_completed_archive_inspected_verified',
             UTC=datetime.now(timezone.utc).isoformat(),remote_root=str(root),archive=str(archive),
             archive_sha256=sha_file(archive),archive_bytes=archive.stat().st_size,inventory=expected,
             file_count=len(expected),control_sha256={name:expected[name]['sha256'] for name in CONTROLS},
             source_sha256=source_digests,source_file_count=len(source_digests),immutable_inputs=inputs,
             immutable_input_count=len(inputs),immutable_inputs_unchanged=True,job=summary,
             checkpoint_loads=0,numeric_array_decodes=0,scientific_model_forwards=0,new_optimizer_updates=0)
 print(json.dumps(result))

if __name__=='__main__':
 try:main()
 except Exception as error:
  print(json.dumps(dict(status='collection_failed_or_unknown_preserved',error_type=type(error).__name__,error=str(error),
                       automatic_retry_allowed=False)))
  sys.exit(1)
'''


def shared_namespace():
    value = dict(hashlib=hashlib, json=json, Path=Path, PurePosixPath=PurePosixPath, tarfile=tarfile,
                 re=__import__('re'))
    exec(compile(COMMON, 'shape010_collection_shared_checks', 'exec'), value)
    return value


CHECKS = shared_namespace()


def sha(path):
    return CHECKS['sha_file'](path)


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def safe_extract(path, destination, inventory):
    destination = Path(destination).absolute()
    if destination.drive.upper() not in ('E:', 'D:') or os.path.lexists(destination):
        raise ValueError('Collection extraction requires a fresh E/D destination')
    CHECKS['verify_archive_inventory'](path, inventory)
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    if root != destination:
        raise ValueError('Extraction root is redirected')
    with tarfile.open(path, 'r:') as archive:
        for member in archive.getmembers():
            name = CHECKS['safe_name'](member.name)
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            resolved = target.parent.resolve()
            if resolved != root and root not in resolved.parents:
                raise ValueError('Resolved extraction escaped the designated root')
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
            descriptor = os.open(target, flags, 0o600)
            with os.fdopen(descriptor, 'wb') as output, archive.extractfile(member) as source:
                shutil.copyfileobj(source, output, length=1024**2)
            if sha(target) != inventory[name]['sha256'] or target.stat().st_size != inventory[name]['bytes']:
                raise ValueError('Extracted SHA/size differs: '+name)
    return root


def collect(args, output):
    digests = local_freeze()
    policy_sha = sha(ROOT/'shape010_policy.json')
    bootstrap = "installers={'__name__':'readonly_completed_shape010_install_checks'}\nexec(compile("+repr(INSTALL_HELPERS)+",'frozen_shape010_install_checks','exec'),installers)"
    script = REMOTE_CODE.replace('SHARED_INSTALLERS', bootstrap).replace('SHARED_VALIDATORS', COMMON)
    mode = 'inspect' if args.inspect_only else 'create'
    command = ['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST,
               PYTHON, '-B', '-', mode, digests['manifest'], digests['bindings'], policy_sha]
    stage = dict(command=command)
    try:
        process = subprocess.run(command, input=script, text=True, encoding='utf8', capture_output=True, timeout=180)
        stage.update(native_ssh_exit_code=process.returncode, stdout=process.stdout, stderr=process.stderr)
        write_new(output/'REMOTE_ARCHIVE_RECEIPT.json', stage)
        if process.returncode != 0:
            raise RuntimeError('Remote collection failed/unknown; inspect existing state before retry')
        remote = json.loads(process.stdout)
        wanted = 'existing_completed_archive_inspected_verified' if args.inspect_only else 'completed_archive_created_verified'
        if remote.get('status') != wanted or remote.get('archive') != REMOTE_ROOT+'/'+ARCHIVE_NAME:
            raise ValueError('Unexpected remote completed archive identity')
        if args.inspect_only:
            write_new(output/'INSPECTION_RECEIPT.json', remote)
            print(json.dumps(dict(status=wanted, receipt=str(output/'INSPECTION_RECEIPT.json'), archive_sha256=remote['archive_sha256'])))
            return
        archive = ROOT/'runs/shape010_20261014'/ARCHIVE_NAME
        extracted = archive.parent/'collected_01'
        if os.path.lexists(archive) or os.path.lexists(extracted):
            raise ValueError('Preserve existing local archive/extraction; no overwrite')
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.parent.resolve() != archive.parent.absolute() or archive.drive.upper() not in ('E:', 'D:'):
            raise ValueError('Archive parent must be the designated unredirected E/D directory')
        if shutil.disk_usage(archive.parent).free < 2*remote['archive_bytes']+256*1024**2:
            raise ValueError('Insufficient archive-plus-extraction free space')
        # Reserve this new, owned output exclusively before scp writes to it.
        with archive.open('xb'):
            pass
        transfer_command = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                            HOST+':'+remote['archive'], str(archive)]
        transferred = subprocess.run(transfer_command, text=True, encoding='utf8', capture_output=True, timeout=180)
        transfer = dict(command=transfer_command, native_scp_exit_code=transferred.returncode,
                        stdout=transferred.stdout, stderr=transferred.stderr)
        write_new(output/'TRANSFER_RECEIPT.json', transfer)
        if transferred.returncode != 0:
            raise RuntimeError('Download failed/unknown; preserve the owned partial file')
        if sha(archive) != remote['archive_sha256'] or archive.stat().st_size != remote['archive_bytes']:
            raise ValueError('Remote/local completed archive SHA/size differs')
        safe_extract(archive, extracted, remote['inventory'])
        documents = {name: json.loads((extracted/name).read_text(encoding='utf8')) for name in (
            'START_RECEIPT.json','runs/primary/supervisor/launch.json','runs/primary/supervisor/exit_receipt.json',
            'runs/primary/artifacts/COMPLETE.json','runs/primary/artifacts/status.json',
            'runs/primary/artifacts/FIRST_UPDATE_PROOF.json','runs/primary/artifacts/FROZEN_INPUTS.json')}
        job = CHECKS['verify_job'](documents)
        complete = documents['runs/primary/artifacts/COMPLETE.json']
        for name, row in complete['artifacts'].items():
            recovered = extracted/'runs/primary/artifacts'/name
            if sha(recovered) != row['sha256'] or recovered.stat().st_size != row['bytes']:
                raise ValueError('Recovered completed artifact differs: '+name)
        control_sha = {name: sha(extracted/name) for name in CONTROLS}
        if control_sha != remote['control_sha256'] or job != remote['job']:
            raise ValueError('Recovered control/native summaries differ')
        result = dict(schema='n01/shape010-collection/v1', UTC=datetime.now(timezone.utc).isoformat(),
                      status='completed_shape010_archive_downloaded_extracted_verified', archive=str(archive),
                      archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                      collected_root=str(extracted), file_count=remote['file_count'], artifact_count=job['artifact_count'],
                      control_sha256=control_sha, job=job, source_file_count=remote['source_file_count'],
                      frozen_source_sha256=remote['source_sha256'], immutable_inputs=remote['immutable_inputs'],
                      immutable_input_count=remote['immutable_input_count'], immutable_inputs_unchanged=True,
                      native_ssh_exit_code=0, native_scp_exit_code=0, checkpoint_loads=0,
                      numeric_array_decodes=0, scientific_model_forwards=0, new_optimizer_updates=0,
                      collector_sha256=sha(__file__), automatic_retry_allowed=False)
        write_new(output/'COLLECTION_RECEIPT.json', result)
        print(json.dumps({key: result[key] for key in ('status','collected_root','archive_sha256','archive_bytes','file_count','artifact_count')}))
    except Exception as error:
        write_new(output/'FAILED_COLLECTION_RECEIPT.json', dict(status='failed_or_unknown_preserved', error_type=type(error).__name__,
            error=str(error), stage=stage, automatic_retry_allowed=False, new_optimizer_updates=0))
        raise


def self_check(output):
    checks = []
    valid = output/'valid.tar'
    contents = {'runs/primary/artifacts/opaque.pt': b'OPAQUE ARTIFICIAL CP BYTES', 'START_RECEIPT.json': b'{}\n'}
    with tarfile.open(valid, 'x:') as archive:
        for name, data in contents.items():
            entry = tarfile.TarInfo(name);entry.size = len(data);archive.addfile(entry, io.BytesIO(data))
    inventory = {name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)) for name, data in contents.items()}
    target = output/'artificial_extracted'
    safe_extract(valid, target, inventory)
    checks.append(dict(name='exclusive_regular_byte_extraction_and_SHA_size', passed=True))

    def reject(name, action):
        try:
            action()
        except (ValueError, FileExistsError) as error:
            checks.append(dict(name=name, passed=True, rejection=str(error)))
            return
        raise AssertionError('Unsafe collection fixture accepted: '+name)

    reject('existing_extraction_preserved', lambda: safe_extract(valid, target, inventory))
    wrong = json.loads(json.dumps(inventory));wrong['START_RECEIPT.json']['sha256'] = 'f'*64
    reject('wrong_inventory_hash', lambda: CHECKS['verify_archive_inventory'](valid, wrong))
    for number, name in enumerate(('../escaped', '/absolute', 'runs//alias', 'runs/./alias', 'runs/../alias', 'C:/escape', 'runs\\escape')):
        path = output/('unsafe_%d.tar'%number)
        with tarfile.open(path, 'x:') as archive:
            entry = tarfile.TarInfo(name);entry.size = 1;archive.addfile(entry, io.BytesIO(b'x'))
        reject('unsafe_path_'+name, lambda path=path: CHECKS['index_tar'](path))
    for number, kind in enumerate((tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE)):
        path = output/('type_%d.tar'%number)
        with tarfile.open(path, 'x:') as archive:
            entry = tarfile.TarInfo('opaque');entry.type = kind;entry.linkname = '/outside';archive.addfile(entry)
        reject('nonregular_type_'+repr(kind), lambda path=path: CHECKS['index_tar'](path))
    duplicate = output/'duplicate.tar'
    with tarfile.open(duplicate, 'x:') as archive:
        for _ in range(2):
            entry = tarfile.TarInfo('same');entry.size = 1;archive.addfile(entry, io.BytesIO(b'x'))
    reject('duplicate_member', lambda: CHECKS['index_tar'](duplicate))
    result = dict(schema='n01/shape010-collection-artificial-qa/v1', status='passed_local_artificial_tar_checks',
                  checks=checks, passed_checks=len(checks), remote_SSH_calls=0, SCP_calls=0, checkpoint_loads=0,
                  numeric_array_decodes=0, scientific_model_forwards=0, new_optimizer_updates=0, collector_sha256=sha(__file__))
    write_new(output/'ENGINEERING_RECEIPT.json', result)
    print(json.dumps(dict(status=result['status'], passed_checks=result['passed_checks'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--self-check', action='store_true')
    modes.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if output.drive.upper() not in ('E:', 'D:'):
        raise ValueError('Collection receipts stay on E/D')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'COLLECTION_RECEIPT.json').exists() or (output/'FAILED_COLLECTION_RECEIPT.json').exists():
        raise ValueError('Preserve prior collection receipt')
    if args.self_check:
        self_check(output)
    else:
        collect(args, output)


if __name__ == '__main__':
    main()
