"""Source freeze, safe native deployment/status/collection for array-only 011.

No numpy/torch imports, array decoding, checkpoint loading or scientific work.
Mutating modes are explicit; dispatch never follows install automatically.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import uuid


ROOT = Path(__file__).resolve().parents[1] if not str(__file__).startswith('<') else Path.cwd()
HOST = 'rylan@100.82.231.102'
PYTHON = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
REMOTE = '/home/rylan/projects/ej276d-n01/stability011_20261014'
OLD010 = '/home/rylan/projects/ej276d-n01/shape010_20261014'
OLD008 = '/home/rylan/projects/ej276d-n01/window008_resume01_20261013'
RUN_ID = 'stability011_20261014'
MANIFEST, BINDINGS, POLICY = 'STABILITY011_SOURCE_MANIFEST.json', 'STABILITY011_BINDINGS.json', 'stability011_policy.json'
EXTRA = ('scripts/stability011_ops.py', 'scripts/supervise_stability011.py', 'scripts/run_stability011.py',
         'scripts/stability011_metrics.py', POLICY, 'docs/STABILITY011_PROTOCOL.md')
SHAPE_INPUTS = dict(shape010_manifest='SHAPE010_SOURCE_MANIFEST.json', shape010_bindings='SHAPE010_BINDINGS.json',
    shape010_complete='runs/primary/artifacts/COMPLETE.json', shape010_initial_eval='runs/primary/artifacts/INITIAL_EVALUATION.json',
    shape010_final_eval='runs/primary/artifacts/FINAL_EVALUATION.json', shape010_eval_bank='runs/primary/artifacts/EVALUATION_BANK.npz',
    shape010_binding_receipt='runs/primary/artifacts/BINDING.json', shape010_native_exit='runs/primary/supervisor/exit_receipt.json')
for _view in ('frozen_A', 'shape010_final'):
    for _hyp in ('fast', 'slow', 'common_level'):
        _key = _view+'_'+_hyp+'_TRAIN'
        SHAPE_INPUTS[_key] = 'runs/primary/artifacts/'+_key+'.npz'
INPUT_PATHS = dict(dataset=OLD008+'/data/window008/DATASET008.npz',
                   statistics=OLD008+'/data/window008/FIT_STATISTICS008.json',
                   **{k: OLD010+'/'+v for k, v in SHAPE_INPUTS.items()})
ZERO_COUNTERS = ('model_instances', 'checkpoint_loads', 'model_forward_rows', 'gradient_evaluations',
                 'G_updates', 'D_updates', 'noise_updates', 'population_updates', 'C2ST_updates')
CONTROLS = ('START_RECEIPT.json', 'REMOTE_SOURCE_INSTALL_RECEIPT.json', 'DISPATCH_ATTEMPT.json', POLICY, MANIFEST, BINDINGS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def read_json(path):
    p = Path(path)
    require(not p.is_symlink() and p.stat().st_size <= 16*1024**2, 'Invalid JSON path/size: '+str(p))
    return json.loads(p.read_text(encoding='utf8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def safe_name(name):
    require(isinstance(name, str) and bool(name) and not name.startswith('/') and '\\' not in name and ':' not in name
            and all(ord(c) >= 32 and ord(c) != 127 for c in name), 'Unsafe archive/source name')
    require(all(p not in ('', '.', '..') for p in name.split('/')) and PurePosixPath(name).as_posix() == name,
            'Noncanonical archive/source name')
    require(all(p == p.rstrip(' .') and len(p) <= 255 and
        re.fullmatch('CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]',p.split('.')[0],re.IGNORECASE) is None
        for p in name.split('/')), 'Unsafe Windows member spelling')
    return name


def unredirected(path):
    p = Path(path).absolute()
    for q in (p, *p.parents):
        require(not q.is_symlink(), 'Redirected path rejected: '+str(q))
    require(p.resolve() == p, 'Unresolved path spelling: '+str(p))
    return p


def fresh_directory(path, *, local=False):
    p = unredirected(path)
    if local and os.name == 'nt':
        require(p.drive.upper() in ('E:', 'D:'), 'Local outputs must stay on E/D')
    require(not os.path.lexists(p), 'Preserve existing directory: '+str(p))
    p.mkdir(parents=True, mode=0o700, exist_ok=False)
    return p


def validate_policy(p):
    require(p.get('schema') == 'n01/stability011-policy/v1' and p.get('run_id') == RUN_ID, 'Wrong011 identity')
    require((p.get('remote_root'), p.get('shape010_remote_root'), p.get('reference_remote_root')) == (REMOTE, OLD010, OLD008), 'Wrong011 roots')
    require(p.get('bootstrap_replicates') == 500 and p.get('source_parent_clusters') == 512 and p.get('draws_per_parent') == 4
        and p.get('REAL_reference_original_groups') == 419 and p.get('REAL_reference_rows') == 2048, 'Wrong011 finite cohort')
    require(all(p.get(k) == 0 for k in ZERO_COUNTERS) and p.get('SOURCE_waveform_decode') is False
        and p.get('DEV_numeric_read') is False and p.get('automatic_restart') is False and p.get('overwrite_outputs') is False,
        '011 must be saved-array-only, with no restart/overwrite')
    require(p.get('allowed_dataset_members') == ['real', 'real_ids', 'real_groups', 'source_ids'], 'Wrong011 dataset whitelist')
    require(p.get('seeds') == dict(source_parent_bootstrap=2026101451, REAL_group_bootstrap=2026101452, evaluation_projection=2026101430), 'Wrong011 seeds')
    require(p.get('resource_limits') == dict(CPU_threads=12, GPU_computation=False, job_tree_RSS_GiB=8, wall_seconds=7200,
        output_MiB=256, poll_seconds=10, TERM_grace_seconds=20), 'Wrong011 explicit resource limits')


def validate_documents(m, b, p, manifest_sha):
    validate_policy(p)
    require(m.get('schema') == 'n01/stability011-source/v1' and m.get('run_id') == RUN_ID
        and m.get('before_scientific_numeric_decode') is True, 'Wrong011 source schema')
    require(b.get('schema') == 'n01/stability011-bindings/v1' and b.get('run_id') == RUN_ID
        and isinstance(b.get('roots'), list) and len(b['roots']) == 3 and set(b['roots']) == {REMOTE, OLD010, OLD008}, 'Wrong011 binding roots')
    require(b.get('source_manifest') == dict(path=REMOTE+'/'+MANIFEST, sha256=manifest_sha), 'Wrong011 manifest binding')
    files = m.get('files')
    require(isinstance(files, dict) and len(files) == 63 and set(EXTRA) <= set(files), 'Expected63 source/metadata copies')
    require(m.get('old_shape010_source_file_count') == 53 and len(m.get('protected_manifests', {})) == 4, 'Wrong protected source chain')
    for name, digest in files.items():
        safe_name(name); require(valid_sha(digest), 'Invalid source SHA')
    require(MANIFEST not in files and BINDINGS not in files, 'Recursive source inventory')
    require(set(b.get('files', {})) == set(INPUT_PATHS) and m.get('bound_inputs') == {k:r['sha256'] for k,r in b['files'].items()}, 'Wrong16 input inventory')
    for key, record in b['files'].items():
        require(record == dict(path=INPUT_PATHS[key], sha256=m['bound_inputs'][key]) and valid_sha(record['sha256']), 'Unbound input path: '+key)
    require(m.get('policy_sha256') == b.get('policy_sha256') == files[POLICY], 'Policy SHA chain differs')
    require(m.get('shape010_source_manifest_sha256') == b['files']['shape010_manifest']['sha256'], 'Old010 source chain differs')
    require(all(files.get(n) == d for n,d in m['protected_manifests'].items()), 'Protected metadata copies differ')
    return files


def tar_index(path, *, maximum_bytes):
    result, total = {}, 0
    with tarfile.open(path, 'r:*') as archive:
        for member in archive:
            name = safe_name(member.name)
            require(name not in result and member.isfile() and not member.issym() and not member.islnk(), 'Duplicate or nonregular tar member')
            total += member.size
            require(member.size >= 0 and total <= maximum_bytes and len(result) < 512, 'Tar resource budget exceeded')
            stream = archive.extractfile(member); require(stream is not None, 'Missing tar bytes')
            h, count = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(1024**2), b''):
                count += len(block); h.update(block)
            require(count == member.size, 'Truncated tar member')
            result[name] = dict(bytes=count, sha256=h.hexdigest())
    return result


def package_payload(path, archive_sha, manifest_sha, bindings_sha):
    require(all(valid_sha(v) for v in (archive_sha,manifest_sha,bindings_sha)) and sha(path) == archive_sha, 'Archive SHA differs')
    index = tar_index(path, maximum_bytes=8*1024**2)
    require(MANIFEST in index and BINDINGS in index and POLICY in index, 'Missing source package controls')
    require(index[MANIFEST]['sha256'] == manifest_sha and index[BINDINGS]['sha256'] == bindings_sha, 'Packaged controls differ')
    with tarfile.open(path, 'r:*') as archive:
        contents = {name:archive.extractfile(name).read() for name in index}
    m, b, p = [json.loads(contents[n]) for n in (MANIFEST, BINDINGS, POLICY)]
    files = validate_documents(m,b,p,manifest_sha)
    require(set(index) == set(files)|{MANIFEST,BINDINGS}, 'Unexpected packaged source members')
    require(all(index[n]['sha256'] == d for n,d in files.items()), 'Packaged source digest differs')
    return dict(contents=contents, manifest=m, bindings=b, policy=p, index=index,
                archive_sha256=archive_sha, manifest_sha256=manifest_sha, bindings_sha256=bindings_sha)


def exclusive_extract(path, target, inventory, *, maximum_bytes):
    index = tar_index(path, maximum_bytes=maximum_bytes)
    require(index == inventory, 'Archive inventory differs before extraction')
    target = fresh_directory(target, local=True)
    with tarfile.open(path, 'r:*') as archive:
        for name in sorted(index):
            destination = target/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            require(unredirected(destination).is_relative_to(target), 'Extraction escaped owned root')
            source = archive.extractfile(name); require(source is not None, 'Missing member stream')
            with destination.open('xb') as stream:
                shutil.copyfileobj(source,stream,1024**2)
            require(sha(destination) == index[name]['sha256'] and destination.stat().st_size == index[name]['bytes'], 'Extracted file differs')
    return target


def local_input_paths(repo, policy):
    base = repo/policy['shape010_local_artifacts']
    collection = base.parents[2]
    paths = dict(dataset=repo/'data/window008/DATASET008.npz', statistics=repo/'data/window008/FIT_STATISTICS008.json',
        shape010_manifest=repo/'SHAPE010_SOURCE_MANIFEST.json', shape010_bindings=repo/'SHAPE010_BINDINGS.json',
        shape010_complete=base/'COMPLETE.json', shape010_initial_eval=base/'INITIAL_EVALUATION.json',
        shape010_final_eval=base/'FINAL_EVALUATION.json', shape010_eval_bank=base/'EVALUATION_BANK.npz',
        shape010_binding_receipt=base/'BINDING.json', shape010_native_exit=base.parent/'supervisor/exit_receipt.json')
    paths.update({k:base/(k+'.npz') for k in SHAPE_INPUTS if k.endswith('_TRAIN')})
    return paths


def freeze_sources(repo, output, backup, *, engineering_fixture=False):
    repo = unredirected(repo)
    p = read_json(repo/POLICY); validate_policy(p)
    old_m, old_b = read_json(repo/'SHAPE010_SOURCE_MANIFEST.json'), read_json(repo/'SHAPE010_BINDINGS.json')
    require(old_m.get('schema') == 'n01/shape010-source/v1' and old_m.get('run_id') == 'shape010_20261014'
        and old_b.get('schema') == 'n01/shape010-bindings/v1', 'Wrong old010 source identity')
    require(len(old_m['files']) == 53 and len(old_m['protected_manifests']) == 4, 'Wrong old010 source chain sizes')
    require(old_m['policy_sha256'] == old_b['policy_sha256'] == sha(repo/'shape010_policy.json')
        and old_m['bound_inputs'] == {k:r['sha256'] for k,r in old_b['files'].items()}
        and old_b['source_manifest']['sha256'] == sha(repo/'SHAPE010_SOURCE_MANIFEST.json'), 'Old010 manifest/input chain differs')
    protected = old_m['protected_manifests']
    for rel,digest in {**old_m['files'], **protected}.items():
        safe_name(rel); require(sha(unredirected(repo/rel)) == digest, 'Protected original source changed: '+rel)
    sources = {**old_m['files'], **protected, **{rel:sha(unredirected(repo/rel)) for rel in EXTRA}}
    require(len(sources) == 63, 'Source recipe must contain exactly63 copies')
    paths = local_input_paths(repo,p)
    inputs = {k:sha(unredirected(v)) for k,v in paths.items()}
    complete = read_json(paths['shape010_complete'])
    for name, record in complete['artifacts'].items():
        safe_name(name); q=paths['shape010_complete'].parent/name
        require(q.stat().st_size == record['bytes'] and sha(q) == record['sha256'], 'Collected010 artifact changed: '+name)
    require(inputs['dataset'] == old_b['files']['dataset']['sha256'] and inputs['statistics'] == old_b['files']['statistics']['sha256'], 'Prepared TRAIN inputs differ')
    native = read_json(paths['shape010_native_exit'])
    require(native.get('child_native_exit_code') == native.get('exit_code') == 0 and native.get('reason') is None, 'Old010 did not finish cleanly')
    output = fresh_directory(output,local=True)
    manifest = dict(schema='n01/stability011-source/v1', run_id=RUN_ID, frozen_UTC=now(), before_scientific_numeric_decode=True,
        policy_sha256=sources[POLICY], files=sources, bound_inputs=inputs, old_shape010_source_file_count=53,
        shape010_source_manifest_sha256=inputs['shape010_manifest'], protected_manifests=protected,
        scientific_budgets={k:0 for k in ZERO_COUNTERS}, bootstrap_replicates=500)
    write_new(output/MANIFEST, manifest)
    bindings = dict(schema='n01/stability011-bindings/v1',run_id=RUN_ID,roots=[REMOTE,OLD010,OLD008],
        policy_sha256=sources[POLICY],source_manifest=dict(path=REMOTE+'/'+MANIFEST,sha256=sha(output/MANIFEST)),
        files={k:dict(path=INPUT_PATHS[k],sha256=d) for k,d in inputs.items()})
    write_new(output/BINDINGS,bindings)
    archive_path = output/'stability011_source.tar.gz'
    with tarfile.open(archive_path,'x:gz') as archive:
        for name in sorted(sources):
            archive.add(repo/name,arcname=name,recursive=False)
        for name in (MANIFEST,BINDINGS):
            archive.add(output/name,arcname=name,recursive=False)
    payload = package_payload(archive_path,sha(archive_path),sha(output/MANIFEST),sha(output/BINDINGS))
    require(engineering_fixture or os.name != 'nt' or Path(backup).absolute().drive.upper() == 'D:', 'Source backup must stay on D')
    backup = fresh_directory(backup,local=True)
    require(shutil.disk_usage(backup).free >= archive_path.stat().st_size+256*1024**2, 'Preserve source backup disk margin')
    with (backup/archive_path.name).open('xb') as f, archive_path.open('rb') as src:
        shutil.copyfileobj(src,f,1024**2)
    require(sha(backup/archive_path.name) == sha(archive_path), 'D source backup differs')
    (backup/'SHA256SUMS.txt').write_text(sha(archive_path)+'  '+archive_path.name+'\n',encoding='utf8')
    receipt=dict(schema='n01/stability011-freeze/v1',status='source_and_saved_inputs_frozen_before_analysis',UTC=now(),
        package=str(archive_path),package_sha256=sha(archive_path),package_bytes=archive_path.stat().st_size,
        source_manifest_sha256=payload['manifest_sha256'],bindings_sha256=payload['bindings_sha256'],policy_sha256=sources[POLICY],
        source_file_count=63,input_count=16,old53_bytes_verified=True,protected_metadata_copies=4,
        D_source_backup=None if engineering_fixture else str(backup/archive_path.name),
        source_backup=str(backup/archive_path.name),D_backup_SHA256_verified=not engineering_fixture,artificial_only=engineering_fixture,
        scientific_array_decodes=0,checkpoint_loads=0,model_forwards=0,gradient_evaluations=0,new_bootstrap_replicates=0)
    write_new(output/'FREEZE_RECEIPT.json',receipt)
    return receipt


def load_freeze(directory):
    directory = unredirected(directory)
    receipt=read_json(directory/'FREEZE_RECEIPT.json')
    payload=package_payload(directory/'stability011_source.tar.gz',receipt['package_sha256'],
        receipt['source_manifest_sha256'],receipt['bindings_sha256'])
    require(sha(Path(__file__)) == payload['manifest']['files']['scripts/stability011_ops.py'], 'Controller bytes changed after freeze')
    return receipt,payload


def verify_bound_native(m, b, target):
    target = unredirected(target)
    hashes = {}
    for key,row in b['files'].items():
        p=unredirected(row['path'])
        require(any(p.is_relative_to(Path(r)) for r in (OLD010,OLD008)), 'Input escaped immutable roots: '+key)
        require(sha(p) == row['sha256'], 'Native input changed: '+key)
        hashes[str(p)]=row['sha256']
    old_m,old_b = read_json(b['files']['shape010_manifest']['path']),read_json(b['files']['shape010_bindings']['path'])
    require(old_m.get('schema') == 'n01/shape010-source/v1' and old_m.get('run_id') == 'shape010_20261014'
        and len(old_m['files']) == 53 and len(old_m['protected_manifests']) == 4, 'Wrong old53 source chain')
    require(old_b.get('schema') == 'n01/shape010-bindings/v1' and set(old_b['roots']) == {OLD010,OLD008}
        and old_b['source_manifest'] == dict(path=OLD010+'/SHAPE010_SOURCE_MANIFEST.json',sha256=b['files']['shape010_manifest']['sha256'])
        and old_m['policy_sha256'] == old_b['policy_sha256'] == old_m['files']['shape010_policy.json']
        and old_m['bound_inputs'] == {k:r['sha256'] for k,r in old_b['files'].items()}, 'Old010 metadata chain differs')
    for rel,d in old_m['files'].items():
        safe_name(rel); original=unredirected(Path(OLD010)/rel)
        require(sha(original) == d and m['files'].get(rel) == d, 'Original53 source or copy changed: '+rel)
        hashes[str(original)]=d
    require(m['protected_manifests'] == old_m['protected_manifests'], 'Protected metadata chain differs')
    # Old010 source bundle never contained these four controls. Only their new
    # package copies are verified natively; root separately guards E originals.
    for rel,d in old_m['protected_manifests'].items():
        require(sha(unredirected(target/rel)) == d, 'Protected metadata copy changed: '+rel)
    for key,row in old_b['files'].items():
        p=unredirected(row['path'])
        require(p.is_relative_to(Path(OLD008)) and valid_sha(row['sha256']) and sha(p) == row['sha256'], 'Old immutable input changed: '+key)
        hashes[str(p)]=row['sha256']
    c=read_json(b['files']['shape010_complete']['path'])
    require(c.get('run_id') == 'shape010_20261014' and c.get('G_updates') == 2000 and c.get('frozen_A_005_unchanged') is True, 'Wrong completed010 state')
    for name,row in c['artifacts'].items():
        safe_name(name); require('/' not in name, 'Nonflat completed010 artifact')
        p=unredirected(Path(OLD010)/'runs/primary/artifacts'/name)
        require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Original010 artifact changed: '+name)
        hashes[str(p)]=row['sha256']
    native=read_json(b['files']['shape010_native_exit']['path'])
    require(native.get('child_native_exit_code') == native.get('exit_code') == 0 and native.get('reason') is None, 'Original010 native exit differs')
    return hashes


def verify_runtime(target, manifest_sha, bindings_sha):
    target=unredirected(target)
    require(str(target) == REMOTE and target.is_dir(), 'Only installed011 runtime is allowed')
    require(sha(target/MANIFEST) == manifest_sha and sha(target/BINDINGS) == bindings_sha, 'Installed source/bind controls differ')
    m,b,p = [read_json(target/n) for n in (MANIFEST,BINDINGS,POLICY)]
    validate_documents(m,b,p,manifest_sha)
    for name,d in m['files'].items():
        require(sha(unredirected(target/name)) == d, 'Installed source changed: '+name)
    require(sha(target/POLICY) == b['policy_sha256'], 'Installed policy changed')
    hashes=verify_bound_native(m,b,target)
    return m,b,p,hashes


def runtime_preflight(root, policy):
    root=unredirected(root)
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    disk=shutil.disk_usage(root).free
    cpus=len(os.sched_getaffinity(0))
    require(available >= policy['preflight']['minimum_available_RAM_GiB']*1024**3
        and disk >= policy['preflight']['minimum_free_disk_GiB']*1024**3 and cpus >= policy['resource_limits']['CPU_threads'], 'Insufficient native RAM/disk/CPU')
    markers=('-m n01.', 'scripts/supervise_window', 'scripts/run_window', 'scripts/supervise_shape010.py',
             'scripts/supervise_mean_random009.py', 'scripts/supervise_stability011.py', 'scripts/run_stability011.py', 'scripts/supervise.py')
    active=[line for line in subprocess.check_output(['ps','-eo','pid,args'],text=True,timeout=10).splitlines()
            if any(mark in line for mark in markers)]
    require(not active, 'Another N01 scientific process is active: '+repr(active))
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,timeout=10).strip()
    return dict(RAM_available_bytes=available,disk_free_bytes=disk,available_CPU_affinity_count=cpus,
                active_project_jobs_before=active,GPU_processes_before=gpu,GPU_computation_allowed=False)


def spawn_supervisor(command,root,fd,env):
    with (root/'supervisor_console.log').open('xb') as log:
        return subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=(fd,),env=env)


def native_start(manifest_sha, bindings_sha):
    require(os.name == 'posix', 'Native dispatch is Linux-only')
    import fcntl
    root=unredirected(REMOTE)
    require(not any(os.path.lexists(root/n) for n in ('runs/primary','START_RECEIPT.json','DISPATCH_ATTEMPT.json','supervisor_console.log')), 'Preserve previous011 dispatch; no restart')
    m,b,p,_=verify_runtime(root,manifest_sha,bindings_sha)
    lock_path=unredirected(root.parent/'.n01_scientific_dispatch.lock')
    fd=os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preflight=runtime_preflight(root,p)
        require(Path(PYTHON).is_file(), 'Registered existing Python runtime is missing')
        write_new(root/'DISPATCH_ATTEMPT.json',dict(run_id=RUN_ID,UTC=now(),source_manifest_sha256=manifest_sha,
            bindings_sha256=bindings_sha,automatic_restart=False,preflight=preflight))
        for name in ('tmp','cache'):
            unredirected(root/name).mkdir(mode=0o700,exist_ok=False)
        env=os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src'),TMPDIR=str(root/'tmp'),
            XDG_CACHE_HOME=str(root/'cache'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12',
            OPENBLAS_NUM_THREADS='12',NUMEXPR_NUM_THREADS='12')
        command=[PYTHON,'-B',str(root/'scripts/supervise_stability011.py'),'--policy',str(root/POLICY),
            '--bindings',str(root/BINDINGS),'--job-dir',str(root/'runs/primary'),'--lock-fd',str(fd)]
        process=spawn_supervisor(command,root,fd,env)
        result=dict(schema='n01/stability011-start/v1',run_id=RUN_ID,UTC=now(),status='supervisor_dispatched_awaiting_first_bootstrap',
            supervisor_pid=process.pid,command=command,exclusive_scientific_lock=str(lock_path),lock_inherited_by_supervisor=True,
            source_manifest_sha256=manifest_sha,bindings_sha256=bindings_sha,policy_sha256=sha(root/POLICY),
            bootstrap_budget=500,automatic_restart=False,first_replicate_verified=False,**preflight)
        write_new(root/'START_RECEIPT.json',result)
        return result
    finally:
        os.close(fd)


def native_status(manifest_sha, bindings_sha):
    root=unredirected(REMOTE)
    m,b,p,verified=verify_runtime(root,manifest_sha,bindings_sha)
    names=('START_RECEIPT.json','REMOTE_SOURCE_INSTALL_RECEIPT.json','DISPATCH_ATTEMPT.json',
        'runs/primary/supervisor/launch.json','runs/primary/supervisor/resource_status.json','runs/primary/supervisor/exit_receipt.json',
        'runs/primary/artifacts/status.json','runs/primary/artifacts/FIRST_BOOTSTRAP_PROOF.json',
        'runs/primary/artifacts/FAILED.json','runs/primary/artifacts/COMPLETE.json')
    result={name:read_json(root/name) if (root/name).exists() else None for name in names}
    console=root/'runs/primary/console.log'
    if console.exists():
        with console.open('rb') as f:
            f.seek(max(0,console.stat().st_size-8000)); result['console_tail']=f.read().decode('utf8',errors='replace')
    result['UTC']=now()
    result['_verified_freeze']=dict(source_manifest_sha256=manifest_sha,bindings_sha256=bindings_sha,
        policy_sha256=sha(root/POLICY),bound_inputs={row['path']:row['sha256'] for row in b['files'].values()},
        new_sources={str(root/n):d for n,d in m['files'].items()})
    spec=importlib.util.spec_from_file_location('stability011_owned_supervisor',root/'scripts/supervise_stability011.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result['durable_child_progress']=module.last_progress(root/'runs/primary/artifacts')
    result['in_flight_actual_completion']='only durable child receipts are asserted; unpersisted work unknown'
    return result


def verify_first(state):
    proof=state.get('runs/primary/artifacts/FIRST_BOOTSTRAP_PROOF.json')
    if proof is None:
        return dict(status='awaiting_first_replicate_proof',first_replicate_verified=False)
    start,launch=state['START_RECEIPT.json'],state['runs/primary/supervisor/launch.json']
    require(isinstance(start,dict) and isinstance(launch,dict) and proof.get('run_id') == RUN_ID
        and start.get('supervisor_pid') == launch.get('supervisor_pid') and launch.get('child_pid') == proof.get('pid'), 'First proof/PID chain differs')
    require(proof.get('artificial_only') is False and proof.get('point_checks_passed') is True
        and proof.get('bootstrap_replicates_completed') >= 1 and proof.get('bootstrap_replicates_attempted') >= 1
        and all(proof.get('counters',{}).get(k) == 0 for k in ZERO_COUNTERS)
        and proof.get('inputs_unchanged') is True and isinstance(proof.get('inputs_sha256'),dict)
        and isinstance(proof.get('sources_sha256'),dict), 'First proof is not actual saved-array-only work')
    require(proof.get('projection_context',{}).get('kind') == 'evaluation'
        and proof['projection_context'].get('seed') == 2026101430 and proof['projection_context'].get('update') is None, 'First proof projection differs')
    freeze=state.get('_verified_freeze',{})
    claimed={**proof['inputs_sha256'],**proof['sources_sha256']}
    require(bool(freeze.get('bound_inputs')) and bool(freeze.get('new_sources'))
        and all(claimed.get(path) == d for path,d in {**freeze['bound_inputs'],**freeze['new_sources']}.items()), 'First proof freeze hashes differ')
    return dict(status='actual_first_bootstrap_replicate_verified',first_replicate_verified=True,
        supervisor_pid=start['supervisor_pid'],child_pid=proof['pid'],completed_replicates=proof['bootstrap_replicates_completed'],
        attempted_replicates=proof['bootstrap_replicates_attempted'],new_model_forwards=0,new_optimizer_updates=0)


def verify_completed(state):
    first=verify_first(state); require(first['first_replicate_verified'], 'No actual first bootstrap proof')
    native=state.get('runs/primary/supervisor/exit_receipt.json')
    c=state.get('runs/primary/artifacts/COMPLETE.json'); s=state.get('runs/primary/artifacts/status.json')
    require(isinstance(native,dict) and native.get('exit_code') == native.get('child_native_exit_code') == 0
        and native.get('reason') is None and native.get('error') is None and native.get('registered_bootstrap_budget') == 500,
        'Native011 did not finish cleanly')
    require(isinstance(c,dict) and c.get('run_id') == RUN_ID and c.get('bootstrap_replicates_completed') == 500
        and c.get('bootstrap_replicates_attempted') == 500 and c.get('artificial_only') is False and c.get('point_checks_passed') is True
        and all(c.get('counters',{}).get(k) == 0 for k in ZERO_COUNTERS), 'Wrong completed011 budget/scope')
    require(state.get('runs/primary/artifacts/FAILED.json') is None and isinstance(s,dict)
        and s.get('bootstrap_replicates_completed') == 500 and s.get('bootstrap_replicates_attempted') == 500
        and s.get('pid') == c.get('pid') == native.get('child_pid') == first['child_pid'], 'Terminal completion/failure/PID chain differs')
    require(c.get('inputs_sha256') == s.get('inputs_sha256') and c.get('sources_sha256') == s.get('sources_sha256'), 'Terminal input/source chain differs')
    inventory=c.get('artifacts')
    require(isinstance(inventory,dict) and bool(inventory), 'Missing complete artifact inventory')
    for name,row in inventory.items():
        safe_name(name)
        require('/' not in name and not name.endswith('.pt') and 'FAIL' not in name.upper() and name != 'COMPLETE.json'
            and set(row) == {'bytes','sha256'} and type(row['bytes']) is int and row['bytes'] >= 0 and valid_sha(row['sha256']), 'Invalid completed artifact record')
    return dict(status='native_zero_and_full500_replicates_verified',bootstrap_replicates_completed=500,
        child_pid=c['pid'],supervisor_pid=native['supervisor_pid'],classification=c.get('classification'),artifact_count=len(inventory))


def collection_inventory(root,state):
    verified=verify_completed(state)
    c=state['runs/primary/artifacts/COMPLETE.json']
    names=list(CONTROLS)+['runs/primary/console.log','runs/primary/supervisor/launch.json',
        'runs/primary/supervisor/resource_status.json','runs/primary/supervisor/exit_receipt.json',
        'runs/primary/artifacts/COMPLETE.json']
    names += ['runs/primary/artifacts/'+n for n in c['artifacts']]
    inventory={}
    for name in names:
        safe_name(name); p=unredirected(root/name)
        require(p.is_file(), 'Missing completed collection file: '+name)
        inventory[name]=dict(bytes=p.stat().st_size,sha256=sha(p))
    for name,row in c['artifacts'].items():
        require(inventory['runs/primary/artifacts/'+name] == row, 'Completed011 artifact changed: '+name)
    require(sum(row['bytes'] for row in inventory.values()) <= 272*1024**2, 'Collection output budget exceeded')
    return inventory,verified


def remote_dispatch(payload):
    require(os.name == 'posix' and payload.get('mode') in ('prepare','install','start','status','collect','collect-inspect'), 'Invalid native mode')
    require(all(valid_sha(payload.get(k)) for k in ('archive_sha256','manifest_sha256','bindings_sha256','controller_sha256')), 'Invalid pinned native SHA')
    mode=payload['mode']; ms,bs=payload['manifest_sha256'],payload['bindings_sha256']
    root=unredirected(REMOTE)
    if mode == 'prepare':
        token=payload.get('token'); require(re.fullmatch('[0-9a-f]{32}',str(token)) is not None,'Invalid private incoming token')
        require(not os.path.lexists(root), 'Preserve existing011 runtime')
        incoming=fresh_directory(root.parent/('.stability011.incoming_'+token))
        return dict(status='private_incoming_prepared',incoming=str(incoming),archive=str(incoming/'source.tar.gz'),target_created=False)
    if mode == 'install':
        token=payload.get('token'); require(re.fullmatch('[0-9a-f]{32}',str(token)) is not None,'Invalid private incoming token')
        incoming=unredirected(root.parent/('.stability011.incoming_'+token))
        pack=package_payload(unredirected(incoming/'source.tar.gz'),payload['archive_sha256'],ms,bs)
        require(pack['manifest']['files']['scripts/stability011_ops.py'] == payload['controller_sha256'], 'Frozen remote controller differs')
        require(not os.path.lexists(root),'Preserve existing011 runtime')
        exclusive_extract(incoming/'source.tar.gz',root,pack['index'],maximum_bytes=8*1024**2)
        m,b,p,hashes=verify_runtime(root,ms,bs)
        result=dict(schema='n01/stability011-install/v1',run_id=RUN_ID,status='source_installed_and_original_hashes_verified',
            UTC=now(),remote_root=REMOTE,source_files=63,input_slots=16,source_manifest_sha256=ms,bindings_sha256=bs,
            source_archive_sha256=payload['archive_sha256'],original_verified_paths=len(hashes),
            original_model_checkpoint_loads=0,numeric_array_decodes=0,bootstrap_replicates=0,scientific_started=False)
        write_new(root/'REMOTE_SOURCE_INSTALL_RECEIPT.json',result)
        return result
    require(root.is_dir() and sha(root/'scripts/stability011_ops.py') == payload['controller_sha256'], 'Installed controller differs')
    if mode == 'start':
        return native_start(ms,bs)
    state=native_status(ms,bs)
    if mode == 'status':
        return state
    inventory,verified=collection_inventory(root,state)
    archive=root/'stability011_complete_01.tar'
    if mode == 'collect':
        require(not os.path.lexists(archive), 'Preserve previous collection archive; use collect-inspect')
        with tarfile.open(archive,'x:') as tar:
            for name in sorted(inventory):
                tar.add(root/name,arcname=name,recursive=False)
    else:
        require(archive.is_file() and not archive.is_symlink(),'No existing collection archive to inspect')
    require(tar_index(archive,maximum_bytes=272*1024**2) == inventory, 'Native collection archive differs')
    return dict(schema='n01/stability011-native-archive/v1',UTC=now(),archive=str(archive),archive_sha256=sha(archive),
        archive_bytes=archive.stat().st_size,inventory=inventory,job=verified,source_manifest_sha256=ms,bindings_sha256=bs)


def ssh_action(payload,output):
    code=Path(__file__).read_text(encoding='utf8')
    require(sha_bytes(code.encode('utf8')) == payload['controller_sha256'],'Controller changed before SSH')
    encoded=base64.urlsafe_b64encode(json.dumps(payload,allow_nan=False).encode()).decode()
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PYTHON+' -B - --_remote '+encoded]
    result=subprocess.run(command,input=code,text=True,capture_output=True,timeout=240)
    receipt=dict(command=command,native_ssh_exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,
                 mode=payload['mode'],UTC=now())
    write_new(output/(payload['mode'].replace('-','_')+'_NATIVE_RECEIPT.json'),receipt)
    require(result.returncode == 0,'Native011 '+payload['mode']+' failed; receipt preserved')
    return json.loads(result.stdout)


def controller_action(mode,freeze,output,collected=None):
    receipt,pack=load_freeze(freeze)
    output=fresh_directory(output,local=True)
    payload=dict(mode=mode,archive_sha256=receipt['package_sha256'],manifest_sha256=receipt['source_manifest_sha256'],
        bindings_sha256=receipt['bindings_sha256'],controller_sha256=pack['manifest']['files']['scripts/stability011_ops.py'])
    if mode == 'deploy':
        payload.update(mode='prepare',token=uuid.uuid4().hex)
        prepared=ssh_action(payload,output)
        command=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',str(Path(freeze)/'stability011_source.tar.gz'),HOST+':'+prepared['archive']]
        transfer=subprocess.run(command,capture_output=True,text=True,timeout=180)
        write_new(output/'TRANSFER_RECEIPT.json',dict(command=command,native_scp_exit_code=transfer.returncode,stdout=transfer.stdout,stderr=transfer.stderr))
        require(transfer.returncode == 0,'SCP upload failed; incoming files preserved')
        payload['mode']='install'; result=ssh_action(payload,output)
        write_new(output/'DEPLOYMENT_RECEIPT.json',result)
        return result
    if mode in ('status','verify-start'):
        payload['mode']='status'; result=ssh_action(payload,output)
        write_new(output/'REMOTE_STATE.json',result)
        if mode == 'verify-start':
            verification=verify_first(result); write_new(output/'START_VALIDATION.json',verification); return verification
        return result
    if mode == 'start':
        result=ssh_action(payload,output); write_new(output/'START_RECEIPT.json',result); return result
    require(mode in ('collect','collect-inspect') and collected is not None,'Missing explicit collection destination')
    collected=unredirected(collected)
    require(not os.path.lexists(collected),'Preserve existing extracted outputs')
    result=ssh_action(payload,output); write_new(output/'REMOTE_ARCHIVE_RECEIPT.json',result)
    archive=output/'stability011_complete_01.tar'; part=output/'stability011_complete_01.tar.part'
    require(not os.path.lexists(part) and not os.path.lexists(archive),'Preserve local transfer files')
    command=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST+':'+result['archive'],str(part)]
    transfer=subprocess.run(command,capture_output=True,text=True,timeout=240)
    write_new(output/'TRANSFER_RECEIPT.json',dict(command=command,native_scp_exit_code=transfer.returncode,stdout=transfer.stdout,stderr=transfer.stderr))
    require(transfer.returncode == 0 and part.stat().st_size == result['archive_bytes'] and sha(part) == result['archive_sha256'], 'Downloaded archive differs')
    part.replace(archive)
    exclusive_extract(archive,collected,result['inventory'],maximum_bytes=272*1024**2)
    recovered={name:read_json(collected/name) if (collected/name).is_file() else None for name in
        ('START_RECEIPT.json','runs/primary/supervisor/launch.json','runs/primary/supervisor/exit_receipt.json',
         'runs/primary/artifacts/FIRST_BOOTSTRAP_PROOF.json','runs/primary/artifacts/COMPLETE.json',
         'runs/primary/artifacts/status.json','runs/primary/artifacts/FAILED.json')}
    recovered['_verified_freeze']=dict(bound_inputs={r['path']:r['sha256'] for r in pack['bindings']['files'].values()},
                                     new_sources={REMOTE+'/'+n:d for n,d in pack['manifest']['files'].items()})
    verified=verify_completed(recovered)
    final=dict(schema='n01/stability011-collection/v1',status='native_outputs_downloaded_exclusively_verified',UTC=now(),
        archive=str(archive),archive_sha256=sha(archive),collected_root=str(collected),file_count=len(result['inventory']),
        job=verified,numeric_array_decodes=0,checkpoint_loads=0,model_forwards=0,gradient_evaluations=0,
        automatic_restart=False,source_manifest_sha256=receipt['source_manifest_sha256'],bindings_sha256=receipt['bindings_sha256'])
    write_new(output/'COLLECTION_RECEIPT.json',final)
    return final


def self_check(output):
    output=fresh_directory(output,local=True)
    checks=[]
    def passed(name):
        checks.append(dict(name=name,passed=True))
    def reject(name,action):
        try:
            action()
        except (ValueError,FileExistsError):
            passed(name); return
        raise AssertionError('Unsafe fixture accepted: '+name)
    policy=read_json(ROOT/POLICY); validate_policy(policy)
    changed=json.loads(json.dumps(policy)); changed['gradient_evaluations']=1
    reject('nonzero_gradient_budget_rejected',lambda:validate_policy(changed))
    changed=json.loads(json.dumps(policy)); changed['allowed_dataset_members'].append('val')
    reject('DEV_member_rejected',lambda:validate_policy(changed))
    for name in ('../outside','/absolute','a//b','a/./b','a\\b','C:/escape','NUL','a./file','a/../b'):
        reject('unsafe_name_'+name,lambda name=name:safe_name(name))
    repo=fresh_directory(output/'artificial_repo',local=True)
    def put(name,data):
        p=repo/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(data); return sha(p)
    old_files={}
    for i in range(52):
        name='src/n01/opaque_%02d.py'%i;old_files[name]=put(name,b'# synthetic source; never executed\n')
    old_files['shape010_policy.json']=put('shape010_policy.json',b'{"synthetic":true}\n')
    protected={}
    for i in range(4):
        name='PROTECTED_%d.json'%i;protected[name]=put(name,b'{"synthetic_protected_metadata":true}\n')
    old_inputs={}
    for key in ('dataset','statistics','data_statistics','noise005','deterministic','base_config','training_complete'):
        rel={'dataset':'data/window008/DATASET008.npz','statistics':'data/window008/FIT_STATISTICS008.json'}.get(key,'opaque/'+key)
        old_inputs[key]=dict(path=OLD008+'/'+rel,sha256=put(rel,b'synthetic opaque bytes; not a scientific array or checkpoint'))
    old_m=dict(schema='n01/shape010-source/v1',run_id='shape010_20261014',files=old_files,
        protected_manifests=protected,policy_sha256=old_files['shape010_policy.json'],bound_inputs={k:v['sha256'] for k,v in old_inputs.items()})
    write_new(repo/'SHAPE010_SOURCE_MANIFEST.json',old_m)
    old_b=dict(schema='n01/shape010-bindings/v1',policy_sha256=old_m['policy_sha256'],files=old_inputs,
        source_manifest=dict(path=OLD010+'/SHAPE010_SOURCE_MANIFEST.json',sha256=sha(repo/'SHAPE010_SOURCE_MANIFEST.json')))
    write_new(repo/'SHAPE010_BINDINGS.json',old_b)
    for name in EXTRA:
        if name == POLICY:
            put(name,(json.dumps(policy)+'\n').encode())
        else:
            put(name,(ROOT/name).read_bytes())
    paths=local_input_paths(repo,policy);base=paths['shape010_complete'].parent;base.mkdir(parents=True)
    for key,p in paths.items():
        if key in ('dataset','statistics','shape010_manifest','shape010_bindings','shape010_complete','shape010_native_exit'):
            continue
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(b'synthetic saved bytes; no numeric decode')
    paths['shape010_native_exit'].parent.mkdir(parents=True,exist_ok=True)
    write_new(paths['shape010_native_exit'],dict(child_native_exit_code=0,exit_code=0,reason=None,artificial_fixture=True))
    original_inventory={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in base.iterdir() if p.is_file()}
    write_new(paths['shape010_complete'],dict(artificial_only=True,artifacts=original_inventory))
    frozen=freeze_sources(repo,output/'artificial_freeze',output/'artificial_backup',engineering_fixture=True)
    payload=package_payload(Path(frozen['package']),frozen['package_sha256'],frozen['source_manifest_sha256'],frozen['bindings_sha256'])
    passed('synthetic_63file_16slot_source_freeze_tar_and_small_backup')
    target=output/'artificial_extracted'
    exclusive_extract(frozen['package'],target,payload['index'],maximum_bytes=8*1024**2)
    passed('exclusive_regular_byte_extract_and_SHA_size')
    reject('existing_extract_preserved',lambda:exclusive_extract(frozen['package'],target,payload['index'],maximum_bytes=8*1024**2))
    for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE):
        path=output/('type_'+str(kind[0])+'.tar')
        with tarfile.open(path,'x:') as tar:
            member=tarfile.TarInfo('opaque');member.type=kind;member.linkname='/outside';tar.addfile(member)
        reject('nonregular_tar_'+str(kind[0]),lambda path=path:tar_index(path,maximum_bytes=100))
    duplicate=output/'duplicate.tar'
    with tarfile.open(duplicate,'x:') as tar:
        for _ in range(2):
            member=tarfile.TarInfo('same');member.size=1;tar.addfile(member,io.BytesIO(b'x'))
    reject('duplicate_tar_rejected',lambda:tar_index(duplicate,maximum_bytes=100))
    tampered=json.loads(json.dumps(payload['bindings']));tampered['files']['dataset']['path']=OLD008+'/other.npz'
    reject('input_path_mutation_rejected',lambda:validate_documents(payload['manifest'],tampered,payload['policy'],payload['manifest_sha256']))
    tampered=json.loads(json.dumps(payload['manifest']));tampered['files'].pop('scripts/run_stability011.py')
    reject('source_inventory_mutation_rejected',lambda:validate_documents(tampered,payload['bindings'],payload['policy'],payload['manifest_sha256']))
    spec=importlib.util.spec_from_file_location('stability011_supervisor_fixture',ROOT/'scripts/supervise_stability011.py')
    supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    limits=supervisor.validate_limits(policy)
    for updates,reason in ((dict(rss=9*1024**3),'RSS_limit'),(dict(gpu=1),'unexpected_GPU_computation'),
        (dict(size=257*1024**2),'output_limit'),(dict(elapsed=7201),'wall_limit'),(dict(interrupted=15),'user_signal')):
        sample=dict(rss=0,gpu=0,size=0,elapsed=0);sample.update(updates)
        require(supervisor.limit_reason(limits,**sample)==reason,'Wrong owned resource reason')
    passed('all_explicit_resource_guards_and_CPU_only')
    original_probe=supervisor.subprocess.check_output
    try:
        supervisor.subprocess.check_output=lambda *args,**kwargs:'PID PPID RSS\n900 3 100\n901 900 200\n902 901 50\n777 4 999999\n'
        pids,rss=supervisor.process_tree(900)
        require(pids == {900,901,902} and rss == 350*1024,'Supervisor+child owned RSS omitted or foreign RSS counted')
    finally:
        supervisor.subprocess.check_output=original_probe
    passed('job_tree_RSS_includes_supervisor_and_owned_children_only')
    import types
    captured={}
    original_spawn=subprocess.Popen
    try:
        subprocess.Popen=lambda command,**kwargs:captured.update(command=command,**kwargs) or types.SimpleNamespace(pid=123)
        spawn_supervisor(['synthetic_supervisor'],fresh_directory(output/'artificial_dispatch',local=True),55,{'CUDA_VISIBLE_DEVICES':''})
        require(captured['pass_fds'] == (55,) and captured['start_new_session'] is True
            and captured['env']['CUDA_VISIBLE_DEVICES'] == '', 'Exclusive lock/session/CPU-only spawn contract lost')
    finally:
        subprocess.Popen=original_spawn
    passed('exclusive_lock_fd_inherited_by_detached_supervisor_mock')
    calls=[]
    class ArtificialChild:
        pid=654321
        returncode=None
        waits=0
        def poll(self):
            return self.returncode
        def wait(self,timeout=None):
            self.waits+=1
            if self.waits == 1:
                raise subprocess.TimeoutExpired('synthetic',timeout)
            self.returncode=-9;return self.returncode
    original_killpg=getattr(supervisor.os,'killpg',None)
    original_sigkill=getattr(supervisor.signal,'SIGKILL',None)
    try:
        supervisor.os.killpg=lambda pid,number:calls.append((pid,number))
        supervisor.signal.SIGKILL=9
        child=ArtificialChild();supervisor.stop_owned(child,20)
        require(calls == [(654321,supervisor.signal.SIGTERM),(654321,9)] and child.returncode == -9,'Owned stop affected another group or lost native code')
    finally:
        if original_killpg is None:delattr(supervisor.os,'killpg')
        else:supervisor.os.killpg=original_killpg
        if original_sigkill is None:delattr(supervisor.signal,'SIGKILL')
        else:supervisor.signal.SIGKILL=original_sigkill
    passed('TERM_then_KILL_only_owned_child_group_native_code_preserved_mock')
    partial=fresh_directory(output/'artificial_interrupted',local=True)
    write_new(partial/'status.json',dict(bootstrap_replicates_completed=0,bootstrap_replicates_attempted=0,phase='point'))
    (partial/'bootstrap_attempts.jsonl').write_text('{"replicate_index":0,"event":"entered"}\n{"replicate_index":1,"event":"entered"}\n',encoding='utf8')
    progress=supervisor.last_progress(partial)
    require(progress['entered_replicates_lower_bound']==2 and progress['completed_replicates']==0
        and progress['entered_not_computed_ids']==[0,1] and 'unknown' in progress['unjournaled_or_in_flight_actual_completion'],'Lost in-flight lower bound')
    (partial/'bootstrap_computed.jsonl').write_text('{"replicate_index":0}\n',encoding='utf8')
    (partial/'bootstrap.jsonl').write_text('{"replicate_index":0}\n{"replicate_',encoding='utf8')
    progress=supervisor.last_progress(partial)
    require(progress['saved_replicates_lower_bound']==1 and progress['computed_replicates_lower_bound']==1
        and progress['durable_journals']['bootstrap.jsonl']['truncated_tail'],'Truncated durable tail miscounted')
    passed('SIGKILL_no_FAILED_preserves_entered_computed_saved_lower_bounds')
    # Exercise the real producer's terminal schema, without numeric decode/math.
    spec=importlib.util.spec_from_file_location('stability011_producer_fixture',ROOT/'scripts/run_stability011.py')
    producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
    store=producer.Store(output/'artificial_producer',policy,artificial=True)
    protection=dict(inputs_sha256={row['path']:row['sha256'] for row in payload['bindings']['files'].values()},
                    sources_sha256={REMOTE+'/'+n:d for n,d in payload['manifest']['files'].items()})
    store.protection=protection;store.point_passed=True;store.saved=store.attempted=store.finished=500;store.phase='complete'
    store.status(dict(classification='synthetic_metadata_only'))
    terminal=read_json(store.root/'status.json')
    require(terminal['inputs_sha256']==protection['inputs_sha256'] and terminal['sources_sha256']==protection['sources_sha256'],'Actual producer terminal omitted hashes')
    c=dict(store.base(),classification='synthetic_metadata_only',artifacts={'status.json':dict(bytes=1,sha256='a'*64)})
    proof=dict(store.base(),bootstrap_replicates_completed=1,bootstrap_replicates_attempted=1,inputs_unchanged=True,
        projection_context=dict(kind='evaluation',seed=2026101430,update=None))
    state={'START_RECEIPT.json':dict(supervisor_pid=111),'runs/primary/supervisor/launch.json':dict(supervisor_pid=111,child_pid=store.base()['pid']),
        'runs/primary/supervisor/exit_receipt.json':dict(exit_code=0,child_native_exit_code=0,reason=None,error=None,registered_bootstrap_budget=500,
                                                     supervisor_pid=111,child_pid=store.base()['pid']),
        'runs/primary/artifacts/FIRST_BOOTSTRAP_PROOF.json':proof,'runs/primary/artifacts/COMPLETE.json':c,
        'runs/primary/artifacts/status.json':terminal,'runs/primary/artifacts/FAILED.json':None,
        '_verified_freeze':dict(bound_inputs=protection['inputs_sha256'],new_sources=protection['sources_sha256'])}
    reject('artificial_actual_producer_not_accepted_as_science',lambda:verify_completed(state))
    # Clone only in memory to exercise the native false flag branch; no such
    # science-looking COMPLETE/status is written to the artificial receipts.
    native_fixture=json.loads(json.dumps(state))
    for name in ('runs/primary/artifacts/FIRST_BOOTSTRAP_PROOF.json','runs/primary/artifacts/COMPLETE.json','runs/primary/artifacts/status.json'):
        native_fixture[name]['artificial_only']=False
    require(verify_completed(native_fixture)['bootstrap_replicates_completed']==500,'Actual producer-to-verifier schema bridge failed')
    passed('actual_Store_terminal_schema_bridge_without_bootstrap')
    for key,value in (('bootstrap_replicates_completed',499),('bootstrap_replicates_attempted',501)):
        bad=json.loads(json.dumps(native_fixture));bad['runs/primary/artifacts/COMPLETE.json'][key]=value
        reject('partial_or_overbudget_'+key,lambda bad=bad:verify_completed(bad))
    bad=json.loads(json.dumps(native_fixture));bad['runs/primary/supervisor/exit_receipt.json']['child_native_exit_code']=-9
    reject('nonzero_native_exit_rejected',lambda:verify_completed(bad))
    bad=json.loads(json.dumps(native_fixture));bad['runs/primary/artifacts/FAILED.json']={'synthetic':True}
    reject('failure_receipt_rejected',lambda:verify_completed(bad))
    bad=json.loads(json.dumps(native_fixture));bad['runs/primary/artifacts/status.json']['inputs_sha256']={}
    reject('terminal_hash_mismatch_rejected',lambda:verify_completed(bad))
    result=dict(schema='n01/stability011-ops-artificial-qa/v1',status='artificial_ops_checks_passed',UTC=now(),checks=checks,passed_checks=len(checks),
        actual_producer_terminal_schema_exercised=True,source_package_copies_are_synthetic=True,
        native_SSH_SCP_calls=0,numeric_array_decodes=0,checkpoint_loads=0,model_forwards=0,gradient_evaluations=0,
        actual_bootstrap_replicates=0,actual_optimizer_updates=0,
        scripts_sha256={n:sha(ROOT/n) for n in EXTRA if n.endswith('.py')})
    write_new(output/'ENGINEERING_RECEIPT.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',nargs='?',choices=('freeze','deploy','start','status','verify-start','collect','collect-inspect','self-check'))
    parser.add_argument('--repo',type=Path,default=ROOT)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--backup',type=Path)
    parser.add_argument('--collected',type=Path)
    parser.add_argument('--_remote')
    args=parser.parse_args()
    if args._remote is not None:
        require(args.mode is None,'Native internal mode cannot mix local actions')
        payload=json.loads(base64.urlsafe_b64decode(args._remote.encode()))
        print(json.dumps(remote_dispatch(payload),allow_nan=False));return
    require(args.mode is not None and args.output is not None,'Explicit mode and fresh --output required')
    if args.mode == 'self-check':
        result=self_check(args.output)
    elif args.mode == 'freeze':
        require(args.backup is not None,'Explicit D small-source --backup required')
        result=freeze_sources(args.repo,args.output,args.backup)
    else:
        require(args.freeze is not None,'Explicit reviewed --freeze directory required')
        result=controller_action(args.mode,args.freeze,args.output,args.collected)
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    main()
