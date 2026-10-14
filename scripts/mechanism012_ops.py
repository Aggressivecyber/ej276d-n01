"""Stdlib-only freeze/deploy/dispatch/readback for bounded mechanism012.

Never performs numeric decoding, checkpoint loading, NN/AD or optimization.
All scientific dispatches are explicit, exclusive and never automatically retried.
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

ROOT=Path(__file__).resolve().parents[1] if not str(__file__).startswith('<') else Path.cwd()
HOST='rylan@100.82.231.102'
PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
REMOTE='/home/rylan/projects/ej276d-n01/mechanism012_20261015'
OLD010='/home/rylan/projects/ej276d-n01/shape010_20261014'
OLD011='/home/rylan/projects/ej276d-n01/stability011_20261014'
OLD008='/home/rylan/projects/ej276d-n01/window008_resume01_20261013'
RUN_ID='mechanism012_20261015'
MANIFEST,BINDINGS,POLICY='MECHANISM012_SOURCE_MANIFEST.json','MECHANISM012_BINDINGS.json','mechanism012_policy.json'
EXTRA=('scripts/mechanism012_ops.py','scripts/supervise_mechanism012.py','scripts/run_mechanism012.py',
       'scripts/mechanism012_losses.py',POLICY,'docs/MECHANISM012_PROTOCOL.md')
CONTROL_COPIES=dict(control_noise_acf005_manifest='NOISE_ACF005_SOURCE_MANIFEST.json',
    control_integration006_manifest='INTEGRATION006_SOURCE_MANIFEST.json',
    control_integration006_binding_fix01_manifest='INTEGRATION006_BINDING_FIX01_SOURCE_MANIFEST.json',
    control_integration006_id_fix02_manifest='INTEGRATION006_ID_FIX02_SOURCE_MANIFEST.json',
    control_window008_manifest='WINDOW008_SOURCE_MANIFEST.json',control_window008_fix01_manifest='WINDOW008_FIX01_SOURCE_MANIFEST.json')
OLD_INPUTS=dict(dataset='data/window008/DATASET008.npz',statistics='data/window008/FIT_STATISTICS008.json',
    data_statistics='data/window008/FIT_DATA_STATISTICS.json',noise005='provenance/integration006_reference/noise005.pt',
    deterministic='provenance/window008_fix01_reference/deterministic_final.pt',base_config='window008_config.json',
    training_complete='runs/science/TRAINING_COMPLETE.json')
SHAPE_INPUTS=dict(head_initial='runs/primary/artifacts/initial.pt',head_1000='runs/primary/artifacts/periodic_01000.pt',
    head_final='runs/primary/artifacts/final.pt',shape010_manifest='SHAPE010_SOURCE_MANIFEST.json',
    shape010_bindings='SHAPE010_BINDINGS.json',shape010_complete='runs/primary/artifacts/COMPLETE.json',
    shape010_initial_eval='runs/primary/artifacts/INITIAL_EVALUATION.json',shape010_final_eval='runs/primary/artifacts/FINAL_EVALUATION.json',
    shape010_eval_bank='runs/primary/artifacts/EVALUATION_BANK.npz',shape010_binding_receipt='runs/primary/artifacts/BINDING.json',
    shape010_native_exit='runs/primary/supervisor/exit_receipt.json')
for _v in ('frozen_A','shape010_final'):
    for _h in ('fast','slow','common_level'):
        _k=_v+'_'+_h+'_TRAIN';SHAPE_INPUTS[_k]='runs/primary/artifacts/'+_k+'.npz'
STABILITY_INPUTS=dict(stability011_manifest='STABILITY011_SOURCE_MANIFEST.json',stability011_bindings='STABILITY011_BINDINGS.json',
    stability011_complete='runs/primary/artifacts/COMPLETE.json',stability011_result='runs/primary/artifacts/STABILITY_RESULT.json',
    stability011_native_exit='runs/primary/supervisor/exit_receipt.json',stability011_frozen_inputs='runs/primary/artifacts/FROZEN_INPUTS.json')
INPUT_PATHS={**{k:OLD008+'/'+v for k,v in OLD_INPUTS.items()},**{k:OLD010+'/'+v for k,v in SHAPE_INPUTS.items()},
    **{k:OLD011+'/'+v for k,v in STABILITY_INPUTS.items()},**{k:REMOTE+'/'+v for k,v in CONTROL_COPIES.items()}}
BUDGET=dict(old_A_rows=2048,noise005_rows=2048,head_final_rows=2048,head_initial_or1000_rows=0,
    scalar_AD_queries=9,masked_endpoint_VJP_queries=2,ADC_hard_or_STE_rows=49152,checkpoint_loads_max=5,
    optimizer_created=0,optimizer_steps=0,G_updates=0,D_updates=0,noise_updates=0,population_updates=0,
    C2ST_updates=0,DEV_rows=0,sampler_created=0,new_latent_draws=0,new_noise_draws=0)
ZERO_COUNTERS=tuple(k for k,v in BUDGET.items() if v == 0)
COUNTER_CAPS={('checkpoint_loads' if k=='checkpoint_loads_max' else k):v for k,v in BUDGET.items()}
RESOURCES=dict(CPU_threads=12,GPU_GiB=10,job_tree_RSS_GiB=8,wall_seconds=900,output_MiB=128,poll_seconds=10,TERM_grace_seconds=20)
CONTROLS=('START_RECEIPT.json','REMOTE_SOURCE_INSTALL_RECEIPT.json','DISPATCH_ATTEMPT.json',POLICY,MANIFEST,BINDINGS)

def require(value,message):
    if not value:raise ValueError(message)

def now():return datetime.now(timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def sha_bytes(value):return hashlib.sha256(value).hexdigest()

def valid_sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def read_json(path):
    p=Path(path);require(not p.is_symlink() and p.stat().st_size<=16*1024**2,'Invalid JSON path/size')
    return json.loads(p.read_text(encoding='utf8'))

def write_new(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def safe_name(name):
    require(isinstance(name,str) and name and not name.startswith('/') and '\\' not in name and ':' not in name
        and all(ord(c)>=32 and ord(c)!=127 for c in name),'Unsafe archive/source name')
    parts=name.split('/')
    require(all(p not in ('','.','..') and p==p.rstrip(' .') and len(p)<=255 and
        re.fullmatch('CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]',p.split('.')[0],re.IGNORECASE) is None for p in parts)
        and PurePosixPath(name).as_posix()==name,'Noncanonical member name')
    return name

def unredirected(path):
    p=Path(path).absolute()
    require(all(not q.is_symlink() for q in (p,*p.parents)) and p.resolve()==p,'Redirected path rejected: '+str(p))
    return p

def fresh_directory(path,local=False):
    p=unredirected(path)
    require(not local or os.name!='nt' or p.drive.upper() in ('E:','D:'),'Local output must stay on E/D')
    require(not os.path.lexists(p),'Preserve existing directory: '+str(p))
    p.mkdir(parents=True,mode=0o700,exist_ok=False);return p

def validate_policy(p):
    require(p.get('schema')=='n01/mechanism012-policy/v1' and p.get('run_id')==RUN_ID,'Wrong012 identity')
    require((p.get('remote_root'),p.get('shape010_remote_root'),p.get('stability011_remote_root'),p.get('reference_remote_root'))
        ==(REMOTE,OLD010,OLD011,OLD008),'Wrong012 roots')
    require(p.get('budgets')==BUDGET and p.get('resource_limits')==RESOURCES,'Wrong012 bounded computation/resources')
    require(p.get('head_update')==2000 and p.get('metadata_only_head_updates')==[0,1000]
        and p.get('forward_chunk_rows')==128 and p.get('head_parameter_names')==['basis_head.weight','basis_head.bias']
        and p.get('head_parameter_count')==1288,'Wrong012 checkpoint/parameter contract')
    require(p.get('DEV_numeric_read') is False and p.get('automatic_restart') is False and p.get('overwrite_outputs') is False,'Wrong012 DEV/restart scope')
    require(p.get('source_file_count')==69 and p.get('source_parent_clusters')==512 and p.get('draws_per_parent')==4
        and p.get('matched_rows')==2048 and p.get('TRAIN_counts')==dict(source=6593,real=21047),'Wrong012 fixed cohort')
    require(p.get('allowed_dataset_members')==['source','source_carrier384','real','real_baseline100','source_ids','real_ids','real_groups','source_role','real_role']
        and p.get('allowed_bank')=='shape010_saved_EVALUATION_BANK_only','Wrong012 numerical read whitelist')
    require(p.get('preflight')==dict(minimum_available_RAM_GiB=10,minimum_free_disk_GiB=4,GPU_empty=True),'Wrong native preflight')
    require(len(p.get('autograd',{}).get('scalar_queries',[]))==9 and len(p['autograd'].get('masked_VJP_queries',[]))==2
        and p['autograd'].get('extra_pilot_queries')==0 and p['autograd'].get('backward_calls')==0,'Extra012 AD query prohibited')
    require(p.get('probe',{}).get('step_sizes_ADC_LSB')==[0.1,1,10] and p['probe'].get('signs')==[-1,1]
        and p['probe'].get('extra_NN_forward_rows')==0 and p['probe'].get('parameter_finite_difference') is False,'Wrong012 probes')

def validate_documents(m,b,p,ms):
    validate_policy(p)
    require(m.get('schema')=='n01/mechanism012-source/v1' and m.get('run_id')==RUN_ID
        and m.get('before_scientific_numeric_decode') is True,'Wrong012 source schema')
    require(b.get('schema')=='n01/mechanism012-bindings/v1' and b.get('run_id')==RUN_ID
        and len(b.get('roots',[]))==4 and set(b['roots'])=={REMOTE,OLD010,OLD011,OLD008}
        and b.get('source_manifest')==dict(path=REMOTE+'/'+MANIFEST,sha256=ms),'Wrong012 binding roots/manifest')
    files=m.get('files');require(isinstance(files,dict) and len(files)==69 and set(EXTRA)<=set(files),'Expected69 frozen source/metadata copies')
    require(m.get('old_shape010_source_file_count')==53 and len(m.get('protected_manifests',{}))==4
        and set(m.get('control_metadata_copies',{}))==set(CONTROL_COPIES.values()),'Wrong012 source/control chain')
    for n,d in files.items():safe_name(n);require(valid_sha(d),'Invalid source SHA')
    require(MANIFEST not in files and BINDINGS not in files,'Recursive source inventory')
    require(set(b.get('files',{}))==set(INPUT_PATHS) and m.get('bound_inputs')=={k:r['sha256'] for k,r in b['files'].items()},'Wrong36 input inventory')
    for k,r in b['files'].items():require(r==dict(path=INPUT_PATHS[k],sha256=m['bound_inputs'][k]) and valid_sha(r['sha256']),'Unbound input: '+k)
    require(m.get('policy_sha256')==b.get('policy_sha256')==files[POLICY] and m.get('budgets')==BUDGET,'Wrong policy/budget SHA chain')
    require(all(files.get(n)==d for n,d in {**m['protected_manifests'],**m['control_metadata_copies']}.items()),'Metadata copy SHA differs')
    require(m.get('shape010_source_manifest_sha256')==b['files']['shape010_manifest']['sha256']
        and m.get('stability011_source_manifest_sha256')==b['files']['stability011_manifest']['sha256'],'Prior source chain differs')
    fact=dict(sha256=p['prior_stability_required']['independent_readback_sha256'],status='PASS',scope='root_verified_local_receipt')
    require(b.get('prior_independent_readback')==m.get('prior_independent_readback')==fact,'Unregistered prior independent readback')
    return files

def tar_index(path,maximum_bytes):
    result,total={},0
    with tarfile.open(path,'r:*') as tar:
        for entry in tar:
            n=safe_name(entry.name);require(n not in result and entry.isfile() and not entry.issym() and not entry.islnk(),'Duplicate/nonregular tar member')
            total+=entry.size;require(entry.size>=0 and total<=maximum_bytes and len(result)<512,'Tar resource budget exceeded')
            stream=tar.extractfile(entry);require(stream is not None,'Missing tar stream');h,count=hashlib.sha256(),0
            for block in iter(lambda:stream.read(1024**2),b''):count+=len(block);h.update(block)
            require(count==entry.size,'Truncated tar member');result[n]=dict(bytes=count,sha256=h.hexdigest())
    return result

def package_payload(path,archive_sha,ms,bs):
    require(all(valid_sha(v) for v in (archive_sha,ms,bs)) and sha(path)==archive_sha,'Archive SHA differs')
    index=tar_index(path,8*1024**2)
    require({MANIFEST,BINDINGS,POLICY}<=set(index) and index[MANIFEST]['sha256']==ms and index[BINDINGS]['sha256']==bs,'Missing/changed package controls')
    with tarfile.open(path,'r:*') as tar:contents={n:tar.extractfile(n).read() for n in index}
    m,b,p=[json.loads(contents[n]) for n in (MANIFEST,BINDINGS,POLICY)]
    files=validate_documents(m,b,p,ms)
    require(set(index)==set(files)|{MANIFEST,BINDINGS} and all(index[n]['sha256']==d for n,d in files.items()),'Packaged source inventory/SHA differs')
    return dict(contents=contents,manifest=m,bindings=b,policy=p,index=index,archive_sha256=archive_sha,manifest_sha256=ms,bindings_sha256=bs)

def exclusive_extract(path,target,inventory,maximum_bytes):
    require(tar_index(path,maximum_bytes)==inventory,'Archive inventory differs before extraction')
    target=fresh_directory(target,local=True)
    with tarfile.open(path,'r:*') as tar:
        for n in sorted(inventory):
            p=target/n;p.parent.mkdir(parents=True,exist_ok=True);require(unredirected(p).is_relative_to(target),'Extraction escaped root')
            with p.open('xb') as f:shutil.copyfileobj(tar.extractfile(n),f,1024**2)
            require(p.stat().st_size==inventory[n]['bytes'] and sha(p)==inventory[n]['sha256'],'Extracted bytes differ')
    return target

def verify_paths(paths):
    for path,digest in paths.items():require(sha(unredirected(path))==digest,'Frozen bytes changed: '+path)
    return len(paths)

def add_path(paths,path,digest):
    path=str(unredirected(path));require(valid_sha(digest),'Invalid pinned SHA')
    require(path not in paths or paths[path]==digest,'Conflicting SHA for one actual path')
    paths[path]=digest

def operation_progress(output):
    p=Path(output)/'operations.jsonl';entered,completed,failed={},{},{};truncated=False
    if p.exists():
        require(not p.is_symlink() and p.stat().st_size<=128*1024**2,'Invalid durable operations journal')
        with p.open('r',encoding='utf8') as f:
            for line in f:
                if not line.endswith('\n'):truncated=True;break
                try:r=json.loads(line)
                except json.JSONDecodeError:truncated=True;break
                oid,event,charges=r.get('operation_id'),r.get('event'),r.get('charges')
                require(type(oid) is int and oid>=0 and event in ('entered','completed','failed') and isinstance(charges,dict),'Invalid durable operation record')
                require(all(k in COUNTER_CAPS and type(v) is int and v>=0 for k,v in charges.items()),'Invalid durable charges')
                target={'entered':entered,'completed':completed,'failed':failed}[event]
                require(oid not in target,'Duplicate durable operation event');target[oid]=r
    require(not set(completed)&set(failed) and set(completed)|set(failed)<=set(entered)
        and all(r['charges']==entered[i]['charges'] for records in (completed,failed) for i,r in records.items()),'Operation lacks matching enter')
    sums=lambda records:{k:sum(r['charges'].get(k,0) for r in records.values()) for k in COUNTER_CAPS}
    d=dict(entered_counts_durable_lower_bound=sums(entered),completed_counts_durable_lower_bound=sums(completed),
        entered_operation_ids=sorted(entered),completed_operation_ids=sorted(completed),unknown_inflight_operation_ids=sorted(set(entered)-set(completed)),
        failed_operation_ids=sorted(failed),failed_records=[failed[i] for i in sorted(failed)],
        journal_truncated_tail=truncated,actual_unjournaled_or_inflight_completion='unknown; never asserted complete')
    p=Path(output)/'status.json'
    d.update(last_persisted_status=read_json(p) if p.exists() else None)
    return d

def boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip() if sys.platform.startswith('linux') else 'artificial_non_native'

def local_input_paths(repo):
    shape=repo/'runs/shape010_20261014/collected_01';stable=repo/'runs/stability011_20261014/collected_01'
    paths={k:repo/v for k,v in OLD_INPUTS.items()}
    paths['training_complete']=repo/'runs/window008_resume01_20261013/runs/science/TRAINING_COMPLETE.json'
    paths.update({k:(repo/v if '/' not in v else shape/v) for k,v in SHAPE_INPUTS.items()})
    paths.update({k:(repo/v if '/' not in v else stable/v) for k,v in STABILITY_INPUTS.items()})
    paths.update({k:repo/v for k,v in CONTROL_COPIES.items()});return paths

def clean_exit(native):
    require(native.get('child_native_exit_code')==native.get('exit_code')==0 and native.get('reason') is None
        and native.get('error') is None,'Prior native job did not finish cleanly')

def old_chain(m,b,count,run,root,manifest_sha):
    require(m.get('run_id')==b.get('run_id')==run and len(m.get('files',{}))==count
        and b.get('source_manifest')==dict(path=root+'/'+('SHAPE010_SOURCE_MANIFEST.json' if count==53 else 'STABILITY011_SOURCE_MANIFEST.json'),sha256=manifest_sha)
        and m.get('bound_inputs')=={k:r['sha256'] for k,r in b['files'].items()}
        and m.get('policy_sha256')==b.get('policy_sha256')==m['files']['shape010_policy.json' if count==53 else 'stability011_policy.json'],
        'Prior manifest/binding chain differs')
    for n,d in m['files'].items():safe_name(n);require(valid_sha(d),'Invalid prior SHA')

def verify_prior_results(paths,policy,engineering_fixture=False):
    c=read_json(paths['shape010_complete']);clean_exit(read_json(paths['shape010_native_exit']))
    require(c.get('run_id')=='shape010_20261014' and c.get('G_updates')==2000 and c.get('frozen_A_005_unchanged') is True,'Wrong completed010')
    stable=read_json(paths['stability011_complete']);result=read_json(paths['stability011_result'])
    clean_exit(read_json(paths['stability011_native_exit']))
    require(stable.get('run_id')==result.get('run_id')=='stability011_20261014'
        and stable.get('artificial_only') is engineering_fixture and stable.get('bootstrap_replicates_completed')==500
        and stable.get('bootstrap_replicates_attempted')==500 and stable.get('point_checks_passed') is True
        and stable.get('classification')==result.get('classification')==policy['prior_stability_required']['classification'],
        'Prior500 stable result not established')
    return c,stable

def freeze_sources(repo,output,backup,engineering_fixture=False):
    repo=unredirected(repo);p=read_json(repo/POLICY);validate_policy(p)
    oldm,oldb=[read_json(repo/n) for n in ('SHAPE010_SOURCE_MANIFEST.json','SHAPE010_BINDINGS.json')]
    sm,sb=[read_json(repo/n) for n in ('STABILITY011_SOURCE_MANIFEST.json','STABILITY011_BINDINGS.json')]
    old_chain(oldm,oldb,53,'shape010_20261014',OLD010,sha(repo/'SHAPE010_SOURCE_MANIFEST.json'))
    old_chain(sm,sb,63,'stability011_20261014',OLD011,sha(repo/'STABILITY011_SOURCE_MANIFEST.json'))
    protected=oldm['protected_manifests'];require(len(protected)==4 and sm['protected_manifests']==protected,'Wrong protected4 metadata')
    fixed={}
    for n,d in {**oldm['files'],**sm['files'],**protected}.items():safe_name(n);add_path(fixed,repo/n,d)
    # E originals and historical protected files are never copied wholesale.
    for n,d in protected.items():
        require(sha(unredirected(repo/n))==d,'Protected original manifest changed')
        for rel,digest in read_json(repo/n)['files'].items():safe_name(rel);add_path(fixed,repo/rel,digest)
    verify_paths(fixed)
    controls={n:sha(unredirected(repo/n)) for n in CONTROL_COPIES.values()}
    sources={**oldm['files'],**protected,**controls,**{n:sha(unredirected(repo/n)) for n in EXTRA}}
    require(len(sources)==69,'Expected exact69 source recipe')
    paths=local_input_paths(repo);inputs={k:sha(unredirected(v)) for k,v in paths.items()}
    require(all(inputs[k]==r['sha256'] for k,r in oldb['files'].items()),'Original7 immutable inputs differ')
    complete,stable=verify_prior_results(paths,p,engineering_fixture=engineering_fixture)
    for slot,c in (('shape010_complete',complete),('stability011_complete',stable)):
        for n,row in c['artifacts'].items():
            safe_name(n);require('/' not in n,'Nonflat prior inventory');q=unredirected(paths[slot].parent/n)
            require(q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Collected prior artifact changed: '+n)
    prior=repo/'receipts/stability011_20261014/user_server_pause_01/independent_readback_01/READBACK.json'
    require(sha(unredirected(prior))==p['prior_stability_required']['independent_readback_sha256']
        and read_json(prior).get('status')=='PASS','Prior independent readback differs')
    fact=dict(sha256=sha(prior),status='PASS',scope='root_verified_local_receipt')
    output=fresh_directory(output,local=True)
    m=dict(schema='n01/mechanism012-source/v1',run_id=RUN_ID,frozen_UTC=now(),before_scientific_numeric_decode=True,
        files=sources,policy_sha256=sources[POLICY],bound_inputs=inputs,budgets=BUDGET,old_shape010_source_file_count=53,
        shape010_source_manifest_sha256=inputs['shape010_manifest'],stability011_source_manifest_sha256=inputs['stability011_manifest'],
        protected_manifests=protected,control_metadata_copies=controls,prior_independent_readback=fact)
    write_new(output/MANIFEST,m)
    b=dict(schema='n01/mechanism012-bindings/v1',run_id=RUN_ID,roots=[REMOTE,OLD010,OLD011,OLD008],policy_sha256=sources[POLICY],
        source_manifest=dict(path=REMOTE+'/'+MANIFEST,sha256=sha(output/MANIFEST)),prior_independent_readback=fact,
        files={k:dict(path=INPUT_PATHS[k],sha256=d) for k,d in inputs.items()})
    write_new(output/BINDINGS,b);archive=output/'mechanism012_source.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for n in sorted(sources):tar.add(repo/n,arcname=n,recursive=False)
        for n in (MANIFEST,BINDINGS):tar.add(output/n,arcname=n,recursive=False)
    pack=package_payload(archive,sha(archive),sha(output/MANIFEST),sha(output/BINDINGS))
    require(engineering_fixture or os.name!='nt' or Path(backup).absolute().drive.upper()=='D:','Small source backup requires D')
    backup=fresh_directory(backup,local=True);require(shutil.disk_usage(backup).free>=archive.stat().st_size+256*1024**2,'Backup disk margin')
    with archive.open('rb') as src,(backup/archive.name).open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
    require(sha(backup/archive.name)==sha(archive),'Source backup differs')
    with (backup/'SHA256SUMS.txt').open('x',encoding='utf8') as f:f.write(sha(archive)+'  '+archive.name+'\n')
    receipt=dict(schema='n01/mechanism012-freeze/v1',UTC=now(),package=str(archive),package_sha256=sha(archive),
        package_bytes=archive.stat().st_size,source_manifest_sha256=pack['manifest_sha256'],bindings_sha256=pack['bindings_sha256'],
        policy_sha256=sources[POLICY],source_file_count=69,input_count=36,old53_and_01163_verified=True,
        protected4_original_and_historical_files_verified=True,control_metadata_copies=6,source_backup=str(backup/archive.name),
        D_backup_SHA256_verified=not engineering_fixture,artificial_only=engineering_fixture,scientific_started=False,
        numeric_decodes=0,checkpoint_loads=0,NN_rows=0,AD_queries=0,optimizer_steps=0)
    write_new(output/'FREEZE_RECEIPT.json',receipt);return receipt

def load_freeze(directory):
    directory=unredirected(directory);r=read_json(directory/'FREEZE_RECEIPT.json')
    pack=package_payload(directory/'mechanism012_source.tar.gz',r['package_sha256'],r['source_manifest_sha256'],r['bindings_sha256'])
    require(sha(Path(__file__))==pack['manifest']['files']['scripts/mechanism012_ops.py'],'Controller changed after freeze')
    require(r.get('artificial_only') is False and r.get('D_backup_SHA256_verified') is True,'Artificial freeze cannot dispatch')
    return r,pack

def verify_runtime(target,ms,bs):
    target=unredirected(target);require(str(target)==REMOTE and target.is_dir(),'Only installed012 root allowed')
    paths={};checked=set()
    def check(path,digest):
        add_path(paths,path,digest);key=str(unredirected(path))
        if key not in checked:require(sha(key)==digest,'Native frozen bytes changed: '+key);checked.add(key)
    check(target/MANIFEST,ms);check(target/BINDINGS,bs)
    m,b=[read_json(target/n) for n in (MANIFEST,BINDINGS)]
    check(target/POLICY,m['files'][POLICY]);p=read_json(target/POLICY);validate_documents(m,b,p,ms)
    sources={}
    for n,d in m['files'].items():check(target/n,d);sources[str(target/n)]=d
    for k,r in b['files'].items():require(r['path']==INPUT_PATHS[k],'Input path escaped slot');check(r['path'],r['sha256'])
    om,ob,sm,sb=[read_json(b['files'][k]['path']) for k in ('shape010_manifest','shape010_bindings','stability011_manifest','stability011_bindings')]
    old_chain(om,ob,53,'shape010_20261014',OLD010,b['files']['shape010_manifest']['sha256'])
    old_chain(sm,sb,63,'stability011_20261014',OLD011,b['files']['stability011_manifest']['sha256'])
    require(m['protected_manifests']==om['protected_manifests']==sm['protected_manifests'],'Protected metadata copies differ')
    for root,prior in ((OLD010,om),(OLD011,sm)):
        for n,d in prior['files'].items():
            check(Path(root)/n,d)
            if n in om['files']:require(m['files'].get(n)==d,'Original53 copied bytes changed')
    # Old010 never packaged protected4 metadata; their verified new copies suffice.
    for n,d in m['protected_manifests'].items():check(target/n,d)
    for n,d in m['control_metadata_copies'].items():check(target/n,d)
    for prior in (ob,sb):
        for r in prior['files'].values():
            q=unredirected(r['path']);require(any(q.is_relative_to(Path(root)) for root in (OLD008,OLD010)),'Prior input escaped readonly roots')
            check(q,r['sha256'])
    bc={k:r['path'] for k,r in b['files'].items()};c,s=verify_prior_results(bc,p)
    for root,comp in ((OLD010,c),(OLD011,s)):
        for n,r in comp['artifacts'].items():
            safe_name(n);require('/' not in n,'Nonflat prior artifact');q=Path(root)/'runs/primary/artifacts'/n
            require(unredirected(q).stat().st_size==r['bytes'],'Prior artifact size differs');check(q,r['sha256'])
    return m,b,p,paths,sources

def runtime_preflight(root,policy):
    root=unredirected(root)
    available=int(next(r.split()[1] for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith('MemAvailable:')))*1024
    disk,cpus=shutil.disk_usage(root).free,len(os.sched_getaffinity(0))
    require(available>=10*1024**3 and disk>=4*1024**3 and cpus>=12,'Insufficient012 native RAM/disk/CPU')
    markers=('-m n01.','scripts/supervise_window','scripts/run_window','scripts/supervise_shape010.py',
        'scripts/supervise_mean_random009.py','scripts/supervise_stability011.py','scripts/run_stability011.py',
        'scripts/supervise_mechanism012.py','scripts/run_mechanism012.py','scripts/supervise.py')
    active=[r for r in subprocess.check_output(['ps','-eo','pid,args'],text=True,timeout=10).splitlines() if any(v in r for v in markers)]
    require(not active,'Another N01 science process active')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,timeout=10).strip()
    require(not apps,'GPU science app active before012')
    memory=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    require(int(memory)>=10*1024,'GPU0 lacks10GiB free')
    return dict(boot_id=boot_id(),RAM_available_bytes=available,disk_free_bytes=disk,CPU_affinity_count=cpus,GPU0_free_MiB=int(memory),
        active_project_jobs_before=active,GPU_apps_before=apps)

def spawn_supervisor(command,root,fd,env):
    with (root/'supervisor_console.log').open('xb') as log:
        return subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=(fd,),env=env)

def native_start(ms,bs):
    require(os.name=='posix','Linux-only dispatch');import fcntl
    root=unredirected(REMOTE)
    require(not any(os.path.lexists(root/n) for n in ('runs/primary','START_RECEIPT.json','DISPATCH_ATTEMPT.json','supervisor_console.log','tmp','cache')),
        'Preserve prior012 dispatch; no retry/restart')
    m,b,p,_,_=verify_runtime(root,ms,bs)
    lock=unredirected(root.parent/'.n01_scientific_dispatch.lock');fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);preflight=runtime_preflight(root,p)
        require(Path(PYTHON).is_file(),'Registered Python missing')
        write_new(root/'DISPATCH_ATTEMPT.json',dict(run_id=RUN_ID,UTC=now(),source_manifest_sha256=ms,bindings_sha256=bs,
            preflight=preflight,budgets=BUDGET,automatic_restart=False))
        for n in ('tmp','cache'):unredirected(root/n).mkdir(mode=0o700,exist_ok=False)
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src'),TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
            CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12',OPENBLAS_NUM_THREADS='12',NUMEXPR_NUM_THREADS='12')
        command=[PYTHON,'-B',str(root/'scripts/supervise_mechanism012.py'),'--policy',str(root/POLICY),'--bindings',str(root/BINDINGS),
            '--job-dir',str(root/'runs/primary'),'--lock-fd',str(fd)]
        child=spawn_supervisor(command,root,fd,env)
        result=dict(schema='n01/mechanism012-start/v1',run_id=RUN_ID,UTC=now(),status='awaiting_formal_first_AD_proof',supervisor_pid=child.pid,
            command=command,exclusive_scientific_lock=str(lock),lock_inherited_by_supervisor=True,source_manifest_sha256=ms,bindings_sha256=bs,
            policy_sha256=m['policy_sha256'],budgets=BUDGET,automatic_restart=False,first_diagnostic_verified=False,**preflight)
        write_new(root/'START_RECEIPT.json',result);return result
    finally:os.close(fd)

STATE_NAMES=('START_RECEIPT.json','REMOTE_SOURCE_INSTALL_RECEIPT.json','DISPATCH_ATTEMPT.json',
    'runs/primary/supervisor/launch.json','runs/primary/supervisor/resource_status.json','runs/primary/supervisor/exit_receipt.json',
    'runs/primary/artifacts/status.json','runs/primary/artifacts/FIRST_DIAGNOSTIC_PROOF.json',
    'runs/primary/artifacts/FAILED.json','runs/primary/artifacts/COMPLETE.json')

def native_status(ms,bs,preserve_partial=False):
    root=unredirected(REMOTE);errors=[]
    if preserve_partial:
        require(sha(root/MANIFEST)==ms and sha(root/BINDINGS)==bs,'Installed012 controls changed')
        m,b,p=[read_json(root/n) for n in (MANIFEST,BINDINGS,POLICY)];validate_documents(m,b,p,ms)
        try:verify_runtime(root,ms,bs)
        except Exception as error:errors.append(repr(error))
    else:m,b,p,_,_=verify_runtime(root,ms,bs)
    result={}
    for n in STATE_NAMES:
        q=unredirected(root/n)
        try:result[n]=read_json(q) if q.exists() else None
        except Exception as error:
            if not preserve_partial:raise
            result[n]=None;errors.append(n+': '+repr(error))
    console=unredirected(root/'runs/primary/console.log')
    if console.exists():
        with console.open('rb') as f:f.seek(max(0,console.stat().st_size-4000));result['console_tail']=f.read().decode('utf8',errors='replace')
    try:progress=operation_progress(root/'runs/primary/artifacts')
    except Exception as error:
        if not preserve_partial:raise
        progress=dict(progress_read_error=repr(error),actual_completion='unknown');errors.append(repr(error))
    result.update(UTC=now(),current_boot_id=boot_id(),durable_child_progress=progress,protection_errors=errors,
        partial_evidence_only=preserve_partial,
        _verified_freeze=dict(source_manifest_sha256=ms,bindings_sha256=bs,policy_sha256=m['policy_sha256'],
            bound_inputs={r['path']:r['sha256'] for r in b['files'].values()},new_sources={REMOTE+'/'+n:d for n,d in m['files'].items()}))
    return result

def validate_counters(attempted,completed):
    require(isinstance(attempted,dict) and isinstance(completed,dict) and set(attempted)==set(completed)==set(COUNTER_CAPS),'Incomplete operation counters')
    require(all(type(completed[k]) is int and type(attempted[k]) is int and 0<=completed[k]<=attempted[k]<=cap
        for k,cap in COUNTER_CAPS.items()),'Invalid/overbudget operation counts')

def proof_hashes(proof,state):
    claimed={**proof.get('inputs_sha256',{}),**proof.get('sources_sha256',{})};freeze=state.get('_verified_freeze',{})
    required={**freeze.get('bound_inputs',{}),**freeze.get('new_sources',{})}
    require(len(freeze.get('new_sources',{}))==69 and len(freeze.get('bound_inputs',{}))>=30
        and all(claimed.get(n)==d for n,d in required.items()),'Proof input/source hashes differ')

def verify_first(state):
    proof=state.get('runs/primary/artifacts/FIRST_DIAGNOSTIC_PROOF.json')
    if proof is None:return dict(status='awaiting_formal_first_AD_proof',first_diagnostic_verified=False)
    start,launch=state.get('START_RECEIPT.json'),state.get('runs/primary/supervisor/launch.json')
    require(isinstance(start,dict) and isinstance(launch,dict) and proof.get('run_id')==RUN_ID
        and start.get('supervisor_pid')==launch.get('supervisor_pid') and launch.get('child_pid')==proof.get('pid'), '012 proof/PID chain differs')
    validate_counters(proof.get('attempted_counters'),proof.get('counters'));c=proof['counters']
    require(proof.get('artificial_only') is False and proof.get('point_checks_passed') is True
        and all(c[k]==2048 for k in ('old_A_rows','noise005_rows','head_final_rows')) and c['scalar_AD_queries']>=1
        and proof.get('finite_formal_AD') is True and proof.get('inputs_unchanged') is True
        and proof.get('frozen_state_unchanged') is True and proof.get('head_parameter_names')==['basis_head.weight','basis_head.bias']
        and proof.get('head_parameter_count')==1288,'First proof lacks actual complete formal NN/globalAD work')
    require(not state.get('protection_errors'),'Protected bytes failed readback');proof_hashes(proof,state)
    return dict(status='formal_forward_and_first_global_AD_verified',first_diagnostic_verified=True,child_pid=proof['pid'],
        supervisor_pid=start['supervisor_pid'],completed_scalar_AD_queries=c['scalar_AD_queries'],NN_rows={k:c[k] for k in ('old_A_rows','noise005_rows','head_final_rows')},
        optimizer_steps=0,scientific_status='response_match_not_established')

def verify_completed(state):
    first=verify_first(state);require(first['first_diagnostic_verified'],'No formal first012 proof')
    native=state.get('runs/primary/supervisor/exit_receipt.json');require(isinstance(native,dict),'Missing native012 exit')
    clean_exit(native);require(native.get('registered_budgets')==BUDGET,'Supervisor registered budget differs')
    c,s=state.get('runs/primary/artifacts/COMPLETE.json'),state.get('runs/primary/artifacts/status.json')
    require(isinstance(c,dict) and isinstance(s,dict) and c.get('run_id')==RUN_ID and c.get('artificial_only') is False
        and c.get('point_checks_passed') is True and c.get('engineering_checks_passed') is True
        and c.get('inputs_unchanged') is True and c.get('frozen_state_unchanged') is True,'012 engineering completion incomplete')
    validate_counters(c.get('attempted_counters'),c.get('counters'));counts=c['counters']
    require(c['attempted_counters']==counts and not c.get('inflight') and not c.get('failed_operations')
        and counts['scalar_AD_queries']==9 and counts['masked_endpoint_VJP_queries']==2
        and all(counts[k]==2048 for k in ('old_A_rows','noise005_rows','head_final_rows')),'Mandatory012 operation budget incomplete')
    direction=c.get('direction_defined');require(type(direction) is bool and counts['ADC_hard_or_STE_rows']==(49152 if direction else 12288), '012 ADC/zero-direction budget differs')
    require(state.get('runs/primary/artifacts/FAILED.json') is None and c['counters']==s.get('counters')
        and c.get('pid')==s.get('pid')==native.get('child_pid')==first['child_pid']
        and c.get('inputs_sha256')==s.get('inputs_sha256') and c.get('sources_sha256')==s.get('sources_sha256'), '012 terminal state/PID/hash mismatch')
    proof_hashes(c,state)
    progress=state.get('durable_child_progress')
    if progress is not None:
        require(not progress.get('journal_truncated_tail') and not progress.get('unknown_inflight_operation_ids')
            and not progress.get('failed_operation_ids') and progress.get('completed_counts_durable_lower_bound')==counts
            and progress.get('entered_counts_durable_lower_bound')==counts,'Terminal counters differ from durable journal')
    inventory=c.get('artifacts');require(isinstance(inventory,dict) and inventory,'Missing012 artifacts')
    for n,r in inventory.items():
        safe_name(n);require('/' not in n and not n.endswith(('.pt','.pth')) and 'FAIL' not in n.upper() and n!='COMPLETE.json'
            and set(r)=={'bytes','sha256'} and type(r['bytes']) is int and r['bytes']>=0 and valid_sha(r['sha256']),'Invalid012 artifact inventory')
    return dict(status='native_zero_and_registered_mechanism_budget_verified',child_pid=c['pid'],supervisor_pid=native['supervisor_pid'],
        counters=counts,artifact_count=len(inventory),direction_defined=direction,scientific_status='response_match_not_established',formal_response_acceptance=False)

def collection_inventory(root,state,partial=False):
    if partial:
        # A stopped job may have no Python FAILED after KILL/reboot. Never repair it.
        native=state.get('runs/primary/supervisor/exit_receipt.json');start=state.get('START_RECEIPT.json')
        if native is None:
            require(isinstance(start,dict),'No owned dispatch to collect')
            if start.get('boot_id')==state.get('current_boot_id'):
                for pid in (start.get('supervisor_pid'),(state.get('runs/primary/supervisor/launch.json') or {}).get('child_pid')):
                    require(type(pid) is not int or not Path('/proc',str(pid)).exists(),'Owned012 process still alive; collect status only')
        verified=dict(status='partial_evidence_only',native=native,durable_progress=state.get('durable_child_progress'),
            protection_errors=state.get('protection_errors'),actual_inflight_completion='unknown',scientific_status='response_match_not_established')
        names=[n for n in CONTROLS if (root/n).is_file()]
        if (root/'supervisor_console.log').is_file():names.append('supervisor_console.log')
        for directory,folders,files in os.walk(root/'runs/primary',followlinks=False):
            require(not any((Path(directory)/n).is_symlink() for n in folders+files),'Redirected partial output')
            for n in files:names.append((Path(directory)/n).relative_to(root).as_posix())
    else:
        verified=verify_completed(state);c=state['runs/primary/artifacts/COMPLETE.json']
        names=list(CONTROLS)+['runs/primary/console.log','runs/primary/supervisor/launch.json','runs/primary/supervisor/resource_status.json',
            'runs/primary/supervisor/exit_receipt.json','runs/primary/artifacts/COMPLETE.json']+['runs/primary/artifacts/'+n for n in c['artifacts']]
    inventory={}
    for n in sorted(set(names)):
        safe_name(n);q=unredirected(root/n);require(q.is_file() and not n.endswith(('.pt','.pth')),'Invalid collected owned output')
        inventory[n]=dict(bytes=q.stat().st_size,sha256=sha(q))
    if not partial:
        for n,r in c['artifacts'].items():require(inventory['runs/primary/artifacts/'+n]==r,'012 saved output changed: '+n)
    require(sum(r['bytes'] for r in inventory.values())<=144*1024**2,'Collection bytes exceed bounded margin')
    return inventory,verified

def remote_dispatch(payload):
    mode=payload.get('mode');require(os.name=='posix' and mode in ('prepare','install','start','status','collect','collect-inspect','collect-failed','collect-failed-inspect'),'Invalid native012 mode')
    require(all(valid_sha(payload.get(k)) for k in ('archive_sha256','manifest_sha256','bindings_sha256','controller_sha256')),'Invalid native SHA pins')
    ms,bs=payload['manifest_sha256'],payload['bindings_sha256'];root=unredirected(REMOTE)
    if mode in ('prepare','install'):
        token=payload.get('token');require(re.fullmatch('[0-9a-f]{32}',str(token)) is not None,'Invalid incoming token')
        incoming=root.parent/('.mechanism012.incoming_'+token);require(not os.path.lexists(root),'Preserve existing012 root')
        if mode=='prepare':
            incoming=fresh_directory(incoming);return dict(status='private_incoming_prepared',incoming=str(incoming),archive=str(incoming/'source.tar.gz'),target_created=False)
        pack=package_payload(unredirected(incoming/'source.tar.gz'),payload['archive_sha256'],ms,bs)
        require(pack['manifest']['files']['scripts/mechanism012_ops.py']==payload['controller_sha256'],'Native controller differs')
        exclusive_extract(incoming/'source.tar.gz',root,pack['index'],8*1024**2)
        m,b,p,paths,_=verify_runtime(root,ms,bs)
        result=dict(schema='n01/mechanism012-install/v1',run_id=RUN_ID,UTC=now(),status='source_installed_and_readonly_chains_verified',
            source_files=69,input_slots=36,source_manifest_sha256=ms,bindings_sha256=bs,source_archive_sha256=payload['archive_sha256'],
            original_verified_paths=len(paths),scientific_started=False,NN_rows=0,AD_queries=0,checkpoint_loads=0)
        write_new(root/'REMOTE_SOURCE_INSTALL_RECEIPT.json',result);return result
    require(root.is_dir() and sha(root/'scripts/mechanism012_ops.py')==payload['controller_sha256'],'Installed012 controller differs')
    if mode=='start':return native_start(ms,bs)
    partial=mode.startswith('collect-failed');state=native_status(ms,bs,preserve_partial=partial)
    if mode=='status':return state
    inventory,verified=collection_inventory(root,state,partial);archive=root/('mechanism012_partial_01.tar' if partial else 'mechanism012_complete_01.tar')
    if not mode.endswith('inspect'):
        require(not os.path.lexists(archive),'Preserve existing collection archive; inspect only')
        with tarfile.open(archive,'x:') as tar:
            for n in sorted(inventory):tar.add(root/n,arcname=n,recursive=False)
    else:require(archive.is_file() and not archive.is_symlink(),'No prior collection archive')
    require(tar_index(archive,144*1024**2)==inventory,'Native collected bytes differ')
    return dict(schema='n01/mechanism012-native-archive/v1',UTC=now(),archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        inventory=inventory,job=verified,partial_evidence_only=partial,source_manifest_sha256=ms,bindings_sha256=bs)

def ssh_action(payload,output):
    code=Path(__file__).read_text(encoding='utf8');require(sha_bytes(code.encode())==payload['controller_sha256'],'Controller changed before SSH')
    encoded=base64.urlsafe_b64encode(json.dumps(payload,allow_nan=False).encode()).decode()
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PYTHON+' -B - --_remote '+encoded]
    r=subprocess.run(command,input=code,text=True,capture_output=True,timeout=240)
    write_new(output/(payload['mode'].replace('-','_')+'_NATIVE_RECEIPT.json'),dict(command=command,native_ssh_exit_code=r.returncode,
        stdout=r.stdout,stderr=r.stderr,mode=payload['mode'],UTC=now()))
    require(r.returncode==0,'Native012 action failed; full receipt preserved');return json.loads(r.stdout)

def controller_action(mode,freeze,output,collected=None):
    receipt,pack=load_freeze(freeze);output=fresh_directory(output,local=True)
    payload=dict(mode=mode,archive_sha256=receipt['package_sha256'],manifest_sha256=receipt['source_manifest_sha256'],
        bindings_sha256=receipt['bindings_sha256'],controller_sha256=pack['manifest']['files']['scripts/mechanism012_ops.py'])
    if mode=='deploy':
        payload.update(mode='prepare',token=uuid.uuid4().hex);prepared=ssh_action(payload,output)
        cmd=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',str(Path(freeze)/'mechanism012_source.tar.gz'),HOST+':'+prepared['archive']]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        write_new(output/'TRANSFER_RECEIPT.json',dict(command=cmd,native_scp_exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr));require(r.returncode==0,'SCP failed; preserve incoming')
        payload['mode']='install';result=ssh_action(payload,output);write_new(output/'DEPLOYMENT_RECEIPT.json',result);return result
    if mode in ('status','verify-start'):
        payload['mode']='status';result=ssh_action(payload,output);write_new(output/'REMOTE_STATE.json',result)
        if mode=='verify-start':
            verification=verify_first(result);write_new(output/'START_VALIDATION.json',verification);return verification
        return dict(status='remote_state_saved',receipt=str(output/'REMOTE_STATE.json'),receipt_sha256=sha(output/'REMOTE_STATE.json'),
            durable_counters=result.get('durable_child_progress',{}).get('completed_counts_durable_lower_bound'),native_exit=result.get('runs/primary/supervisor/exit_receipt.json',{}))
    if mode=='start':
        result=ssh_action(payload,output);write_new(output/'START_RECEIPT.json',result);return result
    require(mode in ('collect','collect-inspect','collect-failed','collect-failed-inspect') and collected is not None,'Explicit fresh collection required')
    collected=unredirected(collected);require(not os.path.lexists(collected),'Preserve existing local collection')
    result=ssh_action(payload,output);write_new(output/'REMOTE_ARCHIVE_RECEIPT.json',result)
    archive=output/Path(result['archive']).name;part=archive.with_name(archive.name+'.part')
    cmd=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST+':'+result['archive'],str(part)]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    write_new(output/'TRANSFER_RECEIPT.json',dict(command=cmd,native_scp_exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
    require(r.returncode==0 and part.stat().st_size==result['archive_bytes'] and sha(part)==result['archive_sha256'],'Downloaded012 archive differs')
    part.replace(archive);exclusive_extract(archive,collected,result['inventory'],144*1024**2)
    if not result['partial_evidence_only']:
        state={n:read_json(collected/n) if (collected/n).is_file() else None for n in STATE_NAMES}
        state['_verified_freeze']=dict(bound_inputs={r['path']:r['sha256'] for r in pack['bindings']['files'].values()},new_sources={REMOTE+'/'+n:d for n,d in pack['manifest']['files'].items()})
        state['durable_child_progress']=operation_progress(collected/'runs/primary/artifacts');verify_completed(state)
    final=dict(schema='n01/mechanism012-collection/v1',status='partial_evidence_preserved' if result['partial_evidence_only'] else 'native_output_bytes_exclusively_verified',UTC=now(),
        archive=str(archive),archive_sha256=sha(archive),collected_root=str(collected),file_count=len(result['inventory']),job=result['job'],
        partial_evidence_only=result['partial_evidence_only'],numeric_decodes=0,checkpoint_loads=0,NN_rows=0,AD_queries=0,optimizer_steps=0,
        automatic_restart=False,source_manifest_sha256=receipt['source_manifest_sha256'],bindings_sha256=receipt['bindings_sha256'])
    write_new(output/'COLLECTION_RECEIPT.json',final);return final

def self_check(output):
    output=fresh_directory(output,local=True);checks=[]
    def passed(n):checks.append(dict(name=n,passed=True))
    def reject(n,function):
        try:function()
        except (ValueError,FileExistsError):passed(n);return
        raise AssertionError('Unsafe artificial contract accepted: '+n)
    p=read_json(ROOT/POLICY);validate_policy(p)
    for key,value in (('DEV_numeric_read',True),('automatic_restart',True),('head_update',1000)):
        changed=json.loads(json.dumps(p));changed[key]=value;reject('policy_'+key,lambda:validate_policy(changed))
    changed=json.loads(json.dumps(p));changed['budgets']['scalar_AD_queries']=10;reject('extra_AD_budget',lambda:validate_policy(changed))
    changed=json.loads(json.dumps(p));changed['allowed_dataset_members'].append('DEV');reject('DEV_decode_member',lambda:validate_policy(changed))
    for n in ('../outside','/absolute','a//b','a/./b','a\\b','C:/escape','NUL','a./file'):
        reject('unsafe_name_'+n,lambda n=n:safe_name(n))
    repo=fresh_directory(output/'artificial_repo',local=True)
    def put(n,data):
        q=repo/n;q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('xb') as f:f.write(data)
        return sha(q)
    old={};protected={}
    for i in range(52):old['src/n01/opaque_%02d.py'%i]=put('src/n01/opaque_%02d.py'%i,b'# synthetic unused source\n')
    old['shape010_policy.json']=put('shape010_policy.json',b'{"artificial_only":true}\n')
    for i in range(4):n='PROTECTED_%d.json'%i;protected[n]=put(n,b'{"files":{},"artificial_only":true}\n')
    for n in CONTROL_COPIES.values():put(n,b'{"files":{},"artificial_only":true}\n')
    paths=local_input_paths(repo)
    for k in OLD_INPUTS:put(paths[k].relative_to(repo).as_posix(),b'opaque artificial input; no decode/load')
    old_inputs={k:dict(path=INPUT_PATHS[k],sha256=sha(paths[k])) for k in OLD_INPUTS}
    om=dict(schema='n01/shape010-source/v1',run_id='shape010_20261014',files=old,protected_manifests=protected,
        policy_sha256=old['shape010_policy.json'],bound_inputs={k:r['sha256'] for k,r in old_inputs.items()})
    write_new(repo/'SHAPE010_SOURCE_MANIFEST.json',om)
    ob=dict(schema='n01/shape010-bindings/v1',run_id='shape010_20261014',files=old_inputs,policy_sha256=om['policy_sha256'],
        source_manifest=dict(path=OLD010+'/SHAPE010_SOURCE_MANIFEST.json',sha256=sha(repo/'SHAPE010_SOURCE_MANIFEST.json')))
    write_new(repo/'SHAPE010_BINDINGS.json',ob)
    old011={**old,**protected}
    for n in ('scripts/stability011_ops.py','scripts/supervise_stability011.py','scripts/run_stability011.py','scripts/stability011_metrics.py',
        'stability011_policy.json','docs/STABILITY011_PROTOCOL.md'):old011[n]=put(n,b'{"synthetic_source_metadata":true}\n')
    sm=dict(schema='n01/stability011-source/v1',run_id='stability011_20261014',files=old011,protected_manifests=protected,
        policy_sha256=old011['stability011_policy.json'],bound_inputs={})
    write_new(repo/'STABILITY011_SOURCE_MANIFEST.json',sm)
    sb=dict(schema='n01/stability011-bindings/v1',run_id='stability011_20261014',files={},policy_sha256=sm['policy_sha256'],
        source_manifest=dict(path=OLD011+'/STABILITY011_SOURCE_MANIFEST.json',sha256=sha(repo/'STABILITY011_SOURCE_MANIFEST.json')))
    write_new(repo/'STABILITY011_BINDINGS.json',sb)
    for k,q in paths.items():
        if q.exists() or k in ('shape010_complete','shape010_native_exit','stability011_complete','stability011_result','stability011_native_exit'):continue
        put(q.relative_to(repo).as_posix(),b'opaque synthetic saved bytes; no NN/CP/array operation')
    for slot in ('shape010_native_exit','stability011_native_exit'):
        paths[slot].parent.mkdir(parents=True,exist_ok=True);write_new(paths[slot],dict(child_native_exit_code=0,exit_code=0,reason=None,error=None,artificial_only=True))
    c=dict(run_id='shape010_20261014',G_updates=2000,frozen_A_005_unchanged=True,artificial_only=True,
        artifacts={q.name:dict(bytes=q.stat().st_size,sha256=sha(q)) for q in paths['shape010_complete'].parent.iterdir() if q.is_file()})
    write_new(paths['shape010_complete'],c)
    result=dict(run_id='stability011_20261014',classification=p['prior_stability_required']['classification'],artificial_only=True)
    write_new(paths['stability011_result'],result)
    # Synthetic old-run budget headers never masquerade as native science.
    stable=dict(run_id='stability011_20261014',bootstrap_replicates_completed=500,bootstrap_replicates_attempted=500,
        point_checks_passed=True,artificial_only=True,classification=result['classification'],
        injected_native_contract_only=True,artifacts={q.name:dict(bytes=q.stat().st_size,sha256=sha(q)) for q in paths['stability011_complete'].parent.iterdir() if q.is_file()})
    write_new(paths['stability011_complete'],stable)
    prior='receipts/stability011_20261014/user_server_pause_01/independent_readback_01/READBACK.json'
    digest=put(prior,b'{"status":"PASS","artificial_only":true}\n');p['prior_stability_required']['independent_readback_sha256']=digest
    for n in EXTRA:put(n,(json.dumps(p)+'\n').encode() if n==POLICY else (ROOT/n).read_bytes())
    receipt=freeze_sources(repo,output/'artificial_freeze',output/'artificial_backup',engineering_fixture=True)
    pack=package_payload(receipt['package'],receipt['package_sha256'],receipt['source_manifest_sha256'],receipt['bindings_sha256'])
    passed('synthetic_69_source_36_slot_hash_freeze_package_backup')
    target=output/'artificial_extracted';exclusive_extract(receipt['package'],target,pack['index'],8*1024**2);passed('exclusive_extract_sha_size')
    reject('existing_extract_rejected',lambda:exclusive_extract(receipt['package'],target,pack['index'],8*1024**2))
    for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE):
        path=output/('type_%s.tar'%kind.decode())
        with tarfile.open(path,'x:') as tar:r=tarfile.TarInfo('bad');r.type=kind;r.linkname='/outside';tar.addfile(r)
        reject('nonregular_tar_'+kind.decode(),lambda path=path:tar_index(path,100))
    duplicate=output/'duplicate.tar'
    with tarfile.open(duplicate,'x:') as tar:
        for i in range(2):r=tarfile.TarInfo('same');r.size=1;tar.addfile(r,io.BytesIO(b'x'))
    reject('duplicate_tar',lambda:tar_index(duplicate,100))
    bad=json.loads(json.dumps(pack['bindings']));bad['files']['dataset']['path']=OLD008+'/wrong.npz'
    reject('input_path_mutation',lambda:validate_documents(pack['manifest'],bad,p,pack['manifest_sha256']))
    # Import actual producer without importing numeric modules or decoding bytes.
    before=set(sys.modules);spec=importlib.util.spec_from_file_location('mechanism012_artificial_producer',ROOT/'scripts/run_mechanism012.py')
    producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
    require(not any(n in sys.modules and n not in before for n in ('numpy','torch')),'Store bridge unexpectedly imported numerical runtime')
    store=producer.OperationStore012(output/'artificial_producer',p,artificial=True,device='cpu')
    store.binding=dict(files_sha256={r['path']:r['sha256'] for r in pack['bindings']['files'].values()},
        sources_sha256={REMOTE+'/'+n:d for n,d in pack['manifest']['files'].items()})
    store.point_passed=True
    mockfirst=dict(old_A_rows=2048,noise005_rows=2048,head_final_rows=2048,scalar_AD_queries=1)
    store.call('synthetic_contract_only_no_forward_no_AD',mockfirst,lambda:None)
    proof=dict(store.base(),finite_formal_AD=True,inputs_unchanged=True,frozen_state_unchanged=True)
    store.json('FIRST_DIAGNOSTIC_PROOF.json',proof)
    store.call('synthetic_remaining_budget_fields_no_science',dict(scalar_AD_queries=8,masked_endpoint_VJP_queries=2,ADC_hard_or_STE_rows=49152,checkpoint_loads=5),lambda:None)
    store.phase='complete';store.status(dict(direction_defined=True))
    c=dict(store.base(),direction_defined=True,engineering_checks_passed=True,inputs_unchanged=True,frozen_state_unchanged=True,
        artifacts={n:dict(bytes=(store.root/n).stat().st_size,sha256=sha(store.root/n)) for n in ('status.json','operations.jsonl','FIRST_DIAGNOSTIC_PROOF.json')})
    store.json('COMPLETE.json',c)
    progress=operation_progress(store.root)
    state={'START_RECEIPT.json':dict(supervisor_pid=111),'runs/primary/supervisor/launch.json':dict(supervisor_pid=111,child_pid=c['pid']),
        'runs/primary/supervisor/exit_receipt.json':dict(exit_code=0,child_native_exit_code=0,reason=None,error=None,registered_budgets=BUDGET,child_pid=c['pid'],supervisor_pid=111),
        'runs/primary/artifacts/COMPLETE.json':c,'runs/primary/artifacts/status.json':read_json(store.root/'status.json'),
        'runs/primary/artifacts/FIRST_DIAGNOSTIC_PROOF.json':proof,'runs/primary/artifacts/FAILED.json':None,'durable_child_progress':progress,
        '_verified_freeze':dict(bound_inputs=store.binding['files_sha256'],new_sources=store.binding['sources_sha256'])}
    reject('artificial_producer_rejected_as_science',lambda:verify_completed(state))
    native=json.loads(json.dumps(state))
    for n in ('runs/primary/artifacts/COMPLETE.json','runs/primary/artifacts/status.json','runs/primary/artifacts/FIRST_DIAGNOSTIC_PROOF.json'):native[n]['artificial_only']=False
    require(verify_completed(native)['counters']['scalar_AD_queries']==9,'Actual Store terminal schema bridge failed');passed('actual_Store_counters_hashes_journal_terminal_bridge_in_memory_only')
    for key,val in (('scalar_AD_queries',8),('head_initial_or1000_rows',1),('optimizer_steps',1)):
        bad=json.loads(json.dumps(native));bad['runs/primary/artifacts/COMPLETE.json']['counters'][key]=val
        reject('terminal_count_'+key,lambda bad=bad:verify_completed(bad))
    bad=json.loads(json.dumps(native));bad['runs/primary/artifacts/COMPLETE.json']['inputs_unchanged']=False
    reject('changed_frozen_bytes',lambda:verify_completed(bad))
    bad=json.loads(json.dumps(native));bad['runs/primary/supervisor/exit_receipt.json']['child_native_exit_code']=-9
    reject('native_nonzero_rejected',lambda:verify_completed(bad))
    bad=json.loads(json.dumps(native));bad['runs/primary/artifacts/FAILED.json']={'artificial':True}
    reject('FAILED_not_complete',lambda:verify_completed(bad))
    failing=producer.OperationStore012(output/'artificial_failed',p,artificial=True,device='cpu')
    failing.call('synthetic_success',dict(old_A_rows=128),lambda:None)
    try:failing.call('synthetic_failure',dict(head_final_rows=128),lambda:(_ for _ in ()).throw(RuntimeError('injected')))
    except RuntimeError:pass
    progress=operation_progress(failing.root)
    require(progress['completed_counts_durable_lower_bound']['old_A_rows']==128
        and progress['completed_counts_durable_lower_bound']['head_final_rows']==0
        and progress['entered_counts_durable_lower_bound']['head_final_rows']==128
        and progress['unknown_inflight_operation_ids']==[1] and progress['failed_operation_ids']==[1],'Failed operation completion overclaimed')
    with (failing.root/'operations.jsonl').open('ab') as f:f.write(b'{"operation_id":2')
    require(operation_progress(failing.root)['journal_truncated_tail'],'Partial journal tail not preserved');passed('failed_completed_and_unknown_inflight_durable_no_FAILED_KILL')
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('mechanism012_artificial_supervisor',ROOT/'scripts/supervise_mechanism012.py')
    sup=importlib.util.module_from_spec(spec);spec.loader.exec_module(sup)
    for k,args in (('RSS_limit',dict(rss=8*1024**3+1,gpu=0,size=0,elapsed=0)),('GPU_limit',dict(rss=0,gpu=10*1024**3+1,size=0,elapsed=0)),
        ('output_limit',dict(rss=0,gpu=0,size=128*1024**2+1,elapsed=0)),('wall_limit',dict(rss=0,gpu=0,size=0,elapsed=901))):
        require(sup.limit_reason(RESOURCES,**args)==k,'Resource guard failed: '+k);passed(k)
    original=sup.subprocess.check_output
    sup.subprocess.check_output=lambda *a,**k:'PID PPID RSS\n100 1 50\n101 100 20\n102 101 10\n200 1 999\n'
    try:pids,rss=sup.process_tree(100);require(pids=={100,101,102} and rss==80*1024,'Owned tree RSS includes/excludes wrong processes')
    finally:sup.subprocess.check_output=original
    passed('owned_supervisor_plus_descendants_RSS_excludes_other_process')
    popen=globals()['subprocess'].Popen
    class MockChild:pid=321
    globals()['subprocess'].Popen=lambda *a,**kw:(require(kw['pass_fds']==(77,) and kw['start_new_session'] and kw['stdin']==subprocess.DEVNULL,'Lock/stdio ownership wrong') or MockChild())
    try:require(spawn_supervisor(['synthetic_no_process'],output,77,{}) .pid==321,'Supervisor lock bridge failed')
    finally:globals()['subprocess'].Popen=popen
    passed('inherited_lock_fd_and_detached_owned_session')
    killed=[];original=getattr(sup.os,'killpg',None);original_signal=getattr(sup.signal,'SIGKILL',None)
    class MockTermChild:
        pid=345
        def poll(self):return None
        def wait(self,timeout=None):
            if timeout is not None:raise subprocess.TimeoutExpired('synthetic',timeout)
            return -9
    sup.os.killpg=lambda pid,sig:killed.append((pid,sig))
    if original_signal is None:sup.signal.SIGKILL=9
    try:sup.stop_owned(MockTermChild(),20)
    finally:
        if original is None:del sup.os.killpg
        else:sup.os.killpg=original
        if original_signal is None:del sup.signal.SIGKILL
    require([pid for pid,sig in killed]==[345,345],'Stop touched unowned group');passed('TERM_KILL_only_owned_child_group')
    result=dict(schema='n01/mechanism012-ops-artificial-qa/v1',status='PASS',UTC=now(),artificial_only=True,checks=checks,passed_checks=len(checks),
        source_package_copies_synthetic=True,old_native_headers_injected_only=True,actual_Store_writer_exercised_without_numeric_import=True,
        actual_SSH_SCP_calls=0,numeric_decodes=0,checkpoint_loads=0,actual_NN_rows=0,actual_AD_queries=0,actual_optimizer_steps=0,
        sources_sha256={n:sha(ROOT/n) for n in EXTRA if n.endswith('.py')})
    write_new(output/'ENGINEERING_RECEIPT.json',result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',nargs='?',choices=('freeze','deploy','start','status','verify-start','collect','collect-inspect','collect-failed','collect-failed-inspect','self-check'))
    parser.add_argument('--repo',type=Path,default=ROOT);parser.add_argument('--freeze',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--backup',type=Path);parser.add_argument('--collected',type=Path);parser.add_argument('--_remote');args=parser.parse_args()
    if args._remote is not None:
        require(args.mode is None,'Do not mix native and local actions');payload=json.loads(base64.urlsafe_b64decode(args._remote.encode()))
        print(json.dumps(remote_dispatch(payload),allow_nan=False));return
    require(args.mode is not None and args.output is not None,'Explicit mode/fresh output required')
    if args.mode=='self-check':
        try:result=self_check(args.output)
        except BaseException as error:
            if args.output.is_dir() and not (args.output/'ENGINEERING_FAILED.json').exists():
                write_new(args.output/'ENGINEERING_FAILED.json',dict(UTC=now(),artificial_only=True,error=repr(error),scientific_executions=0))
            raise
    elif args.mode=='freeze':
        require(args.backup is not None,'Explicit D small-source backup required');result=freeze_sources(args.repo,args.output,args.backup)
    else:
        require(args.freeze is not None,'Explicit reviewed freeze required');result=controller_action(args.mode,args.freeze,args.output,args.collected)
    # Full native state is saved, never sprayed into the user's console.
    print(json.dumps({k:result[k] for k in ('status','first_diagnostic_verified','passed_checks','source_file_count','input_count','supervisor_pid','child_pid','partial_evidence_only') if k in result}
        |dict(receipt_directory=str(args.output)),allow_nan=False))

if __name__=='__main__':main()
