"""015 policy adapter over the exact, audited 014 stdlib operations.

Old source bytes are checked before loading. Archive/SSH/lock/owned supervision,
partial collection, resource and counter tools are reused without new kernels.
Only label literals and the explicit 015 contract hooks below differ.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import sys
import tarfile
import types

RUN_ID='wp3_015_20261015'
REMOTE='/home/rylan/projects/ej276d-n01/wp3_015_20261015'
OLD014='/home/rylan/projects/ej276d-n01/wp2_014_20261015'
CFG,MANIFEST='wp3_015_config.json','WP3_SOURCE_MANIFEST.json'
CONTROLLER,DRIVER='scripts/wp3_controller.py','scripts/run_wp3_015.py'
OWNED=(CONTROLLER,DRIVER,'src/n01/model015.py')
BASE_SHA='3480825e94e954a46c2c633e5df6ddef514e2b7def1332bb2e7688e78ae041a3'
PRIOR_SHA='14db81d3a08a0edede97f544ed8ec4c63147138223c00fc38d833bbc278e2684'
REMOTE_INVENTORY_SHA='4cf27f6c771294561a474ef2de9574ab711d9322c588da488f0d9ce2e49792d5'
T_SHA='f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b'
TRIALS=('seed0_M0','seed0_M1','seed1_M0','seed1_M1')
HYPOTHESES=('fast','slow','common_level')
SELF_ROOT=Path(__file__).resolve().parents[1] if __file__ not in ('<stdin>','<string>') else Path(REMOTE)
_OPS=None


def require(value,message):
    if not value:raise ValueError(message)


def factory_contract(source):
    tree=ast.parse(source);names={'seed_set015','settings015','budget_caps015'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==names,'015 pure factories missing')
    for node in nodes:
        require(not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.With,ast.Global,ast.Nonlocal)) for n in ast.walk(node)),'015 factory imported runtime')
        require(all(isinstance(n.func,ast.Name) and n.func.id in ('dict','list','len','seed_set015')
            for n in ast.walk(node) if isinstance(n,ast.Call)),'015 factory executes nonliteral helper')
    scope=dict(RUN_ID=RUN_ID,HYPOTHESES=HYPOTHESES)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<015-stdlib-factories>','exec'),scope)
    return scope['settings015'](),scope['budget_caps015']


def validate_config(p,*,frozen=True,driver_source=None):
    require(p.get('schema')=='n01/wp3-015-config/v1' and p.get('run_id')==RUN_ID and p.get('remote_root')==REMOTE,'015 identity/root differs')
    require(not frozen or p.get('status')=='frozen','015 draft cannot execute')
    source=driver_source if driver_source is not None else (SELF_ROOT/DRIVER).read_bytes()
    reference,budget=factory_contract(source)
    for key,value in reference.items():
        if key in ('status','evaluation'):continue
        require(p.get(key)==value,'015 setting differs: '+key)
    for key,value in reference['evaluation'].items():
        if key in ('bank_seeds','projection_seed'):continue
        require(p['evaluation'].get(key)==value,'015 evaluation differs: '+key)
    require(len(p['evaluation']['bank_seeds'])==2 and type(p['evaluation']['projection_seed']) is int,'015 fixed evaluation banks/projection required')
    require(p.get('budget_caps')==budget(p),'015 exact operation caps differ')
    required={'dataset','statistics','data_statistics','base_config','deterministic','noise005','source_information',
        'integral_statistics','C1_head_seed0','C1_head_seed1'}
    required.update(f'mean_seed{s}_bank{b}_{h}' for s in range(2) for b in range(2) for h in HYPOTHESES)
    require(required<=set(p.get('inputs',{})),'015 bound inputs incomplete')
    require({'observation','data_roles','acceptance','tolerance_table'}<=set(p.get('contracts',{})),'015 contracts incomplete')
    require(p['contracts']['tolerance_table']['sha256']==T_SHA,'015 original T changed')
    require(REMOTE in p.get('allowed_input_roots',[]),'015 allowed launch root missing')
    evidence=p.get('remote_inventory_evidence',{})
    require(evidence.get('path')==REMOTE+'/WP2_REMOTE_FULL_INVENTORY.json'
        and evidence.get('sha256')==REMOTE_INVENTORY_SHA and evidence.get('bytes')==129532,
        '015 original318 inventory evidence must be explicitly bound')
    require(not frozen or (set(OWNED)<=set(p.get('source_files',{})) and CFG not in p['source_files'] and MANIFEST not in p['source_files']),
        '015 own source inventory missing or recursive')
    return reference


def remote_input_binding015(row,evidence):
    """Freeze provenance for server-only artifacts, without transferring bytes."""
    path=PurePosixPath(row['path']);root=PurePosixPath(OLD014)
    require(path.is_relative_to(root),'015 remote-only input is not in original014 tree')
    name=path.relative_to(root).as_posix();known=evidence['whole_remote_inventory'].get(name)
    require(known is not None and row.get('sha256')==known['sha256'] and row.get('bytes')==known['bytes'],
        '015 remote-only input differs from original full SHA inventory')
    return name


def freeze(repo,output,expected_draft_sha,*,artificial=False):
    b=operations();repo=b.core().unredirected(repo)
    require(b.sha(repo/CFG)==expected_draft_sha,'015 draft SHA differs')
    p=copy.deepcopy(b.core().read_json(repo/CFG));validate_config(p,frozen=False,driver_source=(repo/DRIVER).read_bytes())
    prior_file=repo/'WP2_SOURCE_MANIFEST.json';require(b.sha(prior_file)==PRIOR_SHA,'015 original95 manifest changed')
    prior=b.core().read_json(prior_file);require(len(prior['files'])==95,'015 original95 inventory changed')
    old_config=b.core().read_json(repo/'wp2_014_config.json')
    require(b.sha(repo/'wp2_014_config.json')==prior['config_sha256'],'015 original014 configuration changed')
    checked={str(repo/CFG):expected_draft_sha,str(prior_file):PRIOR_SHA}
    paths={n:repo/n for n in set(prior['files'])|set(OWNED)|{'WP2_SOURCE_MANIFEST.json'}}
    evidence_path=b.checked_local(p['remote_inventory_evidence'],checked)
    evidence=b.core().read_json(evidence_path)
    require(evidence.get('whole_remote_inventory_verified') is True and len(evidence['whole_remote_inventory'])==318
        and evidence.get('source_manifest_sha256')==PRIOR_SHA
        and evidence.get('config_sha256')==prior['config_sha256'],'015 original318 evidence identity differs')
    paths['WP2_REMOTE_FULL_INVENTORY.json']=evidence_path
    protections=copy.deepcopy(old_config.get('protection_files',{}))
    for row in protections.values():b.checked_local(row,checked)
    for name,row in prior['files'].items():
        require(b.named_record(repo/name)==row,'015 original95 source changed: '+name)
        protections['wp2_014:'+name]=dict(path=OLD014+'/'+name,local_path=str(repo/name),**row)
    for name,digest in (('WP2_SOURCE_MANIFEST.json',PRIOR_SHA),('wp2_014_config.json',prior['config_sha256'])):
        protections['wp2_014:'+name]=dict(path=OLD014+'/'+name,local_path=str(repo/name),sha256=digest,bytes=(repo/name).stat().st_size)
    for records in (p['inputs'],p['contracts']):
        for row in records.values():
            if row.get('local_path') and Path(row['local_path']).is_file():
                local=b.checked_local(row,checked)
            else:
                remote_input_binding015(row,evidence);local=None
            native=PurePosixPath(row['path'])
            if native.is_relative_to(PurePosixPath(REMOTE)):
                require(local is not None,'015 newly packaged contract must exist locally')
                name=native.relative_to(PurePosixPath(REMOTE)).as_posix();b.core().safe_name(name)
                require(not name.endswith(('.npz','.pt','.pth','.bin')),'015 source payload cannot provision numeric inputs')
                require(name not in paths or b.named_record(paths[name])==b.named_record(local),'015 contract/source alias differs')
                paths[name]=local
    source_records={name:b.named_record(path) for name,path in sorted(paths.items())}
    closure=b.project_import_closure(source_records,lambda n:paths[n].read_bytes())
    p['source_files']={n:dict(path=REMOTE+'/'+n,**row) for n,row in source_records.items()}
    p['protection_files']=protections;p.update(status='frozen',frozen_UTC=b.core().now())
    validate_config(p,driver_source=(repo/DRIVER).read_bytes())
    roots=[PurePosixPath(v) for v in p['allowed_input_roots']]
    for records in (p['source_files'],p['inputs'],p['contracts'],p['protection_files']):
        require(all(any(PurePosixPath(row['path']).is_relative_to(r) for r in roots) for row in records.values()),'015 frozen path outside allowed roots')
    output=b.core().fresh_directory(output,local=True);b.core().write_new(output/CFG,p);config_sha=b.sha(output/CFG)
    manifest=dict(schema='n01/wp3-source/v1',run_id=RUN_ID,config_sha256=config_sha,frozen_UTC=p['frozen_UTC'],files=source_records,
        prior_WP2_manifest_sha256=PRIOR_SHA,prior_source_file_count=95,project_import_closure=closure,
        before_scientific_decode=True,public_source_only=True,scientific_inputs_copied=False)
    b.core().write_new(output/MANIFEST,manifest);package=output/b.SOURCE_ARCHIVE
    with tarfile.open(package,'x:gz') as bundle:
        for name,path in sorted(paths.items()):b.add_regular(bundle,path,name)
        for name in (CFG,MANIFEST):b.add_regular(bundle,output/name,name)
    pins=dict(source_package_sha256=b.sha(package),config_sha256=config_sha,source_manifest_sha256=b.sha(output/MANIFEST),controller_sha256=source_records[CONTROLLER]['sha256'])
    _,_,inventory=b.read_bundle(package,pins)
    for path,digest in checked.items():require(b.sha(path)==digest,'015 input/source changed during freeze')
    for name,path in paths.items():require(b.named_record(path)==source_records[name],'015 source changed during freeze')
    result=dict(schema='n01/wp3-freeze/v1',run_id=RUN_ID,UTC=b.core().now(),**pins,source_package=str(package),source_package_bytes=package.stat().st_size,
        source_file_count=len(source_records),source_inventory=inventory,protected_source_records=len(protections),project_import_closure=closure,
        all_original_files_unchanged=True,artificial_only=artificial,scientific_started=False,array_decodes=0,checkpoint_loads=0,model_forwards=0)
    b.core().write_new(output/'FREEZE_RECEIPT.json',result);return result


def first_valid(state,p,*,current_boot=True):
    b=operations();b.same_snapshot_not_failed(state)
    terminal=state.get('runs/primary/artifacts/COMPLETE.json') or {}
    if terminal.get('training_skipped') is True:
        verdict=complete_valid(state,p)
        return dict(status='registered_mean_gate_completed_training_skipped',actual_completed_G_updates=0,
            mean_gate_skipped_training=True,actual_first_G1_verified=False,scientific_acceptance=False,terminal=verdict)
    start=state.get('START_RECEIPT.json');launch=state.get('runs/primary/supervisor/launch.json')
    first=state.get('runs/primary/artifacts/FIRST_UPDATE_PROOF.json');init=state.get('runs/primary/artifacts/INIT_PROOF.json')
    require(all(isinstance(v,dict) for v in (start,launch,first,init)),'015 first real G1 not ready')
    require(start['supervisor_pid']==launch['supervisor_pid'] and launch['child_pid']==first['pid']==init['pid'],'015 PID proof differs')
    require(not current_boot or start['boot_id']==state['current_boot_id'],'015 proof is from another boot')
    require(first.get('run_id')==RUN_ID and first.get('artificial_only') is False and init.get('artificial_only') is False,'015 artificial/wrong proof')
    require(init.get('native_initialized') is True and init.get('mean_gate_all6_passed') is True
        and init.get('zero_mean_adapter') is True and init.get('first_trial')==TRIALS[0]
        and init.get('paired_initialization_bytes') is True and init.get('fresh_AdamW') is True,'015 initialization gate/reference incomplete')
    require(init['counters']['checkpoint_loads']==3 and init['counters']['optimizer_created']==1 and init['counters']['G_updates']==0,'015 initial checkpoint/update boundary differs')
    require(first.get('actual_G_updates')==first.get('batch_completed_G_updates')==1 and first.get('trial')==TRIALS[0]
        and first.get('candidate')=='M0' and first.get('gradient_nonzero') is True
        and first.get('actual_parameter_update_nonzero') is True and first.get('frozen_A_005_unchanged') is True,'015 first actual optimizer step missing')
    require(first.get('trainable_parameters')==1288 and first.get('trainable_parameter_names')==['basis_head.weight','basis_head.bias'],'015 first arm whitelist differs')
    require(first['frozen_state_sha256']==init['frozen_state_sha256'] and bool(first['frozen_state_sha256']),'015 frozen tensors differ')
    b.validate_counters(p,first['attempted_counters'],first['counters']);counts=first['counters']
    require(counts['G_updates']==counts['optimizer_steps']==counts['new_AD_queries']==counts['training_backward_queries']==1
        and counts['checkpoint_loads']==3 and not first.get('inflight'),'015 first G1 counters incomplete')
    require(all(counts[k]>=1024 for k in ('old_A_rows','head_actual_parameter_forward_rows','noise005_rows'))
        and counts['hard_ADC_rows']>=3072 and first.get('mean_gate_all6_passed') is True,'015 first formal training family missing')
    expected=state['_verified_freeze']['immutable_files_sha256']
    require(expected and all(first['immutable_files_sha256'].get(k)==v and init['immutable_files_sha256'].get(k)==v for k,v in expected.items()),'015 first immutable files differ')
    return dict(status='actual_first_G1_and_mean_gate_verified',supervisor_pid=start['supervisor_pid'],child_pid=first['pid'],actual_completed_G_updates=1,
        mean_gate_skipped_training=False,actual_first_G1_verified=True,
        current_batch_completed_G_updates=(state.get('runs/primary/artifacts/status.json') or {}).get('counters',{}).get('G_updates'),scientific_acceptance=False)


def complete_valid(state,p):
    b=operations();b.same_snapshot_not_failed(state)
    native=state.get('runs/primary/supervisor/exit_receipt.json') or {};c=state.get('runs/primary/artifacts/COMPLETE.json')
    require(native.get('child_native_exit_code')==native.get('exit_code')==0 and native.get('reason') is None and native.get('error') is None,'015 native failure/resource stop')
    require(isinstance(c,dict) and c.get('schema')=='n01/wp3-batch-complete/v1' and c.get('run_id')==RUN_ID
        and c.get('native_status')=='COMPLETE' and c.get('artificial_only') is False,'015 producer terminal proof missing')
    counts=c['counters'];b.validate_counters(p,c['attempted_counters'],counts)
    require(c['attempted_counters']==counts and not c.get('inflight'),'015 unresolved operation')
    if c.get('training_skipped') is True:
        require(c.get('reason')=='registered_population_mean_evidence_not_met' and c.get('no_checkpoint_NN_optimizer_AD_G') is True
            and not c.get('trials') and c.get('all_four_trials_completed') is False,'015 invalid mean gate skip')
        require(all(v==0 for k,v in counts.items() if not k.startswith('mean_evidence_')),'015 gate skip spent training/model budget')
        require(all(counts[k]==v for k,v in p['budget_caps'].items() if k.startswith('mean_evidence_')),'015 incomplete registered mean gate')
    else:
        first_valid(state,p,current_boot=False)
        require(c.get('all_four_trials_completed') is True and c.get('all_frozen_A_005_unchanged') is True
            and c.get('mean_gate_all6_passed') is True and c.get('no_DEV') is True,'015 full trial scope incomplete')
        require(c.get('G_updates_per_trial')==2000 and [r['directory'] for r in c['trials']]==list(TRIALS)
            and all(r['G_updates']==2000 for r in c['trials']),'015 serial trial boundaries differ')
        require(counts==p['budget_caps'],'015 exact complete operation budgets differ')
    progress=state['durable_child_progress']
    require(not progress['unknown_inflight_ids'] and not progress['failed_ids'] and not progress['truncated_tail'],'015 unresolved durable operation')
    require(all(progress['completed_counts'].get(k,0)==v==progress['entered_counts'].get(k,0) for k,v in counts.items()),'015 terminal journal counts differ')
    expected=state['_verified_freeze']['immutable_files_sha256']
    require(expected and all(c['immutable_files_sha256'].get(k)==v for k,v in expected.items()),'015 final immutable files differ')
    terminal=state.get('runs/primary/artifacts/status.json') or {}
    require(terminal.get('counters')==counts and terminal.get('native_status')=='COMPLETE','015 terminal status differs')
    require(native['supervisor_pid']==state['START_RECEIPT.json']['supervisor_pid'] and native['child_pid']==c['pid'],'015 terminal PID chain differs')
    require(native.get('elapsed_seconds',float('inf'))<=p['resource_limits']['wall_seconds']
        and native.get('peak_output_bytes',float('inf'))<=p['resource_limits']['output_MiB']*1024**2,'015 terminal resources exceed cap')
    return dict(status='registered_mean_gate_skipped_training' if c.get('training_skipped') else 'native_and_four_trial_producer_complete',
        counters=counts,trials=c.get('trials',[]),training_skipped=c.get('training_skipped'),scientific_acceptance=False,root_candidate_choice_pending=True)


def operations():
    global _OPS
    if _OPS is not None:return _OPS
    candidates=[SELF_ROOT/'scripts/wp2_controller.py',Path(OLD014)/'scripts/wp2_controller.py']
    path=next((p for p in candidates if p.is_file()),None)
    require(path is not None and hashlib.sha256(path.read_bytes()).hexdigest()==BASE_SHA,'015 exact reused014 operations unavailable')
    tree=ast.parse(path.read_bytes(),filename=str(path))
    # Format labels only. All filesystem/process/resource/transfer algorithms
    # remain the pinned 014 AST; policy hooks are replaced explicitly below.
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            value=node.value
            if value.startswith('n01/wp2-'):node.value=value.replace('n01/wp2-','n01/wp3-',1)
            elif value.startswith('wp2_') or value.startswith('.wp2.'):node.value=value.replace('wp2_','wp3_',1).replace('.wp2.','.wp3.',1)
    module=types.ModuleType('wp3_reused014_stdlib');module.__file__=__file__
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
    module.__dict__.update(RUN_ID=RUN_ID,REMOTE=REMOTE,CFG=CFG,MANIFEST=MANIFEST,CONTROLLER=CONTROLLER,DRIVER=DRIVER,
        OWNED=OWNED,TRIALS=TRIALS,SELF_ROOT=SELF_ROOT,SOURCE_ARCHIVE='wp3_source.tar.gz',PRIOR_MANIFEST_SHA=PRIOR_SHA,
        factory_contract=factory_contract,validate_config=validate_config,freeze=freeze,first_valid=first_valid,complete_valid=complete_valid)
    _OPS=module;return module


def main(argv=None):return operations().main(argv)


if __name__=='__main__':raise SystemExit(main())
