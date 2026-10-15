"""Local source assembly from immutable016 controller; no native calls."""
from pathlib import Path
import hashlib
import re
ROOT=Path(__file__).resolve().parents[3]
old=ROOT/'scripts/wp3_016_controller.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='ee1780358314b2fb7c425d21bddd5df16184642b2f5b1317c1e1b6a00d403cfb'
s=old.read_text('utf8').replace('016','017').replace('wp3_017','wp4_017').replace('wp3-017','wp4-017')
s=s.replace("'/home/rylan/projects/ej276d-n01/wp4_017_runtime_fix02_20261015'","'/home/rylan/projects/ej276d-n01/wp4_017_20261015'")
s=s.replace("OLD015='/home/rylan/projects/ej276d-n01/wp3_015_20261015'","OLD016='/home/rylan/projects/ej276d-n01/wp3_016_runtime_fix02_20261015'")
s=s.replace("'WP3_017_SOURCE_MANIFEST.json'","'WP4_017_SOURCE_MANIFEST.json'")
s=s.replace("OWNED=(CONTROLLER,DRIVER,'src/n01/model017.py')","OWNED=(CONTROLLER,DRIVER,'src/n01/model017.py','src/n01/stats017.py','src/n01/saved_array_kernels017.py')")
s=s.replace("PRIOR_SHA='f1255dbff0e47edb7ec20e76c7d90057201f9a56f1184b7896f0c8cb0a67825a'","PRIOR_SHA='824780d297c440d0f1bc51b3ea3344e488ec42ccb46af959d9d0412d8240fc14'\nPROPOSAL_SHA='b2db64dfd3549f6cd22cb5253a1e374738248012e22554308ec26885d2d42e8e'")
s=s.replace("TRIALS=('seed0_Q1','seed1_Q1')","TRIALS=('T1_paired_seed0','T1_paired_seed1','T1_unpaired_seed0','T1_unpaired_seed1',\n        'T2_paired_seed0','T2_paired_seed1','T2_unpaired_seed0','T2_unpaired_seed1')")
def function(name,text):
    global s
    s,count=re.subn(r'^def '+name+r'\([^\n]*(?:\n(?!def |if __name__)[^\n]*)*',text.rstrip()+'\n\n',s,count=1,flags=re.M)
    assert count==1,name

function('factory_contract',r'''def factory_contract(source):
    tree=ast.parse(source);names={'settings017','budget_caps017'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==names,'017 pure factories missing')
    for node in nodes:
        require(not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.With,ast.Global,ast.Nonlocal)) for n in ast.walk(node)),'017 factory runtime import')
        require(all(isinstance(n.func,ast.Name) and n.func.id in ('dict','list','len') for n in ast.walk(node) if isinstance(n,ast.Call)),'017 nonliteral factory')
    scope=dict(RUN_ID=RUN_ID,REMOTE=REMOTE,HYPOTHESES=HYPOTHESES,TRIALS=TRIALS,PROPOSAL_SHA=PROPOSAL_SHA)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<017-stdlib-factories>','exec'),scope)
    return scope['settings017'](),scope['budget_caps017']
''')
function('validate_config',r'''def validate_config(p,*,frozen=True,driver_source=None):
    require(p.get('schema')=='n01/wp4-017-config/v1' and p.get('run_id')==RUN_ID and p.get('remote_root')==REMOTE,'017 identity/root differs')
    require(not frozen or p.get('status')=='frozen','017 draft cannot execute')
    reference,budget=factory_contract(driver_source if driver_source is not None else (SELF_ROOT/DRIVER).read_bytes())
    for key,value in reference.items():
        if key!='status':require(p.get(key)==value,'017 setting differs: '+key)
    require(p.get('budget_caps')==budget(p) and p.get('stage1_budget_caps')==budget(p,stage1=True),'017 full/stage1 exact caps differ')
    require(set(p.get('inputs',{}))=={'dataset','statistics','data_statistics','base_config','deterministic','noise005','M0_head_seed0'},'017 seven fixed inputs only')
    require({'proposal','contract_review','configuration','acceptance','observation','data_roles','tolerance_table'}<=set(p.get('contracts',{})),'017 missing contracts')
    require(p['contracts']['proposal']['sha256']==PROPOSAL_SHA and p['contracts']['contract_review']['sha256']=='31a0a106bf033ec7d363abc13811a7f0e6dd87e79d139ea7ad6c867a67927fd7'
        and p['contracts']['tolerance_table']['sha256']==T_SHA,'017 accepted immutable contract differs')
    require(REMOTE in p.get('allowed_input_roots',[]),'017 allowed root missing')
    require(not frozen or (set(OWNED)<=set(p.get('source_files',{})) and CFG not in p['source_files'] and MANIFEST not in p['source_files']),'017 source closure missing/recursive')
    return reference
''')
function('freeze',r'''def freeze(repo,output,expected_draft_sha,*,artificial=False):
    b=operations();repo=b.core().unredirected(repo)
    require(b.sha(repo/CFG)==expected_draft_sha,'017 draft changed')
    p=copy.deepcopy(b.core().read_json(repo/CFG));validate_config(p,frozen=False,driver_source=(repo/DRIVER).read_bytes())
    prior_file=repo/'WP3_016_SOURCE_MANIFEST.json';require(b.sha(prior_file)==PRIOR_SHA,'017 original116 manifest changed')
    prior=b.core().read_json(prior_file);require(len(prior['files'])==116,'017 original116 count changed')
    old_config=b.core().read_json(repo/'wp3_016_config.json');require(b.sha(repo/'wp3_016_config.json')==prior['config_sha256'],'017 old016 config changed')
    checked={str(repo/CFG):expected_draft_sha,str(prior_file):PRIOR_SHA}
    paths={n:repo/n for n in set(prior['files'])|set(OWNED)|{'WP3_016_SOURCE_MANIFEST.json'}}
    protections=copy.deepcopy(old_config.get('protection_files',{}))
    for row in protections.values():b.checked_local(row,checked)
    for name,row in prior['files'].items():
        require(b.named_record(repo/name)==row,'017 original116 bytes changed: '+name)
        protections['wp3_016:'+name]=dict(path=OLD016+'/'+name,local_path=str(repo/name),**row)
    for name,digest in (('WP3_016_SOURCE_MANIFEST.json',PRIOR_SHA),('wp3_016_config.json',prior['config_sha256'])):
        protections['wp3_016:'+name]=dict(path=OLD016+'/'+name,local_path=str(repo/name),sha256=digest,bytes=(repo/name).stat().st_size)
    cpu_path=repo/'receipts/wp3_016_20261015/readback_registration_01/READBACK_EXECUTION.json'
    require(b.sha(cpu_path)=='afa1b6a340e9e71eb5517ca68929c15bb05c9c5a1d990da50ebe350cce0ba8a8','017 oldCPU config changed')
    cpu=b.core().read_json(cpu_path)
    for name,row in cpu['files'].items():
        local=b.checked_local(row,checked);relative=local.relative_to(repo).as_posix();paths[relative]=local
        protections['wp3_016_CPU:'+name]=row
    paths[cpu_path.relative_to(repo).as_posix()]=cpu_path
    protections['wp3_016_CPU:READBACK_EXECUTION.json']=dict(path=cpu['remote_root']+'/READBACK_EXECUTION.json',local_path=str(cpu_path),**b.named_record(cpu_path))
    p['allowed_input_roots']=sorted(set(p['allowed_input_roots'])|{cpu['remote_root']})
    for records in (p['inputs'],p['contracts']):
        for row in records.values():
            local=b.checked_local(row,checked);native=PurePosixPath(row['path'])
            if native.is_relative_to(PurePosixPath(REMOTE)):
                name=native.relative_to(PurePosixPath(REMOTE)).as_posix();b.core().safe_name(name)
                require(not name.endswith(('.npz','.npy','.pt','.pth','.bin')),'017 cannot provision scientific numeric inputs')
                require(name not in paths or b.named_record(paths[name])==b.named_record(local),'017 contract/source alias differs');paths[name]=local
    records={name:b.named_record(path) for name,path in sorted(paths.items())}
    closure=b.project_import_closure(records,lambda n:paths[n].read_bytes())
    p['source_files']={n:dict(path=REMOTE+'/'+n,**row) for n,row in records.items()}
    p['protection_files']=protections;p.update(status='frozen',frozen_UTC=b.core().now())
    validate_config(p,driver_source=(repo/DRIVER).read_bytes());roots=[PurePosixPath(v) for v in p['allowed_input_roots']]
    for collection in (p['source_files'],p['inputs'],p['contracts'],p['protection_files']):
        require(all(any(PurePosixPath(row['path']).is_relative_to(r) for r in roots) for row in collection.values()),'017 frozen path escaped explicit roots')
    output=b.core().fresh_directory(output,local=True);b.core().write_new(output/CFG,p);config_sha=b.sha(output/CFG)
    manifest=dict(schema='n01/wp4-017-source/v1',run_id=RUN_ID,config_sha256=config_sha,frozen_UTC=p['frozen_UTC'],files=records,
        prior_WP3_manifest_sha256=PRIOR_SHA,prior_source_file_count=116,prior_CPU_controls=6,project_import_closure=closure,
        before_scientific_decode=True,public_source_only=True,scientific_inputs_copied=False)
    b.core().write_new(output/MANIFEST,manifest);package=output/b.SOURCE_ARCHIVE
    with tarfile.open(package,'x:gz') as bundle:
        for name,path in sorted(paths.items()):b.add_regular(bundle,path,name)
        for name in (CFG,MANIFEST):b.add_regular(bundle,output/name,name)
    pins=dict(source_package_sha256=b.sha(package),config_sha256=config_sha,source_manifest_sha256=b.sha(output/MANIFEST),controller_sha256=records[CONTROLLER]['sha256'])
    _,_,inventory=b.read_bundle(package,pins)
    for path,digest in checked.items():require(b.sha(path)==digest,'017 bound source/input changed during freeze')
    for name,path in paths.items():require(b.named_record(path)==records[name],'017 source changed during freeze')
    result=dict(schema='n01/wp4-017-freeze/v1',run_id=RUN_ID,UTC=b.core().now(),**pins,source_package=str(package),source_package_bytes=package.stat().st_size,
        source_file_count=len(records),source_inventory=inventory,protected_source_records=len(protections),project_import_closure=closure,
        all_original_files_unchanged=True,artificial_only=artificial,scientific_started=False,array_decodes=0,checkpoint_loads=0,model_forwards=0)
    b.core().write_new(output/'FREEZE_RECEIPT.json',result);return result
''')
function('first_valid',r'''def first_valid(state,p,*,current_boot=True):
    b=operations();b.same_snapshot_not_failed(state)
    start=state.get('START_RECEIPT.json');launch=state.get('runs/primary/supervisor/launch.json')
    first=state.get('runs/primary/artifacts/FIRST_UPDATE_PROOF.json');init=state.get('runs/primary/artifacts/INIT_PROOF.json')
    require(all(isinstance(v,dict) for v in (start,launch,first,init)),'017 first real G1 not ready')
    require(start['supervisor_pid']==launch['supervisor_pid'] and launch['child_pid']==first['pid']==init['pid'],'017 PID proof differs')
    require(not current_boot or start['boot_id']==state['current_boot_id'],'017 proof another boot')
    require(first.get('run_id')==init.get('run_id')==RUN_ID and first.get('artificial_only') is init.get('artificial_only') is False,'017 artificial/wrong proof')
    require(init.get('native_initialized') is True and init.get('all_6593_Q_positive') is True and init.get('one_pass_A_decoded_M0_cache_complete') is True
        and init.get('known_T1_initial_RMS_contrast_at_least3LSB') is True and init.get('mean_adapter_enabled') is False
        and init.get('first_trial')==TRIALS[0] and init.get('fresh_AdamW') is True,'017 fixed cache initialization missing')
    I=init['counters'];F=first['counters']
    require(I['checkpoint_loads']==3 and I['optimizers_created']==1 and I['G_updates']==0
        and I['A_original_forward_rows']==I['original_U_Net_decoded_rows']==I['fixed_M0_head_actual_kernel_rows']==6593,'017 init exact cache/CP boundary')
    require(first.get('actual_G_updates')==first.get('batch_completed_G_updates')==1 and first.get('trial')==TRIALS[0]
        and first.get('both_gradients_nonzero') is True and first.get('both_parameter_updates_nonzero') is True
        and first.get('frozen_A_005_M0_unchanged') is True,'017 first actual2theta optimizer step missing')
    require(first.get('trainable_parameters')==init.get('trainable_parameters')==2
        and first.get('trainable_parameter_names')==init.get('trainable_parameter_names')==['theta_a','theta_b'],'017 whitelist differs')
    require(first['frozen_state_sha256']==init['frozen_state_sha256'] and first['cache_manifest_sha256']==init['cache_manifest_sha256'],'017 frozen tensor/cache proof differs')
    b.validate_counters(p,first['attempted_counters'],F)
    require(F['G_updates']==F['optimizer_steps']==F['new_AD_queries']==F['training_backward_queries']==1 and F['checkpoint_loads']==3 and not first.get('inflight'),'017 firstG counters incomplete')
    require(F['training_student_rows']==F['training_teacher_rows']==1024 and F['hard_ADC_rows']==F['hard_first_argmax_gather_rows']==6144
        and F['noise005_rows']==F['shape_cached_basis_assembly_rows']==2048 and F['paired_training_energy_pair_distances']==7680,'017 first actual training family differs')
    expected=state['_verified_freeze']['immutable_files_sha256']
    require(expected and all(first['immutable_files_sha256'].get(k)==v and init['immutable_files_sha256'].get(k)==v for k,v in expected.items()),'017 first immutable bytes differ')
    health=same_snapshot_health017(state,p,start,launch,first)
    return dict(status='actual_first_2theta_G1_and_fixed_cache_verified',supervisor_pid=start['supervisor_pid'],child_pid=first['pid'],
        actual_completed_G_updates=1,actual_first_G1_verified=True,same_snapshot_health=health,scientific_acceptance=False,
        current_batch_completed_G_updates=(state.get('runs/primary/artifacts/status.json') or {}).get('counters',{}).get('G_updates'))
''')
function('complete_valid',r'''def complete_valid(state,p):
    b=operations();b.same_snapshot_not_failed(state)
    native=state.get('runs/primary/supervisor/exit_receipt.json') or {};c=state.get('runs/primary/artifacts/COMPLETE.json') or {}
    require(native.get('child_native_exit_code')==native.get('exit_code')==0 and native.get('reason') is None and native.get('error') is None,'017 native failure/resource stop')
    require(c.get('schema')=='n01/wp4-017-batch-complete/v1' and c.get('run_id')==RUN_ID and c.get('native_status')=='COMPLETE'
        and c.get('artificial_only') is False,'017 producer terminal missing')
    expected=terminal_policy017(c,p);counts=c['counters'];b.validate_counters(p,c['attempted_counters'],counts)
    require(c['attempted_counters']==counts==expected and not c.get('inflight'),'017 terminal exact counters/inflight differ')
    first_valid(state,p,current_boot=False)
    require(c.get('frozen_A_005_M0_unchanged') is True and c.get('cache_complete') is True and c.get('no_REAL_DEV') is True,'017 terminal scope incomplete')
    progress=state['durable_child_progress'];require(not progress['unknown_inflight_ids'] and not progress['failed_ids'] and not progress['truncated_tail'],'017 unresolved journal')
    require(all(progress['completed_counts'].get(k,0)==v==progress['entered_counts'].get(k,0) for k,v in counts.items()),'017 terminal durable counts differ')
    immutable=state['_verified_freeze']['immutable_files_sha256'];require(immutable and all(c['immutable_files_sha256'].get(k)==v for k,v in immutable.items()),'017 final immutable bytes differ')
    terminal=state.get('runs/primary/artifacts/status.json') or {}
    require(terminal.get('counters')==counts and terminal.get('native_status')=='COMPLETE','017 terminal status differs')
    require(native['supervisor_pid']==state['START_RECEIPT.json']['supervisor_pid'] and native['child_pid']==c['pid'],'017 terminal PID chain differs')
    return dict(status='native_zero_actual_trial_set_complete',counters=counts,trials=c['trials'],terminal_scope=c['terminal_scope'],
        stage1_gate=c['stage1_gate'],stage2_entered=c['stage2_entered'],training_skipped=False,scientific_acceptance=False)
''')
# The mature live-process health branch is unchanged. Only native0 terminal
# policy changes from two fixed016 slots to the actual gated017 set.
s=s.replace("and done.get('all_two_trials_completed') is True and done.get('pid')==first['pid'] and not done.get('inflight')\n            and done.get('attempted_counters')==counts==p['budget_caps']", "and done.get('pid')==first['pid'] and not done.get('inflight')\n            and done.get('attempted_counters')==counts==terminal_policy017(done,p)")
s=s.replace('full native0 completion','exact gated/full native0 completion')
s=s.replace("process_identity=process_identity,child_bounds017=child_bounds017,native_snapshot=native_snapshot)","process_identity=process_identity,child_bounds017=child_bounds017,native_snapshot=native_snapshot,archive_inventory=archive_inventory017)\n    module.STATE_NAMES=module.STATE_NAMES+('runs/primary/artifacts/STAGE1_GATE.json',)")
extra=r'''
def terminal_policy017(c,p):
    gate=c.get('stage1_gate') or {};full=gate.get('stage2_allowed')
    require(type(full) is bool and gate.get('required_trials')==list(TRIALS[:2]) and gate.get('required_cell_count',0)>0,'017 complete lacks stage1 decision')
    require((full and gate.get('status')=='pass' and gate.get('nonpass')==[]) or
        (not full and gate.get('status')=='closed' and bool(gate.get('nonpass'))),'017 stage1 result inconsistent')
    names=list(TRIALS if full else TRIALS[:2])
    require(c.get('stage2_entered') is full and c.get('terminal_scope')==('full_eight_trials' if full else 'stage1_closed_normally')
        and [r['directory'] for r in c.get('trials',[])]==names and c.get('G_updates_per_trial')==2000
        and all(r.get('G_updates')==2000 for r in c['trials']),'017 terminal trial set does not follow fixed gate')
    return p['budget_caps'] if full else p['stage1_budget_caps']

def archive_inventory017(state,p,*,partial=False,include_heads=False):
    b=operations();root=b.core().unredirected(REMOTE)
    if partial:
        start=state.get('START_RECEIPT.json');require(isinstance(start,dict),'017 no owned dispatch for partial collection')
        if start['boot_id']==state['current_boot_id']:
            launch=state.get('runs/primary/supervisor/launch.json') or {}
            require(all(type(pid) is not int or not Path('/proc',str(pid)).exists() for pid in (start.get('supervisor_pid'),launch.get('child_pid'))),'017 owned job still active; status only')
        verified=dict(status='partial_evidence_only',native=state.get('runs/primary/supervisor/exit_receipt.json'),durable_progress=state['durable_child_progress'],actual_inflight_completion='unknown',scientific_acceptance=False)
    else:verified=complete_valid(state,p)
    names=[]
    for directory,folders,files in os.walk(root/'runs/primary',followlinks=False):
        require(not any((Path(directory)/n).is_symlink() for n in folders+files),'017 redirected owned output')
        names.extend((Path(directory)/n).relative_to(root).as_posix() for n in files)
    for name in (CFG,MANIFEST,'START_RECEIPT.json','REMOTE_INSTALL_RECEIPT.json','DISPATCH_ATTEMPT.json','supervisor_console.log'):
        if (root/name).is_file():names.append(name)
    inventory={name:b.named_record(root/name) for name in sorted(set(names))}
    require(sum(r['bytes'] for r in inventory.values())<=(p['resource_limits']['output_MiB']+16)*1024**2,'017 whole inventory cap')
    if not partial:
        manifest=b.core().read_json(root/'runs/primary/artifacts/ARTIFACT_INVENTORY.json')
        for name,row in manifest['files'].items():
            b.core().safe_name(name)
            require(inventory.get('runs/primary/artifacts/'+name)==row,'017 producer immutable artifact inventory differs')
        for trial in state['runs/primary/artifacts/COMPLETE.json']['trials']:
            name='runs/primary/artifacts/'+trial['directory']+'/COMPLETE.json';require(inventory[name]['sha256']==trial['COMPLETE_sha256'],'017 trial COMPLETE changed')
            r=b.core().read_json(root/name);teacher,mode,seed=trial['directory'].split('_');coverage=r.get('coverage') or {};resources=r.get('resources') or {}
            expected_roles={'shared_parent'} if mode=='paired' else {'student0','student1','teacher0','teacher1'}
            require(r.get('schema')=='n01/wp4-017-trial-complete/v1' and r.get('G_updates')==2000 and r.get('trial')==trial['directory']
                and r.get('teacher')==teacher and r.get('mode')==mode and r.get('seed')==int(seed[-1]) and r.get('frozen_A_005_M0_unchanged') is True
                and set(coverage)==expected_roles and all(v.get('fit_parents')==v.get('unique')==5274 and v.get('all_covered') is True for v in coverage.values()),'017 actual trial identity/fit coverage differs')
            require(resources.get('output_bytes',float('inf'))<=p['run_resource_limits']['output_MiB']*1024**2
                and resources.get('elapsed_seconds',float('inf'))<=p['run_resource_limits']['wall_seconds']
                and resources.get('process_RSS_bytes',float('inf'))<=p['run_resource_limits']['job_tree_RSS_GiB']*1024**3,'017 trial resource cap')
    selected,omitted=b.select_inventory(inventory,include_heads=include_heads)
    return inventory,selected,omitted,verified

'''
s=s.replace('\ndef main(argv=None):', '\n'+extra+'\ndef main(argv=None):')
s=s.replace('two-trial identity','gated eight-trial identity').replace('Only017 policy','Only017 policy')
target=ROOT/'scripts/wp4_017_controller.py'
assert not target.exists(),'preserve existing controller'
target.write_text(s,encoding='utf8',newline='\n')
print(target)
