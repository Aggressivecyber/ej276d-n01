"""016 policy adapter over the exact014 archive/journal/owned-tree controller.

The reusable operation algorithms remain pinned. Only016 policy, source freeze,
two-trial identity and collection limit differ; no numerical imports occur here.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import tarfile
import time
import types

RUN_ID='wp3_016_20261015'
REMOTE='/home/rylan/projects/ej276d-n01/wp3_016_runtime_fix02_20261015'
OLD015='/home/rylan/projects/ej276d-n01/wp3_015_20261015'
CFG,MANIFEST='wp3_016_config.json','WP3_016_SOURCE_MANIFEST.json'
CONTROLLER,DRIVER='scripts/wp3_016_controller.py','scripts/run_wp3_016.py'
OWNED=(CONTROLLER,DRIVER,'src/n01/model016.py')
BASE_SHA='3480825e94e954a46c2c633e5df6ddef514e2b7def1332bb2e7688e78ae041a3'
PRIOR_SHA='f1255dbff0e47edb7ec20e76c7d90057201f9a56f1184b7896f0c8cb0a67825a'
T_SHA='f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b'
TRIALS=('seed0_Q1','seed1_Q1')
HYPOTHESES=('fast','slow','common_level')
SELF_ROOT=Path(__file__).resolve().parents[1] if __file__ not in ('<stdin>','<string>') else Path(REMOTE)
_OPS=None


def process_identity(pid):
    """Owned Linux identity; same read/verify method audited in015 CPU readback."""
    require(os.name=='posix' and type(pid) is int and pid>0,'016 owned Linux PID required')
    base=Path('/proc')/str(pid)
    try:
        fields=(base/'stat').read_text().rsplit(')',1)[1].split()
        start_ticks=int(fields[19]);command=(base/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        memory={r.split(':',1)[0]:int(r.split()[1])*1024 for r in (base/'status').read_text().splitlines() if r.startswith(('VmRSS:','VmSize:'))}
        result=dict(alive=fields[0] not in ('Z','X'),pid=pid,parent_pid=int(fields[1]),start_ticks=start_ticks,
            command=[v.decode('utf8',errors='replace') for v in command],cwd=os.readlink(base/'cwd'),CPU_affinity=sorted(os.sched_getaffinity(pid)),**memory)
        require(int((base/'stat').read_text().rsplit(')',1)[1].split()[19])==start_ticks,'016 PID changed while reading identity')
        return result
    except (OSError,ValueError,IndexError) as error:return dict(alive=False,pid=pid,error=str(error))


def child_bounds016():
    """The registered twelve numeric CPU threads also have an owned CPU set."""
    available=sorted(os.sched_getaffinity(0));require(len(available)>=12,'016 CPU12 unavailable')
    os.sched_setaffinity(0,available[:12])


def native_snapshot(pins,*,partial=False):
    b=operations();p,state=b._base_native_snapshot(pins,partial=partial)
    start=state.get('START_RECEIPT.json') or {};launch=state.get('runs/primary/supervisor/launch.json') or {}
    state['live_processes']={name:process_identity(pid) for name,pid in [('supervisor',start.get('supervisor_pid')),('worker',launch.get('child_pid'))] if type(pid) is int}
    if not state.get('runs/primary/supervisor/exit_receipt.json'):
        try:
            supervisor=state['live_processes'].get('supervisor') or {};worker=state['live_processes'].get('worker') or {}
            require(supervisor.get('alive') is True and worker.get('alive') is True,'016 no current owned process chain')
            helpers=b.supervisor_tools();ids,rss=helpers.process_tree(start['supervisor_pid'])
            state['current_resource_health']=dict(collected_by_native_snapshot=True,snapshot_UTC=state['UTC'],owned_process_tree=sorted(ids),
                RSS_bytes=rss,GPU_bytes=helpers.gpu_bytes(ids),output_bytes=helpers.output_bytes(Path(REMOTE)/'runs/primary'),
                elapsed_seconds=time.monotonic()-launch['supervisor_started_monotonic'],resource_limits=p['resource_limits'])
        except Exception as error:state['current_resource_health']=dict(collected_by_native_snapshot=False,error=repr(error))
    return p,state


def same_snapshot_health016(state,p,start,launch,first):
    require(not state.get('protection_errors'),'016 same snapshot immutable protections unhealthy')
    require(start.get('boot_id')==launch.get('boot_id') and start.get('supervisor_start_ticks')==launch.get('supervisor_start_ticks')
        and type(launch.get('supervisor_start_ticks')) is int and launch['supervisor_start_ticks']>0
        and type(launch.get('child_start_ticks')) is int and launch['child_start_ticks']>0,'016 launch boot/start identity differs')
    status=state.get('runs/primary/artifacts/status.json') or {};durable=state.get('durable_child_progress') or {}
    require(status.get('run_id')==RUN_ID and status.get('pid')==first['pid'] and all(status.get('counters',{}).get(k,0)>=v for k,v in first['counters'].items()),'016 current status behind firstG1')
    require(durable.get('failed_ids')==[] and durable.get('truncated_tail') is False and
        all(durable.get('completed_counts',{}).get(k,0)>=v for k,v in first['counters'].items()),'016 same snapshot durable progress unhealthy or behind firstG1')
    native=state.get('runs/primary/supervisor/exit_receipt.json') or {}
    if native:
        done=state.get('runs/primary/artifacts/COMPLETE.json') or {};counts=done.get('counters') or {}
        require(native.get('child_native_exit_code')==native.get('exit_code')==0 and native.get('reason') is None and native.get('error') is None
            and native.get('supervisor_pid')==start['supervisor_pid'] and native.get('child_pid')==first['pid']
            and done.get('schema')=='n01/wp3-016-batch-complete/v1' and done.get('native_status')=='COMPLETE'
            and done.get('all_two_trials_completed') is True and done.get('pid')==first['pid'] and not done.get('inflight')
            and done.get('attempted_counters')==counts==p['budget_caps'],'016 terminal firstG1 lacks full native0 completion')
        require(not durable.get('unknown_inflight_ids') and all(durable.get('entered_counts',{}).get(k,0)==v==durable.get('completed_counts',{}).get(k,0) for k,v in counts.items()),'016 terminal journal incomplete')
        require(native.get('elapsed_seconds',float('inf'))<=p['resource_limits']['wall_seconds'] and
            native.get('peak_RSS_bytes',float('inf'))<=p['resource_limits']['job_tree_RSS_GiB']*1024**3 and
            native.get('peak_GPU_bytes',float('inf'))<=p['resource_limits']['GPU_GiB']*1024**3 and
            native.get('peak_output_bytes',float('inf'))<=p['resource_limits']['output_MiB']*1024**2,'016 terminal resources exceed cap')
        return 'native_completed_zero'
    require(start['boot_id']==state.get('current_boot_id'),'016 active proof from another boot')
    live=state.get('live_processes') or {};supervisor=live.get('supervisor') or {};worker=live.get('worker') or {}
    require(supervisor.get('alive') is True and worker.get('alive') is True and supervisor.get('pid')==start['supervisor_pid'] and worker.get('pid')==first['pid']
        and supervisor.get('start_ticks')==launch['supervisor_start_ticks'] and worker.get('start_ticks')==launch['child_start_ticks']
        and worker.get('parent_pid')==start['supervisor_pid'],'016 historical firstG1 has no current owned process chain')
    require(supervisor.get('cwd')==worker.get('cwd')==REMOTE and supervisor.get('command')==start.get('command')
        and worker.get('command')==launch.get('command') and REMOTE+'/'+CONTROLLER in supervisor.get('command',[])
        and REMOTE+'/'+DRIVER in worker.get('command',[]) and len(worker.get('CPU_affinity',[]))==p['CPU_threads']
        and set(worker['CPU_affinity'])<=set(supervisor.get('CPU_affinity',[])),'016 current command/cwd/CPU identity differs')
    commands=[supervisor.get('command',[]),worker.get('command',[])];config_sha=state['_verified_freeze']['config_sha256']
    require(all('--config-sha256' in cmd and cmd[cmd.index('--config-sha256')+1]==config_sha for cmd in commands)
        and '--mode' in commands[0] and commands[0][commands[0].index('--mode')+1]=='supervise'
        and '--config' in commands[1] and commands[1][commands[1].index('--config')+1]==REMOTE+'/'+CFG,
        '016 live command is not bound to frozen config and supervision mode')
    resource=state.get('current_resource_health') or {};limits=p['resource_limits']
    require(resource.get('collected_by_native_snapshot') is True and resource.get('snapshot_UTC')==state.get('UTC') and resource.get('resource_limits')==limits
        and {start['supervisor_pid'],first['pid']}<=set(resource.get('owned_process_tree',[]))
        and resource.get('elapsed_seconds',float('inf'))<=limits['wall_seconds'] and resource.get('RSS_bytes',float('inf'))<=limits['job_tree_RSS_GiB']*1024**3
        and resource.get('GPU_bytes',float('inf'))<=limits['GPU_GiB']*1024**3 and resource.get('output_bytes',float('inf'))<=limits['output_MiB']*1024**2
        and supervisor.get('VmRSS',float('inf'))+worker.get('VmRSS',float('inf'))<=limits['job_tree_RSS_GiB']*1024**3,
        '016 same snapshot resource health missing or outside limits')
    return 'current_owned_process_chain_and_resources_verified'


def require(value,message):
    if not value:raise ValueError(message)


def factory_contract(source):
    tree=ast.parse(source);names={'seed_set016','settings016','budget_caps016'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==names,'016 pure factories missing')
    for node in nodes:
        require(not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.With,ast.Global,ast.Nonlocal)) for n in ast.walk(node)),'016 factory imported runtime')
        require(all(isinstance(n.func,ast.Name) and n.func.id in ('dict','list','len','seed_set016') for n in ast.walk(node) if isinstance(n,ast.Call)),
            '016 factory executes nonliteral helper')
    scope=dict(RUN_ID=RUN_ID,HYPOTHESES=HYPOTHESES)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<016-stdlib-factories>','exec'),scope)
    return scope['settings016'](),scope['budget_caps016']


def validate_config(p,*,frozen=True,driver_source=None):
    require(p.get('schema')=='n01/wp3-016-config/v1' and p.get('run_id')==RUN_ID and p.get('remote_root')==REMOTE,'016 identity/root differs')
    require(not frozen or p.get('status')=='frozen','016 draft cannot execute')
    source=driver_source if driver_source is not None else (SELF_ROOT/DRIVER).read_bytes();reference,budget=factory_contract(source)
    for key,value in reference.items():
        if key=='status':continue
        require(p.get(key)==value,'016 setting differs: '+key)
    require(p.get('budget_caps')==budget(p),'016 exact operation caps differ')
    required={'dataset','statistics','data_statistics','base_config','deterministic','noise005','source_information','integral_statistics','M0_head_seed0','M0_head_seed1'}
    require(required<=set(p.get('inputs',{})),'016 bound inputs incomplete')
    require(not any(k.startswith('mean_seed') or k.startswith('C1_head') for k in p['inputs']),'016 obsolete015 mean/C1 inputs present')
    require({'observation','data_roles','acceptance','tolerance_table'}<=set(p.get('contracts',{})),'016 contracts incomplete')
    require(p['contracts']['tolerance_table']['sha256']==T_SHA,'016 original T changed')
    require(REMOTE in p.get('allowed_input_roots',[]),'016 allowed launch root missing')
    require(not frozen or (set(OWNED)<=set(p.get('source_files',{})) and CFG not in p['source_files'] and MANIFEST not in p['source_files']),
        '016 own source inventory missing or recursive')
    return reference


def freeze(repo,output,expected_draft_sha,*,artificial=False):
    b=operations();repo=b.core().unredirected(repo)
    require(b.sha(repo/CFG)==expected_draft_sha,'016 draft SHA differs')
    p=copy.deepcopy(b.core().read_json(repo/CFG));validate_config(p,frozen=False,driver_source=(repo/DRIVER).read_bytes())
    prior_file=repo/'WP3_SOURCE_MANIFEST.json';require(b.sha(prior_file)==PRIOR_SHA,'016 original105 manifest changed')
    prior=b.core().read_json(prior_file);require(len(prior['files'])==105,'016 original105 inventory changed')
    old_config=b.core().read_json(repo/'wp3_015_config.json')
    require(b.sha(repo/'wp3_015_config.json')==prior['config_sha256'],'016 original015 configuration changed')
    checked={str(repo/CFG):expected_draft_sha,str(prior_file):PRIOR_SHA}
    paths={n:repo/n for n in set(prior['files'])|set(OWNED)|{'WP3_SOURCE_MANIFEST.json'}}
    protections=copy.deepcopy(old_config.get('protection_files',{}))
    for row in protections.values():b.checked_local(row,checked)
    for name,row in prior['files'].items():
        require(b.named_record(repo/name)==row,'016 original105 source changed: '+name)
        protections['wp3_015:'+name]=dict(path=OLD015+'/'+name,local_path=str(repo/name),**row)
    for name,digest in (('WP3_SOURCE_MANIFEST.json',PRIOR_SHA),('wp3_015_config.json',prior['config_sha256'])):
        protections['wp3_015:'+name]=dict(path=OLD015+'/'+name,local_path=str(repo/name),sha256=digest,bytes=(repo/name).stat().st_size)
    for records in (p['inputs'],p['contracts']):
        for row in records.values():
            require(row.get('local_path') and Path(row['local_path']).is_file(),'016 inputs/contracts must have explicit local SHA verification bytes')
            local=b.checked_local(row,checked);native=PurePosixPath(row['path'])
            if native.is_relative_to(PurePosixPath(REMOTE)):
                name=native.relative_to(PurePosixPath(REMOTE)).as_posix();b.core().safe_name(name)
                require(not name.endswith(('.npz','.npy','.pt','.pth','.bin')),'016 source package cannot provision scientific numeric inputs')
                require(name not in paths or b.named_record(paths[name])==b.named_record(local),'016 contract/source alias differs')
                paths[name]=local
    source_records={name:b.named_record(path) for name,path in sorted(paths.items())}
    closure=b.project_import_closure(source_records,lambda n:paths[n].read_bytes())
    p['source_files']={n:dict(path=REMOTE+'/'+n,**row) for n,row in source_records.items()}
    p['protection_files']=protections;p.update(status='frozen',frozen_UTC=b.core().now())
    validate_config(p,driver_source=(repo/DRIVER).read_bytes())
    roots=[PurePosixPath(v) for v in p['allowed_input_roots']]
    for records in (p['source_files'],p['inputs'],p['contracts'],p['protection_files']):
        require(all(any(PurePosixPath(row['path']).is_relative_to(r) for r in roots) for row in records.values()),'016 frozen path outside allowed roots')
    output=b.core().fresh_directory(output,local=True);b.core().write_new(output/CFG,p);config_sha=b.sha(output/CFG)
    manifest=dict(schema='n01/wp3-016-source/v1',run_id=RUN_ID,config_sha256=config_sha,frozen_UTC=p['frozen_UTC'],files=source_records,
        prior_WP3_manifest_sha256=PRIOR_SHA,prior_source_file_count=105,project_import_closure=closure,
        before_scientific_decode=True,public_source_only=True,scientific_inputs_copied=False)
    b.core().write_new(output/MANIFEST,manifest);package=output/b.SOURCE_ARCHIVE
    with tarfile.open(package,'x:gz') as bundle:
        for name,path in sorted(paths.items()):b.add_regular(bundle,path,name)
        for name in (CFG,MANIFEST):b.add_regular(bundle,output/name,name)
    pins=dict(source_package_sha256=b.sha(package),config_sha256=config_sha,source_manifest_sha256=b.sha(output/MANIFEST),controller_sha256=source_records[CONTROLLER]['sha256'])
    _,_,inventory=b.read_bundle(package,pins)
    for path,digest in checked.items():require(b.sha(path)==digest,'016 input/source changed during freeze')
    for name,path in paths.items():require(b.named_record(path)==source_records[name],'016 source changed during freeze')
    result=dict(schema='n01/wp3-016-freeze/v1',run_id=RUN_ID,UTC=b.core().now(),**pins,source_package=str(package),source_package_bytes=package.stat().st_size,
        source_file_count=len(source_records),source_inventory=inventory,protected_source_records=len(protections),project_import_closure=closure,
        all_original_files_unchanged=True,artificial_only=artificial,scientific_started=False,array_decodes=0,checkpoint_loads=0,model_forwards=0)
    b.core().write_new(output/'FREEZE_RECEIPT.json',result);return result


def first_valid(state,p,*,current_boot=True):
    b=operations();b.same_snapshot_not_failed(state)
    start=state.get('START_RECEIPT.json');launch=state.get('runs/primary/supervisor/launch.json')
    first=state.get('runs/primary/artifacts/FIRST_UPDATE_PROOF.json');init=state.get('runs/primary/artifacts/INIT_PROOF.json')
    require(all(isinstance(v,dict) for v in (start,launch,first,init)),'016 first real G1 not ready')
    require(start['supervisor_pid']==launch['supervisor_pid'] and launch['child_pid']==first['pid']==init['pid'],'016 PID proof differs')
    require(not current_boot or start['boot_id']==state['current_boot_id'],'016 proof is from another boot')
    require(first.get('run_id')==RUN_ID and first.get('artificial_only') is False and init.get('artificial_only') is False,'016 artificial/wrong proof')
    require(init.get('native_initialized') is True and init.get('charge_initialization_complete') is True and init.get('calibration_all_Q_positive') is True
        and init.get('mean_adapter_enabled') is False and init.get('first_trial')==TRIALS[0] and init.get('fresh_AdamW') is True,'016 initialization incomplete')
    require(init['counters']['checkpoint_loads']==3 and init['counters']['optimizer_created']==1 and init['counters']['G_updates']==0
        and init['counters']['calibration_A_rows']==6593 and init['counters']['charge_initialization_fits']==1,'016 initial calibration/CP boundary differs')
    require(first.get('actual_G_updates')==first.get('batch_completed_G_updates')==1 and first.get('trial')==TRIALS[0] and first.get('candidate')=='Q1'
        and first.get('gradient_nonzero') is True and first.get('actual_parameter_update_nonzero') is True
        and first.get('frozen_A_005_unchanged') is True and first.get('protected_M0_shape_unchanged') is True,'016 first actual optimizer step missing')
    require(first.get('trainable_parameters')==init.get('trainable_parameters')==2 and first.get('trainable_parameter_names')==init.get('trainable_parameter_names')==['theta_a','theta_b'],'016 whitelist differs')
    require(first['frozen_state_sha256']==init['frozen_state_sha256'] and first['M0_head_sha256']==init['M0_head_sha256'],'016 frozen tensor references differ')
    b.validate_counters(p,first['attempted_counters'],first['counters']);counts=first['counters']
    require(counts['G_updates']==counts['optimizer_steps']==counts['new_AD_queries']==counts['training_backward_queries']==1 and counts['checkpoint_loads']==3
        and counts['charge_initialization_fits']==1 and counts['calibration_A_rows']==6593 and not first.get('inflight'),'016 first G1 counters incomplete')
    require(counts['head_actual_parameter_forward_rows']==counts['head_total_kernel_rows']==counts['noise005_rows']==counts['charge_module_rows']==counts['charge_latent_scalar_draws']
        and counts['head_total_kernel_rows']>=1024 and counts['hard_ADC_rows']>=3072 and first.get('mean_adapter_enabled') is False,'016 first training family missing')
    expected=state['_verified_freeze']['immutable_files_sha256']
    require(expected and all(first['immutable_files_sha256'].get(k)==v and init['immutable_files_sha256'].get(k)==v for k,v in expected.items()),'016 first immutable files differ')
    health=same_snapshot_health016(state,p,start,launch,first)
    return dict(status='actual_first_charge_G1_and_analytic_initialization_verified',supervisor_pid=start['supervisor_pid'],child_pid=first['pid'],
        actual_completed_G_updates=1,actual_first_G1_verified=True,same_snapshot_health=health,scientific_acceptance=False,
        current_batch_completed_G_updates=(state.get('runs/primary/artifacts/status.json') or {}).get('counters',{}).get('G_updates'))


def complete_valid(state,p):
    b=operations();b.same_snapshot_not_failed(state)
    native=state.get('runs/primary/supervisor/exit_receipt.json') or {};c=state.get('runs/primary/artifacts/COMPLETE.json')
    require(native.get('child_native_exit_code')==native.get('exit_code')==0 and native.get('reason') is None and native.get('error') is None,'016 native failure/resource stop')
    require(isinstance(c,dict) and c.get('schema')=='n01/wp3-016-batch-complete/v1' and c.get('run_id')==RUN_ID
        and c.get('native_status')=='COMPLETE' and c.get('artificial_only') is False,'016 producer terminal missing')
    counts=c['counters'];b.validate_counters(p,c['attempted_counters'],counts)
    require(c['attempted_counters']==counts and not c.get('inflight'),'016 unresolved operation')
    first_valid(state,p,current_boot=False)
    require(c.get('all_two_trials_completed') is True and c.get('all_frozen_A_005_unchanged') is True and c.get('all_M0_heads_unchanged') is True
        and c.get('charge_initialization_complete') is True and c.get('no_DEV') is True,'016 full trial scope incomplete')
    require(c.get('G_updates_per_trial')==2000 and [r['directory'] for r in c['trials']]==list(TRIALS) and all(r['G_updates']==2000 for r in c['trials']),'016 serial trials differ')
    require(counts==p['budget_caps'],'016 exact complete operation budgets differ');progress=state['durable_child_progress']
    require(not progress['unknown_inflight_ids'] and not progress['failed_ids'] and not progress['truncated_tail'],'016 unresolved durable operation')
    require(all(progress['completed_counts'].get(k,0)==v==progress['entered_counts'].get(k,0) for k,v in counts.items()),'016 terminal journal counts differ')
    expected=state['_verified_freeze']['immutable_files_sha256']
    require(expected and all(c['immutable_files_sha256'].get(k)==v for k,v in expected.items()),'016 final immutable files differ')
    terminal=state.get('runs/primary/artifacts/status.json') or {}
    require(terminal.get('counters')==counts and terminal.get('native_status')=='COMPLETE','016 terminal status differs')
    require(native['supervisor_pid']==state['START_RECEIPT.json']['supervisor_pid'] and native['child_pid']==c['pid'],'016 terminal PID chain differs')
    require(native.get('elapsed_seconds',float('inf'))<=p['resource_limits']['wall_seconds'] and native.get('peak_output_bytes',float('inf'))<=p['resource_limits']['output_MiB']*1024**2,'016 terminal resources exceed cap')
    return dict(status='native_and_two_trial_producer_complete',counters=counts,trials=c['trials'],training_skipped=False,scientific_acceptance=False,root_candidate_choice_pending=True)


def operations():
    global _OPS
    if _OPS is not None:return _OPS
    candidates=[SELF_ROOT/'scripts/wp2_controller.py',Path('/home/rylan/projects/ej276d-n01/wp2_014_20261015')/'scripts/wp2_controller.py']
    path=next((p for p in candidates if p.is_file()),None)
    require(path is not None and hashlib.sha256(path.read_bytes()).hexdigest()==BASE_SHA,'016 exact reused014 operations unavailable')
    tree=ast.parse(path.read_bytes(),filename=str(path))
    # Label adaptation is identical to015. The two archive trial indices are
    # generalized to the registered candidate count (one in016, two in014/015).
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            value=node.value
            if value.startswith('n01/wp2-'):node.value=value.replace('n01/wp2-','n01/wp3-016-',1)
            elif value.startswith('wp2_') or value.startswith('.wp2.'):node.value=value.replace('wp2_','wp3_016_',1).replace('.wp2.','.wp3_016.',1)
        if isinstance(node,ast.FunctionDef) and node.name=='archive_inventory':
            for child in ast.walk(node):
                if isinstance(child,ast.BinOp) and isinstance(child.left,ast.Name) and child.left.id=='index' and isinstance(child.op,(ast.Mod,ast.FloorDiv)):
                    require(isinstance(child.right,ast.Constant) and child.right.value==2,'016 unexpected pinned archive index')
                    child.right=ast.Call(func=ast.Name(id='len',ctx=ast.Load()),args=[ast.Subscript(value=ast.Name(id='p',ctx=ast.Load()),slice=ast.Constant(value='candidates'),ctx=ast.Load())],keywords=[])
    # Extend only the launch/start identity records and child CPU binding. The
    # pinned supervisor's owned process/resource/termination loop is unchanged.
    identity_adaptations=dict(launch=0,start=0,child_CPU_binding=0)
    for function in [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('supervise','native_start')]:
        for node in ast.walk(function):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='Popen' and function.name=='supervise':
                node.keywords.append(ast.keyword(arg='preexec_fn',value=ast.Name(id='child_bounds016',ctx=ast.Load())))
                identity_adaptations['child_CPU_binding']+=1
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='dict':continue
            fields={k.arg for k in node.keywords}
            extra={}
            if function.name=='supervise' and {'supervisor_pid','child_pid','command','UTC'}<=fields and 'child_native_exit_code' not in fields:
                extra=dict(boot_id='core().boot_id()',supervisor_start_ticks="process_identity(os.getpid())['start_ticks']",
                    child_start_ticks="process_identity(child.pid)['start_ticks']",supervisor_started_monotonic='clock')
                identity_adaptations['launch']+=1
            elif function.name=='native_start' and 'exclusive_scientific_lock' in fields:
                extra=dict(supervisor_start_ticks="process_identity(child.pid)['start_ticks']")
                identity_adaptations['start']+=1
            node.keywords.extend(ast.keyword(arg=k,value=ast.parse(v,mode='eval').body) for k,v in extra.items())
    module=types.ModuleType('wp3_016_reused014_stdlib');module.__file__=__file__
    require(identity_adaptations==dict(launch=1,start=1,child_CPU_binding=1),'016 pinned launch adaptation incomplete')
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
    module._base_native_snapshot=module.native_snapshot
    module.identity_adaptations=identity_adaptations
    module.__dict__.update(RUN_ID=RUN_ID,REMOTE=REMOTE,CFG=CFG,MANIFEST=MANIFEST,CONTROLLER=CONTROLLER,DRIVER=DRIVER,
        OWNED=OWNED,TRIALS=TRIALS,SELF_ROOT=SELF_ROOT,SOURCE_ARCHIVE='wp3_016_source.tar.gz',PRIOR_MANIFEST_SHA=PRIOR_SHA,
        COLLECTION_LIMIT=256*1024**2,factory_contract=factory_contract,validate_config=validate_config,freeze=freeze,first_valid=first_valid,complete_valid=complete_valid,
        process_identity=process_identity,child_bounds016=child_bounds016,native_snapshot=native_snapshot)
    _OPS=module;return module


def main(argv=None):return operations().main(argv)


if __name__=='__main__':raise SystemExit(main())
