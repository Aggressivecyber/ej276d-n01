"""One bounded independent prelaunch source review; no real saved arrays/SSH."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
ENG=STAGE/'readback_engineering_01'
OLD=ROOT/'receipts/wp3_015_20261015'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
def load(p):return json.loads(p.read_text('utf8'))
def save(p,obj):
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def node_map(path):
    out={}
    for n in ast.parse(path.read_text('utf8')).body:
        if isinstance(n,ast.FunctionDef):out[n.name]=n
        if isinstance(n,ast.ClassDef):out.update({n.name+'.'+m.name:m for m in n.body if isinstance(m,ast.FunctionDef)})
    return out
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out
def check(name,**detail):checks.append(dict(name=name,passed=True,**detail))

checks=[];plan=load(HERE/'READBACK_PLAN.json');ready=load(HERE/'READBACK_WORKER_READY.json');controller_ready=load(ENG/'CONTROLLER_READY.json')
assert sha(HERE/'READBACK_WORKER_READY.json')=='906ed97985c7a7578b6caca3083db19fc78e5a111e275da3306b5719baae9ca8'
assert sha(ENG/'CONTROLLER_READY.json')=='899813215449bfef0f58fc7afdb901cd08462fbb8933ef14421851240e0a42cf'
assert sha(HERE/'READBACK_PLAN.json')=='9fd6ed29156c5567ef9738835e00fb06f3c2a214bb5694a267b22f06b34e938e'
paths={k:Path(ready[k]['path']) for k in ['worker','kernel','controller','plan','inventory']}
sources={p.relative_to(ROOT).as_posix():{k:v for k,v in ref(p).items() if k!='path'} for p in paths.values()}
for key,p in paths.items():
    assert p.stat().st_size==ready[key]['bytes'] and sha(p)==ready[key]['sha256']
    assert ready[key]==controller_ready[key]
assert sha(paths['kernel'])=='fe1366f35f36e97974f1a4037a36ad858c03dd8453957c1b45b65080f0fee3ac'
assert paths['kernel'].read_bytes()==(OLD/'postrun_independent_01/saved_array_kernels.py').read_bytes()
assert ready['expected_operations']==controller_ready['expected_operations']==plan['expected_operations']==200
assert ready['expected_complete_counters']==controller_ready['expected_complete_counters']==plan['expected_complete_counters']
assert ready['scientific_budget']==controller_ready['scientific_budget']==plan['scientific_budget'] and all(v==0 for v in plan['scientific_budget'].values())
check('final_five_source_plan_inventory_and_READY_bindings',sources=sources)

oldnodes=node_map(OLD/'postrun_independent_01/readback_saved_arrays.py');newnodes=node_map(paths['worker'])
unchanged=['decode_window','lineage','wave_moments','Audit.equal','Audit.member','Audit.load_baselines','Audit.one_bank','Audit.pair','Audit.op','Audit.journal','Audit.state','Audit.guard']
assert all(ast.dump(oldnodes[n],include_attributes=False)==ast.dump(newnodes[n],include_attributes=False) for n in unchanged)
worker_tree=ast.parse(paths['worker'].read_text('utf8'));controller_tree=ast.parse(paths['controller'].read_text('utf8'))
def literals(tree):
    out={}
    for n in tree.body:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name):
                    try:out[t.id]=ast.literal_eval(n.value)
                    except (ValueError,TypeError):pass
    return out
wl,cl=literals(worker_tree),literals(controller_tree)
assert wl['EXPECTED_COUNTS']==plan['expected_complete_counters'] and wl['TRIALS']==('seed0_Q1','seed1_Q1')
assert wl['CONFIG_SHA']==ready['source116_frozen_config_sha256'] and wl['MANIFEST_SHA']==ready['manifest_sha256']
assert wl['INVENTORY_SHA']==ready['inventory']['sha256']
assert cl['FILES']==('readback_control.py','readback_saved_arrays.py','saved_array_kernels.py','READBACK_PLAN.json','BOUND_ORIGINAL_INVENTORY.json')
assert cl['PLAN_SHA']==ready['plan']['sha256'] and cl['SIDE']==plan['sidecar_root'] and cl['ORIGINAL']==plan['original_root']
assert cl['CONFIG']=='READBACK_EXECUTION.json'
imports=[n for n in ast.walk(worker_tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
assert all(not any(x in ast.unparse(n) for x in ['torch','n01','run_wp3','run_wp1']) for n in imports)
assert not any(isinstance(n,ast.Attribute) and n.attr in ('rand','randn','default_rng','random','backward') for n in ast.walk(worker_tree))
check('mature_hard4_derived8_pairwise9x9_and_durable_methods_AST_kernel_unchanged',methods=unchanged,
      no_external_runtime_helper=True,charge_r8_added=False,full_CDF_loss_median_RNG_replay=False)

numeric_path=Path(controller_ready['numeric_artificial_engineering']['path']);numeric=load(numeric_path)
stdlib_path=Path(controller_ready['controller_artificial_engineering']['path']);stdlib=load(stdlib_path)
assert numeric['passed'] and stdlib['passed'] and stdlib['passed_count']==7
assert len(numeric['checks'])>=5 and numeric['negative_count']>=27
assert stdlib['controller']['sha256']==sha(paths['controller'])
delta_path=Path(ready['focused_delta_engineering']['path']);delta=load(delta_path)
assert sha(delta_path)=='ba89795215013fa3b3663bea1a1b46660d68d9996d6bd458a8ba7ccfa7316694'
assert delta['passed'] and delta['exact_one_line_bytes_changed'] and delta['worker']==ready['worker']
assert sha(paths['worker'])=='b42aad59d7d27d51386ca6c3296a6c0ce68b9157c7f94646f179b87548da3b7a'
prior_worker=Path(delta['prior_worker']['path'])
assert sha(prior_worker)==delta['prior_worker']['sha256']==numeric['worker']['sha256']==stdlib['worker']['sha256']
oldlines=prior_worker.read_bytes().splitlines(keepends=True);newlines=paths['worker'].read_bytes().splitlines(keepends=True)
assert len(oldlines)==len(newlines) and sum(a!=b for a,b in zip(oldlines,newlines))==1
assert delta['prior_full_charge_suite']['sha256']==sha(numeric_path) and delta['prior_full_stdlib_suite']['sha256']==sha(stdlib_path)
check('engineering_independent_scalar_charge_fixture_and_health200journal_evidence_reviewed',
    numeric=ref(numeric_path),numeric_groups=len(numeric['checks']),numeric_negatives=numeric['negative_count'],
    controller=ref(stdlib_path),stdlib_groups=7,health_negative_cases=13,existing_groups_rerun=False,
    exact_one_line_final_delta=ref(delta_path),final_source_delta_positive_cases=delta['positive_cases'],final_source_delta_negative_cases=delta['negative_cases'])

# Only the final canonical-residual delta is exercised independently here.
# The existing full artificial groups are reused, not rerun.
tmp=ENG/'independent_prelaunch_tmp_01';assert not tmp.exists();tmp.mkdir()
os.environ.update(TEMP=str(tmp),TMP=str(tmp),TMPDIR=str(tmp),PYTHONDONTWRITEBYTECODE='1')
import numpy as np
w=module(paths['worker'],'independent016_readback_source');w.np=np;w.P,w.R,w.N=3,4,12
fixture=module(ENG/'check_charge016.py','independent016_scalar_fixture')
audit=w.Audit.__new__(w.Audit);audit.errors={}
good=fixture.fixture()
verify=lambda x:w.verify_charge_arrays(*x,bank=0,equal=audit.equal)
verify(good)
alternative=copy.deepcopy(good)
alternative[0]['Q_mu_CPU_recomputed_draws_Vns'][0,0]=np.nextafter(alternative[0]['Q_mu_CPU_recomputed_draws_Vns'][0,0],np.inf)
alternative[0]['Q_mu_GPU_minus_CPU_draws_Vns']=alternative[0]['Q_mu_draws_Vns']-alternative[0]['Q_mu_CPU_recomputed_draws_Vns']
alternative[1]['Q_mu_CPU_GPU_absmax_Vns']=float(np.abs(alternative[0]['Q_mu_GPU_minus_CPU_draws_Vns']).max())
verify(alternative)
bad=copy.deepcopy(alternative);bad[0]['Q_mu_GPU_minus_CPU_draws_Vns'][0,0]=np.nextafter(bad[0]['Q_mu_GPU_minus_CPU_draws_Vns'][0,0],-np.inf)
assert not np.array_equal((bad[0]['Q_mu_draws_Vns']-bad[0]['Q_mu_CPU_recomputed_draws_Vns']).view(np.uint64),bad[0]['Q_mu_GPU_minus_CPU_draws_Vns'].view(np.uint64))
try:verify(bad)
except ValueError as error:assert 'residual algebra' in str(error)
else:raise AssertionError('one-ULP saved residual corruption accepted')
focused=dict(schema='n01/wp3-016-readback-independent-canonical-delta/v1',passed=True,
    source=ref(paths['worker']),checks=['independent scalar baseline passes','legal alternateCPU sum1ULP with coherent saved residual passes',
        'oneULP saved residual corruption rejected despite its small relative discrepancy'],
    artificial_array_cases=3,actual_NPZ_CP_reads=0,new_NN_AD_RNG_training=0,SSH=0,
    bridge='mu-newCPU and savedCPU-newCPU checked by predeclared double-sum bound; saved residual equals elementwise mu-savedCPU bitwise',
    full_existing_artificial_groups_rerun=False)
focused_path=ENG/'INDEPENDENT_CANONICAL_DELTA_RECEIPT.json';save(focused_path,focused)
assert 'torch' not in sys.modules and 'n01' not in sys.modules
check('final_canonical_bridge_independent_focused_check',receipt=ref(focused_path),
    sum_comparison='strict predeclared gamma255 bound',saved_residual_algebra='bitwise exact, no reduction involved',
    bound_formula_value_tolerance='64eps relative, atol0',NumPy_moment_tolerance='128eps relative, atol0')

# Inspect the already-completed artificial durable writer trace, not real I/O.
journals=list(stdlib_path.parent.rglob('operations.jsonl'));assert len(journals)==1
entries={};counts={k:0 for k in plan['expected_complete_counters']};entered=completed=0
with journals[0].open('rb') as f:
    for line in f:
        assert line.endswith(b'\n');row=json.loads(line);key=row['operation_id']
        if row['event']=='entered':
            assert key==entered;entered+=1;entries[key]=row['charges']
        elif row['event']=='completed':
            assert key in entries
            for k,v in entries.pop(key).items():counts[k]+=v
            completed+=1
        else:raise AssertionError('unhealthy artificial operation')
assert entered==completed==200 and not entries and counts==plan['expected_complete_counters']
check('existing_artificial200operation_full_journal_independently_aggregated',operations=200,records=400,
      actual_array_or_SHA_work_injected_counters=False,synthetic_annotations_only=True,counters=counts)

cfg=load(ROOT/'wp3_016_config.json');mf=load(ROOT/'WP3_016_SOURCE_MANIFEST.json')
assert sha(ROOT/'wp3_016_config.json')==wl['CONFIG_SHA'] and sha(ROOT/'WP3_016_SOURCE_MANIFEST.json')==wl['MANIFEST_SHA']
assert len(mf['files'])==116
for name,row in mf['files'].items():assert sha(ROOT/name)==row['sha256'] and (ROOT/name).stat().st_size==row['bytes']
assert plan['finite_resource_budget']['CPU_threads']==2 and plan['finite_resource_budget']['process_RSS_MiB']==1536
assert plan['finite_resource_budget']['virtual_memory_backstop_MiB']==3072 and plan['finite_resource_budget']['wall_seconds']==1800
assert plan['finite_resource_budget']['output_MiB']==64
check('original116_source_plus2_controls_preserved_and_finite_sidecar_resources',source_files=116,original_controls=118,
      resources=controller_ready['resource_limits'],raw_SHA_passes=1,raw_decode_passes=1,original_tree_read_only=True,no_automatic_retry=True)

result=dict(schema='n01/wp3-016-readback-prelaunch-independent-review/v1',UTC=datetime.now(timezone.utc).isoformat(),
    status='PASS_BOUNDED_READBACK_PRELAUNCH_SOURCE_REVIEW',passed=True,blockers=[],sources=sources,
    expected_complete_counters=plan['expected_complete_counters'],expected_operations=200,
    worker_ready=ref(HERE/'READBACK_WORKER_READY.json'),controller_ready=ref(ENG/'CONTROLLER_READY.json'),
    plan=ref(HERE/'READBACK_PLAN.json'),scope_note=ref(HERE/'SCOPE_NOTE.json'),checks=checks,
    canonical_semantics=dict(A_repeat_limit_V=1e-7,A_limit_is_existing013_engineering_gate=True,
        actual_mu_q_v_per_draw_preserved=True,no_broadcast_or_denominator_substitution=True,
        theory_v='a*mu_d+b*mu_d^2',expected_sample_variance='mean(v_d)+samplevar(mu_d,ddof1)',
        theoretical_mean_MCSE='sqrt(sum(v_d))/4',centered_second_moment='mean((q-mu)^2), denominator4',
        saved_residual_bitwise_equals_saved_mu_minus_saved_CPU_sum=True,
        different_legal_sums_compared_using_predeclared_gamma255_bound=True,
        no_relative_comparison_between_cancellation_residuals_from_different_sums=True,
        original_failed_run_amplitude='unknown_not_saved'),
    first_pair_health_bridge=dict(saved_charge_banks_completed_minimum=2,actual_hard_banks_completed_minimum=2,
        current_same_snapshot_PID_boot_startticks_parent_command_cwd_CPU_RSS_VM_resources_and_durable_progress=True,
        native0_terminal_branch_requires_COMPLETE_and_exit0=True,full_terminal_validation_requires_exact200_operations_and16_counters=True),
    complete_journal_bridge_covered=True,first_pair_health_bridge_covered=True,
    controller_file_slots=5,execution_control_slots_total=6,new_external_runtime_helpers=[],
    scientific_budget=plan['scientific_budget'],resource_limits=controller_ready['resource_limits'],
    single_pass_raw_files=112,single_pass_raw_bytes=2169655337,hard_bank_pairs=24,
    old015_numeric_or_worker_rerun=False,registration_performed=False,native_or_SSH_started=False,
    actual_review_work=dict(source_JSON_only_except3_artificial_in_memory_cases=True,real_raw_or_CP_payload_reads=0,
        actual_NPZ_opens=0,new_NN_AD_RNG_training=0,SSH=0,source_or_main_metadata_changes=0),
    next_step='Root may bind the exact five source/control files plus execution registration, commit/backup, and authorize the one bounded CPU task. No additional review layer is required by this receipt.',
    scientific_acceptance=False,weights_promoted=False)
path=HERE/'PRELAUNCH_REVIEW.json';save(path,result)
assert load(path)['passed'] and not load(path)['blockers']
print(json.dumps(dict(passed=True,receipt=ref(path),sources=sources,expected_operations=200,checks=len(checks)),indent=2))
