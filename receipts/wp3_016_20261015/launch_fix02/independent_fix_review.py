"""Independent local fix02 source/evidence review. No scientific model execution."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import copy
import difflib
import hashlib
import itertools
import json
import math
import os
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
OUT=HERE/'independent_artificial_02'
TMP=HERE/'independent_tmp_02'
assert ROOT.drive.upper() in ('E:','D:') and not OUT.exists() and not TMP.exists()
TMP.mkdir()
os.environ.update(TEMP=str(TMP),TMP=str(TMP),TMPDIR=str(TMP),MPLCONFIGDIR=str(TMP),PYTHONDONTWRITEBYTECODE='1')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import run_wp3_016 as d
import wp3_016_controller as c
from wp1_ops import sha,write_new

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
def canonical(node):return ast.dump(node,include_attributes=False)
def check(name,**evidence):
    checks.append(dict(name=name,passed=True,**evidence))

started=time.monotonic();checks=[]
ready=read(HERE/'ENGINEERING_FIX_READY.json')
assert sha(HERE/'ENGINEERING_FIX_READY.json')=='2e32de2b805be4f31ceddde73be436222e6065e40aba6f69c63d360e485e869f'
sources={n:{k:v for k,v in ref(ROOT/n).items() if k!='path'} for n in d.OWNED_FILES}
assert sources==ready['sources']
expected={
 'scripts/run_wp3_016.py':'5d3b4892bbe564c887cab29346a11e92edd9eab7ba1d512ca9143280732da4c0',
 'scripts/wp3_016_controller.py':'ee1780358314b2fb7c425d21bddd5df16184642b2f5b1317c1e1b6a00d403cfb',
 'src/n01/model016.py':'04c63f91ea5405937a940a9aee6eaed876db79149c857bfbe6ed8cfe6dcb138b'}
assert {n:r['sha256'] for n,r in sources.items()}==expected
numeric=read(ROOT/ready['numerical_receipt']['path']);health=read(ROOT/ready['stdlib_receipt']['path'])
for name in ['source_diff','diagnosis','numerical_receipt','stdlib_receipt']:
    assert ref(ROOT/ready[name]['path'])==ready[name]
assert numeric['passed'] and numeric['sources']==sources and numeric['negative_cases']==12
assert len(numeric['checks'])==5 and all(v['passed'] for v in numeric['checks'])
assert health['passed'] and health['source_sha256']==expected[c.CONTROLLER]
assert len(health['checks'])==5 and len(health['checks'][2]['negative_cases'])==15
check('final_source_hashes_and_ready_test_bindings',sources=sources)

oldmf=read(STAGE/'runtime_package_fix01/WP3_016_SOURCE_MANIFEST.json')
oldpackage=STAGE/'runtime_package_fix01/wp3_016_source.tar.gz'
assert sha(oldpackage)=='524ba471ac24d376211349280237e02562dfe797b2921fc25b01c46aac8e688e'
assert sha(STAGE/'runtime_package_fix01/wp3_016_config.json')=='c0fe6408b53969da9f0bd70c01bf7649cfabfbf1864752dc5696bb8c9b6d5da9'
assert sha(STAGE/'runtime_package_fix01/WP3_016_SOURCE_MANIFEST.json')=='1488217be443fb49c60db4fcf3d852fcda8ba33507ce7e5ba2ee2f3d1e4d1327'
with tarfile.open(oldpackage,'r:gz') as archive:
    assert len(archive.getmembers())==117
    for name in [c.DRIVER,c.CONTROLLER]:
        before=HERE/'before'/Path(name).name
        assert before.read_bytes()==archive.extractfile(name).read()
        assert ref(before)==ready['before'][name]
    assert archive.extractfile('src/n01/model016.py').read()==(ROOT/'src/n01/model016.py').read_bytes()
changed=[n for n,row in oldmf['files'].items() if sha(ROOT/n)!=row['sha256']]
assert sorted(changed)==sorted([c.DRIVER,c.CONTROLLER])
prior=read(ROOT/'WP3_SOURCE_MANIFEST.json')
assert len(prior['files'])==105 and all(sha(ROOT/n)==row['sha256'] for n,row in prior['files'].items())
assert sha(ROOT/'wp3_015_config.json')==prior['config_sha256']
assert sha(ROOT/'WP3_SOURCE_MANIFEST.json')==c.PRIOR_SHA
check('old_failure_source_and_package_preserved_original105_plus107_controls_unchanged',
      old_package_members=117,old_sources=115,current_changed_sources=changed,other_old_sources_unchanged=113)

oldtree=ast.parse((HERE/'before/run_wp3_016.py').read_bytes());newtree=ast.parse((ROOT/c.DRIVER).read_bytes())
newhelpers={'source_input_evidence016','repeat_detail016','charge_draw_moments016','save_charge_bank016'}
oldengine=next(n for n in oldtree.body if isinstance(n,ast.ClassDef) and n.name=='Engine016')
oldeval=next(n for n in oldengine.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
normalized=copy.deepcopy(newtree)
normalized.body=[n for n in normalized.body if not(isinstance(n,ast.FunctionDef) and n.name in newhelpers)]
newengine=next(n for n in normalized.body if isinstance(n,ast.ClassDef) and n.name=='Engine016')
newengine.body=[copy.deepcopy(oldeval) if isinstance(n,ast.FunctionDef) and n.name=='evaluate' else n for n in newengine.body]
assert canonical(oldtree)==canonical(normalized)
oldctree=ast.parse((HERE/'before/wp3_016_controller.py').read_bytes());newctree=ast.parse((ROOT/c.CONTROLLER).read_bytes())
for tree in [oldctree,newctree]:
    next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets)).value=ast.Constant('REMOTE_NORMALIZED')
assert canonical(oldctree)==canonical(newctree)
delta=''
for name in [c.DRIVER,c.CONTROLLER]:
    delta+=''.join(difflib.unified_diff((HERE/'before'/Path(name).name).read_text('utf8').splitlines(keepends=True),
        (ROOT/name).read_text('utf8').splitlines(keepends=True),fromfile='before/'+name,tofile='after/'+name))
assert delta==(HERE/'SOURCE_DIFF.patch').read_text('utf8')
oldp,oldbudget=c.factory_contract((HERE/'before/run_wp3_016.py').read_bytes())
p,budget=c.factory_contract((ROOT/c.DRIVER).read_bytes())
caps=budget(p);design=read(ROOT/'wp3_016_design.json')
assert p==oldp==ready['settings_factory'] and caps==oldbudget(oldp)==ready['budget_caps']==design['budget_caps_design']
assert c.REMOTE=='/home/rylan/projects/ej276d-n01/wp3_016_runtime_fix02_20261015'
assert c.RUN_ID=='wp3_016_20261015' and list(c.TRIALS)==['seed0_Q1','seed1_Q1']
check('complete_AST_delta_scoped_to_four_evidence_helpers_evaluate_and_REMOTE_science_factories_equal',
      protected_unchanged=['model016','Runtime016','calibrate_charge','Engine016.__init__','Engine016.step','loss','streams','selection','budget_caps016'],
      new_external_helpers=[],fixed_two_trials=list(c.TRIALS))

# Replay only the engineering script's pure-array checks (through line120).
# Drop its torch thread-setting statement; no neural fixture/import/forward follows.
testtree=ast.parse((HERE/'check_fix02.py').read_bytes())
main=next(n for n in testtree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
reject=next(n for n in testtree.body if isinstance(n,ast.FunctionDef) and n.name=='reject')
prefix=[n for n in main.body if n.lineno<=120 and n.lineno!=37]
env=dict(__file__=str(HERE/'check_fix02.py'),ROOT=ROOT,Path=Path,ast=ast,copy=copy,hashlib=hashlib,itertools=itertools,
         json=json,math=math,os=os,sys=sys,time=time,np=np,d=d,sha=sha,write_new=write_new)
argv=sys.argv;sys.argv=[str(HERE/'check_fix02.py'),str(OUT)]
try:exec(compile(ast.fix_missing_locations(ast.Module(body=[reject,*prefix],type_ignores=[])),str(HERE/'check_fix02.py')+'::array_only','exec'),env)
finally:sys.argv=argv
assert env['negatives']==12 and len(env['checks'])==4 and all(v['passed'] for v in env['checks'])
assert 'torch' not in sys.modules
raw=env['raw'];A=raw.copy();A[1,126]+=2e-7;A[1,127]-=2e-7
env['reject'](lambda:env['bank']('zero_Q_cancelling_waveform_error_saved_then_failed',A),'frozen013 replay tolerance')
assert abs(float(A[1,110:366].sum()-raw[1,110:366].sum()))<1e-14
signed=np.zeros((1,4),np.float64);signed[0,1]=-0.
detail,summary=d.repeat_detail016(signed)
assert summary['actual_absmax']==0 and summary['ordered_float64_ULP_max']==1
parent,draw=env['parent'],env['draw'];source=env['source']
sh,fh=d.source_input_evidence016(source,raw[4:],raw[4:],parent[4:],draw[4:],draws=4,start=4)
assert sh.shape==fh.shape==(8,32)
env['reject'](lambda:d.source_input_evidence016(source,raw[4:],raw[4:],parent[4:],draw[4:],draws=4,start=0),'parent/draw identity')
mu=env['mu'];v=env['v'];samples=env['samples'];mom=env['mom']
assert np.array_equal(mom['theory_expected_sample_variance_Vns2'],np.repeat(v.mean(1)+mu.var(1,ddof=1),16))
assert np.array_equal(mom['theory_sample_mean_MCSE_Vns'],np.repeat(np.sqrt(v.sum(1))/4,16))
assert abs(mom['draw_centered_residual_sample_variance_Vns2'].mean()-v.mean())<1e-14
focused=dict(schema='n01/wp3-016-fix02-independent-array-checks/v1',passed=True,checks=env['checks'],negative_cases=14,
    additional_checks=['charge-cancelling A point error still rejected after saving','signed-zero ULP distinction','nonzero chunk start accepted; wrong start rejected',
                       'API theory variance and MCSE arrays independently compared with formulas'],
    source_sha256=expected[c.DRIVER],neural_modules_loaded=False,scientific_or_synthetic_NN_AD_RNG_training=0,private_arrays_read=0,SSH_calls=0)
write_new(OUT/'ENGINEERING_RECEIPT.json',focused)
check('independently_replayed_array_only_four_groups_12_negatives_plus_two_targeted_negatives',receipt=ref(OUT/'ENGINEERING_RECEIPT.json'))
independent_health=HERE/'independent_artificial_stdlib_01/ENGINEERING_RECEIPT.json'
health_replay=read(independent_health)
assert health_replay['passed'] and health_replay['source_sha256']==expected[c.CONTROLLER]
assert len(health_replay['checks'])==5 and len(health_replay['checks'][2]['negative_cases'])==15
check('independently_replayed_current_controller_five_stdlib_groups_and15_health_negatives',receipt=ref(independent_health))

# Reopen only explicitly artificial saved banks; never open actual failure NPZ/CP.
batch=HERE/'artificial_cpu_01/ARTIFICIAL_BATCH';bank_paths=sorted(batch.glob('seed*/*_evaluation/*_CHARGE_PREADC_MOMENTS.npz'))
assert len(bank_paths)==8
inventory={s:len(list(batch.glob('seed*/*_evaluation/*'+s))) for s in ['_HARD_ROWS.npz','_REPORT.npz','_BASELINE100.npz','_CHARGE_PREADC_MOMENTS.npz','_A_NUMERIC_REPEATS.json']}
assert inventory==numeric['checks'][-1]['files']
for file in bank_paths:
    bank=int(file.name[4]);semantic=read(file.parent/'CHARGE_SEMANTICS.json')['banks'][bank]
    diagnostic=read(file.with_name(f'bank{bank}_A_NUMERIC_REPEATS.json'))
    assert diagnostic['saved_arrays_sha256']==sha(file) and diagnostic['within_numerical_tolerance']
    assert diagnostic['Q_mu_CPU_GPU_within_derived_bound'] and diagnostic['diagnostic_and_arrays_saved_before_numerical_gate']
    with np.load(file,allow_pickle=False) as saved:
        A=saved['A_signal_pre384_V'];mu=saved['Q_mu_draws_Vns'];q=saved['q_sig_draws_Vns']
        assert A.shape==(32,384) and A.dtype==np.float64 and mu.shape==q.shape==(8,4)
        assert np.array_equal(saved['A_source_parent_row'],np.repeat(np.arange(8),4))
        assert np.array_equal(saved['draw_index'],np.tile(np.arange(4),8))
        assert np.array_equal(saved['source_parent_row'],np.arange(8))
        assert np.array_equal(saved['Q_mu_Vns'],mu.mean(1))
        a,b=semantic['a_Vns'],semantic['b'];v=a*mu+b*mu**2
        assert np.array_equal(saved['theory_conditional_variance_draws_Vns2'].view(np.uint64),v.view(np.uint64))
        assert np.array_equal(saved['theory_conditional_variance_Vns2'],v.mean(1))
        assert np.array_equal(saved['theory_expected_sample_variance_Vns2'],v.mean(1)+mu.var(1,ddof=1))
        assert np.array_equal(saved['theory_sample_mean_MCSE_Vns'],np.sqrt(v.sum(1))/4)
        assert np.array_equal(saved['draw_centered_residuals_Vns'],q-mu)
        assert np.array_equal(saved['draw_centered_residual_second_moment_Vns2'],np.square(q-mu).mean(1))
        assert np.all(np.abs(mu-A[:,110:366].sum(1).reshape(8,4))<=saved['Q_mu_CPU_GPU_reduction_bound_Vns'])
        assert np.all(A[:,:100]==0)
        for name in ['source_input_sha256_uint8','forward_float64_input_sha256_uint8']:
            hashes=saved[name];assert hashes.shape==(32,32) and hashes.dtype==np.uint8
            assert np.all(hashes.reshape(8,4,32)==hashes.reshape(8,4,32)[:,:1])
events=[json.loads(line) for line in (batch/'operations.jsonl').read_text('utf8').splitlines()]
entered={};completed={};counts={k:0 for k in caps}
for event in events:
    key=event['operation_id']
    if event['event']=='entered':assert key not in entered;entered[key]=event
    elif event['event']=='completed':
        assert key in entered and key not in completed;completed[key]=event
        for k,v in entered[key]['charges'].items():counts[k]+=v
    else:raise AssertionError('unexpected artificial operation event')
assert set(entered)==set(completed) and counts==numeric['checks'][-1]['counters']
assert counts['G_updates']==counts['new_AD_queries']==2 and counts['checkpoint_loads']==0
check('all_eight_artificial_banks_reopened_actual_rows_moments_lineage_and_two_trial_journal_verified',
      banks=8,files=inventory,synthetic_updates=2,synthetic_checkpoint_loads=0,operations=len(entered),counters=counts)

forecast=ready['resource_estimate'];assert forecast['total8banks_A_payload_bytes']==6593*4*384*8*8==648118272
assert forecast['new_actual_A_float64_bytes_per_bank']==81014784
assert forecast['projected_batch_output_bytes']==forecast['existing48hard48report8baseline_empirical_forecast_bytes']+forecast['projected8bank_charge_payload_and_container_no_compression_bytes']+forecast['other_output_reserve_bytes']==2884918689
assert forecast['batch_cap_bytes']==p['resource_limits']['output_MiB']*1024**2==3221225472
assert forecast['trial_cap_bytes']==p['run_resource_limits']['output_MiB']*1024**2==1610612736
assert forecast['projected_batch_output_bytes']<forecast['batch_cap_bytes'] and forecast['projected_trial_average_bytes']<forecast['trial_cap_bytes']
check('output_forecast_arithmetic_and_unchanged_live_caps',new_A_payload_bytes=648118272,
      projected_batch_bytes=2884918689,batch_cap_bytes=3221225472,trial_cap_bytes=1610612736,
      native_measurement=False,worst_case_guarantee=False,per_trial_average_is_not_each_trial_bound=True,
      additional_CPU_512MiB_is_an_estimate=True,extra_NN_ADC_AD_RNG_calls=0)

failure=read(STAGE/'failure_collection_01/FAILURE_PRESERVATION_RECEIPT.json')
actual=failure['actual_completed_counters'];aggregate={k:caps[k]+actual.get(k,0) for k in caps}
assert actual['G_updates']==actual['new_AD_queries']==0
assert aggregate['G_updates']==aggregate['new_AD_queries']==4000
assert 'torch' not in sys.modules
result=dict(schema='n01/wp3-016-independent-fix-review/v2',UTC=datetime.now(timezone.utc).isoformat(),
    status='PASS_FIX02_SOURCE_REVIEW_PENDING_FROZEN_BINDING',passed=True,blockers=[],sources=sources,budget_caps=caps,
    engineering_ready=ref(HERE/'ENGINEERING_FIX_READY.json'),engineering_numeric_receipt=ref(ROOT/ready['numerical_receipt']['path']),
    engineering_health_receipt=ref(ROOT/ready['stdlib_receipt']['path']),source_diff=ref(HERE/'SOURCE_DIFF.patch'),
    independent_health_receipt=ref(independent_health),
    prior_failure_diagnosis=ref(HERE/'FAILURE_DIAGNOSIS_REVIEW.json'),prior_source_review=ref(STAGE/'independent_01/SOURCE_REVIEW.json'),
    checks=checks,scientific_settings_and_budget_unchanged=True,model016_exact_bytes_unchanged=True,
    numerical_contract=dict(A_max_abs_point_tolerance_V=1e-7,origin='unchanged scripts/run_wp1_013.py A_numerical_residual',
        actual_A_mu_q_v_saved_before_threshold_gate=True,parent_draw_and_original_forward_input_bytes_exact=True,
        CPU_GPU_canonical_bound='2*gamma255/(1-gamma255)*L1(A[110:366])+512*smallest_subnormal, rounded upward',
        assumptions='finite same actual float64 inputs, round-to-nearest, at most255 additions per256-term sum, no overflow; subnormal allowance',
        no_new_arbitrary_Q_allclose=True,actual_model_mu_per_draw_unchanged=True,original_failure_amplitude='unknown_not_saved'),
    statistics=dict(theory_v_per_draw='a*mu_d+b*mu_d^2',expected_sample_variance='mean(v_d)+samplevar(mu_d,ddof1)',
        theory_mean_MCSE='sqrt(sum(v_d))/4',draw_centered_second_moment='mean((q_d-mu_d)^2), ddof0; expectation mean(v_d)',
        draw_centered_sample_variance='samplevar(q_d-mu_d,ddof1); expectation mean(v_d)',
        legacy_empirical_MCSE='sqrt(samplevar(q,ddof1)/4), explicitly distinct from theory',
        mean_mu_reporting_only=True,numerical_mu_spread_is_physical_variance=False),
    resource_estimate=forecast,resource_conclusion='Estimated below unchanged caps; not native measurement or a worst-case guarantee. Per-trial average does not prove each trial fits. Existing trial and supervisor guards remain decisive.',
    replacement_accounting=dict(original_failed_actual_counts_preserved=actual,full_replacement_caps=caps,
        failed_plus_planned_replacement_aggregate_caps=aggregate,scientific_seed_trials=2,new_candidate=False,
        cumulative_scientific_G_before_replacement=175667,root_registration_required_before_refreeze=True),
    review_costs=dict(scientific_CP_NN_AD_random_training=0,synthetic_NN_AD_random_training=0,
        actual_private_NPZ_CP_decodes=0,SSH=0,source_or_config_mutations=0),
    next_boundary='Root registers replacement accounting then refreezes/reviews binding; this source review performs or authorizes no native restart.',
    scientific_acceptance=False,weights_promoted=False,elapsed_seconds=time.monotonic()-started)
write_new(HERE/'INDEPENDENT_FIX_REVIEW.json',result)
reopened=read(HERE/'INDEPENDENT_FIX_REVIEW.json');assert reopened['passed'] and reopened['sources']==sources
print(json.dumps(dict(passed=True,checks=len(checks),receipt=ref(HERE/'INDEPENDENT_FIX_REVIEW.json'),focused_receipt=ref(OUT/'ENGINEERING_RECEIPT.json'),sources=sources),indent=2))
