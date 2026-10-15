"""Update canonical local READY after preserving exact prior bytes."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;ENG=HERE.parent
sys.path.insert(0,str(ROOT/'scripts'))
import run_wp4_017 as d
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),relative_path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
def new(p,v):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
before=HERE/'before/receipts/wp4_017_20261015/engineering_01/ENGINEERING_READY.json'
assert sha(before)==sha(ENG/'ENGINEERING_READY.json')=='11efa04b9a071a0b3642918540cdb41f4989d1785d8f3b35562405bfe1a90d03'
original=read(before);r=read(before)
delta=HERE/'artificial_delta_01/ENGINEERING_RECEIPT.json';final=HERE/'artificial_final_stdlib_01/ENGINEERING_RECEIPT.json'
assert read(delta)['passed'] and len(read(delta)['checks'])==27
assert read(final)['passed'] and len(read(final)['checks'])==23
sources={n:ref(ROOT/n) for n in d.OWNED_FILES}
for name,row in read(final)['sources'].items():assert sources[name]['sha256']==row['sha256'] and sources[name]['bytes']==row['bytes']
# The27-case source is preserved verbatim; final additions are metadata/AST
# checked without rerunning numeric groups or loading any original arrays.
for name in ('scripts/run_wp4_017.py','src/n01/stats017.py'):
    assert sha(HERE/'before_preflight_metadata'/name)==read(delta)['sources'][name]['sha256']
protected=read(ENG/'PROTECTED_BEFORE.json')['files'];assert len(protected)==124
assert all(sha(ROOT/n)==row['sha256'] and (ROOT/n).stat().st_size==row['bytes'] for n,row in protected.items())
stamp=dt.datetime.now(dt.timezone.utc).isoformat()
new(HERE/'PROTECTED_AFTER_FIX01.json',dict(passed=True,UTC=stamp,files=protected,count=124,all_exact_original_bytes_unchanged=True))
p=d.settings017();p.update(budget_caps=d.budget_caps017(p),stage1_budget_caps=d.budget_caps017(p,True))
d.validate_settings017(p,frozen=False)
draft=HERE/'SETTINGS_DRAFT_REVISED.json';new(draft,dict(p,inputs_pending_root_binding=True,local_only_delivery=True,native_execution_authorized=False))
assert sha(ENG/'SETTINGS_DRAFT.json')==sha(HERE/'before/receipts/wp4_017_20261015/engineering_01/SETTINGS_DRAFT.json')
temp=ENG/'SETTINGS_DRAFT.fix01.tmp';temp.write_bytes(draft.read_bytes());os.replace(temp,ENG/'SETTINGS_DRAFT.json')
mandatory=['undefined_'+name for name in ('Q','tail','PSD_raw','baseline_mean','baseline_RMS','peak_voltage','rise10_90','FWHM','peak_index')]+[
    'flag__PSD_undefined','flag__rise_undefined','flag__FWHM_undefined','flag__temporal_nonpositive_peak','flag__rise_prepeak_censored',
    'flag__rise_other_undefined','flag__FWHM_prepeak_censored','flag__FWHM_postpeak_censored','flag__FWHM_other_undefined']
fix=dict(schema='n01/wp4-017-source-review-local-fix/v1',passed=True,UTC=stamp,
    prior_ready=ref(before),snapshot=ref(HERE/'before/SNAPSHOT.json'),sources=sources,
    focused27=ref(delta),final_stdlib23=ref(final),original124_protection=ref(HERE/'PROTECTED_AFTER_FIX01.json'),
    changes=['fit/test fixed [Q/qref,(Q/qref)^2] Gram and BOTH-role secondary identification',
        'complete initial/end training and parent RNG state metadata; namespace/seed/role/seen0; checkpoint binds initial receipt',
        'analytic variance ALL original group parents; invalid model/reference cannot use95percent deletion',
        'only18 required undefined/censor causes gate1percent; other retained64-slot flag fractions diagnostic',
        'literal preflight free40GiB/RAM10GiB/GPUempty in factories and both validators',
        'existing B0 finite/nonzero/max/RMS reporting and T2 untestable/inconclusive/no mixture recovery implication'],
    inherited_heavy_tests_reused_by_exact_AST_proof=read(final)['unchanged_full_math_AST'],
    cache_A_B_Q_decoded_algebra_and_eval_only_reporting_metadata_changed=True,
    original_scientific_law_seed_gates_budget_proposal_unchanged=True,extra_gate_distance_bank_RNG_or_science_trial=0,
    original_CP_SOURCE_REAL_DEV_NN_AD_G_RNG_GPU=0,SSH_SCP_or_server_queries=0,
    new_full1024_or_tiny_trial_or_core35_suite_reruns=0,
    independent_SOURCE_REVIEW_still_same_round=True,independent_SOURCE_REVIEW_pending=True)
new(HERE/'SOURCE_REVIEW_FIX_RECEIPT.json',fix)
r.update(UTC=stamp,sources=sources,prior_ready=ref(before),source_review_fix=ref(HERE/'SOURCE_REVIEW_FIX_RECEIPT.json'),
    status='LOCAL_ONLY_ENGINEERING_ALIGNED_AFTER_SAME_SOURCE_REVIEW_DELTA_AWAITING_FINAL_REVIEW',
    configuration_document=ref(ROOT/'docs/WP4_017_CONFIGURATION.md'),data_roles=ref(ROOT/'WP4_017_DATA_ROLES.json'),
    configuration_draft=ref(ENG/'SETTINGS_DRAFT.json'),original124_protection=ref(HERE/'PROTECTED_AFTER_FIX01.json'))
r['engineering_receipts'].update(focused_source_review_delta27=ref(delta),final_source_metadata_stdlib23=ref(final))
r['stage_gate'].update(required_missingness_features=mandatory,required_missingness_cells_two_seed=1296,
    total_required_cells_two_seed=2688,other_retained_flags_diagnostic_only=True)
r['resource_scope'].update(preflight=p['preflight'])
r['secondary_identifiability']=dict(columns=['Q/qref','(Q/qref)^2'],Gram='X.T@X/N',roles=['fit','test'],
    per_column_L2_normalization=False,both_roles_must_have_eigenvalue_ratio_at_least=1e-4)
r['training_random_state_storage']=dict(initial_and_end_full_arrays=True,namespaces_seed_roles_and_initial_seen_zero=True,
    checkpoint_states_and_initial_receipt_bound=True,additional_scientific_RNG_forwards_CP=0)
r['T2_reporting_boundary']=dict(component=0,actual_formal_B0_summary='not computed; no original formal cache has been built',
    cache_will_record_finite_nonzero_max_RMS=True,zero_or_invalid_B0='untestable',nonzero_B0_hard_observed_distinguishability='not_established',
    separate_T1_T2_observed_contrast_computed=False,theoretical_eta_fourth_moment=1.6878,
    charge_recovery_pass_does_not_establish_mixture_recovery=True)
r['heavy_test_reuse_after_delta']=dict(AST_receipt=ref(final),no_heavy_group_rerun=True,
    core_model_energy_controller_kernel_exact_bytes=True,training_step_signal_STE_and_score_math_AST_unchanged=True,
    cache_and_eval_only_T2_boundary_reporting_added=True)
target=HERE/'ENGINEERING_READY_REVISED.json';new(target,r)
tmp=ENG/'ENGINEERING_READY.fix01.tmp';tmp.write_bytes(target.read_bytes());os.replace(tmp,ENG/'ENGINEERING_READY.json')
with (ENG/'ENGINEERING_READY.md').open('a',encoding='utf8') as f:
    f.write('\n同次源码审查局部对齐已完成：27项focused检查+23项最终stdlib字段/AST/哈希检查通过。\nfit/test固定Gram、initial/end/CP随机状态、解析方差不删父、18项缺失原因门与其他flag诊断分离，preflight40/10/GPUempty及T2机制可分辨性边界已落。\n未重跑full1024/两tinytrial/核心35组；精确AST/原字节证明复用。SOURCE_REVIEW仍为原同一轮，待最终回执；继续LOCAL ONLY。\n')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps(dict(passed=True,READY=ref(ENG/'ENGINEERING_READY.json'),fix=ref(HERE/'SOURCE_REVIEW_FIX_RECEIPT.json'),sources=sources),ensure_ascii=False))
