"""Final local hashes/protection and successful receipt aggregation, stdlib only."""
from pathlib import Path
import ast
import datetime as dt
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return dict(path=str(path),relative_path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path))
def write(path,value):
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
sys.path.insert(0,str(ROOT/'scripts'))
import run_wp4_017 as d
import wp4_017_controller as c
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
names=list(d.OWNED_FILES);sources={name:ref(ROOT/name) for name in names}
for name in names:ast.parse((ROOT/name).read_bytes(),filename=name)
assert d.OWNED_FILES==('scripts/run_wp4_017.py','scripts/wp4_017_controller.py','src/n01/model017.py','src/n01/stats017.py','src/n01/saved_array_kernels017.py')
protected=read(HERE/'PROTECTED_BEFORE.json')['files'];assert len(protected)==124
for name,row in protected.items():assert sha(ROOT/name)==row['sha256'] and (ROOT/name).stat().st_size==row['bytes'],name
assert sha(ROOT/'src/n01/saved_array_kernels017.py')=='fe1366f35f36e97974f1a4037a36ad858c03dd8453957c1b45b65080f0fee3ac'
proposal=ROOT/'receipts/wp4_017_20261015/planning_01/ENGINEERING_PROPOSAL.json'
contract=proposal.with_name('INDEPENDENT_CONTRACT_REVIEW.json')
assert sha(proposal)==d.PROPOSAL_SHA and sha(contract)=='31a0a106bf033ec7d363abc13811a7f0e6dd87e79d139ea7ad6c867a67927fd7'
receipts={
    'core16_reused':HERE/'artificial_model_03/ENGINEERING_RECEIPT.json',
    'independent_core25numpy10torch_reused':ROOT/'receipts/wp4_017_20261015/independent_01/PREPARED_SEMANTIC_CHECKS.json',
    'final_stdlib_factories':HERE/'artificial_stdlib_final_01/ENGINEERING_RECEIPT.json',
    'two_tiny_trial_end_to_end':HERE/'artificial_driver_02/ENGINEERING_RECEIPT.json',
    'full1024_three_hypotheses_one_backward':HERE/'artificial_full1024_02/ENGINEERING_RECEIPT.json',
    'serialization_and_new_formula_delta':HERE/'artificial_storage_delta_03/ENGINEERING_RECEIPT.json',
    'actual_original_class_artificial_cache':HERE/'artificial_actual_cache_01/ENGINEERING_RECEIPT.json',
    'saved_format_reopen_and_own_CP_state':HERE/'artificial_reopen_01/ENGINEERING_RECEIPT.json',
    'root_controller11positive42negative':HERE/'artificial_controller_root_01/run_01/ENGINEERING_RECEIPT.json'}
for name,path in receipts.items():
    r=read(path)
    assert r.get('passed',r.get('preparation_complete')) is True,name
driver=read(receipts['two_tiny_trial_end_to_end']);full=read(receipts['full1024_three_hypotheses_one_backward'])
assert driver['sources']['scripts/run_wp4_017.py']['sha256']==sources['scripts/run_wp4_017.py']['sha256']
assert full['sources']['scripts/run_wp4_017.py']['sha256']==sources['scripts/run_wp4_017.py']['sha256']
assert full['sources']['src/n01/model017.py']['sha256']==sources['src/n01/model017.py']['sha256']
assert read(receipts['root_controller11positive42negative'])['source_sha256']==sources['scripts/wp4_017_controller.py']['sha256']
stamp=dt.datetime.now(dt.timezone.utc).isoformat()
write(HERE/'PROTECTED_AFTER.json',dict(passed=True,UTC=stamp,files=protected,count=124,all_exact_original_bytes_unchanged=True,
    baseline=ref(HERE/'PROTECTED_BEFORE.json'),scientific_operations=0,numeric_imports=False))
old_failed=HERE/'artificial_storage_delta_01'
write(HERE/'TEST_ATTEMPTS.json',dict(successful_receipts=[ref(p) for p in receipts.values()],
    retained_prior_attempts=['ARTIFICIAL_IMPORT_STALL_01.json','ARTIFICIAL_IMPORT_STALL_02.json',
        'artificial_driver_01','artificial_full1024_01','artificial_storage_delta_01','artificial_storage_delta_02','artificial_stdlib_01'],
    failed_storage_oracle_attempt=dict(path=str(old_failed),passed=False,reason='test asserted bitwise equality between cdist and explicit broadcast norm reductions',
        later_same_literal_max_absolute_difference=read(receipts['serialization_and_new_formula_delta'])['paired_oracle_max_abs_error'],
        correction='engineering oracle comparison128float64eps; exact5 distances/formula unchanged; no scientific gate/source law changed'),
    failed_or_no_receipt_attempts_counted_as_PASS=False,originals_preserved=True))
p=d.settings017();p.update(budget_caps=d.budget_caps017(p),stage1_budget_caps=d.budget_caps017(p,True))
d.validate_settings017(p,frozen=False)
write(HERE/'SETTINGS_DRAFT.json',dict(p,inputs_pending_root_binding=True,local_only_delivery=True,native_execution_authorized=False))
runtime_api=dict(driver_factories=['settings017()','budget_caps017(p, stage1=False)','validate_settings017(p, frozen=True)'],
    source_inventory='driver.OWNED_FILES / controller.OWNED (5 explicit runtime files; all lazy imports covered by original116 plus these5)',
    model=['CachedFrozenQ1Charge017','deterministic_loading017','assemble_cached017','teacher_shape_coefficients017','energy_score_U017'],
    driver=['build_frozen_cache017','Streams017','ParentSampler017','Runtime017','Engine017.step','evaluate_kernel017',
        'energy_vectors017','compare_kernel017','run017'],
    pure_stats=['partition017','groups_weights017','weighted017','support_cell017','scalar_banks017','coverage017',
        'covariance_cell017','timing_cell017','energy_cell017','variance_curve_cell017','waveform_cells017','stage_gate017'],
    controller='--mode freeze/deploy/start/verify-start/status/collect/collect-failed/collect-inspect; NO native action is authorized or invoked')
ready=dict(schema='n01/wp4-017-engineering-ready/v1',UTC=stamp,passed=True,engineering_ready=True,
    status='LOCAL_ONLY_SOURCE_AND_ARTIFICIAL_ENGINEERING_COMPLETE_AWAITING_SAME_SOURCE_REVIEW',
    user_latest_boundary='做好就行 不用推送服务器 服务器马上要关闭了',
    source_review_pending=True,ready_for_local_source_freeze_after_existing_SOURCE_REVIEW=True,ready_for_native_dispatch=False,
    sources=sources,runtime_source_file_count=5,extra_runtime_helper_files=0,runtime_API=runtime_api,
    exact_required_input_keys=['dataset','statistics','data_statistics','base_config','deterministic','noise005','M0_head_seed0'],
    accepted_proposal=ref(proposal),accepted_contract_review=ref(contract),
    configuration_document=ref(ROOT/'docs/WP4_017_CONFIGURATION.md'),data_roles=ref(ROOT/'WP4_017_DATA_ROLES.json'),
    configuration_draft=ref(HERE/'SETTINGS_DRAFT.json'),original124_protection=ref(HERE/'PROTECTED_AFTER.json'),
    engineering_receipts={k:ref(v) for k,v in receipts.items()},attempt_history=ref(HERE/'TEST_ATTEMPTS.json'),
    exact_full_additive_budget=p['budget_caps'],exact_stage1_additive_budget=p['stage1_budget_caps'],
    invariant_metadata=dict(trainable_parameters_per_optimizer=2,failure_checkpoint_whole_job_maximum=1,
        stage2_failure_CP_still_available_after_successful_stage1=1),
    native_call_counts=dict(SSH=0,SCP=0,deploy=0,start=0,status=0,server_queries=0),
    actual_scientific_counts=dict(original_CP_loads=0,original_SOURCE_REAL_DEV_numeric_reads=0,NN_forward_rows=0,
        AD=0,G=0,new_random_scalar_requests=0,GPU=0),
    manufactured_engineering_counts_are_separate=True,
    artificial_latest_two_trial_counts=driver['manufactured_engineering_counts'],
    artificial_full1024=dict(elapsed_seconds=full['elapsed_seconds'],memory=full['memory'],
        unique_saved_storage_bytes=full['unique_saved_storage_bytes'],backward_queries=1,optimizer_steps=1,
        full1024x1024x257_broadcast=False,actual_gradients=full['metrics']['preclip_gradients']),
    original_class_cache_artificial_forward_rows=4,original_class_cache_artificial_CP_loads=0,
    proper_score=dict(paired_actual_distances_per_parent_h=5,unpaired_actual_matrices='cross1024x1024+self1024x1024; self diagonal masked',
        training_signal_and_ADC_chunk_rows=256,distance_block_rows=64,one_backward=True,no_detach_or_objective_change=True),
    stage_gate=dict(both_seed_full_primary_matrix=1392,missing_censor_cells_additional=True,
        fail_or_pending_closes_stage2=True,normal_stage1_native0_two_slots_supported=True,
        pass_requires_full8_slots_for_normal_complete=True,no_unentered_stage2_counts_in_two_slot_completion=True),
    saved_format=dict(codec='original013 code256+actual baseline mean+optional IEEE XOR',actual_baseline100=True,
        native32_and_derived64_moments=True,actual_q_mu_v_and_parent_draw_ids=True,full_stochastic_preADC_carrier_saved=False,
        original_input_and_actual_float64_forward_input_SHA_separate=True,original_physical_weights_preserved_separately=True),
    resource_scope=dict(CPU_engineering_peak_RSS_verified=True,GPU_peak_or_remote_runtime_measured=False,
        total_output_cap_MiB=17408,selected_cap_MiB=256,compression_or_output_estimates_are_not_guarantees=True,
        output_cap_failure_preserves_partial_no_silent_metric_deletion=True),
    original116_sources_118_training_controls_and6CPU_controls_unchanged=True,
    metadata_or_original_sources_modified_by_this_agent=False,Git_commit_or_D_backup_by_this_agent=False,
    unresolved_engineering_blockers=[],scientific_acceptance=False,hardware_calibration=False,
    scientific_status='response_match_not_established')
write(HERE/'ENGINEERING_READY.json',ready)
summary='017本地施工完成；正式科学运行0。\n\n5运行源码、configuration/docs/roles、精确full/stage1预算factory已齐。\n两tiny trial15项、full1024一次三hyp反向传播、实际原class人工cache10项、存储delta13项、重开12项、root控制器11正例/42负例均通过；核心已通过组复用。\nfull1024本地CPU peakRSS '+str(full['memory']['peak_RSS_bytes'])+'B；GPU/真实数据/服务器未执行。\n所有124旧控制精确保护。\n用户最新LOCAL ONLY：不推送、不查询或启动服务器。待原reviewer同一轮SOURCE_REVIEW后由root本地绑定/冻结/小备份。\n'
with (HERE/'ENGINEERING_READY.md').open('x',encoding='utf8') as f:f.write(summary)
print(json.dumps(dict(passed=True,ready=str(HERE/'ENGINEERING_READY.json'),sha256=sha(HERE/'ENGINEERING_READY.json'),sources=sources),ensure_ascii=False))
