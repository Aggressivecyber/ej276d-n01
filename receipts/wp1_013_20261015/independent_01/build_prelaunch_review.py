"""Independent WP1 contract/source closeout; stdlib only, no scientific inputs."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT/'receipts/wp1_013_20261015/independent_01'
EXPECTED = {
    'USER_MASTER_PLAN_20261015.txt': '56190995374eb2a765a3c8589736514154dba7ff33aa1309b8b1de493e846ebb',
    'OBSERVATION_CONTRACT.json': 'c894abe80d7d9a31e4ffe670484123dd1eea5ee75dad7742e52d304e23054c6c',
    'DATA_ROLES.json': 'bfc27c49a54293c86ff116965f3ce410468611b26266e0fe41112c81ae4bc0aa',
    'docs/ACCEPTANCE_CONTRACT.md': '87f2d605dca67109a72356d44aa411a1aa1ac4d11a26c7f97aab30ae4460bc6c',
    'wp1_013_config.json': '5f29f43cbac524441d44ddad7aaf978176e614f4184ff8bfdfb5dbcbb903e9d5',
    'scripts/wp1_input_information.py': 'bb8278d2651ddcaf10830a4589d66242d69a7e22d10321171ad7f4b71dec8b17',
    'scripts/wp1_boundary_saved.py': 'ea1317c2a6e8616f5b30e2aa6ea1900fde17a711b57e2f02ae84fa5cc5782065',
    'scripts/wp1_parameter_probe.py': 'bf7901db6e1b99084894fdc8f4c5d1685de2ca00cacb7a0a6a9801c39b8823fa',
    'scripts/wp1_evaluation.py': '3f5e4f9e8ca147cfbad8b75085c59d9c2081deda9b7532f1732f1d5e932cc895',
    'scripts/run_wp1_013.py': 'faa2eb81c60d4ab6ff6507513782450eb1ba08059d8643aaed504e63fa0a1680',
    'scripts/wp1_ops.py': '02582f9fe0f41a54a32c167270011f521b5f9fb95f6a5191acb620e0c1786d51',
    'scripts/wp1_controller.py': '7458deaff32b4c725334844df09219a859581efc97c3e808c34e90093febc287',
}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text('utf8'))

def pin(relative):
    p = ROOT/relative
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)

def require(flag, name, detail=None):
    if not flag:
        raise AssertionError(name)
    checks.append(dict(name=name, passed=True, kind='static_or_saved_engineering_receipt', detail=detail))

checks = []
pins = {name: pin(name) for name in EXPECTED}
require(all(pins[k]['sha256'] == v for k, v in EXPECTED.items()), 'authority_contract_and_seven_final_source_bytes')
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
require(head == 'c9ad769697b982b607ccec320da9b727ad37b978', 'registered_base_HEAD')
p = read(ROOT/'wp1_013_config.json')
roles = read(ROOT/'DATA_ROLES.json')
obs = read(ROOT/'OBSERVATION_CONTRACT.json')
require(p['status'] == 'draft_registered' and p['source_files'] == {} and p['scientific_training_updates'] == 0,
        'current_draft_is_not_executable_and_controller_must_freeze')
require(p['authority']['sha256'] == EXPECTED['USER_MASTER_PLAN_20261015.txt'], 'literal_user_authority_pin')
require(len(p['inputs']) == 22 and all({'path','local_path','sha256','bytes'} <= set(r) for r in p['inputs'].values()),
        '22_explicit_immutable_input_records_metadata_only')
require(all(r['sha256'] == EXPECTED[r['path'].split('wp1_013_20261015/',1)[1]] for r in p['contracts'].values()),
        'three_formal_contract_input_SHA_aliases')
require('information_analysis' in p and 'boundary_analysis' in p and 'input_information' not in p and 'boundary_saved' not in p,
        'canonical_sections_without_ambiguous_legacy_aliases')
require(sum(len(r.get('allowed_members', [])) for r in p['information_analysis']['inputs'].values()) == 21
        and len(p['boundary_analysis']['inputs']['row_mechanisms']['allowed_members']) == 1387,
        'SOURCE21_and_saved_boundary1387_named_member_whitelists')
require(p['boundary_analysis']['paths'][-1] == 'H_moving_same_hard_ADC_first_argmax', 'actual012_moving_path_schema')
require(roles['SOURCE']['TRAIN_parents'] == 6593 and roles['SOURCE']['original_internal_val'] == 1599
        and roles['SOURCE']['historically_fixed_DEV'] == 512 and roles['REAL']['TRAIN_rows'] == 21047
        and roles['REAL']['TRAIN_original_logical_groups'] == 422 and roles['REAL']['012_reference_bank_groups'] == 419,
        'full_TRAIN422_vs_old_reference419_and_information_audit_roles')
require(roles['confirmed_bound_real_acquisitions'] == 1 and not roles['external_confirmation']['unknown_is_external']
        and not roles['external_confirmation']['software_delivery_blocked_by_absence']
        and roles['historical_other_raw_role_pools']['WP1_waveform_access'] == 0,
        'single_acquisition_unknown_external_and_other_historical_wave_pool_scope')
require(obs['ADC']['intervals'] == 16383 and obs['ADC']['bits'] == 14
        and obs['peak']['search_indices_half_open'] == [112,144]
        and obs['representation']['output_samples'] == 256 and obs['all_rows_retained'], 'unchanged_hard008_observation_geometry')
require(p['reference']['theta0']['G_updates'] == p['calibration']['reference_checkpoint']['G_updates'] == 2000
        and p['reference']['theta0']['sha256'] == p['inputs']['head_final']['sha256'], 'theta0_is_saved_finalG2000')
require(p['reference']['banks'] == 2 and p['reference']['draws_per_bank'] == 4
        and p['parameter_validation']['parents'] == 1024 and p['parameter_validation']['parents_per_batch'] == 512
        and p['parameter_validation']['batches'] == 2 and p['parameter_validation']['parameter_states'] == 30,
        'full_population_and_fixed_parameter_probe_scope')
require(p['parameter_validation']['nominal_carrier_RMS_ADC_LSB'] == [.1,1,10]
        and p['parameter_validation']['signs'] == [-1,1]
        and p['parameter_validation']['primary_scale_LSB'] == 1
        and p['method_selection']['numeric_sign_epsilon'] == 1e-12,
        'all_signed_three_scales_retained_fixed_primary1LSB')
zero_keys = ['optimizer_created','optimizer_steps','G_updates','D_updates','noise_updates','population_updates','C2ST_updates','new_AD_queries','DEV_model_rows']
require(all(p['budget_caps'][k] == 0 for k in zero_keys), 'zero_AD_optimizer_training_DEV_model_caps')
reference = 6593*4*2
parameter = 1024*4*2
computed_caps = dict(old_A_rows=reference+parameter*30, head_actual_parameter_forward_rows=reference+parameter*30,
    noise005_rows=reference, head_linear_direction_map_rows=parameter*5,
    head_total_kernel_rows=reference+parameter*35, hard_ADC_rows=reference*2*3+parameter*(30+21)*3,
    all_path_gather_rows=reference*2*3+parameter*(30*3+21)*3, checkpoint_loads=3)
require(all(p['budget_caps'][k] == v for k,v in computed_caps.items()), 'actual_and_analytic_work_budget_arithmetic', computed_caps)
formula = {str(d):dict(old_A_rows=52744+8192*d, head_actual_parameter_forward_rows=52744+8192*d,
    noise005_rows=52744, hard_ADC_rows=832560+24576*d, all_path_gather_rows=832560+73728*d, checkpoint_loads=3)
    for d in range(0,31,6)}
require(formula['30']['hard_ADC_rows'] == p['budget_caps']['hard_ADC_rows']
        and formula['30']['all_path_gather_rows'] == p['budget_caps']['all_path_gather_rows'],
        'controller_COMPLETE_formula_all_legal_defined_state_counts', formula)
rr = p['RR']
require(rr['rows_per_side'] == 10475 and rr['groups_per_side'] == 210
        and sum(int(k)*v for k,v in rr['histogram_each_side'].items()) == 10475
        and 2*10475+rr['auxiliary_unpaired_rows'] == 21047 and rr['main_population_exclusions'] == 0,
        'RR_group_isolated_equalN_aux97_do_not_exclude_population')
require(not p['application_tolerances']['physical_hardware_calibrated'] and not p['calibration']['RR_used_to_set_tolerance']
        and p['calibration']['freeze_before_candidate'], 'normative_tolerance_algorithm_before_new_candidate_not_RR_or_hardware_fit')

trees = {}
for name in EXPECTED:
    if name.endswith('.py'):
        trees[name] = ast.parse((ROOT/name).read_text('utf8'), filename=name)
for name in ('scripts/wp1_parameter_probe.py','scripts/wp1_evaluation.py','scripts/run_wp1_013.py'):
    calls = [n for n in ast.walk(trees[name]) if isinstance(n, ast.Call)]
    forbidden = [n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','grad','step')]
    require(not forbidden, 'no_new_AD_or_optimizer_call_AST_'+Path(name).stem)
for name in ('scripts/wp1_ops.py','scripts/wp1_controller.py'):
    # Store.npz lazily imports NumPy only when writing native/synthetic arrays.
    # Importing operations for metadata/controller checks must remain stdlib only.
    imports = [n for n in trees[name].body if isinstance(n,(ast.Import,ast.ImportFrom))]
    require(all(not (getattr(n,'module','') or '').startswith(('torch','numpy')) and
        all(not a.name.startswith(('torch','numpy')) for a in getattr(n,'names',[])) for n in imports),
        'stdlib_only_operations_module_import_AST_'+Path(name).stem)
driver = (ROOT/'scripts/run_wp1_013.py').read_text('utf8')
require('model.to(' not in driver and 'model.cuda(' not in driver and 'baseline.to(device)' in driver,
        'no_whole_wrapper_device_migration_after_frozen_layout')
require("observation_flags=observation_booleans(obs)" in driver and 'real_unknown_flags=REAL_UNKNOWN_FLAGS' in driver
        and 'REPEATS_8_' in driver and 'IEEE754_reconstruction_XOR_uint64' in driver,
        'all_row_native_flags_REAL_unknowns_combined8draws_lossless_window_contract')
require('owned_size+complete_bytes<=' in driver and "time.monotonic()-started<=p['resource_limits']['wall_seconds']" in driver,
        'final_exact_output_and_wall_cap')

old_names = ['SHAPE010_SOURCE_MANIFEST.json','STABILITY011_SOURCE_MANIFEST.json','MECHANISM012_SOURCE_MANIFEST.json']
old = {name:read(ROOT/name) for name in old_names}
require([len(old[name]['files']) for name in old_names] == [53,63,69], 'original_53_63_69_source_inventories')
protected = {root+':'+name:digest for root,doc in old.items() for name,digest in doc['files'].items()}
require(len(protected) == 185 and all(sha(ROOT/key.split(':',1)[1]) == digest for key,digest in protected.items()),
        '185_old_source_protection_records_current_local_SHA')
new = {'scripts/wp1_ops.py','scripts/run_wp1_013.py','scripts/wp1_evaluation.py','scripts/wp1_parameter_probe.py',
       'scripts/wp1_input_information.py','scripts/wp1_boundary_saved.py','scripts/wp1_controller.py',
       'USER_MASTER_PLAN_20261015.txt','OBSERVATION_CONTRACT.json','DATA_ROLES.json','docs/ACCEPTANCE_CONTRACT.md'}
recipe = set(old['MECHANISM012_SOURCE_MANIFEST.json']['files']) | new | set(old_names)
require(len(recipe) == 83 and all((ROOT/n).is_file() for n in recipe)
        and not any(n.startswith('private_inputs/') or n.endswith(('.npz','.pt','.pth','.bin')) for n in recipe),
        '83_source_recipe_separate_from_four_private_inputs')
private = {name:r for name,r in p['inputs'].items() if r['path'].startswith(p['remote_root']+'/private_inputs/')}
require(set(private) == {'g4_cache','training_container','generator_source','provenance_report'}, 'exact_four_private_payload_metadata')

evidence_names = {
    'independent_native_schema4': 'receipts/wp1_013_20261015/independent_01/native_schema_bridge_01/ENGINEERING_RECEIPT.json',
    'independent_machine_rule10': 'receipts/wp1_013_20261015/independent_01/method_rule_01/ENGINEERING_RECEIPT.json',
    'independent_controller12': 'receipts/wp1_013_20261015/independent_01/controller_bridge_01/artificial_stdlib_01/ENGINEERING_RECEIPT.json',
    'producer_information9': 'receipts/wp1_013_20261015/information_engineering_01/artificial_cpu_05/ENGINEERING_RECEIPT.json',
    'producer_Runtime7': 'receipts/wp1_013_20261015/evaluation_engineering_01/runtime_cpu_01/ENGINEERING_RECEIPT.json',
    'producer_final_semantics13': 'receipts/wp1_013_20261015/evaluation_engineering_01/cpu_final_02/ENGINEERING_RECEIPT.json',
    'producer_controller12': 'receipts/wp1_013_20261015/controller_engineering_01/artificial_stdlib_01/ENGINEERING_RECEIPT.json',
}
evidence = {name:dict(pin(path), receipt=read(ROOT/path)) for name,path in evidence_names.items()}
for name,row in evidence.items():
    r = row['receipt']
    require(r.get('status') == 'PASS' or (r.get('passed') == r.get('total') == 9), 'saved_engineering_PASS_'+name)
require(evidence['independent_controller12']['receipt']['native_exit_code'] == 0
        and evidence['independent_controller12']['receipt']['test_count'] == 12
        and evidence['independent_controller12']['receipt']['controller_sha256'] == EXPECTED['scripts/wp1_controller.py']
        and not evidence['independent_controller12']['receipt']['numpy_imported']
        and not evidence['independent_controller12']['receipt']['torch_imported'], 'current_final_controller_independent12_native0')
require(all(d == EXPECTED[n] for n,d in evidence['producer_final_semantics13']['receipt']['source_sha256'].items()),
        'final_semantics13_matches_current_four_source_bytes')

receipt = dict(
    schema='n01/wp1-013-independent-prelaunch/v1', status='PASS', passed=True,
    UTC=datetime.now(timezone.utc).isoformat(), run_id='wp1_013_20261015', reviewer_role='GPT-6.1-Sol-Max independent reviewer',
    base_HEAD=head, source_and_contract_pins=pins, static_checks=checks, unresolved_critical_findings=[],
    engineering_evidence=evidence,
    resolved_findings=[
        dict(issue='saved012 moving path name', resolution='actual H_moving_same_hard_ADC_first_argmax and current cfg'),
        dict(issue='local parent ordinal versus original TRAIN row', resolution='ordinal -> parent_rows[ordinal] and IDs, independently artificial verified'),
        dict(issue='missing/zero primary direction and required repair tolerance', resolution='all12 slots retained; both decisions false if any W/Q/noise T undefined or nonpositive; other complete direction can satisfy ANY'),
        dict(issue='duplicate/unregistered primary cell', resolution='schema rejection; not silently substituted with scientific inconclusive'),
        dict(issue='hard row flags and REAL unavailable preADC facts', resolution='row flags plus integer/float diagnostics retained; available REAL metadata separate from explicit unknowns'),
        dict(issue='precision and combined repeats', resolution='uint16 ADC codes + actual double baseline mean + optional XOR checked byte exact; combined8draw moments included'),
        dict(issue='nested A/head failure accounting and end caps', resolution='A completion durable independently from outer head; unknown head in-flight retained; exact final wall/output checks'),
    ],
    contracts=dict(
        data_roles='All legal TRAIN retained; full REAL TRAIN422 logical groups is distinct from old reference419. SOURCE information census includes8192/1599internal_val/512fixedDEV; model evaluation only6593TRAIN, REAL DEV values0.',
        external='One verified bound acquisition. Other historical raw pools are metadata-only, exposure unknown, not newly blind or external. Their waveform values were not read by this reviewer.',
        whole_pipeline_CV='Each fold must refit all data-dependent statistics/scales/bins,005,A/features,adapters and other fitted modules. Existing head-only holdout does not establish unseen whole-system generalization.',
        tolerances='Normative gain3%,slowtail3%,noiseAC10%,extra Gaussian sigma1 software ns; no hardware calibration. Max7 registered cases of mean2banks self-distribution distances. Positive finite requiredT only; zero/undefined cannot PASS. Numeric map frozen after013 before014 candidate.',
        machine_rule='ANY one primary direction / ALL12 bank×batch×hyp cells, +lambda1LSB only. adequate precedes repair_needed then inconclusive; all unsafe direction evidence retained. Uses J=wave+2QPSD diagnostic, not final scientific acceptance.',
        actual_parameter_probe='Actual original model010 functional_call substitutes only two head tensors1288 at theta2000. Five directions/30 endpoints, both signs/all3scales retained. Nominal lambda from analytic map over both MC banks; actual RMS/max separately reported. Saved012 direction and prospective float64 AdamW formula are not newly computed new-bank gradients or a bitwise native optimizer step.',
        sensitivity_scope='Both random banks are finite TRAIN MC libraries and share pooled scale/tolerance choices. They are not held-out physical acquisitions. Distribution self-distances do not by themselves bound per-event voltage error or identify true conditional response.',
        route='Short candidates<=12, teacher2×2×2=8 separate, final confirmation/deployment separate fixed budget. Inconclusive moves to registered C0/C1 short trials rather than more single-point audit. Missing independent acquisition does not block software delivery or justify relabeling science success.'),
    line_references=dict(
        current_head_restore='scripts/run_wp1_013.py:112',
        original_A_device_before_wrapper='scripts/run_wp1_013.py:119 and src/n01/model010.py:98',
        saved012_weighted_Q_gradient='scripts/run_mechanism012.py:750 and :753',
        true_parameter_substitution='scripts/wp1_parameter_probe.py:94',
        analytic_scale_map='scripts/wp1_parameter_probe.py:106',
        missing_and_undefined_selector='scripts/wp1_parameter_probe.py:183',
        original_native_loss='scripts/wp1_evaluation.py:67',
        actual_first128_proof='scripts/run_wp1_013.py:234',
        lossless_save='scripts/run_wp1_013.py:43',
        combined8draws='scripts/run_wp1_013.py:298',
        tolerance_formula='scripts/run_wp1_013.py:350',
        final_limits='scripts/run_wp1_013.py:515',
        durable_nested_calls='scripts/wp1_ops.py:107',
        owned_supervision='scripts/wp1_ops.py:141',
        source_private_inventory='scripts/wp1_controller.py:169',
        source_recipe='scripts/wp1_controller.py:195',
        formal_first_and_COMPLETE_consumer='scripts/wp1_controller.py:303',
        fresh_exclusive_dispatch='scripts/wp1_controller.py:396',
        failed_partial_collection='scripts/wp1_controller.py:431'),
    source_freeze_state=dict(actual_source_freeze_performed_by_reviewer=False, draft_config_sha256=EXPECTED['wp1_013_config.json'],
        planned_source_files=83, private_payload_files=4, readonly_input_records=22, old_source_protection_records=185,
        next_authorized_root_step='Freeze actual config/source83/private4, verify package SHA and E/D backup, then exclusive native start with actual first128 proof; schedule one wake and do not continuously monitor.'),
    scope=dict(real_scientific_numeric_array_decodes=0, real_checkpoint_loads=0, real_neural_model_forwards=0,
        real_autograd_queries=0, real_optimizer_constructions=0, real_training_updates=0,
        real_DEV_waveform_reads=0, SSH_calls=0, real_remote_dispatches=0, real_source_freezes=0,
        this_closeout_used_stdlib_only=True, outputs_and_temporary_files='E:/GAN and D: only',
        earlier_independent_fixtures='explicit artificial6-row schema and manufactured12-cell decisions; no original scientific inputs'),
    not_verified=['No actual CUDA checkpoint restore/forward or native performance/peak-resource assertion before root starts.',
        'No current scientific wave/QPSD results, new tolerance numeric table, finite-parameter verdict or response acceptance established by this review.',
        'Original/private numeric input SHA claims are read from root-bound metadata; this reviewer did not decode/load these scientific inputs. Root freeze and native gate verify their actual bytes.',
        'Historical outputs/weights beyond registered old source inventories were not broadly re-audited.',
        'No new candidate, teacher run, training, CV or independent acquisition evaluation launched.'],
    interpretation=dict(engineering_prelaunch_ready=True, approved_for_root_source_freeze=True,
        scientific_response_match_not_established=True, finite_method_outcome_not_yet_observed=True),
    conclusion_zh='PASS：七份最终源码与现行合同、自分布T公式、真实参数方向及预算、保存精度、父事件身份和一次启动/回收接口闭合。可由根流程冻结83份source与4份private包后按登记预算启动013；本审核没有读取原始科学波形/CP或执行NN、AD、训练、SSH，不建立科学响应验收。')
target=OUT/'PRELAUNCH_REVIEW.json'
with target.open('x',encoding='utf8') as f:
    json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
loaded=read(target)
assert loaded['passed'] and loaded['status']=='PASS'
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
print(json.dumps(dict(status='PASS',receipt=str(target),sha256=sha(target),static_checks=len(checks),source_files=83,private_files=4,
    old_source_protection_records=185,scientific_NN_rows=0,scientific_AD_queries=0),ensure_ascii=False))
