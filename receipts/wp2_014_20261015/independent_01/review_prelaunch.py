"""Independent stdlib-only WP2 source/configuration closure. No scientific input reads."""
from datetime import datetime, timezone
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT / 'receipts/wp2_014_20261015/independent_01'
EXPECTED_SOURCES = {
    'scripts/run_wp2_014.py': 'e38616e88bd4e98dfc0c9605a0d0405dd9327abcd152b992ff07967dc629c2bc',
    'src/n01/model014.py': 'b45080dd5d99aa51c6846affde021ad1e363aba086c665beb40f8f83a0e0082c',
    'src/n01/losses014.py': '86fded3226ae86a5aaae6d9066220923fb6b0aebfeaa09cb638642498e4f4114',
    'scripts/wp2_controller.py': '3480825e94e954a46c2c633e5df6ddef514e2b7def1332bb2e7688e78ae041a3',
}
EXPECTED_DRAFT = 'cf351f3a72c1c722f8647301dbe7038693f388a7756b2a7aef4fcd4a7ec5e798'
T_SHA = 'f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b'


def load(p):
    return json.loads(p.read_text(encoding='utf8'))


def pin(p):
    return dict(path=str(p), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


checks = []


def ok(value, name):
    assert value, name
    checks.append(dict(name=name, passed=True))


sources = {}
for name, expected in EXPECTED_SOURCES.items():
    sources[name] = pin(ROOT / name)
    ok(sources[name]['sha256'] == expected, 'final_source_SHA:' + name)
    ast.parse((ROOT / name).read_bytes(), filename=name)

cfg_path = ROOT / 'wp2_014_config.json'
cfg = load(cfg_path)
ok(pin(cfg_path)['sha256'] == EXPECTED_DRAFT, 'final_prelaunch_draft_SHA')
ok(cfg['status'] == 'draft_registered_before_any_scientific_operation' and not cfg['source_files'],
   'draft_has_not_frozen_or_authorized_scientific_execution')
spec = importlib.util.spec_from_file_location('independent014_controller', ROOT / 'scripts/wp2_controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
controller.validate_config(cfg, frozen=False, driver_source=(ROOT / 'scripts/run_wp2_014.py').read_bytes())
ok(not {'torch', 'numpy', 'n01'} & set(sys.modules), 'actual_factory_controller_validation_has_no_numeric_import')

# Independent arithmetic, using literal registered sizes rather than the driver's factory.
train = 4 * 2000 * 512 * 2
evaluation_forward = 4 * 2 * 2 * 6593 * 4
probe_forward = 4 * 2 * 2 * 64 * 2
calibration_A = 2 * 128
calibration_head = 2 * 2 * 128
maps = 2 * 128 * 4 * 2 * 18
extra_AD = 4 * 2 * 2 * 3 * 13 * 3
ADC = 3 * (train + probe_forward) + 4 * 2 * 6593 * 4 * 3 * 3
budget = dict(old_A_rows=train + evaluation_forward + probe_forward + calibration_A,
              head_actual_parameter_forward_rows=train + evaluation_forward + probe_forward + calibration_head,
              head_total_kernel_rows=train + evaluation_forward + probe_forward + calibration_head + maps,
              calibration_map_rows=maps, noise005_rows=train + evaluation_forward + probe_forward,
              hard_ADC_rows=ADC, all_path_gather_rows=ADC, checkpoint_loads=2,
              optimizer_created=4, optimizer_steps=8000, G_updates=8000,
              training_backward_queries=8000, probe_AD_queries=extra_AD,
              new_AD_queries=8000 + extra_AD, integral_statistics_fits=1,
              D_updates=0, noise_updates=0, population_updates=0, C2ST_updates=0, DEV_model_rows=0)
ok(budget == cfg['budget_caps'], 'independent_literal_budget_formula_matches_all_registered_caps')

old_cfg_path = ROOT / 'wp1_013_config.json'
old_mf_path = ROOT / 'WP1_SOURCE_MANIFEST.json'
old_cfg, old_mf = load(old_cfg_path), load(old_mf_path)
ok(pin(old_mf_path)['sha256'] == controller.PRIOR_MANIFEST_SHA
   and pin(old_cfg_path)['sha256'] == old_mf['config_sha256'], 'original013_controls_unchanged')
ok(len(old_mf['files']) == 84 and len(old_cfg['protection_files']) == 185, '84_and185_old_protection_records')
for name, record in old_mf['files'].items():
    p = ROOT / name
    assert p.suffix in ('.py', '.md', '.json', '.txt'), name
    assert {k: pin(p)[k] for k in ('bytes', 'sha256')} == record, name
for name, record in old_cfg['protection_files'].items():
    p = Path(record['local_path'])
    assert p.suffix in ('.py', '.md', '.json', '.txt'), name
    assert pin(p)['sha256'] == record['sha256'], name
ok(True, 'all_original84_and185_source_metadata_SHA_verified_without_CP_or_waveform_reads')

e, prior = cfg['evaluation'], old_cfg['reference']
ok(e['bank_seeds'] == prior['bank_seeds'] and e['projection_seed'] == prior['projection']['seed']
   and e['projection_fingerprint'] == prior['projection']['fingerprint'], 'same013_evaluation_bank_and_projection_addresses')

input_metadata = {}
old_inputs = old_cfg['inputs']
for name, row in cfg['inputs'].items():
    p = Path(row['local_path'])
    assert p.drive.upper() in ('E:', 'D:') and p.stat().st_size == row['bytes']
    assert any(PurePosixPath(row['path']).is_relative_to(PurePosixPath(r)) for r in cfg['allowed_input_roots'])
    if p.suffix in ('.npz', '.pt', '.pth'):
        # Scientific inputs are NOT opened. Compare pre-existing bound SHA metadata.
        matches = [v for v in old_inputs.values() if v['sha256'] == row['sha256'] and v.get('bytes') == row['bytes']]
        if name == 'source_information':
            inventory = load(ROOT / 'receipts/wp1_013_20261015/native_repeat_fix01/collection_03/REMOTE_ARCHIVE_RECEIPT.json')['inventory']
            assert inventory['runs/primary/information/PARENT_INFORMATION.npz'] == dict(bytes=row['bytes'], sha256=row['sha256'])
            provenance = 'independently_verified_prior013_collection_inventory; no current array read'
        else:
            assert matches, name
            provenance = 'previous_frozen013_input_SHA_and_size_metadata; no current binary read'
    else:
        assert pin(p)['sha256'] == row['sha256'], name
        provenance = 'current_small_JSON_source_bytes_hashed'
    input_metadata[name] = dict(row, independent_validation=provenance)
ok(len(input_metadata) == 8, 'eight_readonly_input_paths_sizes_and_prior_bindings_consistent')

contracts = {}
for name, row in cfg['contracts'].items():
    p = Path(row['local_path'])
    assert p.suffix in ('.md', '.json')
    assert pin(p)['sha256'] == row['sha256'] and p.stat().st_size == row['bytes'], name
    contracts[name] = pin(p)
ok(len(contracts) == 7 and contracts['tolerance_table']['sha256'] == T_SHA
   and contracts['acceptance']['sha256'] == 'c3e3e5027a4b15dd9124538ecb4b8f8a16d54ab432adf920b03923eb6f272aab',
   'seven_final_contract_SHA_including_pre_candidate_T')

before = ROOT / 'receipts/wp2_014_20261015/control_versions/01_before_explicit_selection_quantifier/wp2_014_config.json'
previous = load(before)
assert pin(before)['sha256'] == 'd23cfae8dbc613e57c497d38de55d0b5ad2cecf07ac8d3d52281a365e57b5a62'
delta = copy.deepcopy(cfg)
selection = delta.pop('development_selection')
delta['contracts']['acceptance'] = previous['contracts']['acceptance']
ok(delta == previous, 'final_control_delta_only_explicit_selection_and_acceptance_binding')
ok(selection['noninferiority_cells'] == 'ALL6_same_candidate_2seeds_x3hypotheses'
   and selection['noninferiority_delta'] == 'mean2bank(final_distance_minus_paired_A_distance)'
   and selection['required_calibrated_metric_missing_or_undefined'] == 'pending_not_eligible_not_passed'
   and selection['tie_absolute_epsilon'] == 1e-12 and selection['tie_preference'] == 'C0'
   and selection['empty_eligible_set'] == 'keep_original_C0_engineering_entry_without_promoting_weights',
   'predeclared_selection_quantifiers_and_pending_no_weight_promotion')

names = set(old_mf['files']) | set(controller.OWNED) | {'WP1_SOURCE_MANIFEST.json'}
paths = {n: ROOT / n for n in names}
for row in cfg['contracts'].values():
    name = PurePosixPath(row['path']).relative_to(PurePosixPath(controller.REMOTE)).as_posix()
    if name in paths:
        assert pin(paths[name])['sha256'] == row['sha256']
    else:
        paths[name] = Path(row['local_path'])
closure = controller.project_import_closure(paths, lambda n: paths[n].read_bytes())
ok(len(paths) == 95 and len(closure) == 33 and 'scripts/stability011_metrics.py' in closure,
   'planned95_source_inventory_has33_reachable_lazy_project_imports')

controller_receipt = OUT / 'controller_stdlib_01/ENGINEERING_RECEIPT.json'
cr = load(controller_receipt)
ok(pin(controller_receipt)['sha256'] == '1524771cfeda45d54e424fc4c2c18f39b42dc1a0ac249e3d0120c3aae079a93f'
   and cr['passed_count'] == 6 and cr['native_test_process_exit'] == 0
   and not cr['numerical_packages_imported'], 'independent_actual_controller_stdlib_six_groups_native0')

ready_path = ROOT / 'receipts/wp2_014_20261015/implementation_01/READY.json'
ready = load(ready_path)
ok(pin(ready_path)['sha256'] == '24012cf772eaf3ed28e12ef74f53259828e07eb5c7faf4c57837dc0daa051b5a', 'implementation_ready_receipt_pin')
engineering = {}
for name in ('focused_full_manual_receipt', 'final_amendment_QA_receipt', 'protected84_receipt', 'budget_resource_receipt'):
    row = ready[name]
    p = ROOT / row['path']
    assert pin(p)['sha256'] == row['sha256'] and p.stat().st_size == row['bytes']
    engineering[name] = pin(p)
manual = load(Path(engineering['focused_full_manual_receipt']['path']))
focus = load(Path(engineering['final_amendment_QA_receipt']['path']))
assert manual['status'] == focus['status'] == 'PASS' and len(manual['checks']) == 9
ok(manual['new_source_SHA256']['scripts/run_wp2_014.py'] == 'a317557308a7644b89fb266d119c11d5b1cef3c52c83bb43e0144b0026d76cc0'
   and focus['source_SHA256'] == {n: EXPECTED_SOURCES[n] for n in ready['new_sources']},
   'honest_prior_manual_driver_SHA_and_final_focused_bridge_not_claiming_full_final_rerun')

semantics = [
    dict(topic='C0 objective equivalence', evidence='src/n01/losses014.py:102; src/n01/train_shape010.py:283',
         finding='C0 directly reuses original components and native QPSD, same wave/2 distribution/.01 regular and frozen hard observation; initialization, seeds and RMS are the newly registered common design, not a replay of historical010 training.'),
    dict(topic='C1 single changed factor', evidence='src/n01/losses014.py:24; src/n01/losses014.py:77',
         finding='Signed Q/T/four fixed disjoint integrals on all rows, frozen allREAL TRAIN median-absolute scales floored by ADC quadrature, asinh, equal coordinate/joint16direction W1; native QPSD only no_grad diagnostic. No denominator or output-Q conditioning in the C1 target.'),
    dict(topic='paired model and RMS initialization', evidence='src/n01/model014.py:23; src/n01/model014.py:111; scripts/run_wp2_014.py:750',
         finding='Same1288 head and gain.05 throughout. One seed-specific cached A calibration, fixed16 lower bracket, <=1LSB actual final RMS; only initial parameters are scaled. C0/C1 load the identical in-memory calibrated head and reset explicit sampler/shape/noise/extension/wave streams. Future capacity is not scaled.'),
    dict(topic='frozen old response and whitelist', evidence='scripts/run_wp2_014.py:243; scripts/run_wp2_014.py:337; scripts/run_wp2_014.py:698',
         finding='OriginalA19k and005 loaded exactly once each for whole batch, no010 headCP. Wrapper is never cast wholesale; only original device placement occurs before wrapper construction. Optimizer contains only basis_head.weight/bias; tensor hashes and original file hashes checked at boundaries.'),
    dict(topic='source before numeric import', evidence='scripts/run_wp2_014.py:101; scripts/run_wp2_014.py:724; scripts/wp2_controller.py:126',
         finding='Full config/source/input/contract SHA gate precedes numeric import and private decode. Controller extracts only three pure factories, refuses changed frozen settings, packages reachable lazy CDF dependency and preserves84/185 old bytes. Generic reused stdlib journal loads at driver import; no numeric code is imported then.'),
    dict(topic='gradient query budget', evidence='scripts/run_wp2_014.py:443; scripts/run_wp2_014.py:519',
         finding='At G0/G2000,2banks×3hyp×up to13 nonempty strata, 3scalar AD terms each; total stored vector is their sum. Probe preserves accumulated .grad, creates no optimizer step and advances no training streams. Conditional subset normalized losses are not additive row/global contributions.'),
    dict(topic='hard evaluation and storage', evidence='scripts/run_wp2_014.py:563; scripts/run_wp2_014.py:678',
         finding='Every trial initial A+head and final head: full6593×4×2 SOURCE and21047REAL, original ADC/crop/baseline, raw signed Q and all four suites. Exact hard codes/baseline means/optional IEEE XOR and original100prefix are saved, all flags retained including unavailable REAL ADC metadata as unknown. Each-bank comparisons are diagnostics; root uses frozen mean2bank ALL6 rule, not a pooled8 empirical distance.'),
    dict(topic='failure and genuine cost', evidence='scripts/run_wp2_014.py:155; scripts/run_wp2_014.py:808; scripts/wp2_controller.py:370',
         finding='Nested A/head and calibration-map actual/attempted charges are durable; a failed call leaves completion unknown. Any exception ends serial batch, saves FAILED/status before best-effort CP and never continues next trial. SAME snapshot FAILED/status/native exit rejects historical firstG1.'),
    dict(topic='owned resources and collection', evidence='scripts/wp2_controller.py:457; scripts/wp2_controller.py:527; scripts/wp2_controller.py:687',
         finding='Fresh root, inherited scientific lock, one owned child process-group; RSS includes supervisor tree, only child group is stopped. Pertrial7200s/1536MiB and batch28800s/6144MiB enforced. Whole remote inventory SHA and selected local bytes are distinguished; raw NPZ retained remotely, all_raw_artifacts_collected=false and independent_numerical_readback_complete=false.'),
]

roles = load(ROOT / 'WP2_DATA_ROLES.json')
ok(roles['evaluation']['split'] == 'TRAIN_development'
   and roles['evaluation']['unseen_parent_generalization'] == 'not_estimated'
   and roles['evaluation']['independent_acquisition_generalization'] == 'unverified'
   and roles['future_confirmation']['all_data_dependent_components_refit_per_fold'],
   'exposed_TRAIN_model_selection_and_future_full_pipeline_refit_boundaries')

result = dict(schema='n01/wp2-independent-prelaunch-review/v1',
    UTC=datetime.now(timezone.utc).isoformat(), status='PASS', passed=True,
    ready_for_root_freeze=True, scientific_launch_performed_or_authorized_by_reviewer=False,
    run_id=cfg['run_id'], final_draft=pin(cfg_path), final_sources=sources,
    contracts=contracts, input_metadata_checks=input_metadata,
    checks=checks, static_semantic_review=semantics,
    native_budgets=budget,
    mandatory_on_complete=['old_A_rows','head_actual_parameter_forward_rows','noise005_rows','hard_ADC_rows',
                           'all_path_gather_rows','checkpoint_loads','optimizer_created','optimizer_steps',
                           'G_updates','training_backward_queries','integral_statistics_fits'],
    maximum_only_due_to_empty_strata_or_already_small_initialization=['calibration_map_rows','head_total_kernel_rows',
                                                                  'probe_AD_queries','new_AD_queries'],
    paired_trial_order=cfg['candidate_trial_order'], development_selection=selection,
    protected_old_source_records=185, old_source_inventory_count=84,
    planned_new_source_count=95, planned_protection_records=271,
    project_import_closure=closure,
    reused_artificial_engineering=engineering,
    independently_rerun_controller_interfaces=pin(controller_receipt),
    clarification=pin(ROOT / 'receipts/wp2_014_20261015/SELECTION_QUANTIFIER_CLARIFICATION.json'),
    resolved_issue='Prelaunch bank aggregation and ALL6 noninferiority quantifiers now explicit; scoring/tie/fallback separate from weight or response acceptance.',
    unresolved_engineering_blockers=[],
    limitations=[
        'Native full-population CUDA wall/RSS/output and actual RMS calibration are not established by tiny artificial checks; enforced limits may stop a real run and costs must remain recorded.',
        'The112s manual9-group check used prior driver a317...; final focused bridge plus independent static review covers subsequent changes. This review did not repeat that entire numerical suite.',
        'This review independently hashed only source, small JSON and contracts; scientific CP/dataset/NPZ inputs were not opened. Their bindings were checked against prior frozen metadata and prior independently verified013 inventory.',
        'Root still must freeze/package and verify actual deployed source SHA, valid first formal G1 plus same-snapshot health, then set one wake. No native dispatch or science was performed here.',
        'Local selected collection will not establish independent numerical readback of omitted raw NPZ; final numeric acceptance requires later saved-array verification within available E/D space.',
        'TRAIN includes originalA/005 fit population; exposedDEV remains exposed,422 logical groups still one confirmed acquisition. This short study selects a development path and does not establish physical response match or independent acquisition generalization.',
    ],
    reviewer_actual=dict(numeric_package_imports=0, scientific_input_binary_reads=0, checkpoint_loads=0,
                         scientific_NN_rows=0, AD_queries=0, optimizer_or_sampler_created=0,
                         training_updates=0, DEV_waveform_reads=0, SSH_calls=0, native_jobs_started=0,
                         artificial_stdlib_checker_groups=6, old42_ops_rerun=False, CPU_requested_max=2),
    scientific_status='response_match_not_established', scientific_acceptance=False)
target = OUT / 'PRELAUNCH_REVIEW.json'
with target.open('x', encoding='utf8', newline='\n') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps(dict(status='PASS', receipt=pin(target), configuration=pin(cfg_path), sources=sources,
                     source_count_planned=len(paths), import_closure=len(closure), checks=len(checks)), ensure_ascii=False))
