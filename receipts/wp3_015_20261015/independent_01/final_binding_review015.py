"""015 independent final source/package bridge. Stdlib, no scientific inputs decoded."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
import tarfile

ROOT = Path('E:/GAN/research/N01')
STAGE = ROOT / 'receipts/wp3_015_20261015'
OUT = STAGE / 'independent_01'
PACKAGE = STAGE / 'runtime_package_01'
REMOTE = '/home/rylan/projects/ej276d-n01/wp3_015_20261015'
OLD_REMOTE = '/home/rylan/projects/ej276d-n01/wp2_014_20261015'
PINS = {
    'src/n01/model015.py': '149f7faefc9ca982700eea8c7ce7850e9c679d74985cb4fce90708743299dadc',
    'scripts/run_wp3_015.py': '5f688f828fdc563822f3fb28ddf1da2d558654280d4756798bd642f5a6445676',
    'scripts/wp3_controller.py': '6415e2c4c8a6339b0187ff564669adf6b15111015cfc41c0fc31fd71b02b2fcd',
}
CFG_SHA = '54c096ad1a5e112493798e5c6c8db481cd3c79b78747200af0bbd47bccba5394'
MF_SHA = 'f1255dbff0e47edb7ec20e76c7d90057201f9a56f1184b7896f0c8cb0a67825a'
TAR_SHA = '9ea44db9ab3fd36cfeb8b68f877c30f89e9a441bcd98fcd31666c6b964bbf422'
DRAFT_SHA = '24dc6a1bdb240ddfe363a3ea07fe811c5cd23ac683a2ab7bd89af4e7a45d5b59'
OLD_CFG_SHA = 'abb1de68d5ca312d1dbaf14dc058d7514429321b0813aa00c70ff37b20e92e32'
OLD_MF_SHA = '14db81d3a08a0edede97f544ed8ec4c63147138223c00fc38d833bbc278e2684'
INV_SHA = '4cf27f6c771294561a474ef2de9574ab711d9322c588da488f0d9ce2e49792d5'
T_SHA = 'f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b'
SOURCE_REVIEW_SHA = 'f800ae2422502025b268de767c8b32a7732d7478febb2de5cb16f97a05447b0b'
READY_SHA = 'dbf7f2a37142a12963d9ea75ab1972920d452e3a5516386f41f629f2a682d1b7'
CPU_SHA = '8651b56b8283e061eedf4e3709f8db426a7170327ad64af04222643c58be34c6'
OPS_SHA = 'c62a66f6922df3c7838e60ab3c5588461124cc5e1d81d7cd94320c58c9bfe126'
TEXT_SUFFIXES = {'.py', '.json', '.md', '.txt'}
CHECKS = []
READ_FILES = {}


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_text_bytes(path):
    path = Path(path)
    require(path.suffix.lower() in TEXT_SUFFIXES, 'non-source/JSON/doc read refused: ' + str(path))
    data = path.read_bytes()
    READ_FILES[str(path)] = dict(bytes=len(data), sha256=digest(data))
    return data


def record(path):
    data = read_text_bytes(path)
    return dict(path=str(path), bytes=len(data), sha256=digest(data))


def load(path, expected=None):
    data = read_text_bytes(path)
    if expected is not None:
        require(digest(data) == expected, 'JSON/source SHA differs: ' + str(path))
    return json.loads(data)


def named(data):
    return dict(bytes=len(data), sha256=digest(data))


def check(name, **details):
    CHECKS.append(dict(name=name, passed=True, **details))


def write_new(path, obj):
    with path.open('xb') as stream:
        stream.write((json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def main():
    cfg = load(PACKAGE / 'wp3_015_config.json', CFG_SHA)
    manifest = load(PACKAGE / 'WP3_SOURCE_MANIFEST.json', MF_SHA)
    freeze = load(PACKAGE / 'FREEZE_RECEIPT.json', '9fb8149e50b94d15da08aa4f4842c1fa5baaf1e23798a942ed36e2d6b776227c')
    draft_path = STAGE / 'control_versions/01_draft_before_frozen_adoption/wp3_015_config.json'
    draft = load(draft_path, DRAFT_SHA)
    root_cfg = load(ROOT / 'wp3_015_config.json', CFG_SHA)
    root_mf = load(ROOT / 'WP3_SOURCE_MANIFEST.json', MF_SHA)
    require(root_cfg == cfg and root_mf == manifest, 'root frozen controls differ from package')
    freeze_keys = {'status', 'source_files', 'protection_files', 'frozen_UTC'}
    require({k: v for k, v in cfg.items() if k not in freeze_keys} ==
            {k: v for k, v in draft.items() if k not in freeze_keys}, 'non-freeze configuration changed')
    changed = sorted(k for k in set(cfg) | set(draft) if cfg.get(k) != draft.get(k))
    require(set(changed) == freeze_keys, 'unexpected configuration diff')
    check('draft_to_frozen_only_four_freeze_fields_root_exact_adoption', changed_fields=changed)

    for name, expected in PINS.items():
        require(record(ROOT / name)['sha256'] == expected, 'final source SHA differs: ' + name)
    source_review = load(OUT / 'SOURCE_REVIEW.json', SOURCE_REVIEW_SHA)
    ready = load(STAGE / 'implementation_01/READY.json', READY_SHA)
    cpu = load(STAGE / 'engineering_01/artificial_cpu_02/ENGINEERING_RECEIPT.json', CPU_SHA)
    ops = load(STAGE / 'engineering_01/controller_stdlib_03/ENGINEERING_RECEIPT.json', OPS_SHA)
    require(source_review['passed'] is True and len(source_review['checks']) == 18, 'source/method review not passed')
    require(source_review['source_SHA256'] == PINS, 'source review pins differ')
    require(cpu['status'] == ops['status'] == 'PASS' and len(cpu['checks']) == 9 and len(ops['checks']) == 5,
            'reused engineering checks not passed')
    require(ops['source_SHA256'] == PINS and ready['exact_budget_caps'] == cfg['budget_caps'], 'READY/control bridge differs')
    require(all(ready['sources'][k]['sha256'] == v for k, v in PINS.items()), 'READY source differs')
    require(all(cpu['sources'][k] == v for k, v in PINS.items() if k in cpu['sources']), 'numeric receipt source differs')
    require(ready['numeric_artificial_QA']['native_exit_code'] == ready['control_artificial_QA']['native_exit_code'] == 0,
            'artificial native exit differs')
    check('unchanged_three_sources_reuse18_source9_numeric5_control_checks_without_rerun')

    # Import only the source-pinned stdlib controller; its numerical runner is AST-parsed, never imported.
    spec = importlib.util.spec_from_file_location('_independent015_controller', ROOT / 'scripts/wp3_controller.py')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    controller.validate_config(cfg, driver_source=read_text_bytes(ROOT / 'scripts/run_wp3_015.py'))
    require('numpy' not in sys.modules and 'torch' not in sys.modules, 'numerical import occurred')
    check('actual_stdlib_controller_literal_factory_validation_no_numerical_import')

    require(manifest['schema'] == 'n01/wp3-source/v1' and manifest['run_id'] == cfg['run_id'] == 'wp3_015_20261015',
            'manifest/run identity differs')
    require(manifest['config_sha256'] == CFG_SHA and manifest['prior_WP2_manifest_sha256'] == OLD_MF_SHA,
            'manifest linkage differs')
    require(manifest['before_scientific_decode'] is True and manifest['public_source_only'] is True
            and manifest['scientific_inputs_copied'] is False, 'payload role differs')
    require(len(manifest['files']) == len(cfg['source_files']) == 105, 'source count differs')
    tar_path = PACKAGE / 'wp3_source.tar.gz'
    tar_bytes = tar_path.read_bytes()  # Source-only archive, not a numeric dataset/archive.
    require(len(tar_bytes) == freeze['source_package_bytes'] == 720300 and digest(tar_bytes) == TAR_SHA,
            'source package SHA/size differs')
    require(freeze['source_package_sha256'] == TAR_SHA and freeze['config_sha256'] == CFG_SHA
            and freeze['source_manifest_sha256'] == MF_SHA and freeze['source_file_count'] == 105,
            'freeze receipt package pins differ')
    require(freeze['scientific_started'] is False and freeze['array_decodes'] == freeze['checkpoint_loads'] == freeze['model_forwards'] == 0,
            'freeze receipt scientific boundary differs')
    packed = {}
    with tarfile.open(tar_path, 'r:gz') as bundle:
        for item in bundle:
            name = item.name
            parts = name.split('/')
            require(item.isfile() and name not in packed and not name.startswith('/') and '\\' not in name
                    and ':' not in name and all(s not in ('', '.', '..') for s in parts), 'unsafe/duplicate/nonregular member')
            require(PurePosixPath(name).suffix in TEXT_SUFFIXES, 'numeric payload in source package')
            packed[name] = bundle.extractfile(item).read()
            require(len(packed[name]) == item.size, 'tar member size differs')
    require(len(packed) == 107 and set(packed) == set(freeze['source_inventory']), '107 package inventory differs')
    require({k: named(v) for k, v in packed.items()} == freeze['source_inventory'], 'package member SHA/size differs')
    require(set(packed) == set(manifest['files']) | {'wp3_015_config.json', 'WP3_SOURCE_MANIFEST.json'}, 'bundle inventory split differs')
    require(packed['wp3_015_config.json'] == read_text_bytes(PACKAGE / 'wp3_015_config.json') and
            packed['WP3_SOURCE_MANIFEST.json'] == read_text_bytes(PACKAGE / 'WP3_SOURCE_MANIFEST.json'), 'packed controls not exact bytes')
    for name, row in manifest['files'].items():
        require(named(packed[name]) == row, 'manifest source member differs: ' + name)
        require(cfg['source_files'][name] == dict(path=REMOTE + '/' + name, **row), 'native source binding differs: ' + name)
        require(named(read_text_bytes(ROOT / name)) == row, 'local source changed: ' + name)
    check('source_only_tar107_regular_unique_members_105_local_sources_exact_sha_size_no_extraction',
          source_files=105, package_members=107, package_bytes=len(tar_bytes), import_closure=len(manifest['project_import_closure']))

    old_cfg = load(ROOT / 'wp2_014_config.json', OLD_CFG_SHA)
    old_mf = load(ROOT / 'WP2_SOURCE_MANIFEST.json', OLD_MF_SHA)
    require(len(old_mf['files']) == 95 and old_mf['config_sha256'] == OLD_CFG_SHA, 'original95/97 chain differs')
    for name, row in old_mf['files'].items():
        require(manifest['files'][name] == row and named(read_text_bytes(ROOT / name)) == row, 'old source changed: ' + name)
    require(len(old_cfg['protection_files']) == 271 and len(cfg['protection_files']) == 368, 'protection count differs')
    require(all(cfg['protection_files'][k] == v for k, v in old_cfg['protection_files'].items()), 'old271 binding changed')
    expected_add = {'wp2_014:' + n for n in old_mf['files']} | {'wp2_014:WP2_SOURCE_MANIFEST.json', 'wp2_014:wp2_014_config.json'}
    require(set(cfg['protection_files']) - set(old_cfg['protection_files']) == expected_add, 'new97 protection identities differ')
    for key, row in cfg['protection_files'].items():
        require(named(read_text_bytes(row['local_path'])) == {k: row[k] for k in ('bytes', 'sha256')},
                'protected local source/metadata changed: ' + key)
        if key in expected_add:
            require(row['path'] == OLD_REMOTE + '/' + key.split(':', 1)[1], 'new protected native path differs')
    require(freeze['protected_source_records'] == 368 and freeze['all_original_files_unchanged'] is True,
            'freeze protection receipt differs')
    check('original95_sources_and97_controls_old271_protections_physically_text_hashed_368_total')

    require(len(cfg['inputs']) == 23 and len(old_cfg['inputs']) == 8, '23-input split differs')
    require(all(cfg['inputs'][k] == v for k, v in old_cfg['inputs'].items()), 'one of original8 inputs changed')
    inv_row = cfg['remote_inventory_evidence']
    inventory = load(inv_row['local_path'], INV_SHA)
    require(named(packed['WP2_REMOTE_FULL_INVENTORY.json']) == {k: inv_row[k] for k in ('bytes', 'sha256')}, 'inventory metadata copy differs')
    require(inventory['whole_remote_inventory_verified'] is True and len(inventory['whole_remote_inventory']) == 318
            and inventory['config_sha256'] == OLD_CFG_SHA and inventory['source_manifest_sha256'] == OLD_MF_SHA,
            'original318 inventory identity differs')
    remote_slots = []
    for seed in range(2):
        for bank in range(2):
            for hyp in ('fast', 'slow', 'common_level'):
                key = f'mean_seed{seed}_bank{bank}_{hyp}'
                row = cfg['inputs'][key]
                expected = f'runs/primary/artifacts/seed{seed}_C1/final_evaluation/bank{bank}_final_{hyp}_REPORT.npz'
                require(controller.remote_input_binding015(row, inventory) == expected, 'remote REPORT lineage differs')
                require(not row.get('local_path') and row.get('remote_only_saved014_REPORT') is True, 'REPORT became local or role changed')
                remote_slots.append(key)
    for key in ('C1_head_seed0', 'C1_head_seed1', 'integral_statistics'):
        controller.remote_input_binding015(cfg['inputs'][key], inventory)
    # Safe JSON inputs can be checked locally, but binary data/checkpoints are metadata-only in this audit.
    safe_json_inputs = []
    for key, row in cfg['inputs'].items():
        if row.get('local_path') and Path(row['local_path']).suffix == '.json':
            require(named(read_text_bytes(row['local_path'])) == {k: row[k] for k in ('bytes', 'sha256')}, 'input JSON changed: ' + key)
            safe_json_inputs.append(key)
    check('23inputs_original8_exact_two_C1_heads_integral_and12_remote_REPORT_provenance_only',
          remote_REPORT_slots=remote_slots, prior_inventory_records=318, locally_hashed_JSON_inputs=safe_json_inputs,
          remote_physical_NPZ_or_CP_SHA_rechecked_here=False, native_install_start_driver_physical_SHA_required=True)

    require(len(cfg['contracts']) == 11 and len(old_cfg['contracts']) == 7, 'contract count differs')
    for key, row in cfg['contracts'].items():
        data = read_text_bytes(row['local_path'])
        require(named(data) == {k: row[k] for k in ('bytes', 'sha256')}, 'current contract changed: ' + key)
        name = PurePosixPath(row['path']).relative_to(PurePosixPath(REMOTE)).as_posix()
        require(packed[name] == data, 'contract package copy differs: ' + key)
    for key, row in old_cfg['contracts'].items():
        require(named(read_text_bytes(row['local_path'])) == {k: row[k] for k in ('bytes', 'sha256')}, 'original7 contract changed: ' + key)
    require(cfg['contracts']['tolerance_table']['sha256'] == T_SHA, 'original T changed')
    design = load(ROOT / 'wp3_015_design.json')
    require(design['exact_budget_caps'] == cfg['budget_caps'] == ready['exact_budget_caps'], 'design/runner budget mismatch')
    require(design['implementation']['sha256'] == READY_SHA and design['independent_source_review']['sha256'] == SOURCE_REVIEW_SHA,
            'final readiness design links differ')
    require(design['computation_started'] is False and design['exact_NN_AD_CP_caps_pending_implementation'] is False,
            'final design stage differs')
    check('11current_contracts_and_original7_contract_bytes_unchanged_T_fixed_final_design_ready')

    caps = cfg['budget_caps']
    training_rows = 4 * 2000 * 512 * 2
    extra_rows = 4 * 2 * 2 * 6593 * 4
    require(caps['old_A_rows'] == caps['head_actual_parameter_forward_rows'] == caps['head_total_kernel_rows'] == caps['noise005_rows'] == training_rows + extra_rows == 8613952,
            'NN caps formula differs')
    require(caps['mean_adapter_rows'] == 2 * (2000 * 1024 + 2 * 2 * 6593 * 4) == 4306976, 'mean cap formula differs')
    require(caps['hard_ADC_rows'] == caps['all_path_gather_rows'] == training_rows * 3 + 4 * 3 * 2 * 6593 * 4 * 3 == 26474784,
            'ADC/gather cap formula differs')
    require(caps['checkpoint_loads'] == caps['optimizer_created'] == 4 and
            all(caps[k] == 8000 for k in ('G_updates', 'optimizer_steps', 'training_backward_queries', 'new_AD_queries')),
            'CP/optimizer/AD cap differs')
    zero_keys = ('calibration_map_rows', 'integral_statistics_fits', 'probe_AD_queries', 'D_updates', 'noise_updates', 'population_updates', 'C2ST_updates', 'DEV_model_rows')
    require(all(caps[k] == 0 for k in zero_keys), 'forbidden operation budget nonzero')
    require([caps[k] for k in ('mean_evidence_REPORT_files', 'mean_evidence_REPORT_parent_rows', 'mean_evidence_REAL_rows', 'mean_evidence_tests')] == [12, 12 * 6593, 21047, 6],
            'gate read budget differs')
    require(cfg['candidate_trial_order'] == ['seed0_M0', 'seed0_M1', 'seed1_M0', 'seed1_M1'] and
            cfg['cumulative_training_before_run'] == 167667 and cfg['global_completed_short_trials_before'] == 4
            and cfg['WP3_reserved_short_trials_this_block'] == 4 and cfg['WP3_remaining_unreserved_maximum'] == 2,
            'fixed route/trial budget differs')
    check('all_exact_maximum_NN_mean_ADC_CP_G_AD_and_gate_only_caps_derive_without_execution', budget_caps=caps)

    gate = cfg['mean_evidence']
    require(gate['minimum_RMS_V'] == 1e-7 and gate['SE_multiplier'] == 3.0 and gate['minimum_bank_residual_cosine'] == 0.5
            and gate['ALL_seed_hypothesis_cells'] == 6 and gate['group_is_independent_acquisition'] is False
            and gate['not_conditional_mean_identification'] is True, 'gate quantifier or scientific boundary differs')
    require(cfg['TRAIN_counts'] == dict(source=6593, real=21047, REAL_logical_groups=422), 'population changed')
    require(cfg['mean_adapter']['control_Q_slice'] == [110, 366] and cfg['mean_adapter']['control_tail_slice'] == [156, 366]
            and cfg['mean_adapter']['M0_trainable_parameters'] == 1288 and cfg['mean_adapter']['M1_trainable_parameters'] == 6
            and cfg['mean_adapter']['stochastic_amplitude_from_original_A'] is True, 'one-factor adapter changed')
    statistics = load(cfg['inputs']['statistics']['local_path'])
    require(statistics['loss']['q_reference'] == 3.7103998046755806 and statistics['loss']['q_floor'] == 0.0005319378913383611,
            'original q-reference/floor differs')
    roles = load(cfg['contracts']['data_roles']['local_path'])
    require(roles['new_mean_evidence_role']['SOURCE_or_DEV_waveforms_read'] is False
            and roles['REAL']['confirmed_independent_acquisitions'] == 1 and roles['REAL']['DEV_numeric_decode'] is False,
            'data roles changed')
    require(cfg['scientific_status'] == roles['scientific_status'] == 'response_match_not_established', 'scientific state expanded')
    check('ALL6_positive_gate_or_legal_no_CP_NN_AD_G_skip_SOURCE_controls_and_roles_no_conditional_truth_claim')

    choice = cfg['development_selection']
    require(choice['noninferiority_references'] == ['paired_A', 'common_initial_C1_reference']
            and choice['noninferiority_cells'] == 'ALL6_same_candidate_2seeds_x3hypotheses'
            and choice['noninferiority_components'] == 'each_positive_defined_T_component'
            and choice['bank_distance_aggregation'] == 'mean_of2_bank_distances_not_pooled8'
            and choice['required_calibrated_metric_missing_or_undefined'] == 'pending_not_eligible_not_pass'
            and choice['score'] == 'arithmetic_mean12_final_hard_wave_distances'
            and choice['tie_preference'] == 'M0' and choice['tie_absolute_epsilon'] == 1e-12
            and choice['replace_research_reference_only_if_score_below_common_initial_by_more_than'] == 1e-12
            and choice['otherwise'] == 'retain014_C1_reference_mean_disabled'
            and choice['reference_selection_implies_weight_promotion'] is False, 'development selection rule differs')
    require(cfg['one_active_scientific_job'] is True and cfg['serial_runs'] is True and
            cfg['automatic_restart'] is False and cfg['overwrite_outputs'] is False and cfg['DEV_numeric_read'] is False,
            'lifecycle/DEV boundary changed')
    require(cfg['CPU_threads'] == 12 and cfg['run_resource_limits'] == dict(GPU_GiB=10, job_tree_RSS_GiB=8, wall_seconds=7200, output_MiB=1536)
            and cfg['resource_limits'] == dict(GPU_GiB=10, job_tree_RSS_GiB=8, wall_seconds=28800, output_MiB=6144, poll_seconds=10, TERM_grace_seconds=20),
            'resource caps differ')
    require(cfg['collection_policy']['selected_archive_MiB'] == 512 and cfg['collection_policy']['raw_NPZ_retained_remote'] is True
            and cfg['collection_policy']['all_raw_collected_claim_allowed'] is False, 'selective collection boundary differs')
    require('one future wake' in cfg['long_job_policy'] and 'same-snapshot health' in cfg['long_job_policy'], 'single-wake policy differs')
    require(record(ROOT / cfg['user_authority']['path'])['sha256'] == cfg['user_authority']['sha256'] ==
            '56190995374eb2a765a3c8589736514154dba7ff33aa1309b8b1de493e846ebb', 'user master authority differs')
    check('ALL6_two_references_selection_separate_usefulness_absolute_T_science_and_single_start_single_wake')

    require('numpy' not in sys.modules and 'torch' not in sys.modules, 'numeric import occurred')
    for name, expected in PINS.items():
        require(record(ROOT / name)['sha256'] == expected, 'source changed during bridge')
    actual_operations = dict(source_and_JSON_read_only=True, NPZ_reads=0, checkpoint_reads=0, model_instantiations=0,
        NN_forward_rows=0, AD_queries=0, optimizer_created=0, G_updates=0, DEV_waveform_reads=0,
        new_random_libraries_or_banks=0, SSH_calls=0, numerical_fixture_reruns=0, package_tar_extractions=0)
    pin_records = dict(config_sha256=CFG_SHA, source_manifest_sha256=MF_SHA, source_package_sha256=TAR_SHA,
        source_package_bytes=720300, draft_original_sha256=DRAFT_SHA, source_SHA256=PINS,
        source_file_count=105, package_member_count=107, protection_record_count=368,
        original_WP2_source_count=95, original_WP2_source_plus_controls=97, prior_protection_records=271,
        inputs=23, contracts=11, original_inventory_records=318, remote_only_REPORTs=12)
    common = dict(UTC=datetime.now(timezone.utc).isoformat(), passed=True, status='PASS',
        run_id='wp3_015_20261015', reviewer_role='continuing independent Sol Max reviewer, not implementation author',
        source_and_control_pins=pin_records, actual_review_operations=actual_operations, unresolved_blockers=[],
        scientific_response_acceptance=False, scientific_status='response_match_not_established')
    prelaunch = dict(common, schema='n01/wp3-independent-prelaunch-review/v1', checks=CHECKS,
        evidence=dict(source_method_review=record(OUT / 'SOURCE_REVIEW.json'), implementation_READY=record(STAGE / 'implementation_01/READY.json'),
            numeric9_artificial=record(STAGE / 'engineering_01/artificial_cpu_02/ENGINEERING_RECEIPT.json'),
            control5_artificial=record(STAGE / 'engineering_01/controller_stdlib_03/ENGINEERING_RECEIPT.json'),
            freeze=record(PACKAGE / 'FREEZE_RECEIPT.json'), independent_script=record(Path(__file__))),
        reused_engineering=dict(source_method_checks=18, numeric_artificial_checks=9, controller_stdlib_checks=5,
            rerun=False, actual_real_resource_peak_or_model_restore_verified=False),
        scope='M0 paired C1 continuation vs M1 frozen C1 with six zero mean parameters; four fixed trials only after registered ALL6 population gate.',
        boundary='12 remote REPORTs and binary input/checkpoint identities are bound to prior verified SHA metadata; this audit does not decode or independently physically rehash those binaries. Native install/start/driver must verify physical SHA before decode.',
        disposition='No unresolved source/configuration/package blocker. Root owns exclusive deployment, native gate or real-G1 same-snapshot proof, then one wake. No scientific result is established by this review.')
    write_new(OUT / 'PRELAUNCH_REVIEW.json', prelaunch)
    frozen = dict(common, schema='n01/wp3-independent-frozen-package-review/v1',
        prelaunch_review=record(OUT / 'PRELAUNCH_REVIEW.json'), freeze_receipt=record(PACKAGE / 'FREEZE_RECEIPT.json'),
        original_draft=record(draft_path), frozen_config=record(PACKAGE / 'wp3_015_config.json'),
        root_exact_config=record(ROOT / 'wp3_015_config.json'), frozen_manifest=record(PACKAGE / 'WP3_SOURCE_MANIFEST.json'),
        root_exact_manifest=record(ROOT / 'WP3_SOURCE_MANIFEST.json'), allowed_configuration_changes=changed,
        source_inventory=freeze['source_inventory'], contracts=cfg['contracts'], budget_caps=caps,
        old_protection_verification='All368 local protection records have source/doc/JSON suffixes and were physically SHA/size checked; original271 binding records are unchanged and97 original014 source/control records were added.',
        remote_only_inputs={k: cfg['inputs'][k] for k in remote_slots},
        package_payload='105 source/doc/metadata records plus exact frozen configuration and manifest;107 regular unique members, no numeric input payload; archive read without extraction.',
        pending_native_facts=['Physical SHA/size of remote12 REPORTs and other binary inputs at native install/start/driver gates',
            'Native all-six mean-evidence decision before checkpoint/NN/optimizer',
            'If gate passes: real firstG1 PID/boot/counter/immutable-hash proof and same-snapshot health',
            'Root107 Git-blob readback and D source backup are separate root deliverables'],
        conclusion_zh='最终源码、合同、预算与冻结包桥接通过，无未解决阻断。12份远端REPORT仅核库存元数据，实际SHA由原生门核验；本审核未读取权重或波形，未执行NN、AD、训练或SSH。')
    write_new(OUT / 'FROZEN_PACKAGE_REVIEW.json', frozen)
    for name in ('PRELAUNCH_REVIEW.json', 'FROZEN_PACKAGE_REVIEW.json'):
        path = OUT / name
        require(load(path)['passed'] is True, 'saved receipt did not reopen')
        print(json.dumps(record(path), ensure_ascii=False))
    print(json.dumps(dict(passed=True, check_count=len(CHECKS), scientific_operations=actual_operations), ensure_ascii=False))


if __name__ == '__main__':
    main()
