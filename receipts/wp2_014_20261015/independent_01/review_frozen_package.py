"""Stdlib-only literal and SHA bridge from approved014 draft to95-source bundle."""
from datetime import datetime, timezone
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT / 'receipts/wp2_014_20261015/independent_01'
PACKAGE = ROOT / 'receipts/wp2_014_20261015/runtime_package_01'
EXPECTED = {
    'wp2_014_config.json': 'abb1de68d5ca312d1dbaf14dc058d7514429321b0813aa00c70ff37b20e92e32',
    'WP2_SOURCE_MANIFEST.json': '14db81d3a08a0edede97f544ed8ec4c63147138223c00fc38d833bbc278e2684',
    'wp2_source.tar.gz': '1d25cf4a77e3722e8c86edfe7c5103f4905932ae37028eb3642b8690aacb056a',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pin(p):
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p.read_bytes()))


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


checks = []


def ok(value, name):
    assert value, name
    checks.append(dict(name=name, passed=True))


review = read(OUT / 'PRELAUNCH_REVIEW.json')
ok(review['status'] == 'PASS'
   and pin(OUT / 'PRELAUNCH_REVIEW.json')['sha256'] == 'd6610141534da1074622ce440b605a0bcd5f69d0f5a74e19623e8b82f58dc819',
   'approved_prelaunch_receipt_pinned')
freeze = read(PACKAGE / 'FREEZE_RECEIPT.json')
pins = {}
for name, expected in EXPECTED.items():
    pins[name] = pin(PACKAGE / name)
    ok(pins[name]['sha256'] == expected, 'actual_frozen_file_SHA:' + name)
ok(freeze['schema'] == 'n01/wp2-freeze/v1' and freeze['run_id'] == 'wp2_014_20261015'
   and freeze['source_file_count'] == 95 and freeze['protected_source_records'] == 271
   and freeze['source_package_bytes'] == 663020 and freeze['source_package_bytes'] == pins['wp2_source.tar.gz']['bytes']
   and freeze['source_package_sha256'] == EXPECTED['wp2_source.tar.gz']
   and freeze['config_sha256'] == EXPECTED['wp2_014_config.json']
   and freeze['source_manifest_sha256'] == EXPECTED['WP2_SOURCE_MANIFEST.json']
   and freeze['scientific_started'] is False and freeze['artificial_only'] is False,
   'freeze_receipt_identity_counts_and_expected_pins')

draft_path = ROOT / 'receipts/wp2_014_20261015/control_versions/02_before_frozen_adoption/wp2_014_config.json'
ok(pin(draft_path)['sha256'] == review['final_draft']['sha256']
   == 'cf351f3a72c1c722f8647301dbe7038693f388a7756b2a7aef4fcd4a7ec5e798', 'exact_approved_draft_original_preserved')
draft, cfg, mf = read(draft_path), read(PACKAGE / 'wp2_014_config.json'), read(PACKAGE / 'WP2_SOURCE_MANIFEST.json')
changed = sorted(k for k in set(draft) | set(cfg) if draft.get(k) != cfg.get(k))
ok(changed == ['frozen_UTC', 'protection_files', 'source_files', 'status'], 'all_and_only_four_freeze_fields_changed')
normalized = copy.deepcopy(cfg)
normalized.pop('frozen_UTC')
for key in ('status', 'source_files', 'protection_files'):
    normalized[key] = draft[key]
ok(normalized == draft and cfg['status'] == 'frozen', 'every_nonfreeze_value_literal_equal_to_reviewed_draft')
for name in ('wp2_014_config.json', 'WP2_SOURCE_MANIFEST.json'):
    ok((ROOT / name).read_bytes() == (PACKAGE / name).read_bytes(), 'ROOT_adopted_identical_frozen_bytes:' + name)

ok(mf['schema'] == 'n01/wp2-source/v1' and mf['run_id'] == cfg['run_id']
   and mf['config_sha256'] == EXPECTED['wp2_014_config.json']
   and len(mf['files']) == 95 and set(mf['files']) == set(cfg['source_files'])
   and mf['prior_WP1_manifest_sha256'] == 'dccd1b6c5676fb481f04e1b94a346aff2c41ada72c41c32e6a9042ac30b1fc04',
   'source_manifest_frozen_config_and_prior013_chain')
remote = cfg['remote_root']
for name, row in mf['files'].items():
    assert cfg['source_files'][name] == dict(path=remote + '/' + name, **row), name
    assert {k: pin(ROOT / name)[k] for k in ('bytes', 'sha256')} == row, name
ok(True, 'all95_local_source_bytes_and_absolute_native_source_bindings')
for name, row in review['final_sources'].items():
    assert mf['files'][name] == dict(bytes=row['bytes'], sha256=row['sha256'])
ok(True, 'four_approved_new_source_files_byte_unchanged')

old_cfg = read(ROOT / 'wp1_013_config.json')
old_mf = read(ROOT / 'WP1_SOURCE_MANIFEST.json')
expected_protected = copy.deepcopy(old_cfg['protection_files'])
old_remote = '/home/rylan/projects/ej276d-n01/wp1_013_repeat_fix01_20261015'
for name, record in old_mf['files'].items():
    assert mf['files'][name] == record, name
    expected_protected['wp1_013:' + name] = dict(path=old_remote + '/' + name, local_path=str(ROOT / name), **record)
for name, expected in (('WP1_SOURCE_MANIFEST.json', mf['prior_WP1_manifest_sha256']),
                       ('wp1_013_config.json', old_mf['config_sha256'])):
    assert pin(ROOT / name)['sha256'] == expected
    expected_protected['wp1_013:' + name] = dict(path=old_remote + '/' + name,
        local_path=str(ROOT / name), sha256=expected, bytes=(ROOT / name).stat().st_size)
ok(len(old_cfg['protection_files']) == 185 and len(old_mf['files']) == 84
   and len(expected_protected) == 271 and cfg['protection_files'] == expected_protected,
   'exact271_protection_map_is_old185_plus84_and2_controls')

archive_payloads, archive_inventory = {}, {}
with tarfile.open(PACKAGE / 'wp2_source.tar.gz', 'r:gz') as archive:
    for item in archive:
        name = item.name
        components = name.split('/')
        assert item.isfile() and not item.issym() and not item.islnk()
        assert name not in archive_payloads and not PurePosixPath(name).is_absolute()
        assert all(v not in ('', '.', '..') for v in components)
        assert '\\' not in name and ':' not in name
        assert not name.endswith(('.npz', '.pt', '.pth', '.onnx', '.bin'))
        data = archive.extractfile(item).read()
        assert len(data) == item.size
        archive_payloads[name] = data
        archive_inventory[name] = dict(bytes=len(data), sha256=digest(data))
ok(set(archive_inventory) == set(mf['files']) | {'wp2_014_config.json', 'WP2_SOURCE_MANIFEST.json'}
   and len(archive_inventory) == 97 and archive_inventory == freeze['source_inventory'],
   'all97_archive_members_regular_safe_exclusive_inventory_with_no_scientific_binary')
ok(all(archive_inventory[name] == row for name, row in mf['files'].items()), 'all95_archive_source_member_SHA_and_sizes')
for name in ('wp2_014_config.json', 'WP2_SOURCE_MANIFEST.json'):
    assert archive_payloads[name] == (PACKAGE / name).read_bytes()
ok(True, 'two_archive_control_bytes_match_frozen_files_and_ROOT_adoption')

spec = importlib.util.spec_from_file_location('independent014_frozen_controller', ROOT / 'scripts/wp2_controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
controller.validate_config(cfg, driver_source=archive_payloads['scripts/run_wp2_014.py'])
closure = controller.project_import_closure(mf['files'], lambda n: archive_payloads[n])
ok(closure == mf['project_import_closure'] == review['project_import_closure'] and len(closure) == 33,
   'same33_reachable_lazy_dependencies_in_actual_archive_including_CDF')
ok(not {'numpy', 'torch', 'n01'} & set(sys.modules), 'all_bridge_work_stdlib_only_no_numeric_import')
for name, row in cfg['contracts'].items():
    rel = PurePosixPath(row['path']).relative_to(PurePosixPath(remote)).as_posix()
    assert archive_inventory[rel] == dict(bytes=row['bytes'], sha256=row['sha256'])
ok(len(cfg['inputs']) == 8 and len(cfg['contracts']) == 7
   and cfg['budget_caps'] == review['native_budgets']
   and cfg['development_selection'] == review['development_selection'],
   'eight_input_seven_contract_budget_and_development_selection_unchanged')

result = dict(schema='n01/wp2-independent-frozen-package-review/v1',
    UTC=datetime.now(timezone.utc).isoformat(), status='PASS', passed=True,
    ready_for_root_deployment_and_formal_start=True,
    scientific_launch_performed_or_authorized_by_reviewer=False,
    prelaunch_review=pin(OUT / 'PRELAUNCH_REVIEW.json'),
    original_approved_draft=pin(draft_path), freeze_receipt=pin(PACKAGE / 'FREEZE_RECEIPT.json'),
    frozen_files=pins, changed_configuration_fields=changed,
    all_nonfreeze_configuration_values_literal_equal=True,
    source_file_count=95, archive_regular_members=97,
    prior84_source_bytes_unchanged=True, protection_record_count=271,
    protection_map_definition='original185 + original84 source records +2 original013 controls; exact literal map verified',
    reachable_lazy_dependencies=33, checks=checks,
    unresolved_engineering_blockers=[],
    independent_Git97_blob_and_D_backup_validation='root owns these separate receipts; not asserted by this package audit',
    actual_reviewer_operations=dict(stdlib_only=True, source_package_SHA_and_member_reads=True,
        scientific_input_binary_reads=0, checkpoint_loads=0, NumPy_or_Torch_imports=0,
        NN_rows=0, AD_queries=0, optimizer_or_sampler_created=0, training_updates=0,
        DEV_waveform_reads=0, SSH_calls=0, native_jobs_started=0, numeric_fixture_repeats=0),
    scope='Frozen-source bridge only. Native resource/firstG1 correctness and response acceptance are not established here.',
    scientific_status='response_match_not_established', scientific_acceptance=False)
target = OUT / 'FROZEN_PACKAGE_REVIEW.json'
with target.open('x', encoding='utf8', newline='\n') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps(dict(status='PASS', receipt=pin(target), checked_freeze_fields=changed,
                     source_files=95, regular_archive_members=97, protection_records=271), ensure_ascii=False))
