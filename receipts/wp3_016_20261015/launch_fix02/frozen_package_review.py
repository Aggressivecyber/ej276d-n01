"""Final fix02 local binding review, stdlib only; no numerical/scientific replay."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
PACKAGE=STAGE/'runtime_package_fix02'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
def check(name,**details):checks.append(dict(name=name,passed=True,**details))
def pinned(row):
    p=ROOT/row['path'];assert ref(p)==row;return p

checks=[]
pins=dict(config_sha256='5279597aed74feae87448bfc71aa1387bd711609591eabde775142fa6d1303ab',
    source_manifest_sha256='824780d297c440d0f1bc51b3ea3344e488ec42ccb46af959d9d0412d8240fc14',
    source_package_sha256='132cdafd0105bacf3d5c173737ec2d8936ded927dfac897b181f31ddf4575739')
cfg=PACKAGE/'wp3_016_config.json';mf=PACKAGE/'WP3_016_SOURCE_MANIFEST.json';tar=PACKAGE/'wp3_016_source.tar.gz'
assert sha(cfg)==pins['config_sha256'] and sha(mf)==pins['source_manifest_sha256'] and sha(tar)==pins['source_package_sha256']
assert cfg.read_bytes()==(ROOT/cfg.name).read_bytes() and mf.read_bytes()==(ROOT/mf.name).read_bytes()
p=read(cfg);m=read(mf);freeze=read(PACKAGE/'FREEZE_RECEIPT.json')
assert all(freeze[k]==v for k,v in pins.items()) and m['config_sha256']==pins['config_sha256']
assert p['status']=='frozen' and p['run_id']==m['run_id']==freeze['run_id']=='wp3_016_20261015'
assert freeze['source_file_count']==len(m['files'])==len(p['source_files'])==116
assert freeze['source_package_bytes']==tar.stat().st_size and freeze['scientific_started'] is False
assert freeze['array_decodes']==freeze['checkpoint_loads']==freeze['model_forwards']==0
adoption=read(STAGE/'FROZEN_CONTROL_ADOPTION_FIX02.json')
assert adoption['configuration_sha256']==pins['config_sha256'] and adoption['source_manifest_sha256']==pins['source_manifest_sha256']
assert adoption['source_package_sha256']==pins['source_package_sha256'] and adoption['source_controls']==118 and adoption['scientific_operations']==0
check('canonical_configuration_manifest_byte_exact_adoption_and_freeze_pins',**pins,source_count=116,controls=118)

review_path=HERE/'INDEPENDENT_FIX_REVIEW.json';review=read(review_path)
assert sha(review_path)=='e237988e6b7bdde9090997ff580d1806297157856d66b862290edeb29b6ee302'
assert review['passed'] and not review['blockers']
for name,row in review['sources'].items():
    assert m['files'][name]==row and {k:p['source_files'][name][k] for k in row}==row
    assert sha(ROOT/name)==row['sha256'] and (ROOT/name).stat().st_size==row['bytes']
assert freeze['controller_sha256']==review['sources']['scripts/wp3_016_controller.py']['sha256']
assert review['sources']['src/n01/model016.py']['sha256']=='04c63f91ea5405937a940a9aee6eaed876db79149c857bfbe6ed8cfe6dcb138b'
check('three_final_sources_bound_to_passed_independent_source_review',sources=review['sources'])

newroot='/home/rylan/projects/ej276d-n01/wp3_016_runtime_fix02_20261015'
oldroot='/home/rylan/projects/ej276d-n01/wp3_016_20261015'
assert p['remote_root']==newroot
expected_inventory={**m['files'],cfg.name:dict(sha256=sha(cfg),bytes=cfg.stat().st_size),mf.name:dict(sha256=sha(mf),bytes=mf.stat().st_size)}
assert expected_inventory==freeze['source_inventory']
inventory={}
with tarfile.open(tar,'r:gz') as archive:
    members=archive.getmembers();assert len(members)==118
    assert len({v.name for v in members})==118 and {v.name for v in members}==set(expected_inventory)
    for member in members:
        name=PurePosixPath(member.name)
        assert member.isfile() and not name.is_absolute() and '..' not in name.parts
        assert name.suffix not in ('.npz','.pt','.pth','.npy','.pkl','.pickle')
        payload=archive.extractfile(member).read()
        row=dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
        assert row==expected_inventory[member.name]
        assert payload==(ROOT/member.name).read_bytes()
        inventory[member.name]=row
for name,row in m['files'].items():
    assert p['source_files'][name]==dict(path=newroot+'/'+name,**row)
assert m['public_source_only'] and m['scientific_inputs_copied'] is False and m['before_scientific_decode']
assert m['project_import_closure']==freeze['project_import_closure']
assert set(m['project_import_closure'])<=set(m['files'])
check('all118_package_members_regular_unique_exact_local_manifest_receipt_bytes_no_scientific_payload',
    source_members=116,control_members=118,scientific_payload_members=0)

oldpackage=STAGE/'runtime_package_fix01'
oldcfg=oldpackage/cfg.name;oldmf=oldpackage/mf.name;oldtar=oldpackage/tar.name
assert sha(oldcfg)=='c0fe6408b53969da9f0bd70c01bf7649cfabfbf1864752dc5696bb8c9b6d5da9'
assert sha(oldmf)=='1488217be443fb49c60db4fcf3d852fcda8ba33507ce7e5ba2ee2f3d1e4d1327'
assert sha(oldtar)=='524ba471ac24d376211349280237e02562dfe797b2921fc25b01c46aac8e688e'
old=read(oldcfg);oldmanifest=read(oldmf)
assert oldcfg.read_bytes()==(HERE/'before_canonical_revision'/cfg.name).read_bytes()
assert oldmf.read_bytes()==(HERE/'before_canonical_revision'/mf.name).read_bytes()
assert set(m['files'])-set(oldmanifest['files'])=={'docs/WP3_016_RUNTIME_FIX02.json'}
assert not(set(oldmanifest['files'])-set(m['files']))
changed=[n for n in oldmanifest['files'] if oldmanifest['files'][n]!=m['files'][n]]
assert sorted(changed)==['scripts/run_wp3_016.py','scripts/wp3_016_controller.py']
assert len([n for n in oldmanifest['files'] if n not in changed])==113
prior=read(ROOT/'WP3_SOURCE_MANIFEST.json')
assert len(prior['files'])==105 and all(m['files'][n]==row for n,row in prior['files'].items())
assert sha(ROOT/'WP3_SOURCE_MANIFEST.json')==m['prior_WP3_manifest_sha256']
assert sha(ROOT/'wp3_015_config.json')==prior['config_sha256']
assert p['protection_files']==old['protection_files'] and len(p['protection_files'])==freeze['protected_source_records']==475
original=STAGE/'runtime_package_01'
assert sha(original/cfg.name)=='20884dc1a3916c2ac23820fa1ff7c47da21861a465a3f08fa5988cb71220281d'
assert sha(original/mf.name)=='9151ca1a9503cde85786c0623effe431e10fa706ec5b92895bc448714ed93789'
assert sha(original/tar.name)=='9d054aeb744cf35c0f50080ddebc0397574a66ac404dd2f68a5d4e9221cf6fb9'
check('original105_107controls_475protection_bindings_old_packages_and_failed_canonical_copies_preserved',
    original_sources=105,original_controls=107,protection_records=475,old_fix01_sources_unchanged=113,
    local_protection_binding_equal=True,remote_protection_reread=False)

assert p['inputs']==old['inputs'] and len(p['inputs'])==11
assert len(old['contracts'])==12 and len(p['contracts'])==13
for key,row in old['contracts'].items():
    expected=dict(row);assert expected['path'].startswith(oldroot+'/')
    expected['path']=newroot+expected['path'][len(oldroot):]
    assert p['contracts'][key]==expected
    local=Path(expected['local_path']);assert sha(local)==expected['sha256'] and local.stat().st_size==expected['bytes']
assert p['contracts']['tolerance_table']['sha256']=='f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b'
amendment_path=ROOT/'docs/WP3_016_RUNTIME_FIX02.json';amendment=read(amendment_path)
assert p['contracts']['runtime_numerical_storage_amendment']==dict(path=newroot+'/docs/WP3_016_RUNTIME_FIX02.json',
    local_path=str(amendment_path),sha256=sha(amendment_path),bytes=amendment_path.stat().st_size)
assert p['runtime_repair']==amendment
assert p['allowed_input_roots']==sorted(set(old['allowed_input_roots'])|{newroot})
check('eleven_input_bindings_unchanged_old12_contract_bytes_only_relocated_plus_one_reviewed_amendment',
    input_count=11,old_contracts=12,new_contracts=13,actual_input_payloads_read=0,original_T_unchanged=True)

allowed_changes={'launch_revision','previous_launch_configuration_sha256','remote_root','frozen_UTC','runtime_repair','source_files',
    'prior_failed_execution','previous_launch_fix_reason','contracts','launch_fix_reason','allowed_input_roots'}
changed_keys={k for k in set(old)|set(p) if old.get(k)!=p.get(k)}
assert changed_keys==allowed_changes
assert p['budget_caps']==review['budget_caps']==old['budget_caps']
assert p['candidate_trial_order']==['seed0_Q1','seed1_Q1'] and p['candidates']==['Q1']
assert p['WP3_reserved_short_trials_this_block']==2 and p['WP3_remaining_unreserved_maximum']==0
assert p['cumulative_training_before_run']==175667
assert p['launch_revision']=='fix02' and p['previous_launch_configuration_sha256']==sha(oldcfg)
assert p['previous_launch_fix_reason']==old['launch_fix_reason']
assert p['development_selection']==old['development_selection']
check('scientific_fields_seed_trials_selection_and_one_replacement_budget_unchanged',
    changed_top_level_fields=sorted(changed_keys),scientific_trial_slots=2,additional_candidate_slots=0,cumulative_G_before=175667)

failure=read(STAGE/'failure_collection_01/FAILURE_PRESERVATION_RECEIPT.json')
for name in ['engineering_ready','independent_review','native_failure','preserved_failure_backup','preserved_failure_local_commit']:
    pinned(amendment[name])
assert amendment['original_failure_difference_magnitude']=='unknown_not_saved'
assert amendment['baseline_repeat_gate']==dict(source='scripts/run_wp1_013.py:A_numerical_residual',max_abs_V=1e-7,
    same_frozen_A=True,actual_A384_and_Q_mu_per_draw_retained=True,diagnostics_written_before_gate=True,
    no_broadcast_or_mean_substitution=True,exceeds_gate='save_evidence_then_fail_without_training_retry')
assert amendment['charge_law_model_objective_optimizer_seed_inputs_observation_T_unchanged']
assert amendment['per_draw_theory']=='E[q_d]=mu_d; Var(q_d)=a*mu_d+b*mu_d**2'
assert amendment['sample_variance_expectation']=='mean(per_draw_theory_variance)+sample_variance(per_draw_mu)'
assert amendment['numerical_mu_spread_is_not_physical_randomness']
assert amendment['A384_is_frozen_average_response_only_not_full_stochastic_preADC_carrier']
assert amendment['prior_actual_cost']==failure['actual_completed_counters']
assert amendment['replacement_attempt_budget_caps']==p['budget_caps']
aggregate={k:v+amendment['prior_actual_cost'][k] for k,v in p['budget_caps'].items()}
assert aggregate==amendment['aggregate_budget_caps']==review['replacement_accounting']['failed_plus_planned_replacement_aggregate_caps']
assert aggregate['old_A_rows']==4346534 and aggregate['checkpoint_loads']==7 and aggregate['G_updates']==aggregate['new_AD_queries']==4000
assert amendment['resource_limits']==p['resource_limits']==old['resource_limits']
assert amendment['run_resource_limits']==p['run_resource_limits']==old['run_resource_limits']
assert amendment['aggregate_wall_cap_seconds']==p['resource_limits']['wall_seconds']+amendment['prior_native_elapsed_seconds']
assert amendment['original_remote_root']==oldroot and amendment['replacement_remote_root']==newroot
assert amendment['scientific_trial_slots']==2 and amendment['additional_candidate_slots']==0 and amendment['completed_prior_candidates']==0
assert amendment['maximum_replacement_native_attempts']==1 and amendment['automatic_restart'] is False
assert amendment['newly_started_operations_by_this_registration']==0
assert amendment['scientific_status']==p['scientific_status']=='response_match_not_established'
for key in ['frozen_configuration','frozen_manifest','failure','collection']:pinned(p['prior_failed_execution'][key])
preparation=read(STAGE/'CONTROL_PREPARATION_RECEIPT_FIX02.json')
assert preparation['prior_actual_cost']==amendment['prior_actual_cost'] and preparation['aggregate_budget_caps']==aggregate
assert preparation['new_scientific_operations']==preparation['new_candidate_trial_slots']==0
check('runtime_amendment_existing_A_gate_per_draw_statistics_failure_accounting_one_replacement_same_two_slots',
    A_gate_V=1e-7,ddof1_formula_bound_to_reviewed_source=True,prior_failure_G_AD=0,
    aggregate_A=4346534,aggregate_CP=7,aggregate_G_AD=4000,automatic_restart=False,original_amplitude_unknown=True)

result=dict(schema='n01/wp3-016-frozen-fix02-package-review/v1',UTC=datetime.now(timezone.utc).isoformat(),
    status='PASS_FIX02_FINAL_STATIC_FROZEN_BINDING',passed=True,blockers=[],sources=review['sources'],budget_caps=p['budget_caps'],
    **pins,controls={cfg.name:ref(ROOT/cfg.name),mf.name:ref(ROOT/mf.name),'source_package':ref(tar),
        'freeze_receipt':ref(PACKAGE/'FREEZE_RECEIPT.json'),'adoption_receipt':ref(STAGE/'FROZEN_CONTROL_ADOPTION_FIX02.json'),
        'preparation_receipt':ref(STAGE/'CONTROL_PREPARATION_RECEIPT_FIX02.json'),'runtime_amendment':ref(amendment_path)},
    source_review=ref(review_path),source_file_count=116,control_member_count=118,contract_count=13,input_count=11,protection_record_count=475,
    verified_package_inventory=inventory,checks=checks,
    runtime_repair=dict(same_two_scientific_slots=True,additional_candidate_slots=0,one_replacement_attempt=True,
        prior_failed_actual_cost=amendment['prior_actual_cost'],replacement_caps=p['budget_caps'],aggregate_caps=aggregate,
        old_and_new_remote_roots=[oldroot,newroot],existing_A_point_tolerance_V=1e-7,original_failure_repeat_magnitude='unknown_not_saved'),
    input_verification_scope='Exact frozen metadata equality to prior independently reviewed11 input bindings; no input payload read or decoded in this thin review.',
    protection_verification_scope='Exact475 inherited protection bindings and local public package controls checked; no fresh remote reads.',
    resource_boundary='Unchanged native trial/supervisor guards; previous output forecast remains an estimate, not a native measurement or worst-case guarantee.',
    review_costs=dict(new_scientific_operations=0,numerical_tests_rerun=0,array_or_checkpoint_payload_reads=0,SSH=0,source_config_or_main_metadata_changes=0),
    scientific_acceptance=False,weights_promoted=False,native_started_by_this_review=False)
path=HERE/'FROZEN_PACKAGE_REVIEW.json'
with path.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
assert read(path)['passed'] and not read(path)['blockers']
print(json.dumps(dict(passed=True,checks=len(checks),receipt=ref(path),**pins,source_count=116,controls=118,contracts=13,inputs=11),indent=2))
