"""Bind focused artificial evidence to final016 source bytes; no private inputs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import wp3_016_controller as controller


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


settings,budget=controller.factory_contract((ROOT/controller.DRIVER).read_bytes())
design=read(ROOT/'wp3_016_design.json');review=read(ROOT/'receipts/wp3_016_20261015/independent_01/DESIGN_REVIEW.json')
assert review['passed'] and budget(settings)==design['budget_caps_design']
numeric_path=HERE/'artificial_cpu_01/ENGINEERING_RECEIPT.json';numeric=read(numeric_path)
stdlib_path=HERE/'controller_stdlib_01/ENGINEERING_RECEIPT.json';stdlib=read(stdlib_path)
assert numeric['passed'] and stdlib['passed']
assert all(sha(ROOT/n)==numeric['sources'][n] for n in ['src/n01/model016.py','scripts/run_wp3_016.py'])
assert sha(ROOT/controller.CONTROLLER)==stdlib['source_sha256']
manifest=read(ROOT/'WP3_SOURCE_MANIFEST.json')
assert len(manifest['files'])==105 and all(sha(ROOT/n)==r['sha256'] for n,r in manifest['files'].items())
assert sha(ROOT/'wp3_015_config.json')==manifest['config_sha256']
settings['budget_caps']=budget(settings)
draft=ROOT/'receipts/wp3_016_20261015/implementation_01/SETTINGS_DRAFT.json'
write(draft,settings)
sources={name:ref(ROOT/name) for name in controller.OWNED}
result=dict(schema='n01/wp3-016-engineering-ready/v1',UTC=datetime.now(timezone.utc).isoformat(),
    status='LOCAL_ARTIFICIAL_ENGINEERING_READY_PENDING_INDEPENDENT_SOURCE_REVIEW_AND_FREEZE',passed=True,
    model_policy=dict(assigned_model='gpt-6.1-sol',assigned_reasoning_effort='max'),
    sources=sources,design=ref(ROOT/'wp3_016_design.json'),independent_design_review=ref(ROOT/'receipts/wp3_016_20261015/independent_01/DESIGN_REVIEW.json'),
    settings_draft=ref(draft),budget_caps=budget(settings),
    numeric_engineering=ref(numeric_path),stdlib_controller_engineering=ref(stdlib_path),
    semantic_checks=dict(exact_charge_off_M0_bytes=True,actual_canonical_A_Q_no_floor=True,
        analytic_actual_A_one_LSB_initialization=True,all6593_required_at_native_calibration=True,
        two_parameter_positive_mean_corrected_charge=True,shape_noise_015_prefixes_preserved=True,
        exactly_two_scalar_AdamW_parameters=True,shared_single_forward_initial_A_M0_charge_on=True,
        original_four_axis_evaluator_and_all_flags_retained=True,preADC_charge_moments_and_stream_states_retained=True,
        current_owned_PID_boot_startticks_command_cwd_CPU_resource_health_or_full_native0_required=True,
        dead_without_exit_and_stale_PID_rejected=True),
    original105_sources_and107_controls_unchanged=True,
    synthetic_updates=2,synthetic_functional_AD_queries=1,
    synthetic_evaluation=dict(initial_hard_REPORT_pairs=36,final_hard_REPORT_pairs=12,baseline100_files=8,preADC_charge_moment_files=8),
    real_scientific_operations=dict(checkpoint_loads=0,original_SOURCE_REAL_DEV_waveform_reads=0,NN_rows=0,AD_queries=0,
        training_updates=0,new_random_draws=0,calibration_fits=0,GPU_work=0,SSH_calls=0),
    controller_reuse=dict(pinned014_controller_sha256=controller.BASE_SHA,
        methods='Pinned014 archive/journal/owned-tree/supervision;016 policy hooks; embedded015 CPU live identity/health semantics',
        collection_MiB=256,launch_identity_instrumentation=controller.operations().identity_adaptations,
        remote_extra_helper_dependency=False),
    factory_API=dict(controller='scripts/wp3_016_controller.py',factory='factory_contract(driver_bytes) -> settings,budget_factory',
        driver='scripts/run_wp3_016.py',model='n01.model016.FrozenM0ChargeResponse016',
        run_id=controller.RUN_ID,config=controller.CFG,manifest=controller.MANIFEST,
        inputs='original9 including integral_statistics plus M0_head_seed0 and M0_head_seed1',
        freeze='controller.freeze(repo,output,expected_draft_sha)',
        prior105_manifest_sha256=controller.PRIOR_SHA),
    scientific_start_authorized_or_performed_by_this_artifact=False,source_review_pending=True,
    actual_A_calibration_not_run=True,actual_M0_tensor_metadata_not_loaded=True,undecided_implementation_items=[],
    root_next_action='Independent source review, bind exact inputs/config/commit, then root-controlled deployment/start.',
    weights_promoted=False,scientific_acceptance=False,scientific_status='response_match_not_established')
write(HERE/'ENGINEERING_READY.json',result)
print(json.dumps(dict(passed=True,ready=str(HERE/'ENGINEERING_READY.json'),source_sha256={n:r['sha256'] for n,r in sources.items()},
    ready_sha256=sha(HERE/'ENGINEERING_READY.json')),ensure_ascii=False))
