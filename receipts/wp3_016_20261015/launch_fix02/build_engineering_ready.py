"""Local metadata/source audit only. No private-array decode or remote calls."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import difflib
import hashlib
import json
import re
import struct
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
sys.path.insert(0,str(ROOT/'scripts'))
import wp3_016_controller as c

read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

numeric=read(HERE/'artificial_cpu_01/ENGINEERING_RECEIPT.json');stdlib=read(HERE/'artificial_stdlib_01/ENGINEERING_RECEIPT.json')
assert numeric['passed'] and stdlib['passed']
sources={n:dict(sha256=sha(ROOT/n),bytes=(ROOT/n).stat().st_size) for n in c.OWNED}
assert sources==numeric['sources'] and stdlib['source_sha256']==sources[c.CONTROLLER]['sha256']
assert sources['src/n01/model016.py']['sha256']=='04c63f91ea5405937a940a9aee6eaed876db79149c857bfbe6ed8cfe6dcb138b'
before={n:ref(HERE/'before'/Path(n).name) for n in (c.DRIVER,c.CONTROLLER)}
assert before[c.DRIVER]['sha256']=='4729208d66c61c3b082827535719e72d8d7bb2ee60f390b688111ae3b4f12972'
assert before[c.CONTROLLER]['sha256']=='ae9afe444a90acb4fabddf1d2bf643550a4cbf29c4b9130408b040abcc843b5b'
save(HERE/'before/SNAPSHOT_SHA256.json',dict(schema='n01/wp3-016-fix02-before/v1',files=before,model016_unchanged=sources['src/n01/model016.py']))
delta=''
for name in (c.DRIVER,c.CONTROLLER):
    old=(HERE/'before'/Path(name).name).read_text('utf8').splitlines(keepends=True);new=(ROOT/name).read_text('utf8').splitlines(keepends=True)
    delta+=''.join(difflib.unified_diff(old,new,fromfile='before/'+name,tofile='after/'+name))
with (HERE/'SOURCE_DIFF.patch').open('x',encoding='utf8') as f:f.write(delta)
oldp,oldbudget=c.factory_contract((HERE/'before/run_wp3_016.py').read_bytes());p,budget=c.factory_contract((ROOT/c.DRIVER).read_bytes())
assert oldp==p and oldbudget(oldp)==budget(p)==read(ROOT/'wp3_016_design.json')['budget_caps_design']
manifest=read(ROOT/'WP3_SOURCE_MANIFEST.json');assert sha(ROOT/'WP3_SOURCE_MANIFEST.json')==c.PRIOR_SHA
assert len(manifest['files'])==105 and all(sha(ROOT/n)==row['sha256'] for n,row in manifest['files'].items())
assert sha(ROOT/'wp3_015_config.json')==manifest['config_sha256']
fix01=STAGE/'runtime_package_fix01'
oldpins={k:ref(fix01/n) for k,n in [('config','wp3_016_config.json'),('manifest','WP3_016_SOURCE_MANIFEST.json'),('package','wp3_016_source.tar.gz')]}
assert oldpins['config']['sha256']=='c0fe6408b53969da9f0bd70c01bf7649cfabfbf1864752dc5696bb8c9b6d5da9'
assert oldpins['manifest']['sha256']=='1488217be443fb49c60db4fcf3d852fcda8ba33507ce7e5ba2ee2f3d1e4d1327'
assert oldpins['package']['sha256']=='524ba471ac24d376211349280237e02562dfe797b2921fc25b01c46aac8e688e'

# Read only the NPY headers of artificial NPZ members. Upper bound new charge
# payload with no compression; RNG arrays stay fixed, parent/draw arrays scale.
npz_files=sorted((HERE/'artificial_cpu_01/ARTIFICIAL_BATCH').glob('seed*/*_evaluation/*_CHARGE_PREADC_MOMENTS.npz'))
assert len(npz_files)==8;projected_payload=0;actual_zip_bytes=0;actual_raw_bytes=0
for path in npz_files:
    actual_zip_bytes+=path.stat().st_size
    with zipfile.ZipFile(path) as z:
        for member in z.infolist():
            with z.open(member) as f:
                assert f.read(6)==b'\x93NUMPY';major,minor=f.read(2)
                size=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0]
                header=ast.literal_eval(f.read(size).decode('latin1'))
            shape=header['shape'];raw=member.file_size-(10 if major==1 else 12)-size;actual_raw_bytes+=raw
            factor=6593/8 if shape and shape[0] in (8,32) else 1
            projected_payload+=int(raw*factor)+(10 if major==1 else 12)+size+1024
projected_payload=int(projected_payload*1.002) # Conservative DEFLATE expansion allowance, plus member overhead above.
previous=ROOT/'receipts/wp3_015_20261015/collection_01/REMOTE_FULL_INVENTORY.json'
inv=read(previous)['whole_remote_inventory']
historical={suffix:dict(count=sum(k.endswith(suffix) for k in inv),bytes=sum(v['bytes'] for k,v in inv.items() if k.endswith(suffix)),
    maximum_file_bytes=max(v['bytes'] for k,v in inv.items() if k.endswith(suffix))) for suffix in ['_HARD_ROWS.npz','_REPORT.npz','_BASELINE100.npz']}
old_path_forecast=48*(historical['_HARD_ROWS.npz']['maximum_file_bytes']+historical['_REPORT.npz']['maximum_file_bytes'])+8*historical['_BASELINE100.npz']['maximum_file_bytes']
reserved=128*1024**2;forecast=old_path_forecast+projected_payload+reserved
output=dict(new_actual_A_float64_bytes_per_bank=6593*4*384*8,total8banks_A_payload_bytes=6593*4*384*8*8,
    artificial8bank_npz_actual_bytes=actual_zip_bytes,artificial8bank_payload_bytes=actual_raw_bytes,
    projected8bank_charge_payload_and_container_no_compression_bytes=projected_payload,
    historical_same_population_NPZ_inventory=ref(previous),historical=historical,
    existing48hard48report8baseline_empirical_forecast_bytes=old_path_forecast,other_output_reserve_bytes=reserved,
    projected_batch_output_bytes=forecast,projected_trial_average_bytes=forecast/2,
    batch_cap_bytes=p['resource_limits']['output_MiB']*1024**2,trial_cap_bytes=p['run_resource_limits']['output_MiB']*1024**2,
    within_batch_cap_estimate=forecast<p['resource_limits']['output_MiB']*1024**2,
    within_trial_average_cap_estimate=forecast/2<p['run_resource_limits']['output_MiB']*1024**2,
    forecast_is_not_measured_native_or_a_worst_case_compression_guarantee=True,existing_caps_unchanged_and_guard_after_bank_save=True,
    additional_peak_CPU_memory_estimate_upper_bytes=512*1024**2,CPU_memory_method='two A copies during concatenate ~162MB; block diagnostics/input hashes and compression reserve within512MiB; actual bank A/moment refs deleted before next bank',
    extra_GPU_forward_activation_rows=0,full_stochastic_preADC_carrier_saved=False)
assert output['within_batch_cap_estimate'] and output['within_trial_average_cap_estimate']

preservation=STAGE/'failure_collection_01/FAILURE_PRESERVATION_RECEIPT.json';failed=read(preservation)
calpath=ROOT/'runs/wp3_016_20261015/failed_collected_01/runs/primary/artifacts/CHARGE_INITIALIZATION.json';cal=read(calpath)
diagnosis=dict(schema='n01/wp3-016-failure-diagnosis-engineering/v1',UTC=datetime.now(timezone.utc).isoformat(),
    status='CONFIRMED_REDUCER_INVARIANT_FAILURE_ACTUAL_DIFFERENCE_MAGNITUDE_UNKNOWN',
    conclusion_zh='真实失败位于初始bank0结束后的严格Q相等reducer；实际差值未保存，不能把GPU/cuDNN舍入说成已证实根因。最小修复保留实际逐draw A/Q/v，先保存再按013原A数值门判定，并修正异均值draw的统计汇总。',
    actual_failure=ref(preservation),confirmed_error=failed['failure'],actual_completed_counters=failed['actual_completed_counters'],
    actual_native_costs_not_all_zero=True,actual_G_AD_updates=0,
    ordering=dict(parents=6593,draws=4,rows=26372,layout='repeat(arange(6593),4); tile(arange(4),6593); concatenate chunks then reshape(6593,4)',
        chunk_rows=128,full_chunks=206,last_chunk_rows=4,parent_crosses_chunk=False,static_order_correct=True,
        actual_original_failure_input_row_hashes_saved=False,new_original_and_forward_input_byte_evidence_added=True),
    calibration=dict(source=ref(calpath),parents=cal['parents'],chunk_rows=256,all_Q_positive=cal['all_Q_mu_finite_strictly_positive'],
        Q_min_Vns=cal['Q_mu_min_Vns'],Q_max_Vns=cal['Q_mu_max_Vns'],expected_RMS_V=cal['actual_A_expected_increment_RMS_V'],target_RMS_V=cal['target_RMS_V'],
        comparison_at_old_line312='within one evaluation bank, same parent across draws; not calibration versus evaluation',new_calibration_or_finite_MC_retuning=False),
    dtype=dict(on_disk_original_source_dtype_in_failed_run='unknown_not_in_collected_JSON_not_decoded',evaluation_and_calibration_forward_input='explicit torch.float64',
        canonical_sum='actual A to float64, indices110:366, software1ns',model_feature_path='model008._features casts to encoder weight dtype; default neural parameters instantiated float32; FP64 input does not make the entire network FP64',
        checkpoint_tensor_dtypes_not_reloaded_for_this_diagnosis=True),
    mechanism=dict(A_path='frozen original A; _baseline_once uses zero phys/local latents; no q latent in A; old shape head uses captured decoded output after A',
        deterministic_controls='use_deterministic_algorithms(True), cudnn.benchmark=False, TF32 matmul/cudnn disabled',
        network_normalization='GroupNorm found; no batch-statistics normalization introduced by016',
        GPU_or_cuDNN_rounding='plausible mechanism, not established by saved evidence; no original actual A/mu matrix to locate stage or magnitude',
        mapping_or_material_A_change='not observed in source ordering; actual byte/point evidence unavailable in failed run; future explicit gates refuse mismatch'),
    fixed_numerical_contract=dict(original_helper=ref(ROOT/'scripts/run_wp1_013.py'),original_config=ref(ROOT/'wp1_013_config.json'),
        original_gate_V=1e-7,canonical_window_samples=256,derived_repeat_Q_component_Vns=2.56e-5,
        GPU_CPU_sum_bound='2*gamma255/(1-gamma255)*L1(A)+512*smallest_subnormal, rounded upward; u=2^-53, gamma255=255*u/(1-255*u)',
        assumptions='same finite actual float64 A, IEEE754 nearest additions, <=255 additions per256-term sum, no overflow; tiny subnormal allowance',
        bound_is_predeclared_not_fit_to_new_results=True,no_independent_charge_allclose=True,no_A_Q_broadcast_or_replacement=True,
        actual_arrays_and_diagnostics_before_A_gate=True,actual_mu_and_v_per_draw=True),
    reducer_formulas=dict(v_d='a*mu_d+b*mu_d^2',report_center='mean(mu_d)',expected_sample_variance='mean(v_d)+samplevar(mu_d,ddof1)',
        theoretical_sample_mean_MCSE='sqrt(sum(v_d))/draws',sample_mean_MCSE='sqrt(samplevar(q,ddof1)/draws)',
        centered_residual='r_d=q_d-mu_d',centered_second_moment='mean(r_d^2), denominator draws; expectation mean(v_d)',
        centered_sample_variance='samplevar(r_d,ddof1); expectation mean(v_d)',numerical_mu_spread_is_physical_variance=False,
        individual_model_denominators_unchanged=True,legacy_shapes_retained_with_explicit_semantics=True),
    documentation=[dict(url='https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html',
        use='Equivalent floating-point or batched/sliced computations need not be bitwise identical; supports rejecting an unconditional equality assumption, not attribution of this failure.'),
        dict(url='https://docs.pytorch.org/docs/2.14/notes/randomness.html',use='Deterministic algorithms address repeat execution choices; source flags do not supply missing measurements of this failure.')],
    independent_diagnosis=ref(HERE/'FAILURE_DIAGNOSIS_REVIEW.json'),artificial_engineering=ref(HERE/'artificial_cpu_01/ENGINEERING_RECEIPT.json'),
    original_failure_magnitude_recovered=False,original_raw_arrays_or_private_models_decoded=0,SSH_calls_after_collection=0,
    source_fix_authorized_by_root=True,real_retry_authorized_by_this_artifact=False,output_memory_estimate=output)
save(HERE/'FAILURE_DIAGNOSIS_ENGINEERING.json',diagnosis)
ready=dict(schema='n01/wp3-016-engineering-fix-ready/v1',UTC=datetime.now(timezone.utc).isoformat(),status='LOCAL_FIX02_READY_FOR_INDEPENDENT_SOURCE_REVIEW',passed=True,
    sources=sources,before=before,source_diff=ref(HERE/'SOURCE_DIFF.patch'),diagnosis=ref(HERE/'FAILURE_DIAGNOSIS_ENGINEERING.json'),
    numerical_receipt=ref(HERE/'artificial_cpu_01/ENGINEERING_RECEIPT.json'),stdlib_receipt=ref(HERE/'artificial_stdlib_01/ENGINEERING_RECEIPT.json'),
    original105_sources_and107_controls_unchanged=True,original_fix01_package_unchanged=oldpins,
    model016_exact_bytes_unchanged=True,all_old_factories_scientific_settings_and_budget_equal=True,settings_factory=p,budget_caps=budget(p),
    remote_root=c.REMOTE,run_id=c.RUN_ID,trial_names=list(c.TRIALS),controller_only_change='REMOTE constant',
    driver_scope='evaluation evidence saving, frozen013 replay gate, actual-row reducer semantics; no scientific forward/train/calibration/law changes',
    extra_imports_already_in_old105=['scripts/run_wp1_013.py'],new_external_helper_files=[],
    tests=dict(numeric_groups=len(numeric['checks']),numeric_negative_cases=numeric['negative_cases'],native_numeric_seconds=numeric['elapsed_seconds'],
        two_synthetic_trial_updates=2,stdlib_groups=len(stdlib['checks']),stdlib_live_health_negative_cases=15,private_CP_NN_AD_random_training=0,SSH=0,
        artificial_costs=numeric['checks'][-1]['counters'],artificial_initialization_and_RNG_counts_are_not_scientific_costs=True),
    resource_estimate=output,unresolved_engineering_items=[],original_actual_repeat_amplitude='unknown; preserved old failure has no A/mu arrays',
    root_next_action='Independent final source review; register additional old native costs, freeze/commit/backup a new package and explicitly authorize one new remote execution if approved.',
    main_metadata_changed_by_engineer=False,refreeze_or_SSH_or_real_retry_performed=False,scientific_acceptance=False,weights_promoted=False)
save(HERE/'ENGINEERING_FIX_READY.json',ready)
print(json.dumps(dict(status=ready['status'],ready_sha256=sha(HERE/'ENGINEERING_FIX_READY.json'),diagnosis_sha256=sha(HERE/'FAILURE_DIAGNOSIS_ENGINEERING.json'),
    sources=sources,output_estimate=output['projected_batch_output_bytes'],output_cap=output['batch_cap_bytes']),ensure_ascii=False))
