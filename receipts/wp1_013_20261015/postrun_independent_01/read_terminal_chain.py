"""Independent local JSON/source/Git terminal-chain audit, before array collection."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('E:/GAN/research/N01')
OUT=ROOT/'receipts/wp1_013_20261015/postrun_independent_01'
BASE=ROOT/'receipts/wp1_013_20261015/native_repeat_fix01'
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
checks=[]
def ok(v,name,detail=None):
    assert v,name
    checks.append(dict(name=name,passed=True,detail=detail))

cfg=read(ROOT/'wp1_013_config.json');mf=read(ROOT/'WP1_SOURCE_MANIFEST.json')
state_path=BASE/'wake_status_01/REMOTE_STATE.json';s=read(state_path)
freeze_path=BASE/'runtime_package_03/FREEZE_RECEIPT.json';fr=read(freeze_path)
reg=read(BASE/'NUMERICAL_STORAGE_REPAIR_REGISTRATION.json')
start=s['START_RECEIPT.json'];launch=s['runs/primary/supervisor/launch.json']
proof=s['runs/primary/artifacts/FIRST_WP1_PROOF.json'];status=s['runs/primary/artifacts/status.json']
native=s['runs/primary/supervisor/exit_receipt.json'];c=s['runs/primary/artifacts/COMPLETE.json']
durable=s['durable_child_progress']
ok(cfg['status']=='frozen' and sha(ROOT/'wp1_013_config.json')==fr['config_sha256']==mf['config_sha256'], 'frozen_config_SHA')
ok(sha(ROOT/'WP1_SOURCE_MANIFEST.json')==fr['source_manifest_sha256'], 'source_manifest_SHA')
ok(len(mf['files'])==len(cfg['source_files'])==fr['source_file_count']==84 and fr['private_file_count']==4,
    'source84_private4_inventory_metadata')
source_pins={name:pin(ROOT/name) for name in mf['files']}
ok(all(r['sha256']==mf['files'][n]['sha256']==cfg['source_files'][n]['sha256']
    and r['bytes']==mf['files'][n]['bytes'] for n,r in source_pins.items()), 'all84_current_local_source_bytes')
protections=cfg['protection_files']
ok(len(protections)==fr['protected_source_records']==185, 'old_source_protection_record_count185')
ok(all(sha(Path(r['local_path']))==r['sha256'] for r in protections.values()), 'all185_old_local_source_protection_SHA')
ok(len(cfg['inputs'])==22 and len(cfg['contracts'])==3, '22_immutable_inputs_three_contract_records')
expected={cfg['remote_root']+'/wp1_013_config.json':sha(ROOT/'wp1_013_config.json')}
for records in (cfg['source_files'],cfg['inputs'],cfg['contracts'],protections):
    for r in records.values():
        ok(r['path'] not in expected or expected[r['path']]==r['sha256'], 'no_conflicting_runtime_SHA_alias_'+r['path'])
        expected[r['path']]=r['sha256']
ok(not s['protection_errors'] and all(s['_verified_freeze']['immutable_files_sha256'].get(k)==v for k,v in expected.items()),
    'native_end_readonly_gate_matches_registered_all_runtime_paths')
ok(all(proof['immutable_files_sha256'].get(k)==v and c['immutable_files_sha256'].get(k)==v for k,v in expected.items()),
    'first_and_complete_immutable_SHA_maps_include_all_bound_paths')
ok(all(start[k]==fr[k] for k in ('config_sha256','source_manifest_sha256','controller_sha256','source_package_sha256','private_package_sha256')),
    'start_receipt_bound_to_actual_freeze')
ok(start['supervisor_pid']==launch['supervisor_pid']==native['supervisor_pid']==57507
    and launch['child_pid']==native['child_pid']==proof['pid']==status['pid']==c['pid']==57514,
    'consistent_supervisor_child_proof_status_complete_PID_chain')
ok(start['boot_id']==s['current_boot_id'] and proof['first_family_rows']==128 and proof['reference_head_G_updates']==2000
    and proof['first_family_completed'] and proof['actual_science'] and proof['extra_pilot_rows']==0,
    'same_boot_actual128_family_G2000_not_extra_pilot')
ok(native['child_native_exit_code']==native['exit_code']==0 and native['reason'] is None and native['error'] is None
    and s['runs/primary/artifacts/FAILED.json'] is None, 'native_child_supervisor_zero_and_no_FAILED')
ok(status['phase']==c['phase']=='complete' and c['engineering_checks_passed'] and c['inputs_unchanged']
    and c['frozen_state_unchanged'] and not c['artificial_only'], 'actual_producer_complete_engineering_flags')
ok(c['probe_defined_states']==c['probe_registered_states']==30 and c['calibration_cases']==21
    and c['all_SOURCE_TRAIN_parents']==6593 and c['all_REAL_TRAIN_rows']==21047 and c['banks']==2,
    'all_registered_populations_states_and_calibration_cases')
caps=cfg['budget_caps']
ok(c['counters']==c['attempted_counters']==status['counters']==caps, 'completed_attempted_and_status_equal_per_attempt_caps')
ok(c['operations_entered']==c['operations_completed']==29201 and not c['inflight']
    and not durable['unknown_inflight_ids'] and not durable['failed_ids'] and not durable['truncated_tail'],
    '29201_durable_operations_completed_no_unknown_or_failed')
ok(all(durable['entered_counts'].get(k,0)==durable['completed_counts'].get(k,0)==native['completed_counts'].get(k,0)==v
    for k,v in caps.items()), 'native_and_durable_entered_completed_charge_maps_equal')
formula=dict(old_A_rows=52744+8192*30,head_actual_parameter_forward_rows=52744+8192*30,
    head_linear_direction_map_rows=8192*5,noise005_rows=52744,hard_ADC_rows=832560+24576*30,
    all_path_gather_rows=832560+73728*30,checkpoint_loads=3)
formula['head_total_kernel_rows']=formula['head_actual_parameter_forward_rows']+formula['head_linear_direction_map_rows']
ok(all(caps[k]==v for k,v in formula.items()), 'full_original_reference_parameter_calibration_budget_arithmetic')
failures=[read(ROOT/f'runs/wp1_013_20261015/failed_attempt_0{i}/runs/primary/artifacts/FAILED.json') for i in (1,2)]
prior={k:sum(f['counters'][k] for f in failures) for k in caps}
total={k:prior[k]+caps[k] for k in caps}
ok(prior==reg['prior_cost']==cfg['engineering_repair']['previous_cost']
    and total==reg['aggregate_budget_caps']==cfg['engineering_repair']['aggregate_budget_caps'], 'two_failed_attempts_charged_exact_cumulative_maximum')
zero=('optimizer_created','optimizer_steps','G_updates','D_updates','noise_updates','population_updates','C2ST_updates','new_AD_queries','DEV_model_rows')
ok(all(total[k]==0 for k in zero) and c['actual_training_updates']==0 and c['formal_response_acceptance'] is False,
    'zero_AD_updates_optimizers_DEVmodel_scientific_acceptance_false')
resources=cfg['resource_limits']
wall=(datetime.fromisoformat(native['UTC'])-datetime.fromisoformat(start['UTC'])).total_seconds()
resource_readback=dict(sampled_peak_RSS_bytes=native['peak_RSS_bytes'],sampled_peak_GPU_bytes=native['peak_GPU_bytes'],
    peak_output_bytes=native['peak_output_bytes'],native_start_to_exit_seconds=wall,
    RSS_limit_bytes=resources['job_tree_RSS_GiB']*1024**3,GPU_limit_bytes=resources['GPU_GiB']*1024**3,
    output_limit_bytes=resources['output_MiB']*1024**2,wall_limit_seconds=resources['wall_seconds'])
ok(resource_readback['sampled_peak_RSS_bytes']<=resource_readback['RSS_limit_bytes']
    and resource_readback['sampled_peak_GPU_bytes']<=resource_readback['GPU_limit_bytes']
    and resource_readback['peak_output_bytes']<=resource_readback['output_limit_bytes'] and wall<=resources['wall_seconds'],
    'recorded_sampled_resource_peaks_and_final_output_wall_within_caps',resource_readback)
ok(native['peak_output_bytes']>1024*1024**2 and resources['output_MiB']==1536, 'actual_output_exceeds_original1024_and_fits_amended1536')

commit_receipt_path=BASE/'SOURCE_COMMIT_READBACK.json';cr=read(commit_receipt_path)
commit=cr['commit'];files={name:r['sha256'] for name,r in source_pins.items()}
files.update({'wp1_013_config.json':sha(ROOT/'wp1_013_config.json'),'WP1_SOURCE_MANIFEST.json':sha(ROOT/'WP1_SOURCE_MANIFEST.json')})
ok(len(files)==cr['frozen_source_and_control_files']==86 and files==cr['exact_working_and_Git_blob_sha256'], 'recorded_source_commit86_exact_inventory')
names=sorted(files)
raw=subprocess.check_output(['git','cat-file','--batch'],input=''.join(f'{commit}:{name}\n' for name in names).encode('utf8'),cwd=ROOT)
cursor=0
for name in names:
    end=raw.index(b'\n',cursor);header=raw[cursor:end].decode('ascii').split();cursor=end+1
    assert header[1]=='blob';size=int(header[2]);body=raw[cursor:cursor+size];cursor+=size
    assert raw[cursor:cursor+1]==b'\n';cursor+=1
    assert hashlib.sha256(body).hexdigest()==files[name],name
ok(cursor==len(raw),'independent86_actual_Git_blobs_match_source_commit_'+commit)
OUT.mkdir(parents=True,exist_ok=True)
receipt=dict(schema='n01/wp1-independent-postrun-terminal-chain/v1',status='PASS_METADATA_ARRAY_REVIEW_PENDING',passed=True,
    UTC=datetime.now(timezone.utc).isoformat(),run_id=cfg['run_id'],snapshot=pin(state_path),freeze=pin(freeze_path),
    configuration=pin(ROOT/'wp1_013_config.json'),source_manifest=pin(ROOT/'WP1_SOURCE_MANIFEST.json'),source_commit=commit,
    source_commit_readback=pin(commit_receipt_path),checks=checks,
    PID_chain=dict(supervisor=57507,child=57514,boot=start['boot_id']),
    current_completed_cost=caps,two_failed_prior_cost=prior,cumulative_actual_cost=total,resources=resource_readback,
    method_choice_native=c['method_choice'],method_choice_independently_recomputed='pending_collection_complete',
    array_requirements=['CALIBRATION.json','CALIBRATED_TOLERANCES.json','PARAMETER_VALIDATION.json','METHOD_SELECTION.json',
        'A_NUMERIC_REPEATS_0.json','A_NUMERIC_REPEATS_1.json','A_NUMERIC_CROSS_BANK.json','REFERENCE_0.npz','REFERENCE_1.npz','REFERENCE_ENCODING.json',
        'SIGNAL_MOMENTS_*.npz','BANK_*.npz','FULL_*.json/npz','REPEATS_8_*.json/npz','operations.jsonl'],
    recomputation_scope='T-table max7case(mean2bank) and method selection algebra from saved metrics, actualA/raw-code decoding and stored moment/covariance checks. No full CDF rerun is planned; calibration raw waveforms were not retained, so native component self-distances cannot independently be replayed from calibration Q/tail alone.',
    limits=['COMPLETE/native0 and adequate are engineering/local finite-direction evidence, not formal response-match acceptance.',
        'RSS/GPU peak fields are sampled at the registered polling interval; they are not continuous instrumentation.',
        'No array collection examined before root collection-complete notification. Original dataset/CP/DEV waveform inputs are not opened.',
        'Native input byte hashes were compared to frozen metadata maps; this reviewer did not load or independently decode the scientific checkpoint/data inputs.'],
    reviewer_scope=dict(local_JSON_source_Git_only=True,scientific_checkpoint_loads=0,scientific_array_decodes=0,
        NN_rows=0,AD_queries=0,training_updates=0,DEV_waveform_reads=0,SSH_calls=0,new_jobs_started=0))
target=OUT/'TERMINAL_CHAIN_READBACK.json'
with target.open('x',encoding='utf8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
assert read(target)['passed']
print(json.dumps(dict(status=receipt['status'],receipt=str(target),sha256=sha(target),checks=len(checks),elapsed_seconds=wall,NN_rows=0)))
