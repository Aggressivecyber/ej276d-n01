"""Close016 from local returned JSON/source/journal only; no raw binary reads."""
from pathlib import Path
from datetime import datetime,timezone
import copy
import hashlib
import json
import math
import tarfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
CPU=ROOT/'runs/wp3_016_20261015/readback_collected_01';ART=ROOT/'runs/wp3_016_20261015/selected_collected_01/runs/primary/artifacts'
read=lambda p:json.loads(p.read_text('utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
def save(name,obj):
 p=HERE/name
 with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 assert read(p)==obj;return ref(p)
def check(name,**details):checks.append(dict(name=name,passed=True,**details))

checks=[];plan=read(HERE/'READBACK_PLAN.json');provisional=read(HERE/'PROVISIONAL_DEVELOPMENT_SELECTION.json')
prelaunch=read(HERE/'PRELAUNCH_REVIEW.json');assert prelaunch['passed'] and not prelaunch['blockers']
collection=read(STAGE/'readback_collection_01/COLLECTION_RECEIPT.json');archive=read(STAGE/'readback_collection_01/REMOTE_ARCHIVE.json')
assert collection['files']==len(archive['inventory'])==19 and not collection['partial_evidence_only'] and not archive['partial_evidence_only']
assert collection['original_raw_NPZ_transferred'] is False and collection['job']['completed_pairs']==24
assert collection['archive_sha256']==archive['archive_sha256']=='765db742ccd336b533975eb3057a21ff8cc0304acd976b5e6aadefddd5aefdd2'
assert {p.relative_to(CPU).as_posix() for p in CPU.rglob('*') if p.is_file()}==set(archive['inventory'])
for name,row in archive['inventory'].items():
 p=CPU/name;assert p.suffix in ('.json','.jsonl','.log','.py') and sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
assert sum(v['bytes'] for v in archive['inventory'].values())==470489
check('all19_collected_public_sidecar_files_bytes_SHA_and_inventory_verified',files=19,bytes=470489,archive_SHA=archive['archive_sha256'])

registration=read(CPU/'READBACK_EXECUTION.json');registration_sha=sha(CPU/'READBACK_EXECUTION.json')
aliases={'readback_saved_arrays.py':HERE/'readback_saved_arrays.py','saved_array_kernels.py':HERE/'saved_array_kernels.py',
 'readback_control.py':STAGE/'readback_engineering_01/readback_control.py','READBACK_PLAN.json':HERE/'READBACK_PLAN.json',
 'BOUND_ORIGINAL_INVENTORY.json':STAGE/'collection_01/REMOTE_FULL_INVENTORY.json'}
assert set(registration['files'])==set(aliases) and registration['expected_operations']==200
for name,p in aliases.items():
 assert (CPU/name).read_bytes()==p.read_bytes()
 assert {k:registration['files'][name][k] for k in ('bytes','sha256')}==prelaunch['sources'][p.relative_to(ROOT).as_posix()]
assert registration['prelaunch_independent_review']['sha256']==sha(HERE/'PRELAUNCH_REVIEW.json')
assert registration['scientific_budget']==plan['scientific_budget'] and all(v==0 for v in registration['scientific_budget'].values())
check('registered_five_sources_plus_execution_exactly_match_passed_prelaunch_plan_and_inventory',registration_sha256=registration_sha,controls=6)

array_path=CPU/'artifacts/ARRAY_READBACK.json';array=read(array_path);done=read(CPU/'artifacts/COMPLETE.json');native=read(CPU/'supervisor/exit_receipt.json')
first=read(CPU/'artifacts/FIRST_PAIR_PROOF.json');start=read(CPU/'START.json');launch=read(CPU/'supervisor/launch.json')
reconcile=read(STAGE/'readback_registration_01/CPU_COMPLETION_RECONCILIATION.json');snapshot=read(STAGE/'readback_start_verification_01/REMOTE_STATE.json')
assert sha(array_path)=='7337c5ab50929eb4381fc29ded4af138420b8257ba0c7b9945a7c82ca6be604a'
assert array['passed'] and array['status']=='PASS' and not array['artificial_only'] and not array['inflight']
assert done['native_status']=='COMPLETE' and done['passed'] and done['ARRAY_READBACK_sha256']==sha(array_path)
assert array['counters']==array['attempted_counters']==done['counters']==plan['expected_complete_counters']
assert array['operations_entered']==array['operations_completed']==done['operations_entered']==done['operations_completed']==200
assert native['exit_code']==native['child_native_exit_code']==0 and native['error'] is None and native['reason'] is None
assert start['registration_sha256']==registration_sha and first['pid']==launch['child_pid']==native['child_pid']
assert start['supervisor_pid']==launch['supervisor_pid']==native['supervisor_pid'] and start['boot_id']==launch['boot_id']
assert first['actual_saved_array_pair_completed'] and first['counters']['charge_npz_files_decoded']>=2
assert reconcile['first_pair_verification']['same_snapshot_health']=='native_completed_zero'
assert reconcile['actual_counts']==array['counters'] and reconcile['unknown_inflight_ids']==reconcile['failed_ids']==[] and not reconcile['truncated_tail']
entries={};counts={k:0 for k in plan['expected_complete_counters']};entered=completed=records=0
with (CPU/'artifacts/operations.jsonl').open('rb') as f:
 for line in f:
  records+=1;assert line.endswith(b'\n');row=json.loads(line);key=row['operation_id']
  if row['event']=='entered':assert key==entered;entered+=1;entries[key]=row['charges']
  elif row['event']=='completed':
   assert key in entries
   for k,v in entries.pop(key).items():counts[k]+=v
   completed+=1
  else:raise AssertionError('unhealthy CPU operation event')
assert not entries and records==400 and entered==completed==200 and counts==array['counters']
assert array['raw_files_SHA_checked']==plan['raw_inventory'] and array['config_sha256']==registration['original_configuration_sha256']
assert array['source_manifest_sha256']==registration['original_source_manifest_sha256']
assert array['readback_script_sha256']==registration['files']['readback_saved_arrays.py']['sha256']
assert array['pure_kernel_sha256']==registration['files']['saved_array_kernels.py']['sha256']
assert array['inventory_sha256']==registration['files']['BOUND_ORIGINAL_INVENTORY.json']['sha256']
check('native0_actual200operations400journal16counters_and_firstpair_terminal_health_provenance',actual_counts=counts,first_pair_health='native_completed_zero')

expected_pairs={(p['trial'],p['snapshot'],p['arm'],p['hypothesis']) for p in plan['pair_specs']}
assert len(array['pairs'])==24 and {(p['trial'],p['snapshot'],p['arm'],p['hypothesis']) for p in array['pairs']}==expected_pairs
for p in array['pairs']:
 assert p['parents']==6593 and p['rows_per_bank']==26372 and p['r4_native_moments_verified'] and p['r8_derived_from_saved4plus4']
 assert p['covariance_denominator']==p['waveform_rank_upper_bound']==7 and p['full_scalar_covariance_shape']==[6593,9,9]
 assert p['pairwise_finite_joint_counts_verified'] and not p['r8_native_report_exists'] and not p['dense256x256_covariance_materialized']
 assert len(p['r4_by_bank'])==2 and p['IEEE_correction_absmax_V']<=4e-15
charge=array['charge_banks'];assert len(charge)==8 and len({(c['trial'],c['snapshot'],c['bank']) for c in charge})==8
for c in charge:
 assert c['A_rows']==26372 and c['A_samples']==384 and c['parents']==6593 and c['draws']==4
 assert c['actual_A_repeat_max_abs_V']<=1e-7 and c['q_min_Vns']>0 and c['actual_mu_min_Vns']>0
 assert c['Q_CPU_GPU_absmax_Vns']<=c['Q_CPU_GPU_bound_max_Vns'] and c['saved_canonical_residual_algebra_verified']
 assert c['independent_frozen_A_saved_array_readback'] and c['mean_or_first_draw_substitution'] is False
 assert c['original_SOURCE_rehashed'] is False and c['q_latent_or_RNG_replay'] is False and c['charge_r8_derived'] is False
 assert c['numerical_mu_spread_is_physical_variance'] is False
assert len({c['saved_source_input_digests_sha256'] for c in charge})==len({c['saved_forward_input_digests_sha256'] for c in charge})==1
assert all(v==0 for v in array['reviewer_operations'].values())
check('bounded_savedADC_moments_covariance_flags_and_actual_charge_A_mu_v_numerical_scope_complete',hard_pairs=24,charge_banks=8,
 max_A_repeat_V=max(c['actual_A_repeat_max_abs_V'] for c in charge),max_mu_repeat_Vns=max(c['actual_mu_repeat_max_abs_Vns'] for c in charge),
 max_GPU_CPU_Q_difference_Vns=max(c['Q_CPU_GPU_absmax_Vns'] for c in charge),all_q_and_mu_positive=True)

# Reaggregate every saved component, without CDF/loss/array work.
T=read(ROOT/'CALIBRATED_TOLERANCES_WP1_013.json')['T_by_hypothesis'];waves={a:[] for a in ('A','M0','initial','final')}
for seed in (0,1):
 initial=read(ART/f'seed{seed}_Q1/initial_evaluation/METRICS.json');final=read(ART/f'seed{seed}_Q1/final_evaluation/METRICS.json')
 for h in ('fast','slow','common_level'):
  cell=next(c for c in provisional['cells'] if c['seed']==seed and c['hypothesis']==h)
  for name,t in T[h].items():
   old=cell['components'][name]
   for arm in waves:
    vals=[(final if arm=='final' else initial)['metrics'][f'bank{b}_{arm}_{h}']['values'][name] for b in (0,1)]
    assert vals==old['bank_values'][arm] and sum(vals)/2==old[arm]
   if isinstance(t,(int,float)) and t>0:
    assert old['final']-old['A']<=t and old['final']-old['M0']<=t and old['status']=='noninferior_to_both_references'
   else:assert old['status']=='pending_not_calibrated'
  for arm in waves:waves[arm].extend(cell['components']['wave']['bank_values'][arm])
scores={a:sum(v)/12 for a,v in waves.items()};assert scores==provisional['reference_scores_mean12']
assert all(c['eligible_cell'] for c in provisional['cells']) and scores['M0']-scores['final']>1e-12
check('saved_JSON_rule_reaggregation_keeps_Q1_final_research_reference',mean12=scores,relative_gain_vs_M0_percent=100*(scores['M0']-scores['final'])/scores['M0'])

selection=copy.deepcopy(provisional)
selection.update(schema='n01/wp3-016-final-independent-development-selection/v1',UTC=datetime.now(timezone.utc).isoformat(),
 status='FINAL_DEVELOPMENT_REFERENCE_AFTER_BOUNDED_SAVED_ARRAY_QA',provisional=False,development_selection_final=True,
 independent_raw_saved_array_readback_complete=True,current_confirmed_research_reference='016_Q1_mean_off_charge_on',
 selected_research_reference='Q1',selected_charge_enabled=True,selected_mean_adapter_enabled=False,
 previous_provisional=ref(HERE/'PROVISIONAL_DEVELOPMENT_SELECTION.json'),array_result=ref(array_path),CPU_collection=ref(STAGE/'readback_collection_01/COLLECTION_RECEIPT.json'),
 expected_followup='Proceed to WP4 known-response recovery under master plan; no extraWP3 trial. Scientific response-match acceptance remains separate.',
 weights_promoted=False,scientific_acceptance=False,scientific_status='response_match_not_established')
selection_ref=save('FINAL_DEVELOPMENT_SELECTION.json',selection)
scope=read(HERE/'SCOPE_NOTE.json')
ratios={name:[c['components'][name]['absolute_distance_over_T'] for c in selection['cells']] for name in T['fast'] if isinstance(T['fast'][name],(int,float)) and T['fast'][name]>0}
summary=dict(mean12=scores,relative_M0_improvement_percent=100*(scores['M0']-scores['final'])/scores['M0'],
 M0_cell_improvements=sum(c['wave_relative_improvement_vs_M0']>0 for c in selection['cells']),M0_cell_regressions=sum(c['wave_relative_improvement_vs_M0']<0 for c in selection['cells']),
 usefulness_ALL6=True,absolute_T_ALL6=False,positive_T_components=15,uncalibrated_components=7,
 absolute_T_ratio_ranges={n:dict(min=min(v),max=max(v),failing_cells=sum(x>1 for x in v)) for n,v in ratios.items()})
result=dict(schema='n01/wp3-016-final-array-and-selection-independent-review/v1',UTC=datetime.now(timezone.utc).isoformat(),
 passed=True,blockers=[],status='COMPLETE_BOUNDED_NUMERICAL_QA_AND_FINAL_DEVELOPMENT_SELECTION',provisional=False,
 selected_research_reference='Q1',selected_charge_enabled=True,selected_mean_adapter_enabled=False,weights_promoted=False,scientific_acceptance=False,
 scientific_status='response_match_not_established',independent_raw_saved_array_readback_complete=True,
 final_development_selection=selection_ref,provisional_selection_preserved=ref(HERE/'PROVISIONAL_DEVELOPMENT_SELECTION.json'),
 sources=dict(array_result=ref(array_path),CPU_collection=ref(STAGE/'readback_collection_01/COLLECTION_RECEIPT.json'),
 original_inventory=ref(STAGE/'collection_01/REMOTE_FULL_INVENTORY.json'),prelaunch=ref(HERE/'PRELAUNCH_REVIEW.json'),
 CPU_reconciliation=ref(STAGE/'readback_registration_01/CPU_COMPLETION_RECONCILIATION.json')),
 checks=checks,actual_CPU_counts=counts,expected_operations=200,journal_records=400,
 independent_CPU_numerical_acceptance=dict(passed=True,scope=scope['independently_recomputed_planned'],
  original_SOURCE_replay=False,full_stochastic_carrier_replay=False,charge_Z_RNG_replay=False,complete_CDF_loss_C1_median_replay=False,
  empirical_hard_rank_upper_bound7_is_not_theoretical_preADC_rank=True,pairwise_missing_covariance_is_not_automatically_PSD=True),
 key_summary=summary,charge_readback_summaries=charge,
 resources=dict(supervisor_elapsed_seconds=native['elapsed_seconds'],supervisor_sampled_owned_RSS_bytes=native['peak_owned_RSS_bytes'],
  supervisor_peak_worker_artifact_bytes=native['peak_worker_artifact_bytes'],worker_getrusage_peak_RSS_bytes=array['resources']['peak_RSS_bytes'],worker=array['resources']),
 limitations=scope,cumulative_training_updates=179667,new_G_AD_or_scientific_operations_by_this_local_review=0,
 original_failed_run_difference_magnitude='unknown_not_saved',raw_files_remain_remote=True,old015_not_rerun=True,six_work_packages_complete=False,
 next_stage='WP4 known-response recovery under the master plan; fixed WP3 short-trial budget6/6 is consumed.',
 figure_recommendations_en=['Paired hard-wave distance and tiny change relative to M0','Absolute discrepancy relative to frozen T',
  'Charge coefficients and finite four-draw mean precision','Native4 and derived8 hard-wave Monte Carlo precision'])
result_ref=save('FINAL_ARRAY_AND_SELECTION_REVIEW.json',result)
print(json.dumps(dict(passed=True,review=result_ref,selection=selection_ref,selected='Q1',summary=summary),indent=2))
