"""Read collected015 metadata and frozen decision algebra. No numeric imports/CP decode."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import statistics
import sys

ROOT = Path('E:/GAN/research/N01')
STAGE = ROOT / 'receipts/wp3_015_20261015'
OUT = STAGE / 'postrun_independent_01'
COLLECTION = STAGE / 'collection_01'
COLLECTED = ROOT / 'runs/wp3_015_20261015/selected_collected_01'
ART = COLLECTED / 'runs/primary/artifacts'
CONFIG_SHA = '54c096ad1a5e112493798e5c6c8db481cd3c79b78747200af0bbd47bccba5394'
MANIFEST_SHA = 'f1255dbff0e47edb7ec20e76c7d90057201f9a56f1184b7896f0c8cb0a67825a'
INVENTORY_SHA = '9d253a57bb79d9f25b112b4f5e0009a239c21360766c89cef7357d040adebe42'
RECEIPT_SHA = '4d11ba12786d4d9be3da32762b64f877c1f191b83996fd0933e7eb7a5903b728'
T_SHA = 'f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b'
TRIALS = ('seed0_M0','seed0_M1','seed1_M0','seed1_M1')
HYPOTHESES = ('fast','slow','common_level')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): value.update(block)
    return value.hexdigest()


def record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def load(path, expected=None):
    path = Path(path)
    require(path.suffix in ('.json', '.jsonl'), 'only metadata JSON decoded')
    if expected: require(sha(path) == expected, 'metadata SHA mismatch: ' + str(path))
    return json.loads(path.read_bytes())


def save(name, value):
    with (OUT/name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def finite(x): return isinstance(x, (int,float)) and not isinstance(x,bool) and math.isfinite(x)
def average(xs): return sum(xs)/len(xs) if xs and all(finite(v) for v in xs) else None
def describe(xs):
    xs = [v for v in xs if finite(v)]
    return dict(n=len(xs), minimum=min(xs), mean=average(xs), median=statistics.median(xs), maximum=max(xs)) if xs else dict(n=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = load(ROOT/'wp3_015_config.json', CONFIG_SHA)
    mf = load(ROOT/'WP3_SOURCE_MANIFEST.json', MANIFEST_SHA)
    inv = load(COLLECTION/'REMOTE_FULL_INVENTORY.json', INVENTORY_SHA)
    receipt = load(COLLECTION/'COLLECTION_RECEIPT.json', RECEIPT_SHA)
    require(receipt['status'] == 'native_inventory_and_selected_bytes_verified' and not receipt['partial_evidence_only'], 'not a complete collection')
    require(inv['config_sha256'] == CONFIG_SHA and inv['source_manifest_sha256'] == MANIFEST_SHA and
            inv['whole_remote_inventory_verified'] is True and len(inv['whole_remote_inventory']) == 300 and
            len(inv['inventory']) == 139 and len(inv['omitted_remote_inventory']) == 161, 'inventory identity/count differs')
    require(set(inv['inventory']).isdisjoint(inv['omitted_remote_inventory']) and
            dict(inv['inventory'], **inv['omitted_remote_inventory']) == inv['whole_remote_inventory'], 'inventory partition differs')
    require(all(x.endswith('.npz') for x in inv['omitted_remote_inventory']), 'unexpected omitted file type')
    selected_bytes = 0
    for name, row in inv['inventory'].items():
        q = COLLECTED/name
        require(q.resolve().is_relative_to(COLLECTED.resolve()) and q.is_file() and not q.is_symlink(), 'selected unsafe path')
        require(q.stat().st_size == row['bytes'] and sha(q) == row['sha256'], 'selected SHA/size differs: ' + name)
        selected_bytes += row['bytes']
    require(selected_bytes == 76908267 and sha(Path(receipt['archive'])) == receipt['archive_sha256'] ==
            'f96889c5a1ebe37b6ed01c08502ef4fb869e76fed2fb6b9e4a67b6e9b182a4a8', 'selected archive SHA differs')
    require(Path(receipt['archive']).stat().st_size == inv['archive_bytes'] == 77158400, 'archive size differs')
    require(sha(COLLECTED/'wp3_015_config.json') == CONFIG_SHA and sha(COLLECTED/'WP3_SOURCE_MANIFEST.json') == MANIFEST_SHA,
            'collected frozen controls differ')
    require(len(mf['files']) == 105 and all(sha(ROOT/k) == v['sha256'] and (ROOT/k).stat().st_size == v['bytes'] for k,v in mf['files'].items()),
            'frozen local105 sources changed')

    state = load(STAGE/'wake_status_01/REMOTE_STATE.json')
    done = load(ART/'COMPLETE.json'); native = load(COLLECTED/'runs/primary/supervisor/exit_receipt.json')
    require(done == state['runs/primary/artifacts/COMPLETE.json'] and native == state['runs/primary/supervisor/exit_receipt.json'], 'collected/snapshot terminal differs')
    first = load(ART/'FIRST_UPDATE_PROOF.json'); init = load(ART/'INIT_PROOF.json')
    start = load(COLLECTED/'START_RECEIPT.json'); launch = load(COLLECTED/'runs/primary/supervisor/launch.json')
    require(start['supervisor_pid'] == launch['supervisor_pid'] == native['supervisor_pid'] == 149185 and
            launch['child_pid'] == native['child_pid'] == first['pid'] == done['pid'] == init['pid'] == 149201, 'PID chain differs')
    require(start['boot_id'] == state['current_boot_id'] and native['child_native_exit_code'] == native['exit_code'] == 0 and
            native['reason'] is None and native['error'] is None, 'native terminal failed')
    require(state['runs/primary/artifacts/FAILED.json'] is None and not state['protection_errors'] and
            done['all_four_trials_completed'] is True and done['training_skipped'] is False and done['mean_gate_all6_passed'] is True,
            'failed/partial producer')
    require(done['counters'] == done['attempted_counters'] == p['budget_caps'] and not done['inflight'] and
            done['operations_entered'] == done['operations_completed'] == 154518, 'exact cost terminal differs')
    immutable = {p['remote_root']+'/wp3_015_config.json': CONFIG_SHA}
    for label in ('source_files','inputs','contracts','protection_files'):
        for row in p[label].values(): immutable[row['path']] = row['sha256']
    require(done['immutable_files_sha256'] == immutable and len(immutable) == 497, 'protected/input/source SHA map differs')
    require(init['frozen_state_sha256'] == first['frozen_state_sha256'], 'initial/frozen proof state differs')
    require(init['paired_initialization_bytes'] is True and init['fresh_AdamW'] is True and init['zero_mean_adapter'] is True,
            'initialization chain missing')
    require(native['peak_RSS_bytes'] <= 8*1024**3 and native['peak_GPU_bytes'] <= 10*1024**3 and
            native['elapsed_seconds'] <= p['resource_limits']['wall_seconds'] and native['peak_output_bytes'] <= 6144*1024**2,
            'resource cap exceeded')

    # Stream the durable operation journal; never hold waveform/CP payloads.
    entered = completed = trial_index = 0
    stage = 'shared_preparation'; pending = {}
    attempted_by_stage = defaultdict(lambda: defaultdict(int)); completed_by_stage = defaultdict(lambda: defaultdict(int))
    with (ART/'operations.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            require(line.endswith('\n'), 'truncated operation journal')
            row = json.loads(line); oid = row['operation_id']; event = row['event']
            if event == 'entered':
                require(oid == entered and oid not in pending, 'nonunique entered operation ID')
                if row['name'].startswith('checkpoint_C1_head_seed'):
                    stage = row['name'] + '_shared_load'
                if row['name'] == 'fresh_arm_AdamW_created':
                    require(trial_index < 4, 'unregistered fifth trial')
                    stage = TRIALS[trial_index]; trial_index += 1
                pending[oid] = (stage, row['charges'], row['name']); entered += 1
                for key, value in row['charges'].items(): attempted_by_stage[stage][key] += value
            elif event == 'completed':
                require(oid in pending, 'unentered/duplicate completed ID')
                label, charges, name = pending.pop(oid)
                require(row['charges'] == charges and row['name'] == name, 'operation identity/charges changed')
                for key, value in charges.items(): completed_by_stage[label][key] += value
                completed += 1
            else: raise ValueError('failed journal event in claimed complete run')
    require(entered == completed == 154518 and not pending and trial_index == 4, 'journal not complete')
    totals = {k:sum(x.get(k,0) for x in completed_by_stage.values()) for k in p['budget_caps']}
    require(totals == p['budget_caps'] and {k:dict(v) for k,v in attempted_by_stage.items()} ==
            {k:dict(v) for k,v in completed_by_stage.items()}, 'journal cost identity differs')
    for durable in (state['durable_child_progress'], native):
        require(not durable['unknown_inflight_ids'] and not durable['failed_ids'] and not durable['truncated_tail'], 'unknown durable work')
        require(all(durable['completed_counts'].get(k,0) == v and durable['entered_counts'].get(k,0) == v for k,v in totals.items()), 'durable lower bounds differ')

    gate = load(ART/'MEAN_RESIDUAL_GATE.json')
    require(gate['all6_passed'] is True and len(gate['cells']) == 6 and gate['REAL_rows'] == 21047 and gate['REAL_logical_groups'] == 422,
            'registered mean gate incomplete')
    for cell in gate['cells']:
        threshold = max(1e-7, 3*math.sqrt(cell['bank_MC_SE_RMS_V']**2 + cell['real_cluster_SE_RMS_V']**2))
        require(math.isclose(threshold, cell['required_residual_RMS_V'], rel_tol=1e-14) and
                cell['passed'] == (cell['residual_RMS_V'] > threshold and cell['bank_residual_cosine'] > .5), 'mean gate scalar algebra differs')
    require(gate['before_checkpoint_load'] and gate['before_NN'] and gate['before_new_random'] and gate['before_optimizer'] and
            not gate['source_numeric_decoded'] and not gate['DEV_numeric_decoded'] and not gate['conditional_mean_identified'], 'gate order/role boundary differs')
    integral = load(ART/'INTEGRAL_TRAIN_STATISTICS_REFERENCE.json')
    original_integral = load(p['inputs']['integral_statistics']['local_path'], p['inputs']['integral_statistics']['sha256'])
    require(integral == original_integral and integral['fit_split'] == 'TRAIN' and integral['fit_rows'] == 21047 and
            integral['candidate_outputs_used'] is False and integral['all_rows_retained'] is True, 'fixed integral statistics use chain differs')
    lengths = (256,210,16,30,50,160)
    require(all(math.isclose(x, math.sqrt(n)/16383, rel_tol=1e-14) for x,n in zip(integral['ADC_quadrature_floor_Vns'],lengths)) and
            all(finite(x) and x >= f for x,f in zip(integral['scale_Vns'],integral['ADC_quadrature_floor_Vns'])), 'fixed scale/floor invalid')

    inputs = {}; trial_summaries = {}; logs = {}; metric_sets = {}
    for trial in TRIALS:
        seed = int(trial[4]); arm = trial.rsplit('_',1)[1]; root = ART/trial
        inp, td, fp = load(root/'RUN_INPUTS.json'), load(root/'COMPLETE.json'), load(root/'FIRST_UPDATE_PROOF.json')
        require(td['native_status'] == 'COMPLETE' and td['G_updates'] == 2000 and td['candidate'] == arm and
                td['frozen_A_005_unchanged'] is True and td['frozen_state_sha256'] == init['frozen_state_sha256'], 'trial frozen/terminal mismatch')
        require(inp['seed'] == td['seed'] == p['seed_sets'][seed] and inp['reference']['input_key'] == f'C1_head_seed{seed}' and
                inp['reference']['sha256'] == p['inputs'][f'C1_head_seed{seed}']['sha256'], 'reference seed/CP binding differs')
        require(inp['mean_theta_initial'] == [0.]*6 and inp['fresh_AdamW'] and not inp['inherit_014_optimizer'] and
                not inp['DEV_numeric_read'] and inp['same_seed_reset_streams'] and not inp['weights_promoted'], 'paired fresh initialization differs')
        names = ['basis_head.weight','basis_head.bias'] if arm == 'M0' else ['mean_theta']
        require(inp['trainable_parameter_names'] == td['trainable_parameter_names'] == fp['trainable_parameter_names'] == names and
                inp['trainable_parameters'] == fp['trainable_parameters'] == (1288 if arm == 'M0' else 6), 'whitelist differs')
        require(fp['actual_G_updates'] == 1 and fp['gradient_nonzero'] and fp['actual_parameter_update_nonzero'] and fp['pid'] == done['pid'], 'trial G1 proof missing')
        if arm == 'M1': require(td['protected_C1_shape_unchanged'] is True, 'M1 shape not protected')
        else: require(td['mean_theta'] == [0.]*6, 'M0 mean enabled')
        cov = td['coverage']
        require(cov['source_unique'] == cov['source_total'] == 6593 and cov['real_unique'] == cov['real_total'] == 21047 and
                cov['source_parent_visits'] == 1024000 and cov['real_row_visits'] == 2048000 and cov['source_epochs'] == 155 and cov['real_epochs'] == 97,
                'all TRAIN coverage mismatch')
        log = [json.loads(x) for x in (root/'training.jsonl').read_text().splitlines()]
        require([x['G_updates'] for x in log] == [1]+list(range(100,2001,100)), 'saved21 update log gap')
        for x in log:
            require(x['trainable_parameters'] == inp['trainable_parameters'] and len(x['source_parent_rows']) == 512 and
                    all(type(n) is int and 0 <= n < 6593 for n in x['source_parent_rows']) and x['real_row_count'] == 1024 and
                    x['all_legal_TRAIN_sampling'] and x['D_updates'] == x['noise_updates'] == 0, 'logged row/parameter schema differs')
            require(math.isclose(x['objective'],x['wave']+x['weighted_distribution']+x['weighted_regular'],rel_tol=1e-13,abs_tol=1e-14), 'objective sum differs')
            require(x['mean_relative_RMS_max'] <= .1+1e-14 and 0 <= x['mean_trust_scale_min'] <= 1, 'mean trust exceeded')
        inputs[trial] = inp; logs[trial] = log
        trial_summaries[trial] = dict(G_updates=2000, coverage=cov, parameters=inp['trainable_parameters'],
            whitelist=names, initial_shape_state_sha256=inp['shape_reference_head_sha256'],
            configured_streams=inp['seed'], configured_streams_are_checkpoint_RNG_state=False,
            last_mean_theta=td['mean_theta'], protected_C1_shape_unchanged=td['protected_C1_shape_unchanged'],
            logged_preclip_gradient_norm=describe([x['preclip_total_gradient_norm'] for x in log]),
            logged_actual_parameter_update_norm=describe([x['actual_parameter_update_norm'] for x in log]),
            logged_mean_relative_RMS_max=describe([x['mean_relative_RMS_max'] for x in log]),
            logs_are21_different_training_minibatches_not_population_curve=True, resources=td['resources'])
        initial = load(root/'initial_evaluation/METRICS.json'); final = load(root/'final_evaluation/METRICS.json')
        require(initial['source_parents'] == final['source_parents'] == 6593 and initial['REAL_rows'] == final['REAL_rows'] == 21047 and
                initial['draws_per_bank'] == final['draws_per_bank'] == 4 and initial['MC_banks'] == final['MC_banks'] == 2 and
                initial['all_signed_rows_retained'] and final['all_signed_rows_retained'] and initial['no_DEV'] and final['no_DEV'], 'hard evaluation population differs')
        require(set(initial['metrics']) == {f'bank{b}_{a}_{h}' for b in (0,1) for a in ('A','initial') for h in HYPOTHESES} and
                set(final['metrics']) == {f'bank{b}_final_{h}' for b in (0,1) for h in HYPOTHESES}, 'metric slots missing or duplicate')
        for m in list(initial['metrics'].values()) + list(final['metrics'].values()):
            require(m['coverage']['conditional_coverage_complete'] and m['coverage']['conditional_bins_used'] == 8 and
                    m['fake_rows'] == 26372 and m['reference_rows'] == 21047 and m['all_rows_retained'] and
                    m['projection_context']['fingerprint'] == p['evaluation']['projection_fingerprint'], 'coverage/projection differs')
            require(math.isclose(m['values']['wave'],.5*m['wave_components']['swd']+.25*m['wave_components']['coordinate']+.25*m['wave_components']['short_lag'],rel_tol=1e-13), 'wave component algebra differs')
            require(math.isclose(m['legacy_objective'],m['values']['wave']+2*m['values']['QPSD'],rel_tol=1e-13), 'legacy objective algebra differs')
        comparison = load(root/'HARD_COMPARISON.json')['per_bank_hypothesis']
        for c in comparison:
            for key,v in c['checks'].items():
                require(v['final'] == final['metrics'][c['bank']+'_final_'+c['hypothesis']]['values'][key] and
                        v['A'] == initial['metrics'][c['bank']+'_A_'+c['hypothesis']]['values'][key] and
                        v['common_initial'] == initial['metrics'][c['bank']+'_initial_'+c['hypothesis']]['values'][key], 'producer comparison not same metrics')
        metric_sets[trial] = dict(initial=initial['metrics'], final=final['metrics'])
    for seed in (0,1):
        x,y = inputs[f'seed{seed}_M0'],inputs[f'seed{seed}_M1']
        require(x['shape_reference_head_sha256'] == y['shape_reference_head_sha256'] and x['reference'] == y['reference'] and x['seed'] == y['seed'], 'within-seed initialization not paired')
        require(metric_sets[f'seed{seed}_M0']['initial'] == metric_sets[f'seed{seed}_M1']['initial'], 'common initial hard metrics differ')
        for a,b in zip(logs[f'seed{seed}_M0'],logs[f'seed{seed}_M1']):
            require(a['source_parent_rows'] == b['source_parent_rows'] and a['projection_context'] == b['projection_context'], 'logged streams not paired')

    T = load(ROOT/'CALIBRATED_TOLERANCES_WP1_013.json', T_SHA)['T_by_hypothesis']
    cells = []; results = {}
    for arm in ('M0','M1'):
        arm_cells = []; final_scores = []; initial_scores = []
        for seed in (0,1):
            source = metric_sets[f'seed{seed}_{arm}']
            for h in HYPOTHESES:
                F = [source['final'][f'bank{b}_final_{h}']['values'] for b in (0,1)]
                A = [source['initial'][f'bank{b}_A_{h}']['values'] for b in (0,1)]
                I = [source['initial'][f'bank{b}_initial_{h}']['values'] for b in (0,1)]
                components = {}; eligible = True; reasons = []; pending_uncalibrated = []
                for key,bound in T[h].items():
                    f,a,i = [average([v.get(key) for v in rows]) for rows in (F,A,I)]
                    calibrated = finite(bound) and bound > 0
                    da = None if f is None or a is None else f-a
                    di = None if f is None or i is None else f-i
                    if not calibrated: status='pending_not_calibrated'; pending_uncalibrated.append(key)
                    elif da is None or di is None: status='pending_missing_required_calibrated_metric'; eligible=False; reasons.append(key+':undefined')
                    elif da > bound or di > bound: status='failed_dual_reference_noninferiority'; eligible=False; reasons.append(key+':delta_above_T')
                    else: status='noninferior_to_both_references'
                    components[key] = dict(final=f,A=a,common_initial=i,T=bound,
                        delta_final_minus_A=da,delta_final_minus_common_initial=di,status=status,
                        absolute_distance_over_T=f/bound if calibrated and f is not None else None,
                        absolute_within_T=f<=bound if calibrated and f is not None else None)
                wf = components['wave']; rel = 1-wf['final']/wf['A']; rel_initial = 1-wf['final']/wf['common_initial']
                cell = dict(candidate=arm,seed=seed,hypothesis=h,eligible_cell=eligible,reasons=reasons,
                    wave_relative_improvement_vs_A=rel,wave_relative_improvement_vs_common_initial=rel_initial,
                    wave_usefulness_2percent=rel>=.02,components=components,pending_uncalibrated=pending_uncalibrated,
                    absolute_T_all_positive_defined=all(v['absolute_within_T'] is True for v in components.values() if finite(v['T']) and v['T']>0))
                cells.append(cell); arm_cells.append(cell)
                final_scores.extend(v['wave'] for v in F); initial_scores.extend(v['wave'] for v in I)
        require(len(arm_cells) == 6 and len(final_scores) == len(initial_scores) == 12, 'ALL6/mean12 cardinality differs')
        results[arm] = dict(eligible=all(x['eligible_cell'] for x in arm_cells),
            dual_reference_noninferior_cells=sum(x['eligible_cell'] for x in arm_cells),
            final_wave_mean12=average(final_scores),common_initial_wave_mean12=average(initial_scores),
            relative_wave_usefulness_cells=sum(x['wave_usefulness_2percent'] for x in arm_cells),
            relative_wave_usefulness_ALL6=all(x['wave_usefulness_2percent'] for x in arm_cells),
            absolute_T_ALL6=all(x['absolute_T_all_positive_defined'] for x in arm_cells),
            calibrated_components_per_hypothesis={h:sum(finite(v) and v>0 for v in T[h].values()) for h in HYPOTHESES})
    eligible = [x for x in ('M0','M1') if results[x]['eligible']]
    winner = None
    if eligible:
        winner = 'M0' if len(eligible)==2 and abs(results['M0']['final_wave_mean12']-results['M1']['final_wave_mean12'])<=1e-12 else min(eligible,key=lambda x:results[x]['final_wave_mean12'])
    improved = winner is not None and results[winner]['common_initial_wave_mean12']-results[winner]['final_wave_mean12'] > 1e-12
    selected = winner if improved else '014_C1_reference_mean_disabled'
    require(selected == 'M0', 'unexpected selection; investigate raw metadata rather than override frozen rule')
    selection = dict(schema='n01/wp3-independent-development-selection/v1',UTC=datetime.now(timezone.utc).isoformat(),
        status='provisional_development_reference_pending_bounded_saved_array_readback',passed=True,
        frozen_configuration_sha256=CONFIG_SHA,tolerance_table_sha256=T_SHA,inventory_sha256=INVENTORY_SHA,
        bank_aggregation='mean2bank separate empirical distances, not pooled8',candidates=results,cells=cells,
        eligible_candidates=eligible,winner_by_mean12=winner,tie_epsilon=1e-12,tie_preference='M0',
        winner_better_than_common_initial_by_more_than_epsilon=improved,selected_research_reference=selected,
        selected_mean_adapter_enabled=False,weights_promoted=False,scientific_acceptance=False,
        absolute_T_ALL6_passed=results[selected]['absolute_T_ALL6'],
        independent_raw_saved_array_readback_complete=False,full_empirical_loss_independently_replayed=False,
        original014_C1_historical_usefulness='5/6 remains unchanged; this015 M0 continuation is a separate completed candidate',
        interpretation_zh='按冻结双参考ALL6非劣和12格wave均值，M0继续形状头成为下一阶段研究参考，均值关闭；M1六参数均值臂未改善共同初始。M0用途6/6，绝对T仍失败，科学状态不变。',
        scientific_status='response_match_not_established')
    save('DEVELOPMENT_SELECTION.json', selection)
    actual = dict(JSON_and_JSONL_only_decoded=True,checkpoint_tensor_loads=0,NPZ_decodes=0,NN_rows=0,AD_queries=0,
        new_training_updates=0,new_random_draws=0,original_SOURCE_REAL_DEV_waveform_reads=0,SSH_calls=0,
        selected_checkpoint_payloads_only_opaque_SHA_checked=True)
    metadata = dict(schema='n01/wp3-independent-local-metadata-readback/v1',UTC=datetime.now(timezone.utc).isoformat(),passed=True,status='PASS',
        collection_receipt=record(COLLECTION/'COLLECTION_RECEIPT.json'),inventory=record(COLLECTION/'REMOTE_FULL_INVENTORY.json'),
        selected_count=139,selected_bytes=selected_bytes,whole_remote_inventory_count=300,remote_NPZ_count=161,
        remote_NPZ_bytes=sum(x['bytes'] for x in inv['omitted_remote_inventory'].values()),all_raw_artifacts_collected=False,
        native=dict(child=0,supervisor=0,worker_pid=149201,supervisor_pid=149185,boot_id=start['boot_id'],resources=native),
        immutable_SHA_records497_match=True,original_A_005_state_hashes=init['frozen_state_sha256'],
        journal=dict(entered=entered,completed=completed,failed=0,unknown_inflight=[],completed_counts=totals,
            actual_completed_by_stage={k:dict(v) for k,v in completed_by_stage.items()}),
        trials=trial_summaries,within_seed_C1_head_initial_hashes_and_hard_initial_metrics_equal=True,
        explicit_stream_addresses_and_saved21_parent_rows_projection_contexts_paired=True,
        not_independent_everyG_sampler_reconstruction=True,checkpoint_tensors_not_loaded=True,
        mean_gate_saved_scalar_formula_verified=True,mean_gate_array_inputs_not_independently_recomputed=True,
        mean_gate=gate,integral_statistics_reference_SHA=record(ART/'INTEGRAL_TRAIN_STATISTICS_REFERENCE.json'),
        integral_statistics_fit21047_floor_and_use_chain_verified=True,median_independently_refitted=False,
        development_selection=record(OUT/'DEVELOPMENT_SELECTION.json'),actual_review_operations=actual,
        raw_array_numeric_status='pending registered one CPU readback; remote inventory verification is not numeric readback',
        native_full_wave_QPSD_noise_distances='producer evidence plus this independent JSON decision algebra, no CDF/loss replay',
        scientific_acceptance=False,scientific_status='response_match_not_established')
    save('LOCAL_METADATA_READBACK.json', metadata)
    require('numpy' not in sys.modules and 'torch' not in sys.modules, 'numeric modules imported')
    for name in ('DEVELOPMENT_SELECTION.json','LOCAL_METADATA_READBACK.json'):print(json.dumps(record(OUT/name)))
    print(json.dumps(dict(selected=selected,candidate_scores=results),ensure_ascii=False))


if __name__ == '__main__': main()
