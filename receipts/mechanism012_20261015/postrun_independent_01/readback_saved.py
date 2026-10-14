"""Read only012 saved outputs with stdlib/NumPy; no Torch/model/AD/CP access."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import os
import sys
import tarfile
import numpy as np

ROOT = Path('E:/GAN/research/N01')
COLLECTED = ROOT / 'runs/mechanism012_20261015/collected_01'
ART = COLLECTED / 'runs/primary/artifacts'
OUT = ROOT / 'receipts/mechanism012_20261015/postrun_independent_01'
H = ('fast', 'slow', 'common_level')
PATHS = ('S_clamp_surrogate_fixed_actual_crop', 'H_fixed_hard_ADC_fixed_actual_crop',
         'H_moving_same_hard_ADC_first_argmax')
NEW = '/home/rylan/projects/ej276d-n01/mechanism012_20261015'
checks = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def check(value, name, **detail):
    checks.append(dict(name=name, passed=bool(value), **detail))


def near(a, b, atol=1e-12):
    return bool(np.allclose(a, b, atol=atol, rtol=0, equal_nan=True))


def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def cosine(a, b):
    n, m = np.linalg.norm(a), np.linalg.norm(b)
    return None if n == 0 or m == 0 else float(np.clip(np.dot(a, b)/(n*m), -1, 1))


def group(mask, parent, wave, q):
    n = int(mask.sum())
    return dict(draws=n, parents=int(np.unique(parent[mask]).size), total_draws=2048, total_parents=512,
                wave_signed_influence_sum=float(wave[mask].sum()), QPSD_signed_influence_sum=float(q[mask].sum()),
                wave_signed_influence_mean=None if n == 0 else float(wave[mask].mean()),
                QPSD_signed_influence_mean=None if n == 0 else float(q[mask].mean()))


def equal_group(a, b):
    return all(a[k] == b[k] if k.endswith('draws') or k.endswith('parents') or a[k] is None
               else near(a[k], b[k]) for k in a)


def main():
    target = OUT/'NUMERIC_READBACK.json'
    if target.exists():
        raise RuntimeError('Preserve previous independent readback')
    collection_path = ROOT/'receipts/mechanism012_20261015/collection_01/COLLECTION_RECEIPT.json'
    collection = load(collection_path)
    complete, status, summary = (load(ART/n) for n in ('COMPLETE.json', 'status.json', 'SUMMARY.json'))
    probe, point, first, frozen = (load(ART/n) for n in ('DIRECTIONAL_PROBE.json', 'POINT_REPLAY.json',
                                                       'FIRST_DIAGNOSTIC_PROOF.json', 'FROZEN_STATE.json'))
    native = load(COLLECTED/'runs/primary/supervisor/exit_receipt.json')
    launch, start = load(COLLECTED/'runs/primary/supervisor/launch.json'), load(COLLECTED/'START_RECEIPT.json')
    mf, bindings, policy = (load(COLLECTED/n) for n in ('MECHANISM012_SOURCE_MANIFEST.json',
                                                      'MECHANISM012_BINDINGS.json', 'mechanism012_policy.json'))
    prelaunch = load(ROOT/'receipts/mechanism012_20261015/independent_01/PRELAUNCH_REVIEW.json')
    freeze = load(ROOT/'receipts/mechanism012_20261015/independent_01/FROZEN_PACKAGE_REVIEW.json')
    # Archive stream and all extracted bytes: does not extract or execute anything.
    archive = Path(collection['archive'])
    check(sha(archive) == collection['archive_sha256'], 'collection_archive_SHA')
    inventory = {}
    with tarfile.open(archive, 'r:') as tar:
        members = tar.getmembers()
        check(len(members) == len({m.name for m in members}) == 24, '24_unique_regular_archive_members')
        for member in members:
            parts = member.name.split('/')
            safe = not member.name.startswith('/') and '\\' not in member.name and ':' not in member.name
            check(safe and all(x not in ('', '.', '..') for x in parts) and member.isfile(), 'regular_member_'+member.name)
            stream = tar.extractfile(member)
            h = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024**2), b''):
                h.update(chunk)
            saved = COLLECTED/member.name
            d = h.hexdigest()
            check(saved.stat().st_size == member.size and sha(saved) == d, 'member_SHA_size_'+member.name)
            inventory[member.name] = dict(bytes=member.size, sha256=d)
    files = {str(p.relative_to(COLLECTED)).replace('\\', '/') for p in COLLECTED.rglob('*') if p.is_file()}
    check(files == set(inventory) and len(files) == collection['file_count'] == 24, 'collected_exact_inventory24')
    check(len(complete['artifacts']) == collection['job']['artifact_count'] == 13, 'COMPLETE_inventory13_plus_COMPLETE_self')
    for name, record in complete['artifacts'].items():
        check(inventory['runs/primary/artifacts/'+name] == record, 'COMPLETE_artifact_'+name)
    check(not (ART/'FAILED.json').exists(), 'no_FAILED_artifact')
    for n, key in [('MECHANISM012_SOURCE_MANIFEST.json', 'source_manifest_sha256'), ('MECHANISM012_BINDINGS.json', 'bindings_sha256')]:
        check(sha(COLLECTED/n) == freeze[key] == collection[key] == sha(ROOT/n), 'frozen_control_SHA_'+n)
    check(sha(COLLECTED/'mechanism012_policy.json') == freeze['policy_sha256'] == bindings['policy_sha256'] == mf['policy_sha256'],
          'frozen_policy_SHA')
    check(len(mf['files']) == 69 and len(bindings['files']) == 36, 'source69_bound36_metadata')
    for rel, d in mf['files'].items():
        check(sha(ROOT/rel) == d == complete['sources_sha256'][NEW+'/'+rel], 'current_frozen_source_'+rel)
    for rel, d in prelaunch['sources_sha256'].items():
        check(mf['files'][rel] == d, 'PRELAUNCH_source_pin_'+rel)
    check(mf['bound_inputs'] == {k: v['sha256'] for k, v in bindings['files'].items()}, 'bound_slot36_MF_consistency')
    for slot, record in bindings['files'].items():
        check(complete['inputs_sha256'][record['path']] == record['sha256'], 'native_declared_bound_SHA_'+slot)
    check(complete['inputs_sha256'] == status['inputs_sha256'] == first['inputs_sha256'], 'native_input_hash_snapshot_unchanged')
    check(complete['sources_sha256'] == status['sources_sha256'] == first['sources_sha256'], 'native_source_hash_snapshot_unchanged')
    pid_ok = (start['supervisor_pid'] == launch['supervisor_pid'] == native['supervisor_pid'] == 20592
              and launch['child_pid'] == native['child_pid'] == complete['pid'] == first['pid'] == status['pid'] == 20599
              and start['boot_id'] == launch['boot_id'] == native['boot_id'])
    check(pid_ok and native['child_native_exit_code'] == native['exit_code'] == 0 and native['reason'] is None,
          'native0_single_boot_supervisor_child_FIRST_COMPLETE_PID_chain')
    check(first['extra_pilot_AD_queries'] == 0 and first['AD_queries_completed'] == 1
          and first['finite_formal_AD'] and first['first_global_query'] == 'wave', 'FIRST_is_formal_wave_q1_no_pilot')
    check(all(first['counters'][k] == 2048 for k in ('old_A_rows','noise005_rows','head_final_rows')),
          'FIRST_complete_global_forward_2048_each')
    check(complete['head_parameter_names'] == ['basis_head.weight','basis_head.bias'] and complete['head_parameter_count'] == 1288,
          'exact_two_tensor1288_parameter_whitelist')
    budget = dict(policy['budgets'])
    budget['checkpoint_loads'] = budget.pop('checkpoint_loads_max')
    check(complete['counters'] == complete['attempted_counters'] == status['counters'] == collection['job']['counters'] == budget,
          'exact_native_full_budget2048_each_9scalar2VJP49152ADC5CP_zeros')
    check(not complete['failed_operations'] and not complete['inflight'] and not native.get('journal_truncated_tail'),
          'no_failed_inflight_or_truncated_operations')
    operations = [json.loads(x) for x in (ART/'operations.jsonl').read_text().splitlines()]
    entered = [x for x in operations if x['event'] == 'entered']
    completed = [x for x in operations if x['event'] == 'completed']
    check(len(operations) == 864 and len(entered) == len(completed) == 432
          and [x['operation_id'] for x in entered] == list(range(432))
          and [x['operation_id'] for x in completed] == list(range(432)), 'journal432_unique_entered_completed_no_gaps')
    running = {k: 0 for k in budget}
    identity_ok = True
    for e, d in zip(entered, completed):
        identity_ok &= e['operation_id'] == d['operation_id'] and e['name'] == d['name'] and e['charges'] == d['charges']
        for key, amount in e['charges'].items():
            running[key] += amount
        identity_ok &= running == d['counters']
    check(identity_ok and running == budget, 'each_operation_identity_charge_and_prefix_counter_reconstruction')
    check(complete['operations_entered'] == complete['operations_completed'] == 432, 'terminal_operation_count432')
    scalar = [x['name'] for x in entered if x['charges'].get('scalar_AD_queries')]
    vjp = [x['name'] for x in entered if x['charges'].get('masked_endpoint_VJP_queries')]
    check(scalar == ['wave','QPSD','RMS','curvature','saturation','total','QPSD_fixed_reference','anchor_wave','anchor_QPSD_fixed']
          and vjp == ['wave_crop_changed','QPSD_crop_changed'], 'exact_ordered_9scalar_2joint_mask_VJP_names')
    for name in ('noise005_and_shared_extensions','single_A_and_final_head_family'):
        check([x['details'] for x in entered if x['name'] == name] == [dict(start=x,stop=x+128) for x in range(0,2048,128)],
              'single_128_chunk_coverage_'+name)
    state_ok = summary['frozen_state_before'] == summary['frozen_state_after'] == frozen['before'] == first['frozen_state_before']
    check(state_ok and complete['frozen_state_unchanged'] and complete['head_tensors_unchanged'], 'saved_tensor_digest_snapshots_all_identical')
    check(frozen['optimizer_instantiated'] is False and frozen['only_final_head_loaded_for_forward']
          and frozen['wrapper_moved_after_construction'] is False and not complete['DEV_numeric_read'], 'saved_freeze_restore_no_optimizer_no_wrapper_move_noDEV')
    check(complete['scientific_status'] == 'response_match_not_established' and complete['formal_response_acceptance'] is False,
          'no_formal_scientific_acceptance')
    limits = policy['resource_limits']
    check(native['peak_sampled_RSS_bytes'] <= limits['job_tree_RSS_GiB']*1024**3
          and native['peak_sampled_GPU_bytes'] <= limits['GPU_GiB']*1024**3
          and native['elapsed_seconds'] <= limits['wall_seconds'] and native['peak_output_bytes'] <= limits['output_MiB']*1024**2,
          'reported_sampled_resources_within_caps')
    # Only the three saved012 NPZs are decoded; no input dataset or checkpoint is opened.
    with np.load(ART/'GRADIENTS.npz', allow_pickle=False) as z:
        g = {k: z[k] for k in z.files}
    with np.load(ART/'ROW_MECHANISMS.npz', allow_pickle=False) as z:
        rows = {k: z[k] for k in z.files}
    with np.load(ART/'EVALUATION_PROJECTIONS.npz', allow_pickle=False) as z:
        projections = {k: z[k] for k in z.files}
    check(list(g['parameter_names']) == complete['head_parameter_names'] and list(g['parameter_shapes']) == ['8x32x5','8'],
          'saved_gradient_parameter_identity_and_shapes')
    names = [k for k, v in g.items() if v.shape == (1288,) and v.dtype.kind == 'f']
    check(len(names) == 14 and all(np.isfinite(g[k]).all() for k in names), 'all14_full1288_gradient_vectors_finite')
    vectors = {}
    total = g['total']
    for name in names:
        v = g[name]
        native_v = summary['gradient_attribution']['vectors'][name]
        norm = float(np.linalg.norm(v))
        record = dict(L2=norm, weight_L2=float(np.linalg.norm(v[:1280])), bias_L2=float(np.linalg.norm(v[1280:])),
                      basis_slice_L2=[float(np.linalg.norm(np.r_[v[:1280].reshape(8,160)[k],v[1280+k]])) for k in range(8)],
                      cosine_with_total=cosine(v,total), signed_projection_on_total=float(v@total/np.linalg.norm(total)))
        match = all(near(record[k], native_v[k]) for k in ('L2','weight_L2','bias_L2','basis_slice_L2','signed_projection_on_total'))
        match &= record['cosine_with_total'] == native_v['cosine_with_total']['value'] if norm == 0 else near(record['cosine_with_total'],native_v['cosine_with_total']['value'])
        check(match, 'independent_gradient_norm_cosine_signed_projection_'+name)
        vectors[name] = record
    pairs = {}
    for a in names:
        pairs[a] = {}
        for b in names:
            c = cosine(g[a],g[b]); expected = summary['gradient_attribution']['pairwise_cosines'][a][b]['value']
            check(c == expected if c is None else near(c,expected), 'pairwise_cosine_'+a+'_'+b)
            pairs[a][b] = c
    component_names = ('wave','QPSD','RMS','curvature','saturation')
    summed = sum(g[k] for k in component_names)
    component_norm_sum = sum(np.linalg.norm(g[k]) for k in component_names)
    sum_error = float(np.linalg.norm(total-summed)); sum_relative = sum_error/component_norm_sum
    check(sum_relative <= 1e-5 and near(sum_relative, summary['gradient_attribution']['total_sum_relative_error']),
          'TOTAL_gradient_five_weighted_components_sum', relative_error=sum_relative)
    check(near(g['regular'],g['RMS']+g['curvature']+g['saturation']), 'regular_gradient_vector_identity')
    check(np.array_equal(g['QPSD'],g['QPSD_fixed_reference']) and summary['fixed_membership_reference']['value_abs_error'] == 0,
          'native_vs_fixed_membership_reference1288_gradient_and_scalar_equality')
    for k in ('wave','QPSD'):
        check(near(g[k],g[k+'_changed']+g[k+'_unchanged']), 'changed_unchanged_joint_VJP_vector_sum_'+k)
    comp = point['comparisons']
    weighted = point['weighted_terms']
    check(near(weighted['wave'],np.mean([comp[h]['actual_wave'] for h in H]))
          and near(weighted['QPSD'],2*np.mean([comp[h]['actual_QPSD'] for h in H]))
          and near(sum(weighted.values()),summary['total_loss']) and weighted == summary['weighted_loss_values'],
          'mean_h_wave_plus2QPSD_plus_regular_TOTAL_scalar_algebra')
    check(near(first['first_query_gradient_L2'],vectors['wave']['L2']), 'FIRST_wave_gradient_matches_saved_vector')
    bank_hash = hashlib.sha256()
    for name in sorted(projections):
        value = np.ascontiguousarray(projections[name])
        bank_hash.update(name.encode('utf8')); bank_hash.update(str(tuple(value.shape)).encode('ascii'))
        bank_hash.update(('torch.'+str(value.dtype)).encode('ascii')); bank_hash.update(value.tobytes())
    check(bank_hash.hexdigest() == policy['projection']['fingerprint'] == point['frozen_projection']['fingerprint'],
          'independent19_saved_projection_bank_fingerprint')
    parent = rows['source_parent_row']
    check(np.array_equal(parent,np.repeat(np.arange(512),4)) and rows['source_parent_ids'].shape == (512,)
          and np.unique(rows['source_parent_ids']).size == np.unique(rows['source_parent_rows']).size == 512
          and np.unique(rows['REAL_reference_rows']).size == np.unique(rows['REAL_reference_ids']).size == 2048,
          '512_parent_major_times4_and2048_unique_REAL_identities')
    q8 = rows['source_native_Q8']
    check(q8.shape == (512,) and np.all((q8>=0)&(q8<8)), 'SOURCE_Q8_fixed_descriptive_parent_strata_all_retained')
    S,A,R = (rows[k] for k in ('signal_pre_V','base_signal_pre_V','random_shape_pre_V'))
    check(S.shape == A.shape == R.shape == (2048,384) and np.isfinite(S).all() and np.isfinite(A).all() and np.isfinite(R).all(),
          'saved_pure_signal_A_random_shape_finite2048x384')
    check(maxerr(S,A+R) <= 1e-14 and np.all(R[:,:100] == 0) and maxerr(R[:,110:366].sum(1),0) <= 1e-10,
          'pure_signal_A_plus_random_prefix0_canonical_random_Q0')
    direction = rows['probe_carrier_direction']
    check(near(direction,R/np.max(np.abs(R))) and maxerr(direction[:,110:366].sum(1),0) <= 1e-10
          and np.all(direction[:,:100] == 0), 'six_probe_direction_saved_R_global_max_detached_zeroQ_prefix0')
    influence, union = {}, np.zeros(512,bool)
    decomposition_errors, adjoint_errors = {}, {}
    for name in ('wave','QPSD'):
        E = g[name+'_endpoint_gradients']
        reconstructed = np.zeros((2048,384),dtype=np.float64)
        check(E.shape == (3,2048,256) and g[name+'_shared_S_partial_gradient'].shape == (2048,384), 'endpoint3_and_sharedS_shapes_'+name)
        for hi,h in enumerate(H):
            # Native saved flags certify all carriers in this actual point lie in inclusive clamp domain.
            check(not rows[h+'_actual_flag_pre_ADC_out_of_range'].any(), 'all_inclusive_clamp_active_saved_flags_'+h+'_'+name)
            idx = rows[h+'_actual_crop_start'][:,None]+np.arange(256)[None,:]
            np.add.at(reconstructed,(np.arange(2048)[:,None],idx),E[hi])
            reconstructed[:,:100] -= E[hi].sum(1)[:,None]/100
        err = maxerr(reconstructed,g[name+'_shared_S_partial_gradient'])
        adjoint_errors[name] = err
        check(err <= 1e-12, 'independent_numpy_fixed_gather_prefix_S_adjoint_'+name, max_abs_error=err,
              scope='uses saved per-row out-of-range flags; raw carrier not saved/recreated')
    for hi,h in enumerate(H):
        actual, anchor = rows[h+'_actual_window_V'], rows[h+'_anchor_window_V']
        mask = rows[h+'_crop_changed'].astype(bool)
        check(np.array_equal(mask,rows[h+'_actual_peak_index'] != rows[h+'_anchor_peak_index'])
              and np.array_equal(rows[h+'_actual_crop_start'],rows[h+'_actual_peak_index']-16)
              and np.array_equal(rows[h+'_actual_crop_stop'],rows[h+'_actual_peak_index']+240), 'actual_anchor_crop_member_identity_'+h)
        union[np.unique(parent[mask])] = True
        influence[h] = {'categories':{}, 'source_Q8':[], 'canonical_Qbin_migration_draws':int(rows[h+'_native_Qbin_migrated'].sum())}
        for term in ('wave','QPSD'):
            recomputed = np.sum(g[term+'_endpoint_gradients'][hi]*(anchor-actual),axis=1)
            check(near(recomputed,rows[h+'_signed_influence_'+term]), 'signed_weighted_endpoint_displacement_dot_'+h+'_'+term)
        categories = dict(all=np.ones(2048,bool),crop_changed=mask,crop_unchanged=~mask)
        for name in ('peak_at_search_boundary','global_peak_outside_search','Q_nonpositive','PSD_undefined','PSD_guarded','ADC_rail_hit',
                     'pre_ADC_out_of_range','shape_peak_tied','shape_peak_floor_active','Q_exact_zero','Q_negative','ADC_halfgrid_tied'):
            categories[name] = rows[h+'_actual_flag_'+name].astype(bool)
        categories['peak_tied'] = rows[h+'_peak_tied'].astype(bool)
        categories['peak_margin_zero'] = rows[h+'_first_runnerup_margin_V'] == 0
        for name,m in categories.items():
            record = group(m,parent,rows[h+'_signed_influence_wave'],rows[h+'_signed_influence_QPSD'])
            check(equal_group(record,summary['influence']['hypotheses'][h]['categories'][name]), 'influence_parent_category_'+h+'_'+name)
            influence[h]['categories'][name] = record
        for b in range(8):
            m = np.repeat(q8,4) == b
            record = dict(bin=b,**group(m,parent,rows[h+'_signed_influence_wave'],rows[h+'_signed_influence_QPSD']))
            native_b = summary['influence']['hypotheses'][h]['source_Q8'][b]
            check(equal_group({k:v for k,v in record.items() if k!='bin'},native_b), 'SOURCE_Q8_influence_'+h+'_'+str(b))
            for category,catmask in categories.items():
                grouped = group(m&catmask,parent,rows[h+'_signed_influence_wave'],rows[h+'_signed_influence_QPSD'])
                check(equal_group(grouped,native_b['categories'][category]), 'SOURCE_Q8_category_'+h+'_'+str(b)+'_'+category)
            influence[h]['source_Q8'].append(record)
        check(np.array_equal(rows[h+'_native_Qbin_migrated'],rows[h+'_anchor_native_Qbin'] != rows[h+'_fake_reference_Qbin']),
              'anchor_native_Qbin_migration_identity_'+h)
        check(rows[h+'_reference_support'].all() and rows[h+'_anchor_native_support'].all(), 'actual_and_anchor8bin_support_complete_'+h)
        for view in ('actual','anchor'):
            window = rows[h+'_'+view+'_window_V']
            q, tail = window.sum(1),window[:,46:].sum(1)
            check(near(q,rows[h+'_'+view+'_Q']) and near(tail,rows[h+'_'+view+'_tail']), 'saved_window_Q_tail_sums_'+h+'_'+view)
            for term in ('Q','tail'):
                prefix = h+'_'+view+'_'+term
                residual = rows[prefix]-sum(rows[prefix+'_'+part] for part in ('signal','noise','ADC_delta'))
                err = maxerr(residual,0)
                decomposition_errors[prefix] = err
                check(err <= 1e-10 and near(residual,rows[prefix+'_identity_residual']), 'three_component_prefix_corrected_'+prefix)
                idx = rows[h+'_'+view+'_crop_start'][:,None]+np.arange(256)[None,:]
                sigwin = np.take_along_axis(S,idx,axis=1)-S[:,:100].mean(1)[:,None]
                sigterm = sigwin.sum(1) if term == 'Q' else sigwin[:,46:].sum(1)
                check(near(sigterm,rows[prefix+'_signal']), 'direct_saved_S_signal_component_'+prefix)
            check(not rows[h+'_'+view+'_flag_PSD_undefined'].any() and near(tail/q,rows[h+'_'+view+'_PSD_raw']),
                  'defined_raw_PSD_ratio_'+h+'_'+view)
    check(int(union.sum()) == summary['influence']['union_crop_changed_parents'], 'changed_parent_union_not_sum_of_overlapping_hypotheses')
    linearization = {}
    for name in ('wave','QPSD'):
        signed = sum(float(rows[h+'_signed_influence_'+name].sum()) for h in H)
        delta = np.mean([comp[h]['anchor_wave']-comp[h]['actual_wave'] for h in H]) if name == 'wave' else 2*np.mean([comp[h]['anchor_QPSD_fixed']-comp[h]['actual_QPSD'] for h in H])
        linearization[name] = dict(signed_influence_sum=signed, actual_anchor_minus_actual_loss=float(delta), remainder=float(delta-signed))
        record = summary['influence']['linearization'][name]
        check(near(signed,record['signed_influence_sum']) and near(delta,record['true_fixed_membership_anchor_minus_actual_loss'])
              and near(delta-signed,record['linearization_remainder']), 'weighted_full_loss_delta_linearization_remainder_'+name)
    gS = g['wave_shared_S_partial_gradient']+g['QPSD_shared_S_partial_gradient']
    ad_direction = float(np.sum(gS*direction))
    check(near(ad_direction,probe['d_AD_loss_per_V']), 'independent_S_dot_direction_AD_derivative')
    probe_records, cd_records = [], []
    check(len(probe['signed_probes']) == 6 and probe['actual_probe_ADC_rows'] == 36864, 'exact6_signed_probes_and36864_additional_ADC')
    for entry in probe['signed_probes']:
        step,sign = entry['step_LSB'],entry['sign']
        check(entry['defined'] and near(entry['delta_V'],sign*step/16383), 'probe_delta_'+str(step)+'_'+str(sign))
        label = str(step).replace('.','p')+'_'+('minus' if sign < 0 else 'plus')
        # Original float .1 serializes as0.1, ints1/10 serialize without suffix.
        if step in (1,10):label = str(int(step))+'_'+('minus' if sign < 0 else 'plus')
        entry_out = dict(step_LSB=step,sign=sign,delta_V=entry['delta_V'],paths={})
        for path in PATHS:
            detail = entry['paths'][path]; recomputed_proxy = []
            path_out = dict(mean_h_proxy=detail['mean_h_proxy'],hypotheses={})
            for h in H:
                d = detail['hypotheses'][h]
                prefix = 'probe_'+label+'_'+path+'_'+h+'_'
                changed = rows[prefix+'peak_index'] != rows[h+'_actual_peak_index']
                migration = rows[prefix+'native_Qbin'] != rows[h+'_fake_reference_Qbin']
                check(np.array_equal(changed,rows[prefix+'peak_changed']) and int(changed.sum()) == d['peak_changed_draws']
                      and np.unique(parent[changed]).size == d['peak_changed_parents'], 'probe_peak_membership_counts_'+prefix)
                check(int(migration.sum()) == d['native_Qbin_changed_draws'], 'probe_native_bin_migration_'+prefix)
                for name,n in d['flag_counts'].items():
                    check(int(rows[prefix+'flag_'+name].sum()) == n, 'probe_flag_count_'+prefix+name)
                check(near(d['proxy_fixed'],d['wave']+2*d['QPSD_fixed']), 'probe_proxy_weighted_algebra_'+prefix)
                sub = d['wave_subcomponents']
                check(near(d['wave'],.5*sub['swd']+.25*sub['coordinate']+.25*sub['short_lag']), 'probe_wave_subcomponent_weights_'+prefix)
                check(d['W1_absolute_pair_or_cross_domain_cusp'] == 'unknown_not_excluded', 'probe_retains_unmeasured_W1_cusp_'+prefix)
                recomputed_proxy.append(d['wave']+2*d['QPSD_fixed'])
                path_out['hypotheses'][h] = {k:d[k] for k in ('wave','QPSD_fixed','QPSD_native','proxy_fixed','ADC_changed_cells',
                                                            'peak_changed_draws','peak_changed_parents','native_Qbin_changed_draws')}
            check(near(detail['mean_h_proxy'],np.mean(recomputed_proxy)), 'probe_mean3_hypotheses_'+path+'_'+label)
            entry_out['paths'][path] = path_out
        probe_records.append(entry_out)
    for step in (.1,1,10):
        minus = next(x for x in probe_records if x['step_LSB'] == step and x['sign'] == -1)
        plus = next(x for x in probe_records if x['step_LSB'] == step and x['sign'] == 1)
        native_cd = next(x for x in probe['central_differences'] if x['step_LSB'] == step)
        paths = {path:(plus['paths'][path]['mean_h_proxy']-minus['paths'][path]['mean_h_proxy'])/(2*step/16383) for path in PATHS}
        for path,value in paths.items():
            check(near(value,native_cd['paths'][path]['central_difference_loss_per_V'],1e-10), 'probe_central_difference_'+str(step)+'_'+path)
        difference = paths[PATHS[2]]-paths[PATHS[1]]
        check(near(difference,native_cd['moving_minus_fixed_central_difference'],1e-10), 'probe_moving_minus_fixed_CD_'+str(step))
        cd_records.append(dict(step_LSB=step,d_AD_loss_per_V=ad_direction,paths=paths,moving_minus_fixed=difference))
    # With fixed members the H paths must match pointwise wherever moving membership did not change.
    fixed_moving_errors = {}
    for entry in probe_records:
        step,sign = entry['step_LSB'],entry['sign']
        label = (str(step).replace('.','p') if step == .1 else str(int(step)))+'_'+('minus' if sign<0 else 'plus')
        for h in H:
            f,m = ('probe_'+label+'_'+path+'_'+h+'_' for path in PATHS[1:])
            unchanged = rows[m+'peak_index'] == rows[h+'_actual_peak_index']
            for name in ('Q','tail','PSD','global_peak_index'):
                err = maxerr(rows[f+name][unchanged],rows[m+name][unchanged]) if unchanged.any() else 0
                fixed_moving_errors[label+'_'+h+'_'+name] = err
                check(err <= 1e-12, 'shared_hard_ADC_same_member_Q_tail_PSD_'+label+'_'+h+'_'+name)
    check(not any(x in sys.modules for x in ('torch','n01')), 'readback_imported_only_stdlib_and_NumPy')
    failed = [x for x in checks if not x['passed']]
    parent_random_mean = R.reshape(512,4,384).mean(1)
    receipt = dict(
        schema='n01/mechanism012-independent-saved-output/v1',run_id='mechanism012_20261015',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS' if not failed else 'FAIL',passed=not failed,
        script_sha256=sha(Path(__file__)),collection_receipt=dict(path=str(collection_path),sha256=sha(collection_path)),
        archive=dict(path=str(archive),sha256=collection['archive_sha256'],bytes=archive.stat().st_size),
        inventory=inventory,checks=checks,failed_checks=failed,
        native=dict(exit_code=0,supervisor_pid=native['supervisor_pid'],child_pid=native['child_pid'],
                    finished_UTC=native['finished_UTC'],elapsed_seconds=native['elapsed_seconds'],
                    sampled_RSS_GiB=native['peak_sampled_RSS_bytes']/1024**3,
                    sampled_GPU_GiB=native['peak_sampled_GPU_bytes']/1024**3,output_bytes=native['peak_output_bytes']),
        operation_accounting=dict(unique_entered=432,unique_completed=432,identity_and_prefix_counts_checked=True,
                                  counters=complete['counters'],scalar_query_names=scalar,masked_VJP_query_names=vjp),
        SHA_chain=dict(source_manifest_sha256=sha(COLLECTED/'MECHANISM012_SOURCE_MANIFEST.json'),bindings_sha256=sha(COLLECTED/'MECHANISM012_BINDINGS.json'),
                       policy_sha256=sha(COLLECTED/'mechanism012_policy.json'),projection_fingerprint=bank_hash.hexdigest(),
                       current_source69_unchanged=True,native_saved_inputs_sha_entries=len(complete['inputs_sha256']),
                       native_saved_source_sha_entries=len(complete['sources_sha256'])),
        frozen_state=dict(before_equals_after=state_ok,states=summary['frozen_state_before']['states'],
                          baseline_stage=summary['frozen_state_before']['baseline_stage'],
                          independent_checkpoint_read=False,scope='saved digests and native protected restore gates; no new parameter decoding'),
        weighted_loss_values=weighted,total_loss=summary['total_loss'],gradient_vectors=vectors,pairwise_cosines=pairs,
        gradient_sum=dict(L2_error=sum_error,relative_error=sum_relative,sum_component_norms=float(component_norm_sum),
                          cancellation_fraction=float(1-np.linalg.norm(summed)/component_norm_sum),
                          norm_fractions_not_causal_or_additive_contributions=True),
        fixed_membership=dict(reference_value_abs_error=summary['fixed_membership_reference']['value_abs_error'],
                              reference_vector_max_abs_error=maxerr(g['QPSD'],g['QPSD_fixed_reference']),
                              anchor_not_A_signal='same current signal+same frozen noise+same ADC+same prefix baseline, only saved A crop member',
                              actual_anchor_native_bin_migration={h:influence[h]['canonical_Qbin_migration_draws'] for h in H},
                              globally_equivalent_objectives_not_established=True),
        S_adjoint_numpy_max_abs_error=adjoint_errors,influence=influence,
        crop_changed_union_parents=int(union.sum()),parent_denominator=512,draws_per_parent=4,
        repeated_draws_not_independent_source_units=True,SOURCE_Q8_parent_counts=np.bincount(q8,minlength=8).tolist(),
        linearization=linearization,Q_tail_decomposition_max_abs_residual=decomposition_errors,
        random_shape=dict(global_RMS_V=float(np.sqrt(np.mean(R**2))),global_max_abs_V=float(np.max(np.abs(R))),
                          max_abs_canonical_Q_residual=float(np.max(np.abs(R[:,110:366].sum(1)))),prefix100_exact_zero=True,
                          parent_4draw_mean_RMS_V=float(np.sqrt(np.mean(parent_random_mean**2))),
                          finite_4draw_parent_means_do_not_test_population_mean_zero=True),
        direction_probe=dict(d_AD_loss_per_V=ad_direction,central_differences=cd_records,signed_probes=probe_records,
                             direction='current saved random-shape voltage direction, not parameter theta perturbation',
                             no_best_step_selection=True,one_direction_three_registered_step_sizes_only=True,
                             unknown_W1_absolute_pair_or_cross_domain_cusps=True),
        scientific_execution_in_this_readback=dict(saved_NPZ_paths_decoded=[str(ART/x) for x in ('GRADIENTS.npz','ROW_MECHANISMS.npz','EVALUATION_PROJECTIONS.npz')],
             original_dataset_reads=0,checkpoint_reads=0,NN_instances=0,NN_forward_rows=0,AD_queries=0,optimizer_created=0,
             sampler_created=0,updates=0,DEV_reads=0,SSH_SCP_calls=0,pipeline_replays=0,new010_or011_evaluations=0),
        limitations=[
            'Checks are saved-array/statistical/linear algebra readback of one fixed TRAIN bank. No new independently collected physical evidence or DEV estimate.',
            'Full proxy W1/CDF loss values were not reevaluated: original selected REAL/statistics/probe waveform banks are not decoded/saved here. Weighted composition and finite differences from SHA-protected scalar outputs are checked.',
            'Native1288 AD and masked VJP correctness cannot be independently regenerated without a model graph; this readback checks the actual recorded11 queries, finite vectors, sum identities and saved-state guards.',
            'S analytic pullback is reconstructed using saved crop/endpoint arrays and native per-row no-out-of-range flags. Original noise/carrier/ADC tensors are not reconstructed.',
            'Signal Q/tail components are independently reconstructed from saved signal; noise/ADC components are checked through saved three-term identities, not new005/noise replay.',
            'SOURCE Q8 uses saved registered strata and parent identities; original raw source carrier and calibration are not reread to refit/re-bin.',
            'Endpoint signed influence is a weighted local first-order linearization, not row loss or a causal per-event decomposition; changed-parent groups across hypotheses overlap.',
            'Probe finite differences are in one saved voltage direction. Quantization, argmax ties, sorting and unmeasured W1 absolute/cross-domain cusps can limit classical derivative interpretation.',
            'Native resource peaks are supervisor samples, not an independently measured continuous maximum.'
        ],
        interpretation=dict(response_match_established=False,formal_response_acceptance=False,
             core='Weighted QPSD dominates the fixed-point head gradient; small regular gradients do not support regularization dominance at this point. The STE voltage derivative matches smooth fixed-member probes approximately, but hard moving-member response changes with registered step and reverses sign at0.1LSB in this single direction.',
             next_step='Use existing saved outputs for a bounded crop-boundary/first-argmax read-only method diagnosis; register any later additional computation separately. No new training/GAN/candidate follows automatically.')
    )
    with target.open('x',encoding='utf8') as f:
        json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(target),sha256=sha(target),status=receipt['status'],total_checks=len(checks),failed_checks=failed,
                         gradient_sum_relative=sum_relative,QPSD_L2=vectors['QPSD']['L2'],total_L2=vectors['total']['L2'],
                         cosine_QPSD_total=vectors['QPSD']['cosine_with_total'],crop_changed_union_parents=int(union.sum()),
                         random_shape_RMS_V=receipt['random_shape']['global_RMS_V'],linearization=linearization,
                         central_differences=cd_records),ensure_ascii=True))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
