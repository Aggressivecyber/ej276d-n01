"""Local JSON/source/inventory closeout; never open NPZ/checkpoints or import NumPy."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib
import json
import math
import subprocess
import tarfile

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT / 'receipts/wp2_014_20261015/postrun_independent_02'
COLLECTED = ROOT / 'runs/wp2_014_20261015/independent_readback_collected_01'
ART = COLLECTED / 'artifacts'
PREVIOUS = ROOT / 'receipts/wp2_014_20261015/postrun_independent_01'
COLLECTION = ROOT / 'receipts/wp2_014_20261015/readback_collection_01'
REGISTRATION = ROOT / 'receipts/wp2_014_20261015/readback_registration_fix01'
SELECTED = ROOT / 'runs/wp2_014_20261015/selected_collected_01/runs/primary/artifacts'
CHECKS = []
EVIDENCE = {}


def require(condition, name):
    if not condition:
        raise AssertionError(name)


def checked(name, condition):
    require(condition, name)
    CHECKS.append(dict(name=name, passed=True))


def read(path):
    path = Path(path)
    require(path.suffix.lower() not in {'.npz', '.npy', '.pt', '.pth', '.ckpt'},
            'closeout forbids numerical array and checkpoint reads')
    return path.read_bytes()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def document(path):
    path = Path(path)
    data = read(path)
    EVIDENCE[str(path.relative_to(ROOT)).replace('\\', '/')] = dict(bytes=len(data), sha256=sha(data))
    return json.loads(data)


def same_float(a, b):
    return math.isclose(a, b, rel_tol=2e-14, abs_tol=2e-15)


def main():
    require(not (OUT / 'CLOSEOUT_REVIEW.json').exists(), 'fresh closeout output')
    reg = document(REGISTRATION / 'READBACK_EXECUTION.json')
    commit = document(REGISTRATION / 'SOURCE_COMMIT_READBACK.json')
    scope = document(PREVIOUS / 'READBACK_SCOPE_NOTE.json')
    ready = document(PREVIOUS / 'READBACK_WORKER_READY.json')
    master = document(ROOT / 'MASTER_EXECUTION_PLAN.json')
    plan_hash = sha(read(ROOT / 'USER_MASTER_PLAN_20261015.txt'))
    cfg = document(ROOT / 'wp2_014_config.json')
    document(ROOT / 'WP2_SOURCE_MANIFEST.json')
    checked('latest_registration_and_source_commit_exact',
            sha(read(REGISTRATION / 'READBACK_EXECUTION.json')) == 'cee1b002cc773acd86e470b6d158f144a7bc55b94b9d0d2139bd70fb18a41fe9'
            and commit['source_commit'] == 'cde0b8db49c49d7beab23011cb5e95f5627cee6d'
            and plan_hash == '56190995374eb2a765a3c8589736514154dba7ff33aa1309b8b1de493e846ebb')
    checked('scope_note_and_worker_READY_actual_bytes_pinned',
            sha(read(PREVIOUS / 'READBACK_SCOPE_NOTE.json')) == 'ad1e3af96fee90a7b9bd0bf01b43342f4663065869b146d3b0d905b3d0e7f8e1'
            and sha(read(PREVIOUS / 'READBACK_WORKER_READY.json')) == 'b74aef6787d930709c24613a52ece3be2c646102d4805c22817585af6dabd434')
    checked('original_frozen_science_configuration_and_manifest_unchanged',
            sha(read(ROOT / 'wp2_014_config.json')) == reg['original_configuration_sha256']
            == 'abb1de68d5ca312d1dbaf14dc058d7514429321b0813aa00c70ff37b20e92e32'
            and sha(read(ROOT / 'WP2_SOURCE_MANIFEST.json')) == reg['original_source_manifest_sha256']
            == '14db81d3a08a0edede97f544ed8ec4c63147138223c00fc38d833bbc278e2684')
    checked('registered_scientific_readback_budget_all_zero', all(v == 0 for v in reg['scientific_budget'].values()))
    checked('registration_resource_and_scope_literal',
            reg['resource_limits'] == dict(CPU_threads=2, RSS_MiB=1536, virtual_memory_MiB=3072, wall_seconds=1800, output_MiB=64)
            and reg['finite_scope']['bank_pairs'] == 36 and reg['finite_scope']['raw_NPZ'] == 176
            and reg['finite_scope']['waveform_rows_decoded'] == 1898784
            and reg['remote_root'] != reg['original_root'] and reg['automatic_retry'] is False)
    for name, record in reg['files'].items():
        local = read(record['local_path'])
        received = read(COLLECTED / name)
        require(len(local) == len(received) == record['bytes'], 'registered file size ' + name)
        require(sha(local) == sha(received) == record['sha256'], 'registered file SHA ' + name)
    for rel, record in commit['files'].items():
        data = read(ROOT / rel)
        raw = subprocess.run(['git', '-C', str(ROOT), 'cat-file', 'blob', commit['source_commit'] + ':' + rel],
                             check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        require(len(raw) == len(data) == record['bytes'] and sha(raw) == sha(data) == record['sha256'], 'Git source blob ' + rel)
    checked('five_collected_sources_and_six_actual_Git_blobs_match_registration', True)
    checked('prepare_only_path_repair_did_not_change_science_scope_or_start_previous_worker',
            reg['engineering_repair']['prior_worker_start_called'] is False
            and reg['engineering_repair']['prior_raw_CP_NN_AD_training'] == 0
            and reg['engineering_repair']['finite_scope_and_resources_unchanged'] is True)

    remote = document(COLLECTION / 'REMOTE_ARCHIVE.json')
    receipt = document(COLLECTION / 'COLLECTION_RECEIPT.json')
    transfer = document(COLLECTION / 'COLLECTION_TRANSFER.json')
    collect_native = document(COLLECTION / 'collect_NATIVE.json')
    inventory = remote['inventory']
    actual_names = {str(p.relative_to(COLLECTED)).replace('\\', '/') for p in COLLECTED.rglob('*') if p.is_file()}
    checked('19_collected_small_files_no_missing_extra_or_NPZ', actual_names == set(inventory) and len(inventory) == 19
            and not any(name.endswith(('.npz', '.pt')) for name in actual_names))
    for name, record in inventory.items():
        data = read(COLLECTED / name)
        require(len(data) == record['bytes'] and sha(data) == record['sha256'], 'collection SHA ' + name)
    archive = COLLECTION / 'readback_output_01.tar'
    archive_bytes = read(archive)
    checked('archive_SHA_native_collection_and_transfer',
            sha(archive_bytes) == remote['archive_sha256'] == receipt['archive_sha256']
            == '2d14a16ca34eb7d629f23b2b57016ab8c141b4c0d5f810a242c7e53e43403202'
            and len(archive_bytes) == remote['archive_bytes'] == 573440
            and transfer['native_scp_exit_code'] == collect_native['exit_code'] == 0
            and receipt['partial_evidence_only'] is False and receipt['original_raw_NPZ_transferred'] is False)
    with tarfile.open(archive, 'r') as tf:
        members = tf.getmembers()
        require(len(members) == 19 and len({m.name for m in members}) == 19, 'tar member unique count')
        for member in members:
            require(member.isfile() and member.name in inventory and '..' not in PurePosixPath(member.name).parts
                    and not member.name.startswith('/'), 'tar regular safe member')
            content = tf.extractfile(member).read()
            require(len(content) == inventory[member.name]['bytes'] and sha(content) == inventory[member.name]['sha256'],
                    'tar member SHA ' + member.name)
    checked('all19_tar_and_collected_bytes_verified_without_extraction', True)

    array = document(ART / 'ARRAY_READBACK.json')
    complete = document(ART / 'COMPLETE.json')
    status = document(ART / 'status.json')
    first = document(ART / 'FIRST_PAIR_PROOF.json')
    start = document(COLLECTED / 'START.json')
    launch = document(COLLECTED / 'supervisor/launch.json')
    exit_receipt = document(COLLECTED / 'supervisor/exit_receipt.json')
    resource = document(COLLECTED / 'supervisor/resource_status.json')
    dispatch = document(COLLECTED / 'DISPATCH_ATTEMPT.json')
    checked('actual_native_PID_chain_first_pair_and_terminal_identity',
            start['supervisor_pid'] == launch['supervisor_pid'] == exit_receipt['supervisor_pid'] == 137650
            and launch['child_pid'] == exit_receipt['child_pid'] == first['pid'] == complete['pid'] == status['pid'] == array['pid'] == 137651
            and start['boot_id'] == '75229583-9bb2-4225-9f0d-12e04e1c1726'
            and start['registration_sha256'] == dispatch['registration_sha256'] == sha(read(REGISTRATION / 'READBACK_EXECUTION.json'))
            and first['actual_saved_array_pair_completed'] is True and first['completed_pair_count'] == 1
            and first['operations_completed'] == first['operations_entered'] == 181
            and first['no_extra_tiny_actual_input'] is True and first['actual_new_CP_NN_AD'] == 0)
    checked('sidecar_output_outside_original_root_and_original_readonly_dispatch_proof',
            launch['command'][-1] == reg['remote_root'] + '/artifacts'
            and launch['command'][launch['command'].index('--run-root') + 1] == reg['original_root']
            and not reg['remote_root'].startswith(reg['original_root'] + '/')
            and dispatch['original_readonly_proof'] == dict(original95_verified=True,
                original_native_complete_verified=True, raw_npz_hash_passes_by_controller=0, checkpoint_reads_by_controller=0))
    checked('native0_complete_no_failure_inflight_or_truncated_journal',
            exit_receipt['child_native_exit_code'] == exit_receipt['exit_code'] == 0
            and exit_receipt['reason'] is None and exit_receipt['error'] is None
            and exit_receipt['unknown_inflight_ids'] == [] and exit_receipt['failed_ids'] == []
            and exit_receipt['truncated_tail'] is False and not (ART / 'FAILED.json').exists()
            and complete['native_status'] == 'COMPLETE' and complete['passed'] is True
            and array['status'] == 'PASS' and array['passed'] is True
            and complete['ARRAY_READBACK_sha256'] == sha(read(ART / 'ARRAY_READBACK.json')))
    expected = dict(raw_files_SHA_checked=176, raw_bytes_SHA_read=3077281496, hard_rows_files_decoded=72,
                    report_npz_files_decoded=72, baseline_npz_files_decoded=16, probe_raw_npz_files_decoded=8,
                    probe_gradient_npz_files_decoded=8, waveform_rows_decoded=1898784, bank_pairs_verified=36,
                    probe_snapshots_verified=8)
    for doc in [complete, status, array]:
        require(doc['counters'] == doc['attempted_counters'] == expected and doc['inflight'] == {}
                and doc['operations_entered'] == doc['operations_completed'] == 308 and doc['artificial_only'] is False
                and doc['scientific_CP_NN_AD_training'] == 0,
                'terminal exact actual counters')
        require(doc['config_sha256'] == reg['original_configuration_sha256']
                and doc['source_manifest_sha256'] == reg['original_source_manifest_sha256']
                and doc['readback_script_sha256'] == reg['files']['readback_saved_arrays.py']['sha256']
                and doc['pure_kernel_sha256'] == reg['files']['saved_array_kernels.py']['sha256']
                and doc['inventory_sha256'] == reg['files']['BOUND_ORIGINAL_INVENTORY.json']['sha256'], 'terminal source binding')
    checked('actual36pairs_176NPZ_1898784rows_8probes_exact_terminal_counters_and_bindings', True)
    checked('reported_resource_usage_within_registered_limits',
            exit_receipt['resource_limits'] == reg['resource_limits'] and array['resources']['CPU_threads'] == 2
            and max(exit_receipt['peak_owned_RSS_bytes'], array['resources']['peak_RSS_bytes']) < 1536 * 1024**2
            and exit_receipt['elapsed_seconds'] < 1800 and exit_receipt['peak_worker_artifact_bytes'] < 64 * 1024**2)

    journal_data = read(ART / 'operations.jsonl')
    require(journal_data.endswith(b'\n'), 'journal final newline')
    entered, completed_ops, seen_entered, seen_completed = {}, {}, set(), set()
    for line in journal_data.splitlines():
        operation = json.loads(line)
        event, index = operation['event'], operation['operation_id']
        require(event in {'entered', 'completed'}, 'no failed journal events')
        if event == 'entered':
            require(index not in entered, 'unique entered ID')
            entered[index] = operation
            seen_entered.add(index)
        else:
            require(index in seen_entered and index not in seen_completed, 'matching once-only completion after entry')
            completed_ops[index] = operation
            seen_completed.add(index)
    require(set(entered) == set(completed_ops) == set(range(308)), '308 complete gap-free operation IDs')
    charges = Counter()
    names = Counter()
    details = defaultdict(list)
    for index, item in entered.items():
        other = completed_ops[index]
        require(all(item[k] == other[k] for k in ['operation_id', 'name', 'charges', 'details']), 'same operation identity ' + str(index))
        charges.update(item['charges'])
        names[item['name']] += 1
        details[item['name']].append(item['details'])
    checked('308_durable_operations_unique_identity_charges_and_no_unknown_completions',
            dict(charges) == expected and dict(names) == dict(raw_NPZ_SHA=176, baseline100_decode=16,
              two_bank_pair_verified=36, hard_bank_saved_statistics=72, probe_saved_vectors_and_rows=8)
            and exit_receipt['entered_counts'] == exit_receipt['completed_counts'] == expected)
    original_inventory = document(COLLECTED / 'BOUND_ORIGINAL_INVENTORY.json')
    original_npz = {k: v for k, v in original_inventory['omitted_remote_inventory'].items() if k.endswith('.npz')}
    checked('raw_SHA176_saved_results_match_exact_original_inventory_and_journal_paths',
            array['raw_files_SHA_checked'] == original_npz and len(original_npz) == 176
            and {x['path'] for x in details['raw_NPZ_SHA']} == set(original_npz)
            and sum(x['bytes'] for x in original_npz.values()) == expected['raw_bytes_SHA_read'])
    trials = ['seed0_C0', 'seed0_C1', 'seed1_C0', 'seed1_C1']
    hypotheses = ['fast', 'slow', 'common_level']
    wanted_pairs = {(trial, snapshot, arm, hyp) for trial in trials for snapshot, arms in
                    [('initial_evaluation', ['A', 'initial']), ('final_evaluation', ['final'])]
                    for arm in arms for hyp in hypotheses}
    pair_keys = [(p['trial'], p['snapshot'], p['arm'], p['hypothesis']) for p in array['pairs']]
    checked('all36_registered_trial_snapshot_arm_hyp_pairs_once',
            len(pair_keys) == len(set(pair_keys)) == 36 and set(pair_keys) == wanted_pairs
            and {tuple(x[k] for k in ['trial', 'snapshot', 'arm', 'hypothesis']) for x in details['two_bank_pair_verified']} == wanted_pairs)
    wanted_banks = {(*key, bank) for key in wanted_pairs for bank in [0, 1]}
    checked('72_bank_decode_identities_and_hashes_16_baselines',
            {tuple(x[k] for k in ['trial', 'snapshot', 'arm', 'hypothesis', 'bank']) for x in details['hard_bank_saved_statistics']} == wanted_banks
            and len(array['decoded_window_hashes']) == 72 and len(array['actual_baseline_hashes']) == 16)
    for pair in array['pairs']:
        require(pair['parents'] == 6593 and pair['rows_per_bank'] == 26372
                and pair['r4_native_moments_verified'] is True and pair['r8_derived_from_saved4plus4'] is True
                and pair['r8_native_report_exists'] is False and pair['covariance_denominator'] == 7
                and pair['waveform_rank_upper_bound'] == 7 and pair['full_scalar_covariance_shape'] == [6593, 9, 9]
                and pair['pairwise_finite_joint_counts_verified'] is True
                and pair['dense256x256_covariance_materialized'] is False, 'r4/r8 interpretation')
        require(len(pair['r4_by_bank']) == 2 and pair['IEEE_correction_absmax_V'] <= 4e-15, 'pair exact decoder and MCSE report')
        require(same_float(pair['hard_wave_RMS_conditional_mean_MCSE_V'] * math.sqrt(8),
                           pair['hard_wave_sqrt_mean_sample_variance_V']), 'r8 MCSE denominator')
        for bank in pair['r4_by_bank']:
            require(same_float(bank['waveform_MCSE_RMS_V'] * 2, bank['waveform_spread_RMS_V']), 'r4 MCSE denominator')
    checked('r4_native_verification_r8_new_derived_moments_factor_and_pairwise_nan_contract', True)
    checked('all8_original_probes_78strata_234storedAD_each_no_new_AD',
            {(p['trial'], p['G_updates']) for p in array['probes']} == {(trial, g) for trial in trials for g in [0, 2000]}
            and len(array['probes']) == 8 and all(p['stratum_records'] == 78 and p['stored_AD_queries'] == 234
              and p['identities_vectors_and_mask_counts_verified'] is True and p['actual_peak_switch_reconstruction'] is False for p in array['probes']))
    checked('scope_limits_preserved_no_second_Y_loss_or_median_full_preADC_replay',
            all(array[k] is False for k in ['original_Y_second_copy_available',
              'native_complete_wave_QPSD_noise_distances_independently_replayed',
              'original_C1_median_fit_independently_replayed', 'preADC384_and_fullcarrier_independent_readback'])
            and all(v == 0 for v in array['reviewer_operations'].values())
            and array['scientific_acceptance'] is False and array['scientific_status'] == 'response_match_not_established')

    previous = document(PREVIOUS / 'LOCAL_METADATA_READBACK.json')
    selection = document(ROOT / 'receipts/wp2_014_20261015/postrun_engineering_01/DEVELOPMENT_SELECTION.json')
    tolerance = document(ROOT / 'CALIBRATED_TOLERANCES_WP1_013.json')
    checked('previous_independent_JSON_audit_and_frozen_T_pinned',
            sha(read(PREVIOUS / 'LOCAL_METADATA_READBACK.json')) == '5d2c572e6a4e63f4c9a816544cf24eb2a17697a79ec4aaf48fb295229c841be7'
            and sha(read(ROOT / 'CALIBRATED_TOLERANCES_WP1_013.json')) == 'f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b')
    for name in ['observation', 'data_roles', 'acceptance', 'tolerance_table', 'configuration']:
        record = cfg['contracts'][name]
        data = read(record['local_path'])
        require(sha(data) == record['sha256'] and len(data) == record['bytes'], 'frozen science contract ' + name)
    checked('five_applicable_science_contracts_original_bytes', True)
    candidates = {}
    selection_cells = {(x['candidate'], x['seed_index'], x['hypothesis']): x for x in selection['cells']}
    for candidate in ['C0', 'C1']:
        all_wave, cells = [], []
        for seed in [0, 1]:
            trial = f'seed{seed}_{candidate}'
            initial = document(SELECTED / trial / 'initial_evaluation/METRICS.json')
            final = document(SELECTED / trial / 'final_evaluation/METRICS.json')
            for snapshot in ['initial_evaluation', 'final_evaluation']:
                name = f'runs/primary/artifacts/{trial}/{snapshot}/METRICS.json'
                data = read(SELECTED / trial / snapshot / 'METRICS.json')
                record = original_inventory['whole_remote_inventory'][name]
                require(len(data) == record['bytes'] and sha(data) == record['sha256'], 'original native metrics SHA ' + name)
            for hyp in hypotheses:
                T = tolerance['T_by_hypothesis'][hyp]
                bank_A = [initial['metrics'][f'bank{bank}_A_{hyp}']['values'] for bank in [0, 1]]
                bank_F = [final['metrics'][f'bank{bank}_final_{hyp}']['values'] for bank in [0, 1]]
                all_wave.extend(x['wave'] for x in bank_F)
                A_wave = sum(x['wave'] for x in bank_A) / 2
                F_wave = sum(x['wave'] for x in bank_F) / 2
                improvement = (A_wave - F_wave) / A_wave
                positive, pending, violations, absolute_failed = [], [], [], []
                for metric, threshold in T.items():
                    if not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or threshold <= 0:
                        pending.append(metric)
                        continue
                    values = [x.get(metric) for x in bank_A + bank_F]
                    require(all(isinstance(v, (float, int)) and math.isfinite(v) for v in values), 'required metric defined')
                    A_value = sum(x[metric] for x in bank_A) / 2
                    F_value = sum(x[metric] for x in bank_F) / 2
                    delta = F_value - A_value
                    positive.append(metric)
                    if delta > threshold:
                        violations.append(metric)
                    if F_value > threshold:
                        absolute_failed.append(metric)
                    producer = selection_cells[(candidate, seed, hyp)]['components'][metric]
                    require(same_float(producer['T'], threshold) and same_float(producer['final'], F_value)
                            and same_float(producer['A'], A_value) and same_float(producer['delta_final_minus_A'], delta), 'unrounded selector component evidence')
                cell = dict(seed=seed, hypothesis=hyp, relative_wave_improvement=improvement,
                            improvement_percent=100 * improvement, useful_at_least_2percent=improvement >= 0.02,
                            positive_T_components=len(positive), noninferiority_violations=violations,
                            absolute_T_failed_components=absolute_failed, uncalibrated_pending=pending)
                require(same_float(selection_cells[(candidate, seed, hyp)]['relative_wave_improvement_vs_A'], improvement), 'unrounded usefulness evidence')
                cells.append(cell)
        score = sum(all_wave) / 12
        eligible = all(not c['noninferiority_violations'] and c['positive_T_components'] == 15 for c in cells)
        useful = sum(c['useful_at_least_2percent'] for c in cells)
        candidates[candidate] = dict(eligible=eligible, mean12_final_wave=score, useful_cells=useful,
            ALL6_usefulness=useful == 6, absolute_T_all_cells=all(not c['absolute_T_failed_components'] for c in cells), cells=cells)
        require(same_float(selection['candidates'][candidate]['final_wave_score'], score)
                and previous['candidates'][candidate]['eligible'] == eligible
                and previous['candidates'][candidate]['useful_cells'] == useful, 'previous independent selection agrees')
    eligible = [c for c in ['C0', 'C1'] if candidates[c]['eligible']]
    winner = min(eligible, key=lambda c: candidates[c]['mean12_final_wave']) if eligible else 'C0'
    if len(eligible) == 2 and abs(candidates['C0']['mean12_final_wave'] - candidates['C1']['mean12_final_wave']) <= 1e-12:
        winner = 'C0'
    failed_cell = next(c for c in candidates['C1']['cells'] if c['seed'] == 1 and c['hypothesis'] == 'fast')
    checked('unrounded_ALL6_noninferiority_mean12_development_choice_C1_only', winner == selection['development_path'] == 'C1'
            and eligible == selection['eligible_set'] == ['C0', 'C1'] and selection['config_development_selection'] == cfg['development_selection'])
    checked('C1_usefulness5of6_seed1_fast_below2percent_no_round_to_pass',
            candidates['C1']['useful_cells'] == 5 and candidates['C0']['useful_cells'] == 0
            and failed_cell['relative_wave_improvement'] < 0.02
            and same_float(failed_cell['improvement_percent'], 1.9887788589535327)
            and selection['no_near_threshold_retry'] is True)
    checked('absolute_T_failed_weights_not_promoted_scientific_response_unestablished',
            not candidates['C0']['absolute_T_all_cells'] and not candidates['C1']['absolute_T_all_cells']
            and selection['weights_promoted'] is False and selection['scientific_accepted'] is False
            and previous['weights_promoted'] is False and previous['scientific_acceptance'] is False)
    errors = array['numerical_comparison_absmax_errors']
    result = dict(
        schema='n01/wp2-independent-saved-array-closeout/v1', UTC=datetime.now(timezone.utc).isoformat(),
        status='PASS', passed=True, reviewer_role='continuing independent Sol Max reviewer; runtime model/effort identifiers not exposed',
        scope='Local stdlib JSON/source/inventory/journal closeout of the already completed registered CPU saved-NPZ audit; no numerical array reopened.',
        master_plan_snapshot=dict(sha256=sha(read(ROOT / 'MASTER_EXECUTION_PLAN.json')),
            status=master.get('status'), active_scientific_jobs=master.get('active_scientific_jobs'),
            note='Local metadata snapshot is not a new live remote process observation.'),
        source_commit=commit['source_commit'], registration_sha256=sha(read(REGISTRATION / 'READBACK_EXECUTION.json')),
        collection=dict(root=str(COLLECTED), regular_files=19, regular_file_bytes=sum(x['bytes'] for x in inventory.values()),
            archive_bytes=len(archive_bytes), archive_sha256=sha(archive_bytes), all19_independently_SHA_and_size_verified=True,
            raw_NPZ_transferred=False, inventory=inventory),
        actual_native=dict(supervisor_pid=137650, child_pid=137651, child_native_exit=0, supervisor_native_exit=0,
            boot_id=start['boot_id'], elapsed_seconds=exit_receipt['elapsed_seconds'],
            worker_reported_peak_RSS_bytes=array['resources']['peak_RSS_bytes'],
            supervisor_sampled_peak_owned_RSS_bytes=exit_receipt['peak_owned_RSS_bytes'],
            resource_limits=reg['resource_limits'], operation_count=308, operations_by_name=dict(names),
            counters=expected, failed_ids=[], unknown_inflight_ids=[]),
        saved_array_audit=dict(ARRAY_READBACK_sha256=sha(read(ART / 'ARRAY_READBACK.json')),
            completed_pairs=36, raw_NPZ_SHA_checked_by_registered_worker=176, raw_NPZ_bytes_SHA_checked_by_worker=3077281496,
            hard_rows=1898784, parents_per_pair=6593, rows_per_bank=26372, probe_snapshots=8,
            distinct_parent_rows_are_not_1898784_independent_sources=True,
            raw_features='Nine row features Q/T/PSD/morphology/baseline, front/crossing auxiliaries, reconstructible flags and counters recalculated by registered CPU worker.',
            encoding='uint16 ADC crop codes + actual float64 baseline mean + optional IEEE XOR; registered decoder/reencoder verified. No second original in-memory Y copy exists for an independent second bit comparison.',
            r4='Native per-bank four-draw REPORT means/variance/scalar covariance/pairwise counts/MCSE verified from decoded rows.',
            r8='New readback-derived 4+4-draw moments, not a native r8 REPORT or pooled8 distribution distance; unbiased denominator7, waveform factor rank<=7; all9x9 pairwise-finite covariance/MCSE verified, no dense256x256 export.',
            covariance_identity='var8=[3*(var4_bank0+var4_bank1)+2*(mean4_bank0-mean4_bank1)^2]/7; MCSE4=sqrt(var4/4), MCSE8=sqrt(var8/8).',
            maximum_reported_IEEE_correction_V=max(p['IEEE_correction_absmax_V'] for p in array['pairs']),
            numerical_absmax_selected={k:errors[k] for k in ['r8 merge mean', 'r8 merge covariance diagonal',
                'r8 full9x9 complete-case covariance', 'stored total=sum3', 'stored gradient norm', 'stored gradient cosine']},
            full_distance_replay=False, C1_median_fit_recomputed=False, complete_preADC384_reconstructed=False,
            probe_actual_peak_switch_reconstructed=False),
        development_selection=dict(development_codepath=winner, eligible_candidates=eligible, candidates=candidates,
            distances='Frozen producer hard distances with independent saved-JSON unrounded aggregation/decision; no full wave/QPSD/noise loss or CDF replay.',
            weights_promoted=False, scientific_acceptance=False, scientific_status='response_match_not_established',
            no_round_to_pass=True, no_extra_training_or_candidate=True),
        prior_science_cost=dict(native_actual_cost=previous['native_actual_cost'],
            new_G_updates_by_this_audit=0, cumulative_G_updates_unchanged=167667,
            previous_native_training_and_fit_checks_reused_from_SHA_pinned_LOCAL_METADATA_READBACK=True),
        unresolved_engineering_blockers=[],
        limitations=[
            '176 original NPZ remain remote. Their bytes were verified/decoded by the registered frozen CPU worker; this closeout reads returned JSON, source and journals only, not a second NPZ pass.',
            'Full wave/QPSD/noise empirical distances remain frozen-producer evidence plus independent JSON aggregation, not raw distance replay.',
            'C1 fit JSON SHA/21047 TRAIN fit/floors/use chain were checked; original TRAIN median was not refitted.',
            'No full preADC384 or pure-carrier bank was saved for this audit, so canonical signal Q/covariance and complete carrier/ADC flags were not independently reconstructed.',
            'Probe full windows and peak indices were not saved; stored peak-switch booleans/counts were checked, not independently reconstructed.',
            'No second original Y copy exists. Byte-reversible encoding, exact native row statistics and hash lineage do not manufacture that second comparison.',
            '8 draws reduce conditional Monte Carlo error under fixed parents; they do not create additional independent SOURCE parents or independent physical acquisitions.',
            'TRAIN development-path selection and normative T do not establish detector response match or physical acceptance. C1 usefulness ALL6 and absolute T gates fail.'
        ],
        actual_closeout_operations=dict(NumPy_imports=0, Torch_imports=0, NPZ_reads=0, CP_loads=0,
            original_TRAIN_or_DEV_waveform_reads=0, NN_rows=0, AD_queries=0, optimizer_or_sampler_created=0,
            training_updates=0, new_random_draws=0, CDF_or_loss_replays=0, SSH_calls=0),
        evidence=EVIDENCE, checks=CHECKS,
        conclusion_zh='独立保存数组审核已按登记范围完成，库存、身份、可逆编码、r4/r8矩与梯度向量证据闭合。C1仅晋级开发代码路径；用途门5/6且绝对T未全过，不晋级权重，科学响应匹配仍未建立。本次收尾没有重新读取NPZ或启动科学计算。')
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with (OUT / 'CLOSEOUT_REVIEW.json').open('xb') as f:
        f.write(data)
    print(json.dumps(dict(status='PASS', checks=len(CHECKS), path=str(OUT / 'CLOSEOUT_REVIEW.json'),
                         sha256=sha(data), bytes=len(data)), ensure_ascii=False))


if __name__ == '__main__':
    main()
