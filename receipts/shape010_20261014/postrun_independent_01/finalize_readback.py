"""Finalize metadata verification and bounded interpretation; no science calls."""
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('E:/GAN/research/N01')
OWN = ROOT/'receipts/shape010_20261014/postrun_independent_01'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


r = read(OWN/'NUMERIC_READBACK.json')
collector_path = ROOT/'receipts/shape010_20261014/collection_01/COLLECTION_RECEIPT.json'
c = read(collector_path)
archive, collection = Path(c['archive']), Path(c['collected_root'])
assert sha(archive) == c['archive_sha256'] and archive.stat().st_size == c['archive_bytes']
members = {}
with tarfile.open(archive, 'r:') as tf:
    for m in tf.getmembers():
        assert m.isfile() and m.name not in members and not m.name.startswith('/') and '..' not in Path(m.name).parts
        stream = tf.extractfile(m)
        h = hashlib.sha256()
        for chunk in iter(lambda:stream.read(2**20), b''):
            h.update(chunk)
        local = collection/m.name
        assert local.is_file() and local.stat().st_size == m.size and sha(local) == h.hexdigest()
        members[m.name] = dict(bytes=m.size, sha256=h.hexdigest())
assert len(members) == c['file_count'] == 39
assert set(str(p.relative_to(collection)).replace('\\', '/') for p in collection.rglob('*') if p.is_file()) == set(members)
for name, expected in c['control_sha256'].items():
    assert sha(collection/name) == expected
source = read(ROOT/'SHAPE010_SOURCE_MANIFEST.json')
for name, digest in source['files'].items():
    assert sha(ROOT/name) == digest == c['frozen_source_sha256'][name]
for name, digest in source['protected_manifests'].items():
    assert sha(ROOT/name) == digest
    for relative, expected in read(ROOT/name)['files'].items():
        assert sha(ROOT/relative) == expected
bound = read(ROOT/'SHAPE010_BINDINGS.json')
for slot, path in [('dataset', ROOT/'data/window008/DATASET008.npz'),
                   ('statistics', ROOT/'data/window008/FIT_STATISTICS008.json'),
                   ('data_statistics', ROOT/'data/window008/FIT_DATA_STATISTICS.json')]:
    assert sha(path) == bound['files'][slot]['sha256']
assert c['immutable_input_count'] == 7 and c['immutable_inputs_unchanged'] is True
assert {k:v['sha256'] for k,v in c['immutable_inputs'].items()} == source['bound_inputs']
assert c['job']['counters'] == r['metadata_chain']['counters']
initial_rms = r['random_shape_pure_signal_readback']['shape010_initial']['actual_all_draw_RMS_V']
final_rms = r['random_shape_pure_signal_readback']['shape010_final']['actual_all_draw_RMS_V']
proxy_a = sum(v['wave_plus_2QPSD_proxy']['A'] for v in r['initial_final_direction'].values())/3
proxy_final = sum(v['wave_plus_2QPSD_proxy']['final'] for v in r['initial_final_direction'].values())/3

interpretation = dict(schema='n01/shape010-independent-interpretation/v1',
    UTC=datetime.now(timezone.utc).isoformat(), run_id='shape010_20261014',
    numerical_readback=dict(path=str(OWN/'NUMERIC_READBACK.json'), sha256=sha(OWN/'NUMERIC_READBACK.json')),
    conclusion='Same fixed TRAIN bank shows modest joint wave/QPSD improvement over frozen A; nonzero random voltage shape remains. Scientific response match is not established.',
    conclusion_zh='同一固定 TRAIN bank 上，终点相对冻结 A 的 wave 与 QPSD 均小幅改善；随机电压形状仍非零。尚未建立科学响应匹配。',
    completed_budget=dict(G=2000, attempted_G=2000, native_exit=0,
        all_source_TRAIN=6593, all_REAL_TRAIN=21047, D=0, noise=0, population=0, C2ST=0,
        DEV_numeric_read=False, additional_training_or_arm=False),
    matched_comparison=dict(source_parents=512, repeats=4, rows=2048, REAL_reference=2048,
        REAL_REAL_control=2048, disjoint=True, same_latent_noise_projection_semantics=True,
        uncertainty_unit='512 source-parent clusters; 2048 are repeated response draws, not 2048 independent source events',
        distances=r['initial_final_direction'],
        joint_proxy=dict(definition='mean_h(wave+2*QPSD), without regular/noise terms',
            A=proxy_a, final=proxy_final, relative_change_percent=100*(proxy_final/proxy_a-1)),
        REAL_REAL_empirical_reference=r['recomputed_REAL_REAL']['components'],
        REAL_REAL_is_acceptance_threshold=False, direct_comparison_to_009_or_old_D=False),
    random_voltage_shape=dict(initial_all_draw_time_RMS_V=initial_rms,
        final_all_draw_time_RMS_V=final_rms, relative_change_percent=100*(final_rms/initial_rms-1),
        fixed_bank_shape_strength_decreased=True, complete_functional_collapse=False,
        parameter_whitelist=['basis_head.weight','basis_head.bias'], trained_parameters=1288,
        parameter_norm_is_not_waveform_activity_proof=True,
        frozen_A_voltage_signal_identical_across_repeats=True, pure_signal_prefix_exact_zero=True,
        canonical_random_Q_max_abs_Vns=r['random_shape_pure_signal_readback']['shape010_final']['canonical_random_Q_absmax_Vns'],
        canonical_signal_Q_identity_max_abs_Vns=r['random_shape_pure_signal_readback']['shape010_final']['canonical_signal_Q_difference_absmax_Vns'],
        conditional_mean_pure_signal=r['random_shape_pure_signal_readback']['shape010_final'],
        explanation='The source construction has zero conditional preADC mean. Four saved draws have a finite sample mean; its MC-variance-corrected parent mean-square interval includes zero.',
        basis8_or_tanh_saturation_not_measured=True),
    observed_mean_and_crop=dict(
        paired_final_vs_A=r['paired_parent_statistics']['shape010_final_minus_A'],
        explanation='After ADC/first-argmax/cropping, corrected parent waveform mean-square differences are positive in the 500 descriptive parent-bootstrap intervals for all hypotheses. Pure-signal zero canonical Q/mean does not force zero observed waveform/Q/PSD change.',
        global_paired_Q_and_PSD_mean_intervals_include_zero_all_hypotheses=True,
        global_mean_cancellation_not_a_causal_test=True,
        no_unique_attribution_to_ADC_vs_peak_vs_crop=True,
        no_mean_path_sole_cause_claim=True),
    numerical_reproducibility=dict(independent_NumPy_wave_QPSD_RR=True,
        max_absolute_component_or_wave_subcomponent_error=r['maximum_distance_recompute_absolute_error'],
        same_crop_Q_decomposition_max_error_Vns=r['maximum_same_crop_Q_identity_error_Vns'],
        native_vs_local_projection_fingerprint_byte_exact=r['common_bank']['projection_fingerprint_byte_exact'],
        cross_CPU_shape_rng_max_absolute_difference=r['common_bank']['shape_stream_recreation_max_error'],
        interpretation='Cross-CPU float32 normal/direction rounding prevented byte-identical RNG/projection reconstruction; numeric loss error is below 4e-10 and the saved native arrays were never replaced.',
        fake_baseline100_and_noise_carriers_not_saved=True,
        full_electronics_array_replay_completed=False, noise_component_independently_recomputed=False,
        native_shared_prefix_guard_frozen005_hash_and_same_crop_noise_checks=True,
        native_reported_noise_component_A_initial_final=0.13848853836634592),
    evidence_bounds=dict(statistical_bootstrap='500 pure NumPy resamples of 512 saved parent clusters; only descriptive mean/RMS/MS/Q/PSD summaries, no whole-loss bootstrap or hyperparameter selection',
        independent_acquisition_uncertainty='unknown', no_predefined_scientific_pass_threshold=True,
        TRAIN_improvement_is_scientific_response_acceptance=False,
        no_GAN_effect_estimated=True, frozen_A_parameters_stage_and_noise005_unchanged=True,
        original_old_weights_not_loaded=True, no_new_model_forwards_gradients_or_optimizer=True),
    next_action='Preserve and close this 2000-G candidate at its registered budget. Pause existing N01 follow-up after final reporting; do not automatically launch a GAN arm or extend training. Any further controlled objective attribution or comparison needs a separately registered bounded plan.',
    scientific_status='response_match_not_established')
write(OWN/'INTERPRETATION.json', interpretation)

validation = dict(schema='n01/shape010-independent-readback-validation/v1',
    UTC=datetime.now(timezone.utc).isoformat(), status='passed_independent_local_readback_with_explicit_numeric_and_saved_array_limits',
    archive=dict(path=str(archive), sha256=sha(archive), bytes=archive.stat().st_size, members=members,
        all_39_archive_member_and_local_SHA_bytes_equal=True),
    collection_receipt=dict(path=str(collector_path), sha256=sha(collector_path)),
    COMPLETE_inventory_29_verified=True, source_53_and_protected008_009_SHA_after_analysis_verified=True,
    local_dataset_and_TRAIN_stat_JSON_SHA_after_analysis_verified=True,
    old_weight_byte_reads=0, new_head_thin_checkpoint_reads=3, new_response_noise_NN_instances=0,
    model_forwards=0, gradient_calls=0, optimizer_updates=0, DEV_or_validation_numeric_reads=0,
    CPU_threads=2, interop_threads=1, temps='E:/GAN/runtime_tmp',
    native_job_pid_chain=dict(supervisor=c['job']['supervisor_pid'], child=c['job']['child_pid']),
    frozen_state_SHA_chain=r['metadata_chain']['frozen_state_sha256'],
    NUMERIC_READBACK=dict(path=str(OWN/'NUMERIC_READBACK.json'), sha256=sha(OWN/'NUMERIC_READBACK.json')),
    INTERPRETATION=dict(path=str(OWN/'INTERPRETATION.json'), sha256=sha(OWN/'INTERPRETATION.json')),
    own_analysis_first_attempts_preserved=dict(binding_CUDA_runtime_guard_extension=True,
        cross_CPU_RNG_exactness_assumption=True, native_experiment_failed=False,
        explanation='Only independent reviewer assumptions were repaired. Original scientific outputs and frozen sources were unchanged.'),
    scientific_status='response_match_not_established')
write(OWN/'READBACK_VALIDATION.json', validation)
print(json.dumps({'status':validation['status'],
    'numeric':validation['NUMERIC_READBACK'], 'interpretation':validation['INTERPRETATION'],
    'validation_sha256':sha(OWN/'READBACK_VALIDATION.json'),
    'archive_members':len(members), 'joint_proxy_relative_change_percent':interpretation['matched_comparison']['joint_proxy']['relative_change_percent'],
    'random_signal_RMS_relative_change_percent':interpretation['random_voltage_shape']['relative_change_percent']}))
