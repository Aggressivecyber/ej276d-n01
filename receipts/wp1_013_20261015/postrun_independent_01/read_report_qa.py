"""WP1 report QA against saved audits/JSON only. No arrays or model calls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT / 'receipts/wp1_013_20261015/postrun_independent_01'
JOB = ROOT / 'runs/wp1_013_20261015/runtime_fix01_collected_03/runs/primary'
report = ROOT / 'reports/wp1_013_20261015/N01_WP1_013_REPORT.md'
html = report.with_suffix('.html')


def load(p):
    return json.loads(p.read_text(encoding='utf8'))


def pin(p):
    return dict(path=str(p), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


text = report.read_text(encoding='utf8')
assert pin(report)['sha256'] == 'd8088e081e5fd324df903b14ddb809e32a85fdb8cae34430338a1fab891cdf00'
old = ROOT / 'reports/wp1_013_20261015_before_statistic_definition_01/N01_WP1_013_REPORT.md'
assert pin(old)['sha256'] == 'e81307cad53c8f2234e5f30c273fbda7ec72989822ab4982a81c0133438bbe9f'
definition = '表内重复散布为 sqrt(mean_parent,time(s²_8draw))，MCSE为 sqrt(mean_parent,time(s²_8draw/8))；下述纯信号散布沿用同一口径。'
assert text.replace(definition + '\n\n', '') == old.read_text(encoding='utf8')
checks = []


def ok(value, name):
    assert value, name
    checks.append(dict(name=name, passed=True))


def table(header_start):
    lines = text.splitlines()
    i = next(i for i, s in enumerate(lines) if s.startswith('| ' + header_start + ' |'))
    rows = []
    for line in lines[i + 2:]:
        if not line.startswith('|'):
            break
        rows.append([s.strip() for s in line.strip('|').split('|')])
    return rows


readback = load(OUT / 'NUMERIC_READBACK.json')
decision = load(OUT / 'DECISION_READBACK.json')
arrays = load(OUT / 'REFERENCE_ARRAY_READBACK.json')
numeric_path = ROOT / 'receipts/wp1_013_20261015/postrun_numeric_01/NATIVE_NUMERIC_READBACK.json'
numeric = load(numeric_path)
T = decision['T']
hyp = ('fast', 'slow', 'common_level')
expected = []
for h in hyp:
    point = readback['full_population_saved_distance_algebra']['by_hypothesis'][h]
    a, z = point['A'], point['theta']
    expected.append([h, f'{a["wave"]:.9f}', f'{z["wave"]:.9f}',
                     f'{point["wave_relative_improvement_percent"]:.4f}%',
                     f'{a["QPSD"]:.9f}', f'{z["QPSD"]:.9f}', f'{z["noise"]:.9f}'])
ok(table('假设 | A wave') == expected, 'full_population_table_and_percentages_match_independent_algebra')

expected = []
for h in hyp:
    z = numeric['full_population']['per_hypothesis'][h]['mean_of_two_bank_distances']['theta']
    expected.append([h] + [f'{z[k] / T[h][k]:.3f}' for k in
                          ('peak_voltage_V', 'rise10_90_ns', 'FWHM_ns', 'baseline_mean_V', 'rise_undefined_rate')])
ok(table('假设 | 峰幅距离/T') == expected, 'absolute_morphology_DC_and_undefined_rate_ratios_match_saved_metrics')

repeats = numeric['selected_saved_array_statistics']['same_parent_eight_repeat_statistics']
expected = []
for h in hyp:
    a, z = (next(v for v in repeats if v['arm'] == arm and v['hypothesis'] == h) for arm in ('A', 'theta'))
    expected.append([h, f'{a["hard_wave_sqrt_mean_sample_variance_V"] * 1000:.6f}',
                     f'{z["hard_wave_sqrt_mean_sample_variance_V"] * 1000:.6f}',
                     f'{z["hard_wave_RMS_conditional_mean_MCSE_V"] * 1000:.6f}'])
ok(table('假设 | A 硬波形重复散布/mV') == expected, 'eight_draw_spread_and_MCSE_table')
ok(definition in text, 'variance_RMS_and_MCSE_definition_precision_note_resolved')
ok('样本协方差分母分别3/7；8次经验协方差秩≤7' in text, 'four_eight_draw_denominator_and_sample_rank_are_explicit')
ok('原Y未另存第二份副本' in text and '本轮未展开稠密矩阵' in text,
   'encoding_originalY_and_dense_covariance_audit_limits_explicit')

boundary = load(JOB / 'boundary/BOUNDARY_SAVED_AUDIT.json')
expected = [[str(v['step_LSB'])] + [f'{v["loss_per_V"][k]:.9f}' for k in
            ('S_clamp_surrogate_fixed_actual_crop', 'H_fixed_hard_ADC_fixed_actual_crop',
             'H_moving_same_hard_ADC_first_argmax')] for v in boundary['central_differences']]
ok(table('名义carrier尺度/LSB') == expected, 'all_three_012_carrier_probe_steps_have_correct_signed_central_differences')
ok(boundary['all_six_signed_probes_reported'] and boundary['changed_event_records'] == 95
   and '95条' in text and '原保存A-anchor' in text,
   'all_six_probes_and_95_state_hyp_events_retained_with_original_anchor_scope')
ok('0.1LSB下动态硬窗中心差分反号，1和10LSB恢复负号' in text,
   'sign_reversal_is_limited_to_saved_carrier_direction_smallest_step')

expected = []
for name in ('negative_saved012_total', 'prospective_AdamW2001'):
    cells = decision['direction_decisions'][name]['cells']
    expected.append([name, f'{sum(c["S_delta"] < -1e-12 for c in cells)}/12',
                     f'{sum(c["H_delta"] < -1e-12 for c in cells)}/12',
                     f'{sum(c["adequate"] for c in cells)}/12', f'{sum(c["repair"] for c in cells)}/12'])
ok(table('正向名义1LSB主方向') == expected, 'same_primary_direction_ALL12_not_mixed_direction_selection')
params = load(JOB / 'artifacts/PARAMETER_VALIDATION.json')
opposed = [v for v in params['records'] if v['S_delta_objective'] < -1e-12 and v['H_delta_objective'] > 1e-12]
opposed_ratio = max(max(v['S_H_self_distance'][k] / T[v['hypothesis']][k] for k in ('wave', 'QPSD')) for v in opposed)
ok(opposed_ratio < 1, 'secondary_S_down_H_up_cells_do_not_reach_registered_materiality')

expected = [[h] + [f'{T[h][k]:.9f}' for k in
             ('wave', 'QPSD', 'noise', 'PSD_raw', 'peak_voltage_V', 'rise10_90_ns', 'FWHM_ns')] for h in hyp]
ok(table('假设 | T_wave') == expected, 'all_report_T_values_and_units_match_frozen_table')
ok('f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b' in text,
   'frozen_numeric_table_SHA_explicit')
ok('零/未定义的灵敏度保持not_calibrated' in text and '实际硬件误差校准仍待独立实验' in text,
   'normative_T_and_unknown_zero_sensitivity_limits_preserved')

costs = readback['native']
expected = []
for k in ('old_A_rows', 'head_actual_parameter_forward_rows', 'head_linear_direction_map_rows',
          'head_total_kernel_rows', 'noise005_rows', 'hard_ADC_rows', 'all_path_gather_rows', 'checkpoint_loads'):
    prior = costs['prior_two_failed_attempts_actual_cost'][k]
    assert prior % 2 == 0
    expected.append([k, str(prior // 2), str(prior // 2), str(costs['current_actual_cost'][k]),
                     str(costs['all_three_attempts_actual_cost'][k])])
ok(table('计数') == expected, 'all_three_attempts_actual_costs_not_zeroed')
ok('1185.785秒' in text and 'RSS 7.472 GiB、GPU 8.125 GiB、输出1316.275 MiB' in text,
   'sampled_resource_units_and_wall_match_native_record')

info = load(ROOT / 'INPUT_INFORMATION_AUDIT.json')
ok((ROOT / 'INPUT_INFORMATION_AUDIT.json').read_bytes() == (JOB / 'information/INPUT_INFORMATION_AUDIT.json').read_bytes(),
   'root_information_report_byte_equal_to_collected_producer_output')
ok(all(set(v['explained_variance_dimensions'].values()) == {1} for k, v in info['PCA'].items() if k.startswith('TRAIN_')),
   'reported_TRAIN_normalizations_480_384_256_PCA_dimensions')
train = info['PCA']['TRAIN_carrier384_canonical_Q']
ok(f'{train["total_variance"]:.9g}' in text
   and f'{train["float32_difference_centered_total_variance"]:.9g}' in text
   and info['generator_float32_exact_rows'] == 8192,
   'SOURCE_generator_and_rounding_variance_numbers_are_saved_metadata_not_new_fit')
sameq = info['same_Q_real_input_shape_feasibility']
ok(sameq['rows_with_testable_shape_contrast'] == 6890
   and f'{sameq["largest_verified_strength"]:.8f}' in text
   and info['response_shape_intervention_executed'] is False
   and '响应干预尚未执行' in text,
   'sameQ_SOURCE_information_is_not_response_or_causal_intervention')

semantic_checks = {
    'scientific_acceptance_not_established': 'response_match_not_established' in text and '仍低于 2% 实用门槛' in text,
    'single_acquisition_logical_groups_and_exposed_DEV': '逻辑组不等于独立采集' in text and '已暴露固定DEV' in text,
    'no_DEV_response_or_model_selection': '未对DEV执行响应网络或用于模型选择' in text,
    'fixed_theta_and_saved_gradient_local_scope': '固定G2000系统、已保存梯度和所列有限步' in text,
    'prospective_AdamW_not_an_executed_optimizer_step': '本轮没有执行优化器步骤' in text,
    'WP2_future_four_serial_short_trials_are_separate_AD_budget': '每次最多2000G' in text and '新增AD及优化器预算在014单列' in text,
    'hard_evaluation_and_lowQ_failure_rows_retained_for_WP2': '最终评价继续原始硬ADC/first-argmax' in text and '全部低Q/非正Q/失效记录保留' in text,
}
for name, v in semantic_checks.items():
    ok(v, name)

links = re.findall(r'\]\(([^)]+)\)', text)
ok(all((report.parent / p).resolve().is_file() for p in links), 'all_report_local_evidence_links_resolve')
result = {
    'schema': 'n01/wp1-independent-report-QA/v1',
    'UTC': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS', 'passed': True,
    'report': pin(report), 'HTML': pin(html), 'initial_report_preserved': pin(old),
    'independent_numeric_readback': pin(OUT / 'NUMERIC_READBACK.json'),
    'reference_array_readback': pin(OUT / 'REFERENCE_ARRAY_READBACK.json'),
    'other_numeric_readback': pin(numeric_path),
    'checks': checks,
    'resolved_precision_request': definition,
    'secondary_opposing_sign_cells': len(opposed),
    'secondary_opposing_sign_max_endpoint_selfdistance_over_T': opposed_ratio,
    'unresolved_numeric_or_scope_errors': [],
    'verdict_ZH': '现行报告数字、统计口径与证据边界一致；保留013科学未通过结论，允许闭合WP1并进入已登记的C0/C1实现审核。',
    'scope': {
        'saved_JSON_and_report_text_only': True, 'new_array_reads_in_report_QA': 0,
        'scientific_CP_reads': 0, 'NN_rows': 0, 'AD_queries': 0, 'training_updates': 0,
        'original_or_DEV_waveform_reads': 0, 'SSH_calls': 0,
        'full_CDF_or_calibration_replays': 0, 'new_jobs': 0,
        'HTML_visual_rendering_review_performed_here': False,
        'HTML_visual_QA_owner': 'root',
    },
    'interpretation_boundaries': [
        'table numbers are mean of bank distances, not the pooled8 empirical distance',
        'R8 spread is sqrt(mean(sample_variance)); MCSE uses sample_variance/8, not average per-parent SD',
        'sameQ contrast uses actual SOURCE inputs; no response intervention was executed',
        '012 carrier probe direction is not a theta-parameter finite difference; sign reversal is limited to the saved0.1LSB case',
        'saved A-anchor influence is a local linearization, not a causal decomposition or the signed probe effect',
        'no new hardware calibration, independent acquisition, true conditional response target or formal acceptance is asserted',
    ],
}
target = OUT / 'REPORT_QA_FINAL.json'
with target.open('x', encoding='utf8', newline='\n') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps({'status': 'PASS', 'receipt': pin(target), 'report': pin(report), 'check_count': len(checks)}, ensure_ascii=False))
