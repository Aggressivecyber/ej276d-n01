"""Read-only editorial QA using existing receipts; no statistics/model calls."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('E:/GAN/research/N01')
OWN = ROOT/'receipts/shape010_20261014/postrun_independent_01'
REPORT = ROOT/'reports/shape010_20261014/N01_SHAPE010_REPORT.md'
RESULT = ROOT/'receipts/shape010_20261014/FINAL_RESULT_RECEIPT.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


text = REPORT.read_text(encoding='utf-8')
result = read(RESULT)
numeric = read(OWN/'NUMERIC_READBACK.json')
interpretation = read(OWN/'INTERPRETATION.json')
training = read(ROOT/'receipts/shape010_20261014/training_readback_01/TRAINING_READBACK.json')
assert result['report_sha256'] == sha(REPORT)
assert result['response_match_established'] is False and result['scientific_status'] == 'response_match_not_established'
assert result['additional_arm_launched'] is False and result['additional_model_forwards'] == result['additional_optimizer_updates'] == 0
for relative, expected in result['evidence'].items():
    assert sha(ROOT/relative) == expected
checks = {}
rows = []
for h in ('fast', 'slow', 'common_level'):
    values = numeric['initial_final_direction'][h]
    expected = '| '+h+' | '+' | '.join(f'{values[v][metric]:.6f}' for v,metric in
        [('A','wave'), ('final','wave'), ('A','QPSD'), ('final','QPSD')])+' | '+f"{values['wave_plus_2QPSD_proxy']['relative_final_minus_A_percent']:.2f}%"+' |'
    assert expected in text
    rows.append(expected)
checks['table_all_12_distances_and_3_relative_changes'] = dict(passed=True, matched_rows=rows)
proxy = result['joint_proxy']
assert '0.336980 降至 0.318604，改善 5.45%' in text
assert proxy == interpretation['matched_comparison']['joint_proxy'] or all(proxy[k] == interpretation['matched_comparison']['joint_proxy'][k] for k in ('A','final','relative_change_percent'))
assert '代理指标不包含正则或噪声项' in text and '此次没有计算该完整代理指标的父簇置信区间' in text
checks['joint_proxy_mean_definition_and_point_estimate'] = dict(passed=True, values=proxy, CI_not_computed_explicit=True)
assert '全部 2048 个 draw ×384 点的纯信号随机 RMS' in text and '2.842→1.385 mV' in text
assert '逐 draw RMS 的均值为 2.050→0.964 mV' in text
checks['RMS_distinguishes_all_draw_time_RMS_from_average_per_draw_RMS'] = dict(passed=True,
    values=interpretation['random_voltage_shape'], no_new_array_statistics=True)
assert '逐父纯信号均值差的 RMS 平均为 0.496 mV' in text and 'MCSE RMS 平均 0.510 mV' in text
assert '4 次重复没有分辨额外的 preADC 均值漂移' in text
assert '116/112/122 个 draw' in text and '5.47–5.96%' in text
assert '逐父均值差 RMS 平均 0.893 mV' in text and 'MCSE RMS 平均 0.862 mV' in text
assert '[2.71e-7,1.12e-6] V²' in text and '全局 ΔQ 和 ΔPSD 均值区间跨零' in text
checks['parent_MCSE_observed_drift_crop_and_CI_scope'] = dict(passed=True,
    parent_unit=512, repeats=4, whole_distribution_proxy_CI_claimed=False,
    preADC_zero_mean_extended_to_postObs=False, global_cancellation_used_as_acceptance=False)
assert 'REAL–REAL' in text and '不是不可约误差下限' in text and '不直接比较' in text
assert 'fake baseline100 和完整噪声载体未保存' in text
assert '噪声分量依赖 native 配对 prefix 守卫及冻结 hash' in text
assert '没有声称完成这些数组的完整重放' in text
assert '本次通过的是数值读回，不是跨平台逐字节重建' in text
checks['same_bank_RR_cross_CPU_and_missing_baseline_limits'] = dict(passed=True,
    maximum_numeric_loss_recompute_error=numeric['maximum_distance_recompute_absolute_error'],
    shape_rng_max_absolute_difference=numeric['common_bank']['shape_stream_recreation_max_error'],
    missing_fake_baseline_full_replay_not_claimed=True,
    old009_or_D_direct_rank_not_claimed=True)
assert '本轮按 2000 G 固定预算关闭' in text and '下一步优先另行登记' in text
assert '2000 步不是收敛证明' in text and '以后若登记 GAN 因果对照' in text
assert '仅切换 adversarial 项' in text and '科学响应匹配仍需另外证据' in text
checks['bounded_conclusions_and_proposed_next_work'] = dict(passed=True, mean_path_sole_cause_claim=False,
    GAN_effect_estimated=False, additional_training_authorized_by_report=False,
    convergence_or_response_acceptance_claim=False, further_work_requires_preregistration=True)
links = re.findall(r'\[[^\]]+\]\(([^)]+)\)', text)
for link in links:
    assert (REPORT.parent/link).resolve().is_file()
checks['relative_evidence_links'] = dict(passed=True, count=len(links))
notes = []
if '最多 1 个 float32 ULP' in text:
    notes.append(dict(severity='editorial_precision', statement='max_one_float32_ULP',
        finding='Receipt measures maximum absolute shape_z difference 2.384185791015625e-7, not a maximum per-value ULP count.',
        recommended='Use the measured maximum absolute difference and keep the separate projection fingerprint distinction.'))
if '新头权重 L2 范数增加 6.82%' in text:
    notes.append(dict(severity='editorial_precision', statement='new_head_weight_L2',
        finding='6.82376821247668% is the L2 ratio of all 1288 new-head parameters, including bias.',
        recommended='Say new-head parameter L2.'))
receipt = dict(schema='n01/shape010-final-report-QA/v1', UTC=datetime.now(timezone.utc).isoformat(),
    status='passed_with_editorial_precision_notes' if notes else 'passed_final_report_numbers_definitions_and_scope',
    report=dict(path=str(REPORT), sha256=sha(REPORT)), final_result_receipt=dict(path=str(RESULT), sha256=sha(RESULT)),
    checks=checks, scientific_conclusion_blockers=[], editorial_precision_notes=notes,
    review_scope=dict(readonly_report_and_existing_JSON_receipts=True, new_statistics=False,
        array_decodes=0, checkpoints_loaded=0, model_instances=0, model_forwards=0,
        gradients=0, optimizer_updates=0, DEV_reads=0, frozen_source_or_raw_output_edits=0),
    source_receipts=dict(NUMERIC_READBACK=sha(OWN/'NUMERIC_READBACK.json'),
        INTERPRETATION=sha(OWN/'INTERPRETATION.json'), TRAINING_READBACK=sha(ROOT/'receipts/shape010_20261014/training_readback_01/TRAINING_READBACK.json')),
    scientific_status='response_match_not_established')
with (OWN/'REPORT_QA.json').open('x',encoding='utf-8') as f:
    json.dump(receipt,f,indent=2,ensure_ascii=False,allow_nan=False)
    f.write('\n')
print(json.dumps({'status':receipt['status'],'report_sha256':sha(REPORT),
    'QA_sha256':sha(OWN/'REPORT_QA.json'),'check_groups':len(checks),'notes':notes}))
