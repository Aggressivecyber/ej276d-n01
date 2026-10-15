"""Merge independent WP1 saved-output audits. Stdlib only; no metric replay."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT / 'receipts/wp1_013_20261015/postrun_independent_01'
BASE = ROOT / 'runs/wp1_013_20261015/runtime_fix01_collected_03'
ART = BASE / 'runs/primary/artifacts'
EVAL = BASE / 'runs/primary/evaluation'


def load(p):
    return json.loads(p.read_text(encoding='utf8'))


def pin(p):
    return {'path': str(p), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def save(name, obj):
    p = OUT / name
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return pin(p)


expected = {
    'TERMINAL_CHAIN_READBACK.json': '70fab5e16a247663cc6cbf6495b27f5025dcc3daf43f343af72afefc00388435',
    'DECISION_READBACK.json': '510916258eebf9c751cd9416059ba0cda331f1671f8981839a890ed5fbce9343',
    'REFERENCE_ARRAY_READBACK.json': '1fc5ab0f5dde0afa2bfb4f6ee516e0571e729479508f16a493bd44ea1634bcd9',
}
audits = {}
for name, digest in expected.items():
    p = OUT / name
    assert pin(p)['sha256'] == digest, name
    audits[name] = load(p)
    assert audits[name]['passed'] is True, name
t, d, a = (audits[n] for n in expected)

table = ROOT / 'CALIBRATED_TOLERANCES_WP1_013.json'
table_receipt = ROOT / 'receipts/wp1_013_20261015/postrun_root_01/TOLERANCE_FREEZE_RECEIPT.json'
assert pin(table)['sha256'] == 'f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b'
assert table.read_bytes() == (ART / 'CALIBRATED_TOLERANCES.json').read_bytes()
assert load(table)['T_by_hypothesis'] == d['T']
assert load(table_receipt)['table_sha256'] == pin(table)['sha256']

# Algebra on twelve saved point results, not recomputation of twelve distances.
points = {}
for h in ('fast', 'slow', 'common_level'):
    points[h] = {}
    for arm in ('A', 'theta'):
        pairs = [load(EVAL / f'FULL_{b}_{arm}_{h}.json')['population'] for b in (0, 1)]
        assert all(p['fake_rows'] == 26372 and p['real_rows'] == 21047
                   and p['all_population_rows_retained'] is True for p in pairs)
        assert all(p['waveform']['coverage']['conditional_coverage_complete'] is True for p in pairs)
        points[h][arm] = {
            k: sum(p['waveform'][field] for p in pairs) / 2
            for k, field in [('wave', 'wave'), ('QPSD', 'QPSD_native'), ('noise', 'noise_prefix100')]
        }
    points[h]['wave_relative_improvement_percent'] = 100 * (1 - points[h]['theta']['wave'] / points[h]['A']['wave'])
    points[h]['QPSD_relative_change_percent'] = 100 * (points[h]['theta']['QPSD'] / points[h]['A']['QPSD'] - 1)
    assert points[h]['wave_relative_improvement_percent'] < 2

direction_summary = {}
for name, v in d['direction_decisions'].items():
    cells = v['cells']
    direction_summary[name] = {
        'primary_cells': len(cells),
        'adequate_cell_count': sum(c['adequate'] for c in cells),
        'repair_cell_count': sum(c['repair'] for c in cells),
        'adequate_all12': v['adequate_all12'],
        'repair_all12': v['repair_all12'],
        'S_legacy_objective_delta_range': [min(c['S_delta'] for c in cells), max(c['S_delta'] for c in cells)],
        'H_moving_legacy_objective_delta_range': [min(c['H_delta'] for c in cells), max(c['H_delta'] for c in cells)],
    }
assert d['decision'] == 'adequate'
assert d['adequate_directions'] == ['negative_saved012_total']
assert direction_summary['negative_saved012_total']['adequate_cell_count'] == 12
assert direction_summary['prospective_AdamW2001']['adequate_cell_count'] == 4

limitations = [
    '独立读回只处理保存输出；原始 DATASET/SOURCE 波形、科学 checkpoint、DEV 波形均未打开，NN/AD/优化器/采样器/SSH/新作业/新随机库均为零。',
    'T 复算是保存 self-distance 的 max_7case(mean_2bank) 代数；未重做126组原波形距离，也未重算完整CDF。校准原波形未完整保留，不能声称独立重放全部原始距离。',
    '已对265个回收文件逐项独立核SHA和大小；归档tar SHA来自已绑定的collection receipt，本次未再次通读归档tar。',
    '硬窗口编码全部解码、ADC重新编码逐位相同，并与另外保存的Q/tail/PSD/peak/矩核对；没有第二份独立原始Y副本用于再次逐字节比较。原生writer已做与内存Y的逐字节roundtrip。',
    '所有父事件的4/8draw波形方差对角及9×9标量协方差/NaN支持计数已复算；完整波形协方差由实际draw中心化因子可重建，本次没有展开每父256×256稠密矩阵。',
    '8draw样本协方差分母7、秩上限7；不能由此估计或验证高于7的条件协方差秩。重复响应不是新增独立SOURCE或独立物理采集。',
    '两bank经验距离的算术平均与合并8draw再算经验距离不同；本次全人口point表使用前者，8draw只用于重复响应矩。',
    'A约5.4nV残差符合既有1e-7V数值重放门，实际draw逐一保留；末4行布局与残差位置相符，但这不是具体CUDA核或硬件原因的实证。',
    'adequate仅是theta_G2000、预定负总梯度、正向名义1LSB、两个固定TRAIN bank×两个父批次×三个观察假设的有限步证据，不是全部后续训练步的保证，也不是因果或正式响应匹配验收。',
    'CPU/RSS/GPU峰值按监督采样记录核限，不能称连续峰值测量；185旧文件保护及input SHA链不替代新的物理校准或外部采集验证。',
]
interfaces = [
    '013可按原注册问题闭合；两次失败及全部三次实际成本保留，禁止将前两次成本记零。',
    '数值T表已冻结在WP2候选之前；零或未定义T仍不能自动通过，RR不作为容忍度来源或不可约误差下限。',
    'WP2先进入预定C0/C1损失表示有限对照，按主计划冻结种子、初始输出RMS、数据/噪声流、预算和硬评价合同；C2观察代理修复不能由本次adequate直接晋级。',
    '候选保留全部合法TRAIN并按原父身份统计；重复draw不能当新父事件，两个bank不能冒充跨采集确认。',
    'wave每假设至少2%相对A、关键指标非劣、绝对T、统计不确定性和最终独立确认分别记录，不以legacy代理变好代替这些门。',
    '原主计划最多12个短候选；C0/C1对照按两种子执行，教师和最终确认使用另行固定预算。本读回不启动任何候选或GAN。',
]
doc = {
    'schema': 'n01/wp1-independent-combined-postrun/v1',
    'UTC': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS', 'passed': True,
    'run_id': 'wp1_013_20261015',
    'collected_root': str(BASE),
    'audit_type': 'local_saved_output_verification_not_scientific_response_acceptance',
    'evidence': {name: pin(OUT / name) for name in expected},
    'audit_scripts': {p.name: pin(p) for p in OUT.glob('*.py')},
    'frozen_configuration': t['configuration'],
    'frozen_source_manifest': t['source_manifest'],
    'source_commit': t['source_commit'],
    'source_and_historical_protection': {'current_source_files': 84, 'source_commit_blobs': 86,
                                         'old_protected_records_verified': 185},
    'collection': d['collection'], 'archive_inventory_receipt': d['archive_receipt'],
    'archive_sha256_collection_record': load(Path(d['collection']['path']))['archive_sha256'],
    'all_local_inventory_size_and_SHA_independently_verified': True,
    'file_count': d['file_count'], 'total_file_bytes': d['total_file_bytes'],
    'native': {'child_exit': 0, 'supervisor_exit': 0, 'PID_chain': t['PID_chain'],
               'operations_entered': d['operation_count'], 'operations_completed': d['operation_count'],
               'in_flight': 0, 'failed_operations': 0,
               'current_actual_cost': t['current_completed_cost'],
               'prior_two_failed_attempts_actual_cost': t['two_failed_prior_cost'],
               'all_three_attempts_actual_cost': t['cumulative_actual_cost'],
               'resources': t['resources']},
    'calibrated_T': {'file': pin(table), 'freeze_receipt': pin(table_receipt),
                     'byte_equal_to_native_saved_table': True,
                     'all_values_equal_independent_algebra': True,
                     'rule': 'max_7_registered_cases(mean_2bank(self_distance))',
                     'primary': d['T_primary'], 'hardware_calibrated': False,
                     'zero_or_undefined_T_automatically_passes': False},
    'parameter_method': {'defined_states': 30, 'evaluated_states': 30, 'unique_calibration_records': 126,
                         'unique_parameter_path_records': 360, 'primary_direction_cells': 24,
                         'selection': d['decision'], 'adequate_directions': d['adequate_directions'],
                         'unsafe_directions': d['unsafe_directions'], 'primary': direction_summary,
                         'conditional_coverage_incomplete_records': d['conditional_coverage_incomplete_metric_records']},
    'full_population_saved_distance_algebra': {
        'statistic': 'mean_of_two_bank_distances_not_distance_of_pooled_eight_draws',
        'population': {'SOURCE_parents': 6593, 'fake_rows_per_bank': 26372, 'REAL_rows': 21047},
        'by_hypothesis': points, 'wave_2percent_pass': False,
    },
    'saved_array_qa': {'receipt': pin(OUT / 'REFERENCE_ARRAY_READBACK.json'),
                       'actual_A_all_draws_saved_float64': a['raw_A_actual_draws_and_two_banks_not_broadcast'],
                       'source_parent_draw_lineage_verified': a['all_source_parent_draw_lineage_verified'],
                       'summary': a['summary'],
                       'pure_signal': a['pure_signal'],
                       'shape_increment_MC_mean_statistic_exact_definition':
                       'sqrt(mean_bank,parent,time((mean_4draw(theta_minus_A))**2)); not mean of per-parent RMS and not pooled8draw mean RMS',
                       'encoded_windows_checked': len(a['encoded_windows']),
                       'stored_numeric_comparisons': len(a['numerical_crosscheck_absmax_errors']),
                       'all_numeric_comparison_absmax_errors_max': max(a['numerical_crosscheck_absmax_errors'].values()),
                       'covariance_scope': a['complete_covariance_recovery'],
                       'encoding_scope': a['encoding_scope']},
    'reviewer_actual': {'scientific_CP_loads': 0, 'NN_rows': 0, 'AD_queries': 0,
                        'optimizer_or_sampler_created': 0, 'training_updates': 0,
                        'original_dataset_waveform_reads': 0, 'DEV_waveform_reads': 0,
                        'new_random_draws': 0, 'full_scientific_metric_or_CDF_reruns': 0,
                        'SSH_calls': 0, 'new_jobs': 0, 'Torch_imports': 0,
                        'NumPy_threads_max': 2},
    'scientific_status': 'response_match_not_established', 'scientific_acceptance': False,
    'can_close_bounded_013_execution_and_readback': True,
    'unresolved_engineering_blockers_found': [],
    'limitations_and_not_performed': limitations,
    'WP2_interface_limits': interfaces,
}
result = save('NUMERIC_READBACK.json', doc)
interpretation = save('INTERPRETATION.json', {
    'schema': 'n01/wp1-independent-interpretation/v1',
    'UTC': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS', 'readback': result,
    'execution_and_storage': '013已在第三次独立目录原生正常退出并完整回收，保护SHA、预算、操作账、保存数组及方法代数均通过独立读回；前两次失败和费用保留。',
    'method_result': '预定negative_saved012_total正向1LSB在全部12格满足adequate；prospective AdamW仅4格，不据其余方向或步幅事后选择。',
    'science_result': '当前头三假设wave相对A改善约0.916%、0.991%、1.691%，均低于2%有用门；slow QPSD约退步2.66%。方法adequate不等于响应匹配。',
    'storage_result': 'A每draw实际float64保留，约5.41nV重复残差不被解释为物理随机；12组编码重建和全部父的4/8draw矩通过。8draw样本协方差秩最多7。',
    'recommended_next': '闭合013，按主计划登记有限C0/C1损失表示对照。T数值已冻，保留硬观察、全部人口和父身份；训练前另审具体实现和预算。',
    'limitations': limitations, 'next_interfaces': interfaces,
    'training_or_new_candidate_started_by_this_review': False,
})
text = ('013独立保存结果读回通过，可闭合本次有界执行。265文件SHA/大小、29201次操作账、源与历史保护、三次累计成本、T公式、预定ANY方向/ALL12判据以及逐draw保存矩均一致。\n'
        'adequate仅由预定负总梯度在正向1LSB的12格全部满足；prospective AdamW仅4格。三假设wave提升均未达2%，响应匹配仍未建立。\n'
        'A逐draw数值残差约5.41nV已如实保留；12组硬窗编码可重建，4/8draw波形方差与标量协方差通过。未重跑126组原距离或完整CDF，未展开稠密256×256协方差，没有第二原始Y拷贝。\n'
        '本审核CP/NN/AD/训练/DEV波形/SSH/新作业均0。下一步仅支持按主计划登记有限C0/C1对照；本审核不启动训练。\n')
p = OUT / 'CONCLUSION_ZH.txt'
with p.open('x', encoding='utf8', newline='\n') as f:
    f.write(text)
print(json.dumps({'status': 'PASS', 'readback': result, 'interpretation': interpretation,
                  'conclusion': pin(p), 'full_population_points': points}, ensure_ascii=False))
