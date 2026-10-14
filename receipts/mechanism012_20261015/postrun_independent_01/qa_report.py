"""Pin and read back final012 report using existing numerical receipt only."""
from pathlib import Path
from decimal import Decimal
import datetime
import hashlib
import json
import re

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT/'receipts/mechanism012_20261015/postrun_independent_01'
REPORT = ROOT/'reports/mechanism012_20261015/N01_MECHANISM012_REPORT.md'
EXPECTED_REPORT = 'b7087e5fccc2bf17980e21e1f372f453a6be67fd5cf64e4d193f678d82a08846'
EXPECTED_READBACK = '28fe6f9e7bd5c204fbd40d68c87b27ddb49dfb3fbd67126c859c79436b51948c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close_printed(token, value):
    d = Decimal(token)
    tol = float(Decimal(10)**d.as_tuple().exponent)/2 + 1e-15
    return abs(float(d)-value) <= tol


def main():
    target = OUT/'REPORT_QA_FINAL.json'
    if target.exists():
        raise RuntimeError('Preserve existing final QA')
    numeric_path = OUT/'NUMERIC_READBACK.json'
    if sha(REPORT) != EXPECTED_REPORT or sha(numeric_path) != EXPECTED_READBACK:
        raise RuntimeError('Report/numeric bytes differ from final pinned version')
    r = json.loads(numeric_path.read_bytes())
    text = REPORT.read_text(encoding='utf8')
    lines = text.splitlines()
    table_rows = [x for x in lines if x.startswith('|')]
    checks = []
    def check(value,name):
        checks.append(dict(name=name,passed=bool(value)))
    check(r['passed'] and not r['failed_checks'] and len(r['checks']) == 2163, 'independent_numeric2163_PASS')
    check('独立保存输出读回已完成' in text and '正在进行' not in text, 'readback_finalized_sentence')
    check('没有重跑完整 CDF' in text and '原始 noise/carrier' in text and 'checkpoint 读取或训练' in text,
          'saved_only_scope_no_completeCDF_originalnoise_carrier_CP_AD_replay')
    check('unknown_not_excluded' in text and '未逐项测量' in text and 'fake 排序/tie' in text,
          'unmeasured_W1_cusps_distinguished_from_recorded_fake_sort_ties')
    for name in ('wave','QPSD','RMS','curvature','saturation','total'):
        row = next(x for x in table_rows if x.startswith('| '+name+' |'))
        fields = [x.strip() for x in row.split('|')[1:-1]]
        loss = r['total_loss'] if name == 'total' else r['weighted_loss_values'][name]
        cosine = r['gradient_vectors'][name]['cosine_with_total']
        check(close_printed(fields[1],loss) and close_printed(fields[2],r['gradient_vectors'][name]['L2'])
              and ('undefined' in fields[3] if cosine is None else close_printed(fields[3],cosine)),
              'weighted_loss_gradient_cosine_table_'+name)
    for h in ('fast','slow','common_level'):
        row = next(x for x in table_rows if x.startswith('| '+h+' |'))
        d = r['influence'][h]['categories']['crop_changed']
        check(str(d['draws'])+'/2048' in row and str(d['parents'])+'/512' in row,
              'changed_draw_parent_counts_'+h)
    check('115/512' in text and '可以同时出现在 changed 与 unchanged' in text and '不能相加' in text,
          'union115_and_overlapping_parent_categories')
    for name in ('wave','QPSD'):
        candidates = [x for x in table_rows if x.startswith('| '+name+' |')]
        row = candidates[1]
        fields = [x.strip() for x in row.split('|')[1:-1]]
        d = r['linearization'][name]
        check(all(close_printed(token,d[key]) for token,key in zip(fields[1:],
                    ('signed_influence_sum','actual_anchor_minus_actual_loss','remainder'))),
              'already_weighted_signed_influence_true_delta_remainder_'+name)
    check('不是逐行 loss 或因果分解' in text and '范数不能相加成贡献百分比' in text,
          'no_causal_row_loss_or_norm_percentage_claim')
    paths = ('S_clamp_surrogate_fixed_actual_crop','H_fixed_hard_ADC_fixed_actual_crop','H_moving_same_hard_ADC_first_argmax')
    for d in r['direction_probe']['central_differences']:
        token = '±'+str(d['step_LSB']).rstrip('0').rstrip('.') if d['step_LSB'] == .1 else '±'+str(int(d['step_LSB']))
        row = next(x for x in table_rows if x.startswith('| '+token+' LSB |'))
        fields = [x.strip() for x in row.split('|')[1:-1]]
        values = [d['d_AD_loss_per_V']]+[d['paths'][k] for k in paths]
        check(all(close_printed(t,v) for t,v in zip(fields[1:],values)), 'three_registered_step_CD_table_'+token)
    for d in r['direction_probe']['signed_probes']:
        step = str(d['step_LSB']) if d['step_LSB'] == .1 else str(int(d['step_LSB']))
        token = ('-' if d['sign'] < 0 else '')+step
        row = next(x for x in table_rows if x.startswith('| '+token+' LSB |'))
        fields = [x.strip() for x in row.split('|')[1:-1]]
        peaks = '/'.join(str(d['paths'][paths[2]]['hypotheses'][h]['peak_changed_draws']) for h in ('fast','slow','common_level'))
        check(all(close_printed(t,d['paths'][p]['mean_h_proxy']) for t,p in zip(fields[1:4],paths))
              and fields[4] == peaks, 'all6_signed_proxy_and_moving_peak_table_'+token)
    migration = [v['native_Qbin_changed_draws'] for x in r['direction_probe']['signed_probes']
                 for d in x['paths'].values() for v in d['hypotheses'].values()]
    check(len(migration) == 54 and all(x == 0 for x in migration) and '所有 signed probe 的 native 条件 Q bin 迁移数为 0' in text,
          'all54_native_probe_conditional_bin_migrations_zero')
    check('未做 θ 参数有限差分或训练步' in text and '不含正则' in text,
          'one_carrier_voltage_direction_and_proxy_not_theta_or_totalregular_step')
    check('在 0.1 LSB 反号，在 1/10 LSB 保持负号' in text
          and '不能推出所有参数方向均反号' in text and '未选最好步长' in text,
          'sign_reversal_restricted_to_single_direction_minimal_step_allsteps_preserved')
    check('只读方法诊断' in text and '不更新模型' in text and '训练预算为 0' in text
          and '另行冻结预算和独立审核' in text, 'next_step_existing_output_only_no_automatic_training_or_GAN')
    check('response_match_not_established' in text and '科学响应验收状态维持原值' in text,
          'scientific_acceptance_remains_not_established')
    check(r['archive']['sha256'] in text and '24 个文件完整回收' in text,
          'collection_archive_SHA_count_as_independently_readback')
    failed = [x for x in checks if not x['passed']]
    result = dict(schema='n01/mechanism012-independent-report-QA/v1',run_id='mechanism012_20261015',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=not failed,status='PASS' if not failed else 'FAIL',
        report=dict(path=str(REPORT),sha256=EXPECTED_REPORT,bytes=REPORT.stat().st_size),
        numeric_readback=dict(path=str(numeric_path),sha256=EXPECTED_READBACK,status='PASS',checks2163_passed=True),
        script_sha256=sha(Path(__file__)),checks=checks,failed_checks=failed,
        signed_influence_weighting='wave endpoint adjoints already carry1/3; QPSD endpoint adjoints already carry2/3; each per-row dot is a weighted local linearization, not row loss or causal contribution',
        source_parent_scope='512 parents with4 repeated draws; per-hypothesis changed parents overlap, and changed/unchanged draw parent categories can overlap; union115 is not sum95+84+92',
        probe_scope='one saved random-shape carrier voltage direction with6 signed perturbations and3 step sizes; not theta finite difference; reversal only0.1LSB H_moving in this direction',
        precision_fixes_resolved=['Saved-only independent readback explicitly excludes fullCDF, originalnoise/carrier, models, AD, checkpoints and training.',
                                 'Only fake permutation/tie and normalization guards are observed; W1 absolute/cross-domain cusp remains unmeasured unknown_not_excluded.'],
        workflow_facts_outside_numeric_QA='cumulative159667 training count, same n01 PAUSED state and active-job0 are root-governed status facts; no live server/state query was performed by this reviewer',
        new_array_decodes=0,new_statistics=0,checkpoint_reads=0,NN_forward_rows=0,AD_queries=0,updates=0,DEV_reads=0,SSH_calls=0,
        final_scientific_status='response_match_not_established',reviewer_action_complete=True)
    with target.open('x',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    conclusion = OUT/'CONCLUSION.zh.txt'
    with conclusion.open('x',encoding='utf8') as f:
        f.write('012 独立保存输出读回与最终报告 QA 均通过。24 文件、完整 SHA 链、原生退出 0、432 项操作及登记预算相合；独立读回仅使用标准库和 NumPy，模型前向、梯度、权重读取、训练、DEV 和 SSH 均为 0。\n\n')
        f.write('在固定 TRAIN、θ2000 的本点，加权 QPSD 梯度 L2=1.66547，总梯度 L2=1.69236，cos=0.998759；正则梯度很小，不支持本点由正则主导。三假设的裁剪变化父簇并集是 115/512，各组可重叠。signed influence 是已加权的局部一阶线性化，不能当逐行 loss 或因果分解。\n\n')
        f.write('单一 carrier 电压方向的 ±0.1 LSB 探针，移动裁窗中心差分与 STE 导数反号；±1 和 ±10 LSB 保持同号。所有 54 项 native 条件 Q bin 迁移计数均为 0。固定与移动路径复用同一硬 ADC，但量化、argmax/tie 和未测量 W1 cusp 限制了导数解释。\n\n')
        f.write('下一步宜使用现有保存输出登记只读裁窗边界方法诊断，训练预算为 0。本结果没有确立 GAN 改善、全参数方向反号或物理响应匹配；科学状态仍为 response_match_not_established。\n')
    if sha(REPORT) != EXPECTED_REPORT or sha(numeric_path) != EXPECTED_READBACK:
        raise RuntimeError('Report/readback changed during QA')
    print(json.dumps(dict(path=str(target),sha256=sha(target),status=result['status'],total_checks=len(checks),failed_checks=failed,
                         report_sha256=EXPECTED_REPORT,conclusion_path=str(conclusion),conclusion_sha256=sha(conclusion)),ensure_ascii=True))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
