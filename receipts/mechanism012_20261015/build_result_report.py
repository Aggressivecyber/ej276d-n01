"""Build a bounded interpretation from persisted JSON; no models or new AD."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('E:/GAN/research/N01')
R = ROOT / 'receipts/mechanism012_20261015'
A = ROOT / 'runs/mechanism012_20261015/collected_01/runs/primary/artifacts'
OUT = ROOT / 'reports/mechanism012_20261015'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read(A/'SUMMARY.json')
    complete = read(A/'COMPLETE.json')
    native = read(A.parent/'supervisor/exit_receipt.json')
    collection = read(R/'collection_01/COLLECTION_RECEIPT.json')
    independent = R/'postrun_independent_01/NUMERIC_READBACK.json'
    independent_value = read(independent) if independent.is_file() else None
    if independent_value is not None:
        assert independent_value['status'] == 'PASS' and independent_value['passed']
        assert not independent_value['failed_checks']
    independent_note = (f"独立保存输出读回已完成，{len(independent_value['checks'])} 项核验通过、失败为 0；只使用标准库和 NumPy 重算保存数组的代数与统计，没有重跑完整 CDF 损失或原始 noise/carrier，也没有新增模型前向、AD、checkpoint 读取或训练。完整范围、代数核验和 SHA 见 NUMERIC_READBACK.json。"
                        if independent_value is not None else '独立数值读回正在进行，本版本尚未封口。')
    vectors = summary['gradient_attribution']['vectors']
    lines = ['# N01 012：新头梯度与裁窗机制诊断结果', '',
        '012 已按登记预算正常完成，原生退出码为 0，模型与新头参数保持不变。这个固定 TRAIN 检查点的总梯度主要由加权 QPSD 项驱动；正则项很小。六个已登记 carrier 扰动显示，硬量化与重新 first-argmax 裁窗会改变局部目标变化，其中 0.1 LSB 的 moving-crop 中心差分与代理梯度反号。下一步先诊断裁窗边界，再决定训练改法。', '',
        independent_note, '',
        '科学状态保持 `response_match_not_established`。本结果描述 θ2000、固定评价库和指定方向的局部机制，没有新增训练或响应验收。', '',
        '## 运行与证据范围', '',
        '- 输入：010 的 final 新头、512 个 SOURCE TRAIN 父事件 × 4 个原 draw、2048 个原 REAL TRAIN 参考行；fast、slow、common_level 三假设共享同一信号及 005 prefix。',
        '- A、005 和 final 新头各前向 2048 行；θ0/θ1000 前向为 0。正式 AD 为 9 次 scalar 查询及 2 次 masked endpoint VJP，总计 11 次。',
        '- ADC 行数 49152，其中原 actual hard/STE 为 12288、六探针为 36864；checkpoint 读取 5 次。432 项操作 entered/completed 全齐，failed 与 in-flight 为空。',
        '- optimizer、sampler、新 latent/noise draw、G/D/noise/population/C2ST 更新和 DEV 数值读取均为 0。累计科学训练更新仍为 159667，最近训练仍为 010。',
        f"- 诊断主体耗时 {summary['elapsed_seconds']:.3f} 秒；监督作业历时 {native['elapsed_seconds']:.3f} 秒。采样峰值 RSS {native['peak_sampled_RSS_bytes']/2**30:.3f} GiB、GPU {native['peak_sampled_GPU_bytes']/2**30:.3f} GiB；输出 {native['peak_output_bytes']/2**20:.3f} MiB。采样峰值不等于连续测得的真实最大值。",
        '- 作业在首次启动核验时已完成，未新增定时苏醒；同一个 n01 定时任务保持 PAUSED，当前活动 N01 科学作业为 0。', '',
        '## 参数梯度：QPSD 主导，正则很小', '',
        '`L = mean_h(W_h) + 2 mean_h(QPSD_h) + .01 (RMS + curvature + mean_h(saturation_h))`。下表各分量已包含原权重，参数向量顺序为 weight.flatten 后接 bias.flatten，共 1288 维。', '',
        '| 分量 | 已加权 loss | 梯度 L2 | 与 total 的 cosine |',
        '|---|---:|---:|---:|']
    names = {'wave':'wave','QPSD':'QPSD','RMS':'RMS','curvature':'curvature','saturation':'saturation'}
    for key, label in names.items():
        cos = vectors[key]['cosine_with_total']['value']
        lines.append(f"| {label} | {summary['weighted_loss_values'][key]:.9g} | {vectors[key]['L2']:.9g} | {cos:.6f} |" if cos is not None else f"| {label} | {summary['weighted_loss_values'][key]:.9g} | {vectors[key]['L2']:.9g} | undefined（零范数） |")
    lines += [f"| total | {summary['total_loss']:.9g} | {vectors['total']['L2']:.9g} | 1 |", '',
        f"QPSD 梯度范数约为 wave 的 {vectors['QPSD']['L2']/vectors['wave']['L2']:.3f} 倍，且与总方向 cosine={vectors['QPSD']['cosine_with_total']['value']:.6f}。RMS、curvature 和 saturation 不支持本点由正则主导的解释。wave 与 QPSD 的 cosine={summary['gradient_attribution']['pairwise_cosines']['wave']['QPSD']['value']:.6f}，两者并非强烈相互抵消。", '',
        f"独立 total 查询与五个分量向量和的 L2 误差为 {summary['gradient_attribution']['total_sum_absolute_error']:.9g}，相对误差 {summary['gradient_attribution']['total_sum_relative_error']:.9g}，低于登记工程容差 1e-5。假设 clip5 系数为 1，实际未裁剪。", '',
        'fixed-membership helper 在原 actual 点的 loss 和 1288 维参数梯度均与 native QPSD 相同（绝对差 0）。S 的解析 pullback 核验复用了正式 wave/QPSD 查询，最大绝对误差分别为 6.94e-18 和 1.73e-18。以上是实现与代数核验。', '',
        '## 同一当前 ADC 上的 actual 与 A-anchor 裁窗', '',
        'anchor 使用当前 ADC 和当前 prefix baseline，只把 gather 索引换成保存的 A 索引；conditional Q membership 与 support 冻结在 actual 点。三假设 changed draw 数如下，分母各为 2048；parent 数分母各为 512。', '',
        '| 假设 | crop changed draws | crop changed parents |', '|---|---:|---:|']
    for h, v in summary['influence']['hypotheses'].items():
        changed = v['categories']['crop_changed']
        lines.append(f"| {h} | {changed['draws']}/2048（{changed['draws']/2048*100:.3f}%） | {changed['parents']}/512 |")
    lines += ['', f"跨假设 changed parent 并集为 {summary['influence']['union_crop_changed_parents']}/512。一个父事件可以同时出现在 changed 与 unchanged draw 组，因此组内 parent 数不能相加。", '',
        '| 已加权分量 | endpoint signed influence 和 | fixed-membership anchor−actual 真 loss 差 | 线性化余项 |',
        '|---|---:|---:|---:|']
    for key, value in summary['influence']['linearization'].items():
        lines.append(f"| {key} | {value['signed_influence_sum']:.9g} | {value['true_fixed_membership_anchor_minus_actual_loss']:.9g} | {value['linearization_remainder']:.9g} |")
    lines += ['',
        '这里的 `inner(endpoint_adjoint, Y_anchor−Y_actual)` 是局部一阶 signed influence。它的线性化余项可见，不能替代真实 loss 差，也不是逐行 loss 或因果分解。changed/unchanged VJP 向量用于核对后验分组；它们的范数不能相加成贡献百分比。', '',
        '## 六探针：固定窗口与移动窗口的方向分离', '',
        f"唯一方向 `v = random_shape_pre_V / global_max_abs(random_shape_pre_V)`；全局最大值为 {summary['probe']['global_max_abs_random_shape_V']:.9g} V。prefix100 为精确零，canonical charge 方向和绝对最大值为 {summary['probe']['direction_canonical_Q_absmax']:.3g}。这是 carrier 方向，未做 θ 参数有限差分或训练步。", '',
        '三路径使用同一 frozen membership 与投影；S 在原硬 ADC 点施加 clamp 增量并固定 actual crop，H_fixed 用硬 ADC 并固定 crop，H_moving 复用同一个硬 ADC、重新 first-argmax。目标为 `mean_h(W + 2 QPSD_fixed)`，不含正则；下表单位均为 loss/V。', '',
        '| 对称步长 | AD 方向导数 | S 中心差分 | H_fixed 中心差分 | H_moving 中心差分 |', '|---|---:|---:|---:|---:|']
    for item in summary['probe']['central_differences']:
        p = item['paths']
        lines.append(f"| ±{item['step_LSB']:g} LSB | {summary['probe']['d_AD_loss_per_V']:.6f} | {p['S_clamp_surrogate_fixed_actual_crop']['central_difference_loss_per_V']:.6f} | {p['H_fixed_hard_ADC_fixed_actual_crop']['central_difference_loss_per_V']:.6f} | {p['H_moving_same_hard_ADC_first_argmax']['central_difference_loss_per_V']:.6f} |")
    lines += ['',
        'S 的三个中心差分都接近代理导数；H_fixed 的三个中心差分均为负但存在量化偏差。H_moving 在 0.1 LSB 反号，在 1/10 LSB 保持负号且幅度明显改变。这表明该方向下移动 crop 的有限差分具有步长依赖，固定-index 代理梯度无法保证硬观察目标的局部变化。六个 signed probe 的数值、peak/crop 变化与排序边界均已保留，未选最好步长。', '',
        '| signed δ | S proxy | H_fixed proxy | H_moving proxy | moving peak changed draws（fast/slow/common） |', '|---|---:|---:|---:|---|']
    for probe in summary['probe']['signed_probes']:
        p = probe['paths']; moving = p['H_moving_same_hard_ADC_first_argmax']
        counts = '/'.join(str(moving['hypotheses'][h]['peak_changed_draws']) for h in ['fast','slow','common_level'])
        lines.append(f"| {probe['sign']*probe['step_LSB']:g} LSB | {p['S_clamp_surrogate_fixed_actual_crop']['mean_h_proxy']:.9f} | {p['H_fixed_hard_ADC_fixed_actual_crop']['mean_h_proxy']:.9f} | {moving['mean_h_proxy']:.9f} | {counts} |")
    lines += ['',
        '所有 signed probe 的 native 条件 Q bin 迁移数为 0；移动/固定路径的差由相同硬 ADC 上的 gather 选择产生。日志记录 fake 排序/tie 及归一化 guard；W1 绝对配对和跨域 support cusp 仍为 unknown_not_excluded，未逐项测量。单个最小步长反号不能推出所有参数方向均反号，也不能判定 GAN 有效。', '',
        '## 下一项有界工作', '',
        '优先对裁窗边界做只读方法诊断：从已保存梯度与逐行数据核对 0.1 LSB 的跳变行、first-argmax 的 tie/margin 和 fixed/moving 差，将量化造成的 fixed-path 偏差与索引切换造成的 moving-path 差分开。使用 012 已保存输出能够完成的计算先在本地登记、核验，不重新抽样、不读取 DEV、不更新模型。若需要新的方向或模型前向，另行冻结预算和独立审核。', '',
        '在边界诊断给出明确结论前，训练预算为 0。后续 loss 权重或观察代理的改动一次只改一个因素，并使用相同新头、初始化、固定随机库、optimizer 和 G 预算作配对对照；不能因 QPSD 梯度较大就直接判断其权重应下调。', '',
        '## 文件与完整性', '',
        f"- 源包：`receipts/mechanism012_20261015/freeze_01/mechanism012_source.tar.gz`；SHA256 `{read(R/'freeze_01/FREEZE_RECEIPT.json')['package_sha256']}`；D 盘小源包备份一致。",
        f"- 结果归档：`receipts/mechanism012_20261015/collection_01/mechanism012_complete_01.tar`；SHA256 `{collection['archive_sha256']}`；24 个文件完整回收。",
        '- 原生结果：`runs/mechanism012_20261015/collected_01/runs/primary/artifacts/` 下的 SUMMARY、GRADIENTS、ROW_MECHANISMS、DIRECTIONAL_PROBE、operations 和 COMPLETE。',
        '- 核验入口：`receipts/mechanism012_20261015/postrun_independent_01/NUMERIC_READBACK.json` 与 `FINAL_RESULT_RECEIPT.json`。', '',
        '旧模型、010/011 结果与既有负结果保留。012 的正常完成是执行与机制诊断完成；科学响应验收状态维持原值。', '']
    destination = OUT/'N01_MECHANISM012_REPORT.md'
    if destination.exists():
        snapshots = R/'report_versions'
        snapshots.mkdir(exist_ok=True)
        old_hash = sha(destination)
        old = snapshots/f'{old_hash}.md'
        if not old.exists():
            old.write_bytes(destination.read_bytes())
    destination.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    evidence = {'schema':'n01/mechanism012-report-json-readback/v1', 'UTC':datetime.now(timezone.utc).isoformat(),
        'report':str(destination), 'report_sha256':sha(destination),
        'inputs':{str(p):sha(p) for p in [A/'SUMMARY.json',A/'COMPLETE.json',A/'DIRECTIONAL_PROBE.json',A.parent/'supervisor/exit_receipt.json',R/'collection_01/COLLECTION_RECEIPT.json']},
        'independent_receipt':str(independent) if independent.is_file() else None,
        'independent_receipt_sha256':sha(independent) if independent.is_file() else None,
        'NN_rows':0,'AD_queries':0,'checkpoint_loads':0,'optimizer_steps':0,
        'scientific_status':'response_match_not_established'}
    build_receipt = R/'REPORT_BUILD_RECEIPT.json'
    if build_receipt.exists():
        versions = R/'report_versions'
        versions.mkdir(exist_ok=True)
        old_receipt = versions/(sha(build_receipt)+'.REPORT_BUILD_RECEIPT.json')
        if not old_receipt.exists():
            old_receipt.write_bytes(build_receipt.read_bytes())
    build_receipt.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')
    assert sha(destination) == evidence['report_sha256']
    print(json.dumps({'report':str(destination),'sha256':sha(destination),'independent_receipt_present':independent.is_file()},ensure_ascii=False))


if __name__ == '__main__':
    main()
