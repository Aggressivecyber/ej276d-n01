"""Build a traceable Chinese013 report from saved results, without science replay."""
from pathlib import Path
import datetime as dt
import hashlib
import html
import json
import re

ROOT=Path('E:/GAN/research/N01')
BASE=ROOT/'runs/wp1_013_20261015/runtime_fix01_collected_03/runs/primary'
OUT=ROOT/'reports/wp1_013_20261015'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
numeric=read(ROOT/'receipts/wp1_013_20261015/postrun_numeric_01/NATIVE_NUMERIC_READBACK.json')
decision=read(ROOT/'receipts/wp1_013_20261015/postrun_independent_01/DECISION_READBACK.json')
arrays=read(ROOT/'receipts/wp1_013_20261015/postrun_independent_01/REFERENCE_ARRAY_READBACK.json')
assert decision['passed'] and arrays['passed']
assert decision['decision']=='adequate'
info=read(ROOT/'INPUT_INFORMATION_AUDIT.json')
boundary=read(BASE/'boundary/BOUNDARY_SAVED_AUDIT.json')
t=read(ROOT/'CALIBRATED_TOLERANCES_WP1_013.json')
terminal=read(ROOT/'receipts/wp1_013_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json')
state=read(ROOT/'receipts/wp1_013_20261015/native_repeat_fix01/wake_status_01/REMOTE_STATE.json')
native=terminal['native_exit']
assert t['T_by_hypothesis']==decision['T']
assert not OUT.exists(), 'Preserve an earlier report version'
OUT.mkdir()
lines=[]
def para(text): lines.extend([text,''])
def heading(text): para('## '+text)
def table(headers, rows):
    lines.extend(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|'])
    lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
    lines.append('')
fmt=lambda x: f'{x:.9f}' if isinstance(x,(int,float)) else str(x)
para('# N01 WP1 / 013：决策与评价合同报告')
para('日期：2026-10-15（Asia/Singapore）。本轮按用户六工作包主实施计划完成 WP1；工程终态、方法选择与科学验收分别记录。')
para('**决策：保留现有硬观察与训练代理，推进 WP2 的 C0/C1 有限损失表示对照。** 当前总梯度方向的真实参数有限更新通过全部 12 格方法规则；当前 010 头的完整人口 wave 改善仍低于 2% 实用门槛，形态与基线均值的绝对距离超过已冻结人工容忍度，科学状态保持 `response_match_not_established`。')
heading('1. 执行与样本范围')
para('第三次工程 attempt 的计算进程与监督进程均原生退出 0。265 个文件已安全回收，库存大小、SHA256、操作日志、输入和源码保护均通过独立读回。本阶段训练更新与新自动微分查询均为 0。')
table(['对象','本轮范围','统计含义'],[
    ['SOURCE 响应评价','6593 TRAIN 父事件 × 每库4次 × 2库 = 52744行','重复响应的独立源单位为父事件'],
    ['REAL 响应评价','全部21047 TRAIN 行，422个原逻辑组','当前确认1次采集；逻辑组不等于独立采集'],
    ['SOURCE 信息审计','全部8192父；TRAIN6593，原内部val1599','512个已暴露固定DEV是内部val子集，余1087曝光未知'],
    ['真实参数验证','1024 TRAIN 父，2批×512，2库、3假设','5方向×3名义RMS尺度×正负号，30/30状态有定义'],
    ['RR 辅助比较','两侧各10475行/210组','仅RR配平排除2组97行；完整人口保留全部21047行']])
para('硬观察合同保持14 bit、384点载体、原100点基线、first-argmax搜索[112,144)、p−16:p+240裁窗、256点输出、1 ns软件采样间隔和tail起点46。θ0引用010的G2000终点；两库是Monte Carlo库，三个电子学假设共用同一训练头。')
heading('2. 完整人口：小幅wave收益与未通过的绝对指标')
rows=[]
for h,r in numeric['full_population']['per_hypothesis'].items():
    m=r['mean_of_two_bank_distances'];a,z=m['A'],m['theta']
    rows.append([h,fmt(a['wave']),fmt(z['wave']),f"{100*r['wave_relative_improvement']:.4f}%",fmt(a['QPSD']),fmt(z['QPSD']),fmt(z['noise'])])
table(['假设','A wave','θ0 wave','wave改善','A QPSD','θ0 QPSD','noise（两臂相同）'],rows)
para('表内数值为每个假设下两库经验距离的算术均值。三种wave收益分别低于2%实用门槛；slow的QPSD距离增加约2.66%。对SOURCE全体重复评价与旧512父参考评价采用不同人口，分别保留。')
rows=[]
for h,r in numeric['full_population']['per_hypothesis'].items():
    z=r['mean_of_two_bank_distances']['theta'];th=t['T_by_hypothesis'][h]
    rows.append([h,*[f'{z[k]/th[k]:.3f}' for k in ['peak_voltage_V','rise10_90_ns','FWHM_ns','baseline_mean_V']],f'{z["rise_undefined_rate"]/th["rise_undefined_rate"]:.3f}'])
table(['假设','峰幅距离/T','上升时间距离/T','半高宽距离/T','基线均值距离/T','上升沿未定义率差/T'],rows)
para('峰幅、上升时间、半高宽和基线均值的绝对偏差仍是当前响应模型的主要缺口。noise_prefix100分布距离在两臂均为0.124825000；随机形状头保留原100点前缀，其噪声分布保持原A。')
rr=numeric['REAL_REAL_control']['metrics']['values']
para(f'同采集RR的wave、QPSD、noise经验距离分别为{rr["wave"]:.9f}、{rr["QPSD"]:.9f}、{rr["noise"]:.9f}。RR用于展示有限样本比较尺度，容忍度由预登记人工扰动确定。')
heading('3. 四套评价和同父重复响应')
table(['评价套件','本轮保存内容','解释范围'],[
    ['全人口','12套bank×A/θ0×假设的完整指标、失效与未定义标记','全部生成和REAL行均保留'],
    ['输出Q条件','各套8箱的PSD/W1、均值、宽度、样本数和支持标记','输出Q条件是观测分组'],
    ['SOURCE输入趋势','按原父事件连接输入Q、幅度、h及形状PC的描述性关联','生成器输入依赖；无配对REAL条件真值'],
    ['同父重复','每库4次及合并8次的均值、方差、协方差重建与MCSE','样本协方差分母分别3/7；8次经验协方差秩≤7']])
rows=[]
repeats=numeric['selected_saved_array_statistics']['same_parent_eight_repeat_statistics']
for h in ['fast','slow','common_level']:
    a=next(x for x in repeats if x['arm']=='A' and x['hypothesis']==h)
    z=next(x for x in repeats if x['arm']=='theta' and x['hypothesis']==h)
    rows.append([h,f'{1000*a["hard_wave_sqrt_mean_sample_variance_V"]:.6f}',f'{1000*z["hard_wave_sqrt_mean_sample_variance_V"]:.6f}',f'{1000*z["hard_wave_RMS_conditional_mean_MCSE_V"]:.6f}'])
table(['假设','A 硬波形重复散布/mV','θ0 硬波形重复散布/mV','θ0 条件均值MCSE/mV'],rows)
para('表内重复散布为 sqrt(mean_parent,time(s²_8draw))，MCSE为 sqrt(mean_parent,time(s²_8draw/8))；下述纯信号散布沿用同一口径。')
para('纯信号θ0的重复散布RMS约1.554 mV，头的随机响应仍有效。A纯信号的重复数值残差约5.69e−12 V；两库实际逐draw的float64 A均已保存。硬波形散布包含电子学、ADC及裁窗的共同影响，真实同父响应目标尚无配对观测。')
para('保存格式用ADC裁窗整数码、原double基线均值及必要的IEEE XOR校正重建硬波形。原生写入时与内存硬输出逐字节比较；独立读回校验编码回写、逐行身份、实际A、均值/方差及4/8draw标量协方差，原Y未另存第二份副本。完整256点条件协方差可由实际draw的中心化因子重建，本轮未展开稠密矩阵。REAL前端未留存的ADC越界/半网格信息继续标unknown。')
heading('4. SOURCE包含什么信息')
para('归一化SOURCE形状主要沿一个维度变化。TRAIN的480点原输入、384点载体及256点clean窗口，canonical-Q与峰值归一化后的95%、99%、99.9%解释方差维数均为1。当前输入来自已登记的EE/hadron双模板混合及可见能量代理，8192行均与原生成公式的float32输出逐字节相合。')
train=info['PCA']['TRAIN_carrier384_canonical_Q']
para(f'TRAIN canonical-Q归一化384点形状的总方差为{train["total_variance"]:.9g}；解析双精度到float32差的中心化方差为{train["float32_difference_centered_total_variance"]:.9g}。PC1与h的Pearson约−0.999946。数值图样个数单独报告；它们包含浮点舍入差异。')
feasible=info['same_Q_real_input_shape_feasibility']
para(f'现有SOURCE中找到了相近Q而形状可区分的父事件：固定相对Q差≤1%、峰归一化形状RMS差>0.001的规则下，{feasible["rows_with_testable_shape_contrast"]}个父有可测对比，最大验证强度{feasible["largest_verified_strength"]:.8f}。这些配对直接来自已有SOURCE输入；响应干预尚未执行。后续可用其检查模型是否保留输入形状依赖。')
para('有效源到实测响应的解释范围受输入与采集信息约束：微观沉积时序、已校准物理权重和独立实测采集仍为unknown/unverified。历史DEV角色和曝光记录保持原样；本轮SOURCE审计未对DEV执行响应网络或用于模型选择。')
heading('5. 保存012边界定位与真实参数有限更新')
table(['名义carrier尺度/LSB','固定窗S中心差分/每V','固定窗硬ADC/每V','动态硬窗/每V'],[
    [r['step_LSB'],*[f'{r["loss_per_V"][k]:.9f}' for k in ['S_clamp_surrogate_fixed_actual_crop','H_fixed_hard_ADC_fixed_actual_crop','H_moving_same_hard_ADC_first_argmax']]]
    for r in boundary['central_differences']])
para('012保存方向的AD值为−0.197338171/每V。0.1LSB下动态硬窗中心差分反号，1和10LSB恢复负号；95条逐状态/假设的峰位变化记录全部保存，含原父身份、ties、crop位移与Q/tail/PSD变化。endpoint signed influence采用原保存A-anchor，只解释其局部线性化。')
table(['正向名义1LSB主方向','S下降','动态硬目标下降','adequate格','repair格'],[
    ['negative_saved012_total','12/12','12/12','12/12','0/12'],
    ['prospective_AdamW2001','4/12','6/12','4/12','0/12']])
para('机器规则要求同一主方向覆盖2库×2父批×3假设全部12格。总梯度方向满足下降与关键指标非劣要求，因此选择adequate；prospective AdamW由旧moments和保存梯度构造，在本轮没有执行优化器步骤。')
para('其余符号、0.1/10LSB及固定Rademacher方向均保留。局部proxy下降而硬目标上升的格未达到预登记材料量级，当前证据支持继续有限训练对照。该方法结论对应固定G2000系统、已保存梯度和所列有限步。')
heading('6. 数字容忍度冻结')
para('公式固定为 `T[h,component] = max_7case(mean_2banks(self_distance))`。七案例为gain±3%、tail±3%、AC噪声幅度±10%及额外Gaussian σ=1软件ns；全部21个敏感性控制同时保留。两套独立读回均从保存self-distance重建公式，误差为0。')
table(['假设','T_wave','T_QPSD','T_noise','T_rawPSD','T_peak/V','T_rise/软件ns','T_FWHM/软件ns'],[
    [h,*[fmt(t['T_by_hypothesis'][h][k]) for k in ['wave','QPSD','noise','PSD_raw','peak_voltage_V','rise10_90_ns','FWHM_ns']]]
    for h in ['fast','slow','common_level']])
para('wave标尺由额外1ns展宽、QPSD标尺由+3%gain、noise标尺由−10%AC噪声幅度确定。每个组件的获胜案例与全部数字另存。零/未定义的灵敏度保持not_calibrated；实际硬件误差校准仍待独立实验。')
para('正式冻结表：`CALIBRATED_TOLERANCES_WP1_013.json`；SHA256：`'+sha(ROOT/'CALIBRATED_TOLERANCES_WP1_013.json')+'`。冻结发生在任何WP2科学计算之前。')
heading('7. 三次attempt的真实成本和资源')
counts=terminal['attempt3_actual_counts']; prior=terminal['prior_two_failed_attempt_actual_counts'];total=terminal['aggregate_actual_counts']
keys=['old_A_rows','head_actual_parameter_forward_rows','head_linear_direction_map_rows','head_total_kernel_rows','noise005_rows','hard_ADC_rows','all_path_gather_rows','checkpoint_loads']
table(['计数','attempt1失败','attempt2失败','attempt3完成','三次合计'],[
    [k,prior[k]//2,prior[k]//2,counts[k],total[k]] for k in keys])
para('attempt1退出1：冻包缺少原011 CDF依赖。attempt2退出1：A逐字节门强于既有1e−7 V数值回放标准。两个现场及各自消耗完整保留。第三次按登记修复保存全部实际A，并将输出上限改为1536 MiB；其余科学计算cap保持原值。三次合计optimizer、G/D/noise/pop/C2ST更新、新AD和DEV模型行数均为0。')
para(f'成功attempt约1185.785秒；采样峰值进程树RSS {native["peak_RSS_bytes"]/2**30:.3f} GiB、GPU {native["peak_GPU_bytes"]/2**30:.3f} GiB、输出{native["peak_output_bytes"]/2**20:.3f} MiB，均在8/10 GiB与1536 MiB登记限制内。原185条旧源码保护记录、84个本轮源码及86个Git原始blob已核验。科学训练累计保持159667。')
heading('8. 进入WP2的固定行动')
para('执行C0原损失与C1无分母积分表示对照，各2训练种子、每次最多2000G，固定4次串行短试验。两候选共用新头结构、按实际输出RMS校准的初始化、SOURCE/REAL流、shape/noise流、wave投影和训练预算；C1只改变分布损失表示。C2关闭。')
para('C1使用Q、T及预设不重叠分段积分，经固定REAL TRAIN尺度与signed-asinh变换。最终评价继续原始硬ADC/first-argmax、原QPSD、rawPSD与条件PSD，全部低Q/非正Q/失效记录保留；按已冻结用途门槛、绝对容忍度、关键组件非劣性和两种子证据选择一个方案。新增AD及优化器预算在014单列。')
para('六工作包主计划持续执行。WP1工程交付完成；在域响应验收和跨采集物理复用分别保留为未建立与未验证。')
heading('9. 证据索引')
references=[
    ('主实施计划','MASTER_EXECUTION_PLAN.json'),
    ('观察合同','OBSERVATION_CONTRACT.json'),('数据角色','DATA_ROLES.json'),
    ('SOURCE信息审计','INPUT_INFORMATION_AUDIT.json'),('验收合同','docs/ACCEPTANCE_CONTRACT.md'),
    ('数字容忍度表','CALIBRATED_TOLERANCES_WP1_013.json'),
    ('数字容忍度冻结回执','receipts/wp1_013_20261015/postrun_root_01/TOLERANCE_FREEZE_RECEIPT.json'),
    ('回收库存回执','receipts/wp1_013_20261015/native_repeat_fix01/collection_03/COLLECTION_RECEIPT.json'),
    ('数值读回','receipts/wp1_013_20261015/postrun_numeric_01/NATIVE_NUMERIC_READBACK.json'),
    ('独立决策读回','receipts/wp1_013_20261015/postrun_independent_01/DECISION_READBACK.json'),
    ('独立数组读回','receipts/wp1_013_20261015/postrun_independent_01/REFERENCE_ARRAY_READBACK.json'),
    ('终态与三次费用','receipts/wp1_013_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json')]
for label,path in references:para(f'[{label}](../../{path})')
md='\n'.join(lines).rstrip()+'\n'
md_path=OUT/'N01_WP1_013_REPORT.md'
md_path.write_text(md,encoding='utf-8',newline='\n')

def inline(text):
    text=html.escape(text)
    text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text)
render=[];i=0;section=0
while i<len(lines):
    line=lines[i]
    if line.startswith('|'):
        block=[]
        while i<len(lines) and lines[i].startswith('|'):
            block.append(lines[i]);i+=1
        cells=lambda row:[x.strip() for x in row.strip('|').split('|')]
        render.append('<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in cells(block[0]))+'</tr></thead><tbody>')
        render.extend('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in cells(row))+'</tr>' for row in block[2:])
        render.append('</tbody></table></div>');continue
    if line.startswith('## '):
        section+=1;render.append(f'<h2 id="section{section}">'+inline(line[3:])+'</h2>')
    elif line.startswith('# '):render.append('<h1>'+inline(line[2:])+'</h1>')
    elif line:render.append('<p>'+inline(line)+'</p>')
    i+=1
doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>N01 WP1 013 决策报告</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf3f8;color:#172638;font:17px/1.7 "Microsoft YaHei",Arial,sans-serif}main{width:min(1280px,96%);margin:24px auto 80px;padding:38px 46px;background:white;border-top:7px solid #1763a8;box-shadow:0 6px 28px #18355012}h1{font-size:31px;line-height:1.4;margin:0 0 24px;color:#103e6b}h2{font-size:23px;margin:40px 0 18px;border-bottom:2px solid #e0eaf4;padding-bottom:9px;color:#174a78;scroll-margin-top:24px}p{margin:13px 0}strong{color:#0e4e7c}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.5;margin:16px 0}th{background:#eaf2f9;color:#16456d;text-align:left}td,th{padding:11px 12px;border:1px solid #dce5ed;vertical-align:top}tbody tr:nth-child(even){background:#f8fbfe}td:first-child{font-weight:600}code{font:14px/1.5 Consolas,monospace;background:#f0f4f7;padding:2px 4px;overflow-wrap:anywhere}a{color:#1263a3;text-decoration:none}.tablewrap{overflow-x:auto}@media print{body{background:white}main{width:100%;margin:0;padding:10px;box-shadow:none}h2{break-after:avoid}tr{break-inside:avoid}}
</style><main>'''+''.join(render)+'</main></html>'
html_path=OUT/'N01_WP1_013_REPORT.html';html_path.write_text(doc,encoding='utf-8',newline='\n')
receipt=dict(schema='n01/wp1-report-build/v1',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
    status='built_from_saved_verified_evidence',method_choice='adequate',scientific_status='response_match_not_established',
    references={p:dict(sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for _,p in references},
    report={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in (md_path,html_path)},
    new_checkpoint_loads=0,new_NN_rows=0,new_AD_queries=0,new_training_updates=0,render_QA_pending=True)
(OUT/'REPORT_BUILD_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=receipt['status'],report=str(md_path),sha256=sha(md_path)),ensure_ascii=False))
