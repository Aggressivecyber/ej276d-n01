"""Render the completed TRAIN-only origin audit and fixed null calibration."""
from pathlib import Path
import hashlib
import html
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runtime/mpl_noise_origin'))
sys.path.insert(0,str(ROOT/'runtime/plotting_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAME='noise_origin_audit_003_20261012'
RUN=ROOT/'runs'/NAME
OUT=ROOT/'reports'/NAME

def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=175,bbox_inches='tight')
    plt.close(fig)

def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    origin=read(RUN/'runs/science/origin/ORIGIN_REPORT.json')
    rr=read(RUN/'continuation_001/runs/science/real_real/RR_CALIBRATION_REPORT.json')
    qa=read(RUN/'continuation_001/runs/replay_qa/REPLAY_QA.json')
    assert origin['same_target_bitwise_passes']==21047 and rr['repeats']==128 and qa['passed']
    stages=origin['stages'];bs=origin['bootstrap']['stages']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    blue='#176DA5';orange='#CB593B';green='#22836D'
    names=['raw_fixed_0','block_fixed_0','q64_fixed_0','q32_fixed_0','postCFD_q32','N01_readc']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    y=np.array([stages[k]['mean_slope_correlation'] for k in names]);ci=np.array([bs[k]['correlation_percentile95'] for k in names])
    axes[0].errorbar(np.arange(6),y,yerr=np.vstack((y-ci[:,0],ci[:,1]-y)),fmt='o-',capsize=4,color=blue)
    axes[0].set(xticks=np.arange(6),xticklabels=['Raw\n0 ns','Block mean\n0 ns','ADC f64\n0 ns','Stored f32\n0 ns','CFD crop','N01 grid'],ylim=(-.79,-.25),ylabel='Correlation: baseline mean vs centered slope',title='Same 21,047 TRAIN records through the pipeline')
    for i,v in enumerate(y):axes[0].text(i,v+.048,f'{v:.4f}',ha='center',fontsize=10)
    x=np.array([0,5,10,15,20])
    for prefix,label,color,offset in [('raw','Earliest archive',blue,-.2),('q32','After ADC / f32',orange,.2)]:
        keys=[f'{prefix}_fixed_{v}' for v in x];yy=np.array([stages[k]['mean_slope_correlation'] for k in keys]);cc=np.array([bs[k]['correlation_percentile95'] for k in keys])
        axes[1].errorbar(x+offset,yy,yerr=np.vstack((yy-cc[:,0],cc[:,1]-yy)),fmt='o-',capsize=4,color=color,label=label)
    axes[1].axhline(stages['N01_readc']['mean_slope_correlation'],ls='--',color=green,label='Actual CFD crop')
    axes[1].set(xlabel='Fixed baseline-window start in archive (ns)',ylabel='Mean-slope correlation',title='Fixed window shifts reproduce the change',xticks=x,ylim=(-.82,.22))
    axes[1].legend(fontsize=9,loc='upper left');axes[1].annotate('20 ns window may include pulse onset',xy=(20,.068),xytext=(7,.18),fontsize=9,arrowprops={'arrowstyle':'->','color':'#444'})
    for ax in axes:ax.grid(alpha=.18)
    save(fig,'01_pipeline_and_window_position')

    fig,axes=plt.subplots(2,2,figsize=(12.8,8),constrained_layout=True)
    for k,label,color in [('block_fixed_0','Fixed 0 ns',blue),('block_fixed_10','Fixed 10 ns',orange),('N01_readc','Actual CFD / N01',green)]:
        profile=stages[k]['profile'];axes[0,0].plot(np.arange(100),np.array(profile['SD_V'])*1000,label=label,color=color)
        axes[0,1].plot(np.arange(1,21),stages[k]['ACF_mean'],'o-',ms=3,label=label,color=color)
    axes[0,0].set(title='Time-dependent ensemble SD',xlabel='Position within 100-sample window (ns)',ylabel='Standard deviation (mV)')
    axes[0,0].legend(fontsize=9);axes[0,1].set(title='Mean event-demeaned baseline ACF',xlabel='Lag (ns)',ylabel='ACF')
    with np.load(RUN/'runs/science/origin/STAGE_COVARIANCES.npz') as cov:
        C=cov['N01_readc']*1e6;limit=np.quantile(abs(C[~np.eye(100,dtype=bool)]),.99)
        im=axes[1,0].imshow(C,cmap='RdBu_r',vmin=-limit,vmax=limit,origin='lower',aspect='auto')
        axes[1,0].set(title='N01 covariance (off-diagonal color scale)',xlabel='Window sample',ylabel='Window sample')
        fig.colorbar(im,ax=axes[1,0],label='Covariance (mV²); diagonal saturated',shrink=.8)
    hist=origin['geometry']['actual_left_histogram'];axes[1,1].bar([int(k) for k in hist],[v/21047*100 for v in hist.values()],color=blue)
    axes[1,1].set(title='Actual CFD window starts',xlabel='Crop start in 500-point record (ns)',ylabel='TRAIN records (%)',xticks=[0,5,10,15,20],ylim=(0,44))
    axes[1,1].text(.03,.92,'12 / 21,047 CFD indices changed after ADC\n0 padded or invalid reconstructed windows',transform=axes[1,1].transAxes,va='top',fontsize=10)
    for ax in [axes[0,0],axes[0,1],axes[1,1]]:ax.grid(alpha=.16)
    save(fig,'02_baseline_structure')

    reps=rr['repetitions'];auc=np.array([v['AUC'] for v in reps]);cis=np.array([v['bootstrap']['percentile_95_interval'] for v in reps]);pvals=np.array([v['conditional_permutation']['plus_one_p_value'] for v in reps])
    fig,axes=plt.subplots(2,2,figsize=(12.8,8),constrained_layout=True)
    axes[0,0].hist(auc,bins=np.linspace(.40,.60,21),color=blue,alpha=.85);axes[0,0].axvline(.5,color='#222',ls='--')
    axes[0,0].set(title='128 independently refit random-label classifiers',xlabel='Held-out AUC (direction not flipped)',ylabel='Repetitions')
    axes[0,0].text(.03,.93,f"Mean {rr['AUC_mean']:.5f}\nSD {rr['AUC_std_ddof1']:.5f}",transform=axes[0,0].transAxes,va='top')
    fail=(cis[:,0]>.5)|(cis[:,1]<.5)
    for is_fail,color in [(False,blue),(True,orange)]:
        idx=np.flatnonzero(fail==is_fail);axes[0,1].vlines(idx,cis[idx,0],cis[idx,1],color=color,alpha=.55,lw=.8);axes[0,1].scatter(idx,auc[idx],color=color,s=9)
    axes[0,1].axhline(.5,color='#222',ls='--');axes[0,1].set(title='Old 95% joint-group bootstrap intervals',xlabel='Predeclared repetition index',ylabel='AUC and interval')
    axes[1,0].hist(pvals,bins=np.linspace(0,1,11),color=green);axes[1,0].axvline(.05,color=orange,ls='--');axes[1,0].set(title='Conditional within-group label permutations',xlabel='Two-sided +1 Monte Carlo p',ylabel='Repetitions')
    keys=['legacy_joint_bootstrap_false_positive','conditional_permutation_false_positive'];rates=np.array([rr[k]['rate'] for k in keys])*100;intervals=np.array([rr[k]['Wilson95'] for k in keys])*100
    axes[1,1].errorbar(np.arange(2),rates,yerr=np.vstack((rates-intervals[:,0],intervals[:,1]-rates)),fmt='o',ms=9,capsize=7,color=blue)
    axes[1,1].axhline(5,ls='--',color='#444',label='Nominal 5%');axes[1,1].set(xticks=[0,1],xticklabels=['Old bootstrap CI','Conditional permutation'],xlim=(-.5,1.5),ylim=(0,16),ylabel='False-positive rate (%)',title='Conditional rates with 95% Wilson intervals')
    for i,key in enumerate(keys):axes[1,1].text(i,rates[i]+1,f"{rr[key]['rejections']}/128 = {rates[i]:.2f}%",ha='center')
    axes[1,1].legend(loc='upper right',fontsize=9)
    for ax in axes.flat:ax.grid(alpha=.15)
    save(fig,'03_real_real_null_calibration')

    compact={'experiment_id':NAME,'origin_rows':21047,'origin_exact_reconstruction':True,'primary_chain':{k:{'rho':stages[k]['mean_slope_correlation'],'CI95':bs[k]['correlation_percentile95']} for k in names},'fixed_window_rho':{str(v):stages[f'raw_fixed_{v}']['mean_slope_correlation'] for v in x.tolist()},'rr':{k:rr[k] for k in ['repeats','AUC_mean','AUC_std_ddof1','legacy_joint_bootstrap_false_positive','conditional_permutation_false_positive','evaluation_classifier_updates']},'replay':{k:qa[k] for k in ['passed','classifier_checkpoints_verified','score_rows_replayed','maximum_absolute_score_error']},'noise_training_updates':0,'response_training_updates':0,'old_gate_passed':False,'independent_acquisition_validation':False}
    write(OUT/'REPORT_DATA.json',compact)
    rows='\n'.join(f'| {label} | {stages[k]["mean_slope_correlation"]:.6f} | [{bs[k]["correlation_percentile95"][0]:.6f}, {bs[k]["correlation_percentile95"][1]:.6f}] |' for k,label in zip(names,['最早容器，固定起点 0 ns','20 点均值降采样，固定 0 ns','14-bit ADC，固定 0 ns','float32 存储，固定 0 ns','实际 CFD 裁窗','N01 再量化']))
    body=f'''# N01 噪声来源与 REAL–REAL 对照校准

本轮诊断已完成。21,047 条合法 TRAIN 的 480 点目标全部逐位重建；128 次 REAL–REAL 全重新拟合完成，全部 65,536 个保存分数 GPU 重放逐位一致。新增噪声模型训练 **0**，脉冲响应训练 **0**；评价分类器共 25,600 更新。上一轮噪声门仍未通过，完整响应匹配尚未成立。

## 负相关早于已知处理链，固定取窗平移可复现其增强

同一批记录从最早可读 NPZ 的前 100 ns 到实际 CFD 基线，均值—中心化 OLS 斜率相关由 −0.3956 变为 −0.6782。固定位置的降采样、ADC 与 float32 转换对该相关系数只产生很小变化。起点固定移到 10 ns 时，原始容器内的相关已为 −0.6997，比实际 CFD 基线还强。因此，已知链中主要变化与取窗位置有关；不能把它归结为 CFD 算法创造了原本不存在的结构。

| 同事件阶段 | 相关系数 | 422 个人工组 bootstrap 95% 区间 |
| --- | ---: | --- |
{rows}

![相同事件处理链与固定窗口位置](figures/01_pipeline_and_window_position.png)

固定原始窗起点 0/5/10/15/20 ns 的相关依次为 −0.3956 / −0.5666 / −0.6997 / −0.6295 / +0.0683。20 ns 窗可能接近脉冲前沿，其均值/斜率与 CFD 位置的关联更明显，不能当成独立、纯净的电子学对照。所有这些窗都来自同一已选 TRAIN 人口，仍带有历史触发与 QC 条件。

在相同实际取窗位置，量化前后相关从 −0.678471 变为 −0.678247。使用量化前/后 CFD 位置截取同一降采样记录，仅 12/21,047 条的整数位置改变，相关从 −0.678452 变为 −0.678471。两条分支均无越界、填充或删行。float32 与 N01 再量化的差异属于微小数值舍入。

这里的“小变化”仅指上述均值—斜率相关。降采样会明显改变其他噪声统计：固定 0 ns 窗的 1 ns 延迟 ACF 从原始 −0.129768 变为降采样后 −0.013888。因此，后续噪声评价仍必须保留完整观察链。

原始固定 0 ns 和最终基线的负交叉协方差中，组内贡献分别为 98.21% 与 98.05%。因此，50 条人工分组之间的整体偏移不能解释主要结构；这些组也不等于独立采集。基线随窗内位置变化的标准差、完整协方差及短延迟 ACF 已保留。

![基线时间结构与裁窗位置](figures/02_baseline_structure.png)

## 阴性对照可以偶发误报，不能凭一次区间排除 0.5 认定实现故障

全部 128 次按预登记 seed 重抽随机域标签、重新初始化并拟合。每域拟合 2048、测试 256；分类器拟合与测试按原组隔离，标准化仅拟合样本参与。12 个伪测试组在数据解码前按 ID/数量确定，原 VAL 波形本轮未解码。

| 判据 | 随机标签下的误报 | 条件误报率 | Wilson 95% 区间 |
| --- | ---: | ---: | ---: |
| 旧 joint-group bootstrap 95% 区间不含 0.5 | 9 / 128 | 7.03% | 3.74%–12.82% |
| 固定分数、组内等配额置换，双侧 p ≤ 0.05 | 7 / 128 | 5.47% | 2.67%–10.86% |

AUC 均值 0.49778，标准差 0.02425。两组误报率区间均包含名义 5%；相差两次不支持“新规则显著更优”。这里的重复是同一历史采集和固定 12 组条件下的随机化，不是 128 次独立采集。旧 VAL 的一次异常可能是随机波动，也可能反映未识别问题；本轮不能为其定性，更不能追认旧噪声门。ACF 与均值—斜率联合统计的旧失败证据仍独立存在。

![固定128次REAL–REAL校准](figures/03_real_real_null_calibration.png)

## 下一步模型依据与停止边界

下一候选应先单独重做噪声模块：允许时间位置相关的结构，用正半定参数化联合表达慢分量的均值/趋势相关，并与短程相关噪声一起重新估计；不能冻结原 FIR 后再强加一个不可行的负协方差补偿。经已绑定的降采样、ADC 和取窗接口评价，逐项检查联合协方差、时间标准差、ACF、谱与分布判别。

基线证据只覆盖约 100 ns 的观测窗，不能直接认定慢分量在整段 480 ns 持续。整段外推及硬件解释需要额外安静长窗或独立采集证据。未来候选开始前，应先登记有限预算和评价规则，并把 REAL–REAL 对照作为校准证据；分类器未拒绝差异不等于分布相等。此次没有实现或拟合该新候选，脉冲响应训练保持停止。

## 执行与复核回执

- 原始来源完整 SHA256、ZIP CRC/EOF、时间网格、角色与身份绑定均已核验；只将 21,047 条 TRAIN 原始行解释为浮点数组，没有保存 10,000 点原始长波形副本。原始重建耗时 49.04 秒。
- 首次 RR 在 CUDA 显存 guard 的设备编号接口处退出 1，当时 target 尚未解码、分类器更新为 0。原失败回执保留。显式同一逻辑 GPU `cuda:0` 修复后，在独立 continuation 从 0 执行原定 128 次；原始重建未重算，科学种子、规则与预算未变。
- continuation 与 RR 原生退出均为 0；RR 科学执行 40.79 秒。独立重放原生退出 0，128 个模型、65,536 个分数全部逐位一致，最大绝对差 0，未新增拟合或重新抽 bootstrap/置换。
- 新增语义测试 16 项通过。旧模型、旧损失实现与 repair_002 失败门保留；执行源码与结果 SHA256 见回执。

最早 NPZ 的创建过程、触发前处理和原始仪器流程尚未得到验证；“已存在于最早归档”不能改写成“已证明硬件漂移”。422 个组为连续记录人工分块。此轮没有新的 VAL 检验或独立采集验证。

方法背景：[Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545)，[Multi-level block permutation](https://pubmed.ncbi.nlm.nih.gov/26074200/)。本项目的具体统计量、随机流和口径以预登记协议及代码为准。

本地完整数据与回执：`runs/{NAME}/`。公开发布包排除实测原始与降采样波形，仅含派生统计、分类器、日志和复核材料。预登记方法文档保留其执行前写作时点，完成状态以本报告及 FINAL_RECEIPT 为准。
'''
    (OUT/'README.md').write_text(body,encoding='utf8')
    # A deliberately small renderer for the report's fixed Markdown subset.
    fragments=[];in_table=False;in_list=False
    def inline(t):
        import re
        t=html.escape(t)
        t=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',t)
        t=re.sub(r'`([^`]+)`',r'<code>\1</code>',t)
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',t)
    for line in body.splitlines():
        if not line.strip():continue
        if not line.startswith('|') and in_table:fragments.append('</tbody></table>');in_table=False
        if not line.startswith('- ') and in_list:fragments.append('</ul>');in_list=False
        if line.startswith('|'):
            if set(line.replace('|','').replace(':','').replace('-','').replace(' ',''))==set():continue
            cells=[inline(x.strip()) for x in line.strip('|').split('|')]
            if not in_table:fragments.append('<table><thead><tr>'+''.join('<th>'+x+'</th>' for x in cells)+'</tr></thead><tbody>');in_table=True
            else:fragments.append('<tr>'+''.join('<td>'+x+'</td>' for x in cells)+'</tr>')
        elif line.startswith('!['):
            alt,src=line[2:].split('](');fragments.append(f'<figure><img alt="{html.escape(alt)}" src="{src[:-1]}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith('# '):fragments.append('<header><div class="eyebrow">N01 / TRAIN-ONLY DIAGNOSTIC / 2026-10-12</div><h1>'+inline(line[2:])+'</h1><div class="badge">来源复算完成 · 对照校准完成 · 噪声门仍未通过</div></header>')
        elif line.startswith('## '):fragments.append('<h2>'+inline(line[3:])+'</h2>')
        elif line.startswith('- '):
            if not in_list:fragments.append('<ul>');in_list=True
            fragments.append('<li>'+inline(line[2:])+'</li>')
        else:fragments.append('<p>'+inline(line)+'</p>')
    if in_list:fragments.append('</ul>')
    if in_table:fragments.append('</tbody></table>')
    css='''body{margin:0;background:#eef3f7;color:#172c3e;font:17px/1.8 "Microsoft YaHei",Arial,sans-serif}main{max-width:1140px;margin:32px auto;padding:46px 56px;background:white;border-radius:14px}header{border-bottom:3px solid #1975a8;padding-bottom:24px}h1{font-size:34px;line-height:1.45;margin:14px 0}h2{font-size:25px;margin-top:40px;color:#15577b}p{margin:18px 0}.eyebrow{font-size:14px;color:#357292;letter-spacing:1px}.badge{display:inline-block;background:#e9f2f8;padding:8px 14px;border-radius:6px;font-size:15px}table{border-collapse:collapse;width:100%;font-size:16px;margin:24px 0}th,td{padding:12px 15px;border-bottom:1px solid #d8e3e9;text-align:left}th{background:#eaf2f8}figure{margin:30px 0}img{display:block;width:100%;height:auto}figcaption{font-size:15px;color:#4d6473;margin-top:9px}code{overflow-wrap:anywhere;font-size:14px;background:#edf3f6}a{color:#146e9f}li{margin:12px 0}strong{color:#9a3d27}@media(max-width:800px){main{margin:0;padding:22px}h1{font-size:28px}table{font-size:13px}th,td{padding:8px}}'''
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>N01 噪声来源与对照校准</title><style>'+css+'</style><main>'+''.join(fragments)+'</main></html>'
    (OUT/'N01_NOISE_ORIGIN_REPORT.html').write_text(page,encoding='utf8')
    write(OUT/'BUILD_RECEIPT.json',{'generator':str(Path(__file__).relative_to(ROOT)),'generator_sha256':digest(Path(__file__)),'inputs':{str(p.relative_to(ROOT)):digest(p) for p in [RUN/'runs/science/origin/ORIGIN_REPORT.json',RUN/'continuation_001/runs/science/real_real/RR_CALIBRATION_REPORT.json',RUN/'continuation_001/runs/replay_qa/REPLAY_QA.json']},'figures':3,'new_scientific_fits':0,'visual_QA':'pending'})
    print(json.dumps({'report':str(OUT/'N01_NOISE_ORIGIN_REPORT.html'),'figures':3,'native_build':'complete'}))

if __name__=='__main__':main()
