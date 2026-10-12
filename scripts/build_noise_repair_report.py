"""Render the fixed repair_002 noise endpoint; no fitting or response generation."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runtime/mpl_noise_repair'))
sys.path.insert(0,str(ROOT/'runtime/plotting_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

RUN=ROOT/'runs/repair_002_20261012'
SCIENCE=RUN/'runs/science'
OUT=ROOT/'reports/repair_002_20261012'

def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def save(fig,name):
    fig.savefig(OUT/'figures'/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(OUT/'figures'/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)
def ellipse(ax,cov,color,label):
    vals,vecs=np.linalg.eigh(np.asarray(cov)*1e6)
    angle=np.degrees(np.arctan2(vecs[1,1],vecs[0,1]))
    ax.add_patch(Ellipse((0,0),2*np.sqrt(vals[1]),2*np.sqrt(vals[0]),angle=angle,
        fill=False,lw=2.5,edgecolor=color,label=label))

def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    gate=load(RUN/'NOISE_GATE.json')
    ident=load(RUN/'identifiability/IDENTIFIABILITY_REPORT.json')
    training=load(SCIENCE/'noise/TRAINING_COMPLETE.json')
    logs=[json.loads(x) for x in (SCIENCE/'noise/training.jsonl').read_text(encoding='utf8').splitlines()]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','axes.titleweight':'bold'})
    colors={'generated':'#176DA5','REAL':'#D15A36','TRAIN':'#2D8A71'}
    labels={'generated':'Generated noise','REAL':'REAL validation','TRAIN':'REAL training'}
    fig,axes=plt.subplots(2,2,figsize=(12,8.4),constrained_layout=True)
    step=np.array([x['update'] for x in logs]); total=np.array([x['total'] for x in logs])
    axes[0,0].plot(step,total,color=colors['generated'],alpha=.16,lw=.5)
    smooth=np.convolve(total,np.ones(100)/100,mode='valid')
    axes[0,0].plot(step[99:],smooth,color=colors['generated'],lw=1.6,label='100-update moving mean')
    axes[0,0].axvline(4000,color='#555555',ls='--',lw=1,label='Predeclared LR change')
    axes[0,0].set(title='Fixed 5,000-update noise fit',xlabel='Noise optimizer update',ylabel='Training loss')
    axes[0,0].legend(fontsize=9)
    x=np.arange(20)+1
    for key in ('TRAIN','REAL','generated'):
        stats=gate['statistics'][key]
        axes[0,1].plot(x,stats['acf_mean'],'o-',ms=3,color=colors[key],label=labels[key])
        axes[1,0].plot(np.arange(3),np.asarray(stats['rms_quantiles_V'])*1000,'o-',color=colors[key],label=labels[key])
        axes[1,1].plot(np.arange(5),stats['psd_band_fraction_mean'],'o-',color=colors[key],label=labels[key])
    axes[0,1].set(title='Demeaned baseline ACF: gate failed',xlabel='Lag (samples)',ylabel='Mean event ACF')
    axes[0,1].legend(fontsize=9)
    axes[1,0].set(title='RMS quantiles: gate passed',xticks=np.arange(3),xticklabels=['10%','50%','90%'],ylabel='RMS (mV)')
    axes[1,1].set(title='Spectral band fractions: gate passed',xticks=np.arange(5),xticklabels=['1','2','3','4','5'],xlabel='Predeclared frequency band',ylabel='Fraction of total power')
    for ax in axes.flat:ax.grid(alpha=.15)
    save(fig,'01_noise_fit_and_observables')

    fig,axes=plt.subplots(1,3,figsize=(13.5,4.6),constrained_layout=True)
    for key in ('TRAIN','REAL','generated'):
        info=ident['saved_gate_summaries'][key]
        ellipse(axes[0],info['covariance_V2'],colors[key],labels[key])
    axes[0].set(xlim=(-.21,.21),ylim=(-.18,.18),aspect='equal',xlabel='Baseline mean fluctuation (mV)',
        ylabel='Centered OLS slope fluctuation (mV)',title='One-SD covariance ellipses')
    axes[0].legend(fontsize=9,loc='upper right')
    keys=['TRAIN','REAL','generated']
    correlations=[ident['saved_gate_summaries'][key]['mean_slope_correlation'] for key in keys]
    axes[1].bar(np.arange(3),correlations,color=[colors[k] for k in keys])
    for i,v in enumerate(correlations):axes[1].text(i,v-.035,f'{v:.3f}',ha='center',va='top')
    axes[1].axhline(0,color='#555555',lw=1)
    axes[1].set(xticks=np.arange(3),xticklabels=['TRAIN','REAL val.','Generated'],ylim=(-.85,.15),
        ylabel='Correlation',title='Mean-slope correlation is missing')
    minimum=ident['latent_subtraction']['eigenvalues_V2'][0]*1e9
    low,high=np.asarray(ident['group_bootstrap']['frozen_K_latent_min_eigenvalue_percentile95_V2'])*1e9
    axes[2].axhspan(0,.8,color='#DCEFE7',label='Nonnegative required')
    axes[2].errorbar([0],[minimum],yerr=[[minimum-low],[high-minimum]],fmt='o',color='#AF2835',capsize=7,ms=8)
    axes[2].axhline(0,color='#555555',lw=1)
    axes[2].set(xlim=(-.8,.8),ylim=(-3,.8),xticks=[0],xticklabels=['TRAIN group bootstrap'],
        ylabel='Smallest eigenvalue (10$^{-9}$ V$^2$)',title='Frozen-K latent subtraction fails')
    axes[2].text(0,-.55,'0 / 512 PSD-feasible resamples',ha='center',fontsize=9)
    axes[2].legend(fontsize=9,loc='lower right')
    for ax in axes:ax.grid(alpha=.15)
    save(fig,'02_covariance_and_identifiability')

    fig,axes=plt.subplots(1,2,figsize=(12,4.9),constrained_layout=True)
    ax=axes[0]
    for i,view in enumerate(('raw','demean','detrend')):
        for j,case in enumerate(('generated_REAL','REAL_REAL')):
            r=gate['classifiers'][view][case]; v=r['AUC'];lo,hi=r['percentile_95_interval'];y=2-i+(j-.5)*.22
            ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3,color=['#176DA5','#D15A36'][j],
                label=['Generated-REAL','REAL-REAL control'][j] if i==0 else None)
    ax.axvline(.5,color='#555555',ls='--',lw=1)
    ax.set(xlim=(.35,.65),yticks=[0,1,2],yticklabels=['Detrended','Demeaned','Raw voltage'],xlabel='Held-out domain AUC; 95% group bootstrap',
        title='Control interval excludes 0.5: inconclusive')
    ax.legend(loc='lower right',fontsize=9);ax.grid(axis='x',alpha=.15)
    ax=axes[1]
    ax.axis('off')
    decisions=[('Noise model fit','5,000 updates','#176DA5'),('Classifier evaluation','1,200 updates','#176DA5'),
        ('Independent noise gate','NOT PASSED','#AF2835'),('New waveform training','0 updates','#AF2835'),
        ('Deterministic / full pipeline','Implemented; not executed','#555555')]
    for i,(name,value,color) in enumerate(decisions):
        y=.94-i*.19
        ax.text(.02,y,name,fontsize=11,transform=ax.transAxes)
        ax.text(.02,y-.065,value,fontsize=14,fontweight='bold',color=color,transform=ax.transAxes)
    save(fig,'03_classifier_control_and_stop')

    evidence={'NOISE_GATE.json':RUN/'NOISE_GATE.json','IDENTIFIABILITY_REPORT.json':RUN/'identifiability/IDENTIFIABILITY_REPORT.json',
        'REOPEN_VERIFICATION.json':RUN/'REOPEN_VERIFICATION.json','REPLAY_VERIFICATION.json':RUN/'REPLAY_VERIFICATION.json',
        'TRAINING_COMPLETE.json':SCIENCE/'noise/TRAINING_COMPLETE.json','PREREGISTRATION.json':ROOT/'receipts/repair_002_20261012/PREREGISTRATION.json'}
    for name,path in evidence.items():shutil.copyfile(path,OUT/name)
    summary=dict(status='repair_attempt_complete_stopped_at_failed_noise_gate',noise_gate_passed=False,
        noise_training_updates=5000,response_training_updates=0,classifier_evaluation_updates=1200,
        implementation_complete=True,full_response_comparison_executed=False,physical_calibration_established=False,
        failed_checks=gate['failed_checks'],metrics=gate['metrics'],mean_slope_correlations=dict(zip(keys,correlations)),
        independent_acquisition_available=False,validation_previously_used=True,train_rows=21047,validation_rows=599,
        native_exit_code=0,noise_elapsed_seconds=training['elapsed_seconds'])
    write(OUT/'SUMMARY.json',summary)
    rows=[('基线均值标准差比','0.85383','0.85–1.15','通过'),('RMS 分位数比','0.99291 / 1.01452 / 1.00251','预注册分位数容差','通过'),
        ('最大 ACF 差','0.021871','≤ 0.015','未通过'),('OLS 协方差误差','0.57436','≤ 0.20','未通过'),
        ('完整 100 点独立 SWD 比','0.98959','≤ 2.0；相同样本数','通过'),('原始基线生成–REAL AUC','0.53114 [0.48612, 0.57958]','点值及区间预注册范围','通过'),
        ('REAL–REAL 阴性对照 AUC','0.45052 [0.40264, 0.49146]','95% 区间须包含 0.5','未通过')]
    table=''.join('<tr>'+''.join('<td>'+html.escape(s)+'</td>' for s in row)+'</tr>' for row in rows)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>N01 电子学修复与独立验证</title>
<style>body{font-family:"Microsoft YaHei",system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 24px;color:#183044;line-height:1.75;background:#f7f9fc}h1{font-size:30px;line-height:1.3}h2{margin-top:34px;font-size:22px}p,li{font-size:16px}.lead{background:#fff0ee;border-left:5px solid #af2835;padding:16px 20px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.stat{padding:18px;background:white;border:1px solid #d8e2ed;border-radius:8px}.stat b{display:block;font-size:26px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:10px 12px;border:1px solid #d7e1eb;text-align:left}img{width:100%;height:auto;background:white;margin-top:16px}a{color:#006ca4}code{background:#e9eef4;padding:2px 4px}.note{background:#edf3f9;padding:14px 18px}.links{display:flex;gap:18px;flex-wrap:wrap}@media(max-width:700px){.stats{grid-template-columns:1fr}body{padding:0 14px}table{font-size:12px}th,td{padding:6px}}</style>
<h1>N01 电子学修复与独立验证</h1><p>repair_002_20261012 · 固定终点 · 2026-10-12 · 保留原有 U-Net 主干</p>
<p class="lead"><b>本轮改造与预注册验证已完成，噪声门未通过。</b>新版解决了“把短窗 OLS 散布直接复制成全窗随机斜率”的实现问题，并放开 FIR、刷新训练投影；但平稳噪声＋独立事件偏置仍不能覆盖观测到的基线联合结构。按预定停止规则，没有继续脉冲长训练，也没有把部分指标通过称为模型校准成功。</p>
<div class="stats"><div class="stat">噪声训练<b>5,000 次</b>全部 21,047 条 TRAIN 均已覆盖</div><div class="stat">独立分类器评价<b>1,200 次</b>3 个视图 × 2 个对照 × 200 次</div><div class="stat">新脉冲训练<b>0 次</b>确定性 / 完整随机比较未执行</div></div>
<h2>已完成的代码与验证</h2><ol>
<li><b>冻结旧模型的机制诊断：</b>已完成分窗口分类器、相同规则 REAL–REAL 对照、固定其余随机数的随机斜率关闭干预及固定投影盲区测试。详见<a href="../noise_audit_20261012/N01_NOISE_MECHANISM_REVIEW.html">前一份机制复核报告</a>，旧模型和原结果保留。</li>
<li><b>电子学 V2：</b>65 点自由 FIR＋精确 L2 归一化、平稳相关噪声、事件偏置及窄初始化的事件 RMS 扰动；取消默认随机斜率。独立拟合预算与判定规则在运行前冻结，终点固定为第 5,000 次更新。</li>
<li><b>分布损失 V2：</b>每次更新刷新方向，同次双遍梯度共享不可变投影上下文；独立冻结评估方向；补充逐坐标分布和固定位置短延迟联合结构。人工反例与双遍梯度一致性检查通过。</li>
<li><b>条件脉冲管线：</b>确定性和完整随机两臂使用同一噪声、相同初始主干和固定人口比较权重，每臂计划 19,000 次。入口核验通过状态、电子学权重、配置和数据身份绑定；失败门在创建响应模型之前阻止执行。评估代码包含独立投影、父事件隔离、六视图分类器及 REAL–REAL 对照，当前仅完成工程测试，尚无本轮脉冲终点。</li></ol>
<h2>预注册噪声门的实际结果</h2><p>噪声拟合只使用 TRAIN 前 100 点。独立评价使用已有验证集的 599 条基线、12 个原始组；分类器每类 TRAIN 2,048 条 / VAL 256 条，组隔离且阴性对照两类在组内随机分配，保留原始标签方向。</p>
<table><thead><tr><th>检查</th><th>实际值</th><th>运行前规则</th><th>结果</th></tr></thead><tbody>__TABLE__</tbody></table>
<img src="figures/01_noise_fit_and_observables.png" alt="噪声训练曲线与 ACF、RMS、频谱统计"><p>基线均值标准差：生成 0.13102 mV，REAL 0.15345 mV，TRAIN 0.15653 mV。RMS、频谱和边缘统计接近并不能替代完整联合结构检查。</p>
<h2>剩余问题来自模型族限制，继续相同训练不能消除</h2>
<p>在基线中心对称坐标下，平稳标量过程的协方差具有时间反演对称性，因此基线均值与中心化 OLS 斜率的协方差为零；独立事件偏置不改变这个结论。当前生成相关系数为 −0.00223，TRAIN 为 −0.67825，REAL 为 −0.65679。该差异不等于已识别出硬件漂移；触发、选样、对齐或前脉冲结构都可能影响观测基线。</p>
<img src="figures/02_covariance_and_identifiability.png" alt="均值斜率协方差与潜在漂移扣除不可行性">
<p>进一步按用户提出的 Cov(β̂) = Cov(β) + A K Aᵀ 复核：固定当前已拟合的平稳 K，TRAIN 协方差扣除 A K Aᵀ 后的最小特征值为 −2.1693×10⁻⁹ V²，所需相关系数为 −1.2155；残余不是合法协方差。422 个 TRAIN 原始组的 512 次重采样中，没有一次通过半正定条件。这里将量化后的 TRAIN 协方差与量化前的 K 相减，属于该分解式的近似数值检查。这个结论以当前 K 和同次采集为条件，不是所有联合噪声模型的失败证明。</p>
<p>因此不能直接把 OLS 相关系数补回模型，也不能把负特征值裁掉就称作校准完成。前 100 点不能唯一确定 480 点的真实漂移持续性。后续若引入联合漂移与噪声，需要同时约束 K、潜在协方差和完整 100 点分布，并明确长窗外推假设；更有判别力的新增证据是覆盖 480 点以上的纯噪声记录、触发与对齐机制以及独立采集。</p>
<h2>分类器结果与证据边界</h2><img src="figures/03_classifier_control_and_stop.png" alt="分类器阴性对照及停止决策">
<p>生成–REAL 原始基线 AUC 为 0.5311，但阴性对照 95% 区间没有覆盖 0.5，故这项通过不足以支持等价结论。没有翻转 AUC、重抽种子或修改门槛。验证集在此前诊断中已经查看，同一采集仅 12 个验证原始组，因此本轮是开发阶段筛查，不是独立硬件校准或跨采集验收。</p>
<div class="note"><b>可复现性复核：</b>5000 个连续更新、5000 个不同训练投影上下文；每条 TRAIN 被访问 121–122 次。6 个分类器权重和得分已重开复算，训练缩放器逐位一致；同设备重新载入噪声终点重放 10,240 条生成样本，最大电压差为 0，未增加训练更新。</div>
<h2>证据与复现入口</h2><div class="links">__LINKS__</div>
<p>当前科学状态：<code>repair_attempt_complete_stopped_at_failed_noise_gate</code>。本轮代码改造、规定的独立噪声运行与诊断已结束；新噪声候选仍未通过，完整波形及条件峰宽仍待通过噪声门后验证。</p></html>'''
    links=''.join('<a href="'+html.escape(name)+'">'+html.escape(name)+'</a>' for name in ['SUMMARY.json',*evidence])
    (OUT/'N01_NOISE_REPAIR_REPORT.html').write_text(page.replace('__TABLE__',table).replace('__LINKS__',links),encoding='utf8')
    (OUT/'README.md').write_text('''# N01 repair_002：噪声独立验证未通过

电子学训练 **5000 次**，评价分类器 **1200 次**，新脉冲训练 **0 次**。保留 U-Net，电子学 V2、投影刷新、逐坐标/短延迟损失及条件两臂训练/独立评价代码已实现；噪声门未通过，本轮两臂比较未执行。

| 检查 | 实际值 | 运行前规则 | 结果 |
|---|---:|---:|---|
| 基线均值标准差比 | 0.85383 | 0.85–1.15 | 通过 |
| 最大 ACF 差 | 0.021871 | ≤0.015 | 未通过 |
| OLS 联合协方差误差 | 0.57436 | ≤0.20 | 未通过 |
| 完整100点独立 SWD 比 | 0.98959 | ≤2.0 | 通过 |
| 生成–REAL AUC | 0.53114 [0.48612,0.57958] | 预注册点值/区间范围 | 通过 |
| REAL–REAL AUC | 0.45052 [0.40264,0.49146] | 95%区间包含0.5 | 未通过 |

![训练与噪声统计](figures/01_noise_fit_and_observables.png)

平稳标量噪声加独立偏置的中心化均值—斜率协方差为零；生成相关系数 −0.00223，TRAIN −0.67825，REAL −0.65679。固定当前 K 后，OLS 协方差相减的潜在协方差非半正定，最小特征值 −2.1693e−9 V²。422 个 TRAIN 组的 512 次重采样均未通过半正定条件。这是固定 K、同次采集及 pre/post-ADC 近似下的诊断，不证明真实硬件漂移，也不能识别480点外推。

![联合统计与可辨识性](figures/02_covariance_and_identifiability.png)

![阴性对照与停止决策](figures/03_classifier_control_and_stop.png)

34 项工程测试通过；6 个分类器权重/得分重开核对；同设备重放10,240条噪声样本逐位一致。验证集已在此前诊断中查看，仅一份采集的12个验证组；没有独立采集或硬件校准结论。

[完整 HTML 报告](N01_NOISE_REPAIR_REPORT.html) · [噪声门全部结果](NOISE_GATE.json) · [协方差诊断](IDENTIFIABILITY_REPORT.json) · [重开核对](REOPEN_VERIFICATION.json) · [同设备重放](REPLAY_VERIFICATION.json) · [预注册](PREREGISTRATION.json)

报告、PNG/PDF 图和 JSON 证据由 `scripts/build_noise_repair_report.py` 从冻结终点生成，不增加训练更新。权重、完整日志和源码在 [repair_002 Release](https://github.com/Aggressivecyber/ej276d-n01/releases/tag/n01-repair-002-20261012)。
''',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
