"""Render saved endpoints without fitting or checkpoint selection."""
from pathlib import Path
import hashlib
import html
import json
import os
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runtime/mpl_noise_improvement_004'))
sys.path.insert(0,str(ROOT/'runtime/plotting_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

ID='noise_improvement_004_20261012'
RUN=ROOT/'runs'/ID
SCI=RUN/'runs/science'
OLD=ROOT/'runs/repair_002_20261012/runs/science'
OUT=ROOT/'reports'/ID
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def save(fig,name):
    fig.savefig(OUT/'figures'/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(OUT/'figures'/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def rho(c):
    c=np.asarray(c);return float(c[0,1]/np.sqrt(c[0,0]*c[1,1]))
def ellipse(ax,cov,color,label):
    vals,vecs=np.linalg.eigh(np.asarray(cov)*1e6)
    ax.add_patch(Ellipse((0,0),2*np.sqrt(vals[1]),2*np.sqrt(vals[0]),
        angle=np.degrees(np.arctan2(vecs[1,1],vecs[0,1])),fill=False,lw=2.2,edgecolor=color,label=label))

def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    new=read(SCI/'noise_validation/NOISE_GATE.json');old=read(OLD/'noise_validation/NOISE_GATE.json')
    complete=read(SCI/'COMPLETE.json');replay=read(RUN/'runs/replay/REPLAY_VERIFICATION.json')
    assert complete['status']=='complete' and complete['noise_training_updates']==5000 and complete['classifier_training_updates']==1200
    assert complete['response_training_updates']==0 and complete['response_integration_allowed'] is False
    for name,want in complete['files'].items():
        assert digest(SCI/name)==want,'Bound scientific payload changed: '+name
    assert replay['status']=='reopened_and_replayed' and replay['complete_sha256']==digest(SCI/'COMPLETE.json')
    assert replay['generator_rows_replayed']==1024 and replay['classifier_score_rows_replayed']==3072
    assert replay['all_REAL_TRAIN_rows_visited']==21047 and replay['sampler_total_rows']==5000*512
    assert replay['generated_evaluation_populations_bitwise_equal'] and replay['AUC_and_all_six_intervals_recomputed']
    assert all(x['bitwise_equal'] for x in replay['classifiers'])
    assert all(x['pre_ADC_bitwise'] and x['post_ADC_bitwise'] for x in replay['generator_chunks'])
    assert replay['noise_training_updates_added']==0 and replay['classifier_updates_added']==0
    assert new['passed']==replay['noise_gate_passed'] and new['failed_checks']==replay['failed_checks']
    logs=[json.loads(x) for x in (SCI/'noise/training.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(logs)==5000 and [x['update'] for x in logs]==list(range(1,5001))
    with np.load(SCI/'noise_validation/baseline_populations.npz',allow_pickle=False) as z:
        fake=z['generated'];real=z['REAL_validation']
    with np.load(OLD/'noise_validation/baseline_populations.npz',allow_pickle=False) as z:
        prev=z['generated'];assert np.array_equal(real,z['REAL_validation'])
    scale=np.maximum(np.asarray(new['statistics']['TRAIN']['time_std_V']),1/16383)
    denom=scale[:,None]*scale[None,:]
    cov={k:np.cov(x,rowvar=False)/denom for k,x in [('REAL',real),('v2',prev),('v3',fake)]}
    errors={k:float(np.linalg.norm(cov[k]-cov['REAL'])/np.linalg.norm(cov['REAL'])) for k in ('v2','v3')}
    correlations={k:rho(s['ols_covariance_V2']) for k,s in [('REAL',new['statistics']['REAL']),('v2',old['statistics']['generated']),('v3',new['statistics']['generated'])]}
    candidate_failed=[k for k,v in new['checks'].items() if not v and not k.startswith('negative_control_')]
    control_failed=[k for k,v in new['checks'].items() if not v and k.startswith('negative_control_')]
    summary=dict(run_id=ID,status='bounded_noise_candidate_evaluated',noise_training_updates=5000,classifier_evaluation_updates=1200,
        response_training_updates=0,old_metrics=old['metrics'],new_metrics=new['metrics'],noise_gate_passed=new['passed'],
        failed_checks=new['failed_checks'],candidate_failed_checks=candidate_failed,control_failed_checks=control_failed,
        mean_slope_correlations=correlations,descriptive_normalized_covariance_errors=errors,
        all_TRAIN_rows_used=21047,validation_rows=599,observed_acquisition_sources=1,independent_acquisition_validations=0,validation_previously_viewed=True,
        model_parameters=5151,supported_samples=100,full480_integration_supported=False,
        generator_replay_rows=replay['generator_rows_replayed'],classifier_replay_rows=replay['classifier_score_rows_replayed'],
        replay_does_not_establish_scientific_acceptance=True)
    write(OUT/'SUMMARY.json',summary)
    colors=dict(REAL='#CA5334',v2='#9A6BA0',v3='#147CAC')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,ax=plt.subplots(2,2,figsize=(12.8,8.8),constrained_layout=True)
    for k,s in [('REAL',new['statistics']['REAL']),('v2',old['statistics']['generated']),('v3',new['statistics']['generated'])]:
        ax[0,0].plot(np.arange(100),np.asarray(s['time_std_V'])*1000,color=colors[k],label=k)
        ellipse(ax[0,1],s['ols_covariance_V2'],colors[k],f'{k}: rho={correlations[k]:.3f}')
    ax[0,0].set(title='Time-dependent baseline SD',xlabel='Aligned baseline sample (1 ns)',ylabel='Ensemble SD (mV)')
    ax[0,0].legend();ax[0,0].grid(alpha=.15)
    ax[0,1].set(xlim=(-.2,.2),ylim=(-.16,.16),aspect='equal',title='Centered mean-slope covariance',xlabel='Mean fluctuation (mV)',ylabel='OLS slope fluctuation (mV)')
    ax[0,1].legend(fontsize=9);ax[0,1].grid(alpha=.15)
    delta={k:cov[k]-cov['REAL'] for k in ('v2','v3')}
    lim=max(np.abs(delta[k]).max() for k in delta)
    for j,k in enumerate(('v2','v3')):
        im=ax[1,j].imshow(delta[k],vmin=-lim,vmax=lim,cmap='RdBu_r',origin='lower')
        ax[1,j].set(title=f'{k} - REAL: normalized covariance\nRelative Frobenius error {errors[k]:.3f}',xlabel='Baseline sample',ylabel='Baseline sample')
        fig.colorbar(im,ax=ax[1,j],shrink=.8,label='Fixed TRAIN SD normalization')
    save(fig,'01_nonstationary_covariance')
    fig,ax=plt.subplots(2,2,figsize=(12.8,8.5),constrained_layout=True)
    for k,s in [('REAL',new['statistics']['REAL']),('v2',old['statistics']['generated']),('v3',new['statistics']['generated'])]:
        ax[0,0].plot(np.arange(1,21),s['acf_mean'],'o-',ms=3,color=colors[k],label=k)
        ax[0,1].plot([.1,.5,.9],np.asarray(s['rms_quantiles_V'])*1000,'o-',color=colors[k],label=k)
        ax[1,0].plot(np.arange(1,6),s['psd_band_fraction_mean'],'o-',color=colors[k],label=k)
    ax[0,0].set(title='Demeaned baseline ACF',xlabel='Lag (samples)',ylabel='Mean event ACF');ax[0,0].legend()
    ax[0,1].set(title='Baseline RMS distribution',xlabel='Quantile',ylabel='RMS (mV)')
    ax[1,0].set(title='Spectral power fractions',xlabel='Registered frequency band',ylabel='Fraction')
    for i,view in enumerate(('raw','demean','detrend')):
        for j,(g,case,k) in enumerate([(old,'generated_REAL','v2'),(new,'generated_REAL','v3'),(new,'REAL_REAL','REAL')]):
            r=g['classifiers'][view][case];lo,hi=r['percentile_95_interval'];y=2-i+(j-1)*.19
            ax[1,1].plot([lo,hi],[y,y],color=colors[k],lw=1.5)
            ax[1,1].plot(r['AUC'],y,'o',color=colors[k],ms=5,label={'REAL':'REAL-REAL control','v2':'v2-REAL','v3':'v3-REAL'}[k] if i==0 else None)
    ax[1,1].axvline(.5,color='#333333',ls='--',lw=1)
    ax[1,1].set(title='Frozen classifier test and control',yticks=[0,1,2],yticklabels=['Detrend','Demean','Raw'],xlabel='AUC; 95% record-group bootstrap')
    ax[1,1].legend(fontsize=9,loc='best')
    for a in ax.flat:a.grid(alpha=.15)
    save(fig,'02_baseline_distributions')
    fig,ax=plt.subplots(1,2,figsize=(12.8,4.6),constrained_layout=True)
    step=np.array([x['update'] for x in logs]);loss=np.array([x['total'] for x in logs])
    ax[0].plot(step,loss,color=colors['v3'],alpha=.18,lw=.5)
    ax[0].plot(step[99:],np.convolve(loss,np.ones(100)/100,mode='valid'),color=colors['v3'],label='100-step moving mean')
    ax[0].axvline(4000,color='#333333',ls='--',lw=1);ax[0].set(title='One fixed 5,000-update candidate',xlabel='Noise optimizer update',ylabel='Unchanged baseline objective');ax[0].legend(fontsize=9)
    keys=['max_mean_acf_difference','ols_covariance_relative_error'];thresholds=[.015,.2]
    pos=np.arange(2)
    for j,(k,g) in enumerate([('v2',old),('v3',new)]):
        values=np.array([g['metrics'][key] for key in keys])/thresholds
        bars=ax[1].bar(pos+(j-.5)*.32,values,width=.3,color=colors[k],label=k)
        ax[1].bar_label(bars,fmt='%.2f',padding=3)
    ax[1].axhline(1,color='#333333',ls='--',label='Original tolerance')
    ax[1].set(title='Original structural checks',xticks=pos,xticklabels=['ACF error','OLS covariance error'],ylabel='Observed error / fixed tolerance');ax[1].legend(fontsize=9)
    for a in ax:a.grid(axis='y',alpha=.15)
    save(fig,'03_fixed_training_and_checks')
    records=[('最大 ACF 差','max_mean_acf_difference','≤ 0.015','acf'),('OLS 联合协方差误差','ols_covariance_relative_error','≤ 0.20','ols_joint'),
        ('基线均值标准差比','mean_std_ratio','0.85–1.15','mean_scatter'),('斜率标准差比','ols_slope_std_ratio','0.85–1.15','ols_slope'),('完整 100 点 SWD 比','full100_independent_SWD_ratio','≤ 2.0','full100')]
    rows=[]
    for title,key,tolerance,check in records:
        rows.append([title,f'{old["metrics"][key]:.6f}',f'{new["metrics"][key]:.6f}',tolerance,'通过' if new['checks'][check] else '未通过'])
    table=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
    raw=new['classifiers']['raw']['generated_REAL'];rr=new['classifiers']['raw']['REAL_REAL']
    candidate='全部通过' if not candidate_failed else '未通过：'+', '.join(candidate_failed)
    body=f'''<h1>N01 噪声改进 004</h1>
<p>固定候选：100 点非平稳协方差 · 2026-10-12 · 对照 v2 原终点</p>
<div class="lead"><b>候选自身检查：{html.escape(candidate)}；总噪声门：{'通过' if new['passed'] else '未通过'}。</b><br>保留原 REAL–REAL 阴性对照判据。仅支持已观测的 100 点基线；480 点噪声未获验证，本轮脉冲训练为 0。</div>
<div class="stats"><div><b>5,000</b>实际噪声优化更新</div><div><b>21,047 / 21,047</b>TRAIN 行覆盖</div><div><b>0</b>脉冲响应更新</div></div>
<h2>改动与结果</h2><p>将平稳 FIR 与独立偏置替换为完整可训练的 100×100 Cholesky 协方差、100 点均值和一个事件尺度。全部 5,151 个参数可训练；尺度满足 E[S²]=1，协方差为 LLᵀ。损失、评价种子和原阈值保持一致，仅使用固定第 5,000 步。</p>
<table><thead><tr><th>检查</th><th>旧 v2</th><th>新 v3</th><th>原容差</th><th>新结果</th></tr></thead><tbody>{table}</tbody></table>
<p>均值—斜率相关系数：REAL {correlations['REAL']:.4f}，v2 {correlations['v2']:.4f}，v3 {correlations['v3']:.4f}。完整 100 点协方差描述性误差：v2 {errors['v2']:.4f}，v3 {errors['v3']:.4f}。后者是额外诊断，不是新增验收门。</p>
<img src="figures/01_nonstationary_covariance.png" alt="时间标准差、均值斜率协方差及完整协方差差值">
<h2>分布与阴性对照</h2><p>新模型–REAL 原始基线 AUC 为 {raw['AUC']:.5f}，95% 区间 [{raw['percentile_95_interval'][0]:.5f}, {raw['percentile_95_interval'][1]:.5f}]。REAL–REAL 对照为 {rr['AUC']:.5f} [{rr['percentile_95_interval'][0]:.5f}, {rr['percentile_95_interval'][1]:.5f}]。旧门的“阴性区间包含 0.5”条件仍按原样执行，未改判历史结果。</p>
<img src="figures/02_baseline_distributions.png" alt="ACF、RMS、频谱与分类器检查">
<h2>执行与复核</h2><p>噪声训练 5,000 次；评价分类器另计 1,200 次；脉冲训练 0 次。最终模型保存后才读取本轮验证波形。重开并重放 1,024 条生成基线、6 个分类器共 3,072 个分数；重放只证明数值可复现。完整回执见下方链接。</p>
<img src="figures/03_fixed_training_and_checks.png" alt="固定训练损失和原有结构检查">
<h2>适用范围</h2><p>结果针对既有采集、筛选、量化和对齐后的观测基线。599 条验证记录来自同一份此前已查看的采集，12 个记录组不构成独立采集。该模型不识别电子学硬件来源，也不能自动外推后续 380 点。高斯全局尺度混合仍限制偏斜、方向相关尾部和多模态；分类器接近随机不等于证明分布完全相同。</p>
<p>本轮停止在已登记的噪声候选终点。下一步应根据剩余失败项和实际可用数据另行绑定完整 480 点的噪声模型，不能把短窗拟合直接拼成整段响应。</p>
<p class="links"><a href="SUMMARY.json">指标摘要</a><a href="NOISE_GATE.json">完整原门结果</a><a href="REPLAY_VERIFICATION.json">重放回执</a><a href="../../docs/NOISE_IMPROVEMENT_004_PROTOCOL.md">预登记协议</a></p>'''
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>N01 噪声改进 004</title><style>body{font-family:"Microsoft YaHei",system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 24px;color:#183044;line-height:1.75;background:#f7f9fc}h1{font-size:30px;line-height:1.35}h2{margin-top:32px;font-size:22px}p{font-size:16px}.lead{background:#fff1e8;border-left:5px solid #b66328;padding:16px 20px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}.stats div{padding:18px;background:white;border:1px solid #d8e2ed;border-radius:8px}.stats b{display:block;font-size:25px}table{width:100%;border-collapse:collapse;background:white;font-size:15px}th,td{padding:10px 12px;border:1px solid #d7e1eb;text-align:left}img{width:100%;height:auto;margin-top:16px;background:white}a{color:#086ba0}.links{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:700px){.stats{grid-template-columns:1fr}body{padding:0 12px}table{font-size:12px}th,td{padding:6px}}</style>'''+body+'</html>'
    (OUT/'N01_NOISE_IMPROVEMENT_REPORT.html').write_text(page,encoding='utf8')
    for name,path in [('NOISE_GATE.json',SCI/'noise_validation/NOISE_GATE.json'),('REPLAY_VERIFICATION.json',RUN/'runs/replay/REPLAY_VERIFICATION.json'),('DESCRIPTIVE_METRICS.json',SCI/'noise_validation/DESCRIPTIVE_METRICS.json'),('COMPLETE.json',SCI/'COMPLETE.json')]:shutil.copy2(path,OUT/name)
    (OUT/'README.md').write_text(f'# N01 噪声改进 004\n\n[打开中文报告](N01_NOISE_IMPROVEMENT_REPORT.html)。\n\n实际噪声更新 5000、分类器评价更新 1200、响应更新 0。候选自身检查：{candidate}；总门通过：{new["passed"]}。仅支持 100 点观测基线，不支持 480 点响应整合。\n\n[预登记协议](../../docs/NOISE_IMPROVEMENT_004_PROTOCOL.md) · [指标摘要](SUMMARY.json) · [完整原门](NOISE_GATE.json) · [重放回执](REPLAY_VERIFICATION.json)\n',encoding='utf8')
    write(OUT/'BUILD_RECEIPT.json',dict(generator_sha256=digest(Path(__file__)),input_gate_sha256=digest(SCI/'noise_validation/NOISE_GATE.json'),old_gate_sha256=digest(OLD/'noise_validation/NOISE_GATE.json'),figures=3,new_fits=0,visual_QA='pending'))
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
