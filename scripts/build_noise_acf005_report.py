"""Build a bounded 005 scientific report only after completed and replayed work."""
from pathlib import Path
import os
import hashlib
import json
import html
import datetime as dt
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ID='noise_acf005_20261012'
RUN=ROOT/'runs'/ID
SCIENCE=RUN/'runs/science'
OUT=ROOT/'reports'/ID
os.environ['MPLCONFIGDIR']='E:/GAN/runtime_tmp/matplotlib_acf005'
sys.path.insert(0,str(ROOT/'runtime/plotting_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

COLORS=dict(TRAIN='#222d36',initial='#788caa',v004='#d36b47',v005='#008782',pre='#7562a1')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
    'axes.labelsize':11,'legend.fontsize':10,'figure.facecolor':'white','axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'savefig.facecolor':'white'})


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=160,bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)
def smooth(values,width=50):
    a=np.asarray(values,dtype=float);return np.arange(width/2,len(a)+1,width),np.array([np.nanmedian(a[i:i+width]) for i in range(0,len(a),width)])
def ols(x):
    t=np.linspace(-1,1,100);mean=x.mean(1);slope=(x-mean[:,None])@t/(t@t)
    return np.stack([mean,slope],axis=1)
def ellipse(ax,values,scale,color,label):
    z=values/scale;center=z.mean(0);cov=np.cov(z,rowvar=False)
    eigen,vectors=np.linalg.eigh(cov);i=np.argsort(eigen)[::-1];eigen=eigen[i];vectors=vectors[:,i]
    angle=np.degrees(np.arctan2(vectors[1,0],vectors[0,0]))
    ax.add_patch(Ellipse(center,width=2*np.sqrt(eigen[0]),height=2*np.sqrt(eigen[1]),angle=angle,
        fill=False,color=color,lw=2,label=label))
    ax.plot(center[0],center[1],'.',color=color)


def main():
    complete=read(SCIENCE/'COMPLETE.json')
    replay=read(RUN/'runs/replay/REPLAY_VERIFICATION.json')
    assert complete['status']=='complete' and complete['noise_training_updates']==5000 and complete['response_training_updates']==0
    assert replay['complete_sha256']==sha(SCIENCE/'COMPLETE.json')
    assert replay['generated_evaluation_populations_bitwise_equal'] and replay['acf005']['all_24576_paired_rows_bitwise']
    for name,digest in complete['files'].items():assert sha(SCIENCE/name)==digest,name
    gate=read(SCIENCE/'noise_validation/LEGACY_GATE.json')
    scoped=read(SCIENCE/'noise_validation/NOISE_GATE.json')
    evaluation=read(SCIENCE/'noise_validation/EVALUATION_RECEIPT.json')
    assert evaluation['implementation_checks']['pending_runner_binding'] is False
    candidate=evaluation['candidate_checks'];decision=evaluation['decision']
    old=read(ROOT/'provenance/noise_acf005_reference/gate004.json')
    old2=read(ROOT/'runs/repair_002_20261012/runs/science/noise_validation/NOISE_GATE.json')
    table=read(SCIENCE/'noise_validation/ACF_PER_LAG.json')
    uncertainty=read(SCIENCE/'noise_validation/ACF_UNCERTAINTY.json')
    logs=[json.loads(line) for line in (SCIENCE/'noise/training.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    assert len(logs)==5000 and [v['update'] for v in logs]==list(range(1,5001))
    with np.load(SCIENCE/'diagnostics/paired_populations.npz',allow_pickle=False) as z:pairs={k:z[k] for k in z.files}
    with np.load(SCIENCE/'diagnostics/trajectory.npz',allow_pickle=False) as z:trajectory={k:z[k] for k in z.files}
    with np.load(SCIENCE/'noise_validation/RAW_CONTROL_CONDITIONAL.npz',allow_pickle=False) as z:permutations={k:z[k] for k in z.files}
    calibration003=read(ROOT/'provenance/noise_acf005_reference/calibration003.json')
    comparison=read(SCIENCE/'diagnostics/005A_COMPARISON.json')
    trajectory_rows=[json.loads(line) for line in (SCIENCE/'diagnostics/trajectory.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    final_diag=trajectory_rows[-1]['post_adc'];old_diag=comparison['v004']['post_adc']
    grad_angles=np.array([row['old_vs_acf']['angle_degrees'] for row in logs],dtype=float)
    grad_ratios=np.array([row['acf20_gradient_norm']/row['old_gradient_norm'] for row in logs],dtype=float)
    mechanism=dict(initial_post_ADC=comparison['initial']['post_adc'],v004_post_ADC=old_diag,v005_post_ADC=final_diag,
        full_covariance_TRAIN_relative_error_increased=final_diag['TRAIN_scaled_fullcov_relative_error']>old_diag['TRAIN_scaled_fullcov_relative_error'],
        gradient_angle_median_degrees=float(np.nanmedian(grad_angles)),
        gradient_angle_above90_fraction=float(np.mean(grad_angles>90)),
        added_to_old_gradient_norm_ratio_median=float(np.median(grad_ratios)),
        interpretation='ACF improved;full covariance did not improve;gradient magnitudes alone do not establish causal conflict or model-family impossibility',
        covariance_is_descriptive_not_added_gate=True)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    lags=np.arange(1,21);series={k:np.asarray(v['mean']) for k,v in table['series'].items()}
    target=series['TRAIN_target']
    fig,axes=plt.subplots(2,2,figsize=(13.6,9),layout='constrained')
    ax=axes[0,0]
    ax.plot(lags,target,'o-',color=COLORS['TRAIN'],ms=3,label='All observed TRAIN')
    for key,label in [('initial','Untrained initialization'),('v004','004 final'),('v005','005 final')]:
        ax.plot(lags,series[key+'_post_adc'],marker='.',color=COLORS[key],label=label)
    ax.set(title='Same explicit diagnostic latents: post-ADC ACF',xlabel='Lag (samples)',ylabel='Mean eventwise ACF')
    ax.legend(loc='lower left',frameon=False)
    ax=axes[0,1]
    for key,label in [('initial','Initialization'),('v004','004'),('v005','005')]:
        ax.plot(lags,series[key+'_post_adc']-target,'o-',ms=3,color=COLORS[key],label=label)
    for limit in [-.015,.015]:ax.axhline(limit,color='#555',ls='--',lw=1)
    ax.axhline(0,color='#777',lw=.6)
    ax.set(title='Diagnostic ACF difference from full TRAIN',xlabel='Lag (samples)',ylabel='Generated minus TRAIN')
    ax.legend(frameon=False)
    ax=axes[1,0]
    olddiff=np.asarray(old['statistics']['generated']['acf_mean'])-np.asarray(old['statistics']['REAL']['acf_mean'])
    newdiff=series['legacy_generated_post_adc']-series['REAL_validation_post_adc']
    intervals=np.asarray(uncertainty['development']['marginal95']).T
    ax.plot(lags,olddiff,'o-',ms=3,color=COLORS['v004'],label='004 development')
    ax.errorbar(lags,newdiff,yerr=np.maximum(np.stack([newdiff-intervals[:,0],intervals[:,1]-newdiff]),0),fmt='o-',ms=3,
        color=COLORS['v005'],capsize=2,label='005; marginal 95% group bootstrap')
    for limit in [-.015,.015]:ax.axhline(limit,color='#555',ls='--',lw=1)
    ax.set(title='Legacy evaluation population versus development REAL',xlabel='Lag (samples)',ylabel='Generated minus development REAL')
    ax.legend(frameon=False,loc='lower left')
    ax=axes[1,1]
    traincov=np.asarray(trajectory['TRAIN_covariance_V2']);t=np.linspace(-1,1,100)
    operator=np.stack([np.ones(100)/100,t/(t@t)])
    trainols=operator@traincov@operator.T;scale=np.sqrt(np.diag(trainols))
    # Absolute centers are retained; only coordinate scales come from TRAIN.
    for key,label in [('initial','Initialization'),('v004','004'),('v005','005')]:
        ellipse(ax,ols(pairs[key+'_post_adc']),scale,COLORS[key],label)
    ax.autoscale_view();ax.set_aspect('equal',adjustable='datalim')
    ax.set(title='Mean/slope covariance contours (one SD)',xlabel='Baseline mean / TRAIN SD',ylabel='OLS slope / TRAIN SD')
    ax.legend(frameon=False)
    for ax in axes.flat:
        if ax is not axes[1,1]:ax.set_xticks([1,4,8,12,16,17,19,20])
    save(fig,'01_acf_and_joint_statistics')

    updates=np.asarray(trajectory['updates']);sd=np.maximum(np.sqrt(np.diag(traincov)),1/16383)
    norm=sd[:,None]*sd[None,:];covden=np.linalg.norm(traincov/norm)
    olsnorm=scale[:,None]*scale[None,:];olsden=np.linalg.norm(trainols/olsnorm)
    fig,axes=plt.subplots(2,2,figsize=(13.6,8.5),layout='constrained')
    for adc,label,color in [('pre','Pre-ADC',COLORS['pre']),('post','Post-ADC',COLORS['v005'])]:
        acf=np.asarray(trajectory[adc+'_ACF20']);covs=np.asarray(trajectory[adc+'_covariance_V2'])
        axes[0,0].plot(updates,np.max(np.abs(acf-target),axis=1),'o-',ms=3,color=color,label=label)
        errors=np.array([np.linalg.norm((c-traincov)/norm)/covden for c in covs])
        axes[0,1].plot(updates,errors,'o-',ms=3,color=color,label=label)
        ocovs=np.array([operator@c@operator.T for c in covs])
        oe=np.array([np.linalg.norm((c-trainols)/olsnorm)/olsden for c in ocovs])
        axes[1,0].plot(updates,oe,'o-',ms=3,color=color,label=label)
        if adc=='post':
            ratios=np.sqrt(np.diagonal(ocovs,axis1=1,axis2=2))/scale
            axes[1,1].plot(updates,ratios[:,0],color=COLORS['v005'],label='Mean SD / TRAIN')
            axes[1,1].plot(updates,ratios[:,1],color=COLORS['pre'],label='Slope SD / TRAIN')
    axes[0,0].axhline(.015,color='#555',ls='--',lw=1,label='Declared TRAIN ACF tolerance')
    axes[0,0].set(title='Fixed-latent TRAIN ACF maximum error',ylabel='Max over all 20 lags')
    axes[0,1].set(title='Full covariance difference from TRAIN (descriptive)',ylabel='TRAIN-SD-normalized relative error')
    axes[1,0].set(title='OLS joint covariance difference (descriptive)',ylabel='TRAIN-scaled relative error')
    axes[1,1].axhline(1,color='#555',ls='--',lw=1)
    axes[1,1].set(title='Post-ADC marginal scatter (descriptive)',ylabel='SD ratio')
    for ax in axes.flat:ax.set_xlabel('Completed optimizer updates');ax.legend(frameon=False)
    save(fig,'02_fixed_latent_trajectory')

    step=np.arange(1,5001);fig,axes=plt.subplots(2,2,figsize=(13.6,8.5),layout='constrained')
    for key,label,color in [('old_total','Original objective',COLORS['v004']),('acf20_loss','Added ACF20',COLORS['v005']),('total','Total',COLORS['TRAIN'])]:
        values=[row[key] for row in logs];x,y=smooth(values)
        axes[0,0].plot(x,y,color=color,label=label)
    axes[0,0].set_yscale('log');axes[0,0].set(title='Objective components: 50-step display medians',ylabel='Loss')
    for key,label,color in [('old_gradient_norm','Original',COLORS['v004']),('acf20_gradient_norm','ACF20',COLORS['v005']),('total_gradient_preclip_norm','Total before clip',COLORS['TRAIN'])]:
        x,y=smooth([row[key] for row in logs]);axes[0,1].plot(x,y,color=color,label=label)
    axes[0,1].axhline(5,color='#555',ls='--',lw=1,label='Unchanged clipping norm')
    axes[0,1].set_yscale('log');axes[0,1].set(title='Parameter gradient norms: 50-step medians',ylabel='L2 norm')
    for key,label,color in [('old_vs_acf','Original versus added ACF20',COLORS['v005']),('old_vs_total','Original versus total',COLORS['pre'])]:
        values=[np.nan if row[key]['angle_degrees'] is None else row[key]['angle_degrees'] for row in logs]
        x,y=smooth(values);axes[1,0].plot(x,y,color=color,label=label)
    axes[1,0].axhline(90,color='#555',ls='--',lw=1)
    axes[1,0].set(title='Pre-clipping gradient angles: 50-step medians',ylabel='Angle (degrees)',ylim=(0,180))
    batchmax=np.max(np.abs(np.array([row['acf20_signed_errors'] for row in logs])),axis=1)
    axes[1,1].plot(step,batchmax,color=COLORS['v005'],alpha=.22,lw=.5,label='Every 512-event batch')
    x,y=smooth(batchmax);axes[1,1].plot(x,y,color=COLORS['TRAIN'],label='50-step median')
    axes[1,1].set(title='Training batch ACF error; not the endpoint screen',ylabel='Maximum absolute error over 20 lags')
    for ax in axes.flat:ax.set_xlabel('Optimizer update');ax.legend(frameon=False)
    save(fig,'03_losses_and_gradients')

    fig,axes=plt.subplots(1,3,figsize=(15.4,4.6),layout='constrained')
    ax=axes[0];ax.axvspan(.35,.65,color=COLORS['v005'],alpha=.08);ax.axvline(.5,color='#777',ls='--',lw=1)
    for i,view in enumerate(['raw','demean','detrend']):
        for delta,case,color,label in [(-.12,'generated_REAL',COLORS['v005'],'Generated–REAL'),(.12,'REAL_REAL',COLORS['v004'],'REAL–REAL')]:
            obj=gate['classifiers'][view][case];v=obj['AUC'];lo,hi=obj['percentile_95_interval']
            ax.errorbar(v,i+delta,xerr=[[v-lo],[hi-v]],fmt='o',color=color,capsize=3,label=label if i==0 else None)
    ax.set_yticks(range(3),['Raw (screen)','Demeaned','Detrended']);ax.invert_yaxis()
    ax.set(title='Fixed classifiers and group intervals',xlabel='AUC');ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.04))
    ax=axes[1]
    for i,(key,label,color) in enumerate([('legacy_joint_bootstrap_false_positive','Legacy interval',COLORS['v004']),('conditional_permutation_false_positive','255-draw permutation',COLORS['pre'])]):
        v=calibration003[key];rate=100*v['rate'];lo,hi=np.array(v['Wilson95'])*100
        ax.errorbar(i,rate,yerr=[[rate-lo],[hi-rate]],fmt='o',color=color,capsize=6,ms=8)
        ax.text(i,hi+.7,f"{v['rejections']}/128",ha='center',color=color)
    ax.axhline(5,color='#555',ls='--',lw=1);ax.set_xticks([0,1],['Legacy interval','Permutation\n(255 draws)'])
    ax.set(title='003 reference: 128 refits, unchanged',ylabel='Conditional false alarm rate (%)',xlim=(-.5,1.5),ylim=(0,16))
    ax=axes[2];values=permutations['permutation_AUC'];observed=float(permutations['observed_AUC'])
    tail=int(np.sum(np.abs(values-.5)>=abs(observed-.5)));p=(tail+1)/(len(values)+1)
    ax.hist(values,bins=40,color=COLORS['pre'],alpha=.8)
    ax.axvline(observed,color=COLORS['v004'],lw=2,label=f'Observed AUC {observed:.4f}')
    ax.set(title=f'Current fixed-score control: 9999 draws\nConditional p = {p:.4f}',xlabel='Permuted AUC',ylabel='Count')
    ax.legend(frameon=False)
    save(fig,'04_candidate_and_control')

    m=gate['metrics'];om=old['metrics'];o2m=old2['metrics']
    raw=gate['classifiers']['raw']['generated_REAL'];rr=gate['classifiers']['raw']['REAL_REAL']
    newdevdiff=series['legacy_generated_post_adc']-series['REAL_validation_post_adc']
    passed=candidate['passed'];scope_status='候选开发筛查满足要求' if passed else '候选开发筛查仍有失败项'
    design=decision['read_only_480_design_check_allowed']
    summary=dict(run_id=ID,noise_training_updates=5000,classifier_evaluation_updates=1200,response_training_updates=0,
        TRAIN_match=decision['TRAIN_match'],TRAIN_improved=decision['TRAIN_improved'],
        candidate_checks_passed=passed,candidate_failed_checks=candidate['failed_checks'],
        legacy_gate_passed=gate['passed'],legacy_failed_checks=gate['failed_checks'],
        control_status='measurement_uncertain_single_exposed_acquisition',
        read_only_480_design_check_allowed=design,formal_response_training_authorized=False,
        supported_samples=100,full480_integration_supported=False,independent_acquisition_validations=0,
        metrics_002=o2m,metrics_004=om,metrics_005=m,raw_generated_REAL=raw,raw_REAL_REAL=rr,
        TRAIN_current_max_ACF_error=decision['TRAIN_current_max_ACF_error'],TRAIN004_max_ACF_error=decision['TRAIN004_max_ACF_error'],
        development_worst_ACF_lag=int(np.argmax(np.abs(newdevdiff))+1),
        development_lags_exceeding_015=[int(k) for k,e in zip(lags,newdevdiff) if abs(e)>.015],
        all_20_development_errors=newdevdiff.tolist(),conditional_control_permutation_p=p,
        TRAIN_full_covariance_relative_error_004=old_diag['TRAIN_scaled_fullcov_relative_error'],
        TRAIN_full_covariance_relative_error_005=final_diag['TRAIN_scaled_fullcov_relative_error'],
        TRAIN_OLS_covariance_relative_error_004=old_diag['TRAIN_scaled_OLS_relative_error'],
        TRAIN_OLS_covariance_relative_error_005=final_diag['TRAIN_scaled_OLS_relative_error'],
        development_lag7_marginal95=[uncertainty['development']['marginal95'][j][6] for j in [0,1]],
        replay_summary=dict(explicit_generator_rows=1024,paired_diagnostic_rows=24576,evaluation_generated_rows=10240,
            classifier_score_rows=3072,bootstrap_draws=2000,conditional_permutations=9999,added_fits=0),
        visual_QA='pending')
    write(OUT/'SUMMARY.json',summary)
    write(OUT/'MECHANISM_ANALYSIS.json',mechanism)
    copies=['LEGACY_GATE.json','NOISE_GATE.json','EVALUATION_RECEIPT.json','ACF_PER_LAG.json','ACF_PER_LAG.csv','ACF_UNCERTAINTY.json','CONTROL_CALIBRATION.json']
    import shutil
    for name in copies:shutil.copy2(SCIENCE/'noise_validation'/name,OUT/name)
    shutil.copy2(RUN/'runs/replay/REPLAY_VERIFICATION.json',OUT/'REPLAY_VERIFICATION.json')
    rows=[]
    for label,key,threshold in [('ACF 最大绝对差','max_mean_acf_difference','≤ 0.015'),('OLS 联合协方差相对误差','ols_covariance_relative_error','≤ 0.20'),
                                ('基线均值散布比','mean_std_ratio','0.85–1.15'),('OLS 斜率散布比','ols_slope_std_ratio','0.85–1.15'),
                                ('完整 100 点投影距离比','full100_independent_SWD_ratio','≤ 2.0')]:
        rows.append(f'<tr><td>{label}</td><td>{o2m[key]:.6f}</td><td>{om[key]:.6f}</td><td><b>{m[key]:.6f}</b></td><td>{threshold}</td></tr>')
    checkrows=''.join(f'<tr><td>{html.escape(key)}</td><td>{"满足" if value else "未满足"}</td></tr>' for key,value in candidate['checks'].items())
    scientific='本轮最多支持 100 点观测噪声候选的开发筛查结论。旧阴性区间失败保留；当前测量状态仍不确定，未开展独立采集验证，正式脉冲训练为 0。'
    designtext=('已满足进入只读 480 点接口设计检查的条件；设计检查结果另见范围说明。' if design else '本轮未满足进入 480 点接口设计检查的条件；该固定候选已停止，不增加步数或改种子。')
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>N01 005 ACF 目标覆盖修复</title>
<style>body{{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#f3f6f8;color:#192d3d;margin:0}}main{{max-width:1120px;margin:auto;padding:32px}}h1{{font-size:30px}}h2{{font-size:23px;border-bottom:2px solid #d8e3e9;padding-bottom:9px}}p,li{{font-size:16px;line-height:1.8}}section,.lead{{background:white;border:1px solid #dbe5eb;border-radius:12px;padding:22px 26px;margin:20px 0}}.lead{{border-left:6px solid #008782}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card{{background:#e9f1f4;border-radius:8px;padding:16px;font-size:15px}}.card b{{display:block;font-size:26px;margin-bottom:8px}}table{{border-collapse:collapse;width:100%;font-size:15px;margin:16px 0}}th,td{{padding:11px;border-bottom:1px solid #dce6eb;text-align:left}}th{{background:#edf3f6}}img{{max-width:100%;height:auto;display:block}}a{{color:#006f91}}code{{overflow-wrap:anywhere}}.status{{font-weight:700;color:#006963}}@media(max-width:800px){{main{{padding:16px}}section{{padding:16px}}.grid{{grid-template-columns:1fr 1fr}}table{{font-size:12px}}th,td{{padding:6px}}}}</style><main>
<h1>N01 005：ACF 目标覆盖修复</h1><div class="lead"><p class="status">{scope_status}；legacy 总门：{"满足" if gate['passed'] else "仍未通过"}</p><p>{scientific}</p><p>改善存在代价：同一诊断群对 TRAIN 的完整协方差相对误差由 {old_diag['TRAIN_scaled_fullcov_relative_error']:.4f} 增至 {final_diag['TRAIN_scaled_fullcov_relative_error']:.4f}，不是所有联合统计都变好。该项预先仅作机制描述，没有事后增设通过门。</p><p>{designtext}</p></div>
<div class="grid"><div class="card"><b>5,000</b>本轮实际噪声更新</div><div class="card"><b>21,047</b>全部合法 TRAIN 事件</div><div class="card"><b>1,200</b>评价分类器更新（另计）</div><div class="card"><b>0</b>正式脉冲响应更新</div></div>
<section><h2>问题、唯一改动与固定终点</h2><p>保留 004 的 ObservedBaselineNoiseV3、全 TRAIN 初始化、随机流、512 批量、Adam 和 5000 次固定预算。原 32 维分布损失保持不变，只新增全 20 阶 ACF 目标：<code>0.5 × (mean(e²) + max(e²))</code>，其中 <code>e=(生成批次逐事件 ACF 均值 − 全 TRAIN 固定目标)/0.015</code>。每事件先减均值、按 100−lag 求乘积均值，再除以该事件方差并保留原 ADC floor。</p><p>初始化和 004、005 末点共用独立固定的 8192 个显式随机输入；训练、诊断和评价随机流分开。TRAIN 只有处理后量化观测，没有伪造的模拟前真值。初始化仅作机制对照，未选作替代末点。</p>
<table><thead><tr><th>开发验证指标</th><th>002</th><th>004</th><th>005</th><th>原要求</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>固定诊断群对 TRAIN 的最大 ACF 差：004 为 {decision['TRAIN004_max_ACF_error']:.6f}，005 为 {decision['TRAIN_current_max_ACF_error']:.6f}。TRAIN 匹配：{decision['TRAIN_match']}；相对 004 的预声明改善条件：{decision['TRAIN_improved']}。开发集超出 0.015 的 lag：{html.escape(str(summary['development_lags_exceeding_015']))}。</p><img src="figures/01_acf_and_joint_statistics.png" alt="全20阶ACF与基线联合统计"><p>图中 TRAIN 比较使用固定诊断群；开发集比较使用旧评价流程的另一固定生成群。误差条是边际 95% 分组 bootstrap；联合误差带另存于不确定性回执，均未用作追加验收门。椭圆只表示模拟样本的一倍标准差协方差轮廓。</p></section>
<section><h2>初始化与协方差轨迹</h2><img src="figures/02_fixed_latent_trajectory.png" alt="固定潜变量下的量化前后协方差与ACF轨迹"><p>每 250 次更新使用同一批随机输入，保存量化前后完整 100×100 协方差、OLS 联合统计、均值和 RMS。协方差误差使用冻结 TRAIN 的时间标准差进行归一化，属于机制描述，不是结果出现后新增的通过门。</p><p>未训练初始化的量化后 TRAIN 最大 ACF 差为 {comparison['initial']['post_adc']['max_abs_acf20_error_vsTRAIN']:.5f}，完整协方差相对误差为 {comparison['initial']['post_adc']['TRAIN_scaled_fullcov_relative_error']:.4f}。它说明协方差初始化本身不保证逐事件归一化 ACF 已匹配；初始化仅作对照，未选为替代终点。004→005 的 TRAIN OLS 联合误差为 {old_diag['TRAIN_scaled_OLS_relative_error']:.4f}→{final_diag['TRAIN_scaled_OLS_relative_error']:.4f}，基本未改善；这与上表开发群的 OLS 改善属于不同参照群。</p></section>
<section><h2>损失与梯度</h2><img src="figures/03_losses_and_gradients.png" alt="原损失新增ACF20及梯度夹角"><p>所有 5000 次更新的原始逐步数据均保留；图中 50 步中位数只便于显示。主夹角比较旧目标与新增 ACF20 目标，附加夹角比较旧目标与总目标；均在裁剪前、同一参数和批次下计算。零范数夹角记为未定义。小批量误差不用于代替固定终点评价。</p><p>新增项/旧目标梯度范数比的全程中位数为 {mechanism['added_to_old_gradient_norm_ratio_median']:.1f}；两者夹角中位数 {mechanism['gradient_angle_median_degrees']:.2f}°，超过 90° 的更新占 {100*mechanism['gradient_angle_above90_fraction']:.2f}%。新增项对更新方向影响很大，但这些描述不足以单独证明因果冲突或模型族极限。训练设置没有据此调整。</p></section>
<section><h2>候选质量、legacy 门和控制校准</h2><p>候选 10 项检查：{sum(bool(v) for v in candidate['checks'].values())}/10 满足；失败项：<code>{html.escape(str(candidate['failed_checks']))}</code>。legacy 失败项仍完整保留：<code>{html.escape(str(gate['failed_checks']))}</code>。</p><table><thead><tr><th>候选检查</th><th>结果</th></tr></thead><tbody>{checkrows}</tbody></table>
<p>原始 generated–REAL AUC={raw['AUC']:.6f}，95% 分组区间 [{raw['percentile_95_interval'][0]:.6f}, {raw['percentile_95_interval'][1]:.6f}]；实用容差筛查不等于统计分布相等。REAL–REAL AUC={rr['AUC']:.6f}，区间 [{rr['percentile_95_interval'][0]:.6f}, {rr['percentile_95_interval'][1]:.6f}]，三种视图均与 004 分数及标签逐位一致。</p>
<img src="figures/04_candidate_and_control.png" alt="候选与控制评价分离"><p>003 的 9/128 和 7/128 校准原样引用，后者使用每控制 255 次置换；两者不足以证明方法优劣。本轮只在同一固定 raw 控制分数上做一次 9999 次组内置换，p={p:.6f}；其 Monte Carlo 区间描述 K/9999 的数值不确定性，不是新的跨采集校准。测量状态仍为 <code>measurement_uncertain_single_exposed_acquisition</code>。</p></section>
<section><h2>点估计与不确定性</h2><p>开发集最差 lag 7 差为 {newdevdiff[6]:.6f}，仅比 0.015 门限低 {0.015-newdevdiff[6]:.6f}；边际 95% 分组区间为 [{summary['development_lag7_marginal95'][0]:.6f}, {summary['development_lag7_marginal95'][1]:.6f}]，并未整体落在 ±0.015 内。该区间按预登记只作描述，不改判候选点值筛查，也不证明等效。raw 控制条件置换 p=0.0515，其尾概率的 Monte Carlo Wilson 区间约 [0.04725, 0.05591] 跨过 0.05；不能据此说控制已修复。</p></section>
<section><h2>支持范围与可复核交付</h2><p>100×100 观测协方差没有指定完整 480 点协方差或跨窗相关性。当前模型继续拒绝 480 点；本轮未做正式确定性/随机响应两臂训练，未做新的 480 点灵敏度模拟，未取得独立采集验证。后续探索须声明长时程假设和接口，再用实际完整响应结果检验 Q、PSD 与波形方差。</p>
<p>已重新打开检查点并逐位重放：1,024 条显式随机输入结果、三模型共 24,576 条配对诊断波形、10,240 条评价生成波形、六个分类器共 3,072 个分数。另独立复算全部 5,000 条损失/梯度日志、2,000 次 ACF bootstrap 和 9,999 次条件置换；复现性不构成科学通过。</p>
<p><a href="ACF_PER_LAG.csv">20 阶误差 CSV</a> · <a href="EVALUATION_RECEIPT.json">分离评价回执</a> · <a href="ACF_UNCERTAINTY.json">分组不确定性</a> · <a href="MECHANISM_ANALYSIS.json">机制统计</a> · <a href="SUPPORT_SCOPE.md">100/480 支持范围与后续设计</a> · <a href="REPLAY_VERIFICATION.json">独立重放</a> · <a href="../../docs/NOISE_ACF005_PROTOCOL.md">预登记协议</a></p></section></main></html>'''
    (OUT/'N01_ACF005_REPORT.html').write_text(body,encoding='utf8',newline='\n')
    (OUT/'README.md').write_text(f'# N01 005 ACF 目标覆盖修复\n\n[打开中文报告](N01_ACF005_REPORT.html)。\n\n实际噪声更新 5000、评价分类器更新 1200、响应更新 0。TRAIN 匹配 {decision["TRAIN_match"]}，改善 {decision["TRAIN_improved"]}；候选开发筛查 {passed}；legacy 总门 {gate["passed"]}。控制测量状态仍不确定。仅支持 100 点。\n\n[20 阶 CSV](ACF_PER_LAG.csv) · [分离回执](EVALUATION_RECEIPT.json) · [预登记协议](../../docs/NOISE_ACF005_PROTOCOL.md)\n',encoding='utf8',newline='\n')
    write(OUT/'BUILD_RECEIPT.json',dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        generator_sha256=sha(Path(__file__)),complete_sha256=sha(SCIENCE/'COMPLETE.json'),
        evaluation_sha256=sha(SCIENCE/'noise_validation/EVALUATION_RECEIPT.json'),
        old004_gate_sha256=sha(ROOT/'provenance/noise_acf005_reference/gate004.json'),
        old002_gate_sha256=sha(ROOT/'runs/repair_002_20261012/runs/science/noise_validation/NOISE_GATE.json'),
        figures=4,new_fits=0,visual_QA='pending'))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
