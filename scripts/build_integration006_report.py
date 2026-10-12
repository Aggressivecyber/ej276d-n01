"""Read-only presentation of completed006 derived artifacts; no fits/data reads.

Only SUMMARY/DECISION/observables/curves/theory/generated sample outputs are
opened. Original REAL/SOURCE waveform containers and model checkpoints are
never opened. PNG/PDF generation does not imply visual/render QA success.
"""
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import html
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
ID='integration_006_20261012'
os.environ['MPLCONFIGDIR']='E:/GAN/runtime_tmp/matplotlib_integration006'
os.environ['TMP']='E:/GAN/runtime_tmp'
os.environ['TEMP']='E:/GAN/runtime_tmp'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
sys.path.insert(0,str(ROOT/'runtime/plotting_deps'))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

for font in (Path('C:/Windows/Fonts/msyh.ttc'),Path('C:/Windows/Fonts/msyh.ttf')):
    if font.is_file():
        font_manager.fontManager.addfont(str(font));break
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10.5,
    'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':9.5,'axes.unicode_minus':False,
    'figure.facecolor':'white','savefig.facecolor':'white','axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.14,'pdf.fonttype':42})
HYPOTHESES=('fast','slow','common_level')
VIEWS=('signal_only','legacy')+HYPOTHESES
ARMS=('deterministic','full')
COLORS=dict(fast='#007a87',slow='#cf762f',common_level='#7860ad',REAL='#253342',legacy='#7f8890',signal_only='#afbac3')
LABELS=dict(fast='快衰减',slow='慢相关',common_level='公共恒定分量',deterministic='冻结确定性臂',full='冻结完整随机臂')
FIGURES=('01_full480_curves','02_fixed_and_output_bins','03_noise_theory_and_seam','04_first_parent_decomposition')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value

def write(path,value):Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def require(condition,message):
    if not bool(condition):raise ValueError(message)

def item(arrays,*names):
    present=[key for key in names if key in arrays]
    if not present:raise KeyError('Derived artifact schema missing one of '+repr(names))
    value=arrays[present[0]]
    for key in present[1:]:require(np.array_equal(value,arrays[key]),'Ambiguous derived aliases differ: '+repr(names))
    return np.asarray(value)

def artifact(science,name):
    candidates=[science/name,science/name.upper()]
    present=list(dict.fromkeys(p for p in candidates if p.is_file()))
    require(len(present)==1,'Need one completed derived artifact: '+name)
    return present[0]

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}

def save(fig,out,name):
    fig.savefig(out/'figures'/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

def parents_for(obs,prefix,rows):
    split=prefix.split('_')[0]
    ids=item(obs,split+'_source_parent_ids')
    parent_row=item(obs,split+'_source_parent_row')
    require(parent_row.shape==(rows,) and np.issubdtype(parent_row.dtype,np.integer),'Parent-major row map changed')
    require((parent_row>=0).all() and (parent_row<len(ids)).all(),'Illegal saved parent row')
    return ids[parent_row]


def bin_statistics(obs,split,arm,cfg):
    prefix=split+'_'+arm+'_';psd=item(obs,prefix+'PSD');q=item(obs,prefix+'Q')
    fixed=item(obs,prefix+'fixed_bins');rebinned=item(obs,prefix+'rebinned_bins')
    require(psd.shape==q.shape and psd.ndim==2 and psd.shape[0]==5,'Readout shapes must be5views x rows')
    require(fixed.shape==(psd.shape[1],) and rebinned.shape==psd.shape,'Fixed/final bin shape changed')
    require(np.isfinite(psd).all() and np.isfinite(q).all(),'Nonfinite saved observations; no silent row removal')
    ids=parents_for(obs,prefix,psd.shape[1]);fixed_sd=np.full((3,8),np.nan);final_sd=np.full((3,8),np.nan)
    counts=np.zeros(8,dtype=int);parent_counts=np.zeros(8,dtype=int)
    for b in range(8):
        mask=fixed==b;counts[b]=mask.sum();parent_counts[b]=len(np.unique(ids[mask]))
        for h in range(3):
            if mask.any():fixed_sd[h,b]=psd[h+2,mask].std(ddof=0)
            final=rebinned[h+2]==b
            if final.any():final_sd[h,b]=psd[h+2,final].std(ddof=0)
    support=(counts>=cfg['evaluation']['minimum_rows_per_fixed_bin'])&(parent_counts>=cfg['evaluation']['minimum_parents_per_fixed_bin'])
    rq=item(obs,'TRAIN_REAL_Q');rp=item(obs,'TRAIN_REAL_PSD')
    require(np.isfinite(rq).all() and np.isfinite(rp).all(),'Nonfinite saved REAL readouts')
    qscale=rq.std(ddof=0);pscale=rp.std(ddof=0)
    require(qscale>0 and pscale>0,'REAL TRAIN/development scales must be positive')
    means_q=q[2:].mean(1);means_psd=psd[2:].mean(1)
    minimum=np.min(fixed_sd,axis=0);maximum=np.max(fixed_sd,axis=0)
    ratio=np.divide(maximum,minimum,out=np.full(8,np.nan),where=(minimum>0))
    return dict(rows=psd.shape[1],parents=len(np.unique(ids)),fixed_rows=counts,fixed_parents=parent_counts,
        sufficient_support=support,fixed_PSD_SD=fixed_sd,output_Q_bin_PSD_SD=final_sd,
        fixed_PSD_SD_max_min_ratio=ratio,mean_Q_by_hypothesis=means_q,mean_PSD_by_hypothesis=means_psd,
        Q_mean_range_over_REAL_SD=np.ptp(means_q)/qscale,PSD_mean_range_over_REAL_SD=np.ptp(means_psd)/pscale,
        nonpositive_Q_by_hypothesis=(q[2:]<=0).sum(1),
        guarded_PSD_by_hypothesis=item(obs,prefix+'PSD_guarded')[2:].sum(1))


def plot_curves(curves,out):
    t=np.arange(480);fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for row,arm in enumerate(ARMS):
        prefix='TRAIN_'+arm+'_'
        mean=item(curves,prefix+'mean',prefix+'waveform_mean');variance=item(curves,prefix+'variance',prefix+'waveform_variance')
        require(mean.shape==variance.shape==(5,480),'Whole-population curves must be5x480')
        real_mean=item(curves,'TRAIN_REAL_mean','TRAIN_REAL_waveform_mean')
        real_var=item(curves,'TRAIN_REAL_variance','TRAIN_REAL_waveform_variance')
        for i,h in enumerate(HYPOTHESES):
            axes[row,0].plot(t,mean[i+2]*1000,color=COLORS[h],label=LABELS[h])
            axes[row,1].plot(t,np.sqrt(np.maximum(variance[i+2],0))*1000,color=COLORS[h],label=LABELS[h])
        axes[row,0].plot(t,real_mean*1000,color=COLORS['REAL'],ls='--',lw=1.5,label='同处理 REAL TRAIN')
        axes[row,1].plot(t,np.sqrt(np.maximum(real_var,0))*1000,color=COLORS['REAL'],ls='--',lw=1.5,label='同处理 REAL TRAIN')
        for col in range(2):
            axes[row,col].axvline(100,color='#90999e',ls=':',lw=1)
            axes[row,col].set(xlabel='原软件坐标（1 ns/点，非硬件采样验证）',ylabel='电压（mV）')
            axes[row,col].legend(frameon=False)
        axes[row,0].set_title(LABELS[arm]+'：完整480点均值')
        axes[row,1].set_title(LABELS[arm]+'：完整480点总体标准差')
    fig.suptitle('TRAIN 全父事件、每父4draw；原count权重1；旧冻结参数下的反事实替换',fontsize=14)
    save(fig,out,FIGURES[0])


def plot_widths(stats,bootstrap,out):
    fig,axes=plt.subplots(3,2,figsize=(14,12),layout='constrained');bins=np.arange(1,9)
    for col,arm in enumerate(ARMS):
        value=stats['TRAIN'][arm]
        for i,h in enumerate(HYPOTHESES):
            axes[0,col].plot(bins,value['fixed_PSD_SD'][i],'o-',ms=4,color=COLORS[h],label=LABELS[h])
            axes[1,col].plot(bins,value['output_Q_bin_PSD_SD'][i],'o-',ms=4,color=COLORS[h],label=LABELS[h])
        bad=~value['sufficient_support']
        for ax in axes[:,col]:
            for b in bins[bad]:ax.axvspan(b-.3,b+.3,color='#d7a051',alpha=.16)
            ax.set(xticks=bins,xlabel='冻结TRAIN坐标的Q箱',ylabel='PSD总体SD（ddof0）')
            if ax is not axes[2,col]:ax.legend(frameon=False)
        axes[0,col].set_title(LABELS[arm]+'：固定无电子学信号Q成员')
        axes[1,col].set_title(LABELS[arm]+'：按最终输出Q重新归箱')
        ci=item(bootstrap,'TRAIN_intervals')[col*11+2:col*11+10]
        axes[2,col].vlines(bins,ci[:,0],ci[:,1],color='#3b6578',lw=1.5,label='1000次父级配对bootstrap 95%区间')
        axes[2,col].plot(bins,value['fixed_PSD_SD_max_min_ratio'],'o',color='#243a45',label='固定成员SD max/min点值')
        axes[2,col].axhline(1.1,color='#8a6960',ls='--',lw=1,label='冻结实用容差1.10')
        axes[2,col].set(ylabel='三假设PSD SD最大/最小比',title=LABELS[arm]+'：固定箱宽度敏感性及MC区间')
        axes[2,col].legend(frameon=False)
    fig.suptitle('固定成员的宽度敏感性与Q迁移分开；着色箱支持不足；不拟合粒子真峰',fontsize=14)
    save(fig,out,FIGURES[1])


def plot_theory(theory,out):
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');t=np.arange(480);bars=[];seams=[]
    for h in HYPOTHESES:
        cov=item(theory,h+'_covariance');require(cov.shape==(480,480),'Theory needs full480 covariance')
        sd=np.sqrt(np.diag(cov));rho=cov[99]/(sd[99]*sd)
        axes[0,0].plot(t,sd*1000,color=COLORS[h],label=LABELS[h])
        axes[0,1].plot(t,rho,color=COLORS[h],label=LABELS[h])
        bars.append([float(item(theory,h+'_variance_Q')),float(item(theory,h+'_variance_tail_charge'))])
        seams.append(float(item(theory,h+'_seam_increment_variance'))/float(item(theory,h+'_prefix_increment_variance90_99').mean()))
    for ax in axes[0]:ax.axvline(100,color='#90999e',ls=':',lw=1);ax.set_xlabel('原软件坐标（ns）');ax.legend(frameon=False)
    axes[0,0].set(title='有效连续噪声：解析逐点SD',ylabel='噪声SD（mV）')
    axes[0,1].set(title='跨窗相关：与第99点的解析相关系数',ylabel='相关系数')
    bars=np.asarray(bars);x=np.arange(3)
    axes[1,0].bar(x-.18,bars[:,0],width=.36,color='#287a84',label='Var(Q)')
    axes[1,0].bar(x+.18,bars[:,1],width=.36,color='#a783bc',label='Var(tail)')
    axes[1,0].set(xticks=x,xticklabels=[LABELS[h] for h in HYPOTHESES],ylabel='方差（V² ns²）',title='电子学电荷保留非零涨落');axes[1,0].legend(frameon=False)
    axes[1,1].bar(x,seams,color=[COLORS[h] for h in HYPOTHESES]);axes[1,1].axhline(1,color='#71797f',ls='--',lw=1)
    axes[1,1].set(xticks=x,xticklabels=[LABELS[h] for h in HYPOTHESES],ylabel='Var(99→100) / 前缀90..99内部增量平均Var',title='边界粗糙度仅报告，不事后修边或选核')
    save(fig,out,FIGURES[2])


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def plot_example(samples,obs,out):
    require(samples['schema_version']=='integration006/v1' and tuple(samples['hypotheses'])==HYPOTHESES and tuple(samples['views'])==VIEWS,'Generated sample schema changed')
    split=samples['splits']['TRAIN']
    require(np.array_equal(as_numpy(split['source_parent_row'])[:128],item(obs,'TRAIN_source_parent_row')[:128]),'Sample rows differ from predeclared parent-major prefix')
    require(np.array_equal(as_numpy(split['draw_index'])[:4],np.arange(4)),'First parent must have draws0..3')
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');t=np.arange(480);shared_noise=None
    for row,arm in enumerate(ARMS):
        signal=as_numpy(split['arms'][arm]['signal_pre']);noise=as_numpy(split['arms'][arm]['noise_pre'])
        require(signal.shape==(128,480) and noise.shape==(3,128,480),'Need exact first128 generated component samples')
        require(np.isfinite(signal).all() and np.isfinite(noise).all(),'Nonfinite sample components')
        if shared_noise is not None:require(np.array_equal(shared_noise,noise),'Noise samples must pair across arms')
        shared_noise=noise
        ids=parents_for(obs,'TRAIN_'+arm+'_',item(obs,'TRAIN_'+arm+'_Q').shape[1])
        require(len(np.unique(ids[:4]))==1,'Predeclared first four rows must share the first SOURCE parent')
        for draw in range(4):axes[row,0].plot(t,signal[draw]*1000,alpha=.7,lw=1,label='draw '+str(draw))
        axes[row,0].set(title=LABELS[arm]+'：预定首父的4个信号响应',ylabel='信号（mV）');axes[row,0].legend(frameon=False)
        for i,h in enumerate(HYPOTHESES):axes[row,1].plot(t,noise[i,0]*1000,color=COLORS[h],lw=.9,label=LABELS[h])
        axes[row,1].set(title='共享同一次噪声抽样：首父draw0的三种延拓',ylabel='有效连续噪声（mV）');axes[row,1].legend(frameon=False)
        for ax in axes[row]:ax.axvline(100,color='#90999e',ls=':',lw=1);ax.set_xlabel('原软件坐标（ns）')
    fig.suptitle('示例只按预定输入顺序取首父；没有挑选最佳波形。信号与噪声分量直接读取',fontsize=14)
    save(fig,out,FIGURES[3])


def table(headers,rows):
    def cell(value):return html.escape(str(value))
    return '<table><thead><tr>'+''.join('<th>'+cell(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

def f(value):return 'NA' if value is None or not np.isfinite(float(value)) else f'{float(value):.6g}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--science',type=Path,required=True);parser.add_argument('--verification',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'integration006_config.json')
    parser.add_argument('--out',type=Path,default=ROOT/'reports'/ID);parser.add_argument('--overwrite',action='store_true')
    a=parser.parse_args();science=a.science.resolve();out=a.out.resolve()
    require(os.name!='nt' or out.drive.upper() in ('E:','D:'),'Local output/caches must be E/D')
    require(a.overwrite or not (out/'N01_INTEGRATION006_REPORT.html').exists(),'Preserve existing report; --overwrite permits display rebuild')
    complete=read(science/'COMPLETE.json');verified=read(a.verification);cfg=read(a.config)
    require(complete['status']=='zero_training_completed' and complete['run_id']==cfg['run_id']==ID,'Only the unique completed006 endpoint can be reported')
    require(complete['provenance']['config_sha256']==sha(a.config),'Report config differs from completed scientific binding')
    require(verified.get('status') in ('passed','verified','complete') and verified.get('passed',True),'Independent replay must be completed successfully')
    require(verified['complete_sha256']==sha(science/'COMPLETE.json'),'Replay/endpoint COMPLETE binding differs')
    require(complete['updates']==dict(noise=0,response=0,classifier=0,GMM_fits=0),'006 must contain zero actual update/fits')
    # Restrict all numeric reads/hashing to derived presentation inputs. Original
    # containers, replay source tensors, checkpoints and latent payloads remain unopened.
    for name in ('SUMMARY.json','DECISION.json','observables.npz','curves.npz','theory.npz','bootstrap.npz','sample_waveforms.pt'):
        require(sha(science/name)==complete['payload_sha256'][name],'Completed derived artifact SHA mismatch: '+name)
    summary=read(artifact(science,'SUMMARY.json'));decision=read(artifact(science,'DECISION.json'))
    paths={name:artifact(science,name+'.npz') for name in ('observables','curves','theory','bootstrap')}
    paths['sample_waveforms']=science/'sample_waveforms.pt'
    obs,curves,theory,bootstrap=[load_npz(paths[name]) for name in ('observables','curves','theory','bootstrap')]
    # Hash-bound generated-only project artifact; never open replay/source payloads.
    samples=torch.load(paths['sample_waveforms'],map_location='cpu',weights_only=False)
    stats={split:{arm:bin_statistics(obs,split,arm,cfg) for arm in ARMS} for split in ('TRAIN','DEV')}
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    plot_curves(curves,out);plot_widths(stats,bootstrap,out);plot_theory(theory,out);plot_example(samples,obs,out)
    outcome=decision.get('outcome',decision.get('status'))
    require(outcome in ('interface_fail','sensitivity_detected','MC_inconclusive','sensitivity_within_family'),'Missing frozen decision outcome')
    outcomes=dict(interface_fail='接口检查失败',sensitivity_detected='检测到长程假设敏感性',MC_inconclusive='固定预算下不确定',sensitivity_within_family='在这三个有限假设内满足稳定性要求')
    report=dict(run_id=ID,outcome=outcome,decision=decision,noise_updates=0,response_updates=0,classifier_updates=0,GMM_fits=0,
        descriptive_readouts=stats,source_summary=summary,predeclared_example_source_parent_id=str(parents_for(obs,'TRAIN_deterministic_',item(obs,'TRAIN_deterministic_Q').shape[1])[0]),
        control_status='measurement_uncertain_single_exposed_acquisition',
        scenario_family_is_not_confidence_set=True,identified_hardware_noise=False,independent_acquisition_validations=0,
        frozen_old_response_counterfactual=True,independent_replay_counts=verified.get('replay_counts'),
        engineering_attempts=[dict(attempt='initial',native_exit_code=1,scientific_forward_rows=0,
            failure='Tensor/list loss-statistic metadata representation comparison'),
            dict(attempt='binding_fix01',native_exit_code=1,noise005_forward_rows=128,extension_forward_rows=384,
                REAL_readout_rows=21047,response_forward_rows=0,failure='Original string ID serialization'),
            dict(attempt='id_fix02',completed_scientific_evaluation=True,unchanged_scientific_configuration=True)],
        verification_attempts=[dict(attempt='initial',native_exit_code=1,response_noise_replay_forward_rows=0,
            REAL_readout_rows_recomputed=21646,failure='Dimensionless A near-zero float64 solve comparison'),
            dict(attempt='numeric_fix01',native_exit_code=1,failure='ADC voltage representation compared as bitwise cross-backend values',
                reconstructed_noise_forward_rows=128,reconstructed_extension_rows=384,reconstructed_response_forward_rows=128,
                first_batch_generated_waveform_hashes_matched=True,REAL_readout_rows_recomputed=21646),
            dict(attempt='adc_fix02',status=verified['status'],verifier_sha256=verified['verifier_sha256'],
                scientific_thresholds_and_completed_outputs_unchanged=True)],
        visual_QA='see_separate_VISUAL_QA.json',PDF_render_QA='see_separate_VISUAL_QA.json')
    write(out/'SUMMARY.json',report)
    metrics=[];support=[];migration=[]
    for split in ('TRAIN','DEV'):
        for arm in ARMS:
            value=stats[split][arm]
            intervals=item(bootstrap,split+'_intervals');offset=ARMS.index(arm)*11
            metrics.append([split,LABELS[arm],value['parents'],value['rows'],f(value['Q_mean_range_over_REAL_SD']),
                '['+', '.join(f(v) for v in intervals[offset])+']',f(value['PSD_mean_range_over_REAL_SD']),
                '['+', '.join(f(v) for v in intervals[offset+1])+']',int(np.sum(value['nonpositive_Q_by_hypothesis'])),int(np.sum(value['guarded_PSD_by_hypothesis']))])
            for b in range(8):support.append([split,LABELS[arm],b+1,int(value['fixed_parents'][b]),int(value['fixed_rows'][b]),f(value['fixed_PSD_SD_max_min_ratio'][b]),'足够' if value['sufficient_support'][b] else '不足'])
            if split=='TRAIN':
                matrix=item(obs,split+'_'+arm+'_migration')
                require(matrix.shape==(5,8,8),'Migration schema changed')
                for i,h in enumerate(HYPOTHESES):
                    rows=[[b+1]+list(map(int,matrix[i+2,b])) for b in range(8)]
                    migration.append('<details><summary>'+html.escape(LABELS[arm]+' / '+LABELS[h]+'：固定信号Q箱 → 最终输出Q箱')+'</summary>'+table(['原箱 / 新箱']+list(range(1,9)),rows)+'</details>')
    noise_rows=[]
    for h in HYPOTHESES:
        noise_rows.append([LABELS[h],f(item(theory,h+'_variance_Q')),f(item(theory,h+'_variance_tail_charge')),f(item(theory,h+'_covariance_Q_tail')),f(item(theory,h+'_tail400480_variance_mean')),f(item(theory,h+'_seam_increment_mean')),f(item(theory,h+'_seam_increment_variance'))])
    scopes='本轮新增噪声、响应、CNN更新和GMM拟合均为0。005前100点候选及旧两臂权重保持冻结；替换噪声是在旧噪声下学习的固定参数上的反事实，不是真正的新响应训练，也不是随机支路的架构因果验证。'
    decisive_rows=[]
    for i,arm in enumerate(ARMS):
        value=stats['TRAIN'][arm];ci=item(bootstrap,'TRAIN_intervals')[i*11+2]
        decisive_rows.append([LABELS[arm],int(value['fixed_parents'][0]),
            f(value['fixed_PSD_SD_max_min_ratio'][0]),'['+', '.join(f(v) for v in ci)+']'])
    next_stage=('可另行登记显式保留假设的响应协议。' if decision['next_response_protocol_allowed'] else
        '按本轮冻结规则，当前候选不进入正式响应训练；不得按结果挑选某个核继续。')
    body=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>N01 006 条件延拓与整合敏感性</title>
<style>body{{font-family:'Microsoft YaHei',system-ui,sans-serif;background:#f2f5f7;color:#203442;margin:0}}main{{max-width:1200px;margin:auto;padding:30px}}h1{{font-size:29px}}h2{{font-size:22px}}section,.lead{{background:white;border:1px solid #dbe4e9;border-radius:10px;padding:22px;margin:18px 0}}.lead{{border-left:6px solid #cf762f}}p,li{{line-height:1.8;font-size:16px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;overflow:auto}}th,td{{padding:9px;border-bottom:1px solid #dce5e9;text-align:right}}th{{background:#edf3f6}}th:first-child,td:first-child{{text-align:left}}details{{margin:14px 0}}summary{{cursor:pointer;font-weight:bold}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f6;padding:14px;font-size:12px}}code{{overflow-wrap:anywhere}}a{{color:#007482}}.scroll{{overflow-x:auto}}@media(max-width:800px){{main{{padding:12px}}section{{padding:14px}}th,td{{padding:6px}}}}</style><main>
<h1>N01 006：480点条件延拓与整合敏感性</h1><div class="lead"><h2>{outcomes[outcome]}</h2><p>{scopes}</p><p>工程接口检查：{'通过' if decision['engineering_pass'] else '未通过'}。下表列出最低电荷固定箱在三假设之间的PSD宽度比，冻结容差为1.10；噪声尾部方差比为{f(decision['noise_tail_variance_ratio'])}，其单独容差为1.25。</p><div class="scroll">{table(['TRAIN冻结臂','最低箱独立父','PSD SD最大/最小','95%父级配对MC区间'],decisive_rows)}</div><p>两臂按各自无电子学信号Q定义固定成员，最低箱人口不同；表中差异不用于比较两臂优劣，也不是与REAL的拟合误差。{next_stage}</p><p>决定仅使用TRAIN；DEV是已多次曝光的同一采集开发集，不影响假设、参数或下一阶段决定。旧测量控制状态仍为 <code>measurement_uncertain_single_exposed_acquisition</code>。本报告不改写005结论。</p></div>
<section><h2>积分电荷、PSD与固定人口</h2><p>原合法 SOURCE TRAIN 6593父、DEV 512父，每父4draw；同处理REAL TRAIN 21047行、DEV 599行。原count权重1，未套用旧人口比较权重。REAL只经过原canonical ADC和相同确定性读出，不添加新噪声、不做随机响应或去噪。</p><div class="scroll">{table(['角色','冻结臂','父数','响应行','Q范围 / REAL TRAIN SD','Q范围95%MC区间','PSD范围 / REAL TRAIN SD','PSD范围95%MC区间','新候选Q≤0行','分母保护行'],metrics)}</div><p>表中是三种假设的点值范围，TRAIN实用容差各0.05；最终结论使用原冻结的1000次父级配对bootstrap及足支持条件，不把点值小解释为物理等价。所有非正Q/保护行保留计数，保护后的PSD不代表有效物理PSD。</p><img src="figures/{FIGURES[0]}.png" alt="两臂完整480点均值与标准差"><p><a href="figures/{FIGURES[0]}.pdf">同图PDF</a></p><p>完整波形总体标准差包含源人口宽度；它不等同于单父条件噪声，也不等同于解析noise-only尾部方差。原软件1 ns网格不是硬件采样验证。</p></section>
<section><h2>PSD宽度：固定成员与Q分箱迁移分开</h2><img src="figures/{FIGURES[1]}.png" alt="固定分箱与输出分箱PSD宽度"><p><a href="figures/{FIGURES[1]}.pdf">同图PDF</a></p><p>首要固定箱按各冻结臂无电子学信号Q确定，同父同draw跨假设不变；下面重新归箱的宽度含Q迁移。每箱至少32个独立父和128行才称足支持，图中着色箱不足。混合箱的SD/IQR不是已识别中子或伽马真峰宽度，本轮GMM拟合0。</p><details><summary>全部TRAIN/DEV固定箱支持与三假设SD最大/最小比</summary><div class="scroll">{table(['角色','臂','箱','独立父','行','SD max/min','支持'],support)}</div></details>{''.join(migration)}<p>8×8矩阵行是固定信号Q成员、列是各情景最终Q归箱，保存完整计数；不按迁移或DEV挑选情景。</p></section>
<section><h2>三个条件假设及跨窗结构</h2><p>参考核固定为 <code>exp(-d/4)</code>、<code>0.75 exp(-d/4)+0.25 exp(-d/80)</code>、<code>0.75 exp(-d/4)+0.25</code>；乘冻结005协方差对角均值后，全480矩阵统一加相对ridge 1e-6。第三项是公共恒定水平，不是随机斜率。</p><p><code>tail=mu_tail+A(prefix-mu100)+S·chol(R)·epsilon</code>，共用005的 <code>S=exp(eta·u-eta²)</code>。前100点pre/postADC逐位检查；联合协方差含实际005非平稳前缀C，因此通常不是参考核K。有限三假设不是长程相关性的置信集合，也不是480点硬件电子学校准。</p><img src="figures/{FIGURES[2]}.png" alt="解析VarQ跨窗相关尾方差与边界增量"><div class="scroll">{table(['假设','VarQ(V²ns²)','VarTail(V²ns²)','Cov(Q,Tail)','400:480平均Var(V²)','跨界增量均值(V)','跨界增量Var(V²)'],noise_rows)}</div><p><a href="figures/{FIGURES[2]}.pdf">同图PDF</a></p><p>这里是有效连续潜变量的解析量；不把它等同于ADC后的精确矩。边界增量只报告，不后加taper、不强行逐点等值、不按seam好坏选择核。</p></section>
<section><h2>预定首父示例与接口隔离</h2><img src="figures/{FIGURES[3]}.png" alt="预定首SOURCE父的信号与共享噪声分解"><p><a href="figures/{FIGURES[3]}.pdf">同图PDF</a></p><p>示例固定输入顺序中的首TRAIN父 <code>{html.escape(report["predeclared_example_source_parent_id"])}</code> 的4draw，三假设噪声图固定draw0；未挑选最佳波形。信号直接由 <code>qsig·unit_shape+signal_noise</code> 读取，候选链只对 <code>signal+new_noise</code> 量化一次。旧量化waveform不用于扣噪或二次量化。</p><p>保留的是噪声前缀。旧time-warp信号可能泄漏到前100点，工程检查单独记录；超过1 ADC step时不能把组装接口称为保留005观测基线。正式两臂响应协议必须另行登记，本轮不自动启动。</p><details><summary>原始派生SUMMARY与冻结决策（包含工程检查、信号泄漏、bootstrap区间）</summary><pre>{html.escape(json.dumps(clean(dict(SUMMARY=summary,DECISION=decision)),ensure_ascii=False,indent=2))}</pre></details></section>
<section><h2>重放与证据边界</h2><p>本报告只读完成后的派生产物，没有读取REAL/SOURCE原波形、运行响应或拟合新模型。独立重放绑定本次COMPLETE；完整重放范围保留于机器摘要及独立receipt，工程重放不代表科学验收。无独立采集验证，旧控制不确定仍保留。首次启动因旧检查点统计量的Tensor与JSON表示比较失败，科学前向为0；第二次因字符串ID记录失败，已完成128条005噪声、384条延拓与21047条REAL读出，响应前向为0。两次失败及非零计数均保留。最终id_fix02原样保留ID，沿用原科学配置、假设和种子；首128输入重复产生，前次未保存的部分输出无法逐位复核。首次独立复核在解析矩阵A的近零元素比较处停止，差异约1.36×10⁻¹⁵，尚未进入响应重放。另行登记的数值QA修订仅调整A的机器精度绝对底线并新增后向误差检查，保留原失败回执，所有科学门槛和完成产物均未更改。第二次复核已逐位匹配首批噪声、信号和生成波形，随后因跨后端ADC电压表示差异误计格点变化而停止；实际整数码完全一致，非零重复前向128条响应、128条噪声及384条延拓按失败路径重建并保留。另行登记的ADC复核修订恢复同设备比较，并独立核验所有480点ADC整数码及电压格点。独立全量重放核验的是最终完成产物。PNG及同图PDF的视觉与重开检查记录在独立QA回执中。</p><p><a href="SUMMARY.json">机器可读交付摘要</a> · <a href="SUMMARY.md">简明摘要</a> · <a href="REPORT_BUILD_RECEIPT.json">生成回执</a> · <a href="VISUAL_QA.json">视觉与PDF重开回执</a> · <a href="../../docs/INTEGRATION_006_PROTOCOL.md">冻结协议</a> · <a href="../../docs/INTEGRATION_006_BINDING_FIX01.md">元数据修订</a> · <a href="../../docs/INTEGRATION_006_ID_FIX02.md">字符串ID修订</a></p></section></main></html>"""
    (out/'N01_INTEGRATION006_REPORT.html').write_text(body,encoding='utf8',newline='\n')
    (out/'SUMMARY.md').write_text('# N01 006 条件延拓整合敏感性\n\n'+outcomes[outcome]+'。\n\n'+scopes+'\n\n'+
        '三个核预先固定，所有前缀和随机输入配对；决定仅用TRAIN。旧测量不确定、同采集曝光DEV、未知长程相关性与物理识别边界均保留。\n\n'+
        '[中文完整报告](N01_INTEGRATION006_REPORT.html) · [机器摘要](SUMMARY.json) · [构建回执](REPORT_BUILD_RECEIPT.json)\n',encoding='utf8',newline='\n')
    rendered=[out/'figures'/(name+'.'+suffix) for name in FIGURES for suffix in ('png','pdf')]
    require(all(path.is_file() and path.stat().st_size>0 for path in rendered),'Figure exports incomplete')
    receipt=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
        complete_sha256=sha(science/'COMPLETE.json'),verification_sha256=sha(a.verification),configuration_sha256=sha(a.config),
        inputs={str(path):sha(path) for path in [artifact(science,'SUMMARY.json'),artifact(science,'DECISION.json'),*paths.values()]},
        figures=4,PNG_exports=4,PDF_exports=4,numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,
        report_artifacts={name:sha(out/name) for name in ('N01_INTEGRATION006_REPORT.html','SUMMARY.json','SUMMARY.md')},rendered_files={path.relative_to(out).as_posix():dict(sha256=sha(path),bytes=path.stat().st_size) for path in rendered},
        scientific_forward_rows=0,original_scientific_arrays_decoded=0,new_fits=0,
        visual_QA='pending_root_inspection',PDF_render_QA='not_performed',scientific_acceptance=False)
    write(out/'REPORT_BUILD_RECEIPT.json',receipt)
    print(json.dumps(dict(report=str(out/'N01_INTEGRATION006_REPORT.html'),outcome=outcome,figures=4,new_fits=0,visual_QA='pending'),ensure_ascii=False))


if __name__=='__main__':main()
