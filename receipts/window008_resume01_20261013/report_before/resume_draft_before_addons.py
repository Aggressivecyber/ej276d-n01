"""Render frozen endpoint artifacts; never load a model or original waveforms."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import html
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']='E:/GAN/runtime_tmp/matplotlib_window008'
os.environ['TMP']=os.environ['TEMP']='E:/GAN/runtime_tmp'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
sys.path.insert(0,str(ROOT/'runtime/plotting_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,
    'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8.5,'axes.unicode_minus':False,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42})
ARMS=('deterministic','full','simple');HYP=('fast','slow','common_level');METRICS=('wave','QPSD','noise')
LABEL={'deterministic':'确定性 U-Net','full':'完整随机 U-Net','simple':'简单随机响应',
       'fast':'短相关','slow':'慢相关','common_level':'恒定公共分量',
       'wave':'256点波形','QPSD':'电荷与PSD','noise':'100点基线'}
COLORS={'fast':'#007f8b','slow':'#cc7626','common_level':'#7958a0','REAL':'#26394b',
        'deterministic':'#22718b','full':'#b15a28','simple':'#687248'}


def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda:stream.read(1024**2),b''):digest.update(piece)
    return digest.hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as archive:return {key:archive[key].copy() for key in archive.files}
def write(path,value):
    with path.open('x',encoding='utf8') as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def fmt(value):return '未定义' if value is None else f'{value:.4g}'
def foot(fig,value):fig.text(.065,.026,value,va='bottom',fontsize=9,color='#35495c',linespacing=1.45)
def title(fig,main,sub):
    fig.suptitle(main,y=.963,fontsize=17)
    fig.text(.5,.921,sub,ha='center',fontsize=10.5)


def load_training_journals(run):
    def rows(p):return [json.loads(line) for line in p.read_text(encoding='utf8').splitlines()]
    old=ROOT/'runs/window008_20261013/runs/science/train'
    fix=ROOT/'runs/window008_fix01_20261013/runs/science/train'
    new=run/'runs/science/train'
    full=[v for v in rows(old/'full/training.jsonl') if v['G_updates']<=3000]
    full.extend(v for v in rows(fix/'full/training.jsonl') if v['G_updates']<=6000)
    full.extend(rows(new/'full/training.jsonl'))
    result={'deterministic':rows(old/'deterministic/training.jsonl'),'full':full,'simple':rows(new/'simple/training.jsonl')}
    for arm,values in result.items():
        if [v['G_updates'] for v in values]!=list(range(1,19001)):raise ValueError('Effective training journal lineage differs: '+arm)
    return result


def evidence_page(pdf,save,complete,replay):
    fig=plt.figure(figsize=(12.4,8.8))
    title(fig,'N01-008：训练与数值证据','256点有效响应；2026年10月14日固定终点交付')
    ax=fig.add_axes([.07,.20,.86,.64]);ax.axis('off')
    lines=[
        ('固定终点','deterministic / full / simple 均完成 19,000 G；每臂三种 D 各 16,000。'),
        ('训练覆盖','每臂覆盖全部 6,593 SOURCE 与 21,047 REAL TRAIN 父事件。'),
        ('原始数值故障','full 第3,270步形状积分为负；保留原 native exit1 和失败现场。'),
        ('修复与用户停机','随机形状基投影为零规范电荷；full 到6,398步按关机指令停止。'),
        ('完整状态接续','从完整6,000步恢复所有状态；首6,001步17项共同科学指标精确复现。'),
        ('当前运行','新增32,000 G / 87,000 D；native exit0，2026-10-14 02:10完成（UTC+8）。'),
        ('有效比较预算','57,000 G / 144,000 D；历史实际57,667 G / 146,001 D。'),
        ('主体数值回放',f"{replay['NPZ_files_checked']}个NPZ、{replay['NPZ_members_checked']}个数组成员和全部数值JSON精确一致。"),
        ('科学状态','response_match_not_established：当前证据是单次采集上的冻结模型内部评价。'),
    ]
    for i,(label,value) in enumerate(lines):
        y=.98-i*.107
        ax.text(0,y,label,fontsize=12,fontweight='bold',va='top',color='#22718b')
        ax.text(.205,y,value,fontsize=11,va='top',wrap=True)
    foot(fig,'噪声、人口和C2ST本轮新增优化均为0。原故障和用户停机的native1作为历史证据保留。\n物理响应匹配仍需独立采集、标定或逐输入实测约束；下页起分别呈现TRAIN和已暴露DEV。')
    save(fig,'00_execution_and_replay',pdf)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='runs/window008_resume01_20261013')
    parser.add_argument('--evaluation',default='runs/window008_resume01_20261013/runs/evaluation_jobs/primary/artifacts')
    parser.add_argument('--out',default='reports/window008_resume01_20261013')
    args=parser.parse_args();run=ROOT/args.run;evaluation=ROOT/args.evaluation;out=ROOT/args.out
    if out.exists():raise FileExistsError('Preserve an earlier report: '+str(out))
    complete=read(run/'runs/science/TRAINING_COMPLETE.json')
    native=read(run/'runs/supervisor/exit_receipt.json');summary=read(evaluation/'SUMMARY.json')
    if complete['effective_comparison_G_updates']!=57000 or complete['effective_comparison_D_updates']!=144000 or native['exit_code']!=0:
        raise ValueError('Report requires the complete fixed native training endpoint')
    replay=read(evaluation.parent.parent/'replay01/REPLAY_VERIFICATION.json')
    if replay['status']!='exact_replay_verified':raise ValueError('Exact numerical replay is required')
    for kind in ('primary','replay01'):
        job=evaluation.parent.parent/kind
        if read(job/'supervisor/exit_receipt.json')['exit_code']!=0 or read(job/'EVALUATOR_NATIVE.json')['exit_code']!=0:raise ValueError('Evaluation native success missing')
    if summary['status']!='fixed_resumed_endpoint_evaluation_complete' or summary['scientific_status']!='response_match_not_established':
        raise ValueError('Wrong endpoint evaluation result')
    for name,digest in summary['artifact_sha256'].items():
        if sha(evaluation/name)!=digest:raise ValueError('Evaluation artifact changed: '+name)
    distributions=read(evaluation/'DISTRIBUTION_DISTANCES.json')['distances']
    generations={split:read(evaluation/(split+'_generation.json')) for split in ('TRAIN','DEV')}
    bootstrap={split:read(evaluation/(split+'_PARENT_BOOTSTRAP.json')) for split in ('TRAIN','DEV')}
    observables={split:load(evaluation/(split+'_observables.npz')) for split in ('TRAIN','DEV')}
    curves={split:load(evaluation/(split+'_curves.npz')) for split in ('TRAIN','DEV')}
    journals=load_training_journals(run)
    if any(len(rows)!=19000 or rows[-1]['G_updates']!=19000 for rows in journals.values()):raise ValueError('Training logs are incomplete')
    out.mkdir(parents=True);(out/'figures').mkdir()
    figures=[]
    def save(fig,name,pdf):
        fig.savefig(out/'figures'/(name+'.png'),dpi=175,facecolor='white')
        pdf.savefig(fig,facecolor='white');plt.close(fig);figures.append(name)
    worst={split:{arm:{metric:max(distributions[split][arm][h]['components'][metric] for h in HYP) for metric in METRICS} for arm in ARMS} for split in ('TRAIN','DEV')}
    relative={split:{reference:{metric:1-worst[split]['full'][metric]/worst[split][reference][metric] if worst[split][reference][metric]>0 else None for metric in METRICS} for reference in ('deterministic','simple')} for split in ('TRAIN','DEV')}
    controls={split:{metric:worst[split]['full'][metric]/distributions[split]['REAL_REAL']['components'][metric] if distributions[split]['REAL_REAL']['components'][metric]>0 else None for metric in METRICS} for split in ('TRAIN','DEV')}
    pdfpath=out/'N01_WINDOW008_RESUME01_REPORT.pdf'
    with PdfPages(pdfpath) as pdf:
        evidence_page(pdf,save,complete,replay)
        fig,axes=plt.subplots(2,3,figsize=(12.4,8.8));fig.subplots_adjust(left=.075,right=.97,bottom=.14,top=.825,hspace=.46,wspace=.32)
        title(fig,'N01-008：256点固定终点比较','三组各19,000 G；合计57,000 G / 144,000 D。训练已完成，物理响应匹配尚未建立。')
        for i,split in enumerate(('TRAIN','DEV')):
            for j,metric in enumerate(METRICS):
                ax=axes[i,j];positions=np.arange(3)
                for k,h in enumerate(HYP):
                    ax.bar(positions+(k-1)*.24,[distributions[split][a][h]['components'][metric] for a in ARMS],width=.22,color=COLORS[h],label=LABEL[h])
                ax.axhline(distributions[split]['REAL_REAL']['components'][metric],color=COLORS['REAL'],lw=1.5,ls='--',label='REAL-REAL')
                ax.set_xticks(positions,['确定性','完整随机','简单随机']);ax.set_ylabel('固定投影距离（越小越接近）')
                ax.set_title(split+' / '+LABEL[metric])
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.902),ncol=4,frameon=False)
        foot(fig,'TRAIN每域匹配2,048个父事件，DEV每域256个；每个生成父事件只取第0次响应作距离比较。\nDEV来自同一次已暴露采集，未用于本轮训练或终点选择。三种噪声假设是敏感性设置，未注册科学通过阈值。')
        save(fig,'01_fixed_endpoint_distances',pdf)

        for split in ('TRAIN','DEV'):
            fig,axes=plt.subplots(2,3,figsize=(12.4,8.8));fig.subplots_adjust(left=.075,right=.975,bottom=.14,top=.825,hspace=.43,wspace=.3)
            title(fig,split+'：256点完整电压分布','每个仿真父事件4次响应；图示保留原人口权重，未按实测电荷挑选输出。')
            times=np.arange(256)-16;values=curves[split]
            for j,arm in enumerate(ARMS):
                for i,stat in enumerate(('mean','std')):
                    ax=axes[i,j]
                    for h in HYP:ax.plot(times,1000*values[arm+'_'+h+'_'+stat],color=COLORS[h],lw=1.35,label=LABEL[h])
                    ax.plot(times,1000*values['REAL_'+stat],color=COLORS['REAL'],lw=1.5,ls='--',label='REAL')
                    ax.set_title(LABEL[arm]+(' / 均值' if stat=='mean' else ' / 标准差'))
                    ax.set_xlabel('相对局部峰时间 / ns');ax.set_ylabel('电压 / mV' if stat=='mean' else '逐点SD / mV')
                    ax.axvline(0,color='#a4adb5',lw=.8,ls=':')
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.900),ncol=4,frameon=False)
            foot(fig,'有效输出为峰前16点至峰后239点；内部384点保留100点基线和有限寻峰上下文。\n此图为人口分布，包含输入人口差异，不能解释为逐输入条件响应的真实方差。')
            save(fig,('02' if split=='TRAIN' else '03')+'_'+split.lower()+'_waveforms',pdf)

        fig,axes=plt.subplots(2,3,figsize=(12.4,8.8));fig.subplots_adjust(left=.075,right=.975,bottom=.14,top=.86,hspace=.45,wspace=.3)
        title(fig,'电荷条件下的PSD宽度','各生成臂和REAL使用相同的输出Q定义与TRAIN分箱边界；箱内父事件人口仍可能不同。')
        for i,split in enumerate(('TRAIN','DEV')):
            for j,arm in enumerate(ARMS):
                ax=axes[i,j]
                for h in HYP:
                    rows=generations[split]['summary'][arm][h]['output_Q_in_REAL_TRAIN_bins']
                    vals=[row['PSD'].get('std',np.nan) for row in rows]
                    ax.plot(np.arange(1,9),vals,'o-',ms=3.5,color=COLORS[h],label=LABEL[h])
                rows=generations[split]['summary']['REAL']['output_Q_in_REAL_TRAIN_bins']
                ax.plot(np.arange(1,9),[row['PSD'].get('std',np.nan) for row in rows],'s--',ms=3.5,color=COLORS['REAL'],label='REAL')
                ax.set_xticks(np.arange(1,9));ax.set_xlabel('输出Q箱（低 → 高）');ax.set_ylabel('PSD SD');ax.set_title(split+' / '+LABEL[arm])
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.900),ncol=4,frameon=False)
        foot(fig,'负PSD与保护阈值附近事件全部保留；未把PSD裁剪到[0,1]。低支持箱保留并在原始JSON中标记。\n宽度变大、变小或靠近REAL本身都不能独立证明物理响应正确。')
        save(fig,'04_output_charge_conditional_psd',pdf)

        fig,axes=plt.subplots(2,2,figsize=(12.4,8.8));fig.subplots_adjust(left=.08,right=.97,bottom=.14,top=.83,hspace=.46,wspace=.29)
        title(fig,'相同SOURCE父人口：随机响应的PSD展宽','固定SOURCE电荷箱；完整随机臂 / 对照臂。带状区域为1,000次父事件重采样的95%百分位区间。')
        unsupported={}
        for i,split in enumerate(('TRAIN','DEV')):
            unsupported[split]={}
            for j,reference in enumerate(('deterministic','simple')):
                ax=axes[i,j];entry=bootstrap[split]['comparisons'][reference]['std_full_over_reference']
                point=np.asarray(entry['point'],float)[1,:,:,1];ci=np.asarray(entry['percentile95'],float)[1,:,:,1]
                support=np.asarray(bootstrap[split]['comparison_supported'][reference],bool)[1]
                unsupported[split][reference]=int((~support).sum())
                for k,h in enumerate(HYP):
                    ax.plot(np.arange(1,9),point[k],'o-',ms=3.5,color=COLORS[h],label=LABEL[h])
                    ax.fill_between(np.arange(1,9),ci[k,:,0],ci[k,:,1],where=support[k],color=COLORS[h],alpha=.13)
                    if (~support[k]).any():ax.scatter(np.arange(1,9)[~support[k]],point[k,~support[k]],marker='x',s=45,color=COLORS[h])
                ax.axhline(1,color='#7f8a95',ls=':',lw=1);ax.set_xticks(np.arange(1,9));ax.set_xlabel('固定SOURCE Q箱');ax.set_ylabel('PSD SD比值')
                ax.set_title(split+' / 完整随机 ÷ '+LABEL[reference])
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.9),ncol=3,frameon=False)
        foot(fig,'父事件的4次响应始终共同重采样；两臂与三种噪声假设共享父权重。REAL未被重采样。\n至少32个父事件才绘制区间；低支持点用叉号标出。该区间描述抽样变化，不包含未知噪声模型或标定误差。')
        save(fig,'05_paired_parent_width_comparisons',pdf)

        fig,axes=plt.subplots(2,2,figsize=(12.4,8.8));fig.subplots_adjust(left=.08,right=.97,bottom=.15,top=.86,hspace=.44,wspace=.3)
        title(fig,'优化记录与独立100点基线','3,000均值 + 12,000联合 + 4,000低学习率；电子学噪声005与人口权重始终冻结。')
        x=np.arange(50,19001,50)
        for arm in ARMS:
            rows=journals[arm]
            axes[0,0].plot(x,np.array([row['weighted_total'] for row in rows]).reshape(-1,50).mean(1),label=LABEL[arm],color=COLORS[arm])
            axes[0,1].plot(x,np.array([np.mean([row['hypotheses'][h]['components']['QPSD'] for h in HYP]) for row in rows]).reshape(-1,50).mean(1),label=LABEL[arm],color=COLORS[arm])
        for ax in axes[0]:
            ax.axvline(3000,color='#7f8a95',ls=':',lw=1);ax.axvline(15000,color='#7f8a95',ls=':',lw=1)
            ax.set_xlabel('每臂G更新');ax.legend(fontsize=8)
        axes[0,0].set_title('加权训练目标 / 50步均值');axes[0,0].set_ylabel('损失（跨阶段含义不同）')
        axes[0,1].set_title('三种假设的QPSD项均值');axes[0,1].set_ylabel('QPSD损失')
        for j,split in enumerate(('TRAIN','DEV')):
            ax=axes[1,j];c=curves[split]
            ax.plot(np.arange(100),1000*c['generated_baseline100_std'],label='生成 / 三臂共享',color=COLORS['full'])
            ax.plot(np.arange(100),1000*c['REAL_baseline100_std'],label='REAL',color=COLORS['REAL'],ls='--')
            ax.set_title(split+' / 实际基线段逐点SD');ax.set_xlabel('内部前100点基线样本');ax.set_ylabel('基线SD / mV');ax.legend()
        foot(fig,'256点脉冲窗的前100点不作为基线；基线来自内部载体的独立前100点。噪声分支未在本轮更新。\n训练损失下降不替代独立评价；REAL-REAL与DEV均来自同一采集，仍缺少独立标定和跨采集检验。')
        save(fig,'06_training_and_baseline',pdf)

    headline=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='fixed_window008_report_built_pending_visual_QA',
        G_updates=57000,D_updates=144000,historical_actual_G_updates=57667,historical_actual_D_updates=146001, exact_replay=replay['status'],C2ST_updates=0,noise_updates=0,population_updates=0,
        effective_samples=256,internal_carrier_samples=384,scientific_status='response_match_not_established',
        worst_hypothesis_distances=worst,relative_reduction_of_hypothesis_maxima=relative,
        full_max_over_REAL_REAL=controls,unsupported_width_comparison_cells=unsupported,
        comparison_note='Maxima may come from different hypotheses; full per-hypothesis values are retained in the source summary.',
        new_model_calls=0,new_training_updates=0,independent_acquisition_available=False,
        evaluation_summary_sha256=sha(evaluation/'SUMMARY.json'),training_complete_sha256=sha(run/'runs/science/TRAINING_COMPLETE.json'))
    write(out/'SUMMARY.json',headline)
    rows=''.join('<tr><td>'+split+'</td><td>'+LABEL[arm]+'</td><td>'+LABEL[h]+'</td>'+''.join('<td>'+fmt(distributions[split][arm][h]['components'][metric])+'</td>' for metric in METRICS)+'</tr>' for split in ('TRAIN','DEV') for arm in ARMS for h in HYP)
    controls_html=''.join('<tr><td>'+split+'</td><td colspan="2">REAL-REAL</td>'+''.join('<td>'+fmt(distributions[split]['REAL_REAL']['components'][metric])+'</td>' for metric in METRICS)+'</tr>' for split in ('TRAIN','DEV'))
    figures_html=''.join('<section><img src="figures/'+name+'.png" alt="'+name+'"></section>' for name in figures)
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>N01-008 256点响应比较</title><style>body{font-family:"Microsoft YaHei",sans-serif;background:#f1f5f8;color:#172e43;margin:0}main{max-width:1220px;margin:32px auto;padding:28px;background:white}h1{font-size:29px;margin-top:0}p{line-height:1.8}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #dae3e9}th:first-child,td:first-child{text-align:left}th{background:#e7f0f6}img{width:100%;display:block}section{margin:32px 0}a{color:#006e87}.status{padding:16px 20px;background:#e7f2f5;border-left:5px solid #007f8b}code{font-family:Consolas,monospace}</style><main>
<h1>N01-008：256点随机响应固定终点比较</h1><p class="status"><b>三组固定训练、终点评价和数值回放已完成；物理响应匹配尚未建立。</b><br>每臂19,000 G，合计57,000 G / 144,000 D；本轮噪声更新、人口权重更新与C2ST训练均为0。</p>
<p>有效观察窗为峰前16点至峰后239点，1 ns/点。模型内部保留384点载体，供原100点基线扣除与有限寻峰使用。损失、判别器、评价和交付波形均为256点。REAL使用既有ADC电压，生成波形只数字化一次；全部合法TRAIN事件保留。</p>
<p>完整随机、确定性及简单随机三组使用相同数据、共同初始化与固定更新数。短相关、慢相关和恒定公共分量分别报告，它们只构成敏感性分析。TRAIN及已暴露DEV分开评价；没有用DEV选择终点、配置或噪声假设。</p>
<p><a href="N01_WINDOW008_RESUME01_REPORT.pdf">下载图表PDF</a> · <a href="SUMMARY.json">读取报告摘要与计算口径</a></p>
<h2>匹配父事件数量的固定投影距离</h2><p>TRAIN每域2,048、DEV每域256个父事件；每个仿真父事件取第0次响应。所有行保留，不对输出进行筛选。较小距离仅代表相应统计更接近，不是科学通过判定。</p>
<table><thead><tr><th>角色</th><th>模型</th><th>噪声假设</th><th>波形</th><th>QPSD</th><th>基线</th></tr></thead><tbody>'''+rows+controls_html+'''</tbody></table>'''+figures_html+'''
<h2>证据边界</h2><p>当前SOURCE仍是依赖少量沉积条件的解析清洁输入，模型是相对于该输入和既有软件电压口径的有效响应候选。源电荷不等于已标定入射能量，单次无配对采集不能独立分开增益、淬灭和输入人口影响。噪声延伸的三种固定协方差不是置信集合；人工软件分组也不代表独立采集。</p><p>没有同一物理事件的重复实测真值，逐输入响应方差尚不能唯一识别。当前主体评价未训练独立域分类器；扩展的逐输入重复、固定随机数输入干预与人口敏感性诊断如有执行，应以其独立回执为准。训练结束、数值回放一致与物理有效性分别报告。</p></main></html>'''
    (out/'N01_WINDOW008_RESUME01_REPORT.html').write_text(body,encoding='utf8')
    artifacts={path.relative_to(out).as_posix():sha(path) for path in sorted(out.rglob('*')) if path.is_file()}
    write(out/'BUILD_RECEIPT.json',dict(UTC=headline['UTC'],files=artifacts,report_pages=len(figures),
        source_data_role='saved fixed TRAIN and exposed DEV endpoint outputs only',new_model_calls=0,new_training_updates=0,
        visual_QA='pending_rendered_inspection',implementation_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status='report_built_pending_visual_QA',out=str(out),pages=len(figures),files=len(artifacts))))


if __name__=='__main__':main()
