"""Present saved partial-stage TRAIN outputs; no model or original data read."""
from pathlib import Path
import os
import sys
import datetime as dt
import hashlib
import html
import json
import shutil
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']='E:/GAN/runtime_tmp/matplotlib_response007'
os.environ['TMP']=os.environ['TEMP']='E:/GAN/runtime_tmp'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
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
ARMS=('deterministic','full');HYP=('fast','slow','common_level')
LABEL={'deterministic':'确定性响应','full':'完整随机响应','fast':'快衰减','slow':'慢相关','common_level':'公共恒定分量'}
COLOR={'fast':'#007a87','slow':'#d07824','common_level':'#7759a6','REAL':'#223344'}
SCI=ROOT/'runs/response_007_20261013/deadline01/runs/science'
OUT=ROOT/'reports/response_007_20261013/deadline01'
FIGURES=['01_matched_distances_and_sensitivity','02_waveform_and_population','03_training_and_source_bins']

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def fmt(v):return '未定义' if v is None else f'{v:.4g}'
def save(fig,name,pdf):
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=170,facecolor='white')
    pdf.savefig(fig,facecolor='white');plt.close(fig)
def foot(fig,txt):fig.text(.06,.025,txt,fontsize=9,color='#33485a',va='bottom')

def main():
    complete=read(SCI/'COMPLETE.json');summary=read(SCI/'TRAIN_diagnostics/SUMMARY.json')
    verification=read(ROOT/'receipts/response_007_20261013/deadline01/ARTIFACT_VERIFICATION.json')
    assert complete['G_updates']==8000 and verification['status']=='passed'
    if OUT.exists():
        backup=ROOT/'runs/response_007_20261013/deadline01/intermediate_backups/report_before_legend_fix'
        assert not backup.exists(),'Preserve the single reviewed layout revision'
        for name,digest in read(OUT/'BUILD_RECEIPT.json')['files'].items():assert sha(OUT/name)==digest
        shutil.copytree(OUT,backup)
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    obs=load(SCI/'TRAIN_diagnostics/TRAIN_observables.npz')
    curves=load(SCI/'TRAIN_diagnostics/TRAIN_curves.npz')
    distance=summary['matched_distances'];sens=summary['sensitivity']
    worst=summary['worst_hypothesis_distances']
    relative={k:1-worst['full'][k]/worst['deterministic'][k] for k in ['wave','QPSD','noise']}
    versus_control={k:worst['full'][k]/distance['REAL_REAL']['components'][k] for k in ['wave','QPSD','noise']}
    journals={a:[json.loads(x) for x in (SCI/'train'/a/'training.jsonl').read_text(encoding='utf8').splitlines()] for a in ARMS}
    pdfpath=OUT/'N01_RESPONSE007_STAGE_REPORT.pdf'
    with PdfPages(pdfpath) as pdf:
        fig=plt.figure(figsize=(12.4,8.8));grid=fig.add_gridspec(2,3,left=.07,right=.97,bottom=.14,top=.84,wspace=.36,hspace=.52)
        fig.suptitle('N01-007：每臂4,000步的TRAIN阶段比较',fontsize=17,y=.966)
        fig.text(.5,.926,'两臂各3,000步平均响应 + 1,000步联合随机；原19,000步/臂末点未完成',ha='center',fontsize=11)
        for j,key in enumerate(['wave','QPSD','noise']):
            ax=fig.add_subplot(grid[0,j]);positions=np.arange(2)
            for i,h in enumerate(HYP):
                values=[distance[a][h]['components'][key] for a in ARMS]
                ax.bar(positions+(i-1)*.24,values,width=.22,color=COLOR[h],label=LABEL[h])
            rr=distance['REAL_REAL']['components'][key]
            ax.axhline(rr,color=COLOR['REAL'],ls='--',lw=1.5,label='REAL-REAL')
            ax.set_xticks(positions,[LABEL[a] for a in ARMS]);ax.set_ylabel('固定投影分布距离（越低越接近）')
            ax.set_title({'wave':'480点波形','QPSD':'电荷与PSD','noise':'100点基线'}[key])
        handles,labels=fig.axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.905),ncol=4,frameon=False)
        for i,a in enumerate(ARMS):
            ax=fig.add_subplot(grid[1,i]);point=np.asarray(sens['point'][i][2:],float);ci=np.asarray(sens['percentile_95_interval'][i][2:],float)
            yerr=np.array([point-ci[:,0],ci[:,1]-point]);ax.errorbar(np.arange(1,9),point,yerr=np.maximum(yerr,0),fmt='o-',color=('#057b8b' if i==0 else '#ba6231'),capsize=3)
            ax.axhline(1,color='#777',lw=1,ls=':');ax.set_xticks(np.arange(1,9));ax.set_xlabel('各臂自身固定信号Q箱')
            ax.set_ylabel('三假设PSD SD最大/最小');ax.set_title(LABEL[a]+'：假设敏感性')
        ax=fig.add_subplot(grid[1,2]);ax.axis('off')
        lines=['随机臂相对确定性臂（最差假设）',
            f"wave：{worst['deterministic']['wave']:.4f} -> {worst['full']['wave']:.4f}（-{relative['wave']:.1%}）",
            f"QPSD：{worst['deterministic']['QPSD']:.4f} -> {worst['full']['QPSD']:.4f}（-{relative['QPSD']:.1%}）",
            'noise：0.1369，冻结基线无变化','',
            '累计G 8,000；本执行新增7,999','训练D 6,000；噪声/C2ST 0',
            '6,593个SOURCE父 × 4次生成','21,047条REAL；距离各2,048条',
            '95%区间：1,000次父级重采样','', '完整末点 / 物理响应匹配未建立']
        ax.text(0,1,'\n'.join(lines),va='top',fontsize=9.8,linespacing=1.35)
        foot(fig,'仅TRAIN；未使用DEV选点。两臂自身固定Q箱的父人口可不同，敏感性比值不是臂优劣排名。\nREAL-REAL是同一采集内对照，不代表独立采集验收；无新增科学通过阈值。')
        save(fig,FIGURES[0],pdf)

        fig,axes=plt.subplots(2,3,figsize=(12.4,8.8));fig.subplots_adjust(left=.07,right=.97,bottom=.14,top=.87,wspace=.34,hspace=.46)
        fig.suptitle('完整原人口：波形与Q-PSD分布',fontsize=17,y=.966)
        fig.text(.5,.925,'每种假设都保留；人口统计不重权，未按结果挑选假设或样本',ha='center',fontsize=11)
        time=np.arange(480)
        for i,a in enumerate(ARMS):
            for j,stat in enumerate(['mean','variance']):
                ax=axes[i,j]
                for h in HYP:
                    v=curves[f'{a}_{h}_{stat}'];v=np.sqrt(np.maximum(v,0)) if stat=='variance' else v
                    ax.plot(time,1000*v,color=COLOR[h],lw=1.3,label=LABEL[h])
                real=curves['REAL_'+stat];real=np.sqrt(np.maximum(real,0)) if stat=='variance' else real
                ax.plot(time,1000*real,color=COLOR['REAL'],lw=1.5,ls='--',label='REAL')
                ax.axvline(100,color='#999',ls=':',lw=1);ax.axvline(400,color='#999',ls=':',lw=1)
                ax.set_xlabel('对齐后采样点（软件1 ns）');ax.set_ylabel('幅度 / mV' if stat=='mean' else '总体SD / mV')
                ax.set_title(LABEL[a]+('：均值' if stat=='mean' else '：标准差'))
                if i==0 and j==0:ax.legend(ncol=2)
            ax=axes[i,2]
            ax.hist(obs['REAL_PSD'],bins=np.linspace(0,1,81),weights=np.full(len(obs['REAL_PSD']),80/len(obs['REAL_PSD'])),histtype='step',lw=1.5,color=COLOR['REAL'],ls='--',label='REAL')
            for h in HYP:
                values=obs[f'{a}_{h}_PSD'];ax.hist(values,bins=np.linspace(0,1,81),weights=np.full(len(values),80/len(values)),histtype='step',lw=1.2,color=COLOR[h],label=LABEL[h])
            ax.set_xlabel('PSD');ax.set_ylabel('密度');ax.set_title(LABEL[a]+'：PSD总体分布')
        outside={a:{h:int(((obs[f'{a}_{h}_PSD']<0)|(obs[f'{a}_{h}_PSD']>1)).sum()) for h in HYP} for a in ARMS}
        outside['REAL']=int(((obs['REAL_PSD']<0)|(obs['REAL_PSD']>1)).sum())
        foot(fig,f"PSD图只显示[0,1]；范围外计数：REAL={outside['REAL']}，确定性[快/慢/恒定]="+
            '/'.join(str(outside['deterministic'][h]) for h in HYP)+'，随机='+
            '/'.join(str(outside['full'][h]) for h in HYP)+'。\n所有事件均计入统计；100点基线不能验证480点长程噪声，仍只有1个已确认采集。')
        save(fig,FIGURES[1],pdf)

        fig,axes=plt.subplots(2,2,figsize=(12.4,8.8));fig.subplots_adjust(left=.08,right=.97,bottom=.14,top=.88,wspace=.3,hspace=.48)
        fig.suptitle('相同起点与步数：训练记录和分箱宽度',fontsize=17,y=.966)
        fig.text(.5,.925,'共同前3,000步作逐位核验；全程日志、失败第1步和恢复状态均保留',ha='center',fontsize=11)
        for i,a in enumerate(ARMS):
            rows=journals[a];x=np.arange(1,4001);y=np.array([r['weighted_total'] for r in rows])
            ax=axes[0,0];ax.plot(x,y,color=('#057b8b' if i==0 else '#ba6231'),alpha=.16,lw=.6)
            blocks=y.reshape(160,25).mean(1);ax.plot(np.arange(25,4001,25),blocks,color=('#057b8b' if i==0 else '#ba6231'),label=LABEL[a]+'（25步均值）')
            ax=axes[0,1]
            for h in HYP:
                vals=np.array([r['hypotheses'][h]['components']['QPSD'] for r in rows]).reshape(160,25).mean(1)
                ax.plot(np.arange(25,4001,25),vals,color=COLOR[h],ls='-' if i==0 else '--',label=LABEL[a]+'/'+LABEL[h])
            ax=axes[1,i]
            for h in HYP:
                vals=[r['PSD']['SD'] for r in summary['population_summary'][a][h]['output_Q_bins']]
                ax.plot(np.arange(1,9),vals,'o-',color=COLOR[h],label=LABEL[h])
            ax.plot(np.arange(1,9),[r['PSD']['SD'] for r in summary['population_summary']['REAL']['output_Q_bins']],'s--',color=COLOR['REAL'],label='REAL')
            ax.set_xticks(np.arange(1,9));ax.set_xlabel('输出Q箱（统一TRAIN边界）');ax.set_ylabel('PSD SD');ax.set_title(LABEL[a]+'：输出Q分箱')
        for ax in axes[0]:ax.axvline(3000,color='#777',ls=':',lw=1);ax.set_xlabel('累计G更新/臂');ax.legend(fontsize=8)
        axes[0,0].set_title('加权训练目标（含对抗项）');axes[0,0].set_ylabel('损失；不同训练阶段不作同尺度验收')
        axes[0,1].set_title('QPSD训练项（25步均值）');axes[0,1].set_ylabel('QPSD损失')
        axes[1,0].legend(ncol=2)
        foot(fig,'输出Q再分箱可引入成员迁移：必须与固定信号Q箱及原SOURCE-Q箱共同阅读。\n训练目标不是物理验收分数；未完成低学习率阶段、DEV评价、C2ST或正式重复父测试。')
        save(fig,FIGURES[2],pdf)
    table=[]
    for a in ARMS:
        for h in HYP:
            row=distance[a][h]['components'];table.append([LABEL[a],LABEL[h]]+[fmt(row[k]) for k in ['wave','QPSD','noise']])
    rr=distance['REAL_REAL']['components'];table.append(['REAL-REAL','同采集内对照']+[fmt(rr[k]) for k in ['wave','QPSD','noise']])
    body=''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in table)
    figures=''.join('<figure><a href="figures/'+name+'.png"><img src="figures/'+name+'.png"></a></figure>' for name in FIGURES)
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>N01 007 有限阶段报告</title>
<style>body{{font-family:Microsoft YaHei,sans-serif;max-width:1160px;margin:36px auto;color:#203344;line-height:1.65;padding:0 24px}}h1{{color:#164f77}}.lead{{background:#eaf2f7;padding:18px 22px;border-left:5px solid #166185}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #c9d8e4;text-align:left}}img{{width:100%;height:auto}}figure{{margin:30px 0}}a{{color:#17639a}}</style>
<h1>N01-007：有限阶段比较已完成</h1><p class="lead">两臂各4,000步，累计8,000 G（本执行新增7,999）与6,000训练D；原每臂19,000步末点尚未完成。科学状态仍为 <b>response_match_not_established</b>。</p>
<p><b>本轮实际收益：</b>完整随机臂在三种假设下的wave和QPSD距离均低于确定性臂；最差假设距离分别降低{relative['wave']:.1%}和{relative['QPSD']:.1%}。完整随机臂的最差wave/QPSD/noise距离仍约为同采集内REAL-REAL对照的{versus_control['wave']:.2f}/{versus_control['QPSD']:.2f}/{versus_control['noise']:.2f}倍。基线噪声固定，未在本轮改善。低Q固定信号箱的PSD宽度假设比为{sens['point'][1][2]:.4f}，父级95%区间[{sens['percentile_95_interval'][1][2][0]:.4f}, {sens['percentile_95_interval'][1][2][1]:.4f}]，不能称长程假设影响已消除。</p>
<p>模型在fast、slow、common_level三种冻结长程噪声假设下训练，逐假设计算损失后等权平均。TRAIN诊断包含6,593个SOURCE父各4次生成和原21,047条REAL；等样本距离使用每侧2,048条及同采集内REAL-REAL对照。没有DEV评价、C2ST或新公开发布。</p>
<p><a href="N01_RESPONSE007_STAGE_REPORT.pdf">打开三页PDF图表</a> · <a href="SUMMARY.json">结构化结果</a> · <a href="../../../receipts/response_007_20261013/deadline01/ARTIFACT_VERIFICATION.json">产物核验</a> · <a href="../../../receipts/response_007_20261013/deadline01/quiet_evidence/QUIET_EVIDENCE.md">长静默证据</a></p>
<h2>等样本分布距离</h2><table><tr><th>响应</th><th>噪声假设</th><th>wave</th><th>QPSD</th><th>noise</th></tr>{body}</table>
<p>各项不是通用通过分数；REAL-REAL给出本次有限样本对照。两臂自身固定信号Q箱可有不同父人口，不能把敏感性比值更小直接解读为更接近REAL。输出Q箱和原SOURCE-Q箱的完整摘要保存在原科学SUMMARY。</p>
{figures}<h2>数据与解释边界</h2><p>已核对的原记录在20点均值降采样后只有500点，半高上升沿位于120-140点，没有480点脉冲前窗口。100点基线只是处理约定；长程噪声、硬件标定与独立采集验证仍缺少证据。此次元数据核对未解码新波形或DEV。</p>
<p>初次运行在第1个G更新后因状态字段重复退出，该检查点与失败回执完整保留。恢复后未重复该次更新。当前报告由主代理自检，原生复现回执另行列入最终交付；不存在第二方独立审核。低学习率阶段与原完整末点未执行，训练集改善不构成物理响应匹配。</p></html>'''
    (OUT/'N01_RESPONSE007_STAGE_REPORT.html').write_text(doc,encoding='utf8')
    exported=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='TRAIN_partial_stage_report',
        budgets=dict(cumulative_G=8000,new_execution_G=7999,D=6000,noise=0,C2ST=0),
        original_endpoint_reached=False,scientific_status='response_match_not_established',
        matched_distances=distance,worst_hypothesis_distances=summary['worst_hypothesis_distances'],sensitivity=sens,
        engineering=summary['engineering'],PSD_outside_display_0_to_1=outside,
        interpretation=dict(full_vs_deterministic_worst_distance_relative_reduction=relative,
            full_worst_distance_over_REAL_REAL_control=versus_control,
            all_three_hypotheses_wave_and_QPSD_lower=all(distance['full'][h]['components'][k]<distance['deterministic'][h]['components'][k] for h in HYP for k in ['wave','QPSD']),
            all_fixed_signal_bins_have_registered_support=all(all(sens['support'][a]['supported']) for a in ARMS),
            scientific_equivalence_established=False,conditional_widths_not_uniformly_matched=True,
            next_action='Preserve full as a candidate. Resume the remaining fixed endpoint only in a later execution window with unchanged protocol; obtain long-quiet and independent-acquisition evidence before physical response claims.'),
        scientific_summary_sha256=sha(SCI/'TRAIN_diagnostics/SUMMARY.json'),complete_sha256=sha(SCI/'COMPLETE.json'),
        verification_sha256=sha(ROOT/'receipts/response_007_20261013/deadline01/ARTIFACT_VERIFICATION.json'))
    (OUT/'SUMMARY.json').write_text(json.dumps(exported,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    files={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='BUILD_RECEIPT.json'}
    (OUT/'BUILD_RECEIPT.json').write_text(json.dumps(dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),files=files,
        report_pages=3,source_data_role='saved TRAIN outputs only',new_model_calls=0,new_training_updates=0,
        visual_QA='pending_rendered_inspection',implementation_sha256=sha(Path(__file__))),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='built_not_yet_visual_QA',report=str(OUT),files=len(files))))

if __name__=='__main__':main()
