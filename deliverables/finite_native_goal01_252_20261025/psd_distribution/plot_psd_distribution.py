"""Plot saved FIT PSD distributions for REAL, M0 and the sole job252 endpoint."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, os

out=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path('E:/GAN/research/N01')
saved=root/'receipts/finite_native_goal01_recovery01_20261025/wake_20261025_0627'
base=root/'receipts/global_memory_repair_20261021/full_population_hard_readout01/terminal/run_01'
config=json.loads((root/'configs/finite_native_goal01_recovery01_P2_TEMPLATE_20261025.json').read_bytes())
receipt=json.loads((saved/'REMOTE_STATE.json').read_bytes())
paths={'REAL':base/'REAL_HARD_READOUT.npz','M0':base/'HARD_READOUT_fast__original.npz','endpoint252':saved/'terminal/P1/FIT_POINT_HARD_READOUT.npz','point':saved/'terminal/P1/FINITE_NATIVE_POINT.npz','state':base/'HARD_METRIC_STATE.npz'}
pins={}
for key,path in paths.items():
    if key in ['REAL','M0','state']:
        remote=config[{'REAL':'REAL_readout','M0':'original_readout','state':'metric_state'}[key]]
        expected=config['active_inputs'][remote]
    else:
        expected=receipt['files']['P1/'+path.name]
    data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    assert len(data)==expected['bytes'] and digest==expected['sha256'],key
    pins[key]={'path':str(path),'bytes':len(data),'sha256':digest}
with np.load(paths['point'],allow_pickle=False) as z:
    source_fit=z['source_FIT_rows'].copy();real_fit=z['REAL_FIT_rows'].copy()
with np.load(paths['endpoint252'],allow_pickle=False) as z:
    generated_indices=z['global_generated_indices'].copy()
with np.load(paths['state'],allow_pickle=False) as z:q_floor=float(z['loss:q_floor'])
assert len(source_fit)==6465 and len(real_fit)==18948 and len(generated_indices)==103440
parent=(generated_indices%52744)//8
assert set(parent)==set(source_fit)
for bank in range(2):
    np.testing.assert_array_equal(np.bincount(parent[generated_indices//52744==bank],minlength=6593)[source_fit],np.full(6465,8))

values={};summary={};samples={}
for key in ['REAL','M0','endpoint252']:
    with np.load(paths[key],allow_pickle=False) as z:
        rows=real_fit if key=='REAL' else generated_indices if key=='M0' else np.arange(len(generated_indices))
        wave=z['window_adc_codes'][rows].astype(np.float64)/16383-.5-z['baseline_mean_V'][rows,None]
        total=wave.sum(1);tail=wave[:,46:256].sum(1)
        denominator=np.where(total<0,-1.,1.)*np.maximum(np.abs(total),q_floor)
        psd=tail/denominator
        assert np.isfinite(psd).all()
        guarded=np.abs(total)<q_floor
        np.testing.assert_array_equal(guarded,z['flag_PSD_guarded'][rows])
        np.testing.assert_array_equal(total<=0,z['flag_Q_nonpositive'][rows])
        flags={k[5:]:int(z[k][rows].sum()) for k in z.files if k.startswith('flag_')}
    values[key]=psd
    samples[key+'_PSD']=psd;samples[key+'_Q']=total;samples[key+'_tail']=tail
    summary[key]={'rows':len(psd),'mean':float(psd.mean()),'std':float(psd.std(ddof=1)),'min':float(psd.min()),'max':float(psd.max()),'q05':float(np.quantile(psd,.05)),'median':float(np.median(psd)),'q95':float(np.quantile(psd,.95)),'guarded_rows':int(guarded.sum()),'nonpositive_Q_rows':int((total<=0).sum()),'flags':flags}
    del wave
samples.update(REAL_FIT_rows=real_fit,SOURCE_FIT_rows=source_fit,global_generated_indices=generated_indices)
np.savez_compressed(out/'PSD_VALUES.npz',**samples)

# One bandwidth from REAL is used for all three visualization curves.
real=values['REAL'];iqr=np.quantile(real,.75)-np.quantile(real,.25)
bandwidth=float(.9*min(real.std(ddof=1),iqr/1.34)*len(real)**(-.2))
x=np.linspace(-.12,.50,1001)
density={};ecdf={}
for key,value in values.items():
    acc=np.zeros(len(x))
    for first in range(0,len(value),1024):
        u=(x[:,None]-value[first:first+1024][None])/bandwidth
        acc+=np.exp(-.5*u*u).sum(1)
    density[key]=acc/(len(value)*bandwidth*np.sqrt(2*np.pi))
    ecdf[key]=np.searchsorted(np.sort(value),x,side='right')/len(value)
    summary[key]['outside_main_xrange']=int(((value<x[0])|(value>x[-1])).sum())
    summary[key]['outside_main_xrange_percent']=100*summary[key]['outside_main_xrange']/len(value)
metrics={}
for key in ['M0','endpoint252']:
    support=np.unique(np.r_[values['REAL'],values[key]])
    f=np.searchsorted(np.sort(values['REAL']),support,side='right')/len(values['REAL'])
    g=np.searchsorted(np.sort(values[key]),support,side='right')/len(values[key])
    metrics[key]={'marginal_PSD_W1':float(np.sum(abs(f[:-1]-g[:-1])*np.diff(support))),'marginal_PSD_KS_distance':float(np.max(abs(f-g)))}
with (out/'PSD_CURVES.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['PSD','REAL_density','M0_density','endpoint252_density','REAL_ECDF','M0_ECDF','endpoint252_ECDF','M0_minus_REAL_ECDF','endpoint252_minus_REAL_ECDF'])
    w.writerows(zip(x,density['REAL'],density['M0'],density['endpoint252'],ecdf['REAL'],ecdf['M0'],ecdf['endpoint252'],ecdf['M0']-ecdf['REAL'],ecdf['endpoint252']-ecdf['REAL']))

font=Path('C:/Windows/Fonts/msyh.ttc');assert font.exists()
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path','pdf.fonttype':42})
colors={'REAL':'#252525','M0':'#0072B2','endpoint252':'#D55E00'}
labels={'REAL':'真实数据 REAL','M0':'原模型 M0','endpoint252':'252 返回端点'}
styles={'REAL':'-','M0':'--','endpoint252':'-'}
fig,(ax,delta_ax)=plt.subplots(1,2,figsize=(12.8,4.8),gridspec_kw={'width_ratios':[1.4,1]})
for key in values:
    ax.plot(x,density[key],color=colors[key],ls=styles[key],lw=2 if key=='REAL' else 1.8,label=labels[key])
ax.set(xlim=(x[0],x[-1]),xlabel=r'PSD = $Q_{\mathrm{tail}}/Q_{\mathrm{total}}$',ylabel='概率密度',title='PSD 分布曲线')
ax.set_ylim(bottom=0);ax.legend(frameon=False,fontsize=10)
for key in ['M0','endpoint252']:
    delta_ax.plot(x,ecdf[key]-ecdf['REAL'],color=colors[key],ls=styles[key],lw=1.6,label=labels[key])
delta_ax.axhline(0,color='#555555',lw=.8)
delta_ax.set(xlim=(x[0],x[-1]),xlabel='PSD',ylabel=r'$F_{\mathrm{model}}-F_{\mathrm{REAL}}$',title='相对真实数据的累计分布差')
delta_ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
for a in [ax,delta_ax]:a.grid(axis='y',alpha=.18)
fig.suptitle('N01 · 真实数据 / M0 / 252 返回端点',fontsize=16,y=.98)
fig.text(.5,.88,'fast 假设 · REAL FIT 18,948 条 · 模型各 6,465 个父事件 × 2 库 × 8 次',ha='center',fontsize=10.5)
outside='；'.join(f"{labels[k]} {summary[k]['outside_main_xrange']}/{summary[k]['rows']}" for k in values)
fig.text(.5,.025,f'三条密度曲线统一高斯带宽 h={bandwidth:.5f}；区间外样本：{outside}（仍计入归一化）',ha='center',fontsize=9.5)
fig.subplots_adjust(left=.065,right=.985,bottom=.17,top=.76,wspace=.30)
for suffix in ['png','svg','pdf']:fig.savefig(out/('PSD_DISTRIBUTION.'+suffix),dpi=200,facecolor='white')
plt.close(fig)

fig,ax=plt.subplots(figsize=(10.2,4.6),layout='constrained')
for key,value in values.items():
    sx=np.sort(value);sy=np.arange(1,len(sx)+1)/len(sx)
    ax.step(np.r_[sx[0]-.02,sx],np.r_[0.,sy],where='post',color=colors[key],ls=styles[key],lw=1.6,label=f'{labels[key]}  (n={len(sx):,})')
ax.set(xlabel='PSD',ylabel='累计概率',ylim=(0,1.015),title='PSD 全范围经验累计分布 · 所有样本均保留')
ax.legend(frameon=False,loc='upper left');ax.grid(alpha=.18)
for suffix in ['png','svg','pdf']:fig.savefig(out/('PSD_FULL_RANGE_CDF.'+suffix),dpi=180,facecolor='white')
plt.close(fig)

metadata={'UTC':datetime.now(timezone.utc).isoformat(),'status':'PLOTTED_FROZEN_PSD_DISTRIBUTIONS','source_job':252,'hypothesis':'fast','definition':'PSD=sum(window_V[:,46:256])/sum(window_V[:,0:256]); original signed q_floor protection retained','q_floor':q_floor,'population':'All18948 REAL FIT; same6465 SOURCE FIT parents with2banks*8draws,103440 rows per model; no row removal or new draws','kde_bandwidth':bandwidth,'bandwidth_rule':'.9*min(REAL sample std,REAL IQR/1.34)*n_REAL**(-1/5); identical absolute bandwidth for all curves','main_xrange':x[[0,-1]].tolist(),'full_range_CDF':'PSD_FULL_RANGE_CDF.png','rows_excluded_from_metrics_or_normalization':0,'summary':summary,'marginal_metrics':metrics,'metric_scope':'Unconditional one-dimensional PSD distances, distinct from the original joint/conditional QPSD objective; no significance inference from repeated parent draws.','input_pins':pins,'parameter_fits':0,'candidate_selections':0,'new_model_calls':0,'new_random_draws':0,'scientific_acceptance':False,'restricted_average_response_acceptance':False,'NAS_access':False}
(out/'PLOT_RECEIPT.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(out/'README.md').write_text(f'''# 252 PSD 分布曲线

比较同一冻结FIT口径下的REAL、原模型M0和252唯一返回端点。主图为统一高斯带宽（h={bandwidth:.6f}）的概率密度及相对REAL的经验累计分布差；完整ECDF单列显示全部范围。全部样本参与密度归一化和累计概率，未裁掉负PSD或其他尾部样本。模型的103440条观察来自6465个父事件各2库×8次，不能视为103440个独立父事件。

PSD按原256点基线扣除窗口计算：总积分为[0,256)，尾积分为[46,256)，原q_floor={q_floor:.12g}保护保留。三组均无非正总积分、未定义PSD或PSD保护触发。这里绘制的是无条件一维PSD分布，不是电荷—PSD联合/条件评价。252的完整wave改善1.76686%，原研究未通过2%入口；本次仅绘图，新增拟合、端点选择、模型生成与随机draw均为0。

- PSD_DISTRIBUTION.png / .svg / .pdf：主要分布及累计差。
- PSD_FULL_RANGE_CDF.png / .svg / .pdf：完整取值范围ECDF。
- PSD_CURVES.csv：主图的数值曲线。
- PSD_VALUES.npz：全体PSD、总/尾积分及原身份索引，可直接重绘。
- PLOT_RECEIPT.json：样本数、尾部数量、数值摘要、定义及输入哈希。
- plot_psd_distribution.py：绘图源码。
''',encoding='utf-8')
print(json.dumps({'out':str(out),'bandwidth':bandwidth,'summary':summary,'marginal_metrics':metrics},ensure_ascii=False,indent=2))
