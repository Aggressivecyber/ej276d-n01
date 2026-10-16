"""Plot saved032 aggregates only; no torch, checkpoints or original data."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1]/'receipts/response032_20261017/completion_01/run_01'
os.environ['MPLCONFIGDIR'] = str(ROOT/'mpl_cache')
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False,
    'font.size':11, 'axes.titlesize':14, 'axes.labelsize':12, 'figure.titlesize':18,
    'legend.fontsize':10, 'axes.spines.top':False, 'axes.spines.right':False,
    'axes.edgecolor':'#9FAAB4', 'xtick.color':'#35424E', 'ytick.color':'#35424E',
    'figure.facecolor':'white', 'axes.facecolor':'white', 'savefig.facecolor':'white'})

HYPOTHESES = ['fast','slow','common_level']
COLORS = {'A':'#2878A6','G0':'#D99B35','final':'#C94747'}
LABELS = {'A':'A：原确定性响应 + 005','G0':'G0：新模型初态','final':'final：2000 次更新'}
read_paths = set()


def read(name):
    path = SOURCE/name
    read_paths.add(path)
    return json.loads(path.read_text(encoding='utf8'))


def save(fig, name):
    fig.savefig(ROOT/name, dpi=180, bbox_inches='tight', pad_inches=.18)
    plt.close(fig)


result = read('RESULT.json')
records = {(split,ep,h):read(f'{split}_{ep}_{h}.json')
    for split,endpoints in [('TRAIN',['A','G0','final']),('DEV',['A','final'])]
    for ep in endpoints for h in HYPOTHESES}
key_numbers = {}
for split,endpoints in [('TRAIN',['A','G0','final']),('DEV',['A','final'])]:
    fig,axes = plt.subplots(1,2,figsize=(12.4,5.1))
    fig.subplots_adjust(top=.73,bottom=.22,wspace=.25)
    title = ('032 TRAIN：final 的两项距离均高于 A' if split == 'TRAIN'
             else '032 DEV：已暴露开发集，final 的两项距离均高于 A')
    fig.suptitle(title,y=.98,fontweight='bold',color='#223746')
    count_text = ('6593 个 SOURCE 父事件 × 4 次生成；REAL TRAIN 21047 条'
                  if split == 'TRAIN' else '512 个 SOURCE 父事件 × 4 次生成；REAL DEV 599 条')
    fig.text(.5,.88,count_text,ha='center',fontsize=12)
    x = np.arange(3)
    width = .23 if split == 'TRAIN' else .30
    for ax,metric,title_metric in zip(axes,('wave','QPSD'),('Native waveform 距离','Native Q/PSD 距离')):
        table = {ep:[records[(split,ep,h)]['values'][metric] for h in HYPOTHESES] for ep in endpoints}
        key_numbers[split+'_'+metric] = table
        thresholds = [result['comparisons'][split][h]['calibrated_components'][metric]['threshold'] for h in HYPOTHESES]
        key_numbers[split+'_'+metric+'_T'] = thresholds
        for index,ep in enumerate(endpoints):
            positions = x+(index-(len(endpoints)-1)/2)*width
            bars = ax.bar(positions,table[ep],width*.90,color=COLORS[ep],label=LABELS[ep],zorder=3)
            ax.bar_label(bars,labels=[f'{value:.3f}' for value in table[ep]],padding=5,fontsize=11)
        for pos,threshold in zip(x,thresholds):
            ax.plot([pos-.42,pos+.42],[threshold,threshold],color='#4D5259',ls='--',lw=1.4,zorder=4)
        ax.set(xticks=x,xticklabels=HYPOTHESES,ylabel='距离（越低越好）',title=title_metric,ylim=(0,.90))
        ax.tick_params(axis='x',labelsize=11)
        ax.grid(axis='y',color='#E2E8EC',lw=.7,zorder=0)
        ax.set_axisbelow(True)
    handles,labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0],[0],color='#4D5259',ls='--',lw=1.4));labels.append('冻结对照容差 T')
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=len(handles),frameon=False)
    fig.text(.5,.015,'四次生成共享父身份；图中无独立重复误差条。DEV 为既有开发证据。',ha='center',fontsize=10,color='#35424E')
    save(fig,f'{split.lower()}_native_distances.png')

metric_path = SOURCE/'metrics.jsonl'
read_paths.add(metric_path)
training = [json.loads(line) for line in metric_path.read_text(encoding='utf8').splitlines()]
steps = np.array([row['step'] for row in training])
data = np.array([row['data'] for row in training])
regular = np.array([row['regular'] for row in training])
gradient = np.array([row['preclip_gradient_norm'] for row in training])
clipped = gradient > 5
fig,axes = plt.subplots(1,3,figsize=(14,4.8))
fig.subplots_adjust(top=.77,bottom=.20,wspace=.30)
fig.suptitle('032 训练轨迹：优化记录不替代响应验收',y=.98,fontweight='bold',color='#223746')
fig.text(.5,.885,'固定 pilot 未通过开发验收；2000 次更新，单一训练 seed',ha='center',fontsize=12,color='#A32F37')
for ax,values,color,title,ylabel in zip(axes,(data,regular,gradient),('#2878A6','#7953A1','#2878A6'),
        ('训练 data：joint energy','原始正则项 R（权重 0.01）','裁剪前梯度范数'),('data','R','梯度范数')):
    ax.plot(steps,values,color=color,alpha=.19,lw=.65,label='逐步记录')
    ax.plot(steps[49:],np.convolve(values,np.ones(50)/50,mode='valid'),color=color,lw=1.8,label='50 步滑动均值')
    ax.set(xlabel='已完成更新次数',ylabel=ylabel,title=title,xlim=(0,2000))
    ax.grid(axis='y',color='#E2E8EC',lw=.65)
axes[0].axhline(0,color='#77838C',lw=.8,ls=':')
axes[0].legend(loc='upper right',frameon=False)
axes[1].text(.98,.96,f'末步 R = {regular[-1]:.3f}',transform=axes[1].transAxes,ha='right',va='top',fontsize=11)
axes[2].set_yscale('log')
axes[2].axhline(5,color='#C94747',ls='--',lw=1.3,label='clip 阈值 5')
axes[2].text(.98,.96,f'触发 clip：{clipped.mean():.1%}\n末 100 步：{clipped[-100:].mean():.0%}',
             transform=axes[2].transAxes,ha='right',va='top',fontsize=11)
axes[2].legend(loc='lower left',frameon=False)
fig.text(.5,.015,'data 保留原有负估计值；R 未加权展示。轨迹来自已保存 metrics.jsonl，无新增训练或梯度查询。',ha='center',fontsize=10,color='#35424E')
save(fig,'training_traces.png')

population = {}
for endpoint in ('A','G0','final'):
    path = SOURCE/f'TRAIN_{endpoint}_fast_examples.npz'
    read_paths.add(path)
    with np.load(path,allow_pickle=False) as archive:
        population[endpoint] = (archive['population_mean_V'].copy(),archive['population_std_V'].copy())
fig,axes = plt.subplots(1,2,figsize=(12.4,5.3))
fig.subplots_adjust(top=.75,bottom=.28,wspace=.25)
fig.suptitle('TRAIN / fast：保存的生成总体均值与标准差',y=.98,fontweight='bold',color='#223746')
fig.text(.5,.88,'仅生成总体，未绘制 REAL 均值；不能据此判断 SOURCE → REAL 匹配',ha='center',fontsize=12,color='#A32F37')
time = np.arange(256)-16
for endpoint,(mean,std) in population.items():
    axes[0].plot(time,mean*1000,color=COLORS[endpoint],lw=1.8,label=LABELS[endpoint])
    axes[1].plot(time,std*1000,color=COLORS[endpoint],lw=1.8,label=LABELS[endpoint])
for ax,title,ylabel in zip(axes,('生成总体均值','生成总体标准差'),('电压均值（mV）','电压标准差（mV）')):
    ax.axvline(0,color='#A4AFB6',ls=':',lw=.9)
    ax.set(title=title,ylabel=ylabel,xlabel='相对选定峰的样点（软件 ns）',xlim=(-16,239))
    ax.grid(color='#E2E8EC',lw=.65)
handles,labels = axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False)
fig.text(.5,.015,'6593 父事件 × 4 次生成 = 26372 条；总体标准差含父事件差异与随机响应，不是均值不确定度。',ha='center',fontsize=10,color='#35424E')
save(fig,'train_fast_generated_population.png')

figures = []
for name in ('train_native_distances.png','dev_native_distances.png','training_traces.png','train_fast_generated_population.png'):
    path = ROOT/name
    with Image.open(path) as image:
        size = image.size
    figures.append(dict(path=str(path),bytes=path.stat().st_size,pixels=list(size),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
receipt = dict(schema='n01/response032-saved-output-plots/v1',status='rendered_pending_visual_QA',
    source_root=str(SOURCE),generator=str(Path(__file__)),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    figures=figures,inputs=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(read_paths)],
    key_numbers=key_numbers,training_summary=dict(steps=len(training),data_first=float(data[0]),data_last=float(data[-1]),
        data_last100_mean=float(data[-100:].mean()),regular_first=float(regular[0]),regular_last=float(regular[-1]),
        preclip_gradient_first=float(gradient[0]),preclip_gradient_last=float(gradient[-1]),clipped_steps=int(clipped.sum()),
        clipped_rate=float(clipped.mean()),last100_clipped_steps=int(clipped[-100:].sum())),
    saved_population_NPZ_members_decoded=['population_mean_V','population_std_V'],saved_population_archives=3,
    saved_event_waveform_members_decoded=0,raw_SOURCE_REAL_reads=0,checkpoint_reads=0,new_NN_calls=0,new_AD_queries=0,new_optimizer_steps=0,
    evidence_limits=dict(pilot_relative_progress=result['pilot_relative_progress'],scientific_status=result['scientific_status'],
        DEV_previously_exposed=True,independent_acquisition_acceptance=False,
        four_draws_are_not_independent_parents=True,population_wave_plot_has_no_REAL_mean=True),
    visual_QA='pending_view_image')
(ROOT/'PLOT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(figures=figures,training_summary=receipt['training_summary']),indent=2))
