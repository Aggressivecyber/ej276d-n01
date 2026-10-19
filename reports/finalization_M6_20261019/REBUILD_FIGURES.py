"""Regenerate six figures from the included population aggregates only."""
import hashlib, json, os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

d=json.loads((ROOT/'data/FIGURE_DATA.json').read_text())
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
    'axes.titleweight':'bold','axes.labelsize':11,'legend.fontsize':10,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
colors={'REAL':'#171e26','M0':'#4474ab','BASE054':'#228983','070':'#d08a20','S':'#ba4148'}
cases=d['cases'];labels=[f'H{i}\n{h.replace("common_level","common")}' for i in (0,1) for h in d['hypotheses']]
manifest={}
def save(fig,name):
    for ext in ('png','svg'):
        p=OUT/(name+'.'+ext);fig.savefig(p,dpi=190)
        manifest[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    plt.close(fig)

# Figure 1: mathematical architecture and exact observation order.
fig,ax=plt.subplots(figsize=(13,7));ax.set(xlim=(0,13),ylim=(0,7));ax.axis('off')
boxes=[(0.2,4.8,2.1,1.3,'Prepared SOURCE\nx: 384 samples\nfloat64, voltage (V)', '#e7eef7'),
       (3.1,5.1,2.5,1.2,'Frozen A0 / BASE054\nmean m(x)', '#e1f1ef'),
       (3.1,2.9,2.5,1.2,'Frozen M0 residual\nr(x,z) = M0(x,z) − A0(x)', '#e7eef7'),
       (6.4,5.1,2.5,1.2,'Constrained mean QP\nm(x) + E G(x) zₘ\n24 spline columns', '#e1f1ef'),
       (6.4,2.9,2.5,1.2,'Linear random transport\nL(x) r(x,z)\nsame conserved space', '#e7eef7'),
       (9.7,3.9,2.7,1.5,'Sum + frozen electronics\n005 noise / extension H\nfast · slow · common', '#fcf0dc'),
       (8.8,.5,3.6,1.4,'Hard observation\n14-bit ADC → baseline subtraction\nfirst argmax → 256-point crop\nQ · front · tail · PSD · flags', '#f7e6e8')]
for x,y,w,h,t,c in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=.09',facecolor=c,edgecolor='#506073',lw=1.1))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11,linespacing=1.5)
for a,b in [((2.3,5.45),(3.05,5.7)),((2.3,5.05),(3.05,3.5)),((5.6,5.7),(6.35,5.7)),
            ((5.6,3.5),(6.35,3.5)),((8.9,5.7),(10.0,5.45)),((8.9,3.5),(10.0,3.85)),((11.1,3.85),(11.1,1.95))]:
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.6,'color':'#506073'})
ax.text(.3,1.7,'Frozen constraints',fontweight='bold',fontsize=13)
ax.text(.3,1.25,'Reference front / tail integrals • support • deterministic mean peak',fontsize=11)
ax.text(.3,.75,'Readout: peak p ∈ [112,144); Q = [p−16,p+240); tail = [p+30,p+240)',fontsize=11)
ax.text(.3,.25,'S is the evaluated research candidate. Its native protection gates failed.',fontsize=11,color='#922e37')
ax.set_title('1  Input-dependent calibration followed by the complete hard readout',loc='left',pad=16,fontsize=16)
save(fig,'FIG01_ARCHITECTURE')

# Figure 2: four executed variants, separate from the planned four-system trial.
fig,axs=plt.subplots(2,2,figsize=(13,8.7),layout='constrained');t=np.arange(256)-16
for case in ['REAL']+cases:
    axs[0,0].plot(t,np.array(d['profiles'][case]['overall_voltage_mean_V'])*1000,color=colors[case],label=case,lw=2 if case in ('REAL','S') else 1.5)
axs[0,0].set(xlabel='Sample time relative to selected peak (ns)',ylabel='Population mean voltage (mV)',title='A  Complete 256-point mean, H0 / fast')
axs[0,0].legend(ncol=3)
for case in ['REAL']+cases:axs[0,1].plot(d['PSD_CDF']['grid'],d['PSD_CDF']['curves'][case],color=colors[case],lw=1.8,label=case)
axs[0,1].set(xlabel='Native PSD = tail / Q',ylabel='Empirical CDF',title='B  PSD distribution, H0 / fast',ylim=(0,1.02))
x=np.arange(6)
for i,key in enumerate(('wave_relative_gain','joint_relative_gain')):
    axs[1,0].bar(x+(i-.5)*.36,[100*r[key] for r in d['S_point_usefulness']],.36,color=['#4474ab','#228983'][i],label=['Full-wave distance','Joint morphology'][i])
axs[1,0].axhline(2,color='#171e26',ls='--',lw=1,label='Point target: 2%')
axs[1,0].set(xticks=x,xticklabels=labels,ylabel='Relative improvement over M0 (%)',ylim=(0,70),title='C  S gains in all six frozen conditions');axs[1,0].legend(loc='upper left',fontsize=9)
for i,key in enumerate(('QPSD','tail_Vns','PSD_guarded')):
    axs[1,1].plot(x,[r['protected_delta_T'][key] for r in d['S_protection']],marker=['o','s','^'][i],color=['#ba4148','#d08a20','#6c5292'][i],label=['QPSD','Tail','PSD'][i])
axs[1,1].axhline(.05,color='#171e26',ls='--',lw=1,label='Allowed increase: 0.05 T')
axs[1,1].set(xticks=x,xticklabels=labels,ylabel='Distance increase over M0 / T',title='D  S readout protection failures');axs[1,1].legend(fontsize=9)
fig.suptitle('2  Executed calibration variants: morphology improves while readout protection fails',fontsize=15,fontweight='bold')
save(fig,'FIG02_EXECUTED_VARIANTS')

# Figure 3: signed algebraic terms and ADC sensitivity.
fig,axs=plt.subplots(2,2,figsize=(13,8.7),layout='constrained')
terms=['tail_BASE054_window_shift_Vns','tail_same_new_window_mean_Vns','tail_same_new_window_random_Vns']
for ax,rows,title in [(axs[0,0],d['decomposition071'],'A  070 − BASE054, verified by 071'),
                      (axs[0,1],d['decompositionS'],'B  S − BASE054')]:
    for j,(key,name,c) in enumerate(zip(terms,['Window relocation','Same-window mean','Same-window random'],['#ba4148','#d08a20','#228983'])):
        vals=[r['tail_terms_mean_Vns'][key] if 'tail_terms_mean_Vns' in r else r['groups']['all']['tail_terms_mean_Vns'][key] for r in rows]
        ax.bar(x+(j-1)*.24,np.array(vals)*1000,.24,color=c,label=name)
    ax.axhline(0,color='black',lw=.7);ax.set(xticks=x,xticklabels=labels,ylabel='Mean signed tail term (mV ns)',title=title)
axs[0,0].legend(fontsize=9,loc='upper left')
for case in ('BASE054','S'):
    axs[1,0].plot(np.array(d['gap_CDF']['grid_V'])*1000,d['gap_CDF']['curves'][case],color=colors[case],lw=2,label=case)
axs[1,0].axvline(1000/16383,color='#171e26',ls='--',label='One ADC code: 0.0610 mV')
axs[1,0].set(xscale='log',xlabel='Top-two deterministic mean peak gap (mV)',ylabel='Fraction of 6,593 SOURCE parents',ylim=(0,1.02),title='C  Near-tied means before stochastic additions');axs[1,0].legend(fontsize=9,loc='upper left')
m=d['mean_margin'][:2];rate=[r['ADC_alone_reselected_rows']/r['rows']*100 for r in m]
bars=axs[1,1].bar(['BASE054','S'],rate,color=[colors['BASE054'],colors['S']],width=.5)
for bar,row in zip(bars,m):axs[1,1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{row['ADC_alone_reselected_rows']:,} / {row['rows']:,}",ha='center')
axs[1,1].set(ylabel='Parents whose selected peak changes (%)',ylim=(0,82),title='D  ADC alone on fixed deterministic means')
fig.suptitle('3  Peak competition connects conserved reference areas to moving native windows',fontsize=15,fontweight='bold')
save(fig,'FIG03_READOUT_MECHANISM')

# Figure 4: existing conditional means, with every finite-value denominator saved.
fig,axs=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
fields=[('peak_voltage_V',1000,'Mean peak voltage (mV)'),('rise10_90_ns',1,'Mean 10–90% rise (ns)'),('FWHM_ns',1,'Mean FWHM (ns)'),('PSD_guarded',1,'Mean native PSD')]
ref=[r for r in d['conditional'] if r['head']==0 and r['H']=='fast' and r['case']=='S']
qx=[r['real_Q_Vns_median'] for r in ref]
for panel,(ax,(key,mult,yl)) in enumerate(zip(axs.flat,fields)):
    ax.plot(qx,[r['real_'+key+'_mean']*mult for r in ref],'-o',color=colors['REAL'],label='REAL',lw=2)
    for case in cases:
        z=[r for r in d['conditional'] if r['head']==0 and r['H']=='fast' and r['case']==case]
        ax.plot(qx,[r['fake_'+key+'_mean']*mult for r in z],'-o',color=colors[case],label=case,lw=1.5,ms=4)
    all_s=np.array([[r['fake_'+key+'_mean']*mult for r in d['conditional'] if r['head']==head and r['H']==h and r['case']=='S'] for head in (0,1) for h in d['hypotheses']])
    ax.fill_between(qx,all_s.min(0),all_s.max(0),color=colors['S'],alpha=.17,label='S range over 2 heads × 3 H')
    ax.set(xlabel='REAL median Q in the shared output-Q bin (V ns)',ylabel=yl,title=chr(65+panel)+'  '+yl.replace('Mean ',''))
axs[0,0].legend(fontsize=9,ncol=2)
fig.suptitle('4  Conditional morphology: common output-Q bins, complete original populations',fontsize=15,fontweight='bold')
save(fig,'FIG04_CONDITIONAL_MORPHOLOGY')

# Figure 5: descriptive population covariance and separate software checks.
fig=plt.figure(figsize=(13,8.7),layout='constrained');gs=fig.add_gridspec(2,3,height_ratios=[1.05,.9])
covs=[np.array(d['profiles'][k]['own_Q_bin3_covariance_first64_V2'])*1e6 for k in ('REAL','M0','S')]
limit=max(float(np.abs(c).max()) for c in covs)
axs=[fig.add_subplot(gs[0,i]) for i in range(3)]
for ax,case,cov in zip(axs,('REAL','M0','S'),covs):
    im=ax.imshow(cov,origin='lower',extent=(-16,48,-16,48),vmin=-limit,vmax=limit,cmap='RdBu_r',aspect='equal')
    n=d['profiles'][case]['own_Q_counts'][3]
    ax.set(title=f'{case}: output-Q bin 4/8 (n={n:,})',xlabel='Native time (ns)',ylabel='Native time (ns)')
fig.colorbar(im,ax=axs,shrink=.8,label='Population covariance (mV²)')
ax=fig.add_subplot(gs[1,:2]);ax.axis('off')
ax.text(.01,.92,'Variance across inputs and repeated responses',fontsize=13,fontweight='bold',transform=ax.transAxes)
ax.text(.01,.68,r'$\mathrm{Var}(f(Y))=\mathrm{Var}_x(\mathrm{E}[f(Y)\mid x])+\mathrm{E}_x[\mathrm{Var}(f(Y)\mid x)]$',fontsize=16,transform=ax.transAxes)
ax.text(.01,.42,'The heatmaps summarize different events within one output-Q bin.\nThe planned 128-parent × 64-draw kernel study has not run.\nS has no new frozen-endpoint C2ST or independent MC confirmation.',fontsize=12,linespacing=1.65,transform=ax.transAxes)
ax=fig.add_subplot(gs[1,2]);names=['028\ngolden','S\ngolden','Replay /\nID / draw'];values=[396,96,31]
ax.bar(names,values,color=['#4474ab','#228983','#d08a20'])
for i,v in enumerate(values):ax.text(i,v+10,str(v),ha='center')
ax.set(ylim=(0,450),ylabel='Byte-exact tensor checks',title='API engineering verification')
fig.suptitle('5  Population covariance and software repeatability are separately documented',fontsize=15,fontweight='bold')
save(fig,'FIG05_COVARIANCE_AND_REPLAY')

# Figure 6: explicit evidence map; unrun studies carry no invented score.
fig,ax=plt.subplots(figsize=(13,8));ax.axis('off')
rows=[['Native artifact + metric replay','VERIFIED','24 complete reports; 158,232 saved rows'],
      ['API / CLI compatibility','VERIFIED','523 byte-exact checks; 1,094 output rows'],
      ['S deterministic constraints','PASSED','13,186 QPs; reference support / area / mean peak'],
      ['S full native protection','FAILED','QPSD, front, tail, PSD and negative-lobe distance'],
      ['New endpoint MC confirmation','NOT TRIGGERED','No eligible endpoint after the S FIT result'],
      ['P0 / P1 / CPU-Net / N01 fair comparison','NOT RUN','Training and comparison outcomes unavailable'],
      ['Three-fold refit + input interventions','NOT RUN','No full-pipeline internal-validation estimate'],
      ['Same-parent kernel + new C2ST','NOT RUN','No new stochastic-kernel acceptance claim'],
      ['Alternate tail 41 / 51 use case','NOT RUN','No alternate-window transfer result']]
table=ax.table(cellText=rows,colLabels=['Evidence item','Observed status','Scope'],cellLoc='left',colLoc='left',
    colWidths=[.34,.16,.50],bbox=[0,.12,1,.80])
table.auto_set_font_size(False);table.set_fontsize(11)
for (row,col),cell in table.get_celld().items():
    cell.set_edgecolor('#d5dce4');cell.PAD=.07
    if row==0:cell.set_facecolor('#20354e');cell.set_text_props(color='white',fontweight='bold')
    else:
        cell.set_facecolor('#f2f5f8' if row%2 else 'white')
        if col==1:
            color='#1d7564' if rows[row-1][1] in ('VERIFIED','PASSED') else '#ad3542' if rows[row-1][1]=='FAILED' else '#5b6472'
            cell.set_text_props(color=color,fontweight='bold')
ax.text(0,.025,'Current scientific status: response_match_not_established.  T denotes a research perturbation scale.',fontsize=12,transform=ax.transAxes)
ax.set_title('6  Evidence available for the present method report',loc='left',pad=20,fontsize=16)
save(fig,'FIG06_EVIDENCE_MAP')
(OUT/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'figures':6,'files':12,'bytes':sum(m['bytes'] for m in manifest.values())}))
