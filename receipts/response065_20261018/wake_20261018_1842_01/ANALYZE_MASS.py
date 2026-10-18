from pathlib import Path
import json,os
import numpy as np
w=Path(__file__).parent;r=w.parents[2];run=w/'collected/run_01';out=r/'reports/response065_20261018';out.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(w/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(p.read_text());load=lambda n:dict(np.load(run/(n+'.npz'),allow_pickle=False))
real=load('REAL_ROWS');rp=load('REAL_PROFILES');res=read(run/'RESULT.json');regions={'pre_peak':(0,16),'peak_neighborhood':(16,24),'falling':(24,46),'early_tail':(46,64),'late_tail':(64,256)}
a=dict(scope='Fixed input-Q and BASE-output-Q paired SOURCE contrasts; own-output-Q SOURCE/REAL distributions are unpaired; Q is not energy truth.',profiles=[],features=[],native_migrations=[],geometry=[],metrics=[],REAL_features=[])
fields=['Q_Vns','front_Vns','tail_Vns','peak_voltage_V','rise10_90_ns','FWHM_ns']
for field in fields:
    x=real[field];mask=np.isfinite(x);x=x[mask];groups=real['output_Q_bin'][mask];mean=x.mean();means=np.array([x[groups==b].mean() for b in range(8)]);var=np.array([x[groups==b].var() for b in range(8)]);weights=np.array([np.mean(groups==b) for b in range(8)])
    a['REAL_features'].append(dict(field=field,mean=float(mean),variance=float(x.var()),within=float(weights@var),between=float(weights@((means-mean)**2)),bin_means=means.tolist(),bin_variances=var.tolist()))
for seed in range(2):
    m0=read(run/f'seed{seed}_M0.json');base=read(run/f'seed{seed}_BASE054.json');mix=read(run/f'seed{seed}_MASS.json')
    for h in ('fast','slow','common_level'):
        a['metrics'].append(dict(seed=seed,hypothesis=h,M0=m0[h]['values'],BASE=base['metrics'][h]['values'],MIX=mix['metrics'][h]['values'],usefulness=mix['point_usefulness'][h],violations=[x for x in mix['decision']['violations'] if x.startswith(h+'/')]))
        br=load(f'seed{seed}_BASE054_{h}_NATIVE_ROWS');mr=load(f'seed{seed}_MASS_{h}_NATIVE_ROWS');shift=mr['native_peak_index']-br['native_peak_index']
        a['native_migrations'].append(dict(seed=seed,hypothesis=h,peak_shift_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(shift,return_counts=True))},Q_bin_moved=int(np.sum(mr['native_output_Q_bin']!=br['native_output_Q_bin'])),tail_delta_by_peak_shift={str(int(k)):dict(rows=int(np.sum(shift==k)),mean=float((mr['tail_Vns']-br['tail_Vns'])[shift==k].mean())) for k in np.unique(shift)}))
    for case,record in [('BASE054',base),('MASS',mix)]:
        geo=load(f'seed{seed}_{case}_GEOMETRY_ROWS');a['geometry'].append(dict(seed=seed,case=case,peak_changed=int(np.sum(geo['old_mean_peak_index']!=geo['new_mean_peak_index'])),edge110=dict(mean=float(geo['boundary_delta_110_V'].mean()),max_absolute=float(np.abs(geo['boundary_delta_110_V']).max())),edge175=dict(mean=float(geo['boundary_delta_175_V'].mean()),max_absolute=float(np.abs(geo['boundary_delta_175_V']).max()))))
        for kind in ('fast','intrinsic'):
            p=load(f'seed{seed}_{case}_{kind}_PROFILES');pr=load(f'seed{seed}_BASE054_{kind}_PROFILES')
            for group in (0,1):
                for view in (0,1):
                    for b in range(8):
                        d=p['mean'][group,view,b]-rp['mean'][0,view,b];c=p['covariance_first64'][group,view,b];c0=pr['covariance_first64'][group,view,b];cr=rp['covariance_first64'][0,view,b]
                        halves=p['parity_mean'][group,view,b]-rp['parity_mean'][0,view,b]
                        a['profiles'].append(dict(seed=seed,case=case,kind=kind,group=group,view=view,bin=b,rows=int(p['counts'][group,b]),full_residual_RMS=float(np.sqrt(np.mean(d*d))),peak=float(p['mean'][group,view,b,16]),covariance_trace_SOURCE_over_REAL=float(np.trace(c)/np.trace(cr)),covariance_trace_SOURCE_over_BASE=float(np.trace(c)/np.trace(c0)),covariance_relative_frobenius=float(np.linalg.norm(c-cr)/np.linalg.norm(cr)),regions={name:dict(residual_integral=float(d[lo:hi].sum()),RMS=float(np.sqrt(np.mean(d[lo:hi]**2))),parity_integrals=halves[:,lo:hi].sum(1).tolist()) for name,(lo,hi) in regions.items()}))
        for kind in ('fast','slow','common_level','intrinsic'):
            rows=load(f'seed{seed}_{case}_'+('INTRINSIC_MEAN_ROWS' if kind=='intrinsic' else kind+'_NATIVE_ROWS'))
            own=load(f'seed{seed}_BASE054_fast_NATIVE_ROWS')['native_output_Q_bin'] if kind=='intrinsic' else rows['native_output_Q_bin']
            fixed=load(f'seed{seed}_BASE054_'+('fast' if kind=='intrinsic' else kind)+'_NATIVE_ROWS')['native_output_Q_bin']
            for group,bins in [('input',rows['input_Q_bin']),('fixed_BASE',fixed),('own_output',own)]:
                if kind=='intrinsic' and group=='own_output':continue
                for field in fields:
                    valid=np.isfinite(rows[field]);x=rows[field][valid];g=bins[valid];mean=x.mean();means=np.array([x[g==b].mean() for b in range(8)]);variance=np.array([x[g==b].var() for b in range(8)]);weights=np.array([np.mean(g==b) for b in range(8)])
                    within=float(weights@variance);between=float(weights@((means-mean)**2));assert abs(within+between-x.var())<1e-10
                    a['features'].append(dict(seed=seed,case=case,kind=kind,group=group,field=field,defined_rows=len(x),mean=float(mean),variance=float(x.var()),within=within,between=between,bin_means=means.tolist(),bin_variances=variance.tolist()))
(w/'MASS_ANALYSIS.json').write_text(json.dumps(a,indent=2)+'\n')
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True);t=np.arange(64)-16
for col,b in enumerate((0,3,7)):
    ax=axes[0,col]
    for name,profile,c in [('REAL',rp,'#d55e00'),('BASE054',load('seed0_BASE054_fast_PROFILES'),'#0072b2'),('MASS',load('seed0_MASS_fast_PROFILES'),'#009e73')]:
        g=0 if name=='REAL' else 1;ax.plot(t,profile['mean'][g,0,b,:64]*1000,label=name,color=c)
    ax.set(title=f'Output Q bin {b+1}/8',xlabel='Sample offset from peak (software ns)',ylabel='Mean voltage (mV)');ax.grid(alpha=.15)
axes[0,0].legend()
ax=axes[1,0];x=np.arange(1,9);rr=next(z for z in a['REAL_features'] if z['field']=='FWHM_ns');ax.plot(x,rr['bin_means'],'o-',color='#d55e00',label='REAL')
for case,c in [('BASE054','#0072b2'),('MASS','#009e73')]:
    z=next(z for z in a['features'] if z['seed']==0 and z['case']==case and z['kind']=='fast' and z['group']=='fixed_BASE' and z['field']=='FWHM_ns');ax.plot(x,z['bin_means'],'o-',label=case,color=c)
ax.set(xlabel='Q bin (SOURCE fixed BASE membership)',ylabel='Mean FWHM (software ns)',title='Conditional width relation');ax.legend(fontsize=8);ax.grid(alpha=.15)
ax=axes[1,1]
for case,c in [('BASE054','#0072b2'),('MASS','#009e73')]:
    z=[z for z in a['profiles'] if z['seed']==0 and z['case']==case and z['kind']=='fast' and z['group']==1 and z['view']==0];ax.plot(x,[z['covariance_trace_SOURCE_over_REAL'] for z in z],'o-',label=case,color=c)
ax.axhline(1,color='black',ls='--',lw=1);ax.set(xlabel='Own output Q bin',ylabel='First64 covariance trace / REAL',title='Within-bin waveform variation');ax.grid(alpha=.15)
ax=axes[1,2]
for case,c in [('BASE054','#0072b2'),('MASS','#009e73')]:
    z=next(z for z in a['features'] if z['seed']==0 and z['case']==case and z['kind']=='fast' and z['group']=='fixed_BASE' and z['field']=='FWHM_ns');ax.bar(np.arange(3)+(-.22 if case=='BASE054' else 0),[z['variance'],z['within'],z['between']],width=.22,label=case,color=c)
ax.bar(np.arange(3)+.22,[rr['variance'],rr['within'],rr['between']],width=.22,label='REAL',color='#d55e00');ax.set_xticks(np.arange(3),['Total','Within Q','Between Q']);ax.set(ylabel='FWHM variance (software ns squared)',title='Width variance decomposition');ax.legend(fontsize=8)
fig.suptitle('065 conditional mass transport | seed0 / fast | all original gates retained; protected QPSD and FWHM censor gates failed',fontsize=13);fig.savefig(out/'DIAGNOSTIC.png',dpi=160);plt.close(fig)
selected=[z for z in a['features'] if z['seed']==0 and z['kind']=='fast' and z['group']=='fixed_BASE' and z['field'] in ('FWHM_ns','peak_voltage_V','rise10_90_ns')]
print(json.dumps(dict(REAL_width=rr,selected=[x for x in selected if x['field']=='FWHM_ns'],geometry=a['geometry']),indent=2))
