from pathlib import Path
import os,json,hashlib
import numpy as np
w=Path(__file__).parent;r=w.parents[2];run=w/'collected/run_01';out=r/'reports/response060_20261018';out.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(w/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(p.read_text());load=lambda n:dict(np.load(run/(n+'.npz'),allow_pickle=False))
real=load('REAL_PROFILES');regions={'pre_peak':(0,16),'peak_neighborhood':(16,24),'falling_side':(24,46),'early_tail':(46,64),'late_tail':(64,256)}
result=dict(scope='Descriptive conditional contrasts, not paired SOURCE/REAL, not preADC truth. Input-Q groups are ordinal sensitivity, not numerically equivalent REAL energy bins.',
    covariance_scope='Full first64 covariance and16 nonoverlapping block integrals; pointwise moments all256.',groups=[],ordered=[],variance=[],peaks={})
for seed in range(2):
    label=f'seed{seed}_BASE054';profiles=[load(label+f'_fast_stage{i}_PROFILES') for i in range(5)]
    intrinsic=load(label+'_intrinsic_PROFILES');native=profiles[-1];stage=load(label+'_fast_STAGE_ROWS')
    result['peaks'][str(seed)]={str(int(k)):int(v) for k,v in zip(*np.unique(stage['selected_peak_index_by_carrier_stage'][0],return_counts=True))}
    for g in range(2):
        for view in range(2):
            for b in range(8):
                ref=real['mean'][0,view,b];src=native['mean'][g,view,b];mean=profiles[0]['mean'][g,view,b];d=src-ref
                # Same parity in the two populations denotes disjoint within-population halves, not event pairing.
                halves=native['parity_mean'][g,view,b]-real['parity_mean'][0,view,b]
                desc={}
                for name,(lo,hi) in regions.items():
                    v=d[lo:hi];h=halves[:,lo:hi];noise=(src-mean)[lo:hi]
                    desc[name]=dict(signed_integral=float(v.sum()),RMS=float(np.sqrt(np.mean(v*v))),
                        min=float(v.min()),max=float(v.max()),parity_signed_integrals=h.sum(1).tolist(),
                        half_RMS_difference=float(np.sqrt(np.mean((h[0]-h[1])**2))),
                        half_sign_agreement_fraction=float(np.mean(np.sign(h[0])==np.sign(h[1]))),
                        ordered_post_mean_RMS=float(np.sqrt(np.mean(noise**2))))
                covariance=native['covariance_first64'][g,view,b];rcov=real['covariance_first64'][0,view,b]
                result['groups'].append(dict(seed=seed,group=g,view=view,bin=b,source_n=int(native['counts'][g,b]),real_n=int(real['counts'][0,b]),
                    peak_difference=float(d[16]),source_peak=float(src[16]),real_peak=float(ref[16]),
                    full_RMS=float(np.sqrt(np.mean(d*d))),full_L1=float(np.abs(d).sum()),regions=desc,
                    first64_covariance_trace_ratio=float(np.trace(covariance)/np.trace(rcov)),
                    first64_covariance_relative_frobenius_difference=float(np.linalg.norm(covariance-rcov)/np.linalg.norm(rcov)),
                    intrinsic_vs_fixed_anchor_RMS=float(np.sqrt(np.mean((intrinsic['mean'][g,view,b]-mean)**2)))))
            weights=native['counts'][g]/6593;rm=real['mean'][0,view];rw=real['counts'][0]/21047
            within=np.einsum('b,bt->t',weights,native['SD_population'][g,view]**2)
            between=np.einsum('b,bt->t',weights,(native['mean'][g,view]-native['overall_mean'][view])**2)
            rwithin=np.einsum('b,bt->t',rw,real['SD_population'][0,view]**2)
            rbetween=np.einsum('b,bt->t',rw,(rm-real['overall_mean'][view])**2)
            result['variance'].append(dict(seed=seed,group=g,view=view,regions={name:dict(source_within=float(within[lo:hi].sum()),source_between=float(between[lo:hi].sum()),REAL_within=float(rwithin[lo:hi].sum()),REAL_between=float(rbetween[lo:hi].sum())) for name,(lo,hi) in regions.items()}))
    ordered=load(label+'_fast_ORDERED_WAVE')
    for g in range(2):
        for b in range(8):
            item=dict(seed=seed,group=g,bin=b,regions={})
            for name,(lo,hi) in regions.items():
                means=ordered['component_mean_V'][g,b,:,lo:hi];cov=ordered['component_covariance_V2'][g,b,:,:,lo:hi]
                item['regions'][name]=dict(component_signed_integrals=means.sum(1).tolist(),component_RMS=np.sqrt(np.mean(means**2,axis=1)).tolist(),
                    integrated_pointwise_covariance=cov.sum(2).tolist(),full_pointwise_variance_sum=float(ordered['final_variance_V2'][g,b,lo:hi].sum()))
            result['ordered'].append(item)
result['REAL_Q_centers_Vns']=real['mean'][0,0].sum(1).tolist()
result['REAL_normalized_peak_argmax']=real['mean'][0,1].argmax(1).tolist()
result['REAL_parity_normalized_peak_argmax']=real['parity_mean'][0,1].argmax(2).tolist()
(w/'CONDITIONAL_WAVE_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
native=load('seed0_BASE054_fast_stage4_PROFILES');mean=load('seed0_BASE054_fast_stage0_PROFILES')
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
t=np.arange(64)-16
for col,b in enumerate((0,3,7)):
    for row,view in enumerate((0,1)):
        ax=axes[row,col];scale=1000 if view==0 else 1
        ax.plot(t,scale*real['mean'][0,view,b,:64],color='#d55e00',label='REAL measured ADC')
        ax.plot(t,scale*native['mean'][1,view,b,:64],color='#0072b2',label='BASE054 native')
        ax.plot(t,scale*mean['mean'][1,view,b,:64],color='#009e73',ls='--',label='Deterministic mean at fixed anchor')
        for half in (0,1):
            ax.plot(t,scale*real['parity_mean'][0,view,b,half,:64],color='#d55e00',alpha=.25,lw=.7)
            ax.plot(t,scale*native['parity_mean'][1,view,b,half,:64],color='#0072b2',alpha=.25,lw=.7)
        ax.set(title=f'Output-Q bin {b+1}/8; SOURCE n={native["counts"][1,b]}',xlabel='Sample offset from selected peak (software ns)',ylabel='Mean voltage (mV)' if view==0 else 'Mean guarded unit-Q shape (1/ns)')
        ax.grid(alpha=.15)
axes[0,0].legend(fontsize=8);fig.suptitle('060 fixed BASE054 conditional waveform audit | seed0 / fast | same bins, unpaired populations',fontsize=13)
fig.savefig(out/'DIAGNOSTIC.png',dpi=160);plt.close(fig)
selected=[x for x in result['groups'] if x['seed']==0 and x['group']==1 and x['view']==0 and x['bin'] in (0,3,7)]
print(json.dumps(dict(peaks=result['peaks'],REAL_Q_centers=result['REAL_Q_centers_Vns'],REAL_peak=result['REAL_normalized_peak_argmax'],selected=selected),indent=2))
