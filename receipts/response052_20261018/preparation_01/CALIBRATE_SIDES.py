from pathlib import Path
import hashlib
import json
import time
import numpy as np

w=Path(__file__).parent;r=w.parents[2]
out=r/'receipts/response051_20261018/repair01_wake_20261018_1104_01/collected/run_01'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
load=lambda p:dict(np.load(p,allow_pickle=False))
started=time.monotonic();T=read(r/'receipts/response039_20261018/preparation_01/CALIBRATED_TOLERANCES_READONLY.json')['T_by_hypothesis']
real=load(out/'REAL_ROWS.npz');base=load(out/'seed0_BASE042_INTRINSIC_MEAN_ROWS.npz');q=base['A0_canonical_Q_Vns']
qref=read(r/'configs/response051_compact_20261018.json')['input_Q_reference_Vns'];logq=np.log(q/qref)
c=base['intrinsic_peak_index'];left=base['left_crossing50_ns'];right=base['right_crossing50_ns']
assert np.all((109.5<left)&(left<c)&(c<right)&(right<158.5))
hat=[]
for key in ('left_crossing10_ns','left_crossing90_ns'):
    x=base[key]
    hat.append(np.where((x>=109.5)&(x<=c),np.where(x<=left,(x-109.5)/(left-109.5),(c-x)/(c-left)),0.))
sensitivity=hat[1]-hat[0];sensitivity[~np.isfinite(base['rise10_90_ns'])]=np.nan
matrix=[];targets=[];groups=[]
for seed in (0,1):
    for h in ('fast','slow','common_level'):
        rows=load(out/f'seed{seed}_BASE042_{h}_ROWS.npz')
        for b in range(8):
            member=rows['BASE042_output_Q_bin']==b;rm=real['output_Q_bin']==b
            for feature in ('rise10_90_ns','FWHM_ns'):
                native=rows[feature+'_by_stage'][-1];mask=member&np.isfinite(native)&np.isfinite(base[feature]);rr=rm&np.isfinite(real[feature])
                target=float(real[feature][rr].mean()-native[mask].mean())
                if feature=='rise10_90_ns':
                    m=np.array([sensitivity[mask].mean(),(sensitivity[mask]*logq[mask]).mean(),0.,0.])
                else:
                    m=np.array([-1.,-logq[mask].mean(),1.,logq[mask].mean()])
                weight=np.sqrt(mask.sum()/6593)/T[h][feature]
                matrix.append(m*weight);targets.append(target*weight)
                groups.append(dict(seed=seed,hypothesis=h,bin=b,feature=feature,rows=int(member.sum()),defined_fit_rows=int(mask.sum()),REAL_rows=int(rm.sum()),REAL_defined_rows=int(rr.sum()),target_change_ns=target,derivative=m.tolist(),weight=float(weight)))
A=np.array(matrix);y=np.array(targets);theta,residuals,rank,singular=np.linalg.lstsq(A,y,rcond=None)
assert rank==4
dL=theta[0]+theta[1]*logq;dR=theta[2]+theta[3]*logq
knots=np.column_stack((np.full(len(q),109.5),left,c,right,np.full(len(q),158.5)))
shift=np.column_stack((np.zeros(len(q)),dL,np.zeros(len(q)),dR,np.zeros(len(q))))
interval=np.diff(knots,axis=1);ds=np.diff(shift,axis=1);negative=ds<0;limits=np.full(ds.shape,np.inf);limits[negative]=-interval[negative]/ds[negative]
bound=float(limits.min());limiting=np.unravel_index(np.argmin(limits),limits.shape);scale=min(1.,.9*bound)
for rec in groups:
    derivative=np.array(rec['derivative']);rec['full_direction_geometric_change_ns']=float(derivative@theta);rec['monotone90pct_direction_geometric_change_ns']=float(scale*derivative@theta)
result=dict(coefficient_order=['left_intercept_ns','left_logQ_slope_ns','right_intercept_ns','right_logQ_slope_ns'],coefficients_ns=theta.tolist(),rank=int(rank),singular_values=singular.tolist(),q_reference_Vns=qref,
    strict_positive_step_upper=bound,limiting_parent_row=int(limiting[0]),limiting_interval=int(limiting[1]),monotone_90pct_step=scale,
    total_scaled_target_squared=float(y@y),unconstrained_scaled_residual_squared=float(np.sum((A@theta-y)**2)),monotone_90pct_scaled_residual_squared=float(np.sum((scale*A@theta-y)**2)),
    source_count=6593,real_count=21047,all_groups=groups,CP_loads=0,raw_TRAIN_decodes=0,DEV_decodes=0,native_waveform_evaluations=0,elapsed_seconds=time.monotonic()-started,
    interpretation='Linear fixed-crossing design fit only, with frozen baseline conditional native residual targets; not actual waveform or response acceptance',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),preparation_contract_sha256=hashlib.sha256((r/'docs/RESPONSE052_PREPARATION_20261018.md').read_bytes()).hexdigest())
assert result['elapsed_seconds']<60
(w/'HALF_HEIGHT_SIDES_FIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='all_groups'}))
for g in groups:
    if g['seed']==0 and g['hypothesis']=='fast':print(g['bin'],g['feature'],g['target_change_ns'],g['full_direction_geometric_change_ns'],g['monotone90pct_direction_geometric_change_ns'])
