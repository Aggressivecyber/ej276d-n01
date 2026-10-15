"""Stream saved actual A/encoded windows and repeated moments; NumPy only."""
from datetime import datetime,timezone
import gc,hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path('E:/GAN/research/N01');OUT=ROOT/'receipts/wp1_013_20261015/postrun_independent_01'
JOB=ROOT/'runs/wp1_013_20261015/runtime_fix01_collected_03/runs/primary';ART=JOB/'artifacts';EVAL=JOB/'evaluation'
P=6593;R=4;N=P*R;H=('fast','slow','common_level')
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[];maxima={};summary={}
def ok(v,name):assert v,name;checks.append(dict(name=name,passed=True))
def equal(a,b,name,*,atol=2e-13,rtol=2e-11):
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape,(name,a.shape,b.shape)
    assert np.array_equal(np.isnan(a),np.isnan(b)),(name,'NaN masks')
    mask=np.isfinite(a)&np.isfinite(b)
    error=float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0.
    assert np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=True),(name,error)
    maxima[name]=error
def data(z,entry):
    assert isinstance(entry,dict) and 'array' in entry
    a=z[entry['array']];assert list(a.shape)==entry['shape'] and str(a.dtype)==entry['dtype']
    return a
def schema_refs(doc):return np.load(EVAL/doc['file'],allow_pickle=False)
def residual(x,y=None):
    m=sq=0.;nz=nr=nb=0;affected=[]
    for st in range(0,P,128):
        en=min(P,st+128);a=x[st*R:en*R].reshape(en-st,R,384)
        b=a[:,:1] if y is None else y[st*R:en*R].reshape(en-st,R,384)
        d=a-b;mask=d!=0;m=max(m,float(np.max(np.abs(d))));sq+=float(np.sum(d*d,dtype=np.float64))
        nz+=int(mask.sum());nr+=int(mask.any(2).sum());nb+=int((a.view(np.uint64)!=b.view(np.uint64)).sum())
        affected.extend((np.flatnonzero(mask.any((1,2)))+st).tolist())
    return dict(actual_max_abs_residual_V=m,actual_RMS_residual_V=math.sqrt(sq/x.size),nonzero_cells=nz,
        nonzero_rows=nr,nonzero_parents=len(affected),different_float64_bit_cells=nb,affected_source_parent_rows=affected)
def scalar_moments(x):
    valid=np.isfinite(x);count=valid.sum(1);s=np.where(valid,x,0.).sum(1)
    mean=np.full_like(s,np.nan);np.divide(s,count,out=mean,where=count>0)
    ss=np.where(valid,(x-mean[:,None])**2,0.).sum(1)
    var=np.full_like(s,np.nan);np.divide(ss,count-1,out=var,where=count>1)
    se=np.sqrt(var/np.maximum(count,1))
    p,_,k=x.shape;cov=np.full((p,k,k),np.nan);joint=np.zeros((p,k,k),np.int64)
    for i in range(k):
        for j in range(i,k):
            pair=valid[:,:,i]&valid[:,:,j];n=pair.sum(1)
            mi=np.zeros(p);mj=np.zeros(p)
            np.divide(np.where(pair,x[:,:,i],0.).sum(1),n,out=mi,where=n>0)
            np.divide(np.where(pair,x[:,:,j],0.).sum(1),n,out=mj,where=n>0)
            q=np.where(pair,(x[:,:,i]-mi[:,None])*(x[:,:,j]-mj[:,None]),0.).sum(1)
            v=np.full(p,np.nan);np.divide(q,n-1,out=v,where=n>1)
            cov[:,i,j]=cov[:,j,i]=v;joint[:,i,j]=joint[:,j,i]=n
    return dict(mean=mean,sample_variance=var,MonteCarlo_SE=se,defined_draws=count,undefined_draws=x.shape[1]-count,
        sample_covariance=cov,jointly_defined_draws=joint)

refs=[np.load(ART/f'REFERENCE_{b}.npz',allow_pickle=False) for b in (0,1)]
ids=refs[0]['source_parent_ids'];ok(len(ids)==len(np.unique(ids))==P and np.array_equal(ids,refs[1]['source_parent_ids']),'same6593_unique_TRAIN_source_ids_two_banks')
for b,z in enumerate(refs):
    assert np.array_equal(z['source_parent_row'],np.repeat(np.arange(P),R))
    assert np.array_equal(z['draw_index'],np.tile(np.arange(R),P))
    with np.load(ART/f'BANK_{b}.npz',allow_pickle=False) as bank:assert np.array_equal(bank['source_parent_rows'],z['source_parent_row'])
ok(True,'exact_parent_major4draw_reference_and_BANK_lineage')
encoding=read(ART/'REFERENCE_ENCODING.json')
ok(encoding['A_actual_float64_per_draw_rows_preserved'] and not encoding['A_signal_broadcast_across_banks_and_draws'],'encoding_metadata_actualA_not_broadcast')
pure_summary={}
for arm,key in (('A','A_signal_pre384'),('theta','theta_signal_pre384')):
    arrays=[z[key] for z in refs]
    assert all(a.shape==(N,384) and a.dtype==np.float64 and np.isfinite(a).all() and not a[:,:100].any() for a in arrays)
    if arm=='A':
        for b,a in enumerate(arrays):
            actual=residual(a);saved=read(ART/f'A_NUMERIC_REPEATS_{b}.json')
            for k,v in actual.items():
                if isinstance(v,list):assert v==saved[k]
                elif isinstance(v,int):assert v==saved[k]
                else:assert math.isclose(v,saved[k],rel_tol=1e-12,abs_tol=1e-25)
            assert 0<actual['actual_max_abs_residual_V']<1e-7 and actual['affected_source_parent_rows']==[6592]
            summary[f'A_repeat_bank{b}']=actual
        actual=residual(arrays[1],arrays[0]);saved=read(ART/'A_NUMERIC_CROSS_BANK.json')
        assert all(actual[k]==saved[k] for k in actual)
        assert np.array_equal(arrays[0].view(np.uint64),arrays[1].view(np.uint64))
        summary['A_crossbank']=actual
    popvar=[]
    for b,a in enumerate(arrays):
        q=a.reshape(P,R,384);mu=q.mean(1);var=np.sum((q-mu[:,None])**2,axis=1)/3
        with np.load(ART/f'SIGNAL_MOMENTS_{b}_{arm}.npz',allow_pickle=False) as saved:
            equal(mu,saved['mean_V'],f'{arm}_pure_mean_bank{b}',atol=1e-15)
            equal(var,saved['sample_variance_V2'],f'{arm}_pure_var_bank{b}',atol=1e-30,rtol=1e-10)
            assert np.array_equal(saved['source_parent_row'],np.arange(P))
        popvar.append(float(var.mean())*.75)
    doc=read(EVAL/f'REPEATS_8_{arm}_fast.json')
    with np.load(EVAL/f'REPEATS_8_{arm}_fast.npz',allow_pickle=False) as saved:
        emu=data(saved,doc['preADC_signal_mean_V']);ev=data(saved,doc['preADC_signal_covariance_diagonal_V2'])
        for st in range(0,P,128):
            en=min(P,st+128);x=np.concatenate([a.reshape(P,R,384)[st:en] for a in arrays],axis=1)
            mu=x.mean(1);var=((x-mu[:,None])**2).sum(1)/7
            equal(mu,emu[st:en],f'{arm}_pure8_mean_block{st}',atol=2e-15)
            equal(var,ev[st:en],f'{arm}_pure8_var_block{st}',atol=1e-27,rtol=2e-10)
    pure_summary[arm]=dict(centered_RMS_per_bank_V=[math.sqrt(v) for v in popvar],all_raw_prefix100_exact_zero=True,
        actual_float64_rows_per_bank=N,actual_source_parents=P)
    del arrays,q,mu,var,emu,ev;gc.collect()
    print(json.dumps(dict(phase='pure_actual_and8moments_PASS',arm=arm)),flush=True)

# The saved stochastic signal increment stays canonical-zero on each actual row.
maximum_charge=0.;square_sum=0.;mean_square_sum=0.
for z in refs:
    a=z['A_signal_pre384'];t=z['theta_signal_pre384'];delta=t-a
    maximum_charge=max(maximum_charge,float(np.max(np.abs(delta[:,110:366].sum(1)))))
    square_sum+=float(np.sum(delta*delta,dtype=np.float64));mean_square_sum+=float(np.sum(delta.reshape(P,R,384).mean(1)**2))
    del a,t,delta;gc.collect()
ok(maximum_charge<=1e-10,'all_actual_stochastic_draw_increments_canonical_zero_Q_and_prefix')
summary['shape_increment']=dict(canonical_charge_max_abs_Vns=maximum_charge,actual_RMS_V=math.sqrt(square_sum/(2*N*384)),
    mean_per_bank_parent_MC_mean_RMS_V=math.sqrt(mean_square_sum/(2*P*384)),
    finite4_or8_MC_mean_not_claimed_population_expectation=True)

encoding_summary=[]
for arm in ('A','theta'):
  for h in H:
    bank_moments=[];scalar_banks=[]
    for b,z in enumerate(refs):
        stem=f'{arm}_{h}__';codes=z[stem+'ADC_crop_codes'];mu=z[stem+'baseline_mean_V'];peaks=z[stem+'peak_index']
        assert codes.dtype==np.uint16 and codes.shape==(N,256) and int(codes.max())<=16383
        assert mu.dtype==np.float64 and mu.shape==(N,) and np.all((peaks>=112)&(peaks<144))
        y=codes.astype(np.float64)/16383-.5-mu[:,None]
        correction=0.;xor_key=stem+'IEEE754_reconstruction_XOR_uint64'
        if xor_key in z.files:
            before=y.copy();xor=z[xor_key];assert xor.dtype==np.uint64 and xor.shape==y.shape
            np.bitwise_xor(y.view(np.uint64),xor,out=y.view(np.uint64));correction=float(np.max(np.abs(y-before)))
            del before,xor
        assert np.isfinite(y).all() and correction<=4e-15
        assert np.array_equal(np.rint((y+mu[:,None]+.5)*16383).astype(np.uint16),codes)
        assert np.array_equal(mu,z[stem+'diagnostic__baseline_mean_V'])
        baseline=z['baseline100'];equal(mu,baseline.mean(1),f'{arm}_{h}_bank{b}_prefix_mean',atol=1e-16)
        doc=read(EVAL/f'FULL_{b}_{arm}_{h}.json');rp=doc['repeated_parent']
        with np.load(EVAL/f'FULL_{b}_{arm}_{h}.npz',allow_pickle=False) as f:
            assert np.array_equal(data(f,rp['parent_rows']),np.arange(P))
            assert np.array_equal(data(f,rp['bank_grid']),np.full((P,4),b))
            assert np.array_equal(data(f,rp['draw_grid']),np.tile(np.arange(4),(P,1)))
            assert np.array_equal(data(f,doc['source_input_trends']['source_ids']),ids)
            features=doc['feature_rows']
            q=y.sum(1);tail=y[:,46:].sum(1);psd=np.divide(tail,q,out=np.full_like(q,np.nan),where=q!=0)
            for key,value in [('Q_Vns',q),('tail_Vns',tail),('front_Vns',q-tail),('PSD_raw',psd),('peak_voltage_V',y[:,16])]:
                equal(value,data(f,features[key]),f'{arm}_{h}_bank{b}_decoded_{key}',atol=2e-12)
            scalar=np.stack([data(f,features[k]) for k in rp['scalar_names']],axis=1).reshape(P,4,-1)
            x=y.reshape(P,4,256);wm=x.mean(1);wv=((x-wm[:,None])**2).sum(1)/3
            equal(wm,data(f,rp['hard_wave_mean_V']),f'{arm}_{h}_bank{b}_wave_mean')
            equal(wv,data(f,rp['hard_wave_covariance_diagonal_V2']),f'{arm}_{h}_bank{b}_wave_var')
            stats=scalar_moments(scalar)
            for k,value in stats.items():
                saved=data(f,rp['scalars'][k])
                if value.dtype.kind in 'iu':assert np.array_equal(value,saved)
                else:equal(value,saved,f'{arm}_{h}_bank{b}_scalar_{k}',atol=2e-11)
        bank_moments.append((wm,wv));scalar_banks.append(scalar)
        encoding_summary.append(dict(arm=arm,hypothesis=h,bank=b,rows=N,codes_dtype='uint16',window_dtype='float64',
            max_IEEE_correction_V=correction,ADC_code_reencode_exact=True,independent_Q_tail_PSD_peak_and_moment_crosschecks=True,
            raw_original_Y_copy_available_for_byte_comparison=False))
        del y,codes,mu,peaks,baseline,x,stats;gc.collect()
    doc=read(EVAL/f'REPEATS_8_{arm}_{h}.json')
    with np.load(EVAL/f'REPEATS_8_{arm}_{h}.npz',allow_pickle=False) as f:
        assert doc['draws_per_parent']==8 and doc['parent_count']==P and doc['covariance_denominator']==7 and doc['covariance_rank_upper_bound']==7
        assert doc['statistical_unit']=='SOURCE_parent_not_MC_draw'
        assert np.array_equal(data(f,doc['bank_grid']),np.tile(np.repeat(np.arange(2),4),(P,1)))
        assert np.array_equal(data(f,doc['draw_grid']),np.tile(np.tile(np.arange(4),2),(P,1)))
        mean=(bank_moments[0][0]+bank_moments[1][0])/2
        var=(3*(bank_moments[0][1]+bank_moments[1][1])+2*(bank_moments[0][0]-bank_moments[1][0])**2)/7
        equal(mean,data(f,doc['hard_wave_mean_V']),f'{arm}_{h}_8wave_mean')
        equal(var,data(f,doc['hard_wave_covariance_diagonal_V2']),f'{arm}_{h}_8wave_var')
        scalar=np.concatenate(scalar_banks,axis=1);stats=scalar_moments(scalar)
        for k,value in stats.items():
            saved=data(f,doc['scalars'][k])
            if value.dtype.kind in 'iu':assert np.array_equal(value,saved)
            else:equal(value,saved,f'{arm}_{h}_8scalar_{k}',atol=2e-11)
    ok(True,f'{arm}_{h}_allparents_4draw_and8draw_wave_and_scalar_covariance_semantics')
    del bank_moments,scalar_banks,scalar,stats,mean,var;gc.collect()
    print(json.dumps(dict(phase='encoded4and8moments_PASS',arm=arm,hypothesis=h)),flush=True)

for z in refs:z.close()
receipt=dict(schema='n01/wp1-independent-reference-array-readback/v1',status='PASS',passed=True,
    UTC=datetime.now(timezone.utc).isoformat(),checks=checks,summary=summary,pure_signal=pure_summary,
    encoded_windows=encoding_summary,numerical_crosscheck_absmax_errors=maxima,
    raw_A_actual_draws_and_two_banks_not_broadcast=True,all_source_parent_draw_lineage_verified=True,
    complete_covariance_recovery='center actual retained draws per parent; factor product/(r-1). All wave diagonals and full9x9 scalar covariances/NaN support counts checked at r4/r8; dense256x256 matrices were not materialized.',
    encoding_scope='All12 encoded windows were decoded, ADC codes re-encoded exactly, and independent saved Q/tail/PSD/peak/moments checked. Native writer roundtrip guarantees the original-Y byte comparison; raw originalY was not separately stored, so this is not an independent byte-for-byte comparison to a second originalY copy.',
    new_frozen_model_or_AD_calls=0,checkpoint_loads=0,raw_dataset_or_DEV_waveform_reads=0,new_random_library_draws=0,
    scientific_loss_or_CDF_reruns=0,SSH_calls=0,Torch_imported=False,
    interpretation='Actual A numerical residual5.4nV is preserved and reported apart from physical randomness; head sample variation and finiteMC means do not establish true conditional response.')
target=OUT/'REFERENCE_ARRAY_READBACK.json'
with target.open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
assert read(target)['passed']
print(json.dumps(dict(status='PASS',receipt=str(target),sha256=sha(target),shape_increment=summary['shape_increment'],encoded_windows=len(encoding_summary))))
