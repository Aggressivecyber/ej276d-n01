"""Independent NumPy checks and explicit-input replay of the frozen006 run.

No optimizer, selection, classifier, mixture fit or new scientific candidate.
Replay preserves the original endpoint and writes only a verification receipt.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from n01.electronics_v3 import ObservedBaselineNoiseV3
from n01.model import N01Response,digitize,charge
from n01.extension006 import ConditionalExtension006,signal_components,assemble_response
from n01.data import load_training,load_validation,write_json

VIEWS=('signal_only','legacy','fast','slow','common_level')
HYPOTHESES=VIEWS[2:]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024**2),b''):h.update(part)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def array_sha(value):
    a=np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def close(actual,expected,*,atol=1e-12,rtol=1e-10,label='numeric'):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape or not np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True):
        difference=float(np.nanmax(abs(a-b))) if a.shape==b.shape else None
        raise AssertionError((label,a.shape,b.shape,difference))
def exact(actual,expected,label):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape or a.dtype!=b.dtype or array_sha(a)!=array_sha(b):
        raise AssertionError((label,a.shape,b.shape,a.dtype.str,b.dtype.str))
def q_np(w):return w[...,100:400].sum(-1)-3*w[...,:100].sum(-1)
def tail_np(w):return w[...,150:400].sum(-1)-2.5*w[...,:100].sum(-1)
def adc_np(w):return np.rint((np.clip(w,-.5,.5)+.5)*16383)/16383-.5


def check_independent_ADC_codes(waveform,pre_adc,*,label):
    """Exact NumPy nearest-even14-bit codes; canonical float64 representation.

    No voltage is corrected or substituted. A reconstruction difference of
    at most4 float64 eps accommodates backend representations of the same
    integer code. The integer-code comparison itself permits zero difference.
    """
    wave=np.asarray(waveform);pre=np.asarray(pre_adc)
    if wave.shape!=pre.shape or wave.ndim!=2 or wave.shape[0]<1 or wave.shape[1]!=480:
        raise AssertionError((label,'ADC waveform/preADC shape',wave.shape,pre.shape))
    if wave.dtype.kind!='f' or wave.dtype.itemsize!=8 or pre.dtype.kind!='f' or pre.dtype.itemsize!=8:
        raise AssertionError((label,'ADC arrays require float64'))
    if not np.isfinite(wave).all() or not np.isfinite(pre).all():
        raise AssertionError((label,'nonfinite ADC waveform/preADC'))
    expected=np.rint((np.clip(pre,-.5,.5)+.5)*16383)
    actual=np.rint((wave+.5)*16383)
    if not np.isfinite(actual).all() or np.any(actual<0) or np.any(actual>16383):
        raise AssertionError((label,'invalid recovered ADC integer code'))
    expected_codes=expected.astype(np.int64);actual_codes=actual.astype(np.int64)
    mismatches=int(np.count_nonzero(expected_codes!=actual_codes))
    if mismatches:
        raise AssertionError((label,'ADC integer code mismatch',mismatches))
    canonical=actual_codes.astype(np.float64)/16383-.5
    error=float(np.max(abs(wave-canonical)))
    bound=float(4*np.finfo(np.float64).eps)
    if not np.isfinite(error) or error>bound:
        raise AssertionError((label,'ADC canonical voltage reconstruction error',error,bound))
    return dict(samples_checked=int(wave.size),integer_code_mismatches=0,
                maximum_canonical_voltage_error_V=error,canonical_voltage_absolute_tolerance=bound)


def psd_np(q,tail,floor):return tail/(np.where(q<0,-1.,1.)*np.maximum(abs(q),floor))
def bins_np(q,statistics):
    q=np.asarray(q,dtype=np.float64)
    q0=statistics['q_floor'];qr=statistics['q_reference']
    coordinate=np.where(q>=q0,np.log(np.maximum(q,q0)/qr),np.log(q0/qr)+np.arcsinh((q-q0)/q0))
    return np.searchsorted(statistics['charge_bin_boundaries'],coordinate,side='left').astype(np.int64)

def extended_intervals(values,probabilities):
    """Linear quantiles retaining positive infinity, while never dropping NaN."""
    values=np.asarray(values);result=np.full((values.shape[1],len(probabilities)),np.nan)
    for column in range(values.shape[1]):
        x=values[:,column]
        if np.isnan(x).any():continue
        x=np.sort(x)
        for j,p in enumerate(probabilities):
            position=(len(x)-1)*p;left=int(np.floor(position));right=int(np.ceil(position));fraction=position-left
            if left==right or fraction==0:result[column,j]=x[left]
            elif np.isposinf(x[right]):result[column,j]=np.inf
            else:result[column,j]=(1-fraction)*x[left]+fraction*x[right]
    return result

def readouts_np(w,floor):
    """Separately compute deterministic readouts without the evaluator module."""
    w=np.asarray(w,dtype=np.float64)
    centered=w-w[:,:100].mean(1,keepdims=True)
    q=q_np(w);tail=tail_np(w)
    raw=np.full_like(q,np.nan);np.divide(tail,q,out=raw,where=q!=0)
    peak_time=np.argmax(centered[:,100:400],axis=1)+100
    peak=centered[np.arange(len(w)),peak_time]
    rise=np.full(len(w),np.nan)
    for row in range(len(w)):
        if peak[row]<=0:continue
        half=peak[row]/2
        for left in range(99,int(peak_time[row])):
            if centered[row,left]<half<=centered[row,left+1]:
                rise[row]=left+(half-centered[row,left])/(centered[row,left+1]-centered[row,left])
                break
    return dict(Q=q,tail=tail,PSD=psd_np(q,tail,floor),PSD_raw=raw,
        Q_nonpositive=q<=0,PSD_guarded=abs(q)<floor,
        peak_voltage=peak,peak_time_ns=peak_time.astype(np.float64),half_rise_ns=rise,
        relative_peak_ns=(peak_time-120).astype(np.float64),relative_half_rise_ns=rise-120,
        baseline_rms_V=np.sqrt(np.mean(centered[:,:100]**2,axis=1)))

def independent_bootstrap(obs,split,cfg):
    """Paired parent resampling, independently reduced from saved event arrays.

    Resampling is implemented as parent multiplicities, not copied event rows.
    The RNG draws match the registered evaluator but all statistics are rebuilt.
    """
    arms=cfg['evaluation']['arms'];n=len(obs[split+'_'+arms[0]+'_Q'][0])//4
    realq=obs['TRAIN_REAL_Q'];realp=obs['TRAIN_REAL_PSD']
    scaleq=np.std(realq,ddof=0);scalep=np.std(realp,ddof=0)
    rng=np.random.default_rng(cfg['seeds']['bootstrap_'+split])
    parent=np.repeat(np.arange(n),4)
    records={};support={}
    for arm in arms:
        pre=split+'_'+arm+'_';q=obs[pre+'Q'][2:];p=obs[pre+'PSD'][2:]
        bins=obs[pre+'fixed_bins'];counts=np.zeros((8,n));sums=np.zeros((8,3,n));squares=sums.copy()
        rows=[];parents=[]
        for b in range(8):
            mask=bins==b
            counts[b]=np.bincount(parent[mask],minlength=n)
            for h in range(3):
                selected=p[h,mask]
                # Translation leaves SD unchanged and keeps a nonzero
                # constant exactly constant under finite-precision reduction.
                center=selected[0] if len(selected) else 0.
                centered=selected-center
                sums[b,h]=np.bincount(parent[mask],weights=centered,minlength=n)
                squares[b,h]=np.bincount(parent[mask],weights=centered**2,minlength=n)
            rows.append(int(mask.sum()));parents.append(int((counts[b]>0).sum()))
        support[arm]=dict(rows=rows,parents=parents,supported=[r>=cfg['evaluation']['minimum_rows_per_fixed_bin'] and k>=cfg['evaluation']['minimum_parents_per_fixed_bin'] for r,k in zip(rows,parents)])
        records[arm]=(q.reshape(3,n,4).mean(2),p.reshape(3,n,4).mean(2),counts,sums,squares)
    count=cfg['evaluation']['bootstrap_draws']
    result={arm:dict(Q_mean_span=np.zeros(count),PSD_mean_span=np.zeros(count),fixed_bin_PSD_sd_ratio=np.full((count,8),np.nan),max_fixed_bin_PSD_sd_ratio=np.full(count,np.nan)) for arm in arms}
    def statistic(arm,weight):
        q,p,count,total,sq=records[arm];den=weight.sum()
        qspan=float(np.ptp(q@weight/den)/scaleq);pspan=float(np.ptp(p@weight/den)/scalep)
        perbin=np.full(8,np.nan)
        for b in range(8):
            c=count[b]@weight
            if c==0 or not support[arm]['supported'][b]:continue
            means=total[b]@weight/c;variance=sq[b]@weight/c-means**2
            sd=np.sqrt(np.maximum(variance,0))
            if np.min(sd)>0:perbin[b]=np.max(sd)/np.min(sd)
            elif np.max(sd)==0:perbin[b]=np.nan
            else:perbin[b]=np.inf
        supported=np.asarray(support[arm]['supported'])
        maximum=float(np.max(perbin[supported])) if supported.any() else np.nan
        return qspan,pspan,perbin,maximum
    points={arm:statistic(arm,np.ones(n)) for arm in arms}
    hashes=[]
    for draw in range(count):
        sampled=rng.integers(0,n,size=n)
        hashes.append(array_sha(sampled))
        weight=np.bincount(sampled,minlength=n).astype(np.float64)
        for arm in arms:
            q,p,b,m=statistic(arm,weight)
            result[arm]['Q_mean_span'][draw]=q
            result[arm]['PSD_mean_span'][draw]=p
            result[arm]['fixed_bin_PSD_sd_ratio'][draw]=b
            result[arm]['max_fixed_bin_PSD_sd_ratio'][draw]=m
    return dict(samples=result,points=points,support=support,REAL_Q_sd=float(scaleq),REAL_PSD_sd=float(scalep),resample_parent_hashes=hashes)

def independent_theory(noise_state,cfg):
    """Use a separately written NumPy block solve, not the006 implementation."""
    model=ObservedBaselineNoiseV3(noise_state['initialization'])
    model.load_state_dict(noise_state['model'],strict=True)
    with torch.no_grad():
        mu=model.mean_vector().numpy();L=model.cholesky_factor().numpy()
        eta=float(model.log_scale_std)
    C=L@L.T;v=float(np.mean(np.diag(C)));mu0=float(np.mean(mu))
    t=np.arange(480);distance=abs(t[:,None]-t[None,:])
    cq=np.zeros(480);cq[:100]=-3.;cq[100:400]=1.
    ct=np.zeros(480);ct[:100]=-2.5;ct[150:400]=1.
    result={}
    for spec in cfg['hypotheses']:
        K=spec['short_weight']*np.exp(-distance/spec['short_length'])
        if spec['long_type']=='exponential':K=K+spec['long_weight']*np.exp(-distance/spec['long_length'])
        if spec['long_type']=='constant':K=K+spec['long_weight']
        K=v*(K+np.eye(480)*cfg['extension']['kernel_ridge_relative'])
        A=np.linalg.solve(K[:100,:100],K[:100,100:]).T
        R=K[100:,100:]-A@K[:100,100:];R=(R+R.T)/2
        B=np.linalg.cholesky(R)
        mean=np.r_[mu,mu0+A@(mu-mu0)]
        # A direct block factor gives an independent PSD construction.
        factor=np.zeros((480,480));factor[:100,:100]=L
        factor[100:,:100]=A@L;factor[100:,100:]=B
        cov=factor@factor.T
        a=A[0].copy();a[99]-=1
        prefix_dv=np.diag(C)[90:99]+np.diag(C)[91:100]-2*C[np.arange(90,99),np.arange(91,100)]
        seam_var=a@C@a+R[0,0]
        result[spec['name']]=dict(mean=mean,covariance=cov,A=A,R=R,cholesky_R=B,kernel_reference=K,
            C100=C,eta=np.array(eta),kernel_variance=np.array(v),mean_reference=np.array(mu0),
            scale_mean=np.array(np.exp(-.5*eta**2)),scale_second_moment=np.array(1.),scale_fourth_moment=np.array(np.exp(4*eta**2)),
            variance_Q=np.array(cq@cov@cq),variance_tail_charge=np.array(ct@cov@ct),
            covariance_Q_tail=np.array(cq@cov@ct),mean_Q=np.array(cq@mean),mean_tail_charge=np.array(ct@mean),
            seam_increment_variance=np.array(seam_var),
            seam_increment_mean=np.array(mean[100]-mean[99]),
            prefix_increment_variance90_99=prefix_dv,
            prefix_increment_start=np.arange(90,99),
            seam_variance_over_prefix_increment_mean=np.array(seam_var/prefix_dv.mean()),
            first_tail_last10_prefix_correlation=cov[100,90:100]/np.sqrt(cov[100,100]*np.diag(cov)[90:100]),
            tail_sample_variance=np.diag(cov)[100:],
            tail400480_variance_mean=np.array(np.diag(cov)[400:480].mean()))
    return result


def theory_A_backward_error(A,independent_kernel,*,label):
    """Fixed normalized infinity-norm residual of the registered block solve.

    Both saved and independent A use the same independent NumPy kernel, never
    each matrix's saved/self-consistent kernel. A is dimensionless; the ratio
    below cancels the voltage-squared normalization of the kernel.
    """
    a=np.asarray(A);kernel=np.asarray(independent_kernel)
    if a.shape!=(380,100) or kernel.shape!=(480,480):
        raise AssertionError((label,'A/kernel shape',a.shape,kernel.shape))
    if a.dtype.kind!='f' or a.dtype.itemsize!=8 or kernel.dtype.kind!='f' or kernel.dtype.itemsize!=8:
        raise AssertionError((label,'A and independent kernel require float64'))
    if not np.isfinite(a).all() or not np.isfinite(kernel).all():
        raise AssertionError((label,'nonfinite A or independent kernel'))
    KBB=kernel[:100,:100];KTB=kernel[100:,:100]
    residual=a@KBB-KTB
    denominator=np.linalg.norm(a,ord=np.inf)*np.linalg.norm(KBB,ord=np.inf)+np.linalg.norm(KTB,ord=np.inf)
    if not np.isfinite(residual).all() or not np.isfinite(denominator) or denominator<=0:
        raise AssertionError((label,'nonfinite residual or invalid backward-error denominator'))
    error=float(np.linalg.norm(residual,ord=np.inf)/denominator)
    bound=64*np.finfo(np.float64).eps
    if not np.isfinite(error) or error>bound:
        raise AssertionError((label,'A normalized infinity-norm backward error',error,float(bound)))
    return error


def check_theory_numeric(saved,independent):
    """Finite theory checks; only dimensionless A gets a64-epsilon floor.

    All other analytical fields keep atol1e-15 and rtol3e-10. There is no
    covariance, mean, noise, readout, bootstrap, or decision tolerance change.
    """
    if set(independent)!=set(HYPOTHESES):
        raise AssertionError('All three registered theory hypotheses are required')
    A_floor=64*np.finfo(np.float64).eps
    diagnostics={}
    for h,values in independent.items():
        for key,value in values.items():
            actual=np.asarray(saved[h+'_'+key]);expected=np.asarray(value)
            if not np.issubdtype(actual.dtype,np.number) or not np.issubdtype(expected.dtype,np.number):
                raise AssertionError((h,key,'nonnumeric analytical field'))
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise AssertionError((h,key,'nonfinite analytical field'))
            close(actual,expected,atol=A_floor if key=='A' else 1e-15,rtol=3e-10,label=h+' analytical '+key)
        reference_kernel=values['kernel_reference']
        saved_A=np.asarray(saved[h+'_A']);independent_A=np.asarray(values['A'])
        saved_error=theory_A_backward_error(saved_A,reference_kernel,label=h+' saved A')
        independent_error=theory_A_backward_error(independent_A,reference_kernel,label=h+' independent A')
        np.linalg.cholesky(values['covariance'])
        diagnostics[h]=dict(A_max_absolute_difference=float(np.max(abs(saved_A-independent_A))),
                            saved_A_backward_error=saved_error,independent_A_backward_error=independent_error)
    return dict(float64_epsilon=float(np.finfo(np.float64).eps),A_absolute_floor=float(A_floor),
                other_theory_absolute_tolerance=1e-15,relative_tolerance=3e-10,
                A_backward_error_limit=float(A_floor),kernel_for_both_A_checks='independent NumPy registered kernel',
                hypotheses=diagnostics)

def check_bindings(cfg_path,manifest_path,container,identities,out):
    cfg=read(cfg_path);manifest=read(manifest_path)
    assert cfg['run_id']=='integration_006_20261012'
    assert manifest['config_sha256']==sha(cfg_path)
    for relative,digest in manifest['files'].items():assert sha(ROOT/relative)==digest,relative
    for item in cfg['references'].values():assert sha(ROOT/item['path'])==item['sha256'],item['path']
    assert sha(container)==cfg['train_container_sha256'] and sha(identities)==cfg['identities_sha256']
    complete=read(out/'COMPLETE.json')
    assert complete['status']=='zero_training_completed'
    assert complete['run_id']==cfg['run_id']
    provenance=complete['provenance']
    assert provenance['config_sha256']==sha(cfg_path)
    assert provenance['protocol_sha256']==manifest['protocol_sha256']==sha(ROOT/manifest['protocol_path'])
    assert provenance['source_manifest_sha256']==sha(manifest_path)
    assert provenance['container_sha256']==cfg['train_container_sha256']
    assert provenance['identities_sha256']==cfg['identities_sha256']
    assert provenance['references']=={key:value['sha256'] for key,value in cfg['references'].items()}
    assert provenance['source_files']==manifest['files']
    required={'observables.npz','curves.npz','theory.npz','bootstrap.npz','DECISION.json','SUMMARY.json',
        'replay128.pt','sample_waveforms.pt','latents/TRAIN.pt','latents/DEV.pt','latents/NOISE_DIAGNOSTIC.pt'}
    payload=complete['payload_sha256']
    assert required<=set(payload),'Completed payload omits a required verification input'
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
    assert set(payload)==actual,'Completed payload set differs from immutable output files'
    assert sum((out/name).stat().st_size for name in payload)<=cfg['output_limit_MiB']*1024**2
    for name,digest in complete['payload_sha256'].items():assert sha(out/name)==digest,name
    assert complete['updates']==dict(noise=0,response=0,classifier=0,GMM_fits=0)
    return cfg,manifest,complete

def read_arrays(path):
    with np.load(path,allow_pickle=False) as f:return {key:f[key].copy() for key in f.files}

def cpu(x):return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
def dictionary_sha(values):
    digest=hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode('utf8')+b'\0'+array_sha(cpu(values[key])).encode('ascii')+b'\n')
    return digest.hexdigest()

class CurveCheck:
    """Stream population, fixed-member and parent statistics independently."""
    def __init__(self):
        self.rows=0;self.parents=0;self.total=np.zeros((5,480));self.square=self.total.copy()
        self.within=self.total.copy();self.bin_total=np.zeros((8,5,480));self.bin_square=self.bin_total.copy()
        self.bin_rows=np.zeros(8,dtype=np.int64)
    def add(self,w,bins):
        n=w.shape[1];assert n%4==0
        self.rows+=n;self.parents+=n//4
        self.total+=w.sum(1);self.square+=(w*w).sum(1)
        self.within+=w.reshape(5,n//4,4,480).var(2,ddof=1).sum(1)
        for b in range(8):
            mask=bins==b;self.bin_rows[b]+=int(mask.sum())
            self.bin_total[b]+=w[:,mask].sum(1);self.bin_square[b]+=(w[:,mask]**2).sum(1)
    def check(self,saved,prefix):
        mean=self.total/self.rows;variance=np.maximum(self.square/self.rows-mean**2,0)
        close(saved[prefix+'mean'],mean,label=prefix+'mean curve')
        close(saved[prefix+'variance'],variance,label=prefix+'variance curve')
        close(saved[prefix+'mean_within_parent_variance'],self.within/self.parents,label=prefix+'within parent curve')
        bm=np.full_like(self.bin_total,np.nan);bv=bm.copy()
        for b,count in enumerate(self.bin_rows):
            if count:
                bm[b]=self.bin_total[b]/count
                bv[b]=np.maximum(self.bin_square[b]/count-bm[b]**2,0)
        close(saved[prefix+'fixed_bin_mean'],bm,label=prefix+'fixed mean')
        close(saved[prefix+'fixed_bin_variance'],bv,label=prefix+'fixed variance')

def check_real_curves(curves,obs,data):
    for split in ('TRAIN','DEV'):
        real=data[split]['real'];bins=obs[split+'_REAL_bins']
        close(curves[split+'_REAL_mean'],real.mean(0),label=split+' REAL mean curve')
        close(curves[split+'_REAL_variance'],real.var(0,ddof=0),label=split+' REAL variance curve')
        for b in range(8):
            mask=bins==b
            mean=real[mask].mean(0) if mask.any() else np.full(480,np.nan)
            variance=real[mask].var(0,ddof=0) if mask.any() else np.full(480,np.nan)
            close(curves[split+'_REAL_bin_mean'][b],mean,label=split+' REAL bin mean')
            close(curves[split+'_REAL_bin_variance'][b],variance,label=split+' REAL bin variance')

def load_models(cfg,device):
    torch.manual_seed(cfg['seeds']['model_constructor'])
    noise_state=torch.load(ROOT/cfg['references']['noise005']['path'],map_location='cpu',weights_only=False)
    noise=ObservedBaselineNoiseV3(noise_state['initialization'])
    noise.load_state_dict(noise_state['model'],strict=True);noise=noise.to(device).eval().requires_grad_(False)
    extension=ConditionalExtension006(noise,cfg['hypotheses'],cfg['extension']['kernel_ridge_relative'])
    arms={}
    for arm in cfg['evaluation']['arms']:
        state=torch.load(ROOT/cfg['references'][arm]['path'],map_location='cpu',weights_only=False)
        assert state['variant']==arm and state['update']==20000
        assert state['config']==read(ROOT/cfg['references']['config001']['path'])
        model=N01Response(state['model_statistics'],variant=arm)
        model.load_state_dict(state['model'],strict=True);model.set_stage(state['stage'])
        arms[arm]=model.to(device).eval().requires_grad_(False)
    return noise_state,noise,extension,arms

def new_streams(saved,expected_seeds):
    assert saved['seeds']==expected_seeds,'RNG seeds differ from the frozen configuration'
    assert set(saved['streams_initial'])==set(saved['streams_final'])==set(expected_seeds)
    streams={}
    for name,seed in saved['seeds'].items():
        streams[name]=torch.Generator(device='cpu').manual_seed(seed)
        assert torch.equal(streams[name].get_state(),saved['streams_initial'][name]),name
    return streams

def chunk_latents(n,streams,noise):
    response={key:torch.randn((n,)+shape,dtype=torch.float32,generator=streams['response']) for key,shape in N01Response.noise_shapes.items()}
    baseline=noise.sample_noise(n,device='cpu',generator=streams['noise005'],length=100)
    epsilon=torch.randn((n,1,380),dtype=torch.float32,generator=streams['extension'])
    return response,baseline,epsilon

@torch.no_grad()
def replay_all(cfg,data,obs,curves,out,device,noise,extension,models):
    """Re-execute only the identical frozen inputs; no new draws or candidates."""
    replay=torch.load(out/'replay128.pt',map_location='cpu',weights_only=False)
    samples=torch.load(out/'sample_waveforms.pt',map_location='cpu',weights_only=False)
    counts=dict(response_forward_events=0,noise005_forward_events=0,extension_events=0,
        regenerated_source_rows=0,checked_view_rows=0,explicit_sample_rows=0,random_stream_chunks=0)
    leakage={};engineering={name:dict(passed=True,failed_checks=0) for name in ('legacy_Q_identity','new_Q_identity')}
    def engineering_check(name,condition):
        if not bool(condition):engineering[name]['passed']=False;engineering[name]['failed_checks']+=1
    for split in ('TRAIN','DEV'):
        saved=torch.load(out/'latents'/f'{split}.pt',map_location='cpu',weights_only=False)
        streams=new_streams(saved,{name:cfg['seeds'][split+'_'+name] for name in ('response','noise005','extension')})
        ds=data[split];parents=np.repeat(np.arange(len(ds['source']),dtype=np.int64),4)
        accum={arm:CurveCheck() for arm in models};leakage[split]={}
        for arm in models:leakage[split][arm]=dict(absmax=0.,sum=0.,sum_square=0.,samples=0)
        expected_start=0
        for record in saved['chunks']:
            start=record['row_start'];stop=record['row_stop'];n=stop-start
            assert start==expected_start and 0<n<=128 and n%4==0
            assert stop==min(start+128,len(parents));expected_start=stop
            response,baseline,epsilon=chunk_latents(n,streams,noise)
            assert dictionary_sha(response)==record['response_sha256']
            assert dictionary_sha(baseline)==record['noise005_sha256']
            assert array_sha(cpu(epsilon))==record['extension_sha256']
            x=torch.as_tensor(ds['source'][parents[start:stop]],dtype=torch.float32,device=device)
            nd={key:value.to(device) for key,value in baseline.items()}
            rd={key:value.to(device) for key,value in response.items()}
            ep=epsilon.to(device);prefix=noise(nd,length=100)
            scale=torch.exp(noise.log_scale_std*nd['baseline_scale'].to(torch.float64)-noise.log_scale_std.square())
            assert array_sha(cpu(prefix))==record['noise005_prefix_sha256']
            assert array_sha(cpu(scale))==record['scale_sha256']
            extended={h:extension.extend_prefix(prefix,scale,ep,h)['noise_pre'] for h in HYPOTHESES}
            for h,value in extended.items():
                assert array_sha(cpu(value))==record['extended_noise_sha256'][h]
                assert torch.equal(value[...,:100],prefix)
                assert torch.equal(digitize(value[...,:100],ste=False),digitize(prefix,ste=False))
            if start==0:
                first=replay['splits'][split]
                for latent_name,latent in response.items():exact(cpu(first['response_latents'][latent_name]),cpu(latent),'explicit response latent')
                for latent_name,latent in baseline.items():exact(cpu(first['noise005_latents'][latent_name]),cpu(latent),'explicit005 latent')
                exact(cpu(first['extension_gaussian']),cpu(epsilon),'explicit epsilon')
                exact(cpu(first['source']),cpu(x),'explicit SOURCE')
                exact(cpu(first['noise005_pre']),cpu(prefix),'explicit prefix')
                exact(cpu(first['shared_scale']),cpu(scale),'explicit scale')
                exact(cpu(first['source_parent_row']),parents[:n],'sample original parent rows')
                exact(cpu(first['source_parent_ids']),ds['source_ids'][parents[:n]],'sample original parent IDs')
                exact(cpu(first['draw_index']),np.tile(np.arange(4,dtype=np.int64),n//4),'sample draw order')
            for arm,model in models.items():
                pre=split+'_'+arm+'_';old=model(x,noise=rd,adc_ste=False)
                signal=old['qsig'][:,None]*old['unit_shape']+old['signal_noise']
                assert torch.equal(signal,signal_components(old))
                assert torch.equal(signal+old['electronics_noise'],old['pre_adc'])
                residual=q_np(cpu(old['pre_adc']))-cpu(old['qsig'])-q_np(cpu(old['electronics_noise']))
                close(obs[pre+'legacy_charge_identity_residual'][start:stop],residual,label=pre+' legacy charge identity')
                close(obs[pre+'legacy_assembly_max_abs_per_event'][start:stop],np.zeros(n),atol=0,rtol=0,label=pre+' legacy component identity')
                # Use the frozen readout reduction for the exact engineering
                # tolerance, and independently check the saved residual above.
                old_residual=charge(old['pre_adc'])-old['qsig']-charge(old['electronics_noise'])
                engineering_check('legacy_Q_identity',torch.all(abs(old_residual)<=1e-12))
                assert array_sha(cpu(signal))==record['arms'][arm]['signal_sha256']
                assert array_sha(cpu(old['pre_adc']))==record['arms'][arm]['legacy_pre_sha256']
                new=[assemble_response(old,extended[h]) for h in HYPOTHESES]
                for item in new:engineering_check('new_Q_identity',torch.all(abs(item['charge_identity_residual'])<=1e-12))
                wave=np.stack([cpu(digitize(signal,ste=False)),cpu(old['waveform'])]+[cpu(xn['waveform']) for xn in new])
                pres=np.stack([cpu(signal),cpu(old['pre_adc'])]+[cpu(xn['pre_adc']) for xn in new])
                noises=np.stack([np.zeros_like(cpu(signal)),cpu(old['electronics_noise'])]+[cpu(extended[h])[:,0] for h in HYPOTHESES])
                assert array_sha(wave)==record['arms'][arm]['output_sha256']
                assert np.array_equal(wave[2,:,:100],wave[3,:,:100]) and np.array_equal(wave[2,:,:100],wave[4,:,:100])
                for iv,view in enumerate(VIEWS):
                    check_independent_ADC_codes(wave[iv],pres[iv],label=pre+view+' independent ADC codes')
                    values=readouts_np(wave[iv],cfg['_stats']['q_floor'])
                    values['baseline_mean_V']=wave[iv,:,:100].mean(1)
                    sat=(pres[iv]<-.5)|(pres[iv]>.5)
                    values.update(saturation_samples=sat.sum(1),saturation_fraction_per_event=sat.mean(1),
                        pre_adc_Q=q_np(pres[iv]),noise_pre_Q=q_np(noises[iv]))
                    values['ADCcharge_change']=values['Q']-values['pre_adc_Q']
                    values['charge_identity_residual']=values['pre_adc_Q']-q_np(cpu(signal))-values['noise_pre_Q']
                    for key,value in values.items():close(obs[pre+key][iv,start:stop],value,label=pre+view+' '+key)
                signal_cpu=cpu(signal);signalq=q_np(signal_cpu)
                close(obs[pre+'signal_pre_Q'][start:stop],signalq,label=pre+' signal Q')
                close(obs[pre+'qsig'][start:stop],cpu(old['qsig']),label=pre+' qsig')
                maximum=np.max(abs(signal_cpu[:,:100]),axis=1)
                close(obs[pre+'signal_baseline_max_per_event'][start:stop],maximum,label=pre+' leakage')
                change=np.mean(wave[2,:,:100]!=cpu(digitize(prefix,ste=False))[:,0],axis=1)
                close(obs[pre+'signal_prefix_ADC_change_fraction'][start:stop],change,label=pre+' baseline ADC change')
                item=leakage[split][arm];sb=signal_cpu[:,:100]
                item['absmax']=max(item['absmax'],float(abs(sb).max()));item['sum']+=float(sb.sum())
                item['sum_square']+=float((sb**2).sum());item['samples']+=sb.size
                fixed=bins_np(signalq,cfg['_stats']);accum[arm].add(wave,fixed)
                if start==0:
                    expected=first['arms'][arm];sample=samples['splits'][split]['arms'][arm]
                    for key,value in dict(signal_pre=signal_cpu,legacy_pre=cpu(old['pre_adc']),old_electronics_noise=cpu(old['electronics_noise']),qsig=cpu(old['qsig']),expected_views=wave,new_pre_adc=pres[2:],new_noise_pre=noises[2:]).items():exact(cpu(expected[key]),value,'explicit replay '+arm+' '+key)
                    for key,value in dict(signal_pre=signal_cpu,noise_pre=noises[2:],waveform=wave,pre_adc=pres).items():exact(cpu(sample[key]),value,'report sample '+arm+' '+key)
                    counts['explicit_sample_rows']+=n
                counts['response_forward_events']+=n;counts['checked_view_rows']+=5*n
            counts['noise005_forward_events']+=n;counts['extension_events']+=3*n
            counts['regenerated_source_rows']+=n;counts['random_stream_chunks']+=1
        assert expected_start==len(parents)
        for name,generator in streams.items():assert torch.equal(generator.get_state(),saved['streams_final'][name]),name
        for arm in models:accum[arm].check(curves,split+'_'+arm+'_')
        print(json.dumps(dict(phase='independent_replay',split=split,rows=len(parents),arms=list(models))),flush=True)
    return counts,leakage,engineering

@torch.no_grad()
def replay_noise_diagnostic(cfg,theory,out,device,noise,extension):
    saved=torch.load(out/'latents/NOISE_DIAGNOSTIC.pt',map_location='cpu',weights_only=False)
    streams=new_streams(saved,dict(noise005=cfg['seeds']['noise_diagnostic005'],extension=cfg['seeds']['noise_diagnostic_extension']))
    waves={h:[] for h in HYPOTHESES};scales=[];rows=0
    for record in saved['chunks']:
        start=record['row_start'];stop=record['row_stop'];n=stop-start
        assert start==rows and stop==min(start+128,8192)
        latents=noise.sample_noise(n,device='cpu',generator=streams['noise005'],length=100)
        epsilon=torch.randn((n,1,380),dtype=torch.float32,generator=streams['extension'])
        assert dictionary_sha(latents)==record['noise005_sha256']
        assert array_sha(cpu(epsilon))==record['extension_sha256']
        gpu={key:value.to(device) for key,value in latents.items()}
        prefix=noise(gpu,length=100)
        scale=torch.exp(noise.log_scale_std*gpu['baseline_scale'].to(torch.float64)-noise.log_scale_std.square())
        assert array_sha(cpu(prefix))==record['noise005_prefix_sha256']
        assert array_sha(cpu(scale))==record['scale_sha256']
        for h in HYPOTHESES:
            value=extension.extend_prefix(prefix,scale,epsilon.to(device),h)['noise_pre']
            assert array_sha(cpu(value))==record['extended_noise_sha256'][h]
            waves[h].append(cpu(value)[:,0])
        scales.append(cpu(scale).reshape(-1));rows=stop
    assert rows==8192
    for name,generator in streams.items():assert torch.equal(generator.get_state(),saved['streams_final'][name]),name
    scale=np.concatenate(scales)
    for h in HYPOTHESES:
        wave=np.concatenate(waves[h]);post=adc_np(wave);mean=wave.mean(0);centered=wave-mean
        derived=dict(diagnostic_mean=mean,diagnostic_covariance=centered.T@centered/len(wave),
            diagnostic_Q=q_np(wave),diagnostic_tail=tail_np(wave),diagnostic_seam_increment=wave[:,100]-wave[:,99],
            diagnostic_scale=scale,diagnostic_ADC_Q=q_np(post),diagnostic_ADC_tail=tail_np(post))
        for key,value in derived.items():close(theory[h+'_'+key],value,label=h+' '+key)
    return dict(noise005_forward_events=rows,extension_events=rows*3,noise_hypothesis_rows=rows*3,random_stream_chunks=len(saved['chunks']))

def check_bootstrap(saved,obs,cfg):
    verified={}
    order=['Q_mean_span_over_REAL_sd','PSD_mean_span_over_REAL_sd']+[f'fixed_bin_{b}_PSD_sd_ratio' for b in range(8)]+['fixed_bin_max_PSD_sd_ratio']
    names=[arm+'/'+key for arm in cfg['evaluation']['arms'] for key in order]
    for split in ('TRAIN','DEV'):
        computed=independent_bootstrap(obs,split,cfg)
        blocks=[];points=[]
        for arm in cfg['evaluation']['arms']:
            p=computed['points'][arm];s=computed['samples'][arm]
            points.extend([p[0],p[1],*p[2],p[3]])
            blocks.append(np.column_stack((s['Q_mean_span'],s['PSD_mean_span'],s['fixed_bin_PSD_sd_ratio'],s['max_fixed_bin_PSD_sd_ratio'])))
        values=np.concatenate(blocks,axis=1);points=np.array(points)
        intervals=extended_intervals(values,cfg['decision']['bootstrap_interval'])
        assert saved[split+'_metric_names'].tolist()==names
        assert int(saved[split+'_seed'])==cfg['seeds']['bootstrap_'+split]
        assert saved[split+'_resample_parent_hashes'].tolist()==computed['resample_parent_hashes']
        close(saved[split+'_metric_values'],values,atol=2e-10,rtol=2e-9,label=split+' parent bootstrap all draws')
        close(saved[split+'_point_metrics'],points,atol=2e-10,rtol=2e-9,label=split+' point metrics')
        close(saved[split+'_intervals'],intervals,atol=2e-10,rtol=2e-9,label=split+' intervals')
        verified[split]=dict(metric_names=names,points=points,intervals=intervals,support=computed['support'])
    return verified

def independent_decision(cfg,obs,verified,theory,leakage,engineering_checks=None):
    limits=cfg['decision'];train=verified['TRAIN'];engineering=[]
    if engineering_checks:
        engineering.extend(name for name,item in engineering_checks.items() if not item['passed'])
    for split in ('TRAIN',):
        for arm in cfg['evaluation']['arms']:
            if leakage[split][arm]['absmax']>limits['signal_baseline_absolute_max_V']:
                engineering.append(split+'/'+arm+'/signal_baseline_leakage')
    for arm in cfg['evaluation']['arms']:
        prefix='TRAIN_'+arm+'_'
        if np.any(obs[prefix+'Q'][2:]<=0) or np.any(obs[prefix+'PSD_guarded'][2:]):
            engineering.append('TRAIN/'+arm+'/invalid_new_readout')
    noise_variances=np.array([float(theory[h+'_tail400480_variance_mean']) for h in HYPOTHESES])
    ratio=float(noise_variances.max()/noise_variances.min())
    all_supported=all(all(item['supported']) for item in train['support'].values())
    lower_fails=[];upper_fails=[];undefined=[]
    for name,ci in zip(train['metric_names'],train['intervals']):
        metric=name.split('/',1)[1]
        threshold=limits['Q_mean_range_over_REAL_sd_max'] if metric.startswith('Q_mean') else limits['PSD_mean_range_over_REAL_sd_max'] if metric.startswith('PSD_mean') else limits['fixed_bin_PSD_sd_ratio_max']
        if np.isnan(ci).any() or np.isneginf(ci).any():undefined.append(name)
        else:
            if ci[0]>threshold:lower_fails.append(name)
            if ci[1]>threshold:upper_fails.append(name)
    if engineering:outcome='interface_fail'
    elif lower_fails or ratio>limits['noise_tail_variance_ratio_max']:outcome='sensitivity_detected'
    elif all_supported and not upper_fails and not undefined:outcome='sensitivity_within_family'
    else:outcome='MC_inconclusive'
    return dict(outcome=outcome,engineering_failures=engineering,lower_above=lower_fails,upper_above=upper_fails,
        undefined_intervals=undefined,all_fixed_bins_supported=all_supported,noise_tail_variance_ratio=ratio)

def check_actual_counts(cfg,complete,summary,data):
    expected={name:{} for name in ('model_forward','noise005_forward','extension_forward','signal_reference_ADC','candidate_ADC')}
    expected.update(REAL_readout_rows={},generated_view_rows={},diagnostic_rows=8192)
    for split in ('TRAIN','DEV'):
        n=len(data[split]['source'])*4;batch=dict(batches=math.ceil(n/128),events=n)
        expected['noise005_forward'][split]=batch.copy()
        expected['REAL_readout_rows'][split]=len(data[split]['real'])
        expected['generated_view_rows'][split]={}
        for h in HYPOTHESES:expected['extension_forward'][split+'/'+h]=batch.copy()
        for arm in cfg['evaluation']['arms']:
            role=split+'/'+arm
            expected['model_forward'][role]=batch.copy()
            expected['signal_reference_ADC'][role]=batch.copy()
            expected['generated_view_rows'][split][arm]=5*n
            for h in HYPOTHESES:expected['candidate_ADC'][role+'/'+h]=batch.copy()
    expected['noise005_forward']['NOISE_DIAGNOSTIC']=dict(batches=64,events=8192)
    for h in HYPOTHESES:expected['extension_forward']['NOISE_DIAGNOSTIC/'+h]=dict(batches=64,events=8192)
    for name in ('model_forward','noise005_forward','extension_forward','signal_reference_ADC','candidate_ADC'):
        expected[name+'_attempted']={role:value.copy() for role,value in expected[name].items()}
    assert complete['actual_counts']==summary['actual_counts']==expected,'Actual execution counts differ from complete original-parent coverage'
    return expected

def check_observables(obs,cfg,stats,data):
    """Check all saved events, readout algebra, fixed bins and migrations."""
    counts={}
    for split in ('TRAIN','DEV'):
        ds=data[split]
        source,real,source_ids,real_ids,real_groups=(ds[k] for k in ('source','real','source_ids','real_ids','real_groups'))
        n=len(source);rows=n*4
        assert len(real)==cfg['evaluation']['train_real_rows' if split=='TRAIN' else 'dev_real_rows']
        exact(obs[split+'_source_parent_ids'],source_ids,split+' source IDs')
        exact(obs[split+'_source_parent_row'],np.repeat(np.arange(n,dtype=np.int64),4),split+' parent rows')
        exact(obs[split+'_draw_index'],np.tile(np.arange(4,dtype=np.int64),n),split+' draw indices')
        close(obs[split+'_source_h'],ds['h'],atol=0,rtol=0,label=split+' original h')
        exact(obs[split+'_REAL_ids'],real_ids,split+' REAL IDs')
        exact(obs[split+'_REAL_groups'],real_groups,split+' REAL groups')
        close(obs[split+'_source_count_weights'],np.ones(n),atol=0,rtol=0)
        close(obs[split+'_REAL_count_weights'],np.ones(len(real)),atol=0,rtol=0)
        for arm in cfg['evaluation']['arms']:
            prefix=split+'_'+arm+'_'
            q=obs[prefix+'Q'];tail=obs[prefix+'tail'];psd=obs[prefix+'PSD']
            assert q.shape==tail.shape==psd.shape==(5,rows)
            close(psd,psd_np(q,tail,stats['q_floor']),label=prefix+'PSD')
            assert np.array_equal(obs[prefix+'PSD_guarded'],abs(q)<stats['q_floor'])
            assert np.array_equal(obs[prefix+'Q_nonpositive'],q<=0)
            raw=np.full_like(q,np.nan);np.divide(tail,q,out=raw,where=q!=0)
            close(obs[prefix+'PSD_raw'],raw,label=prefix+'raw PSD')
            fixed=bins_np(obs[prefix+'signal_pre_Q'],stats)
            assert np.array_equal(obs[prefix+'fixed_bins'],fixed)
            bins=bins_np(q,stats)
            assert np.array_equal(obs[prefix+'rebinned_bins'],bins)
            migrations=np.zeros((5,8,8),np.int64)
            for i in range(5):np.add.at(migrations[i],(fixed,bins[i]),1)
            assert np.array_equal(obs[prefix+'migration'],migrations)
            exact(obs[prefix+'fixed_bin_rows'],np.bincount(fixed,minlength=8),prefix+' bin row support')
            parents=np.array([len(np.unique(np.repeat(np.arange(n,dtype=np.int64),4)[fixed==b])) for b in range(8)],dtype=np.int64)
            exact(obs[prefix+'fixed_bin_parents'],parents,prefix+' bin parent support')
            close(q-obs[prefix+'pre_adc_Q'],obs[prefix+'ADCcharge_change'],label=prefix+' ADC charge')
            close(obs[prefix+'pre_adc_Q']-obs[prefix+'signal_pre_Q'][None,:]-obs[prefix+'noise_pre_Q'],obs[prefix+'charge_identity_residual'],label=prefix+' assembly Q')
            counts[prefix.rstrip('_')]=dict(rows=rows,parents=n,
                new_nonpositive=int((q[2:]<=0).sum()),new_guarded=int((abs(q[2:])<stats['q_floor']).sum()))
        for key,value in readouts_np(real,stats['q_floor']).items():
            close(obs[split+'_REAL_'+key],value,label=split+'REAL_'+key)
        exact(obs[split+'_REAL_bins'],bins_np(q_np(real),stats),split+' REAL bin')
    return counts

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--container',required=True);parser.add_argument('--identities',required=True)
    parser.add_argument('--config',required=True);parser.add_argument('--source-manifest',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--receipt',required=True)
    args=parser.parse_args()
    receipt=Path(args.receipt)
    if receipt.exists():raise FileExistsError('Verification receipt already exists')
    started=dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        cfg,manifest,complete=check_bindings(args.config,args.source_manifest,args.container,args.identities,Path(args.out))
        torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if args.device.startswith('cuda'):
            limit=min(.95,cfg['gpu_memory_limit_GiB']*1024**3/torch.cuda.get_device_properties(args.device).total_memory)
            torch.cuda.set_per_process_memory_fraction(limit,device=args.device)
        out=Path(args.out);obs=read_arrays(out/'observables.npz');curves=read_arrays(out/'curves.npz')
        theory=read_arrays(out/'theory.npz');bootstrap=read_arrays(out/'bootstrap.npz')
        decision=read(out/'DECISION.json');summary=read(out/'SUMMARY.json')
        stats=read(ROOT/cfg['references']['statistics001']['path'])['loss'];cfg['_stats']=stats
        data=dict(TRAIN=load_training(args.container,args.identities),DEV=load_validation(args.container,args.identities))
        assert not np.intersect1d(data['TRAIN']['source_ids'],data['DEV']['source_ids']).size
        assert not np.intersect1d(data['TRAIN']['real_ids'],data['DEV']['real_ids']).size
        assert not np.intersect1d(data['TRAIN']['real_groups'],data['DEV']['real_groups']).size
        actual_counts=check_actual_counts(cfg,complete,summary,data)
        observed_counts=check_observables(obs,cfg,stats,data);check_real_curves(curves,obs,data)
        noise_state,noise,extension,models=load_models(cfg,args.device)
        frozen_states={name:dictionary_sha(model.state_dict()) for name,model in dict(noise005=noise,**models).items()}
        global_rng=torch.get_rng_state().clone()
        expected=independent_theory(noise_state,cfg)
        theory_numeric_checks=check_theory_numeric(theory,expected)
        verified=check_bootstrap(bootstrap,obs,cfg)
        counts,leakage,engineering_checks=replay_all(cfg,data,obs,curves,out,args.device,noise,extension,models)
        diagnostic=replay_noise_diagnostic(cfg,theory,out,args.device,noise,extension)
        derived=independent_decision(cfg,obs,verified,theory,leakage,engineering_checks)
        for name,item in engineering_checks.items():assert summary['engineering'][name]==item,name
        for name in ('prefix_pre_bitwise','prefix_post_bitwise','shared_scale','legacy_assembly_bitwise',
            'response_prefix_across_hypotheses_bitwise','stream_pairing','global_RNG_untouched_by_generation',
            'frozen_states','source_coverage','count_weights_only','single_ADC'):
            assert summary['engineering'][name]==dict(passed=True,failed_checks=0),name
        assert torch.equal(global_rng,torch.get_rng_state()),'Global torch RNG changed during explicit-input replay'
        assert frozen_states=={name:dictionary_sha(model.state_dict()) for name,model in dict(noise005=noise,**models).items()}
        assert set(decision['engineering_failures'])==set(derived['engineering_failures'])
        assert decision['engineering_pass']==(not derived['engineering_failures'])
        for key in ('lower_above','upper_above','undefined_intervals','all_fixed_bins_supported'):
            assert decision[key]==derived[key],key
        assert derived['outcome']==decision['outcome']==complete['decision_outcome']
        close(decision['noise_tail_variance_ratio'],derived['noise_tail_variance_ratio'],label='decision noise variance ratio')
        assert decision['TRAIN_only'] and decision['DEV_descriptive_only'] and not decision['scientific_acceptance']
        assert decision['response_updates_authorized_here']==0
        assert decision['next_response_protocol_allowed']==(derived['outcome']=='sensitivity_within_family')
        assert summary['updates']==complete['updates']==dict(noise=0,response=0,classifier=0,GMM_fits=0)
        # All immutable source/input/payload hashes are checked again after replay.
        check_bindings(args.config,args.source_manifest,args.container,args.identities,out)
        write_json(receipt,dict(status='passed',passed=True,started_UTC=started,completed_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
            run_id=cfg['run_id'],configuration_sha256=sha(args.config),source_manifest_sha256=sha(args.source_manifest),
            verifier_sha256=sha(Path(__file__)),complete_sha256=sha(out/'COMPLETE.json'),
            checks=['all frozen bindings','all SOURCE/REAL identities, roles and original count weights','all readout algebra and bin migrations',
                'independent NumPy conditional covariance and readout moments','all paired parent bootstrap draws and decision',
                'all CPU random streams and all generated waveform hashes','all population/fixed-bin/parent variance curves',
                'all 8192 fixed noise diagnostic draws','explicit first128 report/replay artifacts'],
            observed_counts=observed_counts,original_execution_counts=actual_counts,replay_counts=counts,noise_diagnostic_replay_counts=diagnostic,
            signal_leakage=leakage,independent_decision=derived,theory_numeric_checks=theory_numeric_checks,
            independent_ADC_code_checks=dict(scope='all five views in every split/arm/chunk',integer_code_tolerance=0,
                canonical_voltage_absolute_tolerance=float(4*np.finfo(np.float64).eps),
                code_rule='NumPy rint((clipped preADC+0.5)*16383) equals recovered waveform integer codes',
                baseline_change_reference='original digitize(prefix,ste=False) on the same device'),
            updates=dict(noise=0,response=0,classifier=0,GMM_fits=0),
            scope='Frozen-input numerical reproduction; not independent acquisition or physical noise validation'))
        print(json.dumps(dict(status='passed',receipt=str(receipt),outcome=derived['outcome'])),flush=True)
    except Exception as error:
        write_json(receipt,dict(status='failed',started_UTC=started,failed_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
            error_type=type(error).__name__,error=str(error),verifier_sha256=sha(Path(__file__))))
        raise

if __name__=='__main__':main()
