"""Deterministic artificial charge rows only; independent scalar reference builder."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;POST=HERE.parent/'postrun_independent_01'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def fields_scalar(values):
    if values.ndim==2:values=values[:,:,None]
    n,d,c=values.shape;out={k:np.zeros((n,d),dtype=t) for k,t in [('absmax',np.float64),('absmax_coordinate',np.int64),
        ('relative_max_nonzero_reference',np.float64),('ordered_float64_ULP_max',np.uint64),('ULP_max_coordinate',np.int64),
        ('nonzero_cells',np.int64),('nonzero_with_zero_reference',np.int64)]}
    changes=[];squares=[];bits=0;nonzero_rows=0
    def ordered(x):
        b=struct.unpack('<Q',struct.pack('<d',float(x)))[0]
        return (2**64-1-b) if b>=2**63 else b+2**63
    for p in range(n):
        affected=False
        for j in range(d):
            absolute=[];relative=[];ulp=[];nz=zero_ref=0
            for k in range(c):
                left,right=float(values[p,j,k]),float(values[p,0,k]);diff=left-right
                absolute.append(abs(diff));relative.append(abs(diff)/abs(right) if right else 0.);ulp.append(abs(ordered(left)-ordered(right)))
                squares.append(diff*diff);nz+=diff!=0;zero_ref+=(right==0 and diff!=0);bits+=struct.pack('<d',left)!=struct.pack('<d',right)
            out['absmax'][p,j]=max(absolute);out['absmax_coordinate'][p,j]=absolute.index(max(absolute))
            out['relative_max_nonzero_reference'][p,j]=max(relative);out['ordered_float64_ULP_max'][p,j]=max(ulp)
            out['ULP_max_coordinate'][p,j]=ulp.index(max(ulp));out['nonzero_cells'][p,j]=nz;out['nonzero_with_zero_reference'][p,j]=zero_ref
            affected|=nz>0;nonzero_rows+=nz>0
        if affected:changes.append(p)
    return out,dict(maximum=float(out['absmax'].max()),RMS=math.sqrt(math.fsum(squares)/values.size),
        cells=int(out['nonzero_cells'].sum()),rows=nonzero_rows,parents=changes,bits=bits)


def fixture(delta=2**-25,mu_offset=0.):
    n,d=3,4;A=np.zeros((n*d,384));A[:,126]=np.repeat([1.,2.,3.],d);A[:,127]=.5;A[1,126]+=delta
    mu=A[:,110:366].sum(1).reshape(n,d);mu[0,1]+=mu_offset
    q=mu+np.array([1/64,-1/128,1/128,-1/64]);a,b=2**-12,2**-16
    variance=lambda r:math.fsum((float(x)-math.fsum(map(float,r))/len(r))**2 for x in r)/(len(r)-1)
    mean=lambda r:math.fsum(map(float,r))/len(r)
    vm=np.array([[a*float(x)+b*float(x)*float(x) for x in row] for row in mu]);r=q-mu
    qv=np.array([variance(row) for row in q]);mv=np.array([variance(row) for row in mu]);rv=np.array([variance(row) for row in r])
    arrays=dict(A_signal_pre384_V=A,Q_mu_draws_Vns=mu,q_sig_draws_Vns=q,Q_mu_Vns=np.array([mean(row) for row in mu]),
        sample_mean_Vns=np.array([mean(row) for row in q]),sample_variance_Vns2=qv,conditional_mean_MCSE_Vns=np.sqrt(qv/d),sample_mean_MCSE_Vns=np.sqrt(qv/d),
        theory_conditional_variance_Vns2=np.array([mean(row) for row in vm]),theory_conditional_variance_draws_Vns2=vm,
        theory_expected_sample_variance_Vns2=np.array([mean(row) for row in vm])+mv,theory_sample_mean_MCSE_Vns=np.array([math.sqrt(math.fsum(row))/d for row in vm]),
        Q_mu_numeric_sample_variance_Vns2=mv,Q_mu_numeric_range_Vns=np.array([max(row)-min(row) for row in mu]),
        draw_centered_residuals_Vns=r,draw_centered_residual_mean_Vns=np.array([mean(row) for row in r]),
        draw_centered_residual_second_moment_Vns2=np.array([math.fsum(float(x)**2 for x in row)/d for row in r]),
        draw_centered_residual_sample_variance_Vns2=rv,draw_centered_sample_mean_MCSE_Vns=np.sqrt(rv/d),
        source_parent_row=np.arange(n),A_source_parent_row=np.repeat(np.arange(n),d),draw_index=np.tile(np.arange(d),n),source_parent_ids=np.arange(n)+900)
    for name in ('source_input_sha256_uint8','forward_float64_input_sha256_uint8'):
        arrays[name]=np.repeat(np.array([list(hashlib.sha256((name+str(p)).encode()).digest()) for p in range(n)],np.uint8),d,axis=0)
    af,ac=fields_scalar(A.reshape(n,d,384));mf,mc=fields_scalar(mu)
    arrays.update({'A_numeric_repeat_'+k:v for k,v in af.items()});arrays.update({'Q_mu_numeric_repeat_'+k:v for k,v in mf.items()})
    cpu=A[:,110:366].sum(1).reshape(n,d);u=np.finfo(float).eps/2;g=255*u/(1-255*u)
    bound=np.nextafter(2*g/(1-g)*np.abs(A[:,110:366]).sum(1).reshape(n,d)+512*np.finfo(float).smallest_subnormal,np.inf)
    arrays.update(Q_mu_CPU_recomputed_draws_Vns=cpu,Q_mu_GPU_minus_CPU_draws_Vns=mu-cpu,Q_mu_CPU_GPU_reduction_bound_Vns=bound,
        Q_mu_repeat_bound_from_A_gate_Vns=256e-7+bound+bound[:,:1],charge_rng_initial_state=np.arange(12,dtype=np.uint8),charge_rng_final_state=np.arange(12,dtype=np.uint8)+1)
    diagnostic=dict(bank=0,parent_count=n,actual_rows=n*d,draws_per_parent=d,numerical_tolerance_V=1e-7,tolerance_source='same_frozen_parameter_baseline_replay_1e-7_V',
        within_numerical_tolerance=ac['maximum']<=1e-7,actual_max_abs_residual_V=ac['maximum'],actual_RMS_residual_V=ac['RMS'],
        nonzero_cells=ac['cells'],nonzero_rows=ac['rows'],different_float64_bit_cells=ac['bits'],affected_source_parent_rows=ac['parents'],
        affected_source_ids=(np.arange(n)+900)[ac['parents']].tolist(),Q_mu_CPU_GPU_within_derived_bound=True,Q_mu_CPU_GPU_absmax_Vns=float(np.abs(mu-cpu).max()))
    cv=np.sqrt(a/mu+b)
    semantic=dict(bank=0,a_Vns=a,b=b,actual_A_expected_increment_RMS_V=math.sqrt(math.fsum(float((A[i]*A[i]).sum())*(a/float(mu.ravel()[i])+b) for i in range(n*d))/(n*d*384)),
        conditional_CV_min=float(cv.min()),conditional_CV_mean=float(cv.mean()),conditional_CV_max=float(cv.max()),
        finite_MC_mean_error_RMS_Vns=math.sqrt(mean(np.square(arrays['sample_mean_Vns']-arrays['Q_mu_Vns']))),
        finite_MC_mean_MCSE_RMS_Vns=math.sqrt(mean(np.square(arrays['sample_mean_MCSE_Vns']))))
    return arrays,diagnostic,semantic


def main():
    out=Path(sys.argv[1]).resolve();assert out.drive.upper() in ('E:','D:') and not out.exists();out.mkdir(parents=True)
    (out/'tmp').mkdir();os.environ.update(TEMP=str(out/'tmp'),TMP=str(out/'tmp'),TMPDIR=str(out/'tmp'))
    started=time.monotonic();w=module(POST/'readback_saved_arrays.py','charge016_artificial');w.np=np;w.P,w.R,w.N=3,4,12
    original=out/'empty_original';original.mkdir();audit=w.Audit(SimpleNamespace(run_root=original,out=out/'audit',artificial_only=True));audit.guard=lambda:None
    checks=[];negative=[]
    def verify(case,parent_ids=None,hashes=None):return w.verify_charge_arrays(*case,bank=0,equal=audit.equal,parent_ids=parent_ids,input_hashes=hashes)
    good=fixture();parents,hashes,report=verify(good);assert report['actual_A_repeat_max_abs_V']>0 and report['original_SOURCE_rehashed'] is False
    assert np.array_equal(good[0]['Q_mu_draws_Vns'],good[0]['Q_mu_CPU_recomputed_draws_Vns'])
    checks.append('independent_scalar_reference_nonidentical_mu_and_ddof0_ddof1_moments_pass')
    alternative=copy.deepcopy(good)
    alternative[0]['Q_mu_CPU_recomputed_draws_Vns'][0,0]=np.nextafter(alternative[0]['Q_mu_CPU_recomputed_draws_Vns'][0,0],np.inf)
    alternative[0]['Q_mu_GPU_minus_CPU_draws_Vns']=alternative[0]['Q_mu_draws_Vns']-alternative[0]['Q_mu_CPU_recomputed_draws_Vns']
    alternative[1]['Q_mu_CPU_GPU_absmax_Vns']=float(np.abs(alternative[0]['Q_mu_GPU_minus_CPU_draws_Vns']).max())
    verify(alternative)
    checks.append('legal_one_ULP_CPU_reduction_variation_uses_derived_bound_and_saved_residual_algebra')
    def reject(name,change):
        case=copy.deepcopy(good);change(case)
        try:verify(case)
        except (ValueError,AssertionError):negative.append(name);return
        raise AssertionError('expected reject '+name)
    def reject_case(name,case,**kwargs):
        try:verify(case,**kwargs)
        except (ValueError,AssertionError):negative.append(name);return
        raise AssertionError('expected reject '+name)
    reject_case('above013_A_gate',fixture(delta=2e-7))
    reject_case('canonical_wrong_mu_coherent_other_moments',fixture(mu_offset=2**-10))
    for key in ('source_parent_row','A_source_parent_row','draw_index'):reject('bad_'+key,lambda c,key=key:c[0].__setitem__(key,np.roll(c[0][key],1)))
    reject('A_float32',lambda c:c[0].__setitem__('A_signal_pre384_V',c[0]['A_signal_pre384_V'].astype(np.float32)))
    reject('duplicate_ids',lambda c:c[0]['source_parent_ids'].__setitem__(1,c[0]['source_parent_ids'][0]))
    for key in ('Q_mu_draws_Vns','q_sig_draws_Vns'):
        for value in (0.,-1.,np.nan,np.inf):reject(key+str(value),lambda c,key=key,value=value:c[0][key].__setitem__((0,1),value))
    for key in ('source_input_sha256_uint8','forward_float64_input_sha256_uint8'):
        reject('within_parent_hash_'+key,lambda c,key=key:c[0][key].__setitem__((1,0),c[0][key][1,0]^1))
    changed=copy.deepcopy(hashes);changed[0][0,0]^=1
    reject_case('cross_bank_saved_digest_mismatch',good,hashes=changed)
    reject_case('cross_bank_ID_mismatch',good,parent_ids=parents+1)
    reject('wrong_v',lambda c:c[0]['theory_conditional_variance_draws_Vns2'].__setitem__((0,0),c[0]['theory_conditional_variance_draws_Vns2'][0,0]*1.01))
    reject('omit_mu_numeric_variance',lambda c:c[0].__setitem__('theory_expected_sample_variance_Vns2',c[0]['theory_conditional_variance_Vns2'].copy()))
    zero=fixture(delta=0.);zero[0]['Q_mu_numeric_sample_variance_Vns2'][0]=1e-300
    reject_case('zero_moment_is_strict_zero',zero)
    reject('bound_changed',lambda c:c[0].__setitem__('Q_mu_CPU_GPU_reduction_bound_Vns',c[0]['Q_mu_CPU_GPU_reduction_bound_Vns']*1.01))
    reject('saved_CPU_sum_outside_bound',lambda c:c[0]['Q_mu_CPU_recomputed_draws_Vns'].__setitem__((0,0),c[0]['Q_mu_CPU_recomputed_draws_Vns'][0,0]+1e-5))
    reject('wrong_saved_residual_algebra',lambda c:c[0]['Q_mu_GPU_minus_CPU_draws_Vns'].__setitem__((0,0),1e-16))
    reject('wrong_rng_state_dtype',lambda c:c[0].__setitem__('charge_rng_initial_state',c[0]['charge_rng_initial_state'].astype(np.int64)))
    reject('nonpositive_a',lambda c:c[2].__setitem__('a_Vns',0.))
    checks.append('fixed_A_point_gate_canonical_bound_lineage_positive_q_mu_coefficients_and_moment_negative_cases')
    # Actual new charge-bank loader: one artificial NPZ open, no SOURCE open.
    directory=original/'runs/primary/artifacts/seed0_Q1/initial_evaluation';directory.mkdir(parents=True)
    path=directory/'bank0_CHARGE_PREADC_MOMENTS.npz';np.savez_compressed(path,**good[0])
    good[1]['saved_arrays_sha256']=sha(path);write(directory/'bank0_A_NUMERIC_REPEATS.json',good[1]);write(directory/'CHARGE_SEMANTICS.json',dict(banks=[good[2]]))
    audit.touched[path.relative_to(original).as_posix()]=dict(sha256=sha(path),bytes=path.stat().st_size)
    opens=[];saved_load=np.load
    def only_fixture_load(p,*args,**kwargs):
        assert Path(p)==path;opens.append(str(p));return saved_load(p,*args,**kwargs)
    np.load=only_fixture_load
    before={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in good[0].items()}
    try:audit.op('charge_bank_saved_statistics',dict(charge_npz_files_decoded=1,charge_A_rows_verified=12,charge_A_float64_cells_verified=12*384,
        charge_mu_draws_verified=12,charge_q_draws_verified=12,charge_parent_moment_records_verified=3),lambda:audit.charge_bank('seed0_Q1','initial_evaluation',0))
    finally:np.load=saved_load
    assert len(opens)==1 and audit.entered==audit.completed==1 and not audit.inflight
    assert before=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in good[0].items()}
    assert audit.charge_banks[0]['original_SOURCE_rehashed'] is False and audit.charge_banks[0]['charge_r8_derived'] is False
    checks.append('actual_charge_loader_one_artificial_NPZ_open_readonly_no_source_read_no_RNG_or_model')
    old=ast.parse((ROOT/'receipts/wp3_015_20261015/postrun_independent_01/readback_saved_arrays.py').read_bytes());new=ast.parse((POST/'readback_saved_arrays.py').read_bytes())
    def node(tree,cls,name):
        scope=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
        return next(n for n in scope if isinstance(n,ast.FunctionDef) and n.name==name)
    unchanged=[]
    for cls,name in [(None,n) for n in ('decode_window','lineage','wave_moments')]+[('Audit',n) for n in ('equal','member','load_baselines','one_bank','pair','op','journal','state','guard')]:
        assert ast.dump(node(old,cls,name),include_attributes=False)==ast.dump(node(new,cls,name),include_attributes=False),(cls,name)
        unchanged.append((cls+'.' if cls else '')+name)
    assert sha(POST/'saved_array_kernels.py')=='fe1366f35f36e97974f1a4037a36ad858c03dd8453957c1b45b65080f0fee3ac'
    assert 'torch' not in sys.modules and 'n01' not in sys.modules
    checks.append('mature_savedADC_pair_statistics_durable_methods_AST_and_kernel_bytes_unchanged')
    receipt=dict(schema='n01/wp3-016-readback-charge-artificial-engineering/v1',passed=True,artificial_only=True,checks=checks,
        negative_cases=negative,negative_count=len(negative),unchanged_mature_methods=unchanged,elapsed_seconds=time.monotonic()-started,
        actual_artificial_charge_NPZ_open=1,original_raw_or_CP_reads=0,NN_AD_RNG_training_GPU_SSH=0,
        fixture_uses_no_producer_numeric_functions=True,synthetic_parent_rows=3,synthetic_draw_rows=12,
        worker=dict(bytes=(POST/'readback_saved_arrays.py').stat().st_size,sha256=sha(POST/'readback_saved_arrays.py')),
        kernel=dict(bytes=(POST/'saved_array_kernels.py').stat().st_size,sha256=sha(POST/'saved_array_kernels.py')),
        checker=dict(bytes=Path(__file__).stat().st_size,sha256=sha(Path(__file__))))
    write(out/'ENGINEERING_RECEIPT.json',receipt);print(json.dumps(dict(passed=True,checks=len(checks),negatives=len(negative),receipt_sha256=sha(out/'ENGINEERING_RECEIPT.json'))))


if __name__=='__main__':main()
