"""Artificial-only local fix02 checks. No private arrays, checkpoints or SSH."""
from pathlib import Path
import ast
import copy
import hashlib
import itertools
import json
import math
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
import run_wp3_016 as d
import check_model010 as mf
import check_mean_random009 as fixture
from n01.model014 import FrozenDeterministicVoltageShape014
from n01.model010 import tensor_state_digest010
from n01.train_shape010 import cpu_tree
from n01.losses014 import fit_integral_statistics014
from wp1_ops import sha,write_new


def reject(fn,text):
    try:fn()
    except (ValueError,RuntimeError) as e:
        assert text in str(e),str(e);return str(e)
    raise AssertionError('expected rejection '+text)


def main():
    out=Path(sys.argv[1]).resolve();assert out.drive.upper() in ('E:','D:') and not out.exists();out.mkdir(parents=True)
    (out/'tmp').mkdir();os.environ.update(TEMP=str(out/'tmp'),TMP=str(out/'tmp'),TMPDIR=str(out/'tmp'),MPLCONFIGDIR=str(out/'tmp'))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic();checks=[];here=Path(__file__).resolve().parent
    manifest=json.loads((ROOT/'WP3_SOURCE_MANIFEST.json').read_bytes());original={n:sha(ROOT/n) for n in manifest['files']}
    assert len(original)==105 and all(original[n]==v['sha256'] for n,v in manifest['files'].items())
    old=ast.parse((here/'before/run_wp3_016.py').read_bytes());new=ast.parse((ROOT/'scripts/run_wp3_016.py').read_bytes())
    def funcs(tree):return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    left,right=funcs(old),funcs(new)
    for name,node in left.items():
        if name=='Engine016':
            for method in node.body:
                if isinstance(method,ast.FunctionDef) and method.name!='evaluate':
                    other=next(m for m in right[name].body if isinstance(m,ast.FunctionDef) and m.name==method.name)
                    assert ast.dump(method,include_attributes=False)==ast.dump(other,include_attributes=False),method.name
        else:assert ast.dump(node,include_attributes=False)==ast.dump(right[name],include_attributes=False),name
    oldc=ast.parse((here/'before/wp3_016_controller.py').read_bytes());newc=ast.parse((ROOT/'scripts/wp3_016_controller.py').read_bytes())
    for tree in (oldc,newc):
        next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets)).value=ast.Constant('NORMALIZED_REMOTE')
    assert ast.dump(oldc,include_attributes=False)==ast.dump(newc,include_attributes=False)
    assert sha(ROOT/'src/n01/model016.py')=='04c63f91ea5405937a940a9aee6eaed876db79149c857bfbe6ed8cfe6dcb138b'
    checks.append(dict(name='only_evaluation_helpers_evaluate_and_REMOTE_changed_all_scientific_AST_equal',passed=True))

    source=np.zeros((3,384),np.float64);source[:,126]=[1.,2.,3.];source[:,127]=.5
    parent=np.repeat(np.arange(3),4);draw=np.tile(np.arange(4),3);ids=np.arange(3)+200
    raw=source[parent].copy();forward=raw.copy();sh,fh=d.source_input_evidence016(source,raw,forward,parent,draw,draws=4,start=0)
    def bank(name,A,mu=None):
        target=out/name;target.mkdir();actual=A[:,110:366].sum(1).reshape(3,4) if mu is None else mu
        q=actual*np.exp(np.array([-.01,.02,-.02,.01])[None,:]);a,b=.0003,.00004
        before=[hashlib.sha256(v.tobytes()).hexdigest() for v in (A,actual,q)]
        result=d.save_charge_bank016(target,0,A=A,q=q,mu=actual,a=a,b=b,parent=parent,draw=draw,source_ids=ids,
            source_hashes=sh,forward_hashes=fh,input_dtype=source.dtype.str,extra_arrays={})
        assert before==[hashlib.sha256(v.tobytes()).hexdigest() for v in (A,actual,q)]
        with np.load(target/'bank0_CHARGE_PREADC_MOMENTS.npz',allow_pickle=False) as saved:
            assert np.array_equal(saved['A_signal_pre384_V'].view(np.uint64),A.view(np.uint64))
            assert np.array_equal(saved['Q_mu_draws_Vns'].view(np.uint64),actual.view(np.uint64))
            assert np.array_equal(saved['theory_conditional_variance_draws_Vns2'].view(np.uint64),(a*actual+b*actual**2).view(np.uint64))
        return result
    A=raw.copy();A[1,126]=np.nextafter(A[1,126],np.inf);mu=A[:,110:366].sum(1).reshape(3,4)
    guard=next(n for n in ast.walk(left['Engine016']) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and any(isinstance(v,ast.Constant) and v.value=='016 per-parent A canonical Q changed across draws' for v in n.value.args))
    code=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'old_exact_guard','exec')
    reject(lambda:exec(code,{'np':np,'mu':mu,'require':d.require}),'canonical Q changed')
    _,diag=bank('one_ULP_saved_and_passed_old_gate',A)
    assert not diag['exact_numeric_equal'] and diag['within_numerical_tolerance']
    A=raw.copy();A[1,126]+=5e-8;_,diag=bank('below_existing_1e7_gate',A)
    assert 0<diag['actual_max_abs_residual_V']<=1e-7
    checks.append(dict(name='old_exact_guard_reproduced_by_one_ULP_new_preservation_accepts_only_existing_A_gate',passed=True,
        actual_one_ULP_is_artificial_not_original_failure=True,below_gate_absmax=diag['actual_max_abs_residual_V']))
    A=raw.copy();A[1,126]+=2e-7
    reject(lambda:bank('above_gate_saved_then_failed',A),'frozen013 replay tolerance')
    failed_dir=out/'above_gate_saved_then_failed'
    assert (failed_dir/'bank0_CHARGE_PREADC_MOMENTS.npz').exists()
    assert json.loads((failed_dir/'bank0_A_NUMERIC_REPEATS.json').read_bytes())['within_numerical_tolerance'] is False
    mu=raw[:,110:366].sum(1).reshape(3,4);bad_mu=mu.copy();bad_mu[0,1]+=.001
    reject(lambda:bank('wrong_canonical_mu_saved_then_failed',raw.copy(),bad_mu),'derived float64 sum bound')
    assert (out/'wrong_canonical_mu_saved_then_failed/bank0_A_NUMERIC_REPEATS.json').is_file()
    negatives=2
    for p,di in [(np.roll(parent,1),draw),(parent,np.roll(draw,1))]:
        reject(lambda:d.source_input_evidence016(source,raw,forward,p,di,draws=4,start=0),'parent/draw identity');negatives+=1
    bad_raw=raw.copy();bad_raw[0,0]=-0.
    reject(lambda:d.source_input_evidence016(source,bad_raw,forward,parent,draw,draws=4,start=0),'original SOURCE input bytes');negatives+=1
    bad_forward=forward.copy();bad_forward[0,126]=np.nextafter(1.,np.inf)
    reject(lambda:d.source_input_evidence016(source,raw,bad_forward,parent,draw,draws=4,start=0),'float64 forward input bytes');negatives+=1
    s32=source.astype(np.float32);r32=s32[parent]
    s32hash,f64hash=d.source_input_evidence016(s32,r32,r32.astype(np.float64),parent,draw,draws=4,start=0)
    assert not np.array_equal(s32hash,f64hash)
    for invalid in [0.,-1.,np.nan,np.inf]:
        invalid_mu=mu.copy();invalid_mu[0,1]=invalid
        reject(lambda:d.charge_draw_moments016(mu,invalid_mu,a=.1,b=.1),'strictly positive actual');negatives+=1
    for invalid in [0.,np.nan]:
        invalid_q=mu.copy();invalid_q[0,1]=invalid
        reject(lambda:d.charge_draw_moments016(invalid_q,mu,a=.1,b=.1),'strictly positive actual');negatives+=1
    checks.append(dict(name='above_gate_wrong_canonical_identity_permutations_bytes_and_invalid_mu_q_refused',passed=True,negative_cases=negatives,
        failed_A_and_canonical_banks_keep_actual_NPZ_and_diagnostic=True,original_and_forward_dtypes_separately_verified=True))

    mu=np.array([[1.,2.,3.,4.]],np.float64);a,b=.0003,.00004;v=a*mu+b*mu**2
    rows=[]
    for sign in itertools.product([-1.,1.],repeat=4):rows.append(mu[0]+np.sqrt(v[0])*np.array(sign))
    samples=np.array(rows);mom=d.charge_draw_moments016(samples,np.repeat(mu,16,axis=0),a=a,b=b)
    assert abs(samples.var(1,ddof=1).mean()-(v.mean()+mu.var(ddof=1)))<1e-14
    assert abs(samples.mean(1).var(ddof=0)-v.sum()/16)<1e-14
    assert abs(np.square(samples-mu).mean()-v.mean())<1e-14
    assert np.array_equal(mom['Q_mu_draws_Vns'],np.repeat(mu,16,axis=0))
    checks.append(dict(name='all16_independent_sign_states_verify_heterogeneous_mu_moments_without_MC_fit',passed=True,
        expected_sample_variance=float(v.mean()+mu.var(ddof=1)),expected_mean_variance=float(v.sum()/16),expected_centered_second_moment=float(v.mean())))

    # Actual two-trial tiny engine/evaluator; synthetic parents only, original seeds/law.
    baseline=mf.baseline_fixture();frozen=tensor_state_digest010(baseline)
    reference=FrozenDeterministicVoltageShape014(baseline,seed=2026101540);head=cpu_tree(reference.basis_head.state_dict())
    data,stats,_=fixture.artificial_fixture();original_n=len(data['source']);tiny=8
    for name in list(data):
        if isinstance(data[name],np.ndarray) and data[name].shape[0]==original_n:data[name]=data[name][:tiny].copy()
    integral=fit_integral_statistics014(torch.as_tensor(data['real']))
    p=d.settings016();p['G_updates']=1;p['calibration'].update(parents=tiny,chunk_rows=8)
    p['training'].update(source_parents_per_update=4,draws_per_parent=2,real_rows_per_update=8,forward_chunk_rows=4,status_interval=1,checkpoint_steps=[1])
    p['evaluation'].update(parents=tiny,draws=4,chunk_rows=16)
    p['contracts']=dict(tolerance_table=dict(path='synthetic_only',sha256='synthetic'));p['budget_caps']=d.budget_caps016(p)
    store=d.Store016(out/'ARTIFICIAL_BATCH',p,{},artificial=True)
    info=dict(source_id=data['source_ids'],role=data['source_role'],canonical_Q_Vns=data['source'].sum(1),amplitude_384_V=data['source_carrier384'].max(1),
        PC_scores_384_Qnorm=np.zeros((tiny,2)),PC_valid_384_Qnorm=np.ones(tiny,bool),physical_weight_known=np.zeros(tiny,bool))
    rt=d.Runtime016(p,data,baseline,fixture.ArtificialExtension().eval(),stats,integral,store,'cpu',info);cal=rt.calibrate_charge()
    for index,seed in enumerate(p['seed_sets']):
        engine=d.Engine016(rt,seed,head,cal,{'artificial_only':True},store.root/f'seed{index}_Q1')
        initial=engine.evaluate(initial=True);assert len(initial)==18
        engine.step();engine.checkpoint('ARTIFICIAL_final.pt','final');engine.model.assert_frozen_unchanged(deep=True)
        final=engine.evaluate(initial=False);assert len(final)==6
        for path in engine.out.glob('*_evaluation/*_CHARGE_PREADC_MOMENTS.npz'):
            with np.load(path,allow_pickle=False) as saved:
                assert saved['A_signal_pre384_V'].shape==(tiny*4,384) and saved['A_signal_pre384_V'].dtype==np.float64
                mu=saved['Q_mu_draws_Vns'];assert mu.shape==(tiny,4)
                a,b=map(float,(engine.model.theta_a.new_tensor(cal['a_initial_Vns']),engine.model.theta_b.new_tensor(cal['b_initial'])))
                assert np.array_equal(saved['Q_mu_Vns'],mu.mean(1))
                assert np.all(np.abs(saved['Q_mu_GPU_minus_CPU_draws_Vns'])<=saved['Q_mu_CPU_GPU_reduction_bound_Vns'])
                assert len(saved['source_input_sha256_uint8'])==tiny*4
    expected=d.budget_caps016(p);expected['checkpoint_loads']=0
    assert store.counts==expected and not store.inflight and store.entered==store.completed
    counts={suffix:len(list(store.root.glob('seed*/*_evaluation/*'+suffix))) for suffix in ['_HARD_ROWS.npz','_REPORT.npz','_BASELINE100.npz','_CHARGE_PREADC_MOMENTS.npz','_A_NUMERIC_REPEATS.json']}
    assert counts=={'_HARD_ROWS.npz':48,'_REPORT.npz':48,'_BASELINE100.npz':8,'_CHARGE_PREADC_MOMENTS.npz':8,'_A_NUMERIC_REPEATS.json':8},counts
    assert all(sha(ROOT/n)==digest for n,digest in original.items()) and tensor_state_digest010(baseline)==frozen
    checks.append(dict(name='two_actual_tiny_engines_each_initial_and_final_all48HARD48REPORT8baseline8Acharge_arrays',passed=True,
        artificial_parents=tiny,counters=store.counts,files=counts))
    result=dict(schema='n01/wp3-016-fix02-artificial-engineering/v1',passed=True,artificial_only=True,checks=checks,
        negative_cases=negatives,elapsed_seconds=time.monotonic()-started,original105_sources_unchanged=True,
        scientific_CP_NN_AD_random_training=0,private_arrays_read=0,SSH_calls=0,synthetic_updates=2,synthetic_functional_AD_queries=0,
        synthetic_checkpoint_loads=0,sources={name:dict(sha256=sha(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in d.OWNED_FILES})
    write_new(out/'ENGINEERING_RECEIPT.json',result)
    print(json.dumps(dict(passed=True,checks=len(checks),negatives=negatives,seconds=result['elapsed_seconds'],receipt=str(out/'ENGINEERING_RECEIPT.json'))))


if __name__=='__main__':main()
