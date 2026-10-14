"""New A storage semantics with manufactured arrays, no Torch/model/SSH.

Exercises the actual Runtime writer while old scientific metric calls are
explicitly mocked. These artificial reports are not population evaluations.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest import mock

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve()
    if out.drive.upper() not in ('E:','D:') or out.exists():raise ValueError('Fresh E/D fixture required')
    out.mkdir(parents=True)
    import numpy as np
    import run_wp1_013 as driver
    import wp1_evaluation as evaluation
    from wp1_ops import Store
    checks=[]
    def passed(name,**detail):checks.append(dict(name=name,passed=True,**detail))
    n,r=3,4;parents=np.repeat(np.arange(n),r);draw=np.tile(np.arange(r),n);ids=np.array(['toy0','toy1','toy2'])
    base=np.zeros((n*r,384),np.float64);base[:,100:]=.125
    base[-3,303]=np.nextafter(base[-3,303],np.inf)
    diagnostic=driver.A_numerical_residual(base,parents,draw,ids,draws=r,bank=0,chunk_rows=1)
    assert diagnostic['within_numerical_tolerance'] and not diagnostic['exact_float64_bits_equal']
    assert diagnostic['nonzero_cells']==1 and diagnostic['affected_source_parent_rows'].tolist()==[2]
    assert diagnostic['actual_max_abs_residual_V']==np.spacing(.125) and diagnostic['numerical_tolerance_V']==1e-7
    passed('one_float64_ULP_recorded_nonzero_and_accepted_under_existing_numeric_replay_limit')
    shifted=base.copy();shifted[-2,304]=np.nextafter(shifted[-2,304],np.inf)
    cross=driver.A_numerical_residual(shifted,parents,draw,ids,draws=r,bank=1,other=base,other_parent_rows=parents,other_draw_indices=draw,chunk_rows=1)
    assert cross['nonzero_cells']==1 and cross['actual_max_abs_residual_V']==np.spacing(.125) and cross['parent_draw_identity_verified']
    passed('same_identity_cross_bank_residual_preserved_and_reported')
    for altered_p,altered_d in ((np.roll(parents,1),draw),(parents,np.roll(draw,1))):
        try:driver.A_numerical_residual(base,altered_p,altered_d,ids,draws=r,bank=0)
        except ValueError:pass
        else:raise AssertionError('Identity reorder accepted')
    try:driver.A_numerical_residual(base,parents,draw,ids,draws=r,bank=1,other=shifted,other_parent_rows=np.roll(parents,1),other_draw_indices=draw)
    except ValueError:pass
    else:raise AssertionError('Cross-bank wrong identity accepted')
    passed('parent_draw_and_cross_bank_negative_identity_reorders_rejected')
    p=json.loads((ROOT/'wp1_013_config.json').read_text('utf8'));p['artificial_only']=True
    p['RR'].update(rows_per_side=147,groups_per_side=3,histogram_each_side={'48':1,'49':1,'50':1})
    groups=np.repeat(np.arange(6),[48,48,49,49,50,50])
    # Literal toy ADC cells, no call to any ADC operation.
    codes=np.arange(384,dtype=np.uint16)[None]+10000
    adc=np.broadcast_to(codes.astype(np.float64)/16383-.5,(n*r,384)).copy();peaks=np.full(n*r,126,np.int64)
    prefix=adc[:,:100];mu=prefix.mean(1);wave=(adc-mu[:,None])[:,110:366]
    obs=dict(window_V=wave,baseline100=prefix,peak_index=peaks,adc_carrier=adc,diagnostic__baseline_mean_V=mu)
    def fake_population(w,b,**kwargs):return dict(artificial_metric_mock=True,repeated_parent={},passed_wave_bits=w.view(np.uint64))
    def fake_features(w,b,floor):return dict(Q_Vns=np.asarray(w).sum(1)),{}
    def fake_moments(w,f,parent,d,b,signal):
        # Actual independent covariance construction from the retained raw rows;
        # no model metric/loss or parameter operation is performed.
        order=np.lexsort((d,b,parent));repeats=len(w)//n;result=dict(artificial_metric_mock=True,draws_per_parent=repeats)
        if signal is not None:
            grouped=np.asarray(signal)[order].reshape(n,repeats,384);mean=grouped.mean(1);centered=grouped-mean[:,None]
            result.update(preADC_mean=mean,preADC_covariance_diagonal=(centered**2).sum(1)/(repeats-1))
        return result
    def runtime(target,first,second):
        store=Store(target/'artifacts',p,{},artificial=True);ev=target/'evaluation';ev.mkdir()
        obj=driver.Runtime.__new__(driver.Runtime)
        obj.p,obj.store,obj.out,obj.info=p,store,ev,None
        obj.data=dict(source=np.zeros((n,256)),source_ids=ids,real=np.zeros((len(groups),256)),real_baseline100=np.zeros((len(groups),100)),
            real_groups=groups,real_observation_flags={})
        for key in driver.REAL_METADATA:obj.data[key]=np.zeros(len(groups),np.int64)
        obj.data['real_ids']=np.arange(len(groups))
        obj.real_features=None;obj.loss=type('ToyMockLoss',(),{'q_floor':.001})()
        obj.context=None;obj.component=lambda *args,**kwargs:dict(artificial_metric_mock=True,values={'wave':0.})
        obj.references=[dict(base=x,signal=x+.0001,parent=parents,draw=draw,**{arm:{h:obs for h in driver.HYPOTHESES} for arm in ('A','theta')}) for x in (first,second)]
        return obj
    with mock.patch.object(evaluation,'evaluate_hard_population',side_effect=fake_population),mock.patch.object(evaluation,'observable_features',side_effect=fake_features),mock.patch.object(evaluation,'repeated_parent_statistics',side_effect=fake_moments):
        input_copies=[x.copy() for x in (base,shifted)]
        actual=runtime(out/'healthy_actual_writer',base,shifted);actual.complete_evaluations()
        for bank,original in enumerate((base,shifted)):
            with np.load(actual.store.root/('REFERENCE_%d.npz'%bank),allow_pickle=False) as z:
                saved=z['A_signal_pre384'];assert saved.shape==(12,384) and saved.dtype==np.float64
                assert np.array_equal(saved.view(np.uint64),original.view(np.uint64)) and 'A_signal_per_parent384' not in z
                assert np.array_equal(z['source_parent_ids'],ids) and np.array_equal(z['source_parent_row'],parents)
            group=saved.reshape(n,r,384);center=group-group.mean(1,keepdims=True);cov=np.einsum('prt,pru->ptu',center,center)/(r-1)
            expected=original.reshape(n,r,384);expected=expected-expected.mean(1,keepdims=True)
            assert np.array_equal(cov,np.einsum('prt,pru->ptu',expected,expected)/(r-1)) and np.any(cov[2])
        assert all(np.array_equal(a.view(np.uint64),b.view(np.uint64)) for a,b in zip((base,shifted),input_copies))
        encoding=json.loads((actual.store.root/'REFERENCE_ENCODING.json').read_text('utf8'))
        assert encoding['A_signal_broadcast_across_banks_and_draws'] is False and encoding['A_actual_float64_per_draw_rows_preserved'] is True
        passed('actual_Runtime_fullA_float64_rows_roundtrip_and_complete_covariance_reconstructable_without_broadcast')
        residual=json.loads((actual.store.root/'A_NUMERIC_CROSS_BANK.json').read_text('utf8'))
        assert residual['nonzero_cells']==1 and residual['within_numerical_tolerance']
        passed('actual_writer_cross_bank_residual_and_encoding_metadata')
        excessive=np.zeros((12,384),np.float64);excessive[:,100:]=.125;excessive[-3,303]+=.0000003
        bad=runtime(out/'over_limit',excessive,shifted)
        try:bad.complete_evaluations()
        except ValueError as error:assert 'numerical repeat residual exceeds' in str(error)
        else:raise AssertionError('Above1e-7 not rejected')
        state=json.loads((bad.store.root/'A_NUMERIC_REPEATS_0.json').read_text('utf8'));assert not state['within_numerical_tolerance']
        with np.load(bad.store.root/'REFERENCE_0.npz',allow_pickle=False) as z:assert np.array_equal(z['A_signal_pre384'].view(np.uint64),excessive.view(np.uint64))
        passed('above_1e7_V_numeric_limit_rejected_after_actual_residual_and_raw_reference_saved')
        bank0=np.zeros((12,384),np.float64);bank0[:,100:]=.125;bank1=bank0.copy();bank1[:,100:]+=.0000002
        badbank=runtime(out/'cross_bank_over_limit',bank0,bank1)
        try:badbank.complete_evaluations()
        except ValueError as error:assert 'cross-bank numerical residual exceeds' in str(error)
        else:raise AssertionError('Cross-bank above1e-7 not rejected')
        state=json.loads((badbank.store.root/'A_NUMERIC_CROSS_BANK.json').read_text('utf8'));assert not state['within_numerical_tolerance']
        for bank,original in enumerate((bank0,bank1)):
            with np.load(badbank.store.root/('REFERENCE_%d.npz'%bank),allow_pickle=False) as z:assert np.array_equal(z['A_signal_pre384'].view(np.uint64),original.view(np.uint64))
        passed('cross_bank_above_limit_rejected_after_both_actual_raw_banks_preserved')
    assert 'torch' not in sys.modules
    source_names=('scripts/run_wp1_013.py','scripts/wp1_ops.py','scripts/wp1_controller.py')
    receipt=dict(schema='n01/wp1-A-storage-artificial/v1',status='PASS',artificial_only=True,passed_checks=len(checks),checks=checks,
        old_scientific_metrics_explicitly_mocked=True,actual_saved_data_decodes=0,actual_SOURCE_REAL_decodes=0,
        checkpoint_loads=0,Torch_imported=False,NN_rows=0,ADC_calls=0,AD_queries=0,optimizer_or_training_steps=0,SSH_calls=0,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names})
    (out/'ENGINEERING_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',passed_checks=len(checks),source_sha256=receipt['source_sha256'])))

if __name__=='__main__':main()
