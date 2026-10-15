"""New 015 semantics only; synthetic arrays/untrained A, never private inputs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
import check_model010 as mf
import check_mean_random009 as fixture
import run_wp3_015 as d
from n01.model014 import FrozenDeterministicVoltageShape014
from n01.model015 import FrozenC1MeanResponse015
from n01.model010 import tensor_state_digest010
from n01.losses014 import fit_integral_statistics014
from n01.train_shape010 import cpu_tree
from wp1_ops import write_new,sha


def fail(fn,text):
    try:fn()
    except (ValueError,RuntimeError) as e:assert text in str(e),str(e);return
    raise AssertionError('expected failure '+text)


def main():
    out=Path(sys.argv[1]).resolve();assert out.drive.upper() in ('E:','D:') and not out.exists();out.mkdir(parents=True)
    (out/'tmp').mkdir();os.environ.update(TEMP=str(out/'tmp'),TMP=str(out/'tmp'))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic();checks=[]
    manifest=json.loads((ROOT/'WP2_SOURCE_MANIFEST.json').read_text())
    protected={n:sha(ROOT/n) for n in manifest['files']}
    assert all(protected[n]==r['sha256'] for n,r in manifest['files'].items())
    x,z=mf.fixture();baseline=mf.baseline_fixture();frozen=tensor_state_digest010(baseline)
    reference=FrozenDeterministicVoltageShape014(baseline,seed=2026101540)
    head=cpu_tree(reference.basis_head.state_dict());rng=torch.get_rng_state().clone()
    m0=FrozenC1MeanResponse015(baseline,head,arm='M0',seed=2026101540,q_reference=3.,q_floor=.01)
    m1=FrozenC1MeanResponse015(baseline,head,arm='M1',seed=2026101540,q_reference=3.,q_floor=.01)
    assert torch.equal(rng,torch.get_rng_state())
    with torch.no_grad():
        ref=reference(x,z);a0=m0(x,z);a1=m1(x,z);disabled=m1(x,z,mean_enabled=False)
    assert all(mf.bytes_equal(ref['signal_pre'],v['signal_pre']) for v in (a0,a1,disabled))
    assert tensor_state_digest010(m0.basis_head)==tensor_state_digest010(m1.basis_head)==tensor_state_digest010(reference.basis_head)
    checks.append(dict(name='same_C1_tensor_reference_zero_and_disabled_mean_bitwise_identity',global_CPU_RNG_preserved=True))

    # Zero initialization must preserve IEEE values without masking its derivative.
    zero=m1(x,z);target=torch.linspace(-.7,1.3,384,dtype=torch.float64)[None]
    functional=(zero['signal_pre']*target).sum();functional.backward()
    gradient=m1.mean_theta.grad.detach().clone();assert torch.isfinite(gradient).all() and bool(gradient.ne(0).all())
    assert all(p.grad is None and not p.requires_grad for p in m1.basis_head.parameters())
    m1.mean_theta.grad=None
    finite=[]
    for j in range(6):
        values=[]
        for sign in (-1,1):
            with torch.no_grad():m1.mean_theta.zero_();m1.mean_theta[j]=sign*1e-5
            value=m1.mean_adapter(x,a1['base_signal_pre_V'])['delta_mean_pre_V'];values.append(float((value*target).sum().detach()))
        finite.append((values[1]-values[0])/2e-5)
    with torch.no_grad():m1.mean_theta.zero_()
    assert torch.allclose(torch.tensor(finite,dtype=gradient.dtype),gradient,rtol=2e-6,atol=2e-9)
    checks.append(dict(name='zero_init_nonzero_all_six_gradients_and_parameter_finite_difference',gradient=gradient.tolist(),finite_difference=finite))

    with torch.no_grad():
        m1.mean_theta.fill_(10.)
        controlled=m1(x,z);off=m1(x,z,mean_enabled=False)
    assert mf.bytes_equal(off['signal_pre'],ref['signal_pre'])
    assert mf.bytes_equal(controlled['random_shape_pre_V'],ref['random_shape_pre_V'])
    assert mf.bytes_equal(controlled['basis_amplitude_V'],ref['basis_amplitude_V'])
    assert bool((controlled['delta_mean_pre_V'][:,:100]==0).all())
    assert float(controlled['mean_zeroQ_basis_charge'].abs().max())<1e-10
    assert float(controlled['mean_RMS_relative_A'].max())<=.1+1e-12
    assert float(controlled['canonical_charge_identity_residual'].abs().max())<1e-10
    assert mf.bytes_equal(controlled['qsig'],controlled['Q_mu_Vns']) and mf.bytes_equal(controlled['qbar'],controlled['Q_mu_Vns'])
    assert mf.bytes_equal(controlled['reference_A_qsig'],ref['qsig']) and bool((controlled['Q_mu_Vns']>=.9*ref['qsig']-1e-12).all())
    with torch.no_grad():
        for slot in (2,4):
            m1.mean_theta.zero_();m1.mean_theta[slot]=.2
            v=m1.mean_adapter(x,ref['base_signal_pre_V']);rms=ref['base_signal_pre_V'].square().mean(1).sqrt()
            assert torch.allclose(v['mean_RMS_V'],.1*torch.tanh(torch.tensor(.2))*rms,rtol=1e-7,atol=1e-12)
        m1.mean_theta.zero_()
    checks.append(dict(name='prefix_native_Q_zero_basis_RMS_trust_and_original_A_random_amplitude',max_relative_RMS=float(controlled['mean_RMS_relative_A'].max())))
    assert sum(v.numel() for v in m0.trainable_parameters())==1288 and sum(v.numel() for v in m1.trainable_parameters())==6
    opt=torch.optim.AdamW(m1.trainable_parameters(),lr=.0002,weight_decay=.0001);m1.assert_optimizer_membership(opt)
    bad=torch.optim.AdamW(list(m1.trainable_parameters())+list(m1.basis_head.parameters()),lr=.0002)
    fail(lambda:m1.assert_optimizer_membership(bad),'exactly')
    m1.train();assert not m1.basis_head.training and not baseline.training
    m0.assert_frozen_unchanged(deep=True);m1.assert_frozen_unchanged(deep=True)
    checks.append(dict(name='6_vs1288_whitelist_and_protected_C1_head'))

    # Independent small algebra reference for the registered logical-group SE.
    real=np.arange(9*256,dtype=np.float64).reshape(9,256)*1e-7;groups=np.array([0,0,1,1,1,2,2,3,3])
    mu=real.mean(0);scores=np.stack([(real[groups==g]-mu).sum(0) for g in np.unique(groups)])
    se=float(np.sqrt((4/3*(scores**2).sum(0)/9**2).mean()))
    mean_bank={};pulse=np.exp(-np.arange(256)/30.)*.01
    for s in range(2):
        for b in range(2):
            for h in d.HYPOTHESES:mean_bank[(s,b,h)]=np.tile(mu+pulse+(2*b-1)*1e-7,(5,1))
    gate=d.population_mean_gate015(real,groups,mean_bank,d.settings015()['mean_evidence'])
    assert gate['all6_passed'] and abs(gate['real_cluster_SE_RMS_V']-se)<1e-16
    broken=dict(mean_bank);broken[(1,1,'slow')]=np.tile(mu-pulse,(5,1))
    gate_bad=d.population_mean_gate015(real,groups,broken,d.settings015()['mean_evidence'])
    assert not gate_bad['all6_passed'] and len(gate_bad['cells'])==6
    checks.append(dict(name='population_gate_ALL6_cluster_sandwich_bank_cosine_and_negative_gate',SE_RMS=se))

    # Actual new engine, actual STE observation and C1 objective: one step/arm.
    data,stats,_=fixture.artificial_fixture();integral=fit_integral_statistics014(torch.as_tensor(data['real']))
    p=d.settings015();p['G_updates']=1
    p['training'].update(source_parents_per_update=4,draws_per_parent=2,real_rows_per_update=8,forward_chunk_rows=4,status_interval=1,checkpoint_steps=[1])
    p['evaluation'].update(parents=128,draws=2,chunk_rows=32,projection_seed=2026101590,
        bank_seeds=[dict(shape=191,noise005=192,extension=193),dict(shape=194,noise005=195,extension=196)])
    p['contracts']=dict(tolerance_table=dict(path='synthetic_only',sha256='synthetic'))
    p['budget_caps']=d.budget_caps015(p)
    store=d.Store015(out/'ARTIFICIAL_TWO_ARM_BATCH',p,{},artificial=True)
    info=dict(source_id=data['source_ids'],role=data['source_role'],canonical_Q_Vns=data['source'].sum(1),
        amplitude_384_V=data['source_carrier384'].max(1),PC_scores_384_Qnorm=np.zeros((128,2)),PC_valid_384_Qnorm=np.ones(128,bool),physical_weight_known=np.zeros(128,bool))
    rt=d.Runtime015(p,data,baseline,fixture.ArtificialExtension().eval(),stats,integral,store,'cpu',info)
    states=[];sampling=[]
    for arm in ('M0','M1'):
        engine=d.Engine015(rt,p['seed_sets'][0],arm,head,{'artificial_only':True},store.root/('seed0_'+arm))
        states.append(tensor_state_digest010(engine.model.basis_head));engine.step()
        proof=json.loads((engine.out/'FIRST_UPDATE_PROOF.json').read_text())
        assert proof['artificial_only'] and proof['actual_G_updates']==1
        assert proof['trainable_parameters']==(1288 if arm=='M0' else 6)
        engine.model.assert_frozen_unchanged(deep=True)
        if arm=='M1':assert tensor_state_digest010(engine.model.basis_head)==engine.model.reference_C1_head_sha256
        sampling.append(engine.last_metrics['source_parent_rows'])
        engine.checkpoint('ARTIFICIAL_final.pt','final')
    assert states[0]==states[1] and sampling[0]==sampling[1]
    assert store.counts['G_updates']==store.counts['optimizer_steps']==store.counts['new_AD_queries']==store.counts['training_backward_queries']==2
    assert store.counts['mean_adapter_rows']==8 and store.counts['old_A_rows']==store.counts['head_actual_parameter_forward_rows']==16
    assert store.counts['hard_ADC_rows']==store.counts['all_path_gather_rows']==48 and store.counts['noise005_rows']==16
    assert store.counts['probe_AD_queries']==store.counts['calibration_map_rows']==store.counts['integral_statistics_fits']==0
    assert not store.inflight and store.entered==store.completed
    checks.append(dict(name='actual_two_arm_synthetic_engine_steps_streams_counts_freeze',counters=store.counts))

    # Deliberately injected negative gate proves the controller path cannot load
    # checkpoints/create models/optimizers when the registered evidence is absent.
    previous_gate,previous_evidence,previous_store=d.gate015,d.mean_evidence015,d.Store015
    prior_prepare=d.prepare_numeric015
    try:
        d.gate015=lambda *a:(p,{},{})
        def negative_gate(config,paths,store):
            result={'all6_passed':False,'artificial_only':True};store.json('MEAN_RESIDUAL_GATE.json',result);return result
        d.mean_evidence015=negative_gate
        d.Store015=lambda output,config,checked:previous_store(output,config,checked,artificial=True)
        def forbidden(*a,**kw):raise AssertionError('negative gate reached model/CP preparation')
        d.prepare_numeric015=forbidden
        result=d.run015('synthetic_config','synthetic_sha',out/'ARTIFICIAL_GATE_SKIP','cpu')
        assert result['training_skipped'] and result['counters']['checkpoint_loads']==result['counters']['G_updates']==0
    finally:d.gate015=previous_gate;d.mean_evidence015=previous_evidence;d.Store015=previous_store;d.prepare_numeric015=prior_prepare
    checks.append(dict(name='registered_negative_gate_skips_before_CP_NN_optimizer',artificial_only=True))

    failure=d.Store015(out/'ARTIFICIAL_FAILURE',p,{},artificial=True)
    def injected():raise RuntimeError('synthetic call did not return')
    fail(lambda:failure.call('mean_failure',dict(mean_adapter_rows=3),injected),'did not return')
    assert failure.counts['mean_adapter_rows']==0 and failure.attempted['mean_adapter_rows']==3 and failure.inflight
    checks.append(dict(name='durable_attempted_completed_unknown_preserved',completed=0,attempted=3))
    caps=d.budget_caps015(d.settings015())
    assert caps['old_A_rows']==caps['noise005_rows']==caps['head_total_kernel_rows']==8613952
    assert caps['hard_ADC_rows']==caps['all_path_gather_rows']==26474784 and caps['mean_adapter_rows']==4306976
    assert caps['checkpoint_loads']==4 and caps['G_updates']==caps['new_AD_queries']==8000
    assert all(sha(ROOT/n)==digest for n,digest in protected.items()) and tensor_state_digest010(baseline)==frozen
    checks.append(dict(name='formal_formula_caps_and_original95_bytes_unchanged',budget_caps=caps))
    receipt=dict(schema='n01/wp3-artificial-semantics/v1',status='PASS',artificial_only=True,checks=checks,
        elapsed_seconds=time.monotonic()-started,original95_unchanged=True,
        scientific_CP_NN_AD_training=0,private_arrays_read=0,SSH_calls=0,
        synthetic_updates=2,synthetic_functional_AD_queries=1,
        sources={n:sha(ROOT/n) for n in ('src/n01/model015.py','scripts/run_wp3_015.py')})
    write_new(out/'ENGINEERING_RECEIPT.json',receipt)
    print(json.dumps({'status':'PASS','checks':len(checks),'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
