"""Focused016 semantics with artificial pulses and untrained A only."""
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
import check_model010 as mf
import check_mean_random009 as fixture
import run_wp3_016 as d
from n01.model008 import canonical_charge008
from n01.model014 import FrozenDeterministicVoltageShape014
from n01.model016 import FrozenM0ChargeResponse016,calibration_statistics016,inverse_softplus016,charge_coefficients016,Q_REFERENCE016
from n01.model010 import tensor_state_digest010
from n01.losses008 import ADC_STEP
from n01.losses014 import fit_integral_statistics014
from n01.train_shape010 import Streams010,cpu_tree
from wp1_ops import sha,write_new


def reject(fn,text):
    try:fn()
    except (ValueError,RuntimeError) as e:
        assert text in str(e),str(e);return
    raise AssertionError('expected rejection '+text)


def main():
    out=Path(sys.argv[1]).resolve();assert out.drive.upper() in ('E:','D:') and not out.exists();out.mkdir(parents=True)
    (out/'tmp').mkdir();os.environ.update(TEMP=str(out/'tmp'),TMP=str(out/'tmp'),TMPDIR=str(out/'tmp'))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic();checks=[]
    manifest=json.loads((ROOT/'WP3_SOURCE_MANIFEST.json').read_bytes());original={n:sha(ROOT/n) for n in manifest['files']}
    assert len(original)==105 and all(original[n]==v['sha256'] for n,v in manifest['files'].items())
    protected_config=sha(ROOT/'wp3_015_config.json')
    x,z=mf.fixture();baseline=mf.baseline_fixture();frozen=tensor_state_digest010(baseline)
    reference=FrozenDeterministicVoltageShape014(baseline,seed=2026101540);head=cpu_tree(reference.basis_head.state_dict())
    with torch.no_grad():ref=reference(x,z)
    calibration=calibration_statistics016(ref['base_signal_pre_V'],adc_step=ADC_STEP)
    assert abs(calibration['actual_A_expected_increment_RMS_V']/ADC_STEP-1)<1e-14
    assert calibration['all_Q_mu_finite_strictly_positive']
    rng=torch.get_rng_state().clone();model=FrozenM0ChargeResponse016(baseline,head,seed=2026101540,calibration=calibration)
    assert torch.equal(rng,torch.get_rng_state())
    latent=torch.linspace(-1.4,1.8,len(x),dtype=torch.float64)
    with torch.no_grad():active=model(x,z,latent);off=model(x,z,None,charge_enabled=False)
    assert mf.bytes_equal(off['signal_pre'],ref['signal_pre']) and off['signal_pre'] is off['M0_reference_signal_pre_V']
    assert mf.bytes_equal(active['random_shape_pre_V'],ref['random_shape_pre_V']) and mf.bytes_equal(active['basis_amplitude_V'],ref['basis_amplitude_V'])
    assert bool((active['signal_pre'][:,:100]==0).all()) and float(active['canonical_charge_identity_residual'].abs().max())<1e-10
    assert mf.bytes_equal(active['Q_mu_Vns'],canonical_charge008(active['base_signal_pre_V']))
    checks.append(dict(name='actual_A_one_LSB_analytic_initialization_off_exact_bytes_and_shape_independence',calibration=calibration))

    # Deterministic quadrature of the actual law, without MC sample centering.
    nodes,weights=np.polynomial.hermite.hermgauss(48);nodes=torch.tensor(nodes*math.sqrt(2),dtype=torch.float64)
    weights=torch.tensor(weights/math.sqrt(math.pi),dtype=torch.float64)
    Q=torch.tensor([.05,.4,3.,20.],dtype=torch.float64);a=.03;b=.02
    s2=torch.log1p(a/Q+b);q=Q[:,None]*torch.exp(s2.sqrt()[:,None]*nodes-.5*s2[:,None])
    mean=(q*weights).sum(1);var=((q-Q[:,None]).square()*weights).sum(1)
    assert torch.allclose(mean,Q,rtol=1e-13,atol=1e-13) and torch.allclose(var,a*Q+b*Q.square(),rtol=1e-12,atol=1e-13)
    assert float(torch.abs(Q*torch.exp(-.5*s2)-Q).max())>0
    checks.append(dict(name='positive_mean_corrected_lognormal_conditional_mean_variance_quadrature',mean_absmax=float((mean-Q).abs().max()),variance_absmax=float((var-(a*Q+b*Q.square())).abs().max())))

    def bad_q():
        item=dict(ref);item['base_signal_pre_V']=ref['base_signal_pre_V'].clone();item['base_signal_pre_V'][0].zero_()
        return model.apply_charge(item,latent)
    reject(bad_q,'strictly positive')
    reject(lambda:model(x,z,torch.full_like(latent,float('nan'))),'finite')
    reject(lambda:calibration_statistics016(torch.zeros(2,384,dtype=torch.float64),adc_step=ADC_STEP),'strictly positive')
    before=(model.theta_a.detach().clone(),model.theta_b.detach().clone())
    with torch.no_grad():model.theta_a.fill_(1000);model.theta_b.fill_(1000)
    assert mf.bytes_equal(model(x,z,None,charge_enabled=False)['signal_pre'],ref['signal_pre'])
    with torch.no_grad():model.theta_a.copy_(before[0]);model.theta_b.copy_(before[1])
    reject(lambda:inverse_softplus016(0.),'positive')
    checks.append(dict(name='strict_positive_actual_Q_no_floor_nonfinite_rejected_off_bypasses_compensation'))

    # A fixed smooth functional checks both new parameter derivatives.
    model.zero_grad(set_to_none=True);target=torch.linspace(-.7,1.3,384,dtype=torch.float64)[None]
    value=(model(x,z,latent)['signal_pre']*target).sum();value.backward()
    gradient=torch.stack([p.grad.detach().clone() for p in model.trainable_parameters()]);assert bool(torch.isfinite(gradient).all() and gradient.ne(0).all())
    finite=[]
    for param in model.trainable_parameters():
        original_value=param.detach().clone();values=[]
        for sign in [-1,1]:
            with torch.no_grad():param.copy_(original_value+sign*1e-4)
            values.append(float((model(x,z,latent)['signal_pre']*target).sum().detach()))
        with torch.no_grad():param.copy_(original_value)
        finite.append((values[1]-values[0])/2e-4)
    assert torch.allclose(torch.tensor(finite,dtype=torch.float64),gradient,rtol=2e-6,atol=2e-9)
    assert all(p.grad is None and not p.requires_grad for p in model.basis_head.parameters())
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=.01,weight_decay=0.);model.assert_optimizer_membership(optimizer)
    bad=torch.optim.AdamW(list(model.trainable_parameters())+list(model.basis_head.parameters()),lr=.01,weight_decay=0.)
    reject(lambda:model.assert_optimizer_membership(bad),'exactly')
    bad=torch.optim.AdamW(model.trainable_parameters(),lr=.001,weight_decay=0.)
    reject(lambda:model.assert_optimizer_membership(bad),'lr')
    model.train();assert not model.basis_head.training and not baseline.training;model.assert_frozen_unchanged(deep=True)
    checks.append(dict(name='both_parameter_finite_difference_exact_AdamW_whitelist_and_protected_head',gradient=gradient.tolist(),finite_difference=finite))

    # Adding the new stream leaves every existing draw and generator state exact.
    streams=d.ChargeStreams016(191,192,193,194);old_stream=Streams010(191,192,193)
    qstate=streams.charge.get_state().clone();global_state=torch.get_rng_state().clone()
    for rows in [3,7]:
        actual=streams.draw(rows,'cpu',lambda name,charges,fn,**meta:fn());expected=old_stream.draw(rows,'cpu')
        assert mf.bytes_equal(actual[0],expected[0]) and mf.bytes_equal(actual[2],expected[2])
        assert all(mf.bytes_equal(actual[1][k],expected[1][k]) for k in expected[1])
    assert torch.equal(global_state,torch.get_rng_state()) and not torch.equal(qstate,streams.charge.get_state())
    assert all(torch.equal(streams.existing.state_dict()[k],v) for k,v in old_stream.state_dict().items())
    checks.append(dict(name='independent_charge_stream_preserves_every_old015_prefix_and_global_RNG'))

    # Two tiny actual engine steps and the actual full evaluator/save path.
    data,stats,_=fixture.artificial_fixture();integral=fit_integral_statistics014(torch.as_tensor(data['real']))
    p=d.settings016();p['G_updates']=1;p['calibration'].update(parents=len(data['source']),chunk_rows=32)
    p['training'].update(source_parents_per_update=4,draws_per_parent=2,real_rows_per_update=8,forward_chunk_rows=4,status_interval=1,checkpoint_steps=[1])
    p['evaluation'].update(parents=len(data['source']),draws=4,chunk_rows=32)
    p['contracts']=dict(tolerance_table=dict(path='synthetic_only',sha256='synthetic'));p['budget_caps']=d.budget_caps016(p)
    store=d.Store016(out/'ARTIFICIAL_BATCH',p,{},artificial=True)
    info=dict(source_id=data['source_ids'],role=data['source_role'],canonical_Q_Vns=data['source'].sum(1),
        amplitude_384_V=data['source_carrier384'].max(1),PC_scores_384_Qnorm=np.zeros((len(data['source']),2)),
        PC_valid_384_Qnorm=np.ones(len(data['source']),bool),physical_weight_known=np.zeros(len(data['source']),bool))
    rt=d.Runtime016(p,data,baseline,fixture.ArtificialExtension().eval(),stats,integral,store,'cpu',info)
    cal=rt.calibrate_charge()
    assert store.counts['calibration_A_rows']==store.counts['old_A_rows']==len(data['source']) and store.counts['charge_initialization_fits']==1
    assert all(store.counts[k]==0 for k in ['head_total_kernel_rows','noise005_rows','hard_ADC_rows','charge_latent_scalar_draws','new_AD_queries'])
    for seed_index,seed in enumerate(p['seed_sets']):
        engine=d.Engine016(rt,seed,head,cal,{'artificial_only':True},store.root/f'seed{seed_index}_Q1')
        initial=engine.evaluate(initial=True);assert len(initial)==18
        engine.step();assert engine.last_metrics['trainable_parameters']==2
        engine.checkpoint('ARTIFICIAL_final.pt','final');engine.model.assert_frozen_unchanged(deep=True)
        final=engine.evaluate(initial=False);assert len(final)==6
        source=engine.out/'initial_evaluation/bank0_M0_fast_HARD_ROWS.npz'
        with np.load(source,allow_pickle=False) as saved:assert len(saved['source_parent_row'])==len(data['source'])*4
        checkpoint=torch.load(engine.out/'ARTIFICIAL_final.pt',weights_only=False)
        assert 'new_head' not in checkpoint and checkpoint['trainable_parameter_names']==['theta_a','theta_b']
        assert torch.equal(checkpoint['streams']['charge'],engine.streams.state_dict()['charge'])
    expected=d.budget_caps016(p);expected['checkpoint_loads']=0
    assert store.counts==expected,(store.counts,expected)
    assert not store.inflight and store.entered==store.completed
    assert len(list(store.root.glob('seed*/initial_evaluation/*_HARD_ROWS.npz')))==36
    assert len(list(store.root.glob('seed*/final_evaluation/*_HARD_ROWS.npz')))==12
    assert len(list(store.root.glob('seed*/*_evaluation/*_BASELINE100.npz')))==8
    assert len(list(store.root.glob('seed*/*_evaluation/*_CHARGE_PREADC_MOMENTS.npz')))==8
    checks.append(dict(name='actual_new_engine_two_synthetic_updates_full_four_axis_48hard48REPORT8baseline8charge_moments',counters=store.counts,
        charge_initialization=cal,trial0_first_update=str(store.root/'seed0_Q1/FIRST_UPDATE_PROOF.json')))

    # Durable failed operations never become completed, and evidence stays owned.
    failed=d.Store016(out/'ARTIFICIAL_FAILURE',p,{},artificial=True)
    def explode():raise RuntimeError('injected016_charge_failure')
    reject(lambda:failed.call('injected_charge_failure',dict(charge_module_rows=5),explode),'injected016')
    assert failed.counts['charge_module_rows']==0 and failed.attempted['charge_module_rows']==5 and failed.inflight
    checks.append(dict(name='new_charge_operation_failure_retains_entered_without_false_completion'))
    assert all(sha(ROOT/n)==digest for n,digest in original.items()) and sha(ROOT/'wp3_015_config.json')==protected_config
    assert tensor_state_digest010(baseline)==frozen
    result=dict(schema='n01/wp3-016-focused-artificial-engineering/v1',passed=True,artificial_only=True,checks=checks,
        elapsed_seconds=time.monotonic()-started,original105_sources_and107_controls_unchanged=True,
        scientific_CP_NN_AD_random_training=0,private_arrays_read=0,SSH_calls=0,synthetic_updates=2,synthetic_functional_AD_queries=1,
        sources={name:sha(ROOT/name) for name in d.OWNED_FILES})
    write_new(out/'ENGINEERING_RECEIPT.json',result);print(json.dumps(dict(passed=True,checks=len(checks),seconds=result['elapsed_seconds'],receipt=str(out/'ENGINEERING_RECEIPT.json'))))


if __name__=='__main__':main()
