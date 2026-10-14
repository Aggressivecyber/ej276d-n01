"""New WP1 semantics only; original CPU fixture, zero scientific inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve()
    if out.drive.upper() not in ('E:','D:') or out.exists():raise ValueError('Fresh E/D artifact directory required')
    out.mkdir(parents=True)
    import numpy as np
    import torch
    from check_model010 import baseline_fixture,fixture
    from n01.model010 import FrozenDeterministicVoltageShape010,tensor_state_digest010
    from n01.model008 import control_average008
    from n01.observation008 import observe_torch
    from n01.losses008 import fit_loss_statistics
    from n01.losses008_v2 import N01Loss008
    import wp1_parameter_probe as p
    import wp1_evaluation as e
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    checks=[];calls=rows=0
    def passed(n,**detail):checks.append(dict(name=n,passed=True,**detail))
    def reject(n,fn):
        try:fn()
        except (ValueError,RuntimeError):passed(n);return
        raise AssertionError('Unsafe case accepted: '+n)
    model=FrozenDeterministicVoltageShape010(baseline_fixture()).eval().requires_grad_(False)
    source,z=fixture();old=tensor_state_digest010(model);capture={}
    def hook(module,arguments,result):capture.update(decoded=arguments[0].detach().clone(),logits=control_average008(result).detach().double())
    h=model.basis_head.register_forward_hook(hook)
    with torch.no_grad():reference=model(source,z)
    h.remove();calls+=1;rows+=len(source)
    head={n:np.asarray(v.detach().numpy(),np.float32).copy() for n,v in model.basis_head.state_dict().items()}
    cp=dict(schema='n01/shape010-checkpoint/v1',run_id='shape010_20261014',G_updates=2000,complete_step_boundary=True,
        trainable_parameter_names=list(p.HEAD_NAMES),new_head=head,artificial_only=True,
        optimizer=dict(param_groups=[dict(params=[0,1],lr=.0002,betas=(.9,.999),eps=1e-8,weight_decay=.0001,amsgrad=False,maximize=False)],
        state={0:dict(step=np.asarray(2000.,np.float32),exp_avg=np.full((8,32,5),.003,np.float32),exp_avg_sq=np.full((8,32,5),.01,np.float32)),
            1:dict(step=np.asarray(2000.,np.float32),exp_avg=np.full((8,),-.002,np.float32),exp_avg_sq=np.full((8,),.02,np.float32))}))
    rng=np.random.default_rng(2026101541);g=rng.normal(0,.01,size=1288)
    theta,directions=p.parameter_directions_from_saved_state(cp,dict(total=g,wave=g*.2,QPSD=g*.8),random_seed=2026101541)
    assert len(directions)==5 and all(x['vector'].shape==(1288,) for x in directions)
    passed('five_actual_parameter_directions_from_artificial_saved_state')
    exact=p.parameter_forward(model,source,z,theta,g,0);calls+=1;rows+=len(source)
    assert torch.equal(exact['signal_pre'],reference['signal_pre']) and tensor_state_digest010(model)==old
    passed('zero_parameter_scale_native_byte_equivalence_and_restore')
    dy=p.analytic_head_direction_map(model,capture['decoded'],reference['base_signal_pre_V'],z,g,
        reference_control_logits=capture['logits'])
    plus=p.parameter_forward(model,source,z,theta,g,.001);minus=p.parameter_forward(model,source,z,theta,g,-.001)
    calls+=2;rows+=2*len(source)
    central=(plus['signal_pre']-minus['signal_pre'])/.002
    relative=float(torch.linalg.vector_norm(central-dy)/torch.linalg.vector_norm(dy))
    assert relative<.0003 and not torch.equal(plus['signal_pre'],reference['signal_pre'])
    assert tensor_state_digest010(model)==old and all(x.grad is None for x in model.parameters())
    model.trainable_parameters();model.assert_frozen_unchanged(deep=True)
    passed('analytic_direction_map_vs_true_parameter_finite_difference',relative_error=relative,original_head_unchanged=True)
    # One tiny artificial optimizer reference, wholly unrelated to scientific CP.
    tensors=[torch.nn.Parameter(torch.from_numpy(head[n]).clone()) for n in ('weight','bias')]
    optimizer=torch.optim.AdamW(tensors,lr=.0002,betas=(.9,.999),eps=1e-8,weight_decay=.0001,foreach=False)
    for k,tensor in enumerate(tensors):
        state=cp['optimizer']['state'][k]
        optimizer.state[tensor]={name:torch.from_numpy(np.asarray(value).copy()) for name,value in state.items()}
    tensors[0].grad=torch.from_numpy(g[:1280].reshape(8,32,5).astype(np.float32));tensors[1].grad=torch.from_numpy(g[1280:].astype(np.float32))
    torch.nn.utils.clip_grad_norm_(tensors,5);optimizer.step()
    actual=np.concatenate([x.detach().numpy().astype(np.float64).reshape(-1) for x in tensors])-theta
    reconstructed,_=p.prospective_adamw_direction(cp,g)
    error=float(np.max(np.abs(actual-reconstructed)));assert error<3e-9
    passed('saved_AdamW_formula_vs_tiny_native_reference',absmax_parameter_delta_error=error,artificial_optimizer_steps=1)
    cases=p.sensitivity_cases(dict(gain_percent_signed=[1,3,5],tail_percent_signed=[1,3,5],noise_percent_signed=[5,10,20],width_sigma_software_ns_positive=[.5,1,2]))
    assert len(cases)==21
    noise=torch.randn((8,384),generator=torch.Generator().manual_seed(2026101502),dtype=torch.float64)*.001
    for case in cases:
        carrier,meta=p.apply_sensitivity(reference['signal_pre'],noise,case)
        assert carrier.shape==(8,384) and torch.isfinite(carrier).all() and (meta['signal'][:,:100]==0).all()
        assert torch.equal(reference['signal_pre'],exact['signal_pre'])
    passed('21_preregistered_preADC_controls_no_model_or_input_mutation')
    observed=observe_torch(reference['signal_pre']+noise,ste=False,q_floor=.0001)
    real=observed['window_V'][:5].clone();baseline=observed['baseline100'][:5].clone()
    stats=fit_loss_statistics(real,observed['window_V'],baseline)
    loss=N01Loss008(stats,seed=2026101426,evaluation_seed=2026101430);context=loss.create_evaluation_context()
    info=dict(source_id=np.asarray(['a','b','c','d']),role=np.full(4,'TRAIN'),canonical_Q_Vns=np.arange(1.,5.),
        amplitude_384_V=np.arange(1.,5.),PC_scores_384_Qnorm=np.arange(8.).reshape(4,2),PC_valid_384_Qnorm=np.ones(4,bool))
    parents=np.repeat(np.arange(4),2);draw=np.tile(np.arange(2),4);bank=np.zeros(8,np.int64)
    result=e.evaluate_hard_population(observed['window_V'].numpy(),observed['baseline100'].numpy(),real_wave256=real.numpy(),
        real_baseline100=baseline.numpy(),parent_row=parents,draw_index=draw,bank_index=bank,source_information=info,
        loss=loss,context=context,real_groups=np.asarray([0,0,1,1,2]),signal_pre384=reference['signal_pre'].numpy())
    assert all(n in result for n in ('population','output_Q_conditions','source_input_trends','repeated_parent'))
    assert result['population']['fake_rows']==8 and result['population']['real_rows']==5
    assert result['repeated_parent']['parent_count']==4 and result['repeated_parent']['draws_per_parent']==2
    assert sum(x['fake_rows'] for x in result['output_Q_conditions']['bins'])==8 and not result['population']['scientific_acceptance']
    passed('four_separate_evaluations_unequal_N_and_parent_cluster_semantics')
    duplicate=draw.copy();duplicate[1]=0;reject('duplicate_draw_cannot_be_new_event',lambda:e.repeat_grid(parents,duplicate,bank))
    badinfo=copy.deepcopy(info);badinfo['role'][0]='DEV'
    reject('DEV_forward_lineage_refused',lambda:e.source_input_trends(result['feature_rows'],parents,badinfo,draw,bank))
    zero=observed['window_V'].numpy().copy();zero[0]=0
    feature,flags=e.observable_features(zero,observed['baseline100'].numpy(),float(loss.q_floor))
    assert feature['PSD_raw'].shape==(8,) and np.isnan(feature['PSD_raw'][0]) and flags['PSD_undefined'].sum()==1
    assert e.summary(feature['PSD_raw'])['undefined_rows']==1
    passed('undefined_Q_PSD_morphology_retained_with_counts')
    cached=e.observable_features(real.numpy(),baseline.numpy(),float(loss.q_floor))
    ordinary=e.component_distances(observed['window_V'].numpy(),observed['baseline100'].numpy(),real.numpy(),baseline.numpy(),loss,context)
    reused=e.component_distances(observed['window_V'].numpy(),observed['baseline100'].numpy(),real.numpy(),baseline.numpy(),loss,context,cached_reference=cached)
    assert ordinary==reused
    passed('fixed_reference_feature_cache_identical_to_original_observable_function')
    known={'peak_tied':np.asarray([True,False,False,False,False])}
    withflags=e.evaluate_hard_population(observed['window_V'].numpy(),observed['baseline100'].numpy(),real_wave256=real.numpy(),
        real_baseline100=baseline.numpy(),parent_row=parents,draw_index=draw,bank_index=bank,source_information=info,
        loss=loss,context=context,observation_flags={n:v.numpy() for n,v in observed['flags'].items()},
        real_observation_flags=known,real_unknown_flags=['ADC_halfgrid_tied'],cached_real=cached)
    assert withflags['population']['real_flags']['peak_tied']['count']==1 and 'peak_tied' not in cached[1]
    assert 'ADC_halfgrid_tied' in withflags['population']['real_unknown_flags'] and 'ADC_rail_hit' in withflags['population']['fake_flags']
    passed('cached_REAL_not_mutated_and_known_unknown_hard_flags_separate')
    import run_wp1_013 as driver
    obsrows=driver.observation_rows(dict(observed,adc_carrier=observed.get('adc_carrier',None)),reference['signal_pre']+noise,loss) if observed.get('adc_carrier') is not None else None
    if obsrows is None:
        from n01.observation008 import observe_generated_carrier
        extended=observe_generated_carrier(reference['signal_pre']+noise,ste=False,q_floor=float(loss.q_floor),return_carrier=True)
        obsrows=driver.observation_rows(extended,reference['signal_pre']+noise,loss)
    encoded=driver.lossless_window_codes(obsrows);reconstructed=encoded['ADC_crop_codes'].astype(np.float64)/16383-.5-encoded['baseline_mean_V'][:,None]
    reconstructed=(reconstructed.view(np.uint64)^encoded.get('IEEE754_reconstruction_XOR_uint64',np.zeros(reconstructed.shape,np.uint64))).view(np.float64)
    assert np.array_equal(reconstructed.view(np.uint64),obsrows['window_V'].view(np.uint64)) and 'flag__pre_ADC_out_of_range' in obsrows
    passed('final_observation_rows_and_lossless_float64_encoding')
    cells=[dict(direction='negative_saved012_total',sign=1,nominal_scale_LSB=1.,bank=b,batch=k,hypothesis=h,
        S_delta_objective=-.1,H_delta_objective=-.1,hard_component_delta=dict(wave=.01,QPSD=.01,noise=.01),
        S_H_self_distance=dict(wave=2.,QPSD=2.)) for b in (0,1) for k in (0,1) for h in ('fast','slow','common_level')]
    thresholds={h:dict(wave=1.,QPSD=1.,noise=1.) for h in ('fast','slow','common_level')}
    assert p.select_method_once(cells,thresholds)['choice']=='adequate'
    cells[0]['hard_component_delta']['noise']=None
    selected=p.select_method_once(cells,thresholds)
    assert selected['choice']=='inconclusive' and any(x['reason']=='undefined_primary_evidence' for x in selected['direction_decisions']['negative_saved012_total']['cells'])
    passed('undefined_primary_numerical_evidence_is_scientific_inconclusive')
    receipt=dict(schema='n01/wp1-new-semantic-artificial/v1',status='PASS',artificial_only=True,checks=checks,passed_checks=len(checks),
        artificial_native_model_calls=calls,artificial_native_model_rows=rows,artificial_analytic_direction_rows=8,
        artificial_optimizer_steps=1,actual_scientific_numeric_decodes=0,actual_scientific_CP_loads=0,actual_scientific_NN_rows=0,
        actual_scientific_AD_queries=0,actual_scientific_optimizer_steps=0,SSH_calls=0,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('scripts/wp1_evaluation.py','scripts/wp1_parameter_probe.py','scripts/run_wp1_013.py','scripts/wp1_ops.py')})
    (out/'ENGINEERING_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',passed_checks=len(checks),out=str(out))))

if __name__=='__main__':main()
