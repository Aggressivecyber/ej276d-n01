"""Only new WP1 semantics: tiny artificial Runtime and machine-rule boundary.

Never reads actual data/CP, creates a scientific optimizer, or contacts SSH.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve()
    if out.drive.upper() not in ('E:','D:') or out.exists():raise ValueError('Fresh E/D fixture required')
    out.mkdir(parents=True)
    import numpy as np
    import torch
    from check_model010 import baseline_fixture,fixture
    from n01.model010 import FrozenDeterministicVoltageShape010
    from n01.observation008 import observe_generated_carrier
    from n01.losses008 import fit_loss_statistics
    from n01.losses008_v2 import N01Loss008
    import n01.response008 as r
    import run_wp1_013 as d
    import wp1_parameter_probe as probe
    from wp1_ops import Store,durable_progress,clean
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    checks=[]
    def passed(name,**detail):checks.append(dict(name=name,passed=True,**detail))
    model=FrozenDeterministicVoltageShape010(baseline_fixture()).eval().requires_grad_(False)
    class FakeExtension(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model005=torch.nn.Linear(1,1).eval().requires_grad_(False)
        def assert_frozen(self):assert not any(x.requires_grad or x.grad is not None for x in self.parameters())
    extension=FakeExtension()
    source,z=fixture();base_source=source[::2]
    with torch.no_grad():initial=model(source,z)
    noise=torch.randn((8,384),dtype=torch.float64,generator=torch.Generator().manual_seed(100))*.001
    obs=observe_generated_carrier(initial['signal_pre']+noise,ste=False,q_floor=.001,return_carrier=True)
    waveform=obs['window_V'];groups=np.repeat(np.arange(6),[48,48,49,49,50,50]);take=np.arange(len(groups))%8
    real=waveform[take];realbase=obs['baseline100'][take]
    loss=N01Loss008(fit_loss_statistics(real,waveform,realbase),seed=2026101426,evaluation_seed=2026101430);context=loss.create_evaluation_context()
    data=dict(source=base_source[:,110:366].numpy(),source_carrier384=base_source.numpy(),real=real.numpy(),real_baseline100=realbase.numpy(),
        source_ids=np.arange(4),real_ids=np.arange(len(groups)),real_groups=groups,
        real_observation_flags=dict(peak_tied=np.zeros(len(groups),bool),peak_at_search_boundary=np.zeros(len(groups),bool),global_peak_outside_search=np.zeros(len(groups),bool)))
    for key in d.REAL_METADATA:data[key]=np.ones(len(groups),np.int32) if 'tied' in key else np.zeros(len(groups),bool if 'search' in key else np.int64)
    info=dict(source_id=np.arange(4),role=np.full(4,'TRAIN'),canonical_Q_Vns=np.arange(1.,5.),amplitude_384_V=np.arange(1.,5.),
        PC_scores_384_Qnorm=np.arange(8.).reshape(4,2),PC_valid_384_Qnorm=np.ones(4,bool),physical_weight_known=np.zeros(4,bool))
    cp=dict(schema='n01/shape010-checkpoint/v1',run_id='shape010_20261014',G_updates=2000,complete_step_boundary=True,
        trainable_parameter_names=list(probe.HEAD_NAMES),new_head={k:v.detach().clone() for k,v in model.basis_head.state_dict().items()},
        optimizer=dict(param_groups=[dict(params=[0,1],lr=.0002,betas=(.9,.999),eps=1e-8,weight_decay=.0001,amsgrad=False,maximize=False)],
            state={k:dict(step=np.asarray(2000.,np.float32),exp_avg=np.full(shape,.002,np.float32),exp_avg_sq=np.full(shape,.01,np.float32)) for k,shape in enumerate(probe.HEAD_SHAPES)}))
    rng=np.random.default_rng(42);g=rng.normal(0,.01,1288)
    p=json.loads((ROOT/'wp1_013_config.json').read_text('utf8'));p['artificial_only']=True
    p['reference'].update(SOURCE_TRAIN_parents=4,REAL_TRAIN_rows=len(groups),draws_per_bank=2,chunk_rows=8)
    p['parameter_validation'].update(parents=4,parents_per_batch=2,batches=2)
    p['RR'].update(rows_per_side=147,groups_per_side=3,histogram_each_side={'48':1,'49':1,'50':1})
    p['resource_limits'].update(CPU_threads=2)
    expected=dict(old_A_rows=496,head_actual_parameter_forward_rows=496,head_linear_direction_map_rows=80,
        head_total_kernel_rows=576,noise005_rows=16,hard_ADC_rows=2544,all_path_gather_rows=5424)
    p['budget_caps'].update(expected)
    store=Store(out/'artifacts',p,{},artificial=True);(out/'evaluation').mkdir()
    ctx=dict(model=model,extension=extension,data=data,loss=loss,context=context,device='cpu',head_cp=cp,
        gradients=dict(total=g,wave=.4*g,QPSD=.6*g),snapshot=d.reference_snapshot(model,extension))
    runtime=d.Runtime(p,ctx,store,out/'evaluation',info)
    original_fixed_noises=r.fixed_noises
    def fake_noises(ext,nz,eps):
        n=len(eps);v=next(iter(nz.values())).reshape(n,-1).double();prefix=v[:,:1]*.001+torch.sin(torch.arange(100,dtype=torch.float64))[None]*.0001
        prefix=prefix.expand(n,100);tail=eps[:,0].double()*.001
        return {h:torch.cat((prefix,tail*a),1) for h,a in zip(d.HYPOTHESES,(1.,.7,.4))}
    r.fixed_noises=fake_noises
    try:runtime.references_forward()
    finally:r.fixed_noises=original_fixed_noises
    assert store.counts['old_A_rows']==16 and store.counts['head_actual_parameter_forward_rows']==16
    assert all('flag__ADC_halfgrid_tied' in ref['theta']['fast'] for ref in runtime.references)
    passed('actual_Runtime_reference_nested_A_head_durable_counts_and_hard_flags')
    # Every bit is preserved even after injecting a one-bit numerical residual.
    encoded=d.lossless_window_codes(runtime.references[0]['theta']['fast']);a=encoded['ADC_crop_codes'].astype(np.float64)/16383-.5-encoded['baseline_mean_V'][:,None]
    recovered=(a.view(np.uint64)^encoded.get('IEEE754_reconstruction_XOR_uint64',np.zeros(a.shape,np.uint64))).view(np.float64)
    assert np.array_equal(recovered.view(np.uint64),runtime.references[0]['theta']['fast']['window_V'].view(np.uint64))
    changed=copy.deepcopy(runtime.references[0]['theta']['fast']);changed['window_V'].view(np.uint64)[0,0]^=np.uint64(1)
    assert d.lossless_window_codes(changed)['IEEE754_reconstruction_XOR_uint64'].any()
    passed('lossless_ADC_code_actual_mean_optional_XOR_preserves_float64_bits')
    runtime.complete_evaluations();tolerance=runtime.calibrate();selection=runtime.parameter_states(tolerance)
    assert store.counts==expected|{k:0 for k in store.counts if k not in expected}
    assert not store.inflight and d.reference_snapshot(model,extension)==ctx['snapshot']
    assert all(x.grad is None for x in model.parameters())
    passed('full_artificial_runtime_21_controls_30_actual_parameter_states_exact_budget',counts=store.counts)
    report=json.loads((out/'evaluation/FULL_0_theta_fast.json').read_text('utf8'))
    assert 'ADC_halfgrid_tied' in report['population']['fake_flags'] and 'ADC_halfgrid_tied' in report['population']['real_unknown_flags']
    repeats=json.loads((out/'evaluation/REPEATS_8_theta_fast.json').read_text('utf8'))
    assert repeats['draws_per_parent']==4 # miniature2banks*2draws, production2*4=8
    assert 'reconstruction' in repeats['hard_wave_mean_MCSE_V']
    passed('four_population_suites_observation_flags_unknown_REAL_and_combined_parent_draws')
    state=d.lossless_window_codes(runtime.references[0]['theta']['fast'])
    progress=durable_progress(out/'artifacts');assert progress['completed_counts']==expected and not progress['unknown_inflight_ids']
    passed('completed_jsonl_lower_bounds_match_successful_kernel_counts')
    # Deliberate nested-A success/head failure preserves actual A completion.
    failed=Store(out/'failed',p,{},artificial=True);runtime.store=failed
    hook=model.basis_head.register_forward_pre_hook(lambda *args:(_ for _ in ()).throw(RuntimeError('artificial_after_A')))
    try:runtime.forward(base_source[:1],z[:1])
    except RuntimeError:pass
    else:raise AssertionError('Failure not injected')
    finally:hook.remove()
    progress=durable_progress(out/'failed')
    assert failed.counts['old_A_rows']==1 and failed.counts['head_actual_parameter_forward_rows']==0 and progress['unknown_inflight_ids']==[0]
    assert '_baseline_once' not in model.__dict__ and d.reference_snapshot(model,extension)==ctx['snapshot']
    passed('partial_head_failure_preserves_completed_A_and_unknown_head_no_mutation')
    th={h:dict(wave=1.,QPSD=1.,noise=1.) for h in d.HYPOTHESES}
    def cells(name,H=-.1):return [dict(direction=name,sign=1,nominal_scale_LSB=1.,bank=b,batch=k,hypothesis=h,S_delta_objective=-.1,H_delta_objective=H,
        hard_component_delta=dict(wave=.01,QPSD=.01,noise=.01),S_H_self_distance=dict(wave=2.,QPSD=2.)) for b in (0,1) for k in (0,1) for h in d.HYPOTHESES]
    a,b=p['parameter_validation']['primary_directions']
    assert probe.select_method_once(cells(a),th)['choice']=='adequate'
    assert probe.select_method_once(cells(a)[:-1],th)['choice']=='inconclusive'
    assert probe.select_method_once([],th,undefined_directions={a:'zero',b:'zero'})['choice']=='inconclusive'
    assert probe.select_method_once(cells(a,H=.1),th)['choice']=='repair_needed'
    badth=copy.deepcopy(th);badth['fast']['noise']=None
    assert probe.select_method_once(cells(a,H=.1),badth)['choice']=='inconclusive'
    conflict=probe.select_method_once(cells(a)+cells(b,H=.1),th)
    assert conflict['choice']=='adequate' and conflict['unsafe_directions']==[b]
    try:probe.select_method_once(cells(a)+cells(a)[:1],th)
    except ValueError:pass
    else:raise AssertionError('Duplicate cell accepted')
    passed('machine_ALL12_ANYdirection_conflict_missing_zero_required_T_and_duplicate_rules')
    receipt=dict(schema='n01/wp1-runtime-artificial/v1',status='PASS',artificial_only=True,passed_checks=len(checks),checks=checks,
        miniature_parents=4,draws_per_bank=2,banks=2,scientific_data_decodes=0,scientific_CP_loads=0,scientific_NN_rows=0,
        scientific_AD_queries=0,scientific_optimizer_steps=0,SSH_calls=0,artificial_optimizer_created=0,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('scripts/run_wp1_013.py','scripts/wp1_parameter_probe.py','scripts/wp1_evaluation.py','scripts/wp1_ops.py')})
    (out/'ENGINEERING_RECEIPT.json').write_text(json.dumps(clean(receipt),indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',passed_checks=len(checks),out=str(out))))

if __name__=='__main__':main()
