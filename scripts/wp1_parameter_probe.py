"""Bounded true head-parameter directions, not additive carrier probes.

No file loading, training, optimizer construction or autograd occurs here.
Numerical imports are lazy and require the caller's common SHA/path gate.
"""
from __future__ import annotations
import math

HEAD_NAMES=('basis_head.weight','basis_head.bias')
HEAD_SHAPES=((8,32,5),(8,))


def require(flag,message):
    if not flag:raise ValueError(message)


def numpy_copy(value):
    import numpy as np
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value).copy()


def validated_head_state(cp):
    import numpy as np
    require(cp.get('schema')=='n01/shape010-checkpoint/v1' and cp.get('run_id')=='shape010_20261014'
        and cp.get('G_updates')==2000 and cp.get('complete_step_boundary') is True
        and cp.get('trainable_parameter_names')==list(HEAD_NAMES) and set(cp.get('new_head',{}))=={'weight','bias'},
        'Exact original final thin checkpoint required')
    values=[numpy_copy(cp['new_head'][n]) for n in ('weight','bias')]
    require(all(x.shape==s and x.dtype==np.float32 and np.isfinite(x).all() for x,s in zip(values,HEAD_SHAPES)),
        'Head dtype/shape/finite mismatch')
    return np.concatenate([x.astype(np.float64).reshape(-1) for x in values])


def prospective_adamw_direction(cp,current_saved_gradient,clip_reference=5.):
    """Saved-state algebra, not a native or actually executed optimizer step.

Calculates in float64; the evaluated parameters are cast to original float32.
The reference gradient is the frozen012 bank, not any newly generated bank.
"""
    import numpy as np
    theta=validated_head_state(cp);g=numpy_copy(current_saved_gradient).astype(np.float64)
    require(g.shape==(1288,) and np.isfinite(g).all() and clip_reference==5.,'Bound gradient/clip5 required')
    opt=cp.get('optimizer');require(isinstance(opt,dict) and len(opt.get('param_groups',[]))==1,'Saved optimizer state missing; do not invent direction')
    group=opt['param_groups'][0];require(group.get('params')==[0,1] and group.get('lr')==.0002
        and tuple(group.get('betas',()))==(.9,.999) and group.get('eps')==1e-8 and group.get('weight_decay')==.0001
        and group.get('amsgrad') is False and group.get('maximize') is False,'Original AdamW parameters mismatch')
    moments=[]
    for key,shape in zip(group['params'],HEAD_SHAPES):
        state=opt['state'].get(key,opt['state'].get(str(key)))
        require(isinstance(state,dict) and float(numpy_copy(state.get('step')))==2000.,'Original moments/step2000 missing')
        m,v=map(numpy_copy,(state['exp_avg'],state['exp_avg_sq']))
        require(m.shape==v.shape==shape and m.dtype==v.dtype==np.float32 and np.isfinite(m).all()
            and np.isfinite(v).all() and (v>=0).all(),'Invalid saved moments')
        moments.append((m.astype(np.float64).reshape(-1),v.astype(np.float64).reshape(-1)))
    m=np.concatenate([x[0] for x in moments]);v=np.concatenate([x[1] for x in moments])
    norm=float(np.linalg.norm(g));coefficient=min(1.,clip_reference/(norm+1e-6));gc=g*coefficient
    beta1,beta2=group['betas'];step=2001
    m1=beta1*m+(1-beta1)*gc;v1=beta2*v+(1-beta2)*gc**2
    updated=(1-group['lr']*group['weight_decay'])*theta-(group['lr']/(1-beta1**step))*m1/(np.sqrt(v1/(1-beta2**step))+group['eps'])
    direction=updated-theta
    require(np.isfinite(direction).all(),'Nonfinite prospective direction')
    return direction,dict(source='saved_final_AdamW_moments_plus_saved012_total_gradient',prospective_step=step,
        executed_optimizer_step=False,optimizer_instantiated=False,new_bank_gradient_used=False,
        gradient_norm=norm,registered_clip_reference=clip_reference,clip_coefficient=coefficient,
        parameter_groups={k:group[k] for k in ('lr','betas','eps','weight_decay','amsgrad','maximize')},
        arithmetic='float64_formula_then_original_float32_evaluated_weights',native_optimizer_kernel_rounding_equivalence='not_claimed')


def parameter_directions_from_saved_state(cp,gradients,*,random_seed):
    import numpy as np
    theta=validated_head_state(cp);result=[]
    for key in ('total','wave','QPSD'):
        g=numpy_copy(gradients[key]).astype(np.float64)
        require(g.shape==(1288,) and np.isfinite(g).all(),'Invalid bound012 parameter gradient: '+key)
        result.append(dict(name='negative_saved012_'+key,vector=-g,
            provenance='theta2000_saved012_fixed_BANK_gradient',descent_intent=True))
    vector,metadata=prospective_adamw_direction(cp,gradients['total'])
    result.append(dict(name='prospective_AdamW2001',vector=vector,metadata=metadata,descent_intent=True))
    random=np.random.default_rng(random_seed).integers(0,2,size=1288,dtype=np.int8).astype(np.float64)*2-1
    result.append(dict(name='fixed_Rademacher',vector=random,seed=int(random_seed),descent_intent=False))
    for item in result:
        norm=float(np.linalg.norm(item['vector']));item.update(parameter_norm=norm,direction_defined=norm>0)
        if norm==0:item['undefined_reason']='zero_parameter_direction'
    return theta,result


def split_parameter_vector(flat):
    import numpy as np
    x=np.asarray(flat,np.float64);require(x.shape==(1288,) and np.isfinite(x).all(),'Finite head1288 vector required')
    return {'basis_head.weight':x[:1280].reshape(HEAD_SHAPES[0]),'basis_head.bias':x[1280:].reshape(HEAD_SHAPES[1])}


def parameter_forward(model,source384,shape_latent,theta0,direction,scale):
    """Actual original forward with explicit parameter values; no .data mutation."""
    import torch
    import numpy as np
    flat=np.asarray(theta0,np.float64)+float(scale)*np.asarray(direction,np.float64)
    values=split_parameter_vector(flat);reference=dict(model.named_parameters())
    parameters={n:torch.as_tensor(x,device=reference[n].device,dtype=reference[n].dtype) for n,x in values.items()}
    require(set(parameters)==set(HEAD_NAMES),'Only the two head parameters may be substituted')
    with torch.no_grad():
        return torch.func.functional_call(model,parameters,(source384,shape_latent),kwargs={'enabled':True},strict=False)


def analytic_head_direction_map(model,decoded,base_signal384,shape_latent,flat_direction,*,reference_control_logits):
    """Analytic local head map, for scale selection only, without autograd.

The finite parameter state itself is subsequently evaluated by parameter_forward.
All linear-kernel rows here must be charged separately by the caller.
"""
    import torch
    from torch.nn import functional as F
    from n01.model008 import control_average008,canonical_charge008
    d=split_parameter_vector(flat_direction);head=model.basis_head
    require(decoded.shape==(len(base_signal384),32,384) and shape_latent.shape==(len(decoded),8),'Reference decoded/latent shape mismatch')
    require(head.padding_mode=='zeros','Only the frozen original head kernel accepted')
    with torch.no_grad():
        weight=torch.as_tensor(d[HEAD_NAMES[0]],device=head.weight.device,dtype=head.weight.dtype)
        bias=torch.as_tensor(d[HEAD_NAMES[1]],device=head.bias.device,dtype=head.bias.dtype)
        # Capture this value from the already-budgeted reference head forward.
        # Never recompute head(decoded) once per direction as a hidden pilot.
        logits=reference_control_logits.detach().double()
        dl=control_average008(F.conv1d(decoded,weight,bias,stride=head.stride,padding=head.padding,dilation=head.dilation,groups=head.groups)).double()
        require(logits.shape==dl.shape and bool(torch.isfinite(logits).all()),'Original reference control logits required')
        dc=(1-torch.tanh(logits)**2)*dl
        smooth=torch.einsum('bkc,tc->bkt',dc,model.baseline.expansion)*model.baseline.gate
        projected=smooth-canonical_charge008(smooth)[...,None]*model.baseline.projection
        amplitude=model.amplitude_fraction*base_signal384.abs().amax(1)
        dy=(amplitude[:,None,None]*projected*shape_latent.double()[...,None]).sum(1)
        require(bool(torch.isfinite(dy).all()) and bool((dy[:,:100]==0).all()),'Invalid analytic direction map')
        return dy


def nominal_scales(direction_map_squared_sum,rows,*,ADC_LSB_V=1/16383,targets=(.1,1.,10.)):
    rms=math.sqrt(float(direction_map_squared_sum)/(int(rows)*384))
    require(rows>0 and math.isfinite(rms) and rms>=0,'Invalid pooled direction RMS')
    if rms==0:return dict(direction_defined=False,reason='zero_analytic_output_direction',lambda_per_scale=[None]*3)
    return dict(direction_defined=True,analytic_output_RMS_per_parameter_scale_V=rms,
        nominal_carrier_RMS_ADC_LSB=list(targets),lambda_per_scale=[float(t)*ADC_LSB_V/rms for t in targets],
        actual_finite_RMS_must_be_reported=True,steps_not_loss_adapted=True)


def sensitivity_cases(config):
    cases=[]
    for name,key in (('gain','gain_percent_signed'),('tail','tail_percent_signed'),('noise','noise_percent_signed')):
        for amount in config[key]:
            for sign in (-1,1):cases.append(dict(kind=name,percent=float(amount),sign=sign))
    for sigma in config['width_sigma_software_ns_positive']:cases.append(dict(kind='width',sigma=float(sigma),sign=1))
    require(len(cases)==21,'Exactly21 preregistered artificial cases required')
    return cases


def apply_sensitivity(signal384,noise384,case,*,tail_gate_width=4.,gaussian_radius=12):
    """Explicit preADC calibration control, not a physical model or new REAL."""
    import torch
    from torch.nn import functional as F
    require(signal384.shape==noise384.shape and signal384.ndim==2 and signal384.shape[1]==384,'Aligned reference S/noise384 required')
    require(bool((signal384[:,:100]==0).all()),'Original pure-signal prefix must be zero')
    with torch.no_grad():
        s,n=signal384,noise384;kind=case['kind']
        if kind=='gain':s=s*(1+case['sign']*case['percent']/100.)
        elif kind=='tail':
            require(tail_gate_width>0,'Frozen tail gate width must be positive')
            peak=s[:,112:144].argmax(1)+112
            t=torch.arange(384,device=s.device,dtype=s.dtype)
            gate=torch.sigmoid((t[None]-peak[:,None]-30)/tail_gate_width);gate[:,:100]=0
            s=s*(1+case['sign']*case['percent']/100.*gate)
        elif kind=='noise':
            mu=n[:,:100].mean(1,keepdim=True);n=mu+(1+case['sign']*case['percent']/100.)*(n-mu)
        elif kind=='width':
            require(gaussian_radius==12 and case['sigma']>0,'Frozen positive Gaussian width/radius12 required')
            t=torch.arange(-gaussian_radius,gaussian_radius+1,device=s.device,dtype=s.dtype)
            kernel=torch.exp(-.5*(t/case['sigma'])**2);kernel/=kernel.sum()
            s=F.conv1d(s[:,None],kernel[None,None],padding=gaussian_radius)[:,0];s[:,:100]=0
        else:raise ValueError('Unregistered sensitivity mechanism')
        require(bool(torch.isfinite(s+n).all()) and bool((s[:,:100]==0).all()),'Nonfinite/corrupted prefix in artificial control')
        return s+n,dict(artificial_TRAIN_calibration=True,measurement=False,model_candidate=False,case=case,
            original_noise_prefix_mean_preserved=kind=='noise',pure_signal_prefix100_zero=True,
            canonical_Q_unchanged='not_assumed; report_actual_residual',signal=s,noise=n)


def select_method_once(cells,tolerances,primary_directions=('negative_saved012_total','prospective_AdamW2001'),*,undefined_directions=None):
    """Root's preregistered ALL12-cell rule, without selecting a best step.

cells contain only the positive, nominal1LSB endpoints and explicit deltas.
Random/other scales/signs are reported by the caller, never promoted here.
"""
    undefined_directions=undefined_directions or {}
    by_direction={name:[r for r in cells if r['direction']==name] for name in primary_directions};decisions={}
    expected={(b,k,h) for b in (0,1) for k in (0,1) for h in ('fast','slow','common_level')}
    for name,rows in by_direction.items():
        if name in undefined_directions and not rows:
            rows=[dict(direction=name,bank=b,batch=k,hypothesis=h,sign=1,nominal_scale_LSB=1.,defined=False,
                undefined_reason=undefined_directions[name]) for b in (0,1) for k in (0,1) for h in ('fast','slow','common_level')]
        keys=[(r['bank'],r['batch'],r['hypothesis']) for r in rows]
        require(len(keys)==len(set(keys)) and set(keys)<=expected,'Duplicate or unregistered primary cell')
        # Missing/undefined evidence is a scientific inconclusive slot, not a
        # reason to turn an otherwise preserved WP1 run into native FAILED.
        for b,k,h in sorted(expected-set(keys)):
            rows.append(dict(direction=name,bank=b,batch=k,hypothesis=h,sign=1,nominal_scale_LSB=1.,defined=False,
                undefined_reason=undefined_directions.get(name,'missing_primary_cell')))
        adequate=[];repair=[];details=[]
        for r in rows:
            require(r['sign']==1 and r['nominal_scale_LSB']==1.,'Only preregistered positive1LSB primary endpoints')
            if r.get('defined',True) is False:
                adequate.append(False);repair.append(False);details.append(dict(cell=r,adequate_cell=False,repair_cell=False,
                    reason=r.get('undefined_reason','undefined_primary_direction')));continue
            evidence=(r['S_delta_objective'],r['H_delta_objective'],*[r['hard_component_delta'][k] for k in ('wave','QPSD','noise')])
            if any(v is None or not math.isfinite(v) for v in evidence):
                adequate.append(False);repair.append(False);details.append(dict(cell=r,adequate_cell=False,repair_cell=False,reason='undefined_primary_evidence'));continue
            th=tolerances[r['hypothesis']];positive=lambda k:k in th and th[k] is not None and math.isfinite(th[k]) and th[k]>0
            calibrated=all(positive(k) for k in ('wave','QPSD','noise'))
            ad=calibrated and r['S_delta_objective'] < -1e-12 and r['H_delta_objective'] < -1e-12 and all(r['hard_component_delta'][k]<=th[k] for k in ('wave','QPSD','noise'))
            material=any(positive(k) and r['S_H_self_distance'][k] is not None and r['S_H_self_distance'][k]/th[k]>1 for k in ('wave','QPSD'))
            rp=calibrated and r['S_delta_objective'] < -1e-12 and r['H_delta_objective'] > 1e-12 and material
            adequate.append(ad);repair.append(rp);details.append(dict(cell=r,critical_tolerance_calibrated=calibrated,adequate_cell=ad,repair_cell=rp,
                reason='exact_registered_rule; no tiny-step or random-direction criterion'))
        decisions[name]=dict(adequate_all12=all(adequate),repair_all12=all(repair),cells=details)
    good=[n for n,d in decisions.items() if d['adequate_all12']];unsafe=[n for n,d in decisions.items() if d['repair_all12']]
    choice='adequate' if good else 'repair_needed' if unsafe else 'inconclusive'
    return dict(choice=choice,adequate_directions=good,unsafe_directions=unsafe,direction_decisions=decisions,
        conflict_priority='adequate_with_unsafe_direction_and_scope_reported',
        next_action='C0_C1_short_training_no_more_boundary_diagnostics' if choice=='inconclusive' else 'root_registered_WP2_choice',
        scientific_response_acceptance=False,rule_posthoc_changed=False)
