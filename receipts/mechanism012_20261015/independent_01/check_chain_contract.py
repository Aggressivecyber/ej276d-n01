"""Explicitly artificial CPU check; no trained model, checkpoint or real arrays.

Uses two synthetic leaf tensors with the 1288-parameter layout and a small
nonlinear algebraic signal, not the original A/005/head networks. Six artificial
autograd calls test ancestor/intermediate-return and masked pullback contracts.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys

import torch


PROJECT = Path('E:/GAN/research/N01')
OUTPUT = Path(__file__).resolve().parent/'chain_artificial_01'
sys.path.insert(0,str(PROJECT/'src'))
sys.path.insert(0,str(PROJECT/'scripts'))
from n01.model import digitize, ADC_MIN, ADC_MAX
from n01.model008 import canonical_charge008
from n01.observation008 import observe_generated_carrier
from mechanism012_losses import analytic_window_pullback, q_decomposition


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flat(values):
    return torch.cat([v.detach().reshape(-1).double() for v in values])


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    generator=torch.Generator(device='cpu').manual_seed(20261015071)
    n=8
    features=torch.randn(n,160,generator=generator,dtype=torch.float64)
    weight=(.01*torch.randn(8,32,5,generator=generator,dtype=torch.float64)).requires_grad_()
    bias=torch.linspace(-.01,.01,8,dtype=torch.float64).requires_grad_()
    initial_weight,initial_bias=weight.detach().clone(),bias.detach().clone()
    bases=.01*torch.randn(8,384,generator=generator,dtype=torch.float64)
    bases[:,:100]=0
    bases[:,110:366]-=bases[:,110:366].mean(1,keepdim=True)
    coefficients=torch.tanh(features@weight.reshape(8,160).T+bias)
    time=torch.arange(384,dtype=torch.float64)
    fixed_signal=.1*torch.exp(-((time-126)/9).square())[None,:].repeat(n,1)
    fixed_signal[:,:100]=0
    shared_signal=fixed_signal+coefficients@bases
    noises=[]; observations=[]; hard=[]
    for h in range(3):
        noise=.0003*torch.randn(n,384,generator=generator,dtype=torch.float64)
        noise[:,150]=.61
        noise[:,200]=-.61
        noises.append(noise)
        carrier=shared_signal+noise
        observations.append(observe_generated_carrier(carrier,ste=True,q_floor=.001,return_indices=True,return_carrier=True))
        hard.append(observe_generated_carrier(carrier.detach(),ste=False,q_floor=.001,return_indices=True,return_carrier=True))
    windows=tuple(o['window_V'] for o in observations)
    target=(.11*torch.exp(-((torch.arange(256,dtype=torch.float64)-16)/10).square()))[None,:].repeat(n,1)
    target+=.0002*torch.randn(n,256,generator=generator,dtype=torch.float64)
    # Coupled population summaries: no claim these toy values are the real loss.
    wave=torch.stack([(y.mean(0)-target.mean(0)).square().mean()
                     +(y.var(0,unbiased=False)-target.var(0,unbiased=False)).square().mean() for y in windows]).mean()
    q=2*torch.stack([(y.sum(1).mean()-target.sum(1).mean()).square() for y in windows]).mean()
    targets=(weight,bias)+windows+(shared_signal,)
    gradients={}; endpoints={}; signal_adjoints={}
    artificial_queries=[]
    for name,loss in [('wave',wave),('weighted_Q_toy',q)]:
        values=torch.autograd.grad(loss,targets,retain_graph=True)
        artificial_queries.append(name+'_params_windows_S_one_call')
        gradients[name]=flat(values[:2]);endpoints[name]=values[2:5];signal_adjoints[name]=values[5]
    errors={}
    checks=[]
    def check(name,condition):
        assert bool(condition),name
        checks.append({'name':name,'passed':True})
    check('two_leaf_tensors_1288_entries',weight.numel()+bias.numel()==1288)
    check('hard_STE_ADC_and_windows_byte_values_equal',all(torch.equal(a['adc_carrier'],b['adc_carrier'])
        and torch.equal(a['window_V'],b['window_V']) and torch.equal(a['peak_index'],b['peak_index']) for a,b in zip(observations,hard)))
    check('nonzero_intermediate_prefix_adjoint_is_retained',any(bool(g[:,:100].ne(0).any()) for g in signal_adjoints.values()))
    for name in gradients:
        carrier_pullback=sum(analytic_window_pullback(e,o['peak_index'],shared_signal+noise)
                             for e,o,noise in zip(endpoints[name],observations,noises))
        errors[name+'_analytic_dL_dS']=float((carrier_pullback-signal_adjoints[name]).abs().max())
        d_coeff=(carrier_pullback@bases.T)*(1-coefficients.detach().square())
        analytic_theta=flat([(d_coeff.T@features).reshape_as(weight),d_coeff.sum(0)])
        errors[name+'_analytic_1288_gradient']=float((analytic_theta-gradients[name]).abs().max())
        check(name+'_ancestor_endpoint_and_signal_chain',errors[name+'_analytic_dL_dS']<1e-12
              and errors[name+'_analytic_1288_gradient']<1e-12)
        masks=[(torch.arange(n)%3 == h).double()[:,None] for h in range(3)]
        changed=torch.autograd.grad(windows,(weight,bias),grad_outputs=tuple(e.detach()*m for e,m in zip(endpoints[name],masks)),retain_graph=True)
        artificial_queries.append(name+'_joint_three_window_mask_VJP')
        changed_signal=sum(analytic_window_pullback(e*m,o['peak_index'],shared_signal+noise)
                           for e,m,o,noise in zip(endpoints[name],masks,observations,noises))
        dc=(changed_signal@bases.T)*(1-coefficients.detach().square())
        changed_analytic=flat([(dc.T@features).reshape_as(weight),dc.sum(0)])
        errors[name+'_masked_theta_VJP']=float((flat(changed)-changed_analytic).abs().max())
        unchanged=gradients[name]-flat(changed)
        check(name+'_mask_VJP_has_actual_chain_not_norm_fraction',errors[name+'_masked_theta_VJP']<1e-12
              and torch.allclose(flat(changed)+unchanged,gradients[name],atol=1e-14,rtol=0))
    probe_carrier=torch.tensor([-.61,ADC_MIN,-.49,0.,.49,ADC_MAX,.61],dtype=torch.float64,requires_grad=True)
    hard_value=digitize(probe_carrier,ste=False)
    ste_value=digitize(probe_carrier,ste=True)
    hard_derivative=torch.autograd.grad(hard_value.sum(),probe_carrier,retain_graph=True)[0]
    ste_derivative=torch.autograd.grad(ste_value.sum(),probe_carrier)[0]
    artificial_queries+=['hard_ADC_derivative_toy','STE_ADC_derivative_toy']
    check('hard_round_zero_STE_inclusive_clamp_boundary',torch.equal(hard_value,ste_value)
          and torch.equal(hard_derivative,torch.zeros_like(hard_derivative))
          and torch.equal(ste_derivative,torch.tensor([0,1,1,1,1,1,0],dtype=torch.float64)))
    all_decompositions=[q_decomposition(shared_signal.detach(),noise,o) for noise,o in zip(noises,hard)]
    errors['Q_decomposition']=max(float(v['Q_identity_residual'].abs().max()) for v in all_decompositions)
    errors['tail_decomposition']=max(float(v['tail_identity_residual'].abs().max()) for v in all_decompositions)
    check('Q_and_tail_prefix_ADCdelta_baseline_identity',errors['Q_decomposition']<1e-12 and errors['tail_decomposition']<1e-12)
    raw_source=torch.full((2,384),.02,dtype=torch.float64)
    raw_source[:,110:366]+=.001
    centered=raw_source-raw_source[:,:100].mean(1,keepdim=True)
    correct_q=canonical_charge008(centered)
    wrong_raw_q=canonical_charge008(raw_source)
    check('SOURCE_strata_follow_centered110to366_original_function',torch.allclose(correct_q,torch.full((2,),.256,dtype=torch.float64),atol=1e-14,rtol=0)
          and torch.allclose(wrong_raw_q-correct_q,torch.full((2,),5.12,dtype=torch.float64),atol=1e-14,rtol=0))
    check('no_grad_accumulation_and_parameters_unchanged',weight.grad is None and bias.grad is None
          and torch.equal(weight.detach(),initial_weight) and torch.equal(bias.detach(),initial_bias))
    receipt={'schema':'n01/mechanism012-independent-chain-artificial/v1','status':'PASS','UTC':datetime.now(timezone.utc).isoformat(),
        'artificial_only':True,'checks':checks,'passed_checks':len(checks),'numeric_errors':errors,
        'synthetic_autograd_queries':len(artificial_queries),'synthetic_query_names':artificial_queries,
        'synthetic_signal_rows':n,'synthetic_parameter_entries':1288,'new_actual_scientific_queries':0,
        'trained_NN_instances':0,'original_scientific_array_decodes':0,'checkpoint_loads':0,'scientific_NN_forward_rows':0,
        'optimizer_or_sampler_instances':0,'scientific_gradient_queries':0,'updates':0,'DEV_reads':0,'SSH_calls':0,
        'no_new_011_CDF_evaluations':True,'CPU_threads_effective_torch':torch.get_num_threads(),
        'TEMP_TMP':os.environ.get('TEMP'),'script_sha256':sha(__file__),
        'helper_sha256_at_check':sha(PROJECT/'scripts/mechanism012_losses.py'),
        'helper_functions_under_check':['analytic_window_pullback','q_decomposition'],
        'old_model008_source_sha256':sha(PROJECT/'src/n01/model008.py'),
        'scope':'Artificial algebraic head tensors only. This does not prove the native trained1288 graph, saved-bank equality, model freeze, loss semantics or GPU resource execution.'}
    path=OUTPUT/'ENGINEERING_RECEIPT.json'
    path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','passed_checks':len(checks),'receipt':str(path),'sha256':sha(path),'errors':errors,'science_queries':0}))


if __name__=='__main__':
    main()
