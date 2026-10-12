"""Synthetic006 algebra/replay; no REAL/SIM reads or scientific forwards."""
from pathlib import Path
import copy
import sys
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from n01.data import sha256
from n01.electronics_v3 import ObservedBaselineNoiseV3
from n01 import extension006 as e
from n01.model import charge,digitize


def artificial_model(eta=.3):
    sigma=.0007;idx=torch.arange(100,dtype=torch.float64)
    lower=torch.diag(sigma*(.6+idx/100))
    lower[torch.arange(1,100),torch.arange(99)]=sigma*.16
    mean=.0002*torch.sin(idx/37)+.00004*idx/100
    init=dict(length=100,mean_V=mean.tolist(),cholesky_V=lower.tolist(),voltage_scale=sigma,
              normalized_diagonal_floor=1e-6,initial_log_scale_std=eta,
              metadata=dict(role='fabricated_artificial_nonstationary_process'))
    return ObservedBaselineNoiseV3(init)


def independent_draws(model,n,seed):
    original=model.sample_noise(n,generator=torch.Generator().manual_seed(seed))
    tail=torch.randn(n,1,380,generator=torch.Generator().manual_seed(seed+101))
    return original,tail


class Extension006Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.legacy={p:sha256(p) for p in [ROOT/'src/n01/electronics_v3.py',ROOT/'src/n01/model.py',
                                      ROOT/'src/n01/acf20.py',ROOT/'src/n01/noise_validation.py']}

    @classmethod
    def tearDownClass(cls):
        for p,value in cls.legacy.items():
            if sha256(p)!=value:raise AssertionError('Frozen old source changed: '+str(p))

    def test_original_prefix_and_single_prefix_ADC_are_bitwise_paired_across_hypotheses(self):
        model=artificial_model();extension=e.ConditionalExtension006(model,e.HYPOTHESES006)
        noise,eps=independent_draws(model,7,600)
        original=model(noise,length=100);post=digitize(original,ste=False)
        state=torch.get_rng_state().clone();noise_before={k:v.clone() for k,v in noise.items()};eps_before=eps.clone()
        for name in extension.hypotheses:
            result=extension.extend(noise,eps,name)
            self.assertEqual(result['noise_pre'].shape,(7,1,480));self.assertEqual(result['noise_pre'].dtype,torch.float64)
            self.assertTrue(torch.equal(result['prefix_pre'],original))
            self.assertTrue(torch.equal(result['noise_pre'][...,:100],original))
            self.assertTrue(torch.equal(digitize(result['noise_pre'],ste=False)[...,:100],post))
            self.assertTrue(torch.equal(result['noise_pre'],extension.extend_prefix(original,result['scale'],eps,name)['noise_pre']))
            self.assertFalse(result['noise_pre'].requires_grad)
        self.assertTrue(torch.equal(torch.get_rng_state(),state))
        self.assertTrue(torch.equal(eps,eps_before))
        for key in noise:self.assertTrue(torch.equal(noise[key],noise_before[key]))
        self.assertEqual(set(dict(extension.named_parameters())),{'model005.'+name for name,_ in model.named_parameters()})

    def test_shared_global_scale_controls_prefix_and_new_innovation_together(self):
        model=artificial_model(.45);extension=e.ConditionalExtension006(model,e.HYPOTHESES006)
        noise,eps=independent_draws(model,1,601)
        noise={key:value.expand(3,*value.shape[1:]).clone() for key,value in noise.items()}
        noise['baseline_scale']=torch.tensor([[[-1.]],[[0.]],[[1.]]])
        eps=eps.expand(3,1,380).clone()
        result=extension.extend(noise,eps,'slow');theory=extension.theoretical('slow')
        centered=(result['noise_pre']-theory['mean'][None,None])/result['scale']
        torch.testing.assert_close(centered,centered[:1].expand_as(centered),rtol=2e-13,atol=1e-18)
        expected_scale=torch.exp(model.log_scale_std*noise['baseline_scale'].double()-model.log_scale_std.square())
        self.assertTrue(torch.equal(result['scale'],expected_scale))
        predicted=theory['mean'][100:]+(result['prefix_pre']-model.mean_vector()[None,None])@theory['A'].T
        expected_innovation=result['scale']*(eps.double()@theory['cholesky_R'].T)
        torch.testing.assert_close(result['noise_pre'][...,100:]-predicted,expected_innovation,rtol=1e-12,atol=1e-18)
        zero=extension.extend(noise,torch.zeros_like(eps),'slow')
        self.assertTrue(torch.equal(zero['prefix_pre'],result['prefix_pre']))
        self.assertGreater(float((zero['noise_pre'][...,100:]-result['noise_pre'][...,100:]).abs().max()),0.)

    def test_actual_joint_covariance_factor_and_charge_seam_algebra(self):
        model=artificial_model();extension=e.ConditionalExtension006(model,e.HYPOTHESES006)
        L=model.cholesky_factor().detach();C=L@L.T
        q=torch.zeros(480,dtype=torch.float64);q[:100]=-3;q[100:400]=1
        tail=torch.zeros(480,dtype=torch.float64);tail[:100]=-2.5;tail[150:400]=1
        for name in extension.hypotheses:
            theory=extension.theoretical(name);A=theory['A'];B=theory['cholesky_R'];R=theory['R']
            factor=torch.cat((torch.cat((L,torch.zeros(100,380,dtype=torch.float64)),1),torch.cat((A@L,B),1)),0)
            torch.testing.assert_close(theory['covariance'],factor@factor.T,rtol=3e-13,atol=2e-18)
            self.assertTrue(torch.equal(theory['covariance'][:100,:100],C))
            torch.testing.assert_close(B@B.T,R,rtol=3e-13,atol=2e-18)
            self.assertTrue(bool((torch.linalg.eigvalsh(R)>0).all()))
            torch.linalg.cholesky(theory['covariance'])
            # The replacement C is deliberately nonstationary and not K_BB.
            self.assertGreater(float((theory['covariance']-theory['kernel_reference']).abs().max()),1e-8)
            torch.testing.assert_close(theory['variance_Q'],(factor.T@q).square().sum(),rtol=3e-13,atol=1e-17)
            torch.testing.assert_close(theory['variance_tail_charge'],(factor.T@tail).square().sum(),rtol=3e-13,atol=1e-17)
            torch.testing.assert_close(theory['covariance_Q_tail'],(factor.T@q)@(factor.T@tail),rtol=3e-13,atol=1e-17)
            row=A[0].clone();row[99]-=1
            torch.testing.assert_close(theory['seam_increment_variance'],row@C@row+R[0,0],rtol=3e-13,atol=1e-18)
            self.assertEqual(theory['prefix_increment_variance90_99'].shape,(9,))
            self.assertEqual(theory['first_tail_last10_prefix_correlation'].shape,(10,))
            self.assertEqual(float(theory['scale_second_moment']),1.)
            # Returned tensors are copies; modifying a report cannot alter laws.
            theory['mean'].add_(9.)
            self.assertLess(float(extension.theoretical(name)['mean'].abs().max()),.01)

    def test_sampled_charge_variance_and_scale_moments_match_artificial_law(self):
        model=artificial_model(.25);extension=e.ConditionalExtension006(model,e.HYPOTHESES006)
        noise,eps=independent_draws(model,4096,602)
        result=extension.extend(noise,eps,'common_level');theory=extension.theoretical('common_level')
        value=charge(result['noise_pre'][:,0]);expected=theory['variance_Q']
        self.assertLess(abs(float(value.var(unbiased=True)/expected)-1.),.12)
        self.assertLess(abs(float(value.mean()-theory['mean_Q'])),5*float(expected.sqrt())/4096**.5)
        self.assertLess(abs(float(result['scale'].square().mean())-1.),.05)
        self.assertGreater(float(value.var(unbiased=True)),0.)

    def test_signal_component_assembly_has_one_ADC_no_legacy_noise_and_nonzero_noise_Q(self):
        model=artificial_model();extension=e.ConditionalExtension006(model,e.HYPOTHESES006)
        noise,eps=independent_draws(model,3,603)
        t=torch.arange(480,dtype=torch.float64)
        unit=torch.where(t>=116,torch.exp(-(t-116)/35),0.)[None].expand(3,-1).clone()
        unit=unit/charge(unit)[:,None]
        q=torch.tensor([1.,3.,9.],dtype=torch.float64)
        local=torch.zeros_like(unit);local[:,200]=.0001;local[:,201]=-.0001
        poisoned=torch.full_like(unit,float('nan'))
        outputs=dict(qsig=q,unit_shape=unit,signal_noise=local,waveform=poisoned,pre_adc=poisoned,electronics_noise=poisoned)
        prefixes=[]
        for name in extension.hypotheses:
            extended=extension.extend(noise,eps,name)
            with patch.object(e,'digitize',wraps=digitize) as adc:
                result=e.assemble_response(outputs,extended['noise_pre'])
                self.assertEqual(adc.call_count,1)
            self.assertTrue(torch.equal(result['signal_pre'],q[:,None]*unit+local))
            self.assertTrue(torch.equal(result['pre_adc'],q[:,None]*unit+local+extended['noise_pre'][:,0]))
            self.assertTrue(torch.equal(result['waveform'],digitize(result['pre_adc'],ste=False)))
            torch.testing.assert_close(result['charge_identity_residual'],torch.zeros(3,dtype=torch.float64),rtol=0,atol=1e-13)
            self.assertGreater(float(result['noise_charge'].abs().max()),0.)
            self.assertEqual(float(result['signal_baseline_absolute_max']),0.)
            prefixes.append(result['waveform'][:,:100])
        for prefix in prefixes[1:]:self.assertTrue(torch.equal(prefix,prefixes[0]))
        # Adding a baseline component stays visible; it is never gate-zeroed.
        outputs['signal_noise'][:,0]=2/16383
        result=e.assemble_response(outputs,extension.extend(noise,eps,'fast')['noise_pre'])
        self.assertGreater(float(result['signal_baseline_absolute_max']),1/16383)
        outputs['signal_noise'][:,200]=2.
        result=e.assemble_response(outputs,extension.extend(noise,eps,'fast')['noise_pre'])
        self.assertTrue(bool(result['saturation_mask'][:,200].all()))
        self.assertTrue(bool((result['waveform'][:,200]==.5).all()))

    def test_rejects_unregistered_hypotheses_lengths_latent_keys_dtypes_and_nonfinite(self):
        model=artificial_model();extension=e.ConditionalExtension006(model,e.HYPOTHESES006)
        noise,eps=independent_draws(model,2,604)
        for spec in ([],list(e.HYPOTHESES006)[::-1]):
            with self.assertRaises(ValueError):e.ConditionalExtension006(model,spec)
        bad=copy.deepcopy(e.HYPOTHESES006);bad[0]['short_length']=5.
        with self.assertRaises(ValueError):e.ConditionalExtension006(model,bad)
        with self.assertRaises(ValueError):e.ConditionalExtension006(model,e.HYPOTHESES006,ridge_relative=2e-6)
        with self.assertRaises(ValueError):e.ConditionalExtension006(torch.nn.Identity(),e.HYPOTHESES006)
        with self.assertRaises(ValueError):extension.extend(noise,eps,'other')
        for bad_noise in (dict(noise,ID=torch.arange(2)),{'baseline_gaussian':noise['baseline_gaussian']},
                          dict(noise,baseline_gaussian=torch.zeros(2,1,480))):
            with self.assertRaises(ValueError):extension.extend(bad_noise,eps,'fast')
        prefix=model(noise);scale=torch.ones(2,1,1,dtype=torch.float64)
        for bad_eps in (torch.zeros(2,1,379),torch.zeros(2,1,380,dtype=torch.int64),torch.full((2,1,380),float('nan'))):
            with self.assertRaises(ValueError):extension.extend_prefix(prefix,scale,bad_eps,'fast')
        for bad_prefix in (prefix.float(),prefix[:,0],torch.zeros(2,1,480,dtype=torch.float64)):
            with self.assertRaises(ValueError):extension.extend_prefix(bad_prefix,scale,eps,'fast')
        for bad_scale in (scale.float(),scale*0,scale*-1,torch.full_like(scale,float('inf'))):
            with self.assertRaises(ValueError):extension.extend_prefix(prefix,bad_scale,eps,'fast')
        with self.assertRaises(ValueError):model(noise,length=480)

    def test_frozen_state_guard_detects_parameter_and_cached_law_changes(self):
        model=artificial_model();extension=e.ConditionalExtension006(model,e.HYPOTHESES006)
        noise,eps=independent_draws(model,2,605)
        model.mean_units.data[0]+=.1
        with self.assertRaisesRegex(ValueError,'Frozen005 state changed'):extension.extend(noise,eps,'fast')
        with self.assertRaisesRegex(ValueError,'Frozen005 state changed'):extension.theoretical('fast')
        extension=e.ConditionalExtension006(artificial_model(),e.HYPOTHESES006)
        extension.fast_A[0,0]+=1
        with self.assertRaisesRegex(ValueError,'Frozen006 conditional buffer changed'):extension.theoretical('fast')
        extension=e.ConditionalExtension006(artificial_model(),e.HYPOTHESES006)
        extension.to(dtype=torch.float32)
        with self.assertRaises(ValueError):extension.assert_frozen()


if __name__=='__main__':unittest.main()
