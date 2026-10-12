"""Fabricated-process semantics for strict100 observed-baseline covariance noise."""
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from n01.electronics_v3 import ObservedBaselineNoiseV3,fit_baseline_initialization_v3
from n01.model import digitize


def generator(seed):return torch.Generator(device="cpu").manual_seed(seed)


def known_initialization():
    t=torch.linspace(-1,1,100,dtype=torch.float64)
    # A rank-one mode couples average level and slope negatively. Nonconstant
    # diagonal variance also breaks the stationary Toeplitz restriction.
    u=1-.7*t
    covariance=1e-6*(torch.diag(.2+.5*(t+1)**2)+.7*torch.outer(u,u))
    mean=.00012+.00004*t+.00005*t.square()
    return dict(length=100,mean_V=mean.tolist(),cholesky_V=torch.linalg.cholesky(covariance).tolist(),
        voltage_scale=float(covariance.diagonal().mean().sqrt()),normalized_diagonal_floor=1e-6,
        initial_log_scale_std=.02,metadata=dict(role="fabricated",events=0)),mean,covariance


class ElectronicsV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_full_registered_initializer_ddof1_ridge_and_no_dequantization(self):
        x=torch.randn(700,100,generator=generator(8),dtype=torch.float64)*.002
        t=torch.linspace(-1,1,100,dtype=torch.float64)
        x=x*(1+.3*t)[None]+.0003*t
        config=dict(type="ObservedBaselineNoiseV3",length=100,covariance_ridge_relative=1e-6,
                    normalized_diagonal_floor=1e-6,initial_log_scale_std=.02,scale_second_moment=1,
                    init_covariance_ddof=1,full_mean_and_cholesky_trainable=True)
        init=fit_baseline_initialization_v3(x,**config)
        centered=x-x.mean(0);empirical=centered.T@centered/699
        ridge=1e-6*empirical.diagonal().mean()
        model=ObservedBaselineNoiseV3(init)
        torch.testing.assert_close(model.mean_vector(),x.mean(0),rtol=1e-13,atol=1e-18)
        torch.testing.assert_close(model.covariance(),empirical+ridge*torch.eye(100,dtype=torch.float64),rtol=1e-11,atol=1e-20)
        self.assertEqual(init["metadata"]["events"],700)
        self.assertEqual(init["metadata"]["covariance_ddof"],1)
        self.assertTrue(init["metadata"]["no_ADC_noise_subtraction"])
        self.assertAlmostEqual(init["initial_log_scale_std"],.02)

    def test_exact_parameter_count_dtype_and_analytic_negative_mean_slope_covariance(self):
        init,mean,covariance=known_initialization();model=ObservedBaselineNoiseV3(init)
        self.assertEqual(sum(p.numel() for p in model.parameters()),5151)
        self.assertTrue(all(p.dtype==torch.float64 and p.requires_grad for p in model.parameters()))
        self.assertTrue(all("slope" not in name for name,_ in model.named_parameters()))
        torch.testing.assert_close(model.covariance(),covariance,rtol=1e-12,atol=1e-20)
        mean_weight=torch.ones(100,dtype=torch.float64)/100
        t=torch.linspace(-1,1,100,dtype=torch.float64);slope_weight=t/t.square().sum()
        self.assertLess(float(mean_weight@model.covariance().detach()@slope_weight),0.)
        self.assertGreater(float(torch.linalg.eigvalsh(model.covariance().detach()).min()),0.)
        self.assertFalse(model.supports_response480)

    @torch.no_grad()
    def test_Monte_Carlo_mean_and_mean_slope_joint_covariance(self):
        init,mean,covariance=known_initialization();model=ObservedBaselineNoiseV3(init)
        z=model.sample_noise(20000,generator=generator(18))
        y=model(z)[:,0]
        t=torch.linspace(-1,1,100,dtype=torch.float64)
        weights=torch.stack((torch.ones(100,dtype=torch.float64)/100,t/t.square().sum()),1)
        projected=(y-mean)@weights
        observed=torch.cov(projected.T)
        expected=weights.T@covariance@weights
        torch.testing.assert_close(observed,expected,rtol=.045,atol=1e-10)
        self.assertLess(float(observed[0,1]),0.)
        self.assertLess(float((y.mean(0)-mean).abs().max()),.000065)

    @torch.no_grad()
    def test_second_moment_scale_normalization_covariance_independent_of_eta(self):
        init,mean,covariance=known_initialization();model=ObservedBaselineNoiseV3(init)
        original=model.covariance().clone()
        eta=.3
        model.raw_log_scale_std.copy_(torch.tensor(eta+math.log(-math.expm1(-eta)),dtype=torch.float64))
        theory=model.theoretical_stats()
        self.assertEqual(float(theory["scale_second_moment"]),1.)
        torch.testing.assert_close(model.covariance(),original,rtol=0,atol=0)
        self.assertAlmostEqual(float(theory["scale_mean"]),math.exp(-.5*eta**2),places=14)
        self.assertAlmostEqual(float(theory["scale_fourth_moment"]),math.exp(4*eta**2),places=14)
        u=torch.randn(80000,generator=generator(20),dtype=torch.float64)
        scale=torch.exp(eta*u-eta**2)
        self.assertLess(abs(float(scale.square().mean())-1.),.013)
        self.assertLess(abs(float(scale.pow(4).mean())/math.exp(4*eta**2)-1.),.025)

    @torch.no_grad()
    def test_explicit_noise_replay_batch_reorder_and_checkpoint_restore(self):
        init,_,_=known_initialization();model=ObservedBaselineNoiseV3(init)
        noise=model.sample_noise(8,generator=generator(33),length=100)
        again=model.sample_noise(8,generator=generator(33),length=100)
        self.assertTrue(all(torch.equal(noise[k],again[k]) for k in noise))
        before={k:v.clone() for k,v in model.state_dict().items()}
        a=model(noise,length=100);b=model(noise,length=100)
        self.assertTrue(torch.equal(a,b));self.assertEqual(a.shape,(8,1,100))
        order=torch.tensor([5,1,0,2,7,3,6,4])
        reordered=model({k:v[order] for k,v in noise.items()})
        # BLAS may change last-bit reduction order under row permutations;
        # mathematical batch independence is tested at float64 precision.
        torch.testing.assert_close(reordered,a[order],rtol=2e-13,atol=1e-17)
        restored=ObservedBaselineNoiseV3(init);restored.load_state_dict(before,strict=True)
        self.assertTrue(torch.equal(a,restored(noise)))
        self.assertTrue(torch.equal(digitize(a,ste=False),digitize(restored(noise),ste=False)))
        self.assertTrue(all(torch.equal(model.state_dict()[k],v) for k,v in before.items()))

    def test_all_parameter_groups_finite_gradients_and_mean_step_units(self):
        init,_,_=known_initialization();model=ObservedBaselineNoiseV3(init)
        y=model(model.sample_noise(64,generator=generator(50)))
        units=y/model.voltage_scale
        loss=units.mean()+units.square().mean()+.2*(units[...,:-1]*units[...,1:]).mean()
        loss.backward()
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
            self.assertGreater(float(p.grad.abs().sum()),0.,name)
        mean_before=model.mean_vector().detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        optimizer.step()
        maximum=float((model.mean_vector().detach()-mean_before).abs().max())
        self.assertLessEqual(maximum,float(model.voltage_scale)*.001001)
        self.assertTrue(bool((model.cholesky_factor().detach().diagonal()>0).all()))
        self.assertTrue(bool(torch.isfinite(model.covariance().detach()).all()))

    def test_strict100_rejects480_and_no_ID_source_fields(self):
        init,_,_=known_initialization();model=ObservedBaselineNoiseV3(init)
        noise=model.sample_noise(3,generator=generator(21))
        for length in (480,99,101,100.,None):
            with self.assertRaisesRegex(ValueError,"exactly100"):model.sample_noise(3,length=length)
            with self.assertRaisesRegex(ValueError,"exactly100"):model(noise,length=length)
        with self.assertRaises(ValueError):model(noise|dict(event_id=torch.arange(3)))
        with self.assertRaises(ValueError):fit_baseline_initialization_v3(torch.randn(3,480))
        with self.assertRaises(ValueError):ObservedBaselineNoiseV3(init|dict(length=480))

    def test_invalid_rows_and_degenerate_covariance_fail_explicitly(self):
        init,_,_=known_initialization();model=ObservedBaselineNoiseV3(init)
        noise=model.sample_noise(3,generator=generator(21))
        noise["baseline_gaussian"][2,0,5]=float("nan")
        with self.assertRaisesRegex(ValueError,"Nonfinite"):model(noise)
        bad=torch.zeros(5,100,dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,"positive finite"):fit_baseline_initialization_v3(bad)
        bad[2,8]=float("nan")
        with self.assertRaisesRegex(ValueError,"finite Nx100"):fit_baseline_initialization_v3(bad)
        with self.assertRaises(ValueError):fit_baseline_initialization_v3(torch.randn(5,100),init_covariance_ddof=0)
        with self.assertRaises(ValueError):fit_baseline_initialization_v3(torch.randn(5,100),scale_second_moment=2)


if __name__=='__main__':unittest.main()
