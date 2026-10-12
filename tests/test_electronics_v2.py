"""Artificial-process checks for the independent stationary electronics candidate."""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from n01.electronics_v2 import ElectronicsNoiseV2, fit_electronics_initialization, make_n01_response_v2
from n01.model import charge


torch.set_num_threads(2)


def generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def candidate(**updates):
    return ElectronicsNoiseV2(dict(rms=.002,log_scale_std=.12,offset_mean=.0003,offset_std=.0002)|updates)


class ElectronicsV2Tests(unittest.TestCase):
    @torch.no_grad()
    def test_free_FIR_can_reach_strong_positive_and_negative_correlation(self):
        m=candidate()
        with torch.no_grad():
            h=torch.sin(torch.arange(1,66,dtype=torch.float32)*math.pi/66)
            m.filter.raw.copy_(h)
        self.assertAlmostEqual(float(m.filter.taps().detach().square().sum()),1.,places=13)
        cov=m.autocovariance(1,include_offset=False)
        self.assertAlmostEqual(float(cov[1]/cov[0]),math.cos(math.pi/66),places=7)
        self.assertGreater(float(cov[1]/cov[0]),.99)
        with torch.no_grad():
            m.filter.raw.zero_();m.filter.raw[:2].copy_(torch.tensor([.8,-.6]))
        cov=m.autocovariance(1,include_offset=False)
        self.assertAlmostEqual(float(cov[1]/cov[0]),-.48,places=7)

    @torch.no_grad()
    def test_artificial_MA_process_covariance_and_crop_are_stationary(self):
        m=candidate()
        with torch.no_grad():
            m.filter.raw.zero_();m.filter.raw[:2].copy_(torch.tensor([.8,.6]))
        z=m.sample_noise(10000,generator=generator(172),length=100)
        with torch.no_grad(): y=m(z,length=100)[:,0]
        centered=y-m.offset_mean
        expected=m.autocovariance(3).detach()
        for lag in range(4):
            empirical=(centered[:,:100-lag]*centered[:,lag:]).mean()
            self.assertLess(abs(float(empirical-expected[lag])),float(expected[0])*.025)
        # Real history makes the beginning and end have the same variance.
        for k in (0,50,99):
            self.assertLess(abs(float(centered[:,k].var(unbiased=True)/expected[0]-1)),.045)
        self.assertLess(abs(float(y.mean()-m.offset_mean)),.000025)
        self.assertGreater(float(expected[1]/expected[0]),.4)

    @torch.no_grad()
    def test_lognormal_event_RMS_has_correct_arithmetic_mean(self):
        m=candidate(log_scale_std=.55)
        z=m.sample_noise(30000,generator=generator(34),length=1)
        spread=m.log_scale_std
        sigma=torch.exp(m.log_rms.double()+spread*z["electronics_scale"].double()-.5*spread.square())
        self.assertLess(abs(float(sigma.mean()/m.rms_mean)-1),.012)
        theory=m.theoretical_stats()
        self.assertLess(abs(float(sigma.square().mean()/theory["sigma_squared_mean"])-1),.025)

    @torch.no_grad()
    def test_replay_full_dictionary_compatibility_and_batch_order(self):
        m=candidate()
        response=make_n01_response_v2(variant="deterministic",electronics=m)
        z=response.sample_noise(7,generator=generator(5))
        self.assertEqual(z["electronics_white"].shape,(7,1,544))
        self.assertEqual(z["electronics_scale"].shape,(7,1,1))
        a=m(z)
        b=m(z)
        self.assertTrue(torch.equal(a,b))
        z2=dict(z);z2["electronics_slope"]=torch.full_like(z["electronics_slope"],float("nan"))
        z2["phys"]=z2["phys"]+100
        z2["local_white"]=z2["local_white"]-42
        self.assertTrue(torch.equal(a,m(z2)))
        order=torch.tensor([4,0,6,2,1,5,3])
        reordered={k:v[order] for k,v in z.items()}
        self.assertTrue(torch.equal(a[order],m(reordered)))
        self.assertTrue(torch.equal(a[...,:100],m(z,length=100)))
        small={k:v.clone() for k,v in z.items()};small["electronics_white"]=small["electronics_white"][...,:164]
        self.assertTrue(torch.equal(a[...,:100],m(small,length=100)))
        n1=m.sample_noise(4,generator=generator(8),length=100)
        n2=m.sample_noise(4,generator=generator(8),length=100)
        self.assertEqual(n1["electronics_white"].shape,(4,1,164))
        self.assertTrue(all(torch.equal(n1[k],n2[k]) for k in n1))

    @torch.no_grad()
    def test_electronics_is_independent_of_source_amplitude_and_shape(self):
        r=make_n01_response_v2(variant="deterministic",electronics_initialization=dict(rms=.001,log_scale_std=.02,offset_mean=0.,offset_std=.0002))
        t=torch.arange(480,dtype=torch.float32)
        base=torch.where(t>=110,.12*torch.exp(-(t-140).square()/400),torch.zeros_like(t))
        source=base[None].repeat(3,1)
        changed=source*torch.tensor([.6,1.5,2.])[:,None]
        changed[2]=torch.where(t>=110,.2*torch.exp(-(t-158).square()/650),torch.zeros_like(t))
        noise=r.sample_noise(3,generator=generator(11))
        with torch.no_grad():
            before=r(source,noise=noise,adc_ste=False)
            after=r(changed,noise=noise,adc_ste=False)
        self.assertTrue(torch.equal(before["electronics_noise"],after["electronics_noise"]))
        self.assertFalse(torch.equal(before["q_source"],after["q_source"]))
        self.assertFalse(torch.equal(before["pre_adc"],after["pre_adc"]))
        self.assertTrue(all(not p.requires_grad for p in r.electronics.parameters()))
        r.set_stage("baseline")
        self.assertTrue(all(p.requires_grad for p in r.electronics.parameters()))

    @torch.no_grad()
    def test_nonzero_Q_fluctuations_match_stationary_noise_prediction(self):
        m=candidate(rms=.003,offset_std=.03)
        with torch.no_grad():
            m.filter.raw.zero_();m.filter.raw[:2].copy_(torch.tensor([.8,.6]))
            y=m(m.sample_noise(5000,generator=generator(276)))
        q=charge(y).flatten()
        w=torch.zeros(480,dtype=torch.float64);w[:100]=-3;w[100:400]=1
        predicted=m.linear_statistic_variance(w).detach()
        self.assertGreater(float(predicted),.01)
        self.assertLess(abs(float(q.var(unbiased=True)/predicted-1)),.065)
        self.assertGreater(float(q.std()),.1)
        self.assertTrue(bool((q>0).any() and (q<0).any()))
        # Event-offset variance cancels Q while remaining in baseline means.
        with torch.no_grad():
            previous=m.linear_statistic_variance(w).clone()
            m.log_offset_std.add_(math.log(10.))
        self.assertTrue(torch.equal(previous,m.linear_statistic_variance(w)))

    @torch.no_grad()
    def test_extended_history_and_no_circular_wrap(self):
        m=candidate(offset_mean=0.)
        with torch.no_grad():
            m.filter.raw.zero_();m.filter.raw[:2].copy_(torch.tensor([1.,1.]))
        z={"electronics_white":torch.zeros(1,1,164),"electronics_scale":torch.zeros(1,1,1),"electronics_offset":torch.zeros(1,1,1)}
        z["electronics_white"][0,0,63]=1.
        y=m(z,length=100)
        self.assertGreater(float(y[0,0,0]),0.)
        self.assertTrue(torch.equal(y[...,1:],torch.zeros_like(y[...,1:])))
        z["electronics_white"].zero_();z["electronics_white"][0,0,-1]=1.
        y=m(z,length=100)
        self.assertGreater(float(y[0,0,-1]),0.)
        self.assertTrue(torch.equal(y[...,:-1],torch.zeros_like(y[...,:-1])))

    def test_joint_gradients_and_well_scaled_offset_mean(self):
        m=candidate()
        z=m.sample_noise(32,generator=generator(9),length=100)
        y=m(z,length=100)
        loss=y.mean()/m.voltage_scale+(y.square().mean()+.3*(y[...,1:]*y[...,:-1]).mean())/m.voltage_scale.square()
        loss.backward()
        for name,p in m.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
            self.assertGreater(float(p.grad.abs().sum()),0.,name)
        self.assertFalse(any("slope" in name or "drift" in name for name,_ in m.named_parameters()))
        old=float(m.offset_mean.detach())
        optimizer=torch.optim.Adam(m.parameters(),lr=1e-3)
        optimizer.step()
        delta=abs(float(m.offset_mean.detach())-old)
        self.assertLess(delta,float(m.voltage_scale)*.00101)
        self.assertGreater(delta,float(m.voltage_scale)*.00099)
        self.assertAlmostEqual(float(m.filter.taps().detach().square().sum()),1.,places=13)

    @torch.no_grad()
    def test_TRAIN_initializer_deducts_mean_sampling_noise_without_OLS(self):
        g=generator(10)
        sigma=.002;offset_std=.0004;offset_mean=.0006
        baseline=offset_mean+offset_std*torch.randn(12000,1,generator=g,dtype=torch.float64)+sigma*torch.randn(12000,100,generator=g,dtype=torch.float64)
        initialization=fit_electronics_initialization(baseline)
        self.assertEqual(initialization["log_scale_std"],.02)
        self.assertLess(abs(initialization["rms"]/sigma-1),.02)
        self.assertLess(abs(initialization["offset_std"]/offset_std-1),.06)
        self.assertLess(abs(initialization["offset_mean"]-offset_mean),.00002)
        meta=initialization["metadata"]
        self.assertGreater(meta["stationary_mean_variance_subtracted"],0.)
        self.assertLess(initialization["offset_std"]**2,meta["observed_baseline_mean_variance"])
        m=ElectronicsNoiseV2(initialization)
        w=torch.ones(100,dtype=torch.float64)/100
        self.assertLess(abs(float(m.linear_statistic_variance(w))-meta["observed_baseline_mean_variance"]),1e-12)
        with self.assertRaises(ValueError): ElectronicsNoiseV2({"electronics_offset_std":.01,"electronics_slope_std":.02})

    @torch.no_grad()
    def test_invalid_rows_are_explicit_not_clipped_or_dropped(self):
        m=candidate()
        z=m.sample_noise(2,generator=generator(12),length=100)
        z["electronics_scale"][1,0,0]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):m(z,length=100)
        with self.assertRaises(ValueError):m.sample_noise(0)
        with torch.no_grad():m.filter.raw.zero_()
        z["electronics_scale"].zero_()
        with self.assertRaisesRegex(ValueError,"nonzero L2"):m(z,length=100)
        with self.assertRaises(ValueError):fit_electronics_initialization(torch.zeros(3,100))


if __name__=="__main__": unittest.main()
