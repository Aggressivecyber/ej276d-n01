"""Synthetic ACF20 semantics; no scientific arrays/optimizers are opened."""
from pathlib import Path
import hashlib
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from n01.acf20 import acf20,mean_acf20,ACF20Loss,VARIANCE_FLOOR
from n01.noise_validation import baseline_statistics
from n01.model import digitize


def gen(seed):return torch.Generator(device='cpu').manual_seed(seed)


class ACF20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.legacy={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [ROOT/'src/n01/losses.py',ROOT/'src/n01/losses_v2.py',ROOT/'src/n01/noise_validation.py',ROOT/'src/n01/model.py']}

    @classmethod
    def tearDownClass(cls):
        for p,digest in cls.legacy.items():
            if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
                raise AssertionError('Legacy source changed during ACF20 tests: '+str(p))

    def assert_numpy_agreement(self,x):
        actual=mean_acf20(x).detach().cpu().numpy()
        y=x.detach().cpu().numpy()
        if y.ndim==3:y=y[:,0]
        expected=baseline_statistics(y)['acf_mean']
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-14)
        self.assertEqual(acf20(x).shape,(len(x),20))
        self.assertEqual(mean_acf20(x).shape,(20,))
        self.assertEqual(acf20(x).dtype,torch.float64)

    def test_numpy_equivalence_white_correlated_quantized_and_mixed_amplitude(self):
        white=torch.randn(512,100,generator=gen(2),dtype=torch.float64)*.0006
        correlated=torch.randn(512,101,generator=gen(3),dtype=torch.float64)*.0006
        correlated=.8*correlated[:,:100]+.6*correlated[:,1:]
        scale=torch.exp(torch.randn(512,1,generator=gen(4),dtype=torch.float64))
        for x in (white,correlated,digitize(white,ste=False),correlated*scale+torch.linspace(-.001,.001,512)[:,None]):
            self.assert_numpy_agreement(x)

    def test_float32_adaptation_channels_and_input_gradient_dtype(self):
        x=(torch.randn(5,1,100,generator=gen(5))*.0007).requires_grad_()
        self.assert_numpy_agreement(x)
        value=mean_acf20(x).square().sum();value.backward()
        self.assertEqual(x.grad.dtype,torch.float32)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_constant_near_constant_and_quantization_floor(self):
        constant=torch.full((4,100),.125,dtype=torch.float64)
        near=constant+torch.randn(4,100,generator=gen(6),dtype=torch.float64)*1e-9
        self.assertTrue(torch.equal(acf20(constant),torch.zeros(4,20,dtype=torch.float64)))
        self.assert_numpy_agreement(near)
        b=near-near.mean(1,keepdim=True)
        expected=torch.stack([(b[:,:-k]*b[:,k:]).mean(1)/VARIANCE_FLOOR for k in range(1,21)],1)
        torch.testing.assert_close(acf20(near),expected,rtol=0,atol=0)
        near.requires_grad_();acf20(near).square().sum().backward()
        self.assertTrue(bool(torch.isfinite(near.grad).all()))

    def test_each_lag_uses_overlap_divisor_not100(self):
        x=torch.arange(100,dtype=torch.float64)[None]
        b=x-x.mean(1,keepdim=True)
        variance=b.square().mean(1)
        result=acf20(x)
        overlap=(b[:,:-20]*b[:,20:]).mean(1)/variance
        wrong=(b[:,:-20]*b[:,20:]).sum(1)/100/variance
        torch.testing.assert_close(result[:,19],overlap,rtol=0,atol=0)
        self.assertGreater(float((result[:,19]-wrong).abs().max()),.05)

    def test_event_average_differs_from_pooled_covariance_ratio(self):
        white=torch.randn(1,100,generator=gen(7),dtype=torch.float64)
        t=torch.arange(100,dtype=torch.float64)
        smooth=20*torch.sin(t*.06)[None]
        x=torch.cat((white,smooth))
        expected=acf20(x).mean(0)
        torch.testing.assert_close(mean_acf20(x),expected,rtol=0,atol=0)
        b=x-x.mean(1,keepdim=True)
        pooled=torch.stack([(b[:,:-k]*b[:,k:]).mean()/b.square().mean() for k in range(1,21)])
        self.assertGreater(float((expected-pooled).abs().max()),.3)

    def test_finite_window_demeaned_white_noise_mean_is_not_zero(self):
        x=torch.randn(20000,100,generator=gen(8),dtype=torch.float64)
        result=mean_acf20(x)
        # For Gaussian white samples, expected off-diagonal demeaned ACF is
        # -1/(100-1), even though the unobserved process autocorrelation is zero.
        expected=torch.full((20,),-1/99,dtype=torch.float64)
        torch.testing.assert_close(result,expected,rtol=0,atol=.003)
        self.assertLess(float(result.mean()),-.005)

    def test_gradcheck_and_loss_formula_with_fixed_detached_target(self):
        x=(torch.randn(2,100,generator=gen(9),dtype=torch.float64)*.03).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(mean_acf20,(x,),eps=1e-6,atol=2e-5,rtol=1e-4))
        target=torch.linspace(-.1,.1,20,dtype=torch.float64,requires_grad=True)
        objective=ACF20Loss(target)
        target_before=objective.target_acf20.clone()
        error=(mean_acf20(x)-target.detach())/.015
        expected=.5*(error.square().mean()+error.square().max())
        torch.testing.assert_close(objective(x),expected,rtol=0,atol=0)
        self.assertFalse(objective.target_acf20.requires_grad)
        self.assertEqual(list(objective.parameters()),[])
        with torch.no_grad():target.add_(1.)
        self.assertTrue(torch.equal(objective.target_acf20,target_before))
        objective(x).backward()
        self.assertIsNone(target.grad)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_matching_target_has_zero_loss_and_restore_replays(self):
        x=torch.randn(7,100,generator=gen(10),dtype=torch.float64)
        target=mean_acf20(x)
        a=ACF20Loss(target)
        self.assertEqual(float(a(x)),0.)
        b=ACF20Loss(torch.zeros(20,dtype=torch.float64));b.load_state_dict(a.state_dict(),strict=True)
        y=x+.3*torch.arange(100,dtype=torch.float64)[None]/100
        self.assertTrue(torch.equal(a(y),b(y)))
        self.assertEqual(float(a.scale),.015);self.assertEqual(float(a.coefficient),1.)

    def test_JSON_list_target_keeps_float64_precision(self):
        target=torch.linspace(-.0123456789012345,.0234567890123456,20,dtype=torch.float64)
        tensor_target=ACF20Loss(target)
        list_target=ACF20Loss(target.tolist())
        self.assertTrue(torch.equal(tensor_target.target_acf20,list_target.target_acf20))
        x=torch.randn(3,100,generator=gen(11),dtype=torch.float64)
        self.assertTrue(torch.equal(tensor_target(x),list_target(x)))

    def test_rejects_lengths_channels_empty_nonfinite_and_target_shape(self):
        for x in (torch.zeros(3,480),torch.zeros(3,99),torch.zeros(3,2,100),torch.zeros(0,100),torch.zeros(100),torch.zeros(2,100,dtype=torch.int64)):
            with self.assertRaises(ValueError):acf20(x)
        x=torch.zeros(2,100);x[1,9]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):acf20(x)
        x[1,9]=float('inf')
        with self.assertRaises(ValueError):mean_acf20(x)
        for target in (torch.zeros(19),torch.zeros(1,20),torch.full((20,),float('nan'))):
            with self.assertRaises(ValueError):ACF20Loss(target)


if __name__=='__main__':unittest.main()
