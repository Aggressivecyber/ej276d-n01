"""Artificial CPU checks for the frozen slope diagnostic, no scientific run."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"scripts"))
sys.path.insert(0,str(PROJECT/"src"))
import diagnose_n01_slope as sd
from n01.model import N01Response

torch.set_num_threads(2)


def artificial_pulse():
    t=np.arange(480,dtype=np.float64)
    r=np.maximum(t-116,0)
    y=(1-np.exp(-r/6))*np.exp(-r/50)
    return (.08*y/y.max()).astype(np.float32)


class SlopeDiagnosticTests(unittest.TestCase):
    def test_charge_and_tail_algebra_with_nonzero_offset(self):
        a=sd.slope_algebra()
        self.assertAlmostEqual(a["Q_coefficient"],120000/479,places=11)
        self.assertAlmostEqual(a["Q_tail_coefficient"],112500/479,places=11)
        self.assertAlmostEqual(a["Q_fast_coefficient"],7500/479,places=11)
        self.assertAlmostEqual(a["Q_tail_over_Q_coefficient"],15/16,places=14)
        self.assertAlmostEqual(a["baseline_t_mean"],-380/479,places=14)
        self.assertAlmostEqual(a["baseline_t_sum_squares"],333300/229441,places=13)
        wave=.031+.00073*a["t"]
        self.assertAlmostEqual(float(sd.charge(wave)),a["Q_coefficient"]*.00073,places=12)

    def test_psd_first_derivative_and_exact_finite_slope(self):
        wave=artificial_pulse().astype(np.float64)
        a=sd.slope_algebra(); q=sd.charge(wave); p=sd.raw_psd(wave)
        step=1e-8
        numerical=(sd.raw_psd(wave+step*a["t"])-sd.raw_psd(wave-step*a["t"]))/2/step
        derivative=sd.psd_slope_first_order(q,p,1.)
        self.assertAlmostEqual(float(numerical),float(derivative),places=5)
        delta=.0003
        observed=sd.raw_psd(wave+delta*a["t"])-p
        exact=(a["Q_tail_coefficient"]-p*a["Q_coefficient"])*delta/(q+a["Q_coefficient"]*delta)
        self.assertAlmostEqual(float(observed),float(exact),places=13)

    def test_OLS_white_noise_estimation_variance(self):
        rng=np.random.default_rng(433)
        sigma=.00063
        wave=rng.normal(0,sigma,(16000,100))
        observed=sd.ols_baseline(wave)["slope"].var(ddof=1)
        expected=sigma**2/sd.slope_algebra()["baseline_t_sum_squares"]
        self.assertLess(abs(observed/expected-1),.05)
        h=np.zeros(65); h[0]=1
        theory=sd.noise_variance_theory(h=h,sigma2_mean=sigma**2,offset_std=0,slope_std=0)
        self.assertAlmostEqual(theory["OLS_iid_white_estimation_variance"],expected,places=18)
        self.assertAlmostEqual(theory["OLS_colored_estimation_variance"],expected,places=18)
        self.assertAlmostEqual(theory["Q_filtered_white_variance"],1200*sigma**2,places=16)

    def test_independent_drift_baseline_mean_variance(self):
        rng=np.random.default_rng(982)
        sigma,offset_std,slope_std=.00063,.0004,.0006
        t=sd.slope_algebra()["t"][:100]
        b=rng.normal(0,offset_std,(16000,1))+rng.normal(0,slope_std,(16000,1))*t+rng.normal(0,sigma,(16000,100))
        h=np.zeros(65); h[0]=1
        expected=sd.noise_variance_theory(h=h,sigma2_mean=sigma**2,offset_std=offset_std,slope_std=slope_std)
        self.assertLess(abs(b.mean(1).var(ddof=1)/expected["model_baseline_mean_variance"]-1),.05)
        self.assertGreater(expected["model_baseline_mean_variance"],expected["model_baseline_mean_variance_slope_off"])

    def test_declared_cpu_stream_matches_model_sampler(self):
        torch.manual_seed(725)
        model=N01Response()
        recovered,initial,final=sd.restored_noise(13,seed=17,stream_batch=4)
        generator=torch.Generator(device="cpu").manual_seed(17)
        pieces=[model.sample_noise(n,"cpu",generator) for n in (4,4,4,1)]
        for key in sd.KEYS:
            expected=torch.cat([piece[key] for piece in pieces])
            self.assertTrue(torch.equal(recovered[key],expected),key)
        self.assertTrue(np.array_equal(final,generator.get_state().numpy()))
        self.assertFalse(np.array_equal(initial,final))

    def test_nonpositive_and_zero_Q_are_retained(self):
        wave=np.zeros((3,480),dtype=np.float64)
        wave[0,150]=.1; wave[2,150]=-.1
        observed=sd.observables(wave,1e-5)
        self.assertEqual(len(observed["Q"]),3)
        self.assertEqual(observed["Q_nonpositive"].tolist(),[False,True,True])
        self.assertEqual(observed["Q_exact_zero"].tolist(),[False,True,False])
        self.assertTrue(np.isnan(observed["PSD_raw"][1]))
        result=sd.summary(observed["PSD_raw"])
        self.assertEqual(result["rows"],3)
        self.assertEqual(result["nonfinite_rows"],1)
        self.assertIsNone(result["mean"])
        self.assertTrue(np.isfinite(observed["PSD_guarded"]).all())

    def test_paired_forward_changes_only_slope_and_ddof1_decomposition(self):
        for variant in ("full","deterministic"):
            with self.subTest(variant=variant):
                torch.manual_seed(571)
                model=N01Response(variant=variant)
                model.set_stage("low_learning_rate")
                model.eval().requires_grad_(False)
                state=sd.state_hash(model)
                noise=model.sample_noise(4,"cpu",torch.Generator().manual_seed(192))
                intervention=sd.zero_random_slope(noise)
                for key in sd.KEYS:
                    if key=="electronics_slope": self.assertEqual(float(intervention[key].abs().sum()),0)
                    else: self.assertTrue(torch.equal(noise[key],intervention[key]))
                self.assertGreater(float(noise["electronics_slope"].abs().sum()),0)
                x=torch.from_numpy(np.repeat(artificial_pulse()[None],4,axis=0))
                result=sd.paired_forward(model,x,noise,1e-5)
                self.assertTrue(result["non_electronics_branches_bitwise_identical"].all())
                self.assertLess(np.abs(result["preADC_Q_difference_residual"]).max(),1e-12)
                self.assertLess(np.abs(result["preADC_PSD_exact_residual"]).max(),1e-12)
                self.assertEqual(state,sd.state_hash(model))
                # Add the old-reference channels solely for metadata comparison.
                result["old_baseline_Q"]=result["on_Q"].copy()
                result["old_baseline_PSD_raw"]=result["on_PSD_raw"].copy()
                parent=sd.parent_metrics(result,2,2)
                expected=result["on_Q"].reshape(2,2).var(1,ddof=1)
                self.assertTrue(np.array_equal(parent["on_variance_Q"],expected))
                self.assertLess(np.abs(parent["Q_variance_decomposition_residual"]).max(),1e-12)
                s=sd.summary(parent["on_variance_Q"])
                self.assertEqual(s["mean"],expected.mean())
                self.assertEqual(s["median"],np.median(expected))


if __name__=="__main__": unittest.main(verbosity=2)