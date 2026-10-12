"""Artificial-data checks for the independent frozen006 verifier."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('verify006',ROOT/'scripts/verify_integration006.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


class IndependentVerificationTests(unittest.TestCase):
    def test_readout_and_string_identity(self):
        v.exact(np.array(['unchanged']),np.array(['unchanged']),'string identity')
        x=np.zeros((2,480));x[:,120:150]=.1
        r=v.readouts_np(x,1e-3)
        np.testing.assert_allclose(r['Q'],3)
        np.testing.assert_allclose(r['PSD'],0)
        np.testing.assert_allclose(r['half_rise_ns'],119.5)

    def test_parent_bootstrap_against_explicit_event_resampling(self):
        rng=np.random.default_rng(87);n=256;rows=4*n
        cfg={'evaluation':{'arms':['deterministic','full'],'bootstrap_draws':7,
            'minimum_rows_per_fixed_bin':128,'minimum_parents_per_fixed_bin':32},
            'seeds':{'bootstrap_TRAIN':9,'bootstrap_DEV':10}}
        obs={'TRAIN_REAL_Q':rng.normal(3,1,500),'TRAIN_REAL_PSD':rng.normal(.3,.1,500),
             'DEV_REAL_Q':rng.normal(3,8,100),'DEV_REAL_PSD':rng.normal(.3,4,100)}
        for split in ('TRAIN','DEV'):
            for arm in cfg['evaluation']['arms']:
                pre=split+'_'+arm+'_'
                obs[pre+'Q']=rng.normal(3,.3,(5,rows))
                obs[pre+'PSD']=rng.normal(.3,.05,(5,rows))
                obs[pre+'fixed_bins']=np.arange(rows)%8
            found=v.independent_bootstrap(obs,split,cfg)
            drawrng=np.random.default_rng(cfg['seeds']['bootstrap_'+split])
            for b in range(7):
                parents=drawrng.integers(0,n,size=n)
                indices=(parents[:,None]*4+np.arange(4)).ravel()
                for arm in cfg['evaluation']['arms']:
                    pre=split+'_'+arm+'_';q=obs[pre+'Q'][2:,indices];p=obs[pre+'PSD'][2:,indices]
                    fixed=obs[pre+'fixed_bins'][indices]
                    expected_q=np.ptp(q.mean(1))/obs['TRAIN_REAL_Q'].std(ddof=0)
                    expected_p=np.ptp(p.mean(1))/obs['TRAIN_REAL_PSD'].std(ddof=0)
                    sd=np.array([p[:,fixed==k].std(1,ddof=0) for k in range(8)])
                    ratios=sd.max(1)/sd.min(1)
                    got=found['samples'][arm]
                    np.testing.assert_allclose(got['Q_mean_span'][b],expected_q,atol=1e-12)
                    np.testing.assert_allclose(got['PSD_mean_span'][b],expected_p,atol=1e-12)
                    np.testing.assert_allclose(got['fixed_bin_PSD_sd_ratio'][b],ratios,atol=1e-12)
                    np.testing.assert_allclose(got['max_fixed_bin_PSD_sd_ratio'][b],ratios.max(),atol=1e-12)

    def test_streamed_curves_and_within_parent_variance(self):
        rng=np.random.default_rng(13);wave=rng.normal(size=(5,256,480));bins=np.arange(256)%8
        c=v.CurveCheck();c.add(wave[:,:128],bins[:128]);c.add(wave[:,128:],bins[128:])
        saved={'mean':wave.mean(1),'variance':wave.var(1,ddof=0),
            'mean_within_parent_variance':wave.reshape(5,64,4,480).var(2,ddof=1).mean(1),
            'fixed_bin_mean':np.stack([wave[:,bins==b].mean(1) for b in range(8)]),
            'fixed_bin_variance':np.stack([wave[:,bins==b].var(1,ddof=0) for b in range(8)])}
        c.check(saved,'')

    def test_extended_intervals_preserve_infinity_without_omitting_nan(self):
        x=np.array([[1.,1.,1.],[2.,np.inf,2.],[3.,np.inf,np.nan],[4.,np.inf,4.]])
        actual=v.extended_intervals(x,[.25,.75])
        np.testing.assert_allclose(actual[0],[1.75,3.25])
        self.assertTrue(np.isposinf(actual[1]).all())
        self.assertTrue(np.isnan(actual[2]).all())

    def test_nonzero_constant_width_remains_undefined(self):
        n=256;cfg={'evaluation':{'arms':['full'],'bootstrap_draws':5,
            'minimum_rows_per_fixed_bin':128,'minimum_parents_per_fixed_bin':32},'seeds':{'bootstrap_TRAIN':9}}
        obs={'TRAIN_REAL_Q':np.array([1.,2.,3.]),'TRAIN_REAL_PSD':np.array([.1,.2,.3]),
            'TRAIN_full_Q':np.ones((5,n*4)),'TRAIN_full_PSD':np.full((5,n*4),.3),
            'TRAIN_full_fixed_bins':np.arange(n*4)%8}
        found=v.independent_bootstrap(obs,'TRAIN',cfg)
        self.assertTrue(np.isnan(found['samples']['full']['fixed_bin_PSD_sd_ratio']).all())
        # Near-constant values must retain their genuine tiny spread.
        tiny=np.arange(n*4)%7*1e-12
        obs['TRAIN_full_PSD'][2:]+=(np.array([1.,2.,3.])[:,None]*tiny[None,:])
        found=v.independent_bootstrap(obs,'TRAIN',cfg)
        ratios=found['samples']['full']['fixed_bin_PSD_sd_ratio']
        self.assertTrue(np.isfinite(ratios).all())
        np.testing.assert_allclose(ratios,3.,rtol=3e-5)

    def test_dev_cannot_change_screening_and_undefined_cannot_pass(self):
        cfg={'evaluation':{'arms':['full']},'decision':{
            'signal_baseline_absolute_max_V':1e-4,'noise_tail_variance_ratio_max':1.25,
            'Q_mean_range_over_REAL_sd_max':.05,'PSD_mean_range_over_REAL_sd_max':.05,'fixed_bin_PSD_sd_ratio_max':1.1}}
        obs={'TRAIN_full_Q':np.ones((5,4)),'TRAIN_full_PSD_guarded':np.zeros((5,4),dtype=bool)}
        names=['full/Q_mean_span_over_REAL_sd','full/PSD_mean_span_over_REAL_sd','full/fixed_bin_0_PSD_sd_ratio']
        train={'metric_names':names,'intervals':np.array([[.001,.01],[.002,.02],[1.01,1.05]]),
               'support':{'full':{'supported':[True]*8}}}
        verified={'TRAIN':train}
        theory={h+'_tail400480_variance_mean':1. for h in v.HYPOTHESES}
        leakage={'TRAIN':{'full':{'absmax':0.}},'DEV':{'full':{'absmax':1.}}}
        self.assertEqual(v.independent_decision(cfg,obs,verified,theory,leakage)['outcome'],'sensitivity_within_family')
        train['intervals'][2]=np.nan
        self.assertEqual(v.independent_decision(cfg,obs,verified,theory,leakage)['outcome'],'MC_inconclusive')
        train['intervals'][2]=np.inf
        self.assertEqual(v.independent_decision(cfg,obs,verified,theory,leakage)['outcome'],'sensitivity_detected')
        obs['TRAIN_full_Q'][2,0]=0
        self.assertEqual(v.independent_decision(cfg,obs,verified,theory,leakage)['outcome'],'interface_fail')


if __name__=='__main__':unittest.main()
