"""Artificial conditional-null semantics; no scientific classifier fitting."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import calibrate_real_real_null as null
from n01.data import canonical_real
from n01.evaluate import _weighted_auc


class RealRealNullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_pseudo_roles_use_only_fixed_group_ID_count_and_all_group_rows(self):
        groups=np.repeat(np.arange(35),50)
        fit,test,chosen=null.select_pseudo_roles(groups,12,44,2026101260)
        self.assertEqual(len(chosen),12);self.assertEqual(len(test),600)
        self.assertFalse(np.intersect1d(groups[fit],groups[test]).size)
        self.assertTrue(np.array_equal(test,np.flatnonzero(np.isin(groups,chosen))))
        again=null.select_pseudo_roles(groups,12,44,2026101260)
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip((fit,test,chosen),again)))
        with self.assertRaises(ValueError):null.select_pseudo_roles(np.repeat(np.arange(11),50))
        with self.assertRaises(ValueError):null.check_capacity(np.r_[np.repeat(0,5),np.repeat(1,50)],20)

    def test_role_sampling_equal_group_quotas_and_independent_fixed_seed_ranges(self):
        groups=np.repeat(np.arange(422),50)
        fit,test,chosen=null.select_pseudo_roles(groups)
        cfg=null.FIXED|dict(split_seed_base=2026101261,classifier_seed_base=2026102261,
            permutation_seed_base=2026103261,bootstrap_seed_base=2026104261)
        rows,seeds=null.draw_roles(groups,fit,test,cfg,3)
        self.assertEqual([len(rows[k]) for k in ('fit0','fit1','test0','test1')],[2048,2048,256,256])
        for role in ('fit','test'):
            a,b=rows[role+'0'],rows[role+'1']
            self.assertFalse(np.intersect1d(a,b).size)
            for g in np.unique(groups[a]):self.assertEqual(np.sum(groups[a]==g),np.sum(groups[b]==g))
        self.assertFalse(np.intersect1d(groups[rows['fit0']],groups[rows['test0']]).size)
        self.assertEqual(seeds['test_label_seed']-seeds['fit_label_seed'],128)
        self.assertEqual(seeds['conditional_permutation_seed'],2026103264)

    def test_rank_AUC_matches_old_tie_aware_AUC(self):
        rng=np.random.default_rng(9)
        for _ in range(30):
            labels=np.r_[np.zeros(20),np.ones(20)]
            scores=np.round(rng.normal(size=40)*2)/2
            self.assertAlmostEqual(null.rank_auc(labels,null.average_ranks(scores)),_weighted_auc(labels,scores),places=14)
        self.assertEqual(null.rank_auc(np.array([0,1]),null.average_ranks(np.ones(2))),.5)

    def test_matrix_bootstrap_all1000_draws_equals_old_joint_weighted_statistic(self):
        groups=np.repeat(np.arange(6),20)
        labels=np.tile(np.r_[np.zeros(10),np.ones(10)],6)
        scores=np.round(np.random.default_rng(12).normal(size=120)*3)/3
        seed=81;result=null.joint_group_bootstrap(labels,scores,groups,seed,1000)
        rng=np.random.default_rng(seed);old=[]
        for _ in range(1000):
            multiplicity=np.bincount(rng.integers(0,6,6),minlength=6)[groups]
            old.append(_weighted_auc(labels,scores,multiplicity))
        np.testing.assert_allclose(result['bootstrap_AUC'],old,rtol=0,atol=1e-15)
        np.testing.assert_allclose(result['percentile_95_interval'],np.quantile(old,[.025,.975]),rtol=0,atol=1e-15)
        leave=[_weighted_auc(labels[groups!=g],scores[groups!=g]) for g in range(6)]
        np.testing.assert_allclose(result['leave_one_group_AUC'],leave,rtol=0,atol=1e-15)
        self.assertEqual(result['CI_contains_half'],bool(result['percentile_95_interval'][0]<=.5<=result['percentile_95_interval'][1]))

    def test_permutation_holds_scores_parent_set_and_every_group_label_quota(self):
        groups=np.repeat(np.arange(3),8)
        labels=np.tile(np.r_[np.zeros(4),np.ones(4)],3).astype(np.int8)
        scores=np.arange(24,dtype=np.float64)
        real_rank_auc=null.rank_auc
        with patch.object(null,'rank_auc',wraps=real_rank_auc) as rank:
            result=null.conditional_test_permutations(labels,scores,groups,51,255)
        self.assertEqual(rank.call_count,256)
        for call in rank.call_args_list:
            y,r=call.args
            self.assertEqual(len(y),24)
            for group in range(3):self.assertEqual(np.sum(y[groups==group]),4)
            np.testing.assert_array_equal(r,null.average_ranks(scores))
        values=result['permutation_AUC']
        tail=int(np.sum(abs(values-.5)>=abs(result['observed_AUC']-.5)))
        self.assertEqual(result['tail_count'],tail)
        self.assertEqual(result['plus_one_p_value'],(1+tail)/256)
        self.assertFalse(result['classifier_refitted'])

    def test_ties_are_in_tail_and_perfect_separation_has_small_p(self):
        groups=np.repeat(np.arange(3),8)
        labels=np.tile(np.r_[np.zeros(4),np.ones(4)],3).astype(np.int8)
        tied=null.conditional_test_permutations(labels,np.ones(24),groups,30)
        self.assertEqual(tied['tail_count'],255);self.assertEqual(tied['plus_one_p_value'],1.)
        perfect=null.conditional_test_permutations(labels,labels.astype(float),groups,31)
        self.assertEqual(perfect['observed_AUC'],1.)
        self.assertLessEqual(perfect['plus_one_p_value'],.02)
        self.assertGreaterEqual(perfect['plus_one_p_value'],1/256)

    def test_Wilson_known_boundary_and_nonzero_counts(self):
        lo,hi=null.wilson_interval(0,128)
        self.assertEqual(lo,0.)
        self.assertAlmostEqual(hi,1.959963984540054**2/(128+1.959963984540054**2),places=15)
        lo,hi=null.wilson_interval(64,128)
        self.assertAlmostEqual(lo+hi,1.,places=14)
        self.assertLess(lo,.5);self.assertGreater(hi,.5)
        with self.assertRaises(ValueError):null.wilson_interval(1,0)

    def test_reader_opens_only_target_and_uses_canonical_first100(self):
        target=np.linspace(-.1,.1,480,dtype=np.float32)[None].repeat(4,axis=0)
        reads=[]
        class Archive:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __getitem__(self,key):
                reads.append(key)
                if key!='target':raise AssertionError('Forbidden scientific key')
                return target
        with patch.object(null,'read_identity_metadata',return_value=(np.arange(4),np.array([1,1,2,2]),{})),patch.object(null.np,'load',return_value=Archive()):
            x,ids,groups,meta=null.read_train_baselines_only('artificial','identities')
        self.assertEqual(reads,['target'])
        np.testing.assert_array_equal(x,canonical_real(target)[:,:100])
        self.assertEqual(x.dtype,np.float64)

    def test_fabricated_classifier_scaler_fit_only_replay_and_partial_count(self):
        rng=np.random.default_rng(18)
        train0=rng.normal(size=(64,100))*.001;train1=rng.normal(size=(64,100))*.001
        test0=rng.normal(size=(16,100))*.001;test1=rng.normal(size=(16,100))*.001
        progress={}
        a,state,loss=null.fit_raw_classifier(train0,train1,test0,test1,3,torch.device('cpu'),updates=3,progress=progress)
        b,other,loss2=null.fit_raw_classifier(train0,train1,test0+10,test1-10,3,torch.device('cpu'),updates=3)
        np.testing.assert_array_equal(state['center'],other['center']);np.testing.assert_array_equal(state['scale'],other['scale'])
        self.assertTrue(all(torch.equal(state['model'][k],other['model'][k]) for k in state['model']))
        np.testing.assert_array_equal(loss,loss2)
        self.assertFalse(np.array_equal(a,b))
        self.assertEqual(progress['updates_executed'],3);self.assertFalse(progress['optimizer_step_in_progress'])
        self.assertEqual(sum(v.numel() for v in state['model'].values()),4897)


if __name__=='__main__':unittest.main()
