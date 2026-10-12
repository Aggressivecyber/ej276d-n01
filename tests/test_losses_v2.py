"""Artificial N01 v2 tests; no main-model training or validation data reading."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import audit_projection_blindspot as audit
from n01.losses import N01Loss, PopulationWeights, fit_loss_statistics, weighted_wasserstein_1d, window_charge
from n01.losses_v2 import (BaselineDistributionLoss, N01LossV2, ProjectionContext,
    baseline_ols_features, create_evaluation_context, create_projection_context,
    empirical_wasserstein_1d, fit_baseline_statistics)


class W1FastPathTests(unittest.TestCase):
    def test_uniform_equalN_value_and_position_gradients_match_CDF(self):
        x0=torch.tensor([[.2,-.4],[1.7,.8],[3.1,2.2]],dtype=torch.float64)
        y0=torch.tensor([[.5,-.1],[2.3,.7],[4.,3.4]],dtype=torch.float64)
        x,y=x0.clone().requires_grad_(),y0.clone().requires_grad_()
        fast=empirical_wasserstein_1d(x,y,torch.ones(3,dtype=torch.float64))
        fg=torch.autograd.grad(fast,(x,y))
        x,y=x0.clone().requires_grad_(),y0.clone().requires_grad_()
        exact=weighted_wasserstein_1d(x,y,torch.ones(3,dtype=torch.float64))
        eg=torch.autograd.grad(exact,(x,y))
        torch.testing.assert_close(fast,exact,rtol=0,atol=1e-12)
        for actual,expected in zip(fg,eg):
            torch.testing.assert_close(actual,expected,rtol=0,atol=1e-12)

    def test_learnable_constant_weights_nonuniform_and_unequalN_fall_back(self):
        x=torch.tensor([0.,2.],dtype=torch.float64)
        y=torch.tensor([.5,.6],dtype=torch.float64)
        w=torch.ones(2,dtype=torch.float64,requires_grad=True)
        with patch("n01.losses_v2._weighted_wasserstein_1d",wraps=weighted_wasserstein_1d) as fallback:
            value=empirical_wasserstein_1d(x,y,w)
            value.backward()
            self.assertTrue(fallback.called)
        self.assertIsNotNone(w.grad)
        self.assertGreater(float(w.grad.abs().max()),0)
        self.assertTrue(torch.isfinite(w.grad).all())
        for yy,ww in ((torch.tensor([.5],dtype=torch.float64),None),
                      (y,torch.tensor([1.,1.+1e-10],dtype=torch.float64))):
            with patch("n01.losses_v2._weighted_wasserstein_1d",wraps=weighted_wasserstein_1d) as fallback:
                value=empirical_wasserstein_1d(x,yy,ww)
                self.assertTrue(fallback.called)
                torch.testing.assert_close(value,weighted_wasserstein_1d(x,yy,ww),rtol=0,atol=0)


class LossV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.real=audit.synthetic_waveforms(96)
        cls.stats=fit_loss_statistics(cls.real,cls.real)

    def test_context_refresh_evaluation_independence_serialization_and_no_global_RNG(self):
        state=torch.random.get_rng_state().clone()
        a,b=create_projection_context(3,42),create_projection_context(3,42)
        c=create_projection_context(4,42)
        e=create_evaluation_context(43)
        self.assertEqual(a.fingerprint,b.fingerprint)
        self.assertNotEqual(a.fingerprint,c.fingerprint)
        self.assertNotEqual(a.fingerprint,e.fingerprint)
        self.assertEqual(a.fingerprint,ProjectionContext.from_state_dict(a.state_dict()).fingerprint)
        torch.testing.assert_close(state,torch.random.get_rng_state(),rtol=0,atol=0)
        a.banks["wave:voltage"].add_(0.)
        with self.assertRaisesRegex(RuntimeError,"mutated"):
            a.validate()
        with self.assertRaises(ValueError):
            N01LossV2(self.stats,seed=11,evaluation_seed=11)

    def test_same_context_two_pass_losses_and_waveform_gradients_are_exact(self):
        objective=N01LossV2(self.stats,seed=41,evaluation_seed=42)
        context=objective.create_projection_context(17)
        outputs=[]
        for _ in range(2):
            fake=(self.real*1.025+1e-5).clone().requires_grad_()
            parts=objective.components({"waveform":fake},self.real,projection_context=context)
            total=parts["wave"]+2*parts["QPSD"]+parts["noise"]
            gradient=torch.autograd.grad(total,fake)[0]
            outputs.append((parts,gradient))
        for key in outputs[0][0]:
            torch.testing.assert_close(outputs[0][0][key],outputs[1][0][key],rtol=0,atol=0)
        torch.testing.assert_close(outputs[0][1],outputs[1][1],rtol=0,atol=0)
        self.assertTrue(torch.isfinite(outputs[0][1]).all())
        self.assertGreater(float(outputs[0][1].norm()),0)
        self.assertEqual(objective.last_diagnostics["projection_context"]["update"],17)
        with self.assertRaisesRegex(ValueError,"explicit"):
            objective.components({"waveform":self.real},self.real)

    def test_old_joint_nullspace_is_detected_with_same_old_bank_and_coordinate_lag_terms(self):
        old=N01Loss(self.stats,seed=10001)
        delta,_=audit.nullspace_perturbation(self.stats,audit.frozen_bank(old))
        altered=self.real+delta
        with torch.no_grad():
            old_value=old.components({"waveform":altered},self.real)["wave"]
        self.assertLess(float(old_value),2e-5)
        objective=N01LossV2(self.stats,seed=101,evaluation_seed=102)
        context=objective.create_projection_context(0)
        bank={key:value.clone() for key,value in context.banks.items()}
        for name in audit.VIEWS:
            bank["wave:"+name]=getattr(old,name+"_directions").clone()
        fixed_old_context=ProjectionContext("training",101,0,bank)
        parts=objective.wave_components(altered,self.real,None,fixed_old_context)
        self.assertLess(float(parts["swd"]),1e-11)
        self.assertGreater(float(parts["coordinate"]),1e-4)
        self.assertGreater(float(parts["short_lag"]),1e-4)
        total=objective.components({"waveform":altered},self.real,projection_context=fixed_old_context)["wave"]
        self.assertGreater(float(total),1e-4)

    def test_population_weights_only_grad_and_global_sparse_support_semantics(self):
        objective=N01LossV2(self.stats,seed=17,evaluation_seed=18)
        fake=(self.real[:61]*1.03).clone().requires_grad_()
        population=PopulationWeights(window_charge(fake),torch.linspace(0,1,len(fake)))
        context=objective.create_projection_context(5)
        value=objective.population_objective(fake,self.real,population(),projection_context=context)
        value.backward()
        self.assertIsNone(fake.grad)
        self.assertTrue(torch.isfinite(population.theta.grad).all())
        self.assertGreater(float(population.theta.grad.norm()),0)
        self.assertEqual(objective.last_diagnostics["fake_global_rows"],61)
        self.assertTrue(objective.last_diagnostics["global_term_includes_all_rows"])
        self.assertTrue(objective.last_diagnostics["conditional_rebatch_needed"])


class BaselineV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.real=audit.synthetic_waveforms(96)[:,:100]
        cls.stats=fit_baseline_statistics(cls.real)

    def test_baseline_only_statistics_OLS_and_context_reuse(self):
        full=torch.cat((self.real,torch.full((96,380),float("nan"),dtype=torch.float64)),1)
        stats=fit_baseline_statistics(full)
        self.assertEqual(stats["fit_rows"],96)
        self.assertFalse(stats["metadata"]["full_pulse_features_used"])
        objective=BaselineDistributionLoss(stats,seed=21,evaluation_seed=22)
        context=objective.create_projection_context(9)
        values=objective.components(self.real,self.real,projection_context=context)
        self.assertEqual(set(values),set(objective.weights))
        self.assertEqual(float(objective.total(values)),0.)
        t=2*torch.arange(100,dtype=torch.float64)/479-1
        line=.001+.0002*t
        ols=baseline_ols_features(line[None])
        self.assertAlmostEqual(float(ols[0,1]),.0002,places=12)
        self.assertAlmostEqual(float(ols[0,2]),.001,places=12)

    def test_artificial_noise_gain_has_finite_gradients_and_optimizes(self):
        objective=BaselineDistributionLoss(self.stats,seed=31,evaluation_seed=32)
        eval_context=objective.create_evaluation_context()
        gain=torch.nn.Parameter(torch.tensor(1.4,dtype=torch.float64))
        optimizer=torch.optim.Adam([gain],lr=.04)
        with torch.no_grad():
            initial=float(objective.total(objective.components(gain*self.real,self.real,projection_context=eval_context)))
        for update in range(12):
            context=objective.create_projection_context(update)
            parts=objective.components(gain*self.real,self.real,projection_context=context)
            total=objective.total(parts)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            self.assertTrue(torch.isfinite(gain.grad))
            optimizer.step()
        with torch.no_grad():
            final=float(objective.total(objective.components(gain*self.real,self.real,projection_context=eval_context)))
        self.assertLess(final,.8*initial)


if __name__=="__main__":
    unittest.main()
