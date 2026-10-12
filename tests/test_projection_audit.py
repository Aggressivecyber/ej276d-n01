"""Meaningful artificial regression checks for the isolated projection audit."""
from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"src"))
import audit_projection_blindspot as audit
from n01.losses import N01Loss, fit_loss_statistics
from n01.model import CausalFilter


class ProjectionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original=audit.synthetic_waveforms()
        cls.stats=fit_loss_statistics(cls.original,cls.original)
        cls.objective=N01Loss(cls.stats,seed=10001)
        cls.bank=audit.frozen_bank(cls.objective)
        cls.delta,cls.geometry=audit.nullspace_perturbation(cls.stats,cls.bank)
        cls.altered=cls.original+cls.delta

    def test_joint_nullspace_preserves_observables_and_evades_current_wave_term(self):
        self.assertGreater(self.geometry["nullity"],0)
        self.assertEqual(tuple(self.geometry["matrix"].shape),(194,300))
        self.assertLess(float((self.geometry["matrix"]@self.delta[:,180:].T).abs().max()),1e-11)
        torch.testing.assert_close(self.delta.square().mean(1).sqrt(),torch.full((96,),.001,dtype=torch.float64))
        invariant=audit.preservation(self.original,self.altered)
        self.assertEqual(invariant["baseline_max_abs_error_V"],0.)
        self.assertEqual(invariant["peak_max_abs_error_V"],0.)
        self.assertTrue(invariant["peak_argmax_unchanged"])
        self.assertLess(invariant["charge_max_abs_error_Vns"],1e-12)
        self.assertLess(invariant["PSD_max_abs_error"],1e-12)
        reference=audit.wave_distance(self.original,self.altered,self.stats,self.bank)
        self.assertLess(reference["wave"],1e-11)
        with torch.no_grad():
            actual=self.objective.components({"waveform":self.altered},self.original)
        self.assertLess(float(actual["wave"]),2e-5)
        self.assertEqual(float(actual["noise"]),0.)

    def test_independent_frozen_refresh_coordinate_and_lag_views_detect_same_delta(self):
        independent=audit.frozen_bank(N01Loss(self.stats,seed=audit.EVALUATION_SEED))
        distance=audit.wave_distance(self.original,self.altered,self.stats,independent)
        self.assertGreater(distance["wave"],1e-4)
        for seed in audit.REFRESH_SEEDS:
            fresh=audit.frozen_bank(N01Loss(self.stats,seed=seed))
            self.assertGreater(audit.wave_distance(self.original,self.altered,self.stats,fresh)["wave"],1e-4)
        detected=audit.coordinate_and_short_lag_detection(self.original,self.altered,self.stats)
        self.assertGreater(detected["mean_coordinate_W1"],1e-4)
        self.assertTrue(all(v["time_resolved_pair_projected_W1"]>1e-4 for v in detected["short_lag_joint"].values()))

    def test_exact_fir_supremum_differs_from_linearized_bound_and_matches_current_kernel(self):
        bounds=audit.fir_bounds()
        self.assertAlmostEqual(bounds["first_order_linearized_abs_rho1_bound"],1/129,places=14)
        self.assertAlmostEqual(bounds["exact_stationary_white_FIR_abs_rho1_supremum"],192/16705,places=14)
        kernel=CausalFilter(65,"rms").double()
        with torch.no_grad():
            kernel.raw.fill_(100.)
            kernel.raw[0]=-100.
        rho=audit.kernel_rho1(kernel.taps())
        self.assertAlmostEqual(rho,bounds["exact_stationary_white_FIR_abs_rho1_supremum"],places=14)
        self.assertGreater(rho,bounds["first_order_linearized_abs_rho1_bound"])
        self.assertLess(rho,bounds["conservative_triangle_abs_rho1_bound"])
        rng=torch.Generator().manual_seed(443)
        raw=6*torch.randn(1000,65,generator=rng,dtype=torch.float64)
        h=.5*torch.tanh(raw)/65;h[:,0]+=1
        r=(h[:,:-1]*h[:,1:]).sum(1)/h.square().sum(1)
        self.assertLess(float(r.abs().max()),bounds["exact_stationary_white_FIR_abs_rho1_supremum"])


if __name__=="__main__":
    unittest.main()
