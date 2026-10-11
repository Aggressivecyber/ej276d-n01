import copy
import unittest

import numpy as np
import torch

from n01.data import FullCoverageSampler, charge_np, fit_model_statistics
from n01.losses import N01Loss, fit_loss_statistics
from n01.model import N01Response
from n01.train import exact_two_pass_backward

torch.set_num_threads(2)


def synthetic(count=64):
    t = torch.arange(480).double()
    phase = (t-116).clamp_min(0)
    pulse = (1-torch.exp(-phase/4))*torch.exp(-phase/50)
    rng = torch.Generator().manual_seed(719)
    amp = torch.linspace(.03,.3,count)
    source = amp[:,None]*pulse[None]
    real = .8*source+torch.randn(count,480,generator=rng).double()*.0006
    return source.float(),real


class TrainingChecks(unittest.TestCase):
    def test_exact_joint_gradient_by_reforward(self):
        source,real = synthetic()
        torch.manual_seed(42)
        model = N01Response(dict(q_ref=5.,a_ref=.1,log_q_min=-2.,log_q_max=4.,electronics_rms=.0006))
        model.set_stage("joint_stochastic")
        twin = copy.deepcopy(model)
        loss = N01Loss(fit_loss_statistics(real,source),seed=41)
        batches = [(source[:3],model.sample_noise(3)),(source[24:27],model.sample_noise(3))]
        direct = [model(x,noise=n) for x,n in batches]
        wave = torch.cat([o["waveform"] for o in direct])
        components = loss.components(dict(waveform=wave),real[:24])
        scalar = components["wave"]+2*components["QPSD"]+components["noise"]
        scalar += sum(.01*o["regular"]+.1*o["trend"] for o in direct)/len(direct)
        scalar.backward()
        with torch.no_grad():
            frozen = torch.cat([twin(x,noise=n)["waveform"] for x,n in batches])
        leaf = frozen.detach().requires_grad_(True)
        components = loss.components(dict(waveform=leaf),real[:24])
        scalar = components["wave"]+2*components["QPSD"]+components["noise"]
        derivative = torch.autograd.grad(scalar,leaf)[0]
        exact_two_pass_backward(twin,batches,derivative,verify_waveforms=frozen)
        numerator = denominator = 0.
        for (name,p),(name2,q) in zip(model.named_parameters(),twin.named_parameters()):
            self.assertEqual(name,name2)
            self.assertEqual(p.grad is None,q.grad is None,name)
            if p.grad is not None:
                self.assertTrue(torch.isfinite(q.grad).all(),name)
                numerator += float((p.grad.double()-q.grad.double()).square().sum())
                denominator += float(p.grad.double().square().sum())
        relative_error = (numerator/max(denominator,1e-30))**.5
        self.assertLess(relative_error,1e-6)

    def test_no_dropped_epoch_tails_and_resume(self):
        sampler = FullCoverageSampler(7,3)
        rows = sampler.draw(20)
        self.assertEqual(set(rows[:7]),set(range(7)))
        self.assertEqual(set(rows[7:14]),set(range(7)))
        restored = FullCoverageSampler(7,999)
        restored.load_state_dict(sampler.state_dict())
        np.testing.assert_array_equal(sampler.draw(11),restored.draw(11))
        self.assertEqual(int(sampler.visits.sum()),31)
        self.assertLessEqual(int(np.ptp(sampler.visits)),1)

    def test_electronics_slope_coordinate(self):
        source,_ = synthetic(200)
        t = 2*np.arange(480)/479-1
        rng = np.random.default_rng(64)
        b0 = rng.normal(.001,.0002,200)
        b1 = rng.normal(0,.0004,200)
        real = source.numpy()+b0[:,None]+b1[:,None]*t
        stats = fit_model_statistics(dict(source=source.numpy(),real=real))
        self.assertAlmostEqual(stats["electronics_offset_mean"],float(b0.mean()),places=12)
        self.assertAlmostEqual(stats["electronics_slope_std"],float(b1.std(ddof=1)),places=12)


if __name__ == "__main__":
    unittest.main()
