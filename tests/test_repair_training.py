import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from n01.electronics_v2 import fit_electronics_initialization,make_n01_response_v2
from n01.losses import fit_loss_statistics
from n01.noise_validation import balanced_real_domains,baseline_statistics
from n01.train import exact_two_pass_backward
from n01.train_repair import BoundLossV2,state_digest,train_responses

torch.set_num_threads(2)


class RepairIntegrationChecks(unittest.TestCase):
    def test_failed_noise_gate_stops_before_any_response_construction(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            gate=root/'NOISE_GATE.json'
            gate.write_text(json.dumps(dict(passed=False,status='development_noise_gate_inconclusive_control',
                failed_checks=['acf','ols_joint','negative_control_interval_contains_half'])),encoding='utf8')
            with patch('n01.train_repair.RepairEngine') as engine:
                with self.assertRaisesRegex(RuntimeError,'Noise gate has not passed'):
                    train_responses(None,None,None,gate,root/'responses','cpu')
                engine.assert_not_called()
            self.assertFalse((root/'responses').exists())

    def test_two_pass_gradient_after_electronics_replacement(self):
        rng=torch.Generator().manual_seed(611)
        t=torch.arange(480,dtype=torch.float64);phase=(t-115).clamp_min(0)
        pulse=(1-(-phase/4).exp())*(-phase/50).exp()
        source=torch.linspace(.03,.3,64)[:,None]*pulse[None]
        real=.85*source+torch.randn(64,480,generator=rng)*.0006
        initialization=fit_electronics_initialization(real[:,:100])
        stats=dict(q_ref=5.,a_ref=.1,log_q_min=-2.,log_q_max=4.,electronics_rms=.0006)
        for variant in ('full','deterministic'):
            torch.manual_seed(31)
            model=make_n01_response_v2(stats,variant,electronics_initialization=initialization)
            model.electronics.requires_grad_(False)
            before=state_digest(model.electronics.state_dict())
            twin=copy.deepcopy(model)
            loss=BoundLossV2(fit_loss_statistics(real,source),seed=72,evaluation_seed=73)
            ctx=loss.create_projection_context(7);loss.bind(ctx)
            batches=[(source[:2].float(),model.sample_noise(2)),(source[31:33].float(),model.sample_noise(2))]
            direct=[model(x,noise=n) for x,n in batches]
            waveform=torch.cat([item['waveform'] for item in direct])
            parts=loss.components(dict(waveform=waveform),real[:24])
            objective=parts['wave']+2*parts['QPSD']+parts['noise']
            objective+=sum(.01*item['regular']+.1*item['trend'] for item in direct)/len(direct)
            objective.backward()
            with torch.no_grad():frozen=torch.cat([twin(x,noise=n)['waveform'] for x,n in batches])
            leaf=frozen.detach().requires_grad_(True)
            parts=loss.components(dict(waveform=leaf),real[:24])
            gradient=torch.autograd.grad(parts['wave']+2*parts['QPSD']+parts['noise'],leaf)[0]
            exact_two_pass_backward(twin,batches,gradient,verify_waveforms=frozen)
            numerator=denominator=0.
            for (name,p),(name2,q) in zip(model.named_parameters(),twin.named_parameters()):
                self.assertEqual(name,name2)
                self.assertEqual(p.grad is None,q.grad is None,name)
                if name.startswith('electronics.'):
                    self.assertIsNone(p.grad)
                if p.grad is not None:
                    numerator+=float((p.grad.double()-q.grad.double()).square().sum())
                    denominator+=float(p.grad.double().square().sum())
            self.assertLess((numerator/max(denominator,1e-30))**.5,1e-6)
            self.assertEqual(before,state_digest(twin.electronics.state_dict()))
            self.assertEqual(ctx.fingerprint,loss.bound_projection_context.fingerprint)

    def test_matched_real_control_uses_all_groups_without_row_reuse(self):
        train=np.repeat(np.arange(422),50)
        validation=np.repeat(np.arange(500,512),50)[:-1]
        for groups,n in [(train,2048),(validation,256)]:
            a,b=balanced_real_domains(groups,n,173)
            self.assertEqual(len(a),n);self.assertEqual(len(b),n)
            self.assertEqual(len(np.intersect1d(a,b)),0)
            self.assertEqual(set(groups[a]),set(groups))
            self.assertEqual(set(groups[b]),set(groups))
            for g in np.unique(groups):self.assertEqual(int((groups[a]==g).sum()),int((groups[b]==g).sum()))
        self.assertFalse(set(train)&set(validation))

    def test_statistics_detect_stationary_correlation_and_offset_separately(self):
        rng=np.random.default_rng(725)
        white=rng.normal(size=(8000,101))*.0006
        independent=white[:,:100]
        colored=(white[:,:100]+.4*white[:,1:])/np.sqrt(1.16)
        shifted=independent+rng.normal(size=(len(white),1))*.0004
        a,b,c=map(baseline_statistics,[independent,colored,shifted])
        self.assertGreater(b['acf_mean'][0]-a['acf_mean'][0],.30)
        self.assertGreater(c['mean_std_V']/a['mean_std_V'],5.)
        np.testing.assert_allclose(a['rms_quantiles_V'],c['rms_quantiles_V'],rtol=1e-12)


if __name__=='__main__':unittest.main()
