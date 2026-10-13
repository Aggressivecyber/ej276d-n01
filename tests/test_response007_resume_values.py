"""Compare optimizer scalar values across CPU/CUDA storage after restore.

The previous GPU test already matched model, loss, samplers and streams; its
last assertion used torch.equal across devices for Adam's scalar step counter.
This test changes only comparison placement, with exact value equality.
"""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from test_response007 import fabricated
from n01.response007 import Response007Engine,common_initial_state,state_digest
from n01.train_response007_statusfix01 import restore_first_update,training_status

class ResumeValueTests(unittest.TestCase):
    def test_exact_resume_values(self):
        data,stats,noise,cfg=fabricated();device=os.environ.get('RESPONSE007_TEST_DEVICE','cpu')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        common=common_initial_state(stats['model'],cfg['seeds']['initialization'])
        a=Response007Engine(data,stats,cfg,noise,common,arm='deterministic',device=device)
        a.set_stage('mean_response');a.step(verify_replay=True)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'first.pt';a.checkpoint(p)
            b=Response007Engine(data,stats,cfg,noise,common,arm='deterministic',device=device)
            restore_first_update(b,p,cfg)
            x=a.step(verify_replay=True);y=b.step(verify_replay=True)
            self.assertEqual(state_digest(a.model.state_dict()),state_digest(b.model.state_dict()))
            self.assertEqual(state_digest(a.discriminators.state_dict()),state_digest(b.discriminators.state_dict()))
            self.assertEqual(x['hypotheses'],y['hypotheses']);self.assertEqual(x['weighted_total'],y['weighted_total'])
            for k in a.streams.state_dict():self.assertTrue(torch.equal(a.streams.state_dict()[k],b.streams.state_dict()[k]))
            np.testing.assert_array_equal(a.source_sampler.visits,b.source_sampler.visits)
            np.testing.assert_array_equal(a.real_sampler.visits,b.real_sampler.visits)
            sa,sb=a.g_optimizer.state_dict(),b.g_optimizer.state_dict();self.assertEqual(sa['param_groups'],sb['param_groups'])
            for key,state in sa['state'].items():
                for field,value in state.items():self.assertTrue(torch.equal(value.detach().cpu(),sb['state'][key][field].detach().cpu()))
            result=training_status(y,2,0,1.)
            self.assertEqual(result['noise_updates'],0);self.assertEqual(result['new_execution_G_updates'],1)

if __name__=='__main__':unittest.main()
