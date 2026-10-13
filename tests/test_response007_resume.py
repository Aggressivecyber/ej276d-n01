import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_response007 import fabricated
from n01.response007 import Response007Engine,common_initial_state,state_digest
from n01.train_response007_statusfix01 import training_status,restore_first_update


class Response007ResumeTests(unittest.TestCase):
    def test_status_with_existing_noise_field_and_cumulative_count(self):
        metrics=dict(noise_updates=0,G_updates=2,arm='deterministic',weighted_total=1.)
        result=training_status(metrics,2,0,3.)
        self.assertEqual(result['noise_updates'],0)
        self.assertEqual(result['total_G_updates'],2)
        self.assertEqual(result['new_execution_G_updates'],1)
        self.assertEqual(result['previous_completed_G_updates'],1)
        self.assertNotIn('total_G_updates',metrics)

    def test_exact_first_step_checkpoint_continuation(self):
        data,stats,noise,cfg=fabricated();device=os.environ.get('RESPONSE007_TEST_DEVICE','cpu')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        common=common_initial_state(stats['model'],cfg['seeds']['initialization'])
        a=Response007Engine(data,stats,cfg,noise,common,arm='deterministic',device=device)
        a.set_stage('mean_response');a.step(verify_replay=True)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'step1.pt';a.checkpoint(checkpoint)
            b=Response007Engine(data,stats,cfg,noise,common,arm='deterministic',device=device)
            restore_first_update(b,checkpoint,cfg)
            ma=a.step(verify_replay=True);mb=b.step(verify_replay=True)
            self.assertEqual(a.update,b.update);self.assertEqual(a.update,2)
            self.assertEqual(state_digest(a.model.state_dict()),state_digest(b.model.state_dict()))
            self.assertEqual(state_digest(a.discriminators.state_dict()),state_digest(b.discriminators.state_dict()))
            self.assertEqual(ma['hypotheses'],mb['hypotheses'])
            self.assertEqual(ma['weighted_total'],mb['weighted_total'])
            for key in a.streams.state_dict():self.assertTrue(torch.equal(a.streams.state_dict()[key],b.streams.state_dict()[key]))
            np.testing.assert_array_equal(a.source_sampler.visits,b.source_sampler.visits)
            np.testing.assert_array_equal(a.real_sampler.visits,b.real_sampler.visits)
            sa,sb=a.g_optimizer.state_dict(),b.g_optimizer.state_dict()
            self.assertEqual(sa['param_groups'],sb['param_groups'])
            for key,state in sa['state'].items():
                for field,value in state.items():
                    self.assertTrue(torch.equal(value,sb['state'][key][field]))


if __name__=='__main__':unittest.main()
