import copy
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from n01.data import fit_model_statistics
from n01.electronics_v3 import ObservedBaselineNoiseV3,fit_baseline_initialization_v3
from n01.losses import fit_loss_statistics
from n01.response007 import (ARMS,HYPOTHESES,PairedStreams,Response007Engine,candidates,
    common_initial_state,fixed_noises,make_extension,make_signal,replay_backward,saturation_penalty,state_digest,loss_tensor_statistics)

torch.set_num_threads(2)


def fabricated():
    rng=np.random.default_rng(741)
    t=np.arange(480);phase=np.maximum(t-115,0)
    pulse=(1-np.exp(-phase/4))*np.exp(-phase/50)
    source=np.linspace(.025,.25,64)[:,None]*pulse[None,:]
    real=.85*source+rng.normal(0,.0006,source.shape)
    data=dict(source=source.astype('float32'),real=real,source_ids=np.array([f'p{i}' for i in range(64)]),
        real_ids=np.arange(64),real_groups=np.repeat(np.arange(8),8),h=np.linspace(0,1,64))
    stats=dict(model=fit_model_statistics(data),loss=fit_loss_statistics(torch.from_numpy(real),torch.from_numpy(source)))
    from n01.evaluate_integration006 import clean
    stats=clean(stats)
    init=fit_baseline_initialization_v3(torch.from_numpy(rng.normal(0,.0006,(160,100))))
    noise=ObservedBaselineNoiseV3(init)
    checkpoint=dict(initialization=init,model=noise.state_dict())
    cfg=json.loads((Path(__file__).resolve().parents[1]/'response007_config.json').read_text(encoding='utf8'))
    cfg.update(distribution_microbatches=2,source_parents_per_microbatch=2,real_rows_per_microbatch=4)
    return data,stats,checkpoint,cfg


class Response007Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data,cls.stats,cls.noise,cls.cfg=fabricated()

    def test_explicit_common_initialization_and_mean_equality(self):
        common=common_initial_state(self.stats['model'],31)
        models=[make_signal(self.stats['model'],arm,common,75,'cpu') for arm in ARMS]
        for name,value in models[0].state_dict().items():self.assertTrue(torch.equal(value,models[1].state_dict()[name]),name)
        response,_,_=PairedStreams(7,8,9).draw(4,'cpu')
        source=torch.from_numpy(self.data['source'][:4])
        a,b=[m.signal_forward(source,response) for m in models]
        self.assertTrue(torch.equal(a['signal_pre'],b['signal_pre']))
        self.assertFalse(any(name.startswith('electronics.') for name,_ in models[1].named_parameters()))
        self.assertEqual(float((a['pre_adc']-a['signal_pre']).abs().max().detach()),0.)

    def test_json_statistics_roundtrip_and_paired_mean_update(self):
        before=json.dumps(self.stats,sort_keys=True)
        tensors=loss_tensor_statistics(self.stats['loss'])
        for name,value in self.stats['loss']['views'].items():
            self.assertEqual(tensors['views'][name]['center'].tolist(),value['center'])
            self.assertEqual(tensors['views'][name]['scale'].tolist(),value['scale'])
        self.assertEqual(json.dumps(self.stats,sort_keys=True),before)
        common=common_initial_state(self.stats['model'],self.cfg['seeds']['initialization'])
        engines=[Response007Engine(self.data,self.stats,self.cfg,self.noise,common,arm=arm,device='cpu') for arm in ARMS]
        for engine in engines:engine.set_stage('mean_response');engine.step(verify_replay=True)
        det,full=[e.model.state_dict() for e in engines]
        for name,value in det.items():self.assertTrue(torch.equal(value,full[name]),name)
        self.assertEqual(engines[0].last_metrics['hypotheses'],engines[1].last_metrics['hypotheses'])

    def test_binding_rejects_omitted_execution_sources_before_data_decode(self):
        from n01.train_response007 import bind
        from n01.data import sha256
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=root/'config.json';manifest=root/'manifest.json'
            config.write_text(json.dumps(self.cfg),encoding='utf8')
            manifest.write_text(json.dumps(dict(run_id=self.cfg['run_id'],config_sha256=sha256(config),
                before_new_scientific_forward=True,before_new_scientific_array_decode=True,files={})),encoding='utf8')
            with patch('n01.train_response007.ROOT',root),patch('n01.train_response007.load_training') as load:
                with self.assertRaisesRegex(ValueError,'omits required'):
                    bind(config,manifest,root/'unopened_container',root/'unopened_identities')
                load.assert_not_called()

    def test_paired_streams_restore_and_prefix_same_for_all_hypotheses(self):
        a,b=PairedStreams(1,2,3),PairedStreams(1,2,3)
        state=a.state_dict();ra,na,ea=a.draw(5,'cpu');rb,nb,eb=b.draw(5,'cpu')
        for k in ra:self.assertTrue(torch.equal(ra[k],rb[k]))
        for k in na:self.assertTrue(torch.equal(na[k],nb[k]))
        self.assertTrue(torch.equal(ea,eb));a.load_state_dict(state)
        _,_,ec=a.draw(5,'cpu');self.assertTrue(torch.equal(ea,ec))
        extension=make_extension(self.noise,'cpu');before=state_digest(extension.model005.state_dict())
        noises=fixed_noises(extension,na,ea)
        for h in HYPOTHESES:self.assertTrue(torch.equal(noises[h][:,:100],noises['fast'][:,:100]))
        self.assertGreater(float((noises['fast'][:,100:]-noises['slow'][:,100:]).abs().max()),0.)
        self.assertTrue(all(not x.requires_grad for x in noises.values()))
        self.assertEqual(before,state_digest(extension.model005.state_dict()))

    def test_three_hypothesis_two_pass_matches_direct_parameter_gradient(self):
        common=common_initial_state(self.stats['model'],31)
        for arm in ARMS:
            direct=make_signal(self.stats['model'],arm,common,31,'cpu');direct.set_stage('joint_stochastic')
            replay=copy.deepcopy(direct)
            streams=PairedStreams(51,52,53);extension=make_extension(self.noise,'cpu')
            batches=[]
            for rows in (slice(3,5),slice(48,50)):
                source=torch.from_numpy(self.data['source'][rows]);response,nz,epsilon=streams.draw(len(source),'cpu')
                noises=fixed_noises(extension,nz,epsilon)
                # Deliberately saturate only one hypothesis to exercise its
                # individual STE mask and actual-candidate regularizer.
                noises['slow']=noises['slow'].clone();noises['slow'][:,160:180]+=.55
                batches.append((source,response,noises))
            outputs=[direct.signal_forward(x,n) for x,n,_ in batches]
            waves={h:torch.cat([candidates(o['signal_pre'],noises)[h] for o,(_,_,noises) in zip(outputs,batches)]) for h in HYPOTHESES}
            objective=sum((i+1)*waves[h].square().mean()/3 for i,h in enumerate(HYPOTHESES))
            for o,(_,_,noises) in zip(outputs,batches):
                regular=o['base_regular']+sum(saturation_penalty(o['signal_pre']+noises[h]) for h in HYPOTHESES)/3
                objective+=(.01*regular+.1*o['trend'])/len(batches)
            objective.backward()
            with torch.no_grad():
                snapshots={h:torch.cat([candidates(replay.signal_forward(x,n)['signal_pre'],noises)[h] for x,n,noises in batches]) for h in HYPOTHESES}
            leaves={h:v.detach().requires_grad_(True) for h,v in snapshots.items()}
            gradients={h:torch.autograd.grad((i+1)*leaves[h].square().mean()/3,leaves[h])[0] for i,h in enumerate(HYPOTHESES)}
            replay_backward(replay,batches,gradients,expected=snapshots)
            difference=reference=0.
            for (name,p),(name2,q) in zip(direct.named_parameters(),replay.named_parameters()):
                self.assertEqual(name,name2);self.assertEqual(p.grad is None,q.grad is None,name)
                if p.grad is not None:
                    difference+=float((p.grad.double()-q.grad.double()).square().sum())
                    reference+=float(p.grad.double().square().sum())
            self.assertLess((difference/max(reference,1e-30))**.5,2e-6,arm)

    def test_real_optimizer_steps_freeze_noise_and_count_each_D(self):
        common=common_initial_state(self.stats['model'],self.cfg['seeds']['initialization'])
        engine=Response007Engine(self.data,self.stats,self.cfg,self.noise,common,arm='full',device='cpu')
        noise_hash=state_digest(engine.extension.model005.state_dict())
        engine.set_stage('mean_response');a=engine.step(audit=True,verify_replay=True)
        self.assertEqual(a['G_updates'],1);self.assertEqual(sum(a['D_updates'].values()),0)
        engine.set_stage('joint_stochastic');b=engine.step(audit=True,verify_replay=True)
        self.assertEqual(b['G_updates'],2);self.assertEqual(sum(b['D_updates'].values()),3)
        for name in ('physical_head','basis_head','envelope_head','local_filter'):
            self.assertGreater(b['G_branch_postclip_gradient_norms'][name],0.,name)
        engine.set_stage('low_learning_rate');c=engine.step(verify_replay=True)
        self.assertEqual(c['G_updates'],3);self.assertEqual(sum(c['D_updates'].values()),6)
        self.assertEqual(noise_hash,state_digest(engine.extension.model005.state_dict()))
        self.assertTrue(all(p.grad is None for p in engine.extension.parameters()))
        self.assertEqual(c['noise_updates'],0);self.assertEqual(c['population_updates'],0)

    def test_evaluation_paths_on_fabricated_data(self):
        from n01.evaluate_response007 import (generate_split,paired_width_bootstrap,distance_receipt,
            fit_classifier,repeats,intervention,population_sensitivity)
        from n01.losses_v2 import N01LossV2
        cfg=copy.deepcopy(self.cfg);ev=cfg['evaluation']
        ev.update(chunk_rows=16,sample_waveform_rows=16,repeat_parents=2,repeat_draws=4,
            intervention_pairs=2,intervention_draws=2,bootstrap_draws=12,updates_per_classifier=2,classifier_batch_per_class=4)
        common=common_initial_state(self.stats['model'],31)
        models={arm:make_signal(self.stats['model'],arm,common,31,'cpu') for arm in ARMS}
        for model in models.values():model.set_stage('low_learning_rate');model.eval().requires_grad_(False)
        extension=make_extension(self.noise,'cpu')
        objective=N01LossV2(loss_tensor_statistics(self.stats['loss']),seed=cfg['seeds']['train_projection'],evaluation_seed=cfg['seeds']['evaluation_projection'])
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);selected=np.arange(8)
            result=generate_split('DEV',self.data,models,extension,self.stats,cfg,out,selected,'cpu')
            self.assertEqual(len(result['parents']),256)
            self.assertEqual(result['matched']['full']['fast'].shape,(8,480))
            report,samples=paired_width_bootstrap(result,self.data,self.stats,cfg,'DEV')
            self.assertEqual(samples.shape,(12,2,10))
            dist=distance_receipt(objective,result['matched']['full']['fast'],self.data['real'][:8],objective.create_evaluation_context())
            self.assertTrue(np.isfinite(dist['components']['wave']))
            rep=repeats(self.data,models,extension,self.stats,cfg,out,'cpu')
            self.assertEqual(rep['engineering']['deterministic']['nonpositive_Q'],0)
            dep=intervention(self.data,models,extension,self.stats,cfg,out,'cpu')
            self.assertLessEqual(dep['full']['permutation_maximum_absolute_error_V'],dep['full']['permutation_float32_absolute_tolerance_V'])
            pop=population_sensitivity(result,self.data,cfg)
            self.assertEqual(pop['additional_generation_rows'],0)
            c=fit_classifier(self.data['real'][:8],self.data['real'][8:16],self.data['real'][16:24],self.data['real'][24:32],
                np.arange(8),np.arange(8,16),cfg,out/'classifier','cpu')
            self.assertEqual(c['updates'],2);self.assertTrue(0<=c['AUC']<=1)
            with np.load(out/'DEV_sample_waveforms.npz') as values:
                for h in HYPOTHESES:
                    assembled=values['full_signal_pre']+values['noise_'+h]
                    expected=np.rint((np.clip(assembled,-.5,.5)+.5)*16383)/16383-.5
                    np.testing.assert_array_equal(np.rint((values['full_'+h]+.5)*16383),np.rint((expected+.5)*16383))
                    np.testing.assert_allclose(values['full_'+h],expected,rtol=0,atol=4*np.finfo(np.float64).eps)


if __name__=='__main__':unittest.main()
