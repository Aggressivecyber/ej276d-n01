"""Synthetic005 evaluation semantics; no scientific arrays/classifier fits."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from n01 import noise_validation_acf005 as m
from n01.acf20 import VARIANCE_FLOOR,mean_acf20
from n01.data import sha256,write_json
from n01.noise_validation import balanced_real_domains,baseline_statistics


def synthetic_gate():
    controls=dict(AUC=.58,percentile_95_interval=[.3,.66])
    return dict(checks={key:key not in m.CONTROL_KEYS for key in m.CANDIDATE_KEYS+m.CONTROL_KEYS},
        passed=False,failed_checks=list(m.CONTROL_KEYS),next_action='historical_action_must_never_be_executed',
        classifiers={view:dict(generated_REAL=dict(AUC=.52,percentile_95_interval=[.43,.61]),
                             REAL_REAL=copy.deepcopy(controls)) for view in ('raw','demean','detrend')})


class Evaluation005Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.helpers=m._null_helpers(ROOT)
        cls.legacy={p:sha256(p) for p in [ROOT/'src/n01/noise_validation.py',ROOT/'src/n01/acf20.py',
            ROOT/'src/n01/model.py',ROOT/'scripts/calibrate_real_real_null.py']}

    @classmethod
    def tearDownClass(cls):
        for path,value in cls.legacy.items():
            if sha256(path)!=value:raise AssertionError('Frozen legacy source changed: '+str(path))

    def test_shared_event_ACF_matches_preserved_numpy_without_pooled_ratios(self):
        rng=np.random.default_rng(220)
        x=rng.normal(size=(31,100))*.0006
        x[:10]=np.round(x[:10]*16383)/16383
        x[10:20]=x[10:20]*np.exp(rng.normal(size=(10,1)))+rng.normal(size=(10,1))*.003
        x[20:24]=.002
        x[24:]=.001+rng.normal(size=(7,100))*1e-10
        _,a=m._events(x)
        b=x-x.mean(1,keepdims=True)
        expected=np.stack([(b[:,:-lag]*b[:,lag:]).mean(1)/np.maximum((b*b).mean(1),VARIANCE_FLOOR)
                           for lag in range(1,21)],1)
        np.testing.assert_allclose(a,expected,rtol=2e-13,atol=2e-14)
        np.testing.assert_allclose(a.mean(0),baseline_statistics(x)['acf_mean'],rtol=2e-13,atol=2e-14)
        np.testing.assert_array_equal(m._event_mean(a),mean_acf20(torch.as_tensor(x)).numpy())
        for bad in (np.zeros((0,100)),np.zeros((2,480)),np.full((2,100),np.nan)):
            with self.assertRaises(ValueError):m._events(bad)

    def test_candidate_checks_are_separate_and_never_modify_legacy_gate(self):
        gate=synthetic_gate();before=copy.deepcopy(gate)
        result=m.candidate_screen(gate)
        self.assertTrue(result['passed']);self.assertEqual(len(result['checks']),10)
        self.assertFalse(gate['passed']);self.assertEqual(gate,before)
        self.assertEqual(set(result['checks']),set(m.CANDIDATE_KEYS))
        self.assertIn('not statistical',result['raw_classifier']['interpretation'])
        gate['checks']['acf']=False
        result=m.candidate_screen(gate)
        self.assertFalse(result['passed']);self.assertEqual(result['failed_checks'],['acf'])
        gate['checks']['extra_adaptive_gate']=True
        with self.assertRaisesRegex(ValueError,'check set changed'):m.candidate_screen(gate)

    def test_control_reproduction_is_bitwise_including_dtype_and_signed_zero(self):
        base=dict(labels=np.array([0.,1.]),scores=np.array([0.,.5]),groups0=np.array([5]),groups1=np.array([5]))
        self.assertTrue(all(m.compare_control_arrays(base,copy.deepcopy(base)).values()))
        for key in ('labels','scores','groups0','groups1'):
            changed=copy.deepcopy(base)
            if key=='scores':changed[key][1]=np.nextafter(changed[key][1],1.)
            else:changed[key][0]+=1
            with self.assertRaisesRegex(ValueError,key):m.compare_control_arrays(base,changed)
        changed=copy.deepcopy(base);changed['scores'][0]=-0.
        with self.assertRaisesRegex(ValueError,'scores'):m.compare_control_arrays(base,changed)
        changed=copy.deepcopy(base);changed['labels']=changed['labels'].astype(np.float32)
        with self.assertRaisesRegex(ValueError,'labels'):m.compare_control_arrays(base,changed)

    def test_conditional_permutation_preserves_group_quotas_ties_plus_one_and_MC_scope(self):
        prediction=dict(labels=np.r_[np.zeros(4),np.ones(4)],scores=np.full(8,.25),
                        groups0=np.array([2,2,4,4]),groups1=np.array([2,2,4,4]))
        controls=dict(conditional_seed=350,conditional_permutations=31)
        receipt,numeric=m.conditional_raw_receipt(prediction,controls,self.helpers)
        np.testing.assert_array_equal(numeric['positive_quota'],[2,2])
        np.testing.assert_array_equal(numeric['permutation_AUC'],np.full(31,.5))
        self.assertEqual(receipt['tail_count'],31);self.assertEqual(receipt['plus_one_p_value'],1.)
        self.assertEqual(receipt['conditional_tail_fraction_K_over_draws'],1.)
        self.assertEqual(receipt['conditional_tail_probability_MC_Wilson95'],self.helpers.wilson_interval(31,31))
        self.assertFalse(receipt['classifier_refitted']);self.assertTrue(receipt['scores_fixed'])
        self.assertIn('not_p_or_acquisitionCI',receipt['Wilson_scope'])
        # A separated score example makes +1, rather than the raw tail fraction,
        # observable while retaining two equal class quotas in every group.
        prediction['scores']=np.arange(8,dtype=np.float64)
        receipt,numeric=m.conditional_raw_receipt(prediction,controls,self.helpers)
        expected=int(np.sum(abs(numeric['permutation_AUC']-.5)>=abs(receipt['observed_AUC']-.5)))
        self.assertEqual(receipt['tail_count'],expected)
        self.assertEqual(receipt['plus_one_p_value'],(expected+1)/32)
        np.testing.assert_array_equal(numeric['permutation_AUC'],m.conditional_raw_receipt(prediction,controls,self.helpers)[1]['permutation_AUC'])
        prediction['groups1'][0]=4
        with self.assertRaisesRegex(ValueError,'quotas'):m.conditional_raw_receipt(prediction,controls,self.helpers)

    def test_group_bootstrap_is_event_weighted_and_independent_generated_resampling(self):
        # Unequal group sizes distinguish event-weighted means from equal-group
        # averages. Replicate columns to check all20 dimensions together.
        gen=np.repeat(np.array([.1,.3,.7])[:,None],20,1)
        real=np.repeat(np.array([0.,1.,1.,1.])[:,None],20,1)
        groups=np.array([11,22,22,22]);seed=441;draws=13
        summary,numeric=m.bootstrap_acf_difference(gen,real,groups,seed,draws)
        rng=np.random.default_rng(seed);expected=[]
        for i in range(draws):
            selected_groups=rng.integers(0,2,2)
            mult=np.bincount(selected_groups,minlength=2)
            selected_gen=rng.integers(0,3,3)
            rows=np.concatenate([np.flatnonzero(groups==[11,22][j]) for j in selected_groups])
            real_mean=real[rows].mean(0);gen_mean=gen[selected_gen].mean(0)
            np.testing.assert_array_equal(numeric['REAL_group_multiplicity'][i],mult)
            np.testing.assert_allclose(numeric['REAL_bootstrap_means'][i],real_mean,rtol=0,atol=0)
            np.testing.assert_allclose(numeric['generated_bootstrap_means'][i],gen_mean,rtol=0,atol=0)
            expected.append(gen_mean-real_mean)
        np.testing.assert_allclose(numeric['differences'],expected,rtol=0,atol=0)
        both=np.flatnonzero(np.all(numeric['REAL_group_multiplicity']==[1,1],axis=1))
        self.assertTrue(len(both)>0);self.assertEqual(numeric['REAL_bootstrap_means'][both[0],0],.75)
        self.assertNotEqual(numeric['REAL_bootstrap_means'][both[0],0],.5)
        point=gen.mean(0)-real.mean(0)
        radius=np.quantile(np.max(abs(numeric['differences']-point),axis=1),.95)
        np.testing.assert_array_equal(summary['marginal95'],np.quantile(expected,[.025,.975],axis=0))
        np.testing.assert_array_equal(summary['simultaneous95_centered_max'],np.stack([point-radius,point+radius]))
        self.assertFalse(summary['acceptance_gate']);self.assertFalse(summary['training_randomness_included'])
        np.testing.assert_array_equal(numeric['differences'],m.bootstrap_acf_difference(gen,real,groups,seed,draws)[1]['differences'])
        with self.assertRaises(ValueError):m.bootstrap_acf_difference(gen,real[:,:19],groups,seed,draws)

    def test_train_decision_strict_improvement_and_absolute_100_only_boundary(self):
        target=np.zeros(20);old=np.full(20,.02)
        result=m.train_decision(np.full(20,.01),old,target,True)
        self.assertTrue(result['TRAIN_match']);self.assertTrue(result['TRAIN_improved'])
        self.assertTrue(result['read_only_480_design_check_allowed'])
        self.assertEqual(result['TRAIN_current_acf20_loss'],(.01/.015)**2)
        for new,reference,candidate,match,improved in [
            (np.full(20,.015),old,True,True,True),
            (np.full(20,np.nextafter(.015,1)),old,True,False,True),
            (old,old,True,False,False),
            (np.zeros(20),np.zeros(20),True,True,False),
            (np.full(20,.01),old,False,True,True),
            (np.r_[.012,np.zeros(19)],np.full(20,.011),True,True,False)]:
            result=m.train_decision(new,reference,target,candidate)
            self.assertEqual(result['TRAIN_match'],match);self.assertEqual(result['TRAIN_improved'],improved)
            self.assertEqual(result['read_only_480_design_check_allowed'],bool(match and improved and candidate))
            self.assertFalse(result['response_training_authorized']);self.assertFalse(result['scientifically_qualified'])
            self.assertFalse(result['independent_acquisition_validated']);self.assertEqual(result['numerical_tolerance'],0.)
        with self.assertRaises(ValueError):m.train_decision(np.zeros(19),old,target,True)

    def test_reference_SHA_checked_before_array_decode(self):
        with tempfile.TemporaryDirectory(prefix='acf005_refs_',dir=r'E:/GAN/runtime_tmp') as temporary:
            root=Path(temporary);path=root/'ref.txt';path.write_text('frozen synthetic reference',encoding='utf8')
            config={'references':{'toy':dict(path='ref.txt',sha256=sha256(path))}}
            self.assertEqual(m.verify_references(config,root)['toy']['sha256'],sha256(path))
            path.write_text('changed',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):m.verify_references(config,root)

    def test_synthetic_complete_receipt_preserves_legacy_and_pending_runner_binding(self):
        with tempfile.TemporaryDirectory(prefix='acf005_eval_',dir=r'E:/GAN/runtime_tmp') as temporary:
            root=Path(temporary);out=root/'evaluation';diag=root/'diagnostic';out.mkdir();diag.mkdir()
            train=np.zeros((21047,100));validation=np.zeros((599,100))
            train_groups=np.arange(21047)%422
            val_groups=422+np.arange(599)//50
            selected0,selected1=balanced_real_domains(val_groups,256,seed=516)
            np.savez_compressed(out/'baseline_populations.npz',generated=np.zeros((8192,100)),
                REAL_validation=validation,real_validation_groups=val_groups,
                real_validation_test0=selected0,real_validation_test1=selected1)
            zero=np.zeros((8192,100));ramp=np.broadcast_to(np.linspace(-.001,.001,100),(8192,100)).copy()
            np.savez_compressed(diag/'paired_populations.npz',initial_pre_adc=zero,initial_post_adc=zero,
                v004_pre_adc=ramp,v004_post_adc=ramp,v005_pre_adc=zero,v005_post_adc=zero)
            gate=synthetic_gate();write_json(out/'LEGACY_GATE.json',gate);oldhash=sha256(out/'LEGACY_GATE.json')
            predictions=dict(labels=np.r_[np.zeros(256),np.ones(256)],scores=np.linspace(.2,.8,512),
                groups0=val_groups[selected0],groups1=val_groups[selected1])
            references={}
            for view in ('raw','demean','detrend'):
                oldpath=root/('control004_'+view+'.npz');np.savez_compressed(oldpath,**predictions)
                current=out/'classifiers'/view/'REAL_REAL';current.mkdir(parents=True)
                np.savez_compressed(current/'predictions.npz',**predictions)
                references['control004_'+view]=dict(path=oldpath.name,sha256=sha256(oldpath))
            calibration=dict(repeats=128,binding=dict(registered_rules=dict(test_label_permutations=255)),
                legacy_joint_bootstrap_false_positive=dict(rejections=9,trials=128,Wilson95=self.helpers.wilson_interval(9,128)),
                conditional_permutation_false_positive=dict(rejections=7,trials=128,Wilson95=self.helpers.wilson_interval(7,128)))
            write_json(root/'calibration003.json',calibration)
            references['calibration003']=dict(path='calibration003.json',sha256=sha256(root/'calibration003.json'))
            config=dict(run_id='synthetic_only',references=references,
                acf20=dict(lags=list(range(1,21)),scale=.015,coefficient=1.,variance_floor_V2=VARIANCE_FLOOR),
                controls=dict(conditional_seed=517,conditional_permutations=9999),
                uncertainty=dict(bootstrap_resamples=1000,TRAIN_seed=518,development_seed=519))
            with patch.object(m,'_null_helpers',return_value=self.helpers):
                receipt=m.evaluate_acf005(gate,train,validation,train_groups,val_groups,config,out,root=root,diagnostic_dir=diag)
            self.assertEqual(sha256(out/'LEGACY_GATE.json'),oldhash);self.assertEqual(receipt['legacy_gate'],gate)
            self.assertFalse(receipt['legacy_gate']['passed']);self.assertTrue(receipt['candidate_checks']['passed'])
            implementation=receipt['implementation_checks']
            self.assertTrue(implementation['pending_runner_binding']);self.assertTrue(all(implementation['checks'].values()))
            self.assertEqual(implementation['status'],'module_checks_passed_pending_runner_binding')
            self.assertEqual(implementation['runner_assertions_required'],[
                'initial_state_equals004','immutable_config_source_input','all_TRAIN_coverage',
                'fixed_TRAIN20_target','fixed5000_checkpoint_reopened','frozen_generator_during_evaluation',
                'explicit1024_replay','diagnostic_RNG_isolation','TRAIN_initialization_only','VAL_only_after_endpoint',
                'exact_classifier_replay'])
            self.assertTrue(receipt['decision']['read_only_480_design_check_allowed'])
            self.assertFalse(receipt['decision']['response_training_authorized']);self.assertFalse(receipt['decision']['scientifically_qualified'])
            self.assertEqual(receipt['support_scope']['supported_samples'],100)
            self.assertFalse(receipt['support_scope']['full480_integration_supported'])
            self.assertEqual(receipt['support_scope']['independent_acquisition_validations'],0)
            control=receipt['control_calibration']
            self.assertEqual(control['status'],'measurement_uncertain_single_exposed_acquisition')
            self.assertFalse(control['measurement_certified']);self.assertEqual(control['newly_fitted_classifiers'],0)
            self.assertEqual(control['current_raw']['draws'],9999);self.assertEqual(control['reference003']['method_draws'],255)
            self.assertEqual(control['reference003']['refits'],128)
            self.assertEqual(receipt['target_ACF20_sha256'],hashlib.sha256(np.zeros(20,dtype='<f8').tobytes()).hexdigest())
            self.assertEqual(len(receipt['artifacts']),7)
            for name,value in receipt['artifacts'].items():self.assertEqual(sha256(out/name),value['sha256'])
            with np.load(out/'bootstrap_TRAIN.npz',allow_pickle=False) as saved:
                self.assertEqual(saved['differences'].shape,(1000,20));self.assertEqual(saved['REAL_group_multiplicity'].shape,(1000,422))
            table=json.loads((out/'ACF_PER_LAG.json').read_text(encoding='utf8'))
            self.assertEqual(len(table['rows']),20);self.assertEqual(len(table['series']),9)
            self.assertNotEqual(table['series']['v005_post_adc']['population_role'],table['series']['legacy_generated_post_adc']['population_role'])
            with self.assertRaisesRegex(ValueError,'Preserve completed'):m.evaluate_acf005(gate,train,validation,train_groups,val_groups,config,out,root=root,diagnostic_dir=diag)


if __name__=='__main__':unittest.main()
