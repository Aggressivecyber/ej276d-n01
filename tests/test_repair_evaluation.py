"""Artificial repair endpoint invariants; never open scientific TRAIN/VAL."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from n01.electronics_v2 import make_n01_response_v2
from n01.losses import fit_loss_statistics
from n01.losses_v2 import create_projection_context
from n01.data import sha256,write_json
from n01.evaluate_repair import (EXPERIMENT_ID,_state_digest,validate_repair_checkpoint,validate_frozen_inputs,
    restore_repair_model,make_evaluation_objective,distance_receipt,equal_sample_selection,
    matched_classifier_selection,allrow_observables,noise_stream_receipt,_raw_PSD_receipt,
    _artifact_table,_verify_completion,evaluate_repair_checkpoint,annotate_matched_uncertainty)


def synthetic(count=32):
    g=torch.Generator().manual_seed(701)
    t=torch.arange(480,dtype=torch.float64)
    pulse=torch.where(t>=110,.11*torch.exp(-(t-145).square()/700),torch.zeros_like(t))
    return pulse[None]*(.7+.6*torch.rand(count,1,generator=g,dtype=torch.float64))+.0007*torch.randn(count,480,generator=g,dtype=torch.float64)


class RepairEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.wave=synthetic()
        cls.stats=fit_loss_statistics(cls.wave,cls.wave)
        cls.init=dict(rms=.001,log_scale_std=.02,offset_mean=0.,offset_std=.0001)
        cls.model=make_n01_response_v2(variant="deterministic",electronics_initialization=cls.init)
        cls.value=dict(schema_version=2,loss_version=2,model=cls.model.state_dict(),
            model_statistics=cls.model.stats,loss_statistics=cls.stats,noise_initialization=cls.init,
            electronics_state_sha256=_state_digest(cls.model.electronics.state_dict()),noise_gate_sha256="a"*64,
            config=dict(projection_seed=41,response=dict(updates_per_arm=19000,evaluation_projection_seed=43),
                        train_container_sha256="b"*64,identities_sha256="c"*64),
            seed=9,variant="deterministic",arm="deterministic",update=19000,
            learn_population=False,population=dict(theta=torch.zeros(3)))

    def test_old_and_interim_checkpoints_are_rejected(self):
        for changes in (dict(schema_version=1),dict(loss_version=1),dict(update=15000),
                        dict(learn_population=True),dict(population=dict(theta=torch.tensor([.01,0,0])))):
            with self.assertRaises(ValueError):validate_repair_checkpoint(self.value|changes)
        with self.assertRaisesRegex(ValueError,"missing"):validate_repair_checkpoint(dict(model={}))
        bad=copy.deepcopy(self.value);bad["config"]["response"]["evaluation_projection_seed"]=41
        with self.assertRaisesRegex(ValueError,"independent"):validate_repair_checkpoint(bad)

    def test_strict_v2_restore_and_frozen_electronics_digest(self):
        restored=restore_repair_model(self.value)
        self.assertEqual(_state_digest(restored.state_dict()),_state_digest(self.model.state_dict()))
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        bad=self.value|dict(electronics_state_sha256="0"*64)
        with self.assertRaisesRegex(ValueError,"electronics state"):restore_repair_model(bad)
        state=dict(self.value["model"]);state.pop("electronics.voltage_scale")
        with self.assertRaises(RuntimeError):restore_repair_model(self.value|dict(model=state))

    def test_evaluation_bank_is_independent_and_ignores_training_loss_state(self):
        class ExplodingState(dict):
            def __iter__(self):raise AssertionError("Training loss_state must not be read")
            def __getitem__(self,key):raise AssertionError("Training loss_state must not be read")
        a,context=make_evaluation_objective(self.value|dict(loss_state=ExplodingState()))
        b,other=make_evaluation_objective(self.value|dict(loss_state=dict(arbitrary=torch.zeros(99))))
        self.assertEqual(context.metadata(),other.metadata())
        self.assertEqual(context.kind,"evaluation");self.assertEqual(context.seed,43)
        self.assertNotEqual(context.fingerprint,create_projection_context(19000,41).fingerprint)
        with torch.no_grad():
            before=distance_receipt(a,self.wave.numpy(),self.wave.numpy())
            after=distance_receipt(b,self.wave.numpy(),self.wave.numpy())
        self.assertEqual(before,after)
        self.assertTrue(all(abs(x)<1e-12 for x in before["components"].values()))
        self.assertTrue(before["uniform_comparison_weights"])

    def test_positive_distance_and_bank_remains_frozen(self):
        objective,bank=make_evaluation_objective(self.value)
        old=bank.fingerprint
        altered=self.wave.clone();altered[:,150:220]+=.015
        with torch.no_grad():receipt=distance_receipt(objective,altered.numpy(),self.wave.numpy())
        self.assertGreater(receipt["components"]["wave"],0.)
        self.assertGreater(receipt["components"]["QPSD"],0.)
        self.assertEqual(bank.fingerprint,old);bank.validate()
        self.assertEqual(set(receipt["wave_subcomponents"]),{"swd","coordinate","short_lag"})

    def test_equal_N_one_parent_one_draw_and_same_REAL_quota(self):
        fake_ids=np.repeat(np.arange(512),4)
        groups=np.repeat(np.arange(12),50)
        a=equal_sample_selection(fake_ids,groups)
        b=equal_sample_selection(fake_ids,groups)
        self.assertEqual(len(a["fake_rows"]),256)
        self.assertEqual(len(np.unique(fake_ids[a["fake_rows"]])),256)
        self.assertEqual(len(a["real0_rows"]),256);self.assertEqual(len(a["real1_rows"]),256)
        self.assertEqual(np.intersect1d(a["real0_rows"],a["real1_rows"]).size,0)
        for g in np.unique(groups):
            self.assertEqual(np.sum(groups[a["real0_rows"]]==g),np.sum(groups[a["real1_rows"]]==g))
        self.assertTrue(all(np.array_equal(a[k],b[k]) for k in a))

    def test_primary_C2ST_TRAIN_VAL_parent_group_isolation_and_all_groups(self):
        train_ids=np.arange(6593)
        val_ids=np.repeat(np.arange(10000,10512),4)
        train_groups=np.repeat(np.arange(422),50)
        val_groups=np.repeat(np.arange(1000,1012),50)
        case,rr,s=matched_classifier_selection(train_ids,val_ids,train_groups,val_groups)
        self.assertEqual([len(case[k]) for k in ("fit0","fit1","test0","test1")],[2048,2048,256,256])
        self.assertEqual([len(rr[k]) for k in ("fit0","fit1","test0","test1")],[2048,2048,256,256])
        self.assertEqual(np.intersect1d(s["source_train_ids"],s["source_VAL_ids"]).size,0)
        self.assertEqual(np.unique(s["source_train_ids"]).size,2048)
        self.assertEqual(np.unique(s["source_VAL_ids"]).size,256)
        self.assertEqual(np.unique(train_groups[s["real_train0_rows"]]).size,422)
        self.assertEqual(np.unique(val_groups[s["real_VAL0_rows"]]).size,12)
        self.assertTrue(np.array_equal(case["fit1"],rr["fit0"]))
        self.assertTrue(np.array_equal(case["test1"],rr["test0"]))
        for groups,a,b in ((train_groups,s["real_train0_rows"],s["real_train1_rows"]),
                            (val_groups,s["real_VAL0_rows"],s["real_VAL1_rows"])):
            self.assertEqual(np.intersect1d(a,b).size,0)
            for g in np.unique(groups):self.assertEqual(np.sum(groups[a]==g),np.sum(groups[b]==g))
        with self.assertRaisesRegex(ValueError,"parent overlap"):
            matched_classifier_selection(train_ids,np.repeat(np.arange(512),4),train_groups,val_groups)
        with self.assertRaisesRegex(ValueError,"group overlap"):
            matched_classifier_selection(train_ids,val_ids,train_groups,np.repeat(np.arange(12),50))

    def test_nonpositive_Q_and_raw_zero_PSD_are_saved(self):
        wave=np.zeros((3,480),dtype=np.float64)
        wave[0,150:400]=.1;wave[2,150:400]=-.1
        obs=allrow_observables(wave,1e-5)
        self.assertEqual(len(obs["PSD_raw"]),3)
        self.assertTrue(np.isnan(obs["PSD_raw"][1]))
        self.assertEqual(obs["Q_nonpositive"].tolist(),[False,True,True])
        self.assertEqual(obs["PSD_guarded"].tolist(),[False,True,False])
        receipt=_raw_PSD_receipt(obs)
        self.assertEqual(receipt["nonfinite_rows"],1);self.assertIsNone(receipt["mean"])
        self.assertTrue(receipt["all_rows_preserved"])

    def test_latent_stream_fingerprint_same_across_variants(self):
        deterministic=make_n01_response_v2(variant="deterministic",electronics_initialization=self.init)
        full=make_n01_response_v2(variant="full",electronics_initialization=self.init)
        a=noise_stream_receipt(deterministic,256,51)
        b=noise_stream_receipt(full,256,51)
        c=noise_stream_receipt(full,256,52)
        self.assertEqual(a,b);self.assertNotEqual(a["noise_tensor_stream_sha256"],c["noise_tensor_stream_sha256"])

    def test_registered_container_identity_and_actual_passed_gate_binding(self):
        with tempfile.TemporaryDirectory(dir=r'E:\GAN\runtime_tmp',prefix='repair_eval_test_') as td:
            run=Path(td);checkpoint=run/'full/stage_19000.pt';checkpoint.parent.mkdir()
            gate=run/'noise_validation/NOISE_GATE.json'
            write_json(gate,dict(passed=True,electronics_state_sha256=self.value["electronics_state_sha256"]))
            v=self.value|dict(noise_gate_sha256=sha256(gate))
            receipt=validate_frozen_inputs(checkpoint,v,"b"*64,"c"*64)
            self.assertTrue(receipt["passed"])
            with self.assertRaisesRegex(ValueError,"registered"):
                validate_frozen_inputs(checkpoint,v,"0"*64,"c"*64)
            write_json(gate,dict(passed=False,electronics_state_sha256=self.value["electronics_state_sha256"]))
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                validate_frozen_inputs(checkpoint,v,"b"*64,"c"*64)
            v=v|dict(noise_gate_sha256=sha256(gate))
            with self.assertRaisesRegex(ValueError,"did not pass"):
                validate_frozen_inputs(checkpoint,v,"b"*64,"c"*64)
            write_json(gate,dict(passed=True,electronics_state_sha256="0"*64))
            with self.assertRaisesRegex(ValueError,"electronics differ"):
                validate_frozen_inputs(checkpoint,v|dict(noise_gate_sha256=sha256(gate)),"b"*64,"c"*64)

    def test_invalid_checkpoint_is_rejected_before_VAL_load(self):
        with tempfile.TemporaryDirectory(dir=r'E:\GAN\runtime_tmp',prefix='repair_eval_test_') as td:
            root=Path(td);c=root/'old.pt';data=root/'dummy.bin';identity=root/'identity.bin'
            torch.save(dict(schema_version=1,model={}),c);data.write_bytes(b'artificial');identity.write_bytes(b'identities')
            with patch('n01.evaluate_repair.load_validation') as load:
                with self.assertRaises(ValueError):evaluate_repair_checkpoint(c,data,identity,root/'out',device='cpu')
                load.assert_not_called()

    def test_recursive_artifact_manifest_detects_changed_classifier(self):
        with tempfile.TemporaryDirectory(dir=r'E:\GAN\runtime_tmp',prefix='repair_eval_test_') as td:
            out=Path(td);nested=out/'matched_classifier/view/classifier.pt';nested.parent.mkdir(parents=True)
            nested.write_bytes(b'artificial classifier state')
            artifacts=_artifact_table(out)
            self.assertIn('matched_classifier/view/classifier.pt',artifacts)
            gate=out/'frozen_gate.json'
            write_json(gate,dict(passed=True,electronics_state_sha256='d'*64))
            gate_sha=sha256(gate)
            report=dict(experiment_id=EXPERIMENT_ID,checkpoint_sha256='a'*64,container_sha256='b'*64,identity_sha256='c'*64,
                noise_gate=dict(path=str(gate),sha256=gate_sha),noise_gate_sha256=gate_sha,electronics_state_sha256='d'*64)
            write_json(out/'endpoint_report.json',report)
            write_json(out/'EVALUATION_COMPLETE.json',dict(report_sha256=sha256(out/'endpoint_report.json'),artifacts=artifacts))
            self.assertEqual(_verify_completion(out,'a'*64,'b'*64,'c'*64),report)
            write_json(gate,dict(passed=False,electronics_state_sha256='d'*64))
            with self.assertRaisesRegex(ValueError,"noise gate changed"):_verify_completion(out,'a'*64,'b'*64,'c'*64)
            write_json(gate,dict(passed=True,electronics_state_sha256='d'*64))
            nested.write_bytes(b'changed artificial classifier state')
            with self.assertRaisesRegex(ValueError,"artifact changed"):_verify_completion(out,'a'*64,'b'*64,'c'*64)

    def test_limited_acquisition_uncertainty_metadata_preserves_interval(self):
        original=dict(uncertainty=dict(limited_group_count=False,resamples=400,raw=dict(percentile_95_interval=[.4,.6])))
        result=annotate_matched_uncertainty(original,12)
        self.assertTrue(result["limited_original_group_count"])
        self.assertTrue(result["uncertainty"]["limited_original_group_count"])
        self.assertTrue(result["uncertainty"]["limited_group_count"])
        self.assertFalse(result["uncertainty"]["legacy_helper_limited_group_count"])
        self.assertEqual(result["uncertainty"]["raw"]["percentile_95_interval"],[.4,.6])
        self.assertEqual(result["uncertainty"]["resamples"],400)


if __name__=='__main__':unittest.main()
