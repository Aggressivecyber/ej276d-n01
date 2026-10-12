"""Artificial CPU tests for005; no scientific decode, VAL or GPU."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from n01 import train_noise_acf005 as runner
from n01.acf20 import ACF20Loss,mean_acf20
from n01.electronics_v3 import ObservedBaselineNoiseV3,fit_baseline_initialization_v3
from n01.losses_v2 import BaselineDistributionLoss,fit_baseline_statistics

TMP=r"E:\GAN\runtime_tmp"


def config():
    root=Path(__file__).resolve().parents[1]
    value=json.loads((root/"noise_acf005_config.json").read_text(encoding="utf-8"))
    value["noise"]["batch_size"]=32;value["cpu_threads"]=2;value["wall_limit_s"]=60
    value["diagnostics"]["rows"]=64;value["diagnostics"]["chunk_size"]=16
    value["diagnostics"]["trajectory_updates"]=[0,1,2,3]
    return value


def synthetic():
    r=np.random.default_rng(202610125)
    t=np.linspace(-1,1,100)
    offsets=r.normal(0,.00015,64);slopes=-.65*offsets+r.normal(0,.00005,64)
    return offsets[:,None]+slopes[:,None]*t+r.normal(0,.0006,(64,100)),np.repeat(np.arange(4),16)


def bindings():
    return dict(config_sha256="artificial",source_manifest_sha256="test_manifest",
                protocol_sha256="test_protocol",preregistration_sha256="test_preregistration",
                source_files={},reference_hashes={})


def references(root,cfg,real):
    init=fit_baseline_initialization_v3(real,**cfg["model"])
    model=ObservedBaselineNoiseV3(init)
    original=runner.common.snapshot(model);digest=runner.common.state_digest(original)
    for role,update in (("initial004",0),("final004",5000)):
        state=copy.deepcopy(original)
        if update:state["mean_units"]+=.04  # Synthetic final differs; no optimizer fit.
        path=root/(role+".pt")
        runner.common.atomic_save(dict(update=update,engineering=False,model_scope="observed_baseline100_only",
            model=state,model_state_sha256=runner.common.state_digest(state),initial_model_state=original,
            initialization=init,config=copy.deepcopy(cfg)),path)
        cfg["references"][role]["path"]=path.name
    return digest


class ACF005Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_canonical_gradient_is_sum_and_zero_angle_is_undefined(self):
        a=torch.nn.Parameter(torch.tensor([.21,-.43,.77],dtype=torch.float64))
        b=torch.nn.Parameter(torch.tensor([-.17,.37],dtype=torch.float64))
        old=(a.sin().square().sum()+b.square().sum())
        added=(a.cos().sum()+.6*b.sum()).square()
        g0=torch.autograd.grad(old,[a,b],retain_graph=True)
        ga=torch.autograd.grad(added,[a,b],retain_graph=True)
        record=runner.differentiated_step(old,added,old+added,[a,b])
        for p,x,y in zip([a,b],g0,ga):torch.testing.assert_close(p.grad,x+y,rtol=1e-14,atol=1e-14)
        self.assertLess(record["gradient_sum_max_absolute_difference"],1e-13)
        self.assertAlmostEqual(record["total_gradient_preclip_norm"],float(torch.cat([a.grad,b.grad]).norm()),places=13)
        pair,_,_=runner.gradient_pair([torch.zeros(2)],[torch.ones(2)])
        self.assertIsNone(pair["cosine"]);self.assertIsNone(pair["angle_degrees"])

    def test_fixed_diagnostic_does_not_mutate_streams_sampler_context_or_model(self):
        real,_=synthetic();cfg=config()
        model=ObservedBaselineNoiseV3(fit_baseline_initialization_v3(real,**cfg["model"]))
        objective=BaselineDistributionLoss(fit_baseline_statistics(torch.as_tensor(real)))
        ctx=objective.create_projection_context(7)
        sampler=runner.FullCoverageSampler(len(real),123);sampler.draw(9)
        rng=torch.Generator().manual_seed(9876)
        saved_rng=rng.get_state().clone();saved_visits=sampler.visits.copy()
        bank=runner.diagnostic_latents(model,cfg)
        p,q=runner.generate_diagnostic(model,bank,training_rng=rng,sampler=sampler,context=ctx)
        p2,q2=runner.generate_diagnostic(model,bank,training_rng=rng,sampler=sampler,context=ctx)
        self.assertTrue(np.array_equal(p,p2));self.assertTrue(np.array_equal(q,q2))
        self.assertTrue(torch.equal(saved_rng,rng.get_state()));np.testing.assert_array_equal(saved_visits,sampler.visits)
        self.assertEqual((bank["rows"],bank["chunk_size"],len(bank["chunks"])),(64,16,4))
        self.assertEqual(bank["seed"],cfg["noise"]["generation_seed"]+20000)
        self.assertTrue(model.training);ctx.validate()

    def test_engineering_retains_fixed_target_initial_role_gradients_and_paired_outputs(self):
        with tempfile.TemporaryDirectory(prefix="acf005_fit_",dir=TMP) as temp:
            root=Path(temp);real,groups=synthetic();cfg=config();initial=references(root,cfg,real)
            out=root/"science"
            model,state,checks=runner.fit005(real,groups,cfg,out,bindings(),root=root,device="cpu",engineering=True)
            self.assertEqual(state["update"],3);self.assertTrue(checks["initial_state_matches004"])
            self.assertEqual(state["initial004_model_state_sha256"],initial)
            self.assertEqual(runner.common.state_digest(state["initial_model_state"]),initial)
            torch.testing.assert_close(state["acf20_target"],mean_acf20(torch.as_tensor(real)),rtol=0,atol=0)
            self.assertEqual(state["acf20_target_sha256"],runner.array_digest(state["acf20_target"]))
            self.assertEqual(int(state["sampler"]["visits"].sum()),96)
            self.assertTrue((state["sampler"]["visits"]>0).all())
            self.assertEqual(checks["trajectory_updates"],[0,1,2,3])
            logs=[json.loads(x) for x in (out/"noise/training.jsonl").read_text().splitlines()]
            self.assertEqual(len(logs),3)
            for line in logs:
                self.assertEqual(line["forward_state_update"],line["update"]-1)
                self.assertEqual(len(line["acf20_signed_errors"]),20)
                self.assertAlmostEqual(line["old_total"]+line["acf20_loss"],line["total"],places=12)
                self.assertLess(line["gradient_sum_max_absolute_difference"],1e-10)
                self.assertEqual(line["noise_training_updates"],0)
            with np.load(out/"diagnostics/trajectory.npz") as z:
                self.assertEqual(z["pre_covariance_V2"].shape,(4,100,100))
                np.testing.assert_array_equal(z["updates"],[0,1,2,3])
            with np.load(out/"diagnostics/paired_populations.npz") as z:
                self.assertEqual(set(z.files),{"initial_pre_adc","initial_post_adc","v004_pre_adc","v004_post_adc","v005_pre_adc","v005_post_adc"})
                self.assertTrue(all(z[key].shape==(64,100) for key in z.files))
                self.assertFalse(np.array_equal(z["initial_pre_adc"],z["v004_pre_adc"]))
            self.assertFalse((out/"noise/final.pt").exists())
            with self.assertRaisesRegex(ValueError,"5000"):runner.validate_endpoint(state,cfg,bindings())
            state["engineering"]=False;state["update"]=5000
            with self.assertRaisesRegex(ValueError,"coverage"):runner.validate_endpoint(state,cfg,bindings())
            with self.assertRaises(FileExistsError):runner.fit005(real,groups,cfg,out,bindings(),root=root,device="cpu",engineering=True)
            replay=runner.common.save_replay(model,cfg,root/"replay.pt",bindings())
            self.assertTrue(runner.common.verify_replay(model,replay))

    def test_context_failure_after_step_keeps_update_and_failure_state(self):
        with tempfile.TemporaryDirectory(prefix="acf005_failure_",dir=TMP) as temp:
            root=Path(temp);real,groups=synthetic();cfg=config();references(root,cfg,real)
            holder={};create=BaselineDistributionLoss.create_projection_context;step=torch.optim.Adam.step
            def capture(self,*a,**kw):
                value=create(self,*a,**kw);holder["ctx"]=value;return value
            def corrupt(self,*a,**kw):
                value=step(self,*a,**kw);holder["ctx"].banks["baseline_wave"].add_(1e-6);return value
            out=root/"science"
            with mock.patch.object(BaselineDistributionLoss,"create_projection_context",capture),mock.patch.object(torch.optim.Adam,"step",corrupt):
                with self.assertRaisesRegex(RuntimeError,"mutated"):
                    runner.fit005(real,groups,cfg,out,bindings(),root=root,device="cpu",engineering=True)
            failure=json.loads((out/"noise/FAILURE.json").read_text())
            self.assertEqual(failure["update"],1);self.assertTrue(failure["optimizer_step_completed"])
            state=torch.load(out/"noise/failure_state.pt",weights_only=False)
            self.assertEqual(state["state"]["update"],1)
            self.assertTrue(state["state"]["projection_context"]["invalid_context"])
            self.assertIsNotNone(state["last_noise"]);self.assertIsNone(failure["capture_error"])

    def test_six_classifier_reopen_is_bitwise_and_score_tamper_stops(self):
        from n01.noise_validation import BaselineClassifier,_view,balanced_real_domains
        with tempfile.TemporaryDirectory(prefix="acf005_cnn_replay_",dir=TMP) as temp:
            out=Path(temp);r=np.random.default_rng(778)
            fake=r.normal(0,.0006,(512,100));real=r.normal(0,.0006,(599,100))
            groups=np.arange(599)%12;rows0,rows1=balanced_real_domains(groups,256,909)
            np.savez_compressed(out/"baseline_populations.npz",generated=fake,REAL_validation=real,
                real_validation_groups=groups,real_validation_test0=rows0,real_validation_test1=rows1)
            center=np.zeros(100,dtype=np.float32);scale=np.full(100,.0006,dtype=np.float32)
            # Artificial untrained networks; checkpoint update role is a fixture,
            # not a claimed training run. No optimizer exists in this test.
            for view in ("raw","demean","detrend"):
                for domain in ("generated_REAL","REAL_REAL"):
                    path=out/"classifiers"/view/domain;path.mkdir(parents=True)
                    if domain=="generated_REAL":a,b=fake[:256],real[rows0];g0,g1=np.arange(256),groups[rows0]
                    else:a,b=real[rows0],real[rows1];g0,g1=groups[rows0],groups[rows1]
                    values=(np.concatenate([_view(a,view),_view(b,view)]).astype(np.float32)-center)/scale
                    model=BaselineClassifier().eval()
                    with torch.no_grad():scores=model(torch.as_tensor(values)).numpy()
                    torch.save(dict(model=model.state_dict(),center=center,scale=scale,updates=200),path/"classifier.pt")
                    np.savez_compressed(path/"predictions.npz",scores=scores,labels=np.r_[np.zeros(256),np.ones(256)],groups0=g0,groups1=g1)
            rng_before=torch.get_rng_state().clone()
            with mock.patch.object(torch.optim.Adam,"step",side_effect=AssertionError("No fit permitted")):
                receipt=runner.verify_classifier_replay(out,"cpu")
            self.assertTrue(torch.equal(rng_before,torch.get_rng_state()))
            self.assertEqual(receipt["classifiers_reopened"],6)
            self.assertEqual(receipt["classifier_training_updates"],0)
            p=out/"classifiers/raw/generated_REAL/predictions.npz"
            with np.load(p) as z:values={k:z[k].copy() for k in z.files}
            values["scores"][0]=np.nextafter(values["scores"][0],np.float32(np.inf))
            np.savez_compressed(p,**values)
            with self.assertRaisesRegex(RuntimeError,"score bitwise replay mismatch"):
                runner.verify_classifier_replay(out,"cpu")

    def test_source_and_reference_bindings_fail_before_array_decode(self):
        with tempfile.TemporaryDirectory(prefix="acf005_bind_",dir=TMP) as temp:
            root=Path(temp)
            cfg=json.loads((Path(__file__).resolve().parents[1]/"noise_acf005_config.json").read_text(encoding="utf-8"))
            for name in runner.REQUIRED_SOURCE:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("# artificial source fixture")
            for name,spec in cfg["references"].items():
                p=root/(name+".fixture");p.write_text("bound artificial reference")
                spec.update(path=p.name,sha256=runner.sha256(p))
            runner.write_json(root/"config.json",cfg)
            (root/"protocol.md").write_text("artificial frozen protocol")
            (root/"prereg.json").write_text("{}")
            manifest=dict(config_path="config.json",config_sha256=runner.sha256(root/"config.json"),
                protocol_path="protocol.md",protocol_sha256=runner.sha256(root/"protocol.md"),
                preregistration_path="prereg.json",preregistration_sha256=runner.sha256(root/"prereg.json"),
                files={name:runner.sha256(root/name) for name in runner.REQUIRED_SOURCE})
            runner.write_json(root/"manifest.json",manifest)
            with mock.patch.object(runner.np,"load",side_effect=AssertionError("No numeric arrays allowed")):
                runner.bind(root/"config.json",root/"manifest.json",root=root,
                    expected_config_sha=manifest["config_sha256"],expected_protocol_sha=manifest["protocol_sha256"])
                (root/cfg["references"]["final004"]["path"]).write_text("tampered reference")
                with self.assertRaisesRegex(ValueError,"Reference SHA mismatch"):
                    runner.bind(root/"config.json",root/"manifest.json",root=root,
                        expected_config_sha=manifest["config_sha256"],expected_protocol_sha=manifest["protocol_sha256"])

    def test_initializer_tamper_and_receipt_failed_check_cannot_pass(self):
        with tempfile.TemporaryDirectory(prefix="acf005_role_",dir=TMP) as temp:
            root=Path(temp);real,_=synthetic();cfg=config();references(root,cfg,real)
            p=root/cfg["references"]["initial004"]["path"]
            state=torch.load(p,weights_only=False);state["model"]["mean_units"]+=.01
            state["model_state_sha256"]=runner.common.state_digest(state["model"])
            runner.common.atomic_save(state,p)
            with self.assertRaisesRegex(ValueError,"rebuilt initializer"):
                runner.reference_models(real,cfg,root,"cpu")
        r=dict(implementation_checks=dict(checks={"module":True},pending_runner_binding=True,runner_assertions_required=["actual_assertion"]))
        merged=runner.merge_runner_checks(r,{"actual_assertion":True})
        self.assertFalse(merged["implementation_checks"]["pending_runner_binding"])
        self.assertEqual(merged["implementation_checks"]["status"],"passed")
        with self.assertRaises(RuntimeError):
            runner.merge_runner_checks(dict(implementation_checks=dict(checks={"module":False})),{"runner":True})
        with self.assertRaises(RuntimeError):
            runner.merge_runner_checks(dict(implementation_checks=dict(checks={"module":True})),{"runner":False})
        with self.assertRaisesRegex(RuntimeError,"Required runner assertions"):
            runner.merge_runner_checks(dict(implementation_checks=dict(checks={"module":True},
                runner_assertions_required=["exact_classifier_replay"])),{"explicit1024_replay":True})


if __name__=="__main__":unittest.main()