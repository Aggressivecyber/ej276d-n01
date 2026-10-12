"""Runner engineering tests; artificial data only, no GPU/VAL scientific execution."""
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
from n01 import train_noise_v3 as runner
from n01.losses_v2 import BaselineDistributionLoss

TMP=r"E:\GAN\runtime_tmp"


def configuration():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/"noise_improvement_004_config.json").read_text(encoding="utf-8"))
    cfg["noise"]["batch_size"]=32;cfg["cpu_threads"]=2;cfg["wall_limit_s"]=60
    return cfg


def artificial_baselines():
    rng=np.random.default_rng(123456)
    t=np.linspace(-1,1,100)
    # Known correlated population offset/slope plus positive white covariance.
    offset=rng.normal(0,.00015,64)
    slope=-.65*offset+rng.normal(0,.00005,64)
    return offset[:,None]+slope[:,None]*t+rng.normal(0,.0006,(64,100)),np.repeat(np.arange(4),16)


def bindings():
    return dict(config_sha256="test_config",source_manifest_sha256="test_manifest",
                protocol_sha256="test_protocol",preregistration_sha256="test_preregister",source_files={})


class TrainNoiseV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_TRAIN_loader_accesses_only_target_and_training_identities(self):
        with tempfile.TemporaryDirectory(prefix="noise_v3_fields_",dir=TMP) as temp:
            p=Path(temp);target=np.zeros((4,480),dtype=np.float32)
            target[:,:100]=np.arange(4)[:,None]*.0001
            np.savez_compressed(p/"data.npz",target=target,target_val=np.full((599,480),np.nan),
                                source=np.full((1,480),np.nan))
            np.savez_compressed(p/"ids.npz",real_train_original_index=np.arange(4),
                                real_train_group=np.array([0,0,1,1]),real_val_group=np.array([0]))
            cfg=dict(train_container_sha256=runner.sha256(p/"data.npz"),identities_sha256=runner.sha256(p/"ids.npz"))
            access=[];load=np.load
            class Spy:
                def __init__(self,z):self.z=z
                def __enter__(self):return self
                def __exit__(self,*args):self.z.close()
                def __getitem__(self,key):access.append(key);return self.z[key]
            with mock.patch.object(runner.np,"load",side_effect=lambda *a,**k:Spy(load(*a,**k))):
                real,groups=runner.load_training_baselines(p/"data.npz",p/"ids.npz",cfg,
                                                          expected_rows=4,expected_groups=2)
            self.assertEqual(real.shape,(4,100))
            self.assertEqual(access,["target","real_train_original_index","real_train_group"])
            self.assertEqual(len(groups),4)

    def test_source_bindings_reject_tamper_before_any_numeric_decode(self):
        with tempfile.TemporaryDirectory(prefix="noise_v3_bind_",dir=TMP) as temp:
            root=Path(temp)
            for name in runner.REQUIRED_SOURCE:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# artificial source fixture\n")
            cfg=configuration();runner.write_json(root/"config.json",cfg)
            (root/"protocol.md").write_text("frozen artificial protocol")
            (root/"prereg.json").write_text("{}")
            manifest=dict(config_path="config.json",config_sha256=runner.sha256(root/"config.json"),
                          protocol_path="protocol.md",protocol_sha256=runner.sha256(root/"protocol.md"),
                          preregistration_path="prereg.json",preregistration_sha256=runner.sha256(root/"prereg.json"),
                          files={name:runner.sha256(root/name) for name in runner.REQUIRED_SOURCE})
            runner.write_json(root/"manifest.json",manifest)
            with mock.patch.object(runner.np,"load",side_effect=AssertionError("No arrays allowed")):
                runner.validate_bindings(root/"config.json",root/"manifest.json",root=root,
                                         expected_config_sha=manifest["config_sha256"])
                (root/runner.REQUIRED_SOURCE[0]).write_text("# tampered")
                with self.assertRaisesRegex(ValueError,"source hash mismatch"):
                    runner.validate_bindings(root/"config.json",root/"manifest.json",root=root,
                                             expected_config_sha=manifest["config_sha256"])

    def test_three_step_engineering_saves_initial_state_coverage_and_exact_latent_replay(self):
        with tempfile.TemporaryDirectory(prefix="noise_v3_fit_",dir=TMP) as temp:
            real,groups=artificial_baselines();cfg=configuration()
            model,state=runner.fit_noise_baselines(real,groups,cfg,Path(temp)/"noise",bindings(),
                                                  device="cpu",engineering=True)
            self.assertEqual(state["update"],3)
            self.assertTrue(torch.isfinite(torch.cat([p.flatten() for p in model.parameters()])).all())
            self.assertTrue((state["sampler"]["visits"]>0).all())
            self.assertEqual(int(state["sampler"]["visits"].sum()),96)
            initial=torch.load(Path(temp)/"noise/initial_state.pt",weights_only=False)
            self.assertNotEqual(initial["model_state_sha256"],state["model_state_sha256"])
            self.assertEqual(runner.state_digest(state["initial_model_state"]),initial["model_state_sha256"])
            with self.assertRaisesRegex(ValueError,"fixed5000"):
                runner.validate_endpoint(state,cfg,bindings())
            replay=runner.save_replay(model,cfg,Path(temp)/"replay.pt",bindings())
            self.assertEqual((replay["rows"],replay["chunk_size"],len(replay["chunks"])),(1024,256,4))
            self.assertEqual(replay["seed"],cfg["noise"]["generation_seed"]+10000)
            self.assertTrue(runner.verify_replay(model,replay))
            altered=copy.deepcopy(replay)
            altered["chunks"][0]["pre_adc"][0,0,0]+=1e-6
            with self.assertRaisesRegex(RuntimeError,"replay mismatch"):runner.verify_replay(model,altered)
            with self.assertRaises(FileExistsError):
                runner.fit_noise_baselines(real,groups,cfg,Path(temp)/"noise",bindings(),device="cpu",engineering=True)

    def test_post_optimizer_context_failure_counts_completed_step_and_preserves_invalid_context(self):
        with tempfile.TemporaryDirectory(prefix="noise_v3_fail_",dir=TMP) as temp:
            real,groups=artificial_baselines();cfg=configuration();out=Path(temp)/"noise"
            holder={};create=BaselineDistributionLoss.create_projection_context
            step=torch.optim.Adam.step
            def capture(self,*args,**kwargs):
                ctx=create(self,*args,**kwargs);holder["context"]=ctx;return ctx
            def mutate_after_step(self,*args,**kwargs):
                value=step(self,*args,**kwargs)
                holder["context"].banks["baseline_wave"].add_(1e-6)
                return value
            with mock.patch.object(BaselineDistributionLoss,"create_projection_context",capture),mock.patch.object(torch.optim.Adam,"step",mutate_after_step):
                with self.assertRaisesRegex(RuntimeError,"mutated"):
                    runner.fit_noise_baselines(real,groups,cfg,out,bindings(),device="cpu",engineering=True)
            failure=json.loads((out/"FAILURE.json").read_text())
            self.assertEqual(failure["update"],1)
            self.assertTrue(failure["optimizer_step_completed"])
            self.assertEqual(failure["update_count_status"],"known_completed_optimizer_steps")
            saved=torch.load(out/"failure_state.pt",weights_only=False)
            self.assertEqual(saved["state"]["update"],1)
            self.assertTrue(saved["state"]["projection_context"]["invalid_context"])
            self.assertIsNotNone(saved["last_noise"])
            self.assertFalse((out/"engineering.pt").exists())

    def test_budget_and_coverage_checks_do_not_accept_partial_scientific_endpoint(self):
        with tempfile.TemporaryDirectory(prefix="noise_v3_budget_",dir=TMP) as temp:
            p=Path(temp);(p/"bytes").write_bytes(b"0123456789")
            cfg=configuration();cfg["output_limit_MiB"]=1e-6
            with self.assertRaisesRegex(RuntimeError,"output limit"):
                runner.resource_check(p,cfg,runner.time.monotonic(),"cpu")
            cfg["output_limit_MiB"]=512;cfg["wall_limit_s"]=0
            with self.assertRaises(TimeoutError):
                runner.resource_check(p,cfg,runner.time.monotonic()-1,"cpu")
            real,groups=artificial_baselines();cfg=configuration()
            _,state=runner.fit_noise_baselines(real,groups,cfg,p/"fit",bindings(),device="cpu",engineering=True)
            state["engineering"]=False;state["update"]=5000
            with self.assertRaisesRegex(ValueError,"coverage draw accounting"):
                runner.validate_endpoint(state,cfg,bindings())


if __name__=="__main__":unittest.main()