"""Artificial-only tests. All temporary data stay E:/GAN/runtime_tmp."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("audit_noise_origin",ROOT/"scripts/audit_noise_origin.py")
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
REFERENCE=ROOT/"provenance/noise_origin_003_reference/interface_core.py"
REF_SHA="b1e902a2627dd0017e6a716b681e9b99a01952a9d9139d17e22e76d55f9df8a2"


def synthetic_block(seed):
    rng=np.random.default_rng(seed)
    block=rng.normal(0,.0004,500)
    block[125:145]=np.linspace(.01,.2,20)
    block[145:]=.2*np.exp(-np.arange(355)/45)
    block[111]=.15  # an isolated earlier threshold crossing must not define main-peakCFD
    return block


def fixture(directory,reference,mismatch=False):
    directory=Path(directory)
    block=np.stack([synthetic_block(1),np.zeros(500),synthetic_block(3)])
    raw=np.repeat(block,20,axis=1)
    raw[1]=np.nan  # unselected raw row: must never be decoded or validated numerically
    raw_file=directory/"synthetic_raw.npz"
    np.savez_compressed(raw_file,waveforms=raw,time_ns=np.arange(10000,dtype=np.float64)*.05)
    ids=np.array([2,0],dtype=np.int64)  # preserve target order despite sorted raw streaming
    target=np.stack([audit.replay_block(block[i],reference)["window"] for i in ids])
    if mismatch:target[1,0]=np.nextafter(target[1,0],np.float32(np.inf))
    container=directory/"TRAIN_PRIVATE.npz"
    np.savez_compressed(container,target=target,target_val=np.full((1,480),np.nan))
    identities=directory/"PARENT_IDENTITIES.npz"
    np.savez_compressed(identities,real_train_original_index=ids,real_train_group=np.array([0,0]),
                        real_val_original_index=np.array([1]),real_val_group=np.array([999]))
    members=[]
    with zipfile.ZipFile(raw_file) as z:
        for info in z.infolist():
            shape=(3,10000) if info.filename=="waveforms.npy" else (10000,)
            members.append(dict(name=info.filename,compressed_bytes=info.compress_size,
                uncompressed_bytes=info.file_size,crc32=hex(info.CRC),dtype="<f8",
                shape=list(shape),row_bytes=80000))
    size=raw_file.stat().st_size
    config=dict(experiment_id="noise_origin_audit_003_20261012",registered_UTC="synthetic_fixture_only",
        training_container=dict(path=str(container),sha256=audit.sha256(container)),
        identities=dict(path=str(identities),sha256=audit.sha256(identities)),
        reference_interface=dict(path=str(REFERENCE),sha256=REF_SHA),
        raw_source=dict(container=dict(path=str(raw_file),bytes=size,sha256=audit.sha256(raw_file)),
            members=members,memory_limit_bytes=2**33,
            io_caps=dict(hash=size,metadata=1048576,
                compressed=sum(m["compressed_bytes"] for m in members),
                inflate=sum(m["uncompressed_bytes"]+1 for m in members),
                source_total=size+1048576+sum(m["compressed_bytes"] for m in members))),
        origin_audit=dict(expected_train_rows=2,expected_train_groups=1,
            baseline_samples=100,raw_baseline_samples=2000,fixed_window_starts=[0,5,10,15,20],
            bootstrap_resamples=8,bootstrap_seed=2026101259,save_block_means=True),
        resource=dict(rss_GiB=8,wall_limit_s=60))
    config_path=directory/"config.json"
    audit.write_json(config_path,config)
    return config_path,block,ids


class NoiseOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=audit.load_reference(REFERENCE,REF_SHA)

    def test_historical_conversion_and_integer_main_peak_branch_match(self):
        block=synthetic_block(5)
        result=audit.replay_block(block,self.reference)
        a=dict(block_mean_V_f64=np.full((1,500),np.nan),
            window_normalized_f32=np.full((1,480),np.nan,dtype=np.float32),
            window_V_f32=np.full((1,480),np.nan,dtype=np.float32),
            normalization_f32=np.zeros((1,2),dtype=np.float32),
            geometry_i32=np.zeros((1,9),dtype=np.int32),
            signed_metrics_f64=np.zeros((1,8)),QC_reason_u16=np.zeros(1,dtype=np.uint16))
        self.reference.convert_row(0,np.repeat(block,20).astype("<f8").tobytes(),a,None,{})
        self.assertEqual(result["window"].tobytes(),a["window_V_f32"][0].tobytes())
        self.assertEqual(result["geometry"]["tp_post"],int(a["geometry_i32"][0,1]))
        self.assertGreater(result["geometry"]["tp_post"],111)

    def test_raw_block_centered_slope_endpoint_scaling_is_explicit(self):
        raw=.0002+3e-6*np.arange(10000)*.05
        block=raw.reshape(500,20).mean(1)
        fraw,_=audit.baseline_features(raw[:2000],.05)
        fblock,_=audit.baseline_features(block[:100],1.)
        np.testing.assert_allclose(fraw[:,0],fblock[:,0],atol=1e-16,rtol=0)
        np.testing.assert_allclose(fraw[:,2],fblock[:,2],atol=1e-16,rtol=0)
        self.assertAlmostEqual(fraw[0,1]/fblock[0,1],99.95/99,places=12)
        accumulator=audit.RawStageAccumulator(1);accumulator.add(0,raw[:2000])
        self.assertEqual(len(accumulator.profile()["mean_V"]),2000)
        np.testing.assert_allclose(accumulator.profile()["mean_V"],raw[:2000])

    def test_float64_paired_slices_and_invalid_starts_never_pad(self):
        wave=np.tile(np.arange(500)*1e-9+.0010000001234567,(4,1))
        sliced,valid=audit.slice_baselines(wave,np.array([-1,5,20,21]))
        np.testing.assert_array_equal(valid,[False,True,True,False])
        self.assertTrue(np.isnan(sliced[[0,3]]).all())
        np.testing.assert_array_equal(sliced[1],wave[1,5:105])
        self.assertFalse(np.array_equal(sliced[1],sliced[1].astype(np.float32).astype(np.float64)))
        features=np.full((4,3),np.nan);features[valid],_=audit.baseline_features(sliced[valid],1.)
        summary=audit.covariance_summary(features,np.array([0,0,1,1]),4)
        self.assertEqual((summary["n_total"],summary["n_valid"],summary["n_invalid"]),(4,2,2))

    def test_pre_post_CFDSensitivity_is_recorded_without_precision_cast(self):
        found=None
        for epsilon in (1e-7,1e-6,5e-6,1e-5):
            b=synthetic_block(21);base=b[:100].mean();threshold=(.2+base)/2
            b[111]=0;b[125:129]=threshold-.01
            b[129]=threshold-epsilon;b[130]=threshold+epsilon
            b[131:145]=np.linspace(threshold+.01,.2,14)
            result=audit.replay_block(b,self.reference)
            if result["geometry"]["left_pre"]!=result["geometry"]["left_post"]:
                found=(b,result);break
        self.assertIsNotNone(found)
        b,result=found;g=result["geometry"]
        self.assertEqual(g["shift_post_minus_pre"],g["left_post"]-g["left_pre"])
        pre,_=audit.slice_baselines(b[None],np.array([g["left_pre"]]))
        post,_=audit.slice_baselines(b[None],np.array([g["left_post"]]))
        q,_=audit.slice_baselines(result["q64"][None],np.array([g["left_post"]]))
        self.assertFalse(np.array_equal(pre,post))
        self.assertFalse(np.array_equal(post,q))

    def test_paired_covariance_of_difference_is_not_covariance_change(self):
        a=np.array([[1.,2.,3.],[2.,4.,6.],[4.,8.,12.]])
        b=2*a
        pair=audit.paired_delta(a,b,np.array([0,0,1]))
        covariance=np.cov(a[:,[0,2]],rowvar=False)
        np.testing.assert_allclose(pair["covariance_of_paired_feature_difference_mean_physical_slope"],covariance)
        np.testing.assert_allclose(pair["covariance_change_second_minus_first_same_valid_rows"],3*covariance)

    def test_stream_only_selected_rows_complete_bitwise_and_no_VAL_member_access(self):
        with tempfile.TemporaryDirectory(prefix="n01_origin_test_",dir=r"E:\GAN\runtime_tmp") as temp:
            cfg,block,ids=fixture(temp,self.reference)
            out=Path(temp)/"output";access=[]
            real_load=np.load
            class Spy:
                def __init__(self,value):self.value=value
                def __enter__(self):return self
                def __exit__(self,*args):self.value.close()
                def __getitem__(self,key):access.append(key);return self.value[key]
            def traced(*args,**kwargs):return Spy(real_load(*args,**kwargs))
            with mock.patch.object(audit.np,"load",side_effect=traced):
                report=audit.audit(cfg,out)
            self.assertEqual(access,["target","real_train_original_index","real_train_group"])
            self.assertEqual(report["raw_numeric_decoded_rows"],2)
            self.assertEqual(report["same_target_bitwise_passes"],2)
            self.assertEqual(report["source_io"]["hash_passes"],1)
            self.assertTrue(all(m["unselected_numeric_decodes"]==0 for m in report["source_io"]["members"]))
            with np.load(out/"BLOCK_MEANS.npz") as z:
                np.testing.assert_allclose(z["block_mean_V_f64"],block[ids],rtol=0,atol=1e-16)
                np.testing.assert_array_equal(z["original_index"],ids)
            complete=json.loads((out/"COMPLETE.json").read_text())
            self.assertTrue(complete["bitwise_target_equal"])
            self.assertEqual(complete["report_sha256"],audit.sha256(out/"ORIGIN_REPORT.json"))
            receipt=json.loads((out/"AUDIT_COMPLETE.json").read_text())
            for name,expected in receipt["output_hashes"].items():
                self.assertEqual(expected,audit.sha256(out/name))
            self.assertEqual(len(report["stages"]["raw_fixed_0"]["profile"]["mean_V"]),2000)
            self.assertFalse(report["selection_effect_identified"])
            with self.assertRaises(FileExistsError):audit.audit(cfg,out)

    def test_bitwise_failure_stops_on_first_selected_row_and_keeps_failure_receipt(self):
        with tempfile.TemporaryDirectory(prefix="n01_origin_fail_",dir=r"E:\GAN\runtime_tmp") as temp:
            cfg,_,_=fixture(temp,self.reference,mismatch=True)
            out=Path(temp)/"failure"
            with self.assertRaisesRegex(ValueError,"bitwise_mismatch"):audit.audit(cfg,out)
            receipt=json.loads((out/"FAILURE_RECEIPT.json").read_text())
            self.assertEqual(receipt["selected_rows_numeric_decoded"],1)
            self.assertEqual(receipt["current_original_index"],0)
            self.assertFalse(receipt["accepted_complete"])
            self.assertFalse((out/"COMPLETE.json").exists())
            self.assertFalse((out/"BLOCK_MEANS.npz").exists())
            with self.assertRaises(FileExistsError):audit.audit(cfg,out)


if __name__=="__main__":unittest.main()