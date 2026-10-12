"""TRAIN-only, paired raw-archive-to-N01 noise-origin audit.

No learning, no VAL decoding, no claim that early windows are pure electronics.
Only explicitly selected raw waveform rows are numerically decoded.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import time
import traceback

for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_key]="2"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import numpy as np

ADC_LEVELS=16383
FIXED_STARTS=(0,5,10,15,20)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):
            h.update(b)
    return h.hexdigest()


def _clean(x):
    if isinstance(x,dict):return {str(k):_clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_clean(v) for v in x]
    if isinstance(x,np.ndarray):return _clean(x.tolist())
    if isinstance(x,np.generic):return _clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def write_json(path,value):
    Path(path).write_text(json.dumps(_clean(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def load_reference(path,expected_sha):
    if sha256(path)!=expected_sha:raise ValueError("Frozen reference interface hash mismatch")
    spec=importlib.util.spec_from_file_location("n01_noise_origin_frozen_interface",str(path))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.np=np
    return module


def replay_block(block,reference):
    """Exact historical ADC/cast/CFD/window, plus paired float64 preADC CFD."""
    block=np.asarray(block,dtype=np.float64)
    if block.shape!=(500,) or not np.isfinite(block).all():raise ValueError("Finite500 block means required")
    q64=reference.digitize(block)
    q32=np.ascontiguousarray(q64,dtype=np.float32)
    normalized=reference._normalize_waveform(q32)
    tp_post=reference.constant_fraction_tp0(normalized,.5)
    pre_base=block[:100].mean()
    pre_peak=(block-pre_base).max()
    if pre_peak<=0:raise ValueError("TRAIN preADC positive peak required; no event selection")
    tp_pre=reference.constant_fraction_tp0((block-pre_base)/pre_peak,.5)
    base32=float(np.mean(q32[:100]))
    peak32=float(np.max(q32-base32))
    if peak32<=0:raise ValueError("TRAIN q32 positive peak required; no event selection")
    window=reference._window_waveform(q32,tp_post,480,120)
    left_post,left_pre=int(tp_post)-120,int(tp_pre)-120
    codes=np.round((block+.5)*ADC_LEVELS)
    geometry=dict(tp_post=tp_post,tp_pre=tp_pre,left_post=left_post,left_pre=left_pre,
                  shift_post_minus_pre=left_post-left_pre,
                  pad_left=max(0,-left_post),pad_right=max(0,left_post+480-500),
                  pre_valid_full480=0<=left_pre<=20,post_valid_full480=0<=left_post<=20,
                  adc_low_count=int((codes<0).sum()),adc_high_count=int((codes>ADC_LEVELS).sum()),
                  peak_index_q32=int(np.argmax(q32)),normalization_base32=base32,
                  peak_q32_V=peak32,
                  Q_Vns=float(window[100:400].astype(np.float64).sum()-3*window[:100].astype(np.float64).sum()))
    return dict(q64=q64,q32=q32,window=window,geometry=geometry)


def slice_baselines(x,starts):
    """Direct float64 slices. Illegal full480 starts become NaN, never padded."""
    x=np.asarray(x,dtype=np.float64)
    starts=np.asarray(starts,dtype=np.int64)
    if x.ndim!=2 or x.shape[1]!=500 or len(x)!=len(starts):raise ValueError("Nx500/start length mismatch")
    valid=(starts>=0)&(starts<=20)
    output=np.full((len(x),100),np.nan)
    if valid.any():
        output[valid]=x[valid][np.arange(valid.sum())[:,None],starts[valid,None]+np.arange(100)]
    return output,valid


def baseline_features(x,dt_ns):
    """Mean, slope on centered [-1,1], physical slope V/ns; ACF at1..20ns."""
    x=np.asarray(x,dtype=np.float64)
    if x.ndim==1:x=x[None]
    length=x.shape[1]
    t=np.linspace(-1,1,length)
    mean=x.mean(1)
    centered=x-mean[:,None]
    slope=centered@t/(t@t)
    physical=slope*2/((length-1)*dt_ns)
    variance=(centered*centered).mean(1)
    acf=[]
    for lag_ns in range(1,21):
        lag=round(lag_ns/dt_ns)
        numerator=(centered[:,:-lag]*centered[:,lag:]).mean(1)
        acf.append(np.divide(numerator,variance,out=np.full(len(x),np.nan),where=variance>0))
    return np.column_stack((mean,slope,physical)),np.array(acf).T


def _covariance(x):
    if len(x)<2:return np.full((x.shape[1],x.shape[1]),np.nan)
    return np.atleast_2d(np.cov(x,rowvar=False,ddof=1))


def _corr(x,y):
    x,y=np.asarray(x),np.asarray(y)
    good=np.isfinite(x)&np.isfinite(y)
    if good.sum()<2:return None
    a,b=x[good]-x[good].mean(),y[good]-y[good].mean()
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator else None


def covariance_summary(features,groups,total_rows):
    features=np.asarray(features)
    good=np.isfinite(features).all(1)
    x,g=features[good],np.asarray(groups)[good]
    cov=_covariance(x[:,:2])
    within=between=np.full((2,2),np.nan)
    if len(x)>1:
        unique,ix=np.unique(g,return_inverse=True)
        count=np.bincount(ix)
        sums=np.zeros((len(unique),2));np.add.at(sums,ix,x[:,:2])
        means=sums/count[:,None]
        wc=x[:,:2]-means[ix]
        bc=means-x[:,:2].mean(0)
        within=wc.T@wc/(len(x)-1)
        between=bc.T@(count[:,None]*bc)/(len(x)-1)
    return dict(n_total=total_rows,n_valid=int(good.sum()),n_invalid=int((~good).sum()),
                valid_original_groups=len(np.unique(g)),
                covariance_mean_normalized_slope_V2=cov,
                covariance_mean_physical_slope=_covariance(x[:,[0,2]]),
                mean_slope_correlation=_corr(x[:,0],x[:,1]),
                means=x.mean(0) if len(x) else np.full(3,np.nan),
                SDs=x.std(0,ddof=1) if len(x)>1 else np.full(3,np.nan),
                within_covariance_contribution_V2=within,
                between_covariance_contribution_V2=between,
                contribution_denominator="both_Nvalid_minus1;sum_equals_valid_subset_covariance")


def paired_delta(a,b,groups):
    """Same-event B-A, using physical slope so raw/block span differences are explicit."""
    valid=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    difference=np.full_like(a,np.nan)
    difference[valid]=b[valid]-a[valid]
    return dict(n_total=len(a),n_valid=int(valid.sum()),n_invalid=int((~valid).sum()),
                direction="second_minus_first",
                mean_delta_mean_V=float(difference[valid,0].mean()) if valid.any() else None,
                mean_delta_physical_slope_V_per_ns=float(difference[valid,2].mean()) if valid.any() else None,
                covariance_of_paired_feature_difference_mean_physical_slope=_covariance(difference[valid][:,[0,2]]),
                covariance_change_second_minus_first_same_valid_rows=(
                    _covariance(b[valid][:,[0,2]])-_covariance(a[valid][:,[0,2]])),
                covariance_change_is_not_covariance_of_difference=True,
                difference_summary=covariance_summary(difference,groups,len(a)),
                no_padding_or_silent_full_population_drop=True)


def bootstrap_stages(features,groups,resamples,seed):
    """One shared group resampling plan across all paired stages."""
    unique,inverse=np.unique(groups,return_inverse=True)
    rng=np.random.default_rng(seed)
    draws=np.array([np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))
                    for _ in range(resamples)])
    result={}
    for name,x in features.items():
        good=np.isfinite(x).all(1)
        index=inverse[good];x=x[good,:2]
        count=np.bincount(index,minlength=len(unique))
        sums=np.zeros((len(unique),2));second=np.zeros((len(unique),2,2))
        np.add.at(sums,index,x);np.add.at(second,index,x[:,:,None]*x[:,None,:])
        rho=[];cross=[]
        for draw in draws:
            n=draw@count
            if n<2:continue
            s=draw@sums
            c=(np.einsum("g,gij->ij",draw,second)-np.outer(s,s)/n)/(n-1)
            cross.append(c[0,1])
            if c[0,0]>0 and c[1,1]>0:rho.append(c[0,1]/np.sqrt(c[0,0]*c[1,1]))
        result[name]=dict(n_valid_rows=int(good.sum()),valid_resamples=len(cross),
            correlation_valid_resamples=len(rho),
            correlation_percentile95=np.quantile(rho,[.025,.975]) if rho else [None,None],
            cross_covariance_percentile95_V2=np.quantile(cross,[.025,.975]) if cross else [None,None])
    return dict(seed=seed,resamples=resamples,original_groups=len(unique),
                method="original_group_clusters_shared_draws_across_stages",
                scope="this_selected_TRAIN_pool_and_archive;not_independent_acquisitions",
                stages=result)


class RawStageAccumulator:
    """Retain event features +2000 profile moments, never long raw rows."""
    def __init__(self,rows):
        self.features=np.full((rows,3),np.nan)
        self.acf=np.full((rows,20),np.nan)
        self.sum=np.zeros(2000);self.square=np.zeros(2000);self.count=0

    def add(self,index,segment):
        f,a=baseline_features(segment,.05)
        self.features[index]=f[0];self.acf[index]=a[0]
        samples=np.asarray(segment)
        if len(samples)!=2000:raise ValueError("Raw profile must have2000 points")
        self.sum+=samples;self.square+=samples**2;self.count+=1

    def profile(self):
        mean=self.sum/self.count
        variance=np.maximum((self.square-self.count*mean**2)/max(1,self.count-1),0)
        return dict(mean_V=mean,SD_V=np.sqrt(variance),
                    sample_indices_within_raw_segment=np.arange(2000),
                    sample_interval_ns=.05,profile_is_full2000_but_not_covariance_matrix=True)


def _resolve(config_path,path):
    path=Path(path)
    return path if path.is_absolute() else Path(config_path).resolve().parent/path


def audit(config_path,out,container=None,identities=None,raw_path=None,reference_interface_path=None):
    config_path=Path(config_path).resolve()
    cfg=json.loads(config_path.read_text(encoding="utf-8"))
    out=Path(out).resolve()
    if os.name=="nt" and out.drive.upper() not in ("E:","D:"):raise ValueError("Outputs must stay E/D")
    if out.exists() and any(out.iterdir()):raise FileExistsError("Preserve all existing audit files;choose a fresh output directory")
    out.mkdir(parents=True,exist_ok=True)
    began=time.monotonic()
    cost=dict(reserved=dict.fromkeys(("hash","metadata","compressed","inflate"),0),
              actual_returned=dict.fromkeys(("hash","metadata","compressed","inflate"),0),
              seeks=0,members=[],hash_passes=0)
    decoded=0;bitwise_count=0;current_id=None
    base=dict(experiment_id=cfg["experiment_id"],config_sha256=sha256(config_path),
              new_training_updates=0,validation_member_numeric_decodes=0,
              raw_long_waveforms_saved=False,unselected_raw_numeric_decodes=0,
              raw_npz_is_not_proven_native_instrument_data=True)
    try:
        if cfg["experiment_id"]!="noise_origin_audit_003_20261012" or not cfg.get("registered_UTC"):
            raise ValueError("Require registered noise-origin003 config")
        options=cfg["origin_audit"]
        if options["baseline_samples"]!=100 or options["raw_baseline_samples"]!=2000:
            raise ValueError("Registered windows must be100 block/2000 raw samples")
        if tuple(options["fixed_window_starts"])!=FIXED_STARTS:
            raise ValueError("Fixed window starts mismatch")
        container=Path(container) if container else _resolve(config_path,cfg["training_container"]["path"])
        identities=Path(identities) if identities else _resolve(config_path,cfg["identities"]["path"])
        reference_path=Path(reference_interface_path) if reference_interface_path else _resolve(config_path,cfg["reference_interface"]["path"])
        if sha256(container)!=cfg["training_container"]["sha256"]:raise ValueError("TRAIN container SHA mismatch")
        if sha256(identities)!=cfg["identities"]["sha256"]:raise ValueError("Identity SHA mismatch")
        reference=load_reference(reference_path,cfg["reference_interface"]["sha256"])
        with np.load(container,allow_pickle=False) as z:target=z["target"].copy()
        with np.load(identities,allow_pickle=False) as z:
            ids=z["real_train_original_index"].astype(np.int64)
            groups=z["real_train_group"].copy()
        n=len(ids)
        if n!=options["expected_train_rows"] or target.shape!=(n,480) or target.dtype!=np.float32:
            raise ValueError("Original target TRAIN shape/dtype/count mismatch")
        if len(groups)!=n or len(np.unique(groups))!=options["expected_train_groups"] or len(np.unique(ids))!=n:
            raise ValueError("Complete TRAIN identity/group mismatch")
        plan=copy.deepcopy(cfg["raw_source"])
        if raw_path:plan["container"]["path"]=str(Path(raw_path))
        plan["memory_limit_bytes"]=min(plan["memory_limit_bytes"],int(cfg["resource"]["rss_GiB"]*2**30))
        waveform_spec=next(m for m in plan["members"] if m["name"]=="waveforms.npy")
        if waveform_spec["shape"][1]!=10000 or waveform_spec["dtype"]!="<f8":
            raise ValueError("Bound raw waveform schema mismatch")
        if (ids<0).any() or (ids>=waveform_spec["shape"][0]).any():raise ValueError("TRAIN ID outside raw archive")
        block=np.full((n,500),np.nan,dtype=np.float64)
        raw_names=["raw_fixed_"+str(start) for start in FIXED_STARTS]+["raw_aligned_post","raw_aligned_pre_valid"]
        raw_stages={name:RawStageAccumulator(n) for name in raw_names}
        positions={int(identity):i for i,identity in enumerate(ids)}
        geometry={}
        time_grid={}
        until=began+cfg["resource"]["wall_limit_s"]

        def on_time(index,data):
            t=np.frombuffer(data,dtype="<f8")
            if index!=0 or t.shape!=(10000,) or not np.isfinite(t).all() or not np.all(abs(np.diff(t)-.05)<=1e-9):
                raise ValueError("Bound .05ns raw time grid mismatch")
            differences=np.diff(t)
            time_grid.update(start_ns=float(t[0]),end_ns=float(t[-1]),samples=len(t),
                software_grid_dt_mean_ns=float(differences.mean()),software_grid_dt_min_ns=float(differences.min()),
                software_grid_dt_max_ns=float(differences.max()),native_hardware_timing_proven=False)

        def on_wave(identity,data):
            nonlocal decoded,bitwise_count,current_id
            current_id=int(identity);decoded+=1
            position=positions[current_id]
            row=np.frombuffer(data,dtype="<f8")
            if row.shape!=(10000,) or not np.isfinite(row).all():raise ValueError("Selected TRAIN raw row invalid;no row deletion")
            down=row.reshape(500,20).mean(axis=1)
            replay=replay_block(down,reference)
            info=replay["geometry"]
            if replay["window"].tobytes()!=np.ascontiguousarray(target[position]).tobytes():
                error=ValueError("same_target_bitwise_mismatch_original_index="+str(current_id))
                error.max_absolute_difference=float(np.max(abs(replay["window"].astype(np.float64)-target[position])))
                raise error
            bitwise_count+=1
            block[position]=down
            if not geometry:
                geometry.update({key:np.zeros(n,dtype=np.float64 if isinstance(value,float) else np.int64)
                                 for key,value in info.items()})
            for key,value in info.items():geometry[key][position]=value
            for start in FIXED_STARTS:
                raw_stages["raw_fixed_"+str(start)].add(position,row[start*20:start*20+2000])
            for branch in ("post","pre"):
                left=info["left_"+branch]
                if 0<=left<=20:
                    raw_stages["raw_aligned_post" if branch=="post" else "raw_aligned_pre_valid"].add(position,row[left*20:left*20+2000])
            if decoded%1000==0:
                progress=dict(status="streaming_selected_TRAIN",selected_rows_decoded=decoded,
                    same_target_bitwise_passes=bitwise_count,total_selected_rows=n,
                    current_original_index=current_id,elapsed_seconds=time.monotonic()-began)
                write_json(out/"status.json",progress)
                print(json.dumps(progress),flush=True)

        reference.read_raw_source(plan,ids.tolist(),cost,until,on_time,on_wave)
        if decoded!=n or bitwise_count!=n or not np.isfinite(block).all():
            raise ValueError("Incomplete selected row/replay delivery")
        q64=reference.digitize(block)
        q32=q64.astype(np.float32)
        stages={};feature_arrays={};acf_arrays={};covariances={}

        def add_raw(name,accumulator):
            summary=covariance_summary(accumulator.features,groups,n)
            summary.update(sample_interval_ns=.05,window_samples=2000,
                           endpoint_span_ns=99.95,support_duration_ns=100.,
                           ACF_lags_ns=np.arange(1,21),
                           ACF_mean=np.nanmean(accumulator.acf,axis=0) if accumulator.count else [None]*20,
                           profile=accumulator.profile() if accumulator.count else None,
                           full_raw2000_covariance_saved=False)
            stages[name]=summary;feature_arrays[name]=accumulator.features;acf_arrays[name]=accumulator.acf
        for name,accumulator in raw_stages.items():add_raw(name,accumulator)

        def add_stage(name,wave):
            wave=np.asarray(wave,dtype=np.float64)
            valid=np.isfinite(wave).all(1)
            features=np.full((n,3),np.nan);acfs=np.full((n,20),np.nan)
            if valid.any():features[valid],acfs[valid]=baseline_features(wave[valid],1.)
            summary=covariance_summary(features,groups,n)
            summary.update(sample_interval_ns=1.,window_samples=100,
                           endpoint_span_ns=99.,support_duration_ns=100.,
                           ACF_lags_ns=np.arange(1,21),
                           ACF_mean=np.nanmean(acfs,axis=0) if valid.any() else [None]*20,
                           ACF_undefined_rows=int((~np.isfinite(acfs).all(1)).sum()),
                           profile=dict(mean_V=wave[valid].mean(0),SD_V=wave[valid].std(0,ddof=1)) if valid.sum()>1 else None,
                           full_covariance_npz_key=name)
            stages[name]=summary;feature_arrays[name]=features;acf_arrays[name]=acfs
            covariances[name]=_covariance(wave[valid])
        for start in FIXED_STARTS:
            add_stage("block_fixed_"+str(start),block[:,start:start+100])
            add_stage("q32_fixed_"+str(start),q32[:,start:start+100])
        add_stage("q64_fixed_0",q64[:,:100])
        aligned_block_post,post_valid=slice_baselines(block,geometry["left_post"])
        aligned_q64_post,_=slice_baselines(q64,geometry["left_post"])
        aligned_block_pre,pre_valid=slice_baselines(block,geometry["left_pre"])
        add_stage("block_aligned_post",aligned_block_post)
        add_stage("q64_aligned_post",aligned_q64_post)
        add_stage("block_aligned_pre_valid",aligned_block_pre)
        add_stage("postCFD_q32",target[:,:100].astype(np.float64))
        # N01 canonical_real repeats ADC lattice rounding without clipping; validate its legal interval.
        if not np.isfinite(target).all() or ((target<-.5)|(target>.5)).any():raise ValueError("N01 target outside recorded ADC range")
        n01=np.rint((target[:,:100].astype(np.float64)+.5)*ADC_LEVELS)/ADC_LEVELS-.5
        add_stage("N01_readc",n01)
        pairs=[
            ("raw_fixed_0","block_fixed_0"),("block_fixed_0","q64_fixed_0"),
            ("q64_fixed_0","q32_fixed_0"),("q32_fixed_0","postCFD_q32"),
            ("postCFD_q32","N01_readc"),
            ("block_aligned_post","q64_aligned_post"),
            ("q64_aligned_post","postCFD_q32"),
            ("block_aligned_pre_valid","block_aligned_post"),
            ("raw_aligned_post","block_aligned_post")]
        paired={a+"__to__"+b:paired_delta(feature_arrays[a],feature_arrays[b],groups) for a,b in pairs}
        associations={name:{predictor:dict(mean_V_correlation=_corr(values[:,0],geometry[predictor]),
                                           slope_V_per_ns_correlation=_corr(values[:,2],geometry[predictor]))
                            for predictor in ("left_post","left_pre","peak_q32_V","Q_Vns")}
                      for name,values in feature_arrays.items()}
        left_values,left_counts=np.unique(geometry["left_post"],return_counts=True)
        delta_values,delta_counts=np.unique(geometry["shift_post_minus_pre"],return_counts=True)
        report=dict(**base,status="complete",registered_UTC=cfg["registered_UTC"],
            target_train_rows=n,original_train_groups=len(np.unique(groups)),
            raw_numeric_decoded_rows=decoded,same_target_bitwise_passes=bitwise_count,
            raw_values_numeric_decoded=decoded*10000,time_grid_values_decoded=10000,
            source_io=cost,time_grid=time_grid,raw_archive_provenance="earliest_traceable_archive_not_proven_native_instrument_capture",
            stages=stages,primary_chain=["raw_fixed_0","block_fixed_0","q64_fixed_0","q32_fixed_0","postCFD_q32","N01_readc"],
            geometry=dict(actual_left_histogram=dict(zip(left_values.tolist(),left_counts.tolist())),
                          post_padding_rows=int(((geometry["pad_left"]>0)|(geometry["pad_right"]>0)).sum()),
                          post_invalid_full480_rows=int((~post_valid).sum()),
                          pre_invalid_full480_rows=int((~pre_valid).sum()),
                          CFD_shift_post_minus_pre_histogram=dict(zip(delta_values.tolist(),delta_counts.tolist())),
                          CFD_is_integer_main_peak_connected_branch=True,
                          preADC_CFD_normalization="float64_first100_baseline_and_positive_peak",
                          actual_q32_CFD_normalization="reference_float32_first100_baseline_and_peak",
                          no_invalid_pre_branch_padding_or_silent_global_drop=True),
            paired_comparisons=paired,associations_descriptive_not_causal=associations,
            bootstrap=bootstrap_stages(feature_arrays,groups,options["bootstrap_resamples"],options["bootstrap_seed"]),
            fixed_raw_support="raw[start*20:start*20+2000] matches block[start:start+100]",
            centered_slope_units=dict(normalized="V_on_window_endpoints_minus1_plus1",
                                      physical="V_per_ns;2*normalized_slope/endpoint_span_ns",
                                      raw_endpoint_span_ns=99.95,block_endpoint_span_ns=99.,
                                      raw_profile_has_full2000_positions=True),
            selection_effect_identified=False,
            event_features_columns=["mean_V","centered_slope_normalized_V","centered_slope_V_per_ns"],
            interpretation_limits=["selected_legal_TRAIN_pool_conditions_all_stages",
                                   "early_or_preCFD_positions_are_not_proven_pure_electronics",
                                   "CFD_peak_charge_association_does_not_establish_cause",
                                   "raw_archive_native_creator_unverified",
                                   "no_new_VAL_or_independent_acquisition_validation"],
            replay_precision=dict(block="float64_mean20",q64="reference_ADC_float64",q32="cast_float32",
                                  paired_alignments="direct_float64_slice;reference_window_only_for_exact_q32_target_replay"),
            runtime_seconds=time.monotonic()-began,cpu_numeric_threads=2,
            effective_paths=dict(container=str(container),identities=str(identities),
                                 raw=plan["container"]["path"],reference_interface=str(reference_path)))
        if options["save_block_means"]:
            np.savez_compressed(out/"BLOCK_MEANS.npz",block_mean_V_f64=block,original_index=ids,artificial_group=groups)
        np.savez_compressed(out/"EVENT_FEATURES.npz",original_index=ids,artificial_group=groups,
                            **{"metadata_"+k:v for k,v in geometry.items()},
                            **{"features_"+k:v for k,v in feature_arrays.items()},
                            **{"ACF_"+k:v for k,v in acf_arrays.items()})
        np.savez_compressed(out/"STAGE_COVARIANCES.npz",**covariances)
        write_json(out/"ORIGIN_REPORT.json",report)
        write_json(out/"status.json",dict(status="complete",selected_rows_decoded=decoded,
            same_target_bitwise_passes=bitwise_count,total_selected_rows=n,
            elapsed_seconds=time.monotonic()-began))
        receipt=dict(**base,status="complete",same_target_bitwise_passes=bitwise_count,
                     selected_rows_decoded=decoded,source_io=cost,
                     implementation_sha256=sha256(__file__),
                     output_hashes={p.name:sha256(p) for p in out.iterdir() if p.is_file()},
                     elapsed_seconds=time.monotonic()-began)
        write_json(out/"AUDIT_COMPLETE.json",receipt)
        write_json(out/"COMPLETE.json",dict(status="complete",report_sha256=sha256(out/"ORIGIN_REPORT.json"),
            rows_completed=n,bitwise_target_equal=bitwise_count==n,
            noise_training_updates=0,response_training_updates=0,validation_waveforms_decoded=False,
            implementation_sha256=sha256(__file__)))
        return report
    except BaseException as error:
        write_json(out/"FAILURE_RECEIPT.json",dict(**base,status="failed",
            selected_rows_numeric_decoded=decoded,same_target_bitwise_passes=bitwise_count,
            current_original_index=current_id,source_io=cost,error=repr(error),
            max_absolute_difference=getattr(error,"max_absolute_difference",None),
            traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-began,
            accepted_complete=False,partial_arrays_saved=False))
        write_json(out/"status.json",dict(status="failed",selected_rows_decoded=decoded,
            same_target_bitwise_passes=bitwise_count,current_original_index=current_id,
            elapsed_seconds=time.monotonic()-began,error=repr(error)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--container")
    parser.add_argument("--identities")
    parser.add_argument("--raw-path")
    parser.add_argument("--reference-interface-path")
    args=parser.parse_args()
    result=audit(args.config,args.out,args.container,args.identities,args.raw_path,args.reference_interface_path)
    print(json.dumps(_clean(dict(status=result["status"],rows=result["target_train_rows"],
                                bitwise=result["same_target_bitwise_passes"],
                                elapsed_seconds=result["runtime_seconds"])),indent=2))


if __name__=="__main__":main()