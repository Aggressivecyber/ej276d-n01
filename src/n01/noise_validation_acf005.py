"""005 separated descriptive evaluation; no fitting or response authorization."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from .acf20 import acf20,VARIANCE_FLOOR
from .data import sha256,write_json
from .evaluate import _clean
from .noise_validation import baseline_statistics


CANDIDATE_KEYS=("mean_location","mean_scatter","rms_quantiles","acf","spectrum","ols_slope",
                "ols_joint","full100","classifier_point","classifier_interval")
CONTROL_KEYS=("negative_control_point","negative_control_interval_contains_half")
DIAGNOSTIC_KEYS=("initial_pre_adc","initial_post_adc","v004_pre_adc","v004_post_adc","v005_pre_adc","v005_post_adc")


def _require(condition,message):
    if not bool(condition):raise ValueError(message)


def array_digest(value):
    x=np.ascontiguousarray(value)
    h=hashlib.sha256();h.update(str(x.shape).encode());h.update(str(x.dtype).encode());h.update(x.tobytes())
    return h.hexdigest()


def _event_mean(event_acf):
    # Same float64 reduction as shared mean_acf20/fixed training target.
    return torch.as_tensor(event_acf,dtype=torch.float64).mean(0).cpu().numpy()


def _events(wave):
    if isinstance(wave,torch.Tensor):wave=wave.detach().cpu().numpy()
    x=np.asarray(wave,dtype=np.float64)
    _require(x.ndim==2 and x.shape[1]==100 and len(x)>0 and np.isfinite(x).all(),"Expected finite nonemptyNx100,no row removal")
    with torch.no_grad():a=acf20(torch.as_tensor(x)).cpu().numpy()
    _require(np.isfinite(a).all(),"Nonfinite event ACF20")
    np.testing.assert_allclose(a.mean(0),baseline_statistics(x)["acf_mean"],rtol=2e-13,atol=2e-14,
                               err_msg="Shared/legacy NumPy ACF20 disagrees")
    return x,a


def verify_references(config,root):
    verified={}
    for name,value in config["references"].items():
        path=Path(root)/value["path"]
        _require(path.is_file() and sha256(path)==value["sha256"],"Frozen reference SHA mismatch: "+name)
        verified[name]=dict(path=str(path),sha256=value["sha256"])
    return verified


def compare_control_arrays(old,new):
    checks={}
    for key in ("labels","scores","groups0","groups1"):
        a,b=np.asarray(old[key]),np.asarray(new[key])
        same=(a.shape==b.shape and a.dtype==b.dtype and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes())
        _require(same,"Exact004/005 REAL-REAL reproduction failed: "+key)
        checks[key]=True
    return checks


def candidate_screen(gate):
    _require(set(gate["checks"])==set(CANDIDATE_KEYS+CONTROL_KEYS),"Legacy gate check set changed")
    checks={key:bool(gate["checks"][key]) for key in CANDIDATE_KEYS}
    return dict(passed=all(checks.values()),checks=checks,failed_checks=[key for key,value in checks.items() if not value],
        thresholds_unchanged=True,raw_classifier=dict(AUC=gate["classifiers"]["raw"]["generated_REAL"]["AUC"],
            percentile_95_interval=gate["classifiers"]["raw"]["generated_REAL"]["percentile_95_interval"],
            interpretation="predeclared practical tolerance screen;not statistical equality/equivalence evidence"))


def train_decision(current_mean,reference004_mean,target,candidate_passed):
    current=np.asarray(current_mean)-np.asarray(target)
    previous=np.asarray(reference004_mean)-np.asarray(target)
    _require(current.shape==previous.shape==(20,) and np.isfinite(current).all() and np.isfinite(previous).all(),"Decision requires finite20-vectors")
    def objective(error):
        squared=(error/.015)**2
        return float(.5*(squared.mean()+squared.max()))
    current_max=float(abs(current).max());previous_max=float(abs(previous).max())
    new_loss,old_loss=objective(current),objective(previous)
    match=current_max<=.015
    improved=new_loss<old_loss and current_max<=previous_max
    allowed=bool(match and improved and candidate_passed)
    if not match:status="stop_TRAIN_ACF_mismatch"
    elif not improved:status="stop_no_strict_TRAIN_improvement"
    elif not candidate_passed:status="stop_development_candidate_checks_failed"
    else:status="improved100_candidate_read_only480_design_check_only"
    return dict(TRAIN_match=bool(match),TRAIN_improved=bool(improved),
        TRAIN_current_max_ACF_error=current_max,TRAIN004_max_ACF_error=previous_max,
        TRAIN_current_acf20_loss=new_loss,TRAIN004_acf20_loss=old_loss,
        comparison="same8192_diagnostic_latents_postADC_vs_allTRAIN_observed_target",
        numerical_tolerance=0.,status=status,read_only_480_design_check_allowed=allowed,
        response_training_authorized=False,scientifically_qualified=False,
        independent_acquisition_validated=False)


def bootstrap_acf_difference(generated_acf,real_acf,groups,seed,resamples=1000):
    """Whole REAL clusters weighted by all their events;generated iid events."""
    gen=np.asarray(generated_acf,dtype=np.float64);real=np.asarray(real_acf,dtype=np.float64);groups=np.asarray(groups)
    _require(gen.ndim==real.ndim==2 and gen.shape[1]==real.shape[1]==20 and len(gen)>0 and len(real)==len(groups),"Invalid bootstrap shape")
    _require(np.isfinite(gen).all() and np.isfinite(real).all(),"Nonfinite ACF bootstrap data")
    unique,membership=np.unique(groups,return_inverse=True);G=len(unique)
    _require(G>0 and resamples>=2,"Bootstrap needs nonempty groups and at least2 draws")
    counts=np.bincount(membership,minlength=G)
    sums=np.zeros((G,20),dtype=np.float64);np.add.at(sums,membership,real)
    rng=np.random.default_rng(seed)
    group_multiplicity=np.empty((resamples,G),dtype=np.int64)
    generated_means=np.empty((resamples,20));real_means=np.empty_like(generated_means)
    for i in range(resamples):
        multiplicity=np.bincount(rng.integers(0,G,G),minlength=G)
        selected=rng.integers(0,len(gen),len(gen))
        group_multiplicity[i]=multiplicity
        real_means[i]=(multiplicity@sums)/(multiplicity@counts)
        generated_means[i]=gen[selected].mean(0)
    differences=generated_means-real_means
    point=gen.mean(0)-real.mean(0)
    marginal=np.quantile(differences,[.025,.975],axis=0)
    radius=float(np.quantile(np.max(abs(differences-point),axis=1),.95))
    summary=dict(difference_point=point,marginal95=marginal,
        simultaneous95_centered_max=np.stack((point-radius,point+radius)),bound_radius95=radius,
        seed=int(seed),bootstrap_resamples=resamples,generated_rows=len(gen),REAL_rows=len(real),REAL_original_groups=G,
        REAL_resampling="G original groups with replacement;all within-group events via sums/counts",
        generated_resampling="independent_generated_events_with_replacement",
        simultaneous_method="unstudentized95percentile_max_abs(bootstrap_difference-point_difference)",
        acceptance_gate=False,training_randomness_included=False,independent_acquisition_uncertainty=False,
        scope="descriptive conditional_on_fixed_model_and_one_previously_exposed_acquisition")
    numeric=dict(difference_point=point,differences=differences,generated_bootstrap_means=generated_means,
        REAL_bootstrap_means=real_means,group_ids=unique,group_counts=counts,group_sums=sums,
        REAL_group_multiplicity=group_multiplicity)
    return summary,numeric


def _null_helpers(root):
    path=Path(root)/"scripts/calibrate_real_real_null.py"
    spec=importlib.util.spec_from_file_location("n01_acf005_null_helpers",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def conditional_raw_receipt(prediction,controls,helpers):
    labels=np.asarray(prediction["labels"]);scores=np.asarray(prediction["scores"])
    groups=np.r_[prediction["groups0"],prediction["groups1"]]
    _require(len(labels)==len(scores)==len(groups) and np.isfinite(scores).all(),"Malformed raw control scores")
    result=helpers.conditional_test_permutations(labels,scores,groups,int(controls["conditional_seed"]),int(controls["conditional_permutations"]))
    unique=np.unique(groups)
    quota=np.asarray([np.sum((groups==group)&(labels==1)) for group in unique],dtype=np.int64)
    _require(np.array_equal(quota,[np.sum((groups==group)&(labels==0)) for group in unique]) and (quota>0).all(),"Raw permutation group quotas unequal")
    values=result.pop("permutation_AUC");K=result["tail_count"];M=result["permutations"]
    result.update(seed=int(controls["conditional_seed"]),draws=M,
        arrays_sha256={"labels":array_digest(labels),"scores":array_digest(scores),"groups":array_digest(groups)},
        conditional_tail_fraction_K_over_draws=K/M,
        conditional_tail_probability_MC_Wilson95=helpers.wilson_interval(K,M),
        Wilson_scope="Monte_Carlo_tail_probability_conditional_on_fixed_scores_parents_groups;not_p_or_acquisitionCI",
        interpretation="descriptive only;p>.05 does not validate measurement or erase legacy control failure")
    numeric=dict(labels=labels,scores=scores,groups=groups,group_ids=unique,positive_quota=quota,
        permutation_AUC=values,seed=np.array(controls["conditional_seed"]),observed_AUC=np.array(result["observed_AUC"]))
    return result,numeric


def evaluate_acf005(gate,train,validation,train_groups,val_groups,config,out,*,root,diagnostic_dir):
    out,root,diagnostic_dir=map(Path,(out,root,diagnostic_dir))
    if os.name=="nt" and out.resolve().drive.upper() not in ("E:","D:"):raise ValueError("All local output must be on E/D")
    _require(not (out/"EVALUATION_RECEIPT.json").exists(),"Preserve completed005 evaluation")
    _require(config["acf20"]["lags"]==list(range(1,21)) and config["acf20"]["scale"]==.015 and config["acf20"]["coefficient"]==1.,"Frozen ACF rule changed")
    _require(config["acf20"]["variance_floor_V2"]==VARIANCE_FLOOR,"Quantization floor changed")
    _require(config["uncertainty"]["bootstrap_resamples"]==1000 and config["controls"]["conditional_permutations"]==9999,"Fixed diagnostic budget changed")
    references=verify_references(config,root) # hash every reference before decoding one
    legacy_path=out/"LEGACY_GATE.json"
    _require(legacy_path.is_file(),"Runner must first preserve exact LEGACY_GATE.json")
    legacy_saved=json.loads(legacy_path.read_text(encoding="utf8"))
    legacy=_clean(copy.deepcopy(gate))
    _require(legacy==legacy_saved,"Legacy gate argument/saved artifact differs")
    legacy_hash=sha256(legacy_path)
    train,train_acf=_events(train);validation,validation_acf=_events(validation)
    train_groups,val_groups=np.asarray(train_groups),np.asarray(val_groups)
    _require(train.shape==(21047,100) and validation.shape==(599,100),"Original allTRAIN/development row counts changed")
    _require(train_groups.shape==(21047,) and val_groups.shape==(599,),"Original group count/identity length mismatch")
    _require(len(np.unique(train_groups))==422 and len(np.unique(val_groups))==12 and not np.intersect1d(train_groups,val_groups).size,"Original TRAIN/DEV group roles changed")
    with np.load(diagnostic_dir/"paired_populations.npz",allow_pickle=False) as z:
        _require(set(z.files)==set(DIAGNOSTIC_KEYS),"Diagnostic population six-key schema changed")
        diagnostic={key:z[key].copy() for key in DIAGNOSTIC_KEYS}
    events={}
    for key,value in diagnostic.items():
        _require(value.shape==(8192,100) and value.dtype==np.float64,"Diagnostic values must be8192x100float64")
        _,events[key]=_events(value)
    with np.load(out/"baseline_populations.npz",allow_pickle=False) as z:
        legacy_generated=z["generated"].copy();legacy_real=z["REAL_validation"].copy();legacy_groups=z["real_validation_groups"].copy()
        selected0=z["real_validation_test0"].copy();selected1=z["real_validation_test1"].copy()
    _require(legacy_generated.shape==(8192,100),"Legacy evaluation generation count changed")
    _require(selected0.shape==selected1.shape==(256,) and np.issubdtype(selected0.dtype,np.integer) and np.issubdtype(selected1.dtype,np.integer),"Legacy test quota must be256per class")
    _require(np.unique(selected0).size==np.unique(selected1).size==256 and not np.intersect1d(selected0,selected1).size and
        ((selected0>=0)&(selected0<599)).all() and ((selected1>=0)&(selected1<599)).all(),"Legacy test rows must be disjoint legal unique parents")
    _require(np.array_equal(legacy_real,validation) and np.array_equal(legacy_groups,val_groups),"Legacy/005 development inputs differ")
    _,legacy_acf=_events(legacy_generated)
    target=_event_mean(train_acf);dev=_event_mean(validation_acf)
    series={"TRAIN_target":dict(mean=target,rows=21047,population_role="all_canonical_observed_TRAIN;no_analog_counterpart"),
            "REAL_validation_post_adc":dict(mean=dev,rows=599,population_role="development_reused_VAL_same_acquisition"),
            "legacy_generated_post_adc":dict(mean=_event_mean(legacy_acf),rows=8192,population_role="legacy_evaluation_stream8192;not_diagnostic_stream")}
    for key,a in events.items():series[key]=dict(mean=_event_mean(a),rows=8192,population_role="paired_fixed_diagnostic_latent_stream8192")
    rows=[]
    for k in range(20):
        row={"lag":k+1}
        for name,value in series.items():
            row[name]=float(value["mean"][k])
            if name not in ("TRAIN_target","REAL_validation_post_adc"):
                row[name+"_minus_TRAIN"]=float(value["mean"][k]-target[k])
                row[name+"_minus_DEVELOPMENT"]=float(value["mean"][k]-dev[k])
        rows.append(row)
    table=dict(lags=list(range(1,21)),definition="eventwise_demean;mean_overlap100-lag;variancefloorADCstep^2/12;equal_event_mean",
        series=series,rows=rows,TRAIN_is_postADC_observed_only=True,diagnostic_and_legacy_evaluation_streams_are_distinct=True)
    write_json(out/"ACF_PER_LAG.json",_clean(table))
    with (out/"ACF_PER_LAG.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    reproduction={};predictions={}
    for view in ("raw","demean","detrend"):
        with np.load(references["control004_"+view]["path"],allow_pickle=False) as z:old={key:z[key].copy() for key in z.files}
        current_path=out/"classifiers"/view/"REAL_REAL/predictions.npz"
        with np.load(current_path,allow_pickle=False) as z:new={key:z[key].copy() for key in z.files}
        reproduction[view]=compare_control_arrays(old,new);predictions[view]=new
        _require(new["labels"].shape==new["scores"].shape==(512,) and new["groups0"].shape==new["groups1"].shape==(256,),"RR saved test shape differs from256per class")
        _require(np.array_equal(new["labels"],np.r_[np.zeros(256),np.ones(256)]) and np.isfinite(new["scores"]).all(),"RR saved class order/nonfinite scores changed")
    raw=predictions["raw"]
    _require(np.array_equal(raw["groups0"],val_groups[selected0]) and np.array_equal(raw["groups1"],val_groups[selected1]),"Control selected test groups/input roles differ")
    helpers=_null_helpers(root)
    conditional,perm_numeric=conditional_raw_receipt(raw,config["controls"],helpers)
    perm_numeric.update(validation_test0_rows=selected0,validation_test1_rows=selected1)
    conditional["selected_test_rows_sha256"]={"class0":array_digest(selected0),"class1":array_digest(selected1)}
    np.savez_compressed(out/"RAW_CONTROL_CONDITIONAL.npz",**perm_numeric)
    calibration=json.loads(Path(references["calibration003"]["path"]).read_text(encoding="utf8"))
    _require(calibration["repeats"]==128 and calibration["legacy_joint_bootstrap_false_positive"]["rejections"]==9 and calibration["conditional_permutation_false_positive"]["rejections"]==7,"Reference003 calibrated endpoints differ")
    _require(calibration["binding"]["registered_rules"]["test_label_permutations"]==255,"Reference003 finite permutation rule differs")
    control=dict(status="measurement_uncertain_single_exposed_acquisition",exact_score_reproduction=reproduction,
        reference003=dict(refits=128,method_draws=255,
            legacy_joint_bootstrap_false_positive=calibration["legacy_joint_bootstrap_false_positive"],
            conditional_permutation_false_positive=calibration["conditional_permutation_false_positive"],
            scope="conditional_historical_TRAIN_pseudo_groups;7/128 does not calibrate9999draw_rule"),
        current_raw=conditional,legacy_raw_point=legacy["classifiers"]["raw"]["REAL_REAL"]["AUC"],
        legacy_raw_interval95=legacy["classifiers"]["raw"]["REAL_REAL"]["percentile_95_interval"],
        legacy_raw_practical_interval_contained=bool(np.all(np.asarray(legacy["classifiers"]["raw"]["REAL_REAL"]["percentile_95_interval"])>=.35) and np.all(np.asarray(legacy["classifiers"]["raw"]["REAL_REAL"]["percentile_95_interval"])<=.65)),
        newly_fitted_classifiers=0,other_views_conditional_tests=0,measurement_certified=False)
    write_json(out/"CONTROL_CALIBRATION.json",_clean(control))
    train_uncertainty,train_numeric=bootstrap_acf_difference(events["v005_post_adc"],train_acf,train_groups,config["uncertainty"]["TRAIN_seed"],1000)
    dev_uncertainty,dev_numeric=bootstrap_acf_difference(legacy_acf,validation_acf,val_groups,config["uncertainty"]["development_seed"],1000)
    train_uncertainty["generated_population_role"]="v005_fixed_diagnostic8192_postADC"
    dev_uncertainty["generated_population_role"]="legacy_evaluation8192_postADC"
    np.savez_compressed(out/"bootstrap_TRAIN.npz",**train_numeric)
    np.savez_compressed(out/"bootstrap_DEVELOPMENT.npz",**dev_numeric)
    uncertainty=dict(TRAIN=train_uncertainty,development=dev_uncertainty,acceptance_gate=False)
    write_json(out/"ACF_UNCERTAINTY.json",_clean(uncertainty))
    candidate=candidate_screen(legacy)
    decision=train_decision(_event_mean(events["v005_post_adc"]),_event_mean(events["v004_post_adc"]),target,candidate["passed"])
    checks=dict(reference_SHA_verified=True,legacy_gate_preserved_exactly=True,strict100_and_all_rows=True,
        TRAIN_development_groups_disjoint=True,shared_legacy_ACF_definition_agrees=True,
        paired_diagnostic_and_legacy_eval_streams_separated=True,REAL_REAL_three_views_bitwise_reproduced=True,
        raw_conditional_test_fixed_scores_and_group_quotas=True,bootstrap_whole_REAL_groups_event_weighted=True,
        no_new_classifier_or_generator_fits=True)
    _require(sha256(legacy_path)==legacy_hash,"Legacy gate changed during separated evaluation")
    generated_files=("ACF_PER_LAG.json","ACF_PER_LAG.csv","CONTROL_CALIBRATION.json","RAW_CONTROL_CONDITIONAL.npz",
                     "ACF_UNCERTAINTY.json","bootstrap_TRAIN.npz","bootstrap_DEVELOPMENT.npz")
    receipt=dict(run_id=config["run_id"],implementation_checks=dict(status="module_checks_passed_pending_runner_binding",
            checks=checks,pending_runner_binding=True,
            runner_assertions_required=["initial_state_equals004","immutable_config_source_input","all_TRAIN_coverage",
                "fixed_TRAIN20_target","fixed5000_checkpoint_reopened","frozen_generator_during_evaluation",
                "explicit1024_replay","diagnostic_RNG_isolation","TRAIN_initialization_only","VAL_only_after_endpoint",
                "exact_classifier_replay"]),
        legacy_gate=legacy,candidate_checks=candidate,control_calibration=control,decision=decision,
        support_scope=dict(supported_samples=100,full480_integration_supported=False,independent_acquisition_validations=0,
            development_validation_previously_exposed=True,model="effective_observed_baseline_not_hardware_identification",
            bootstrap_training_randomness_included=False,new_acceptance_gates_introduced=False),
        paired_population_sha256=sha256(diagnostic_dir/"paired_populations.npz"),
        legacy_population_sha256=sha256(out/"baseline_populations.npz"),target_ACF20_sha256=hashlib.sha256(np.ascontiguousarray(target,dtype="<f8").tobytes()).hexdigest(),
        target_ACF20_digest_definition="raw_contiguous_little_endian_float64_bytes;identical_fixed_TRAIN_target",
        references=references,artifacts={name:dict(sha256=sha256(out/name),bytes=(out/name).stat().st_size) for name in generated_files})
    write_json(out/"EVALUATION_RECEIPT.json",_clean(receipt))
    return _clean(receipt)
