"""Fixed128 REAL-REAL full-refit null calibration, using legal TRAIN only.

No electronics/response generator is instantiated or trained. Scientific arrays
are decoded only by run_calibration after configured input hashes are verified.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from n01.data import canonical_real,sha256,write_json
from n01.evaluate import _weighted_auc,_clean
from n01.noise_validation import BaselineClassifier,balanced_real_domains


FIXED=dict(repeats=128,classifier_updates=200,classifier_fit_per_class=2048,
           classifier_test_per_class=256,bootstrap_resamples=1000,test_label_permutations=255,
           pseudo_test_groups=12,min_group_rows=44,view="raw",significance_alpha=.05)
SEED_KEYS=("pseudo_split_seed","split_seed_base","classifier_seed_base","permutation_seed_base","bootstrap_seed_base")


def validate_config(config):
    cfg=config["rr_calibration"]
    for key,value in FIXED.items():
        if cfg.get(key)!=value:raise ValueError(f"Registered REAL-REAL budget/rule changed: {key}")
    for key in SEED_KEYS:
        if not isinstance(cfg.get(key),int) or cfg[key]<0:raise ValueError(f"Invalid seed {key}")
    if config.get("validation_waveform_decode_allowed") is not False or config.get("modify_old_gate_allowed") is not False:
        raise ValueError("This null calibration may not read VAL waveforms or revise the old gate")
    for key in ("noise","response"):
        if config["training_updates"].get(key)!=0:raise ValueError("Noise/response training prohibited")
    for field in ("training_container","identities"):
        digest=config[field]["sha256"]
        if len(digest)!=64 or any(c not in "0123456789abcdefABCDEF" for c in digest):raise ValueError("Invalid registered input hash")
    return cfg


def read_identity_metadata(path):
    with np.load(path,allow_pickle=False) as z:
        ids=z["real_train_original_index"].copy();groups=z["real_train_group"].copy()
        val_ids=z["real_val_original_index"].copy() if "real_val_original_index" in z.files else np.array([],dtype=ids.dtype)
        val_groups=z["real_val_group"].copy() if "real_val_group" in z.files else np.array([],dtype=groups.dtype)
    if ids.ndim!=1 or groups.shape!=ids.shape or len(np.unique(ids))!=len(ids):raise ValueError("Malformed/nonunique TRAIN identities")
    if np.intersect1d(ids,val_ids).size or np.intersect1d(groups,val_groups).size:raise ValueError("Registered TRAIN/VAL identity/group overlap")
    return ids,groups,dict(TRAIN_rows=len(ids),TRAIN_groups=len(np.unique(groups)),
        VAL_identity_rows=len(val_ids),TRAIN_VAL_ID_overlap=0,TRAIN_VAL_group_overlap=0,VAL_waveforms_decoded=False)


def select_pseudo_roles(groups,count=12,min_rows=44,seed=2026101260):
    groups=np.asarray(groups);unique,counts=np.unique(groups,return_counts=True)
    eligible=unique[counts>=min_rows]
    if len(eligible)<count:raise ValueError("Insufficient complete legal-TRAIN groups for prespecified pseudo-test")
    chosen=np.random.default_rng(seed).choice(eligible,count,replace=False)
    test=np.flatnonzero(np.isin(groups,chosen));fit=np.flatnonzero(~np.isin(groups,chosen))
    if np.intersect1d(groups[fit],groups[test]).size:raise AssertionError("Pseudo role group leakage")
    return fit,test,chosen


def check_capacity(groups,per_class):
    unique,counts=np.unique(groups,return_counts=True)
    if len(unique)==0 or int(counts.min())<2*math.ceil(per_class/len(unique)):
        raise ValueError("Old balanced group quota is unsupported; no adaptive cap/count/group change")
    return dict(groups=len(unique),minimum_rows=int(counts.min()),maximum_rows=int(counts.max()),
                quota_per_class_floor=per_class//len(unique),quota_per_class_ceiling=math.ceil(per_class/len(unique)))


def draw_roles(groups,fit_pool,test_pool,cfg,index):
    base=int(cfg["split_seed_base"])+index
    # Disjoint role seed ranges avoid initializing fit/test to the same RNG.
    test_seed=base+int(cfg["repeats"])
    fa,fb=balanced_real_domains(groups[fit_pool],cfg["classifier_fit_per_class"],base)
    ta,tb=balanced_real_domains(groups[test_pool],cfg["classifier_test_per_class"],test_seed)
    result=dict(fit0=fit_pool[fa],fit1=fit_pool[fb],test0=test_pool[ta],test1=test_pool[tb])
    for role in ("fit","test"):
        a,b=result[role+"0"],result[role+"1"]
        if np.intersect1d(a,b).size:raise AssertionError("Random-label domains reused an event")
        ga,ca=np.unique(groups[a],return_counts=True);gb,cb=np.unique(groups[b],return_counts=True)
        if not np.array_equal(ga,gb) or not np.array_equal(ca,cb):raise AssertionError("Random labels have unequal group quota")
    if np.intersect1d(groups[np.r_[result["fit0"],result["fit1"]]],groups[np.r_[result["test0"],result["test1"]]]).size:
        raise AssertionError("Classifier fit/test original group leakage")
    return result,dict(split_seed=base,fit_label_seed=base,test_label_seed=test_seed,
        classifier_seed=int(cfg["classifier_seed_base"])+index,
        optimizer_batch_seed=int(cfg["classifier_seed_base"])+index+1,
        bootstrap_seed=int(cfg["bootstrap_seed_base"])+index,
        conditional_permutation_seed=int(cfg["permutation_seed_base"])+index)


def average_ranks(scores):
    x=np.asarray(scores,dtype=np.float64)
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError("Scores must be finite one-dimensional values")
    order=np.argsort(x,kind="stable");sorted_x=x[order]
    start=np.r_[0,np.flatnonzero(np.diff(sorted_x))+1];end=np.r_[start[1:],len(x)]
    ranks=np.empty(len(x),dtype=np.float64)
    for a,b in zip(start,end):ranks[order[a:b]]=(a+1+b)/2.
    return ranks


def rank_auc(labels,ranks):
    y=np.asarray(labels);r=np.asarray(ranks)
    if len(y)!=len(r) or not np.isin(y,[0,1]).all():raise ValueError("Invalid binary labels")
    n1=int(y.sum());n0=len(y)-n1
    if not n0 or not n1:raise ValueError("Both labels are required")
    return float((r[y==1].sum()-n1*(n1+1)/2.)/(n0*n1))


def joint_group_bootstrap(labels,scores,groups,seed,resamples=1000):
    """Same old joint-cluster weighted AUC, with comparisons precomputed once."""
    y=np.asarray(labels);s=np.asarray(scores);g=np.asarray(groups)
    if len(y)!=len(s) or len(y)!=len(g) or not np.isfinite(s).all() or not np.isin(y,[0,1]).all():raise ValueError("Invalid bootstrap inputs")
    unique=np.unique(g);G=len(unique)
    if G<2:raise ValueError("At least two groups needed for leave-one-group/bootstrap calibration")
    comparisons=np.zeros((G,G),dtype=np.float64)
    n1=np.array([np.sum((g==group)&(y==1)) for group in unique],dtype=np.float64)
    n0=np.array([np.sum((g==group)&(y==0)) for group in unique],dtype=np.float64)
    if not np.array_equal(n1,n0) or (n1<=0).any():raise ValueError("Joint null bootstrap requires matched nonzero labels within every group")
    for a,ga in enumerate(unique):
        positive=s[(g==ga)&(y==1)]
        for b,gb in enumerate(unique):
            negative=s[(g==gb)&(y==0)]
            comparisons[a,b]=np.sum(positive[:,None]>negative[None,:])+.5*np.sum(positive[:,None]==negative[None,:])
    draws=np.random.default_rng(seed).integers(0,G,(resamples,G))
    multiplicity=np.zeros((resamples,G),dtype=np.float64)
    for i,row in enumerate(draws):multiplicity[i]=np.bincount(row,minlength=G)
    denominator=(multiplicity@n1)*(multiplicity@n0)
    values=np.einsum("bi,ij,bj->b",multiplicity,comparisons,multiplicity)/denominator
    interval=np.quantile(values,[.025,.975])
    leave=[]
    for i in range(G):
        keep=np.ones(G);keep[i]=0
        leave.append(float(keep@comparisons@keep/((keep@n1)*(keep@n0))))
    return dict(method="joint_original_group_across_random_labels",resamples=resamples,
        percentile_95_interval=interval,CI_contains_half=bool(interval[0]<=.5<=interval[1]),
        bootstrap_SE=float(values.std(ddof=1)),bootstrap_AUC=values,
        leave_one_group_AUC=np.asarray(leave),leave_one_group_AUC_range=[min(leave),max(leave)],
        groups=G,comparison_matrix= comparisons,limited_original_group_count=True,
        independent_acquisition_uncertainty=False)


def conditional_test_permutations(labels,scores,groups,seed,permutations=255):
    """Scores/test parents fixed; label counts preserved inside each group."""
    y=np.asarray(labels,dtype=np.int8);g=np.asarray(groups);ranks=average_ranks(scores)
    if len(y)!=len(g):raise ValueError("Permutation shape mismatch")
    unique=np.unique(g);rows=[np.flatnonzero(g==group) for group in unique]
    positive=[int(y[row].sum()) for row in rows]
    if any(2*n!=len(row) or n==0 for n,row in zip(positive,rows)):
        raise ValueError("Every test group needs equal nonzero label quotas")
    observed=rank_auc(y,ranks);deviation=abs(observed-.5)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(permutations):
        permuted=np.zeros(len(y),dtype=np.int8)
        for row,n in zip(rows,positive):permuted[rng.permutation(row)[:n]]=1
        values.append(rank_auc(permuted,ranks))
    values=np.asarray(values)
    tail=int(np.sum(abs(values-.5)>=deviation))
    return dict(method="conditional_within_group_equal_quota_test_label_permutation",
        permutations=permutations,observed_AUC=observed,statistic=deviation,
        statistic_definition="abs(AUC-.5)",tail_rule=">= including ties",tail_count=tail,
        plus_one_p_value=(tail+1)/(permutations+1),permutation_AUC=values,
        scores_fixed=True,test_parent_set_fixed=True,classifier_refitted=False)


def wilson_interval(successes,trials,z=1.959963984540054):
    if trials<=0 or not 0<=successes<=trials:raise ValueError("Invalid binomial counts")
    p=successes/trials;denom=1+z*z/trials
    center=(p+z*z/(2*trials))/denom
    half=z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denom
    return [max(0.,center-half),min(1.,center+half)]


def fit_raw_classifier(train0,train1,test0,test1,seed,device,updates=200,progress=None):
    """Same architecture/scaler/Adam/minibatch rules as noise_validation._classifier."""
    x=np.concatenate([train0,train1]).astype(np.float32)
    center=x.mean(0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(x.std(0,dtype=np.float64),1/16383).astype(np.float32)
    train=torch.as_tensor((x-center)/scale,device=device)
    target=torch.as_tensor(np.r_[np.zeros(len(train0)),np.ones(len(train1))],device=device,dtype=torch.float32)
    test=torch.as_tensor((np.concatenate([test0,test1]).astype(np.float32)-center)/scale,device=device)
    torch.manual_seed(seed)
    model=BaselineClassifier().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    rng=np.random.default_rng(seed+1);losses=[]
    if progress is not None:progress.update(updates_executed=0,optimizer_step_in_progress=False)
    for step in range(updates):
        selected=rng.choice(len(train),128,replace=False)
        loss=F.binary_cross_entropy_with_logits(model(train[selected]),target[selected])
        if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite null classifier loss")
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if progress is not None:progress["optimizer_step_in_progress"]=True
        optimizer.step()
        if progress is not None:progress.update(updates_executed=step+1,optimizer_step_in_progress=False)
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():scores=model(test).cpu().numpy()
    checkpoint=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},center=center,scale=scale,
                    seed=seed,updates=updates,architecture="noise_validation.BaselineClassifier_Conv8_16_MLP32",view="raw")
    if not np.isfinite(scores).all():raise FloatingPointError("Nonfinite null classifier scores")
    return scores,checkpoint,np.asarray(losses)


def read_train_baselines_only(container,identities):
    ids,groups,metadata=read_identity_metadata(identities)
    with np.load(container,allow_pickle=False) as z:target=z["target"]
    if target.shape!=(len(ids),480):raise ValueError("TRAIN target shape/identity mismatch")
    baseline=canonical_real(target)[:,:100].copy()
    return baseline,ids,groups,metadata


def _resolve(path,config_path):
    p=Path(path)
    return p if p.is_absolute() else Path(config_path).parent/p


def run_calibration(config_path,out,device="cuda",container=None,identities=None):
    config_path,out=Path(config_path),Path(out)
    if os.name=="nt" and out.resolve().drive.upper() not in ("E:","D:"):raise ValueError("All local outputs must be on E/D")
    if out.exists() and any(out.iterdir()):raise FileExistsError("Preserve previous/partial null calibration; choose a fresh output directory")
    config=json.loads(config_path.read_text(encoding="utf8"));cfg=validate_config(config)
    container=Path(container) if container else _resolve(config["training_container"]["path"],config_path)
    identities=Path(identities) if identities else _resolve(config["identities"]["path"],config_path)
    if sha256(container)!=config["training_container"]["sha256"] or sha256(identities)!=config["identities"]["sha256"]:
        raise ValueError("Registered TRAIN container/identity hash mismatch")
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic();completed=[]
    partial_counter=dict(updates_executed=0,optimizer_step_in_progress=False,repetition=None)
    torch.set_num_threads(int(config["resource"]["cpu_threads"]))
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    target=torch.device(device)
    if target.type=="cuda":
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        budget=float(config["resource"]["gpu_memory_GiB"])*1024**3
        torch.cuda.set_per_process_memory_fraction(min(1.,budget/torch.cuda.get_device_properties(target).total_memory),target)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha256(__file__),
        "src/n01/noise_validation.py":sha256(ROOT/"src/n01/noise_validation.py"),
        "src/n01/evaluate.py":sha256(ROOT/"src/n01/evaluate.py"),"src/n01/data.py":sha256(ROOT/"src/n01/data.py")}
    try:
        baseline,ids,groups,identity_meta=read_train_baselines_only(container,identities)
        fit_pool,test_pool,test_groups=select_pseudo_roles(groups,cfg["pseudo_test_groups"],cfg["min_group_rows"],cfg["pseudo_split_seed"])
        fit_capacity=check_capacity(groups[fit_pool],cfg["classifier_fit_per_class"])
        test_capacity=check_capacity(groups[test_pool],cfg["classifier_test_per_class"])
        visits_fit=np.zeros(len(ids),dtype=np.int64);visits_test=np.zeros(len(ids),dtype=np.int64)
        np.savez_compressed(out/"TRAIN_POOL_BINDING.npz",original_TRAIN_ids=ids,original_TRAIN_groups=groups,
            pseudo_fit_pool_rows=fit_pool,pseudo_test_pool_rows=test_pool,pseudo_test_groups=test_groups)
        binding=dict(experiment_id=config["experiment_id"],config_sha256=sha256(config_path),training_container_sha256=sha256(container),
            identities_sha256=sha256(identities),training_container=str(container),identities=str(identities),
            identity_metadata=identity_meta,pseudo_test_groups=test_groups,pseudo_split_seed=cfg["pseudo_split_seed"],
            pseudo_fit_capacity=fit_capacity,pseudo_test_capacity=test_capacity,registered_rules=cfg,
            canonicalization="stored target -> n01.data.canonical_real inclusive14bit float64 -> first100",
            decoded_container_array_keys=["target"],source_or_validation_waveforms_decoded=False,
            split_seed_rule="fit_label_seed=split_base+i;test_label_seed=split_base+128+i;other_bases+i",
            dependencies_sha256=dependencies,device=str(target),torch=torch.__version__,numpy=np.__version__,
            noise_model_training_updates=0,response_model_training_updates=0,old_noise_gate_modified=False)
        write_json(out/"binding.json",_clean(binding))
        for i in range(cfg["repeats"]):
            if time.monotonic()-started>config["resource"]["wall_limit_s"]:raise TimeoutError("Registered wall budget reached")
            rows,seeds=draw_roles(groups,fit_pool,test_pool,cfg,i)
            partial_counter=dict(updates_executed=0,optimizer_step_in_progress=False,repetition=i)
            scores,state,losses=fit_raw_classifier(baseline[rows["fit0"]],baseline[rows["fit1"]],
                baseline[rows["test0"]],baseline[rows["test1"]],seeds["classifier_seed"],target,cfg["classifier_updates"],progress=partial_counter)
            test_rows=np.r_[rows["test0"],rows["test1"]]
            labels=np.r_[np.zeros(len(rows["test0"])),np.ones(len(rows["test1"]))].astype(np.int8)
            test_group=groups[test_rows]
            bootstrap=joint_group_bootstrap(labels,scores,test_group,seeds["bootstrap_seed"],cfg["bootstrap_resamples"])
            permutations=conditional_test_permutations(labels,scores,test_group,seeds["conditional_permutation_seed"],cfg["test_label_permutations"])
            path=out/f"repetition_{i:03d}";path.mkdir()
            torch.save(state,path/"classifier.pt")
            np.savez_compressed(path/"audit.npz",**rows,seeds_json=np.array(json.dumps(seeds,sort_keys=True)),
                test_original_ids=ids[test_rows],test_original_groups=test_group,labels=labels,scores=scores,fit_loss=losses,
                fit0_original_ids=ids[rows["fit0"]],fit1_original_ids=ids[rows["fit1"]],
                bootstrap_AUC=bootstrap["bootstrap_AUC"],leave_one_group_AUC=bootstrap["leave_one_group_AUC"],
                bootstrap_comparison_matrix=bootstrap["comparison_matrix"],permutation_AUC=permutations["permutation_AUC"])
            for key in ("bootstrap_AUC","leave_one_group_AUC","comparison_matrix"):bootstrap.pop(key)
            permutations.pop("permutation_AUC")
            result=dict(repetition=i,seeds=seeds,AUC=_weighted_auc(labels,scores),bootstrap=bootstrap,conditional_permutation=permutations,
                n_fit=[2048,2048],n_test=[256,256],test_original_groups=12,classifier_updates=200,
                noise_model_training_updates=0,response_model_training_updates=0,orientation_flipped=False,scaler_fit_only=True)
            write_json(path/"result.json",_clean(result));completed.append(_clean(result))
            partial_counter=dict(updates_executed=0,optimizer_step_in_progress=False,repetition=None)
            np.add.at(visits_fit,np.r_[rows["fit0"],rows["fit1"]],1);np.add.at(visits_test,test_rows,1)
            progress=dict(status="fitting_null_classifiers",completed_repetitions=i+1,planned_repetitions=128,
                evaluation_classifier_updates_completed=(i+1)*200,last_AUC=result["AUC"],
                last_CI_contains_half=bootstrap["CI_contains_half"],last_permutation_p=permutations["plus_one_p_value"],
                elapsed_s=time.monotonic()-started,noise_model_training_updates=0,response_model_training_updates=0)
            write_json(out/"status.json",progress);print(json.dumps(progress),flush=True)
            with (out/"progress.jsonl").open("a",encoding="utf8") as stream:stream.write(json.dumps(progress)+"\n")
        auc=np.array([v["AUC"] for v in completed]);p=np.array([v["conditional_permutation"]["plus_one_p_value"] for v in completed])
        coverage=np.array([v["bootstrap"]["CI_contains_half"] for v in completed])
        reject_boot=int((~coverage).sum());reject_perm=int((p<=cfg["significance_alpha"]).sum())
        np.savez_compressed(out/"ALL_REPETITIONS.npz",AUC=auc,permutation_p=p,bootstrap_CI_contains_half=coverage,
            bootstrap_CI=np.asarray([v["bootstrap"]["percentile_95_interval"] for v in completed]),
            fit_original_id_visit_count=visits_fit,test_original_id_visit_count=visits_test)
        checkpoint_bytes=sum(v.stat().st_size for v in out.glob("repetition_*/classifier.pt"))
        if checkpoint_bytes>5*1024**2:raise RuntimeError("Null classifier checkpoint total exceeds5MiB")
        if any(sha256(ROOT/name)!=digest for name,digest in dependencies.items()):raise RuntimeError("Bound source changed during calibration")
        output_bytes=sum(v.stat().st_size for v in out.rglob("*") if v.is_file())
        if output_bytes>float(config["resource"]["output_limit_GiB"])*1024**3:
            raise RuntimeError("Registered output-size limit reached")
        # status is updated after report finalization and bound separately in
        # COMPLETE; including its old hash here would create a circular receipt.
        artifacts={str(v.relative_to(out)):dict(bytes=v.stat().st_size,sha256=sha256(v))
            for v in sorted(out.rglob("*")) if v.is_file() and v.name!="status.json"}
        report=dict(experiment_id=config["experiment_id"],status="fixed128_refit_null_calibration_complete",repeats=128,
            AUC_mean=float(auc.mean()),AUC_std_ddof1=float(auc.std(ddof=1)),
            null_quantile_probabilities=[.005,.025,.05,.5,.95,.975,.995],
            null_AUC_quantiles=np.quantile(auc,[.005,.025,.05,.5,.95,.975,.995]),
            legacy_joint_bootstrap_coverage=float(coverage.mean()),
            legacy_joint_bootstrap_false_positive=dict(rejections=reject_boot,trials=128,rate=reject_boot/128,Wilson95=wilson_interval(reject_boot,128)),
            conditional_permutation_false_positive=dict(rejections=reject_perm,trials=128,alpha=.05,rate=reject_perm/128,Wilson95=wilson_interval(reject_perm,128)),
            conditional_permutation_p_quantiles=np.quantile(p,[.025,.05,.5,.95,.975]),
            classifier_architecture="noise_validation.BaselineClassifier",classifier_updates_each=200,evaluation_classifier_updates=25600,
            checkpoint_total_bytes=checkpoint_bytes,elapsed_s=time.monotonic()-started,binding=binding,repetitions=completed,
            Monte_Carlo_FPR_resolution=1/128,permutation_p_resolution=1/256,
            Wilson_scope="conditional_on_one_fixed_TRAIN_pool_and_fixed12_pseudo_test_groups;independent_label_and_fit_randomizations",
            permutation_scope="conditional_on_each_fixed_classifier_scores_and_selected_test_events;not_full_refit_replacement",
            original_groups_are_artificial_50_record_blocks=True,independent_acquisition_validation=False,
            original_VAL_waveforms_decoded=False,old_noise_gate_modified=False,
            noise_model_training_updates=0,response_model_training_updates=0,artifacts=artifacts)
        write_json(out/"RR_CALIBRATION_REPORT.json",_clean(report))
        write_json(out/"status.json",dict(status="complete",completed_repetitions=128,planned_repetitions=128,
            evaluation_classifier_updates_completed=25600,elapsed_s=time.monotonic()-started,
            report_sha256=sha256(out/"RR_CALIBRATION_REPORT.json"),noise_model_training_updates=0,response_model_training_updates=0))
        write_json(out/"COMPLETE.json",dict(status="complete",completed_UTC=datetime.now(timezone.utc).isoformat(),
            report_sha256=sha256(out/"RR_CALIBRATION_REPORT.json"),status_sha256=sha256(out/"status.json"),evaluation_classifier_updates=25600,
            noise_model_training_updates=0,response_model_training_updates=0))
        return report
    except BaseException as error:
        write_json(out/"FAILED.json",dict(status="failed",exception=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            completed_repetitions=len(completed),completed_block_classifier_updates=len(completed)*200,
            partial_refit=partial_counter,evaluation_classifier_updates_known=len(completed)*200+partial_counter["updates_executed"],
            partial_optimizer_state_unknown=partial_counter["optimizer_step_in_progress"],
            elapsed_s=time.monotonic()-started,noise_model_training_updates=0,response_model_training_updates=0))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path);parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--device",default="cuda",choices=("cpu","cuda"))
    parser.add_argument("--container",type=Path);parser.add_argument("--identities",type=Path)
    args=parser.parse_args(argv)
    run_calibration(args.config,args.out,args.device,args.container,args.identities)


if __name__=="__main__":main()
