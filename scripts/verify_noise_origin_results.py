"""Post-run128-endpoint REAL-REAL artifact and exact-score replay QA only.

No model fitting, bootstrap resampling, label permutation, or seed selection.
Only the target TRAIN array is decoded; replay uses the original whole512 batch.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from n01.data import sha256,write_json
from n01.noise_validation import BaselineClassifier
from n01.evaluate import _weighted_auc,_clean
from calibrate_real_real_null import read_train_baselines_only,validate_config,wilson_interval


def require(condition,message):
    if not bool(condition):raise ValueError(message)


def same(actual,expected,message):
    require(np.array_equal(np.asarray(actual),np.asarray(expected)),message)


def close(actual,expected,message):
    require(np.shape(actual)==np.shape(expected) and np.allclose(actual,expected,rtol=0,atol=1e-15),message)


def tensor_digest(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        digest.update(key.encode());digest.update(str(tuple(x.shape)).encode());digest.update(str(x.dtype).encode());digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def replay_scores(state,test_baseline,device):
    require(np.shape(test_baseline)==(512,100),"Replay must retain the original whole512x100 batch")
    center=np.asarray(state["center"]);scale=np.asarray(state["scale"])
    require(center.shape==scale.shape==(100,) and np.isfinite(center).all() and np.isfinite(scale).all() and (scale>0).all(),"Malformed stored scaler")
    x=(np.asarray(test_baseline).astype(np.float32)-center)/scale
    before=tensor_digest(state["model"])
    model=BaselineClassifier()
    model.load_state_dict(state["model"],strict=True)
    model.eval().requires_grad_(False).to(device)
    with torch.no_grad():scores=model(torch.as_tensor(x,device=device)).cpu().numpy()
    require(np.isfinite(scores).all(),"Nonfinite replay scores")
    require(tensor_digest(model.state_dict())==before,"Replay modified model parameters")
    del model
    return scores


def comparison_matrix(labels,scores,groups):
    unique=np.unique(groups);matrix=np.zeros((len(unique),len(unique)))
    for i,g in enumerate(unique):
        positive=scores[(groups==g)&(labels==1)]
        for j,h in enumerate(unique):
            negative=scores[(groups==h)&(labels==0)]
            matrix[i,j]=np.sum(positive[:,None]>negative[None,:])+.5*np.sum(positive[:,None]==negative[None,:])
    return matrix


def verify_results(root,container,identities,config_path,device,out):
    root,container,identities,config_path,out=map(Path,(root,container,identities,config_path,out))
    if os.name=="nt" and out.resolve().drive.upper() not in ("E:","D:"):raise ValueError("Local QA outputs must be on E/D")
    if out.exists() and any(out.iterdir()):raise FileExistsError("Preserve previous QA; choose a new output directory")
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic();receipts=[]
    try:
        rr=root if (root/"RR_CALIBRATION_REPORT.json").is_file() else root/"real_real"
        report_path=rr/"RR_CALIBRATION_REPORT.json"
        report=json.loads(report_path.read_text(encoding="utf8"));complete=json.loads((rr/"COMPLETE.json").read_text(encoding="utf8"))
        binding=json.loads((rr/"binding.json").read_text(encoding="utf8"))
        config=json.loads(config_path.read_text(encoding="utf8"));cfg=validate_config(config)
        require(report["status"]=="fixed128_refit_null_calibration_complete" and report["repeats"]==128,"Incomplete/null-budget mismatch")
        require(complete["status"]=="complete" and complete["report_sha256"]==sha256(report_path),"Report completion hash mismatch")
        require(complete["status_sha256"]==sha256(rr/"status.json"),"Final null status changed")
        require(report["evaluation_classifier_updates"]==25600 and report["noise_model_training_updates"]==report["response_model_training_updates"]==0,"Scientific update accounting mismatch")
        require(report["old_noise_gate_modified"] is False and report["original_VAL_waveforms_decoded"] is False,"Scientific scope violated")
        require(binding==report["binding"],"Report/binding disagreement")
        require(binding["config_sha256"]==sha256(config_path),"Configuration changed")
        require(binding["training_container_sha256"]==config["training_container"]["sha256"]==sha256(container),"TRAIN input hash mismatch")
        require(binding["identities_sha256"]==config["identities"]["sha256"]==sha256(identities),"Identity input hash mismatch")
        require(binding["registered_rules"]==cfg,"Null rules changed")
        for name,digest in binding["dependencies_sha256"].items():require(sha256(ROOT/name)==digest,"Bound source changed: "+name)
        for name,value in report["artifacts"].items():
            file=rr/name
            require(file.is_file() and file.stat().st_size==value["bytes"] and sha256(file)==value["sha256"],"Artifact changed: "+name)
        expected={f"repetition_{i:03d}" for i in range(128)}
        require({p.name for p in rr.glob("repetition_*") if p.is_dir()}==expected,"Unexpected or missing endpoint directory")
        torch.set_num_threads(int(config["resource"]["cpu_threads"]))
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        target=torch.device('cuda:0' if device=='cuda' else device)
        if target.type=="cuda":
            require(torch.cuda.is_available(),"CUDA unavailable")
            limit=float(config["resource"]["gpu_memory_GiB"])*1024**3
            torch.cuda.set_per_process_memory_fraction(min(1.,limit/torch.cuda.get_device_properties(target).total_memory),target)
        baseline,ids,groups,meta=read_train_baselines_only(container,identities)
        with np.load(rr/"TRAIN_POOL_BINDING.npz",allow_pickle=False) as z:pool={k:z[k].copy() for k in z.files}
        same(ids,pool["original_TRAIN_ids"],"Full original TRAIN identities changed")
        same(groups,pool["original_TRAIN_groups"],"Original TRAIN grouping changed")
        fit_pool,test_pool=pool["pseudo_fit_pool_rows"],pool["pseudo_test_pool_rows"]
        require(not np.intersect1d(groups[fit_pool],groups[test_pool]).size,"Pseudo fit/test group overlap")
        require(np.unique(groups[test_pool]).size==12,"Pseudo test groups must remain12")
        same(np.sort(np.r_[fit_pool,test_pool]),np.arange(len(ids)),"Role pools do not cover original TRAIN")
        same(test_pool,np.flatnonzero(np.isin(groups,pool["pseudo_test_groups"])),"Pseudo test contains partial group role")
        auc=[];pvalues=[];intervals=[];coverage=[];fit_visits=np.zeros(len(ids),dtype=np.int64);test_visits=fit_visits.copy()
        for i in range(128):
            directory=rr/f"repetition_{i:03d}"
            # Weights are the user's trusted, hash-bound artifacts from this project.
            state=torch.load(directory/"classifier.pt",map_location="cpu",weights_only=False)
            result=json.loads((directory/"result.json").read_text(encoding="utf8"))
            with np.load(directory/"audit.npz",allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
            require(state["updates"]==result["classifier_updates"]==200,"Checkpoint update count mismatch")
            require(result["repetition"]==i and result==report["repetitions"][i],"Per-endpoint result differs from aggregate")
            same(a["fit_loss"].shape,(200,),"Loss trace length is not200")
            require(np.isfinite(a["fit_loss"]).all(),"Nonfinite training loss trace")
            seeds=json.loads(str(a["seeds_json"]))
            expected_seeds=dict(split_seed=cfg["split_seed_base"]+i,fit_label_seed=cfg["split_seed_base"]+i,
                test_label_seed=cfg["split_seed_base"]+128+i,classifier_seed=cfg["classifier_seed_base"]+i,
                optimizer_batch_seed=cfg["classifier_seed_base"]+i+1,bootstrap_seed=cfg["bootstrap_seed_base"]+i,
                conditional_permutation_seed=cfg["permutation_seed_base"]+i)
            require(seeds==result["seeds"]==expected_seeds and state["seed"]==expected_seeds["classifier_seed"],"Endpoint seed binding differs")
            for role,count,role_pool in (("fit",2048,fit_pool),("test",256,test_pool)):
                rows0,rows1=a[role+"0"],a[role+"1"]
                require(rows0.shape==rows1.shape==(count,) and np.unique(rows0).size==np.unique(rows1).size==count,"Endpoint row multiplicity/count mismatch")
                require(np.isin(rows0,role_pool).all() and np.isin(rows1,role_pool).all() and not np.intersect1d(rows0,rows1).size,"Endpoint role/domain event leakage")
                g0,n0=np.unique(groups[rows0],return_counts=True);g1,n1=np.unique(groups[rows1],return_counts=True)
                same(g0,g1,"Random-label group support differs");same(n0,n1,"Within-group label quota differs")
                same(g0,np.unique(groups[role_pool]),"Role did not cover every fixed group")
            require(not np.intersect1d(groups[a["fit0"]],groups[a["test0"]]).size,"Endpoint fit/test original group overlap")
            rows=np.r_[a["test0"],a["test1"]];labels=np.r_[np.zeros(256),np.ones(256)].astype(np.int8)
            same(a["labels"],labels,"Stored label order differs");same(a["test_original_ids"],ids[rows],"Test IDs differ")
            same(a["test_original_groups"],groups[rows],"Test groups differ")
            same(a["fit0_original_ids"],ids[a["fit0"]],"Fit0 IDs differ");same(a["fit1_original_ids"],ids[a["fit1"]],"Fit1 IDs differ")
            fit=np.concatenate([baseline[a["fit0"]],baseline[a["fit1"]]]).astype(np.float32)
            same(state["center"],fit.mean(0,dtype=np.float64).astype(np.float32),"Scaler center is not fit-only")
            same(state["scale"],np.maximum(fit.std(0,dtype=np.float64),1/16383).astype(np.float32),"Scaler scale is not fit-only")
            scores=a["scores"];require(scores.shape==(512,) and np.isfinite(scores).all(),"Invalid stored scores")
            observed=_weighted_auc(labels,scores);close(observed,result["AUC"],"Saved AUC does not recompute")
            boot=a["bootstrap_AUC"];require(boot.shape==(1000,) and np.isfinite(boot).all() and ((boot>=0)&(boot<=1)).all(),"Invalid saved1000 bootstrap values")
            interval=np.quantile(boot,[.025,.975]);contains=bool(interval[0]<=.5<=interval[1])
            close(interval,result["bootstrap"]["percentile_95_interval"],"Bootstrap quantiles differ")
            close(boot.std(ddof=1),result["bootstrap"]["bootstrap_SE"],"Bootstrap SE differs")
            require(result["bootstrap"]["CI_contains_half"]==contains,"CI coverage flag differs")
            same(comparison_matrix(labels,scores,groups[rows]),a["bootstrap_comparison_matrix"],"Saved comparison matrix differs")
            close([a["leave_one_group_AUC"].min(),a["leave_one_group_AUC"].max()],result["bootstrap"]["leave_one_group_AUC_range"],"Leave-one-group range differs")
            perm=a["permutation_AUC"];require(perm.shape==(255,) and np.isfinite(perm).all() and ((perm>=0)&(perm<=1)).all(),"Invalid saved255 permutation values")
            tail=int(np.sum(abs(perm-.5)>=abs(observed-.5)));p=(tail+1)/256
            close(p,result["conditional_permutation"]["plus_one_p_value"],"Permutation +1 p differs")
            require(tail==result["conditional_permutation"]["tail_count"] and result["conditional_permutation"]["tail_rule"]==">= including ties","Permutation tail rule differs")
            replay=replay_scores(state,baseline[rows],target)
            exact=np.array_equal(scores,replay);error=float(abs(scores.astype(np.float64)-replay.astype(np.float64)).max())
            receipts.append(dict(repetition=i,score_rows=512,exact_score_equality=exact,max_absolute_score_error=error,
                checkpoint_sha256=sha256(directory/"classifier.pt"),audit_sha256=sha256(directory/"audit.npz"),
                result_sha256=sha256(directory/"result.json"),metrics_verified=True,model_parameters_unchanged=True))
            auc.append(observed);pvalues.append(p);intervals.append(interval);coverage.append(contains)
            np.add.at(fit_visits,np.r_[a["fit0"],a["fit1"]],1);np.add.at(test_visits,rows,1)
            progress=dict(status="replaying_fixed_classifier_endpoints",completed_checkpoints=i+1,total_checkpoints=128,
                score_rows_replayed=(i+1)*512,last_score_bitwise_equal=exact,last_maxerror=error,numeric_new_fits=0,updated_models=0)
            write_json(out/"status.json",progress);print(json.dumps(progress),flush=True)
        with np.load(rr/"ALL_REPETITIONS.npz",allow_pickle=False) as z:
            close(auc,z["AUC"],"Aggregate AUC differs");close(pvalues,z["permutation_p"],"Aggregate p differs")
            same(coverage,z["bootstrap_CI_contains_half"],"Aggregate coverage differs");close(intervals,z["bootstrap_CI"],"Aggregate intervals differ")
            same(fit_visits,z["fit_original_id_visit_count"],"Fit visit counts differ");same(test_visits,z["test_original_id_visit_count"],"Test visit counts differ")
        for key,k in (("legacy_joint_bootstrap_false_positive",int(np.sum(~np.asarray(coverage)))),
                      ("conditional_permutation_false_positive",int(np.sum(np.asarray(pvalues)<=.05)))):
            require(report[key]["rejections"]==k and report[key]["trials"]==128,"Aggregate rejection count differs")
            close(report[key]["rate"],k/128,"Aggregate FPR differs");close(report[key]["Wilson95"],wilson_interval(k,128),"Aggregate Wilson interval differs")
        close(report["null_AUC_quantiles"],np.quantile(auc,report["null_quantile_probabilities"]),"Null quantiles differ")
        passed=all(item["exact_score_equality"] for item in receipts)
        qa=dict(status="complete" if passed else "complete_with_score_replay_mismatch",passed=passed,
            classifier_checkpoints_verified=128,score_rows_replayed=65536,replay_batch_size=512,
            all_exact_score_equality=passed,maximum_absolute_score_error=max(v["max_absolute_score_error"] for v in receipts),
            all_saved_numeric_metrics_verified=True,updated_models=0,numeric_new_fits=0,new_bootstrap_resamples=0,new_label_permutations=0,
            noise_training_updates=0,response_training_updates=0,old_gate_modified_by_verifier=False,
            old_gate_scientific_artifact_claim_unmodified=report["old_noise_gate_modified"] is False,
            old_gate_scope="no old gate written;artifact claim/source binding checked,not independent reread of remote old gate",
            original_VAL_waveforms_decoded=False,container_array_keys_decoded=["target"],device=str(target),
            input_bindings=dict(config_sha256=sha256(config_path),container_sha256=sha256(container),identity_sha256=sha256(identities),
                               RR_report_sha256=sha256(report_path),verifier_sha256=sha256(__file__),sources=binding["dependencies_sha256"]),
            elapsed_s=time.monotonic()-start,checkpoints=receipts,independent_acquisition_validation=False)
        write_json(out/"REPLAY_QA.json",_clean(qa));write_json(out/"status.json",dict(status=qa["status"],passed=passed,score_rows_replayed=65536,updated_models=0,numeric_new_fits=0))
        return qa
    except BaseException as error:
        write_json(out/"FAILED.json",dict(exception=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            checkpoints_replayed=len(receipts),score_rows_replayed=len(receipts)*512,updated_models=0,numeric_new_fits=0))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("root","container","identities","config","out"):parser.add_argument("--"+key,required=True,type=Path)
    parser.add_argument("--device",default="cuda",choices=("cuda","cpu"))
    args=parser.parse_args(argv)
    result=verify_results(args.root,args.container,args.identities,args.config,args.device,args.out)
    return 0 if result["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
