"""Fixed repair_002 endpoint evaluation; reused development VAL, no selection.

Scientific data are opened only inside evaluate_repair_checkpoint. The v1
model/evaluator remain intact. Evaluation never loads checkpoint loss_state.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import charge_np,load_training,load_validation,sha256,write_json
from .electronics_v2 import make_n01_response_v2
from .evaluate import (EVALUATION_SEED,GENERATION_BATCH,MARGINAL_DRAWS,REPEAT_PARENTS,REPEAT_DRAWS,
    _clean,_generate,_observables,_population_summary,_conditional_peaks,_repeat_receipt,
    _classifier_evaluation,_waveform_curves,_distance_receipt,_split_groups,_summary)
from .losses_v2 import create_evaluation_context
from .noise_validation import balanced_real_domains


EXPERIMENT_ID="N01_REPAIR_002"
MATCHED_N=256
MATCHED_DISTANCE_SEED=2026101251
MATCHED_CLASSIFIER_SEED=2026101203
FIT_PER_CLASS=2048
TEST_PER_CLASS=256
WINDOW_VIEWS=("full","baseline_raw","baseline_demean","baseline_detrend","pulse_demean","pulse_shape")
ROOT=Path(__file__).resolve().parents[2]


def _state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        v=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


def validate_repair_checkpoint(value):
    required={"config","model","model_statistics","loss_statistics","noise_initialization",
              "electronics_state_sha256","noise_gate_sha256","seed","variant","arm","update"}
    if not required<=set(value):
        raise ValueError(f"Repair endpoint checkpoint missing {sorted(required-set(value))}; v1 checkpoints are unsupported")
    if value.get("schema_version",0)<2 or value.get("loss_version")!=2:
        raise ValueError("Repair evaluator requires schema>=2 and loss_version=2; legacy endpoints are rejected")
    if value["variant"] not in ("full","deterministic") or value["arm"]!=value["variant"]:
        raise ValueError("Only fixed full/deterministic repair arms are supported")
    if value["update"]!=19000 or value["config"]["response"]["updates_per_arm"]!=19000:
        raise ValueError("Repair evaluation requires the declared fixed19000 endpoint, not an interim checkpoint")
    if value.get("learn_population",False):
        raise ValueError("Repair comparison requires uniform count weights without population learning")
    if "population" in value and not bool((value["population"]["theta"]==0).all()):
        raise ValueError("Saved population theta is nonzero; uniform comparison cannot be assumed")
    for key in ("electronics_state_sha256","noise_gate_sha256"):
        digest=value[key]
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(f"Invalid {key}")
    if value["config"]["projection_seed"]==value["config"]["response"]["evaluation_projection_seed"]:
        raise ValueError("Frozen evaluation projection seed must be independent of training")
    return value


def validate_frozen_inputs(checkpoint,value,container_hash,identity_hash):
    """Verify registered data and a real passed frozen gate before data loading."""
    cfg=value["config"]
    if container_hash!=cfg["train_container_sha256"] or identity_hash!=cfg["identities_sha256"]:
        raise ValueError("Endpoint data differ from registered checkpoint configuration")
    path=Path(checkpoint).parent.parent/"noise_validation/NOISE_GATE.json"
    if not path.is_file() or sha256(path)!=value["noise_gate_sha256"]:
        raise ValueError("Frozen noise gate file missing or hash mismatch")
    gate=json.loads(path.read_text(encoding="utf8"))
    if gate.get("passed") is not True:
        raise ValueError("Noise gate did not pass; response endpoint evaluation is prohibited")
    if gate.get("electronics_state_sha256")!=value["electronics_state_sha256"]:
        raise ValueError("Noise gate and response checkpoint electronics differ")
    return dict(path=str(path),sha256=sha256(path),passed=True,electronics_state_sha256=gate["electronics_state_sha256"])


def restore_repair_model(checkpoint,device="cpu"):
    value=validate_repair_checkpoint(checkpoint)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EVALUATION_SEED)
        model=make_n01_response_v2(value["model_statistics"],value["variant"],
            electronics_initialization=value["noise_initialization"],stage="low_learning_rate")
    model.load_state_dict(value["model"],strict=True)
    if _state_digest(model.electronics.state_dict())!=value["electronics_state_sha256"].lower():
        raise ValueError("Restored electronics state differs from validated frozen candidate")
    return model.eval().requires_grad_(False).to(device)


def make_evaluation_objective(checkpoint):
    """New TRAIN-scaled objective plus independent bank; never load loss_state."""
    from .train_repair import BoundLossV2
    cfg=checkpoint["config"]
    objective=BoundLossV2(checkpoint["loss_statistics"],seed=int(cfg["projection_seed"]),
                         evaluation_seed=int(cfg["response"]["evaluation_projection_seed"]))
    context=create_evaluation_context(int(cfg["response"]["evaluation_projection_seed"]))
    objective.bind(context)
    return objective,context


@torch.no_grad()
def distance_receipt(objective,fake,real):
    result=_distance_receipt(objective,fake,real,np.ones(len(fake)))
    result["wave_subcomponents"]={key:float(value) for key,value in objective.last_wave_components.items()}
    result["evaluation_projection_context"]=objective.bound_projection_context.metadata()
    result["uniform_comparison_weights"]=True
    result["unequal_sample_counts"]=len(fake)!=len(real)
    result["no_scientific_pass_threshold"]=True
    return result


def equal_sample_selection(fake_ids,real_groups,n=MATCHED_N,seed=MATCHED_DISTANCE_SEED):
    """One saved draw per uniformly sampled parent, and balanced same-group RR."""
    ids=np.asarray(fake_ids)
    groups=np.asarray(real_groups)
    unique=np.unique(ids)
    if len(unique)<n:raise ValueError("Insufficient unique source parents for fixed equal-N comparison")
    rng=np.random.default_rng(seed)
    parents=rng.choice(unique,n,replace=False)
    fake_rows=np.asarray([rng.choice(np.flatnonzero(ids==parent)) for parent in parents],dtype=np.int64)
    real0,real1=balanced_real_domains(groups,n,seed+1)
    if np.unique(ids[fake_rows]).size!=n or np.intersect1d(real0,real1).size:
        raise AssertionError("Equal sample comparison reused parent or REAL event")
    a,b=np.unique(groups[real0],return_counts=True),np.unique(groups[real1],return_counts=True)
    if not np.array_equal(a[0],b[0]) or not np.array_equal(a[1],b[1]):
        raise AssertionError("REAL-REAL group quotas differ")
    return dict(fake_rows=fake_rows,fake_parent_ids=ids[fake_rows],real0_rows=real0,real1_rows=real1,
                REAL_quota_group=a[0],REAL_quota_per_domain=a[1],seed=np.array(seed))


def _window_helpers():
    path=ROOT/"scripts/diagnose_n01_windows.py"
    spec=importlib.util.spec_from_file_location("n01_repair_window_helpers",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.SEED!=MATCHED_CLASSIFIER_SEED or module.UPDATES!=200:
        raise ValueError("Published window-classifier fixed rules changed")
    return module


def matched_classifier_selection(train_source_ids,val_fake_ids,train_real_groups,val_real_groups):
    """TRAIN2048/VAL256 per class, with paired balanced random-label RR."""
    helpers=_window_helpers()
    train_source_ids,val_fake_ids=np.asarray(train_source_ids),np.asarray(val_fake_ids)
    train_real_groups,val_real_groups=np.asarray(train_real_groups),np.asarray(val_real_groups)
    if len(np.unique(train_source_ids))!=len(train_source_ids):raise ValueError("Nonunique SOURCE TRAIN parents")
    if np.intersect1d(train_source_ids,val_fake_ids).size:raise ValueError("SOURCE TRAIN/VAL parent overlap")
    if np.intersect1d(train_real_groups,val_real_groups).size:raise ValueError("REAL TRAIN/VAL group overlap")
    if len(train_source_ids)<FIT_PER_CLASS or len(np.unique(val_fake_ids))<TEST_PER_CLASS:
        raise ValueError("Insufficient prespecified SOURCE parent support")
    rng=np.random.default_rng(MATCHED_CLASSIFIER_SEED)
    train_source_rows=rng.choice(len(train_source_ids),FIT_PER_CLASS,replace=False)
    val_fake_rows=helpers.choose_rows(np.arange(len(val_fake_ids)),TEST_PER_CLASS,rng,val_fake_ids)
    tr0,tr1=balanced_real_domains(train_real_groups,FIT_PER_CLASS,MATCHED_CLASSIFIER_SEED+10)
    vr0,vr1=balanced_real_domains(val_real_groups,TEST_PER_CLASS,MATCHED_CLASSIFIER_SEED+11)
    real_count=len(train_real_groups)
    source=dict(domain0="SOURCE",fit0=np.arange(FIT_PER_CLASS),test0=FIT_PER_CLASS+np.arange(TEST_PER_CLASS),
                fit1=tr0,test1=real_count+vr0)
    rr=dict(domain0="REAL",fit0=tr0,fit1=tr1,test0=real_count+vr0,test1=real_count+vr1)
    selected=dict(source_train_rows=train_source_rows,source_train_ids=train_source_ids[train_source_rows],
        source_VAL_saved_rows=val_fake_rows,source_VAL_ids=val_fake_ids[val_fake_rows],
        real_train0_rows=tr0,real_train1_rows=tr1,real_VAL0_rows=vr0,real_VAL1_rows=vr1)
    for groups,r0,r1 in ((train_real_groups,tr0,tr1),(val_real_groups,vr0,vr1)):
        if np.intersect1d(r0,r1).size:raise AssertionError("RR event reuse")
        u,c=np.unique(groups[r0],return_counts=True);v,d=np.unique(groups[r1],return_counts=True)
        if not np.array_equal(u,np.unique(groups)) or not np.array_equal(u,v) or not np.array_equal(c,d):
            raise AssertionError("Prespecified RR quotas do not cover/match every group")
    if np.unique(selected["source_VAL_ids"]).size!=TEST_PER_CLASS:raise AssertionError("More than one saved draw per test parent")
    return source,rr,selected


def annotate_matched_uncertainty(report,real_test_groups):
    uncertainty=report["uncertainty"]
    uncertainty["legacy_helper_limited_group_count"]=uncertainty.get("limited_group_count")
    uncertainty["limited_group_count"]=True
    uncertainty["limited_original_group_count"]=True
    uncertainty["REAL_original_test_groups"]=int(real_test_groups)
    uncertainty["scope"]="one_previously_viewed_acquisition_development_reused_VAL"
    uncertainty["independent_acquisition_uncertainty"]=False
    report["limited_original_group_count"]=True
    return report


def _matched_classifiers(model,train,fake,real,ids,groups,device,out):
    helpers=_window_helpers()
    case,rr,selection=matched_classifier_selection(train["source_ids"],ids,train["real_groups"],groups)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"fixed_splits.npz",**selection,
        **{prefix+k:v for prefix,c in (("SOURCE_REAL_",case),("RR_",rr)) for k,v in c.items() if k!="domain0"})
    source_rows=selection["source_train_rows"]
    train_generated=_generate(model,train["source"],source_rows,1,EVALUATION_SEED+61,device)
    ftrain=train_generated["waveform"]
    fval=fake[selection["source_VAL_saved_rows"]]
    combined_fake=np.concatenate((ftrain,fval))
    combined_ids=np.r_[selection["source_train_ids"],selection["source_VAL_ids"]]
    combined_real=np.concatenate((train["real"],real))
    combined_groups=np.r_[train["real_groups"],groups]
    np.savez_compressed(out/"classifier_populations.npz",generated_TRAIN=ftrain,generated_VAL=fval,
        SOURCE_TRAIN_parent_ids=selection["source_train_ids"],SOURCE_VAL_parent_ids=selection["source_VAL_ids"],
        SOURCE_TRAIN_waveform=train["source"][source_rows],
        REAL_TRAIN0=train["real"][selection["real_train0_rows"]],REAL_TRAIN1=train["real"][selection["real_train1_rows"]],
        REAL_VAL0=real[selection["real_VAL0_rows"]],REAL_VAL1=real[selection["real_VAL1_rows"]],
        REAL_TRAIN0_groups=train["real_groups"][selection["real_train0_rows"]],REAL_TRAIN1_groups=train["real_groups"][selection["real_train1_rows"]],
        REAL_VAL0_groups=groups[selection["real_VAL0_rows"]],REAL_VAL1_groups=groups[selection["real_VAL1_rows"]])
    results={}
    for view in WINDOW_VIEWS:
        results[view]={}
        f,r=helpers.view_array(combined_fake,view),helpers.view_array(combined_real,view)
        results[view]["SOURCE_REAL"]=helpers.fit_classifier(f,r,combined_ids,combined_groups,case,view,device,out/view/"SOURCE_REAL",False)
        results[view]["REAL_REAL_within_group_random_labels"]=helpers.fit_classifier(r,r,combined_groups,combined_groups,rr,view,device,out/view/"REAL_REAL",True)
        for name,folder in (("SOURCE_REAL","SOURCE_REAL"),("REAL_REAL_within_group_random_labels","REAL_REAL")):
            annotate_matched_uncertainty(results[view][name],len(np.unique(groups)))
            write_json(out/view/folder/"result.json",_clean(results[view][name]))
    return dict(status="fixed_matched_TRAIN_fit_VAL_test_classifiers_complete",views=results,
        fit_per_class=FIT_PER_CLASS,test_per_class=TEST_PER_CLASS,updates_per_classifier=200,
        classifiers=12,seed=MATCHED_CLASSIFIER_SEED,one_generated_draw_per_SOURCE_parent=True,
        generated_TRAIN_noise_stream=noise_stream_receipt(model,2048,EVALUATION_SEED+61),
        SOURCE_fit_test_parent_overlap=0,REAL_fit_test_group_overlap=0,
        all_REAL_TRAIN_and_VAL_groups_covered=True,
        REAL_TRAIN_groups=len(np.unique(train["real_groups"])),REAL_VAL_groups=len(np.unique(groups)),
        REAL_events_and_group_quota_matched_to_control=True,
        control="disjoint_random_labels_within_every_original_group;TRAIN_fit_VAL_test",
        independent_acquisition_validation=False,
        fit_role="legal_TRAIN_waveforms_only_for_independent_C2ST;not_loss_statistics_or_model_refitting",
        inference="development_reused_VAL;no_architecture_count_seed_or_checkpoint_selection")


def allrow_observables(waveform,q_floor):
    observed=_observables(waveform,q_floor)
    y=np.asarray(waveform,dtype=np.float64)
    q=observed["Q"]
    tail=y[:,150:400].sum(1)-2.5*y[:,:100].sum(1)
    raw=np.full(len(y),np.nan,dtype=np.float64)
    np.divide(tail,q,out=raw,where=q!=0)
    return observed|dict(PSD_raw=raw,PSD_raw_nonfinite=~np.isfinite(raw),Q_exact_zero=q==0,
                         baseline_mean_V=y[:,:100].mean(1))


def _raw_PSD_receipt(observed):
    x=observed["PSD_raw"]
    return dict(rows=len(x),nonfinite_rows=int((~np.isfinite(x)).sum()),nonpositive_Q_rows=int(observed["Q_nonpositive"].sum()),
        mean=float(x.mean()) if np.isfinite(x).all() else None,
        std_ddof1=float(x.std(ddof=1)) if np.isfinite(x).all() and len(x)>1 else None,
        all_rows_preserved=True,finite_subset_moments_not_reported=True,
        interpretation="negative_or_nearzero_charge_ratios_are_saved_but_not_physical_PSD")


def noise_stream_receipt(model,count,seed,save_path=None):
    """Fingerprint latent values in the exact CPU/batch128 generation order."""
    g=torch.Generator(device="cpu").manual_seed(seed)
    h=hashlib.sha256();saved={}
    initial=g.get_state().numpy()
    for start in range(0,count,GENERATION_BATCH):
        noise=model.sample_noise(min(count-start,GENERATION_BATCH),device="cpu",generator=g)
        for key,v in noise.items():
            h.update(key.encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
            if save_path is not None:saved.setdefault(key,[]).append(v.numpy())
    if save_path is not None:
        np.savez_compressed(save_path,seed=np.array(seed),batch=np.array(GENERATION_BATCH),CPU_rng_initial=initial,
            CPU_rng_final=g.get_state().numpy(),**{k:np.concatenate(v) for k,v in saved.items()})
    return dict(seed=seed,CPU_generator=True,generation_batch=GENERATION_BATCH,draws=count,
                noise_tensor_stream_sha256=h.hexdigest(),key_shapes=model.noise_shapes)


def _input_dependency_repair(model,source,ids,rows,q_floor,stats,device,out):
    selected=rows[:8];draws=32;seed=EVALUATION_SEED+31
    original=source[selected].astype(np.float64);baseline=original[:,:100].mean(1,keepdims=True)
    reference=source[0].astype(np.float64);reference-=reference[:100].mean()
    template=reference/charge_np(reference[None])[0]
    variants={"original":original,"common_normalized_shape_same_Q":baseline+charge_np(original)[:,None]*template,
              "same_shape_scale_1p2":baseline+1.2*(original-baseline),"same_shape_scale_0p8":baseline+.8*(original-baseline)}
    numeric=dict(source_parent_ids=ids[selected],reference_shape_parent_id=ids[0],explicit_noise_seed=np.array(seed),draws=np.array(draws))
    generations={};observed={};results={}
    low,high=np.exp(stats["log_q_min"]),np.exp(stats["log_q_max"])
    for name,parents in variants.items():
        generations[name]=_generate(model,parents.astype(np.float32),np.arange(len(selected)),draws,seed,device)
        observed[name]=allrow_observables(generations[name]["waveform"],q_floor)
        numeric[name+"_source_waveform"]=parents
        for key,value in generations[name].items():
            if key!="parent_row":numeric[name+"_generated_"+key]=value.reshape((len(selected),draws)+value.shape[1:])
        for key,value in observed[name].items():numeric[name+"_"+key]=value.reshape(len(selected),draws)
        q=charge_np(parents)
        results[name]=dict(input_Q=q,input_Q_below_TRAIN=q<low,input_Q_above_TRAIN=q>high,
            input_train_charge_range_Vns=[low,high],input_voltage_change_RMS_V=np.sqrt(((parents-original)**2).mean(1)),
            output_raw_PSD=_raw_PSD_receipt(observed[name]),
            output_guarded_PSD_rows=int(observed[name]["PSD_guarded"].sum()),
            saturated_draws=int((generations[name]["saturation_fraction_per_event"]>0).sum()))
        if name!="original":
            difference=(generations[name]["waveform"]-generations["original"]["waveform"]).reshape(len(selected),draws,480)
            results[name]["mean_voltage_change_RMS_V"]=np.sqrt((difference.mean(1)**2).mean(1))
            results[name]["electronics_charge_same_noise_max_abs_difference"]=float(np.max(abs(generations[name]["electronics_charge"]-generations["original"]["electronics_charge"])))
            results[name]["paired_changes"]={}
            for feature in ("Q","PSD","baseline_mean_V","relative_half_rise_ns","relative_peak_ns"):
                delta=(observed[name][feature]-observed["original"][feature]).reshape(len(selected),draws)
                numeric[name+"_"+feature+"_paired_difference"]=delta
                results[name]["paired_changes"][feature]=dict(parent_mean=np.mean(delta,axis=1),parent_SE=np.std(delta,axis=1,ddof=1)/np.sqrt(draws),nonfinite_rows=int((~np.isfinite(delta)).sum()))
    np.savez_compressed(out/"same_noise_input_interventions.npz",**numeric)
    stream=noise_stream_receipt(model,len(selected)*draws,seed,out/"same_noise_input_latents.npz")
    return dict(parents=len(selected),draws_per_parent=draws,parent_ids=ids[selected],paired_latents_identical=True,
                noise_stream=stream,changing_input_tests=results,ID_as_network_input=False,
                all_rows_retained=True,extrapolation_flagged=True,interpretation="paired development probes;out-of-TRAIN charge inputs are extrapolation,not physical success evidence")


def _artifact_table(out):
    excluded={"endpoint_report.json","EVALUATION_COMPLETE.json","ARTIFACT_MANIFEST.json"}
    return {str(p.relative_to(out)).replace("\\","/"):dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p))
            for p in sorted(out.rglob("*")) if p.is_file() and p.name not in excluded}


def _verify_completion(out,checkpoint_hash,container_hash,identity_hash):
    report_path=out/"endpoint_report.json";marker=out/"EVALUATION_COMPLETE.json"
    if not report_path.exists() or not marker.exists():raise RuntimeError("Preserve incomplete evaluation; choose a fresh E/D output directory")
    report=json.loads(report_path.read_text(encoding="utf8"));complete=json.loads(marker.read_text(encoding="utf8"))
    for key,expected in (("checkpoint_sha256",checkpoint_hash),("container_sha256",container_hash),("identity_sha256",identity_hash)):
        if report.get(key)!=expected:raise FileExistsError("Existing endpoint evaluation has different frozen inputs")
    if report.get("experiment_id")!=EXPERIMENT_ID or complete.get("report_sha256")!=sha256(report_path):raise ValueError("Repair report binding/hash mismatch")
    gate_receipt=report.get("noise_gate")
    if not gate_receipt or not Path(gate_receipt["path"]).is_file() or sha256(gate_receipt["path"])!=report["noise_gate_sha256"]:
        raise ValueError("Frozen completed evaluation noise gate changed or is missing")
    gate=json.loads(Path(gate_receipt["path"]).read_text(encoding="utf8"))
    if gate.get("passed") is not True or gate.get("electronics_state_sha256")!=report["electronics_state_sha256"]:
        raise ValueError("Completed evaluation has no matching passed noise gate")
    for value in complete["artifacts"].values():
        if not Path(value["path"]).exists() or sha256(value["path"])!=value["sha256"]:raise ValueError("Frozen evaluation artifact changed")
    return report


def summarize_repair_comparison(run_root):
    root=Path(run_root);reports={}
    if os.name=="nt" and root.resolve().drive.upper() not in ("E:","D:"):
        raise ValueError("Repair comparison outputs must stay on E/D")
    for arm in ("deterministic","full"):
        out=root/arm/"evaluation"
        if (out/"EVALUATION_COMPLETE.json").exists():
            value=json.loads((out/"endpoint_report.json").read_text(encoding="utf8"))
            reports[arm]=_verify_completion(out,value["checkpoint_sha256"],value["container_sha256"],value["identity_sha256"])
    if len(reports)==2:
        for key in ("container_sha256","identity_sha256","electronics_state_sha256","noise_gate_sha256","evaluation_projection_context","generation_streams","equal_sample_selection_sha256"):
            if reports["full"][key]!=reports["deterministic"][key]:raise ValueError("Repair arm common binding differs: "+key)
    rows=[]
    for arm in ("deterministic","full"):
        r=reports.get(arm)
        if r is None:rows.append(dict(arm=arm,status="pending_fixed_endpoint"));continue
        rows.append(dict(arm=arm,status="descriptive_completed_fixed_endpoint",checkpoint_sha256=r["checkpoint_sha256"],
            raw_distances=r["raw_population"]["distribution_distances"],equal_N_distances=r["controls"]["equal_N256"],
            raw_ROC_AUC=r["independent_domain_classifier"].get("raw_ROC_AUC"),matched_C2ST=r["matched_independent_classifiers"],
            repeated_variance_Q=r["repeated_parent_response"]["conditional_variance_Q"],
            repeated_variance_PSD=r["repeated_parent_response"]["conditional_variance_PSD"],
            nonpositive_Q_rows=r["raw_population"]["nonpositive_Q_rows"],saturated_marginal_draws=r["generated_saturated_draws"]))
    result=dict(experiment_id=EXPERIMENT_ID,arms=rows,all_required_arms_complete=len(reports)==2,
        common_electronics_and_latents_verified=len(reports)==2,weights="uniform_ones",validation_role="development_reused_VAL",
        independent_acquisition_validation=False,scientific_pass_claimed=False)
    write_json(root/"ARM_COMPARISON.json",_clean(result));return result


def evaluate_repair_checkpoint(checkpoint,container,identities,out,device="cuda"):
    checkpoint,container,identities,out=map(Path,(checkpoint,container,identities,out))
    if os.name=="nt" and out.resolve().drive.upper() not in ("E:","D:"):raise ValueError("All repair outputs must be on E/D")
    checkpoint_hash,container_hash,identity_hash=sha256(checkpoint),sha256(container),sha256(identities)
    value=validate_repair_checkpoint(torch.load(checkpoint,map_location="cpu",weights_only=False))
    gate_receipt=validate_frozen_inputs(checkpoint,value,container_hash,identity_hash)
    if out.exists() and any(out.iterdir()):
        prior=_verify_completion(out,checkpoint_hash,container_hash,identity_hash)
        summarize_repair_comparison(out.parent.parent);return prior
    out.mkdir(parents=True,exist_ok=True)
    sources_before={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/"src/n01").glob("*.py"))}
    sources_before["scripts/diagnose_n01_windows.py"]=sha256(ROOT/"scripts/diagnose_n01_windows.py")
    target=torch.device(device)
    if target.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    model=restore_repair_model(value,target);initial_state=_state_digest(model.state_dict())
    objective,context=make_evaluation_objective(value)
    torch.save(context.state_dict(),out/"evaluation_projection_bank.pt")
    data=load_validation(container,identities)
    source,real,ids,real_ids,groups=(data[k] for k in ("source","real","source_ids","real_ids","real_groups"))
    if len(np.unique(ids))!=512 or len(np.unique(real_ids))!=599:raise ValueError("Original validation identities are not unique")
    q_floor,q_reference=float(objective.q_floor),float(objective.q_reference)
    generated=_generate(model,source,np.arange(512),4,EVALUATION_SEED+1,target)
    fake=generated["waveform"];fake_ids=ids[generated["parent_row"]]
    if len(fake)!=2048 or any(np.sum(fake_ids==i)!=4 for i in ids):raise AssertionError("Fixed marginal parent counts changed")
    obs,real_obs=allrow_observables(fake,q_floor),allrow_observables(real,q_floor)
    # Extra raw PSD flags are archived separately: legacy summaries retain their
    # signed guarded all-row population definition and explicit invalid counts.
    raw=_population_summary(_observables(fake,q_floor),fake,None,objective,fake_ids,EVALUATION_SEED+2)
    real_summary=_population_summary(_observables(real,q_floor),real,None,objective,groups,EVALUATION_SEED+3)
    raw["distribution_distances"]=distance_receipt(objective,fake,real)
    raw["raw_PSD_diagnostic"]=_raw_PSD_receipt(obs);real_summary["raw_PSD_diagnostic"]=_raw_PSD_receipt(real_obs)
    raw["baseline_mean_V"]=_summary(obs["baseline_mean_V"]);real_summary["baseline_mean_V"]=_summary(real_obs["baseline_mean_V"])
    conditional=_conditional_peaks(obs,real_obs,q_reference,q_floor,objective.charge_bin_boundaries.numpy(),np.ones(2048),fake_ids)
    numeric=dict(source_parent_ids=ids,source_parent_count_weights=np.ones(512),source_parent_comparison_weights=np.ones(512),
        source_h=data["h"],real_original_ids=real_ids,real_original_groups=groups,generated_source_parent_ids=fake_ids,
        generated_comparison_weights=np.ones(2048),REAL_waveform=real,generated_waveform=fake)
    numeric.update({"generated_"+k:v for k,v in generated.items() if k!="waveform"})
    numeric.update({"generated_"+k:v for k,v in obs.items()});numeric.update({"REAL_"+k:v for k,v in real_obs.items()})
    for name,y in (("raw",fake),("REAL",real)):
        numeric.update({name+"_"+k:v for k,v in _waveform_curves(y).items()})
    np.savez_compressed(out/"endpoint_populations.npz",**numeric)
    selection=equal_sample_selection(fake_ids,groups)
    np.savez_compressed(out/"equal_sample_selection.npz",**selection)
    # Hash selection array values, rather than ZIP timestamps, for cross-arm equality.
    selection_digest=hashlib.sha256(b"".join(k.encode()+np.ascontiguousarray(v).tobytes() for k,v in sorted(selection.items()))).hexdigest()
    f,r0,r1=selection["fake_rows"],selection["real0_rows"],selection["real1_rows"]
    controls=dict(equal_N256=dict(status="equal_N_one_SOURCE_draw_per_parent_same_group_quota_REAL_REAL",
        N_per_population=256,generated_REAL=distance_receipt(objective,fake[f],real[r0]),
        REAL_REAL=distance_receipt(objective,real[r0],real[r1]),
        no_direct_hard_pass_rule=True,comparison_note="both use N256 and the same frozen independent directions;one-acquisition development control"))
    def adc_receipt(y):
        a=np.rint((np.clip(y,-.5,.5)+.5)*16383)/16383-.5
        b=np.rint((a+.5)*16383)/16383-.5
        return dict(rows=len(y),canonicalization_max_abs_change=float(abs(a-y).max()),second_application_idempotent=bool(np.array_equal(a,b)))
    controls["common_ADC"]=dict(REAL=adc_receipt(real),generated=adc_receipt(fake),source_input_preclipping=False)
    legacy_classifier=_classifier_evaluation(fake,real,fake_ids,groups,np.ones(2048),value["loss_statistics"],target,out)
    # The additional legal TRAIN data are used solely to fit the independent
    # matched C2ST; TRAIN loss/model scales remain the checkpoint's fixed values.
    classifier_train=load_training(container,identities)
    matched=_matched_classifiers(model,classifier_train,fake,real,fake_ids,groups,target,out/"matched_classifiers")
    del classifier_train
    selected=np.sort(np.random.default_rng(EVALUATION_SEED+20).choice(512,128,replace=False))
    repeated=_generate(model,source,selected,64,EVALUATION_SEED+22,target)
    repeat=_repeat_receipt(repeated,source,selected,ids[selected],q_floor,out)
    repeat_obs=allrow_observables(repeated["waveform"],q_floor)
    with np.load(out/"repeated_parent_responses.npz",allow_pickle=False) as z:repeat_arrays={k:z[k].copy() for k in z.files}
    for k in ("PSD_raw","PSD_raw_nonfinite","Q_nonpositive","Q_exact_zero","PSD_guarded","baseline_mean_V"):
        repeat_arrays[k]=repeat_obs[k].reshape(128,64)
    repeat_arrays["charge_identity_residual"]=repeated["charge_identity_residual"].reshape(128,64)
    repeat_arrays["pre_adc_charge"]=repeated["pre_adc_charge"].reshape(128,64)
    np.savez_compressed(out/"repeated_parent_responses.npz",**repeat_arrays)
    repeat.update(raw_PSD_diagnostic=_raw_PSD_receipt(repeat_obs),variance_ddof=1,
                  saturated_draws=int((repeated["saturation_fraction_per_event"]>0).sum()))
    dependency=_input_dependency_repair(model,source,ids,selected,q_floor,value["model_statistics"],target,out)
    streams=dict(marginal=noise_stream_receipt(model,2048,EVALUATION_SEED+1),
                 repeated=noise_stream_receipt(model,8192,EVALUATION_SEED+22),input_dependency=dependency["noise_stream"],
                 matched_classifier_TRAIN=matched["generated_TRAIN_noise_stream"])
    if initial_state!=_state_digest(model.state_dict()):raise RuntimeError("Frozen model changed during evaluation")
    context.validate()
    if any(sha256(ROOT/name)!=digest for name,digest in sources_before.items()):
        raise RuntimeError("Bound evaluator/model/loss/helper source changed during endpoint evaluation")
    write_json(out/"evaluation_binding.json",dict(checkpoint_sha256=checkpoint_hash,container_sha256=container_hash,
        identity_sha256=identity_hash,noise_gate=gate_receipt,source_hashes=sources_before,
        evaluation_projection_context=context.metadata(),generation_streams=streams,
        matched_classifier_fit_per_class=2048,matched_classifier_test_per_class=256,
        matched_classifier_selection_scope="TRAIN_original_parents_fit_VAL_original_parents_groups_test",
        validation_role="development_reused_VAL",model_state_unchanged=True))
    artifacts=_artifact_table(out)
    write_json(out/"ARTIFACT_MANIFEST.json",dict(experiment_id=EXPERIMENT_ID,checkpoint_sha256=checkpoint_hash,
        container_sha256=container_hash,identity_sha256=identity_hash,artifacts=artifacts))
    path=out/"ARTIFACT_MANIFEST.json";artifacts[path.name]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path))
    report=dict(schema_version=2,experiment_id=EXPERIMENT_ID,arm=value["arm"],variant=value["variant"],update=19000,
        training_seed=value["seed"],evaluation_seed=EVALUATION_SEED,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,
        container=str(container),container_sha256=container_hash,identity_path=str(identities),identity_sha256=identity_hash,
        electronics_state_sha256=value["electronics_state_sha256"],noise_gate_sha256=value["noise_gate_sha256"],noise_gate=gate_receipt,
        evaluation_projection_context=context.metadata(),training_loss_state_loaded=False,generation_streams=streams,
        equal_sample_selection_sha256=selection_digest,output_dir=str(out),device=str(target),maximum_generation_batch=128,
        validation_role="development_reused_VAL_previously_used_for_mechanism_diagnostics",
        validation_source_parents=512,validation_REAL_original_parents=599,marginal_draws_per_source_parent=4,
        population_weights=dict(mode="uniform_ones",learned=False),raw_population=raw,comparison_weighted_population=raw,
        REAL_population=real_summary,conditional_PSD_peaks_and_original_widths=conditional,controls=controls,
        independent_domain_classifier=legacy_classifier,legacy_classifier_role="descriptive_continuity_VAL_internal_original_group_split",
        matched_independent_classifiers=matched,primary_classifier_role="independent_TRAIN2048_fit_VAL256_test_all_REAL_groups",
        repeated_parent_response=repeat,same_noise_input_dependency=dependency,
        charge_identity_max_abs_Vns=float(abs(generated["charge_identity_residual"]).max()),
        generated_saturation_fraction_mean=float(generated["saturation_fraction_per_event"].mean()),
        generated_saturated_draws=int((generated["saturation_fraction_per_event"]>0).sum()),
        model_state_unchanged=True,scientific_training_updates=0,classifier_fit_updates=13*200,
        statistical_status="descriptive_fixed_endpoint_no_new_pulse_acceptance_threshold",
        physical_calibration="unknown",independent_acquisition_count=1,independent_acquisition_validation=False,
        true_particle_labels_available=False,physical_weights_status="unknown",original_parents_preserved=True,
        artifacts=artifacts,evaluator_source_sha256=sha256(Path(__file__)),window_helpers_sha256=sha256(ROOT/"scripts/diagnose_n01_windows.py"))
    report=_clean(report);write_json(out/"endpoint_report.json",report)
    write_json(out/"EVALUATION_COMPLETE.json",dict(status="complete",experiment_id=EXPERIMENT_ID,arm=value["arm"],
        completed_UTC=datetime.now(timezone.utc).isoformat(),report_path=str(out/"endpoint_report.json"),
        report_sha256=sha256(out/"endpoint_report.json"),checkpoint_sha256=checkpoint_hash,
        container_sha256=container_hash,identity_sha256=identity_hash,artifacts=artifacts))
    summarize_repair_comparison(out.parent.parent)
    return report
