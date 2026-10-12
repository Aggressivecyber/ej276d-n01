"""Prospective005 ACF20 coverage repair. New driver; preserved004 is immutable."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from . import train_noise_v3 as common
from .acf20 import ACF20Loss,mean_acf20
from .data import FullCoverageSampler,sha256,write_json
from .electronics_v3 import ObservedBaselineNoiseV3,fit_baseline_initialization_v3
from .losses_v2 import BaselineDistributionLoss,fit_baseline_statistics
from .model import digitize
from .noise_validation import baseline_statistics

ROOT=Path(__file__).resolve().parents[2]
CONFIG_SHA="8eb87e705ac4abb4864a919194070c573cf39866e4ff694947965b87f3465e0a"
PROTOCOL_SHA="e7e855143fbbeb4a227779c6ab4bdef25838e2d7390e261f40a814effab940f5"
SOURCE_MANIFEST="NOISE_ACF005_SOURCE_MANIFEST.json"
REQUIRED_SOURCE=set(common.REQUIRED_SOURCE)|{"src/n01/train_noise_acf005.py","src/n01/acf20.py","src/n01/noise_validation_acf005.py"}


def array_digest(value):
    a=torch.as_tensor(value,dtype=torch.float64).detach().cpu().numpy().astype("<f8",copy=False)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def bind(config_path,manifest_path,*,root=ROOT,expected_config_sha=CONFIG_SHA,expected_protocol_sha=PROTOCOL_SHA):
    root=Path(root).resolve();config_path=Path(config_path)
    cfg=json.loads(config_path.read_text(encoding="utf-8"))
    manifest=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    digest=sha256(config_path)
    if digest!=expected_config_sha or manifest["config_sha256"]!=digest:raise ValueError("Frozen005 config mismatch")
    if cfg["run_id"]!="noise_acf005_20261012" or not cfg.get("registered_UTC"):raise ValueError("Unregistered005")
    if cfg["response_updates"]!=0 or cfg["response_integration_allowed"] or cfg["old_gates_changed"]:
        raise ValueError("No response integration/gate alteration permitted")
    if cfg["noise"]["updates"]!=5000 or cfg["model"]["length"]!=100:raise ValueError("Fixed5000/100scope required")
    if cfg["acf20"]["scale"]!=.015 or cfg["acf20"]["coefficient"]!=1 or cfg["acf20"]["lags"]!=list(range(1,21)):
        raise ValueError("Frozen ACF20 contract mismatch")
    if cfg["diagnostics"]["rows"]!=8192 or cfg["diagnostics"]["chunk_size"]!=256 or cfg["diagnostics"]["seed"]!=cfg["noise"]["generation_seed"]+20000:
        raise ValueError("Frozen independent diagnostic stream mismatch")
    if not REQUIRED_SOURCE<=set(manifest["files"]):raise ValueError("Missing executed005 source")
    for name,expected in manifest["files"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha256(path)!=expected:raise ValueError("Source SHA mismatch:"+name)
    for role in ("config","protocol","preregistration"):
        path=(root/manifest[role+"_path"]).resolve()
        if not path.is_relative_to(root) or sha256(path)!=manifest[role+"_sha256"]:raise ValueError("Document SHA mismatch:"+role)
    if manifest["protocol_sha256"]!=expected_protocol_sha:raise ValueError("Frozen005 protocol mismatch")
    for name,spec in cfg["references"].items():
        path=(root/spec["path"]).resolve()
        if not path.is_relative_to(root) or sha256(path)!=spec["sha256"]:raise ValueError("Reference SHA mismatch:"+name)
    bindings=dict(config_sha256=digest,source_manifest_sha256=sha256(manifest_path),
                  protocol_sha256=manifest["protocol_sha256"],preregistration_sha256=manifest["preregistration_sha256"],
                  source_files=manifest["files"],reference_hashes={k:v["sha256"] for k,v in cfg["references"].items()})
    return cfg,bindings


def gradient_pair(a,b):
    av=torch.cat([v.detach().reshape(-1).double() for v in a])
    bv=torch.cat([v.detach().reshape(-1).double() for v in b])
    na,nb=float(av.norm()),float(bv.norm());dot=float(av@bv)
    cosine=angle=None
    if na>0 and nb>0:
        cosine=min(1.,max(-1.,dot/(na*nb)))
        angle=float(np.degrees(np.arccos(cosine)))
    return dict(dot=dot,cosine=cosine,angle_degrees=angle),na,nb


def differentiated_step(old,added,total,parameters):
    """Diagnostics do not assign/rescale gradients; actual total.backward is canonical."""
    g0=torch.autograd.grad(old,parameters,retain_graph=True,allow_unused=True)
    ga=torch.autograd.grad(added,parameters,retain_graph=True,allow_unused=True)
    g0=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,g0)]
    ga=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,ga)]
    total.backward()
    gt=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]
    pair,n0,na=gradient_pair(g0,ga);turn,_,nt=gradient_pair(g0,gt)
    difference=max(float((g0i+gai-gti).abs().max()) for g0i,gai,gti in zip(g0,ga,gt))
    return dict(old_vs_acf=pair,old_vs_total=turn,old_gradient_norm=n0,
                acf20_gradient_norm=na,total_gradient_preclip_norm=nt,
                gradient_sum_max_absolute_difference=difference)


def _token(value):
    return hashlib.sha256(json.dumps(common.clean(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def diagnostic_latents(model,cfg):
    device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(cfg["diagnostics"]["seed"])
    chunks=[]
    for _ in range(cfg["diagnostics"]["rows"]//cfg["diagnostics"]["chunk_size"]):
        noise=model.sample_noise(cfg["diagnostics"]["chunk_size"],device=device,generator=generator,length=100)
        chunks.append(dict(noise={k:v.detach().cpu().clone() for k,v in noise.items()}))
    return dict(seed=cfg["diagnostics"]["seed"],seed_rule="generation_seed+20000",
                rows=cfg["diagnostics"]["rows"],chunk_size=cfg["diagnostics"]["chunk_size"],chunks=chunks)


def generate_diagnostic(model,latents,*,training_rng=None,sampler=None,context=None):
    device=next(model.parameters()).device
    before=dict(global_rng=_token(common.global_rng_state(device)),model=common.state_digest(common.snapshot(model)),
                train_rng=array_digest(training_rng.get_state()) if training_rng is not None else None,
                sampler=_token(sampler.state_dict()) if sampler is not None else None,
                context=context.metadata() if context is not None else None)
    pre=[];post=[];was_training=model.training
    model.eval()
    try:
        with torch.no_grad():
            for chunk in latents["chunks"]:
                noise={k:v.to(device) for k,v in chunk["noise"].items()}
                y=model(noise,length=100);q=digitize(y,ste=False)
                pre.append(y[:,0].cpu().numpy().copy());post.append(q[:,0].cpu().numpy().copy())
    finally:model.train(was_training)
    after=dict(global_rng=_token(common.global_rng_state(device)),model=common.state_digest(common.snapshot(model)),
               train_rng=array_digest(training_rng.get_state()) if training_rng is not None else None,
               sampler=_token(sampler.state_dict()) if sampler is not None else None,
               context=context.metadata() if context is not None else None)
    if before!=after:raise RuntimeError("Diagnostic mutated model/optimizer stream/global RNG/context")
    if context is not None:context.validate()
    return np.concatenate(pre),np.concatenate(post)


def training_statistics(train):
    c=np.cov(train,rowvar=False,ddof=1);stats=baseline_statistics(train)
    acf=mean_acf20(torch.as_tensor(train,dtype=torch.float64)).detach().cpu()
    scale=np.maximum(train.std(0,ddof=1),1/16383)
    return dict(covariance_V2=c,OLS_covariance_V2=np.asarray(stats["ols_covariance_V2"]),
                timeSD_V=scale,target_acf20=acf,baseline_statistics=stats)


def population_summary(values,reference):
    stats=baseline_statistics(values);c=np.cov(values,rowvar=False,ddof=1)
    a=mean_acf20(torch.as_tensor(values,dtype=torch.float64)).detach().cpu().numpy()
    target=reference["target_acf20"].numpy();signed=a-target
    e=signed/.015;cf=reference["covariance_V2"]
    scale=reference["timeSD_V"];outer=scale[:,None]*scale[None,:]
    ols=np.asarray(stats["ols_covariance_V2"]);cr=reference["OLS_covariance_V2"]
    oscale=np.sqrt(np.diag(cr));oouter=oscale[:,None]*oscale[None,:]
    output=dict(rows=len(values),acf20_mean=a,acf20_signed_errors_vsTRAIN=signed,
                max_abs_acf20_error_vsTRAIN=float(abs(signed).max()),
                acf20_loss_vsTRAIN=float(.5*(np.mean(e**2)+np.max(e**2))),
                OLS_covariance_V2=ols,mean_slope_correlation=float(ols[0,1]/np.sqrt(ols[0,0]*ols[1,1])),
                TRAIN_scaled_OLS_relative_error=float(np.linalg.norm((ols-cr)/oouter)/np.linalg.norm(cr/oouter)),
                TRAIN_scaled_fullcov_relative_error=float(np.linalg.norm((c-cf)/outer)/np.linalg.norm(cf/outer)),
                time_mean_V=values.mean(0),timeSD_V=values.std(0,ddof=1),
                mean_mean_V=stats["mean_mean_V"],mean_scatter_SD_V=stats["mean_std_V"],
                rms_quantiles_V=stats["rms_quantiles_V"],observed_scope100=True)
    return output,c


def trajectory_entry(update,model,pre,post,reference,cfg):
    sp,cp=population_summary(pre,reference);sq,cq=population_summary(post,reference)
    return dict(update=update,forward_state_update=update,parameter_state="post_optimizer_update" if update else "untrained",
                model_state_sha256=common.state_digest(common.snapshot(model)),
                diagnostic_seed=cfg["diagnostics"]["seed"],population_rows=len(pre),
                rng_isolation_verified=True,pre_adc=sp,post_adc=sq),cp,cq


def save_trajectory(path,entries,pre_cov,post_cov,reference):
    np.savez_compressed(path,updates=np.array([e["update"] for e in entries]),
        pre_covariance_V2=np.stack(pre_cov),post_covariance_V2=np.stack(post_cov),
        pre_time_mean_V=np.array([e["pre_adc"]["time_mean_V"] for e in entries]),
        post_time_mean_V=np.array([e["post_adc"]["time_mean_V"] for e in entries]),
        pre_timeSD_V=np.array([e["pre_adc"]["timeSD_V"] for e in entries]),
        post_timeSD_V=np.array([e["post_adc"]["timeSD_V"] for e in entries]),
        pre_ACF20=np.array([e["pre_adc"]["acf20_mean"] for e in entries]),
        post_ACF20=np.array([e["post_adc"]["acf20_mean"] for e in entries]),
        pre_OLS_covariance_V2=np.array([e["pre_adc"]["OLS_covariance_V2"] for e in entries]),
        post_OLS_covariance_V2=np.array([e["post_adc"]["OLS_covariance_V2"] for e in entries]),
        TRAIN_covariance_V2=reference["covariance_V2"],TRAIN_timeSD_V=reference["timeSD_V"],
        TRAIN_ACF20=reference["target_acf20"].numpy(),TRAIN_OLS_covariance_V2=reference["OLS_covariance_V2"])


def reference_models(real,cfg,root,device):
    init=fit_baseline_initialization_v3(real,**cfg["model"])
    model=ObservedBaselineNoiseV3(init).to(device)
    state0=torch.load(Path(root)/cfg["references"]["initial004"]["path"],map_location="cpu",weights_only=False)
    state4=torch.load(Path(root)/cfg["references"]["final004"]["path"],map_location="cpu",weights_only=False)
    for state,update in ((state0,0),(state4,5000)):
        if state["update"]!=update or state["engineering"] is not False or state["model_scope"]!="observed_baseline100_only":
            raise ValueError("Bound004 reference role/update mismatch")
        if common.state_digest(state["model"])!=state["model_state_sha256"]:raise ValueError("004 model digest mismatch")
        if state["config"]["noise"]!=cfg["noise"] or state["config"]["model"]!=cfg["model"]:raise ValueError("004 optimization/architecture changed")
    if common.state_digest(common.snapshot(model))!=state0["model_state_sha256"]:
        raise ValueError("FullTRAIN rebuilt initializer is not bitwise004 initial")
    if common.state_digest(state4["initial_model_state"])!=state0["model_state_sha256"]:
        raise ValueError("004 final did not begin at bound initial")
    old=ObservedBaselineNoiseV3(state4["initialization"]).to(device)
    old.load_state_dict(state4["model"],strict=True);old.requires_grad_(False)
    return model,old,init,state0["model_state_sha256"]


def fit005(real,groups,cfg,out,bindings,*,root=ROOT,device="cuda:0",engineering=False):
    out=common.fresh_output(out);start=time.monotonic();d=common.configure_runtime(cfg,device)
    if not engineering and str(d)!="cuda:0":raise ValueError("Scientific005 requires cuda:0")
    (out/"noise").mkdir();(out/"diagnostics").mkdir()
    model,old,initialization,initial_match=reference_models(real,cfg,root,d)
    initial_state=common.snapshot(model);reference=training_statistics(real)
    target=reference["target_acf20"].clone();target_hash=array_digest(target)
    stats=fit_baseline_statistics(torch.as_tensor(real,dtype=torch.float64))
    old_objective=BaselineDistributionLoss(stats,seed=cfg["noise"]["projection_seed"],
                         evaluation_seed=cfg["noise"]["evaluation_projection_seed"]).to(d)
    added=ACF20Loss(target).to(d)
    latents=diagnostic_latents(model,cfg);common.atomic_save(dict(**latents,**bindings),out/"diagnostics/latents.pt")
    p0,q0=generate_diagnostic(model,latents);p4,q4=generate_diagnostic(old,latents)
    s0,c0=population_summary(p0,reference);t0,b0=population_summary(q0,reference)
    s4,c4=population_summary(p4,reference);t4,b4=population_summary(q4,reference)
    np.savez_compressed(out/"diagnostics/005A_covariances.npz",TRAIN_covariance_V2=reference["covariance_V2"],
        initial_pre_adc_covariance_V2=c0,initial_post_adc_covariance_V2=b0,
        v004_pre_adc_covariance_V2=c4,v004_post_adc_covariance_V2=b4)
    write_json(out/"diagnostics/005A_COMPARISON.json",common.clean(dict(TRAIN=reference["baseline_statistics"],
        TRAIN_ACF20=target,initial=dict(pre_adc=s0,post_adc=t0),
        v004=dict(pre_adc=s4,post_adc=t4),initial_state_matches004=True,
        initial_model_state_sha256=initial_match,diagnostic_seed=latents["seed"],rows=len(p0),
        all_population_outputs_generated=True,no_analog_TRAIN_counterpart=True,
        acf20_target_sha256=target_hash,**bindings)))
    model.train();optimizer=torch.optim.Adam(model.parameters(),lr=cfg["noise"]["initial_learning_rate"])
    sampler=FullCoverageSampler(len(real),cfg["noise"]["seed"]+101)
    rng=torch.Generator(device=d).manual_seed(cfg["noise"]["seed"]+102)
    real_tensor=torch.as_tensor(real,dtype=torch.float64,device=d)
    parameter_order=[name for name,_ in model.named_parameters()];parameters=list(model.parameters())
    if sum(p.numel() for p in parameters)!=5151:raise ValueError("Original parameter scope/order changed")
    update=0;last_noise=last_rows=context=None;attempted=completed=False
    entries=[];pre_cov=[];post_cov=[]
    entry,cp,cq=trajectory_entry(0,model,p0,q0,reference,cfg)
    entries.append(entry);pre_cov.append(cp);post_cov.append(cq)
    with (out/"diagnostics/trajectory.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(common.clean(entry),allow_nan=False)+"\n")
    save_trajectory(out/"diagnostics/trajectory.npz",entries,pre_cov,post_cov,reference)
    target_data=dict(acf20_target=target,acf20_target_sha256=target_hash,acf20_target_rows=len(real),
                     acf20_target_role="all_canonical_REAL_TRAIN",gradient_parameter_order=parameter_order)
    base=dict(schema_version=5,config=cfg,initialization=initialization,loss_statistics=stats,
              initial_model_state=initial_state,initial_state_matches004=True,initial004_model_state_sha256=initial_match,
              model_type="ObservedBaselineNoiseV3",model_scope="observed_baseline100_only",
              response_training_updates=0,response_integration_allowed=False,**target_data,**bindings)
    common.atomic_save(dict(**base,model=initial_state,update=0,engineering=engineering,
                      model_state_sha256=initial_match),out/"noise/initial_state.pt")
    write_json(out/"noise/fit_statistics.json",common.clean(dict(**base,TRAIN_rows=len(real),TRAIN_original_groups=len(np.unique(groups)),
                 validation_decoded=False,row_selection="none",fit_split="TRAIN")))
    def checkpoint(invalid=False):
        projection=None
        if context is not None:
            try:projection=context.state_dict()
            except RuntimeError:
                if not invalid:raise
                projection=dict(**context.metadata(),banks={k:v.detach().cpu().clone() for k,v in context.banks.items()},invalid_context=True)
        state=common.snapshot(model)
        return dict(**base,model=state,model_state_sha256=common.state_digest(state),
                    update=update,engineering=engineering,optimizer=optimizer.state_dict(),
                    sampler=sampler.state_dict(),noise_rng=rng.get_state().cpu(),
                    global_rng=common.global_rng_state(d),projection_context=projection,saved_UTC=common.utc())
    try:
        end=3 if engineering else 5000
        for _ in range(end):
            tick=time.monotonic();attempted=completed=False
            for g in optimizer.param_groups:g["lr"]=cfg["noise"]["low_learning_rate"] if update>=4000 else cfg["noise"]["initial_learning_rate"]
            last_rows=sampler.draw(cfg["noise"]["batch_size"])
            last_noise=model.sample_noise(cfg["noise"]["batch_size"],device=d,generator=rng,length=100)
            context=old_objective.create_projection_context(update+1)
            optimizer.zero_grad(set_to_none=True)
            waveform=digitize(model(last_noise,length=100))[:,0]
            components=old_objective.components(waveform,real_tensor[torch.as_tensor(last_rows,device=d)],projection_context=context)
            loss_old=old_objective.total(components);loss_acf=added(waveform);total=loss_old+loss_acf
            error=mean_acf20(waveform)-added.target_acf20
            if not torch.isfinite(total):raise FloatingPointError("Nonfinite005 objective")
            gradient=differentiated_step(loss_old,loss_acf,total,parameters)
            torch.nn.utils.clip_grad_norm_(parameters,cfg["noise"]["gradient_clip_norm"],error_if_nonfinite=True)
            attempted=True;optimizer.step();completed=True;update+=1;context.validate()
            if any(not torch.isfinite(p).all() for p in parameters):raise FloatingPointError("Nonfinite005 parameter")
            if not torch.equal(added.target_acf20.detach().cpu(),target):raise RuntimeError("Fixed TRAIN ACF20 target changed")
            metrics=dict(update=update,forward_state_update=update-1,utc=common.utc(),
                old_total=float(loss_old.detach()),acf20_loss=float(loss_acf.detach()),total=float(total.detach()),
                components={k:float(v.detach()) for k,v in components.items()},acf20_signed_errors=error.detach().cpu().tolist(),
                gradient_preclip_norm=gradient["total_gradient_preclip_norm"],**gradient,
                projection_context=context.metadata(),learning_rate=optimizer.param_groups[0]["lr"],
                real_unique=int((sampler.visits>0).sum()),real_total=len(real),
                noise_training_updates=0 if engineering else update,engineering_updates=update if engineering else 0,
                response_training_updates=0,update_seconds=time.monotonic()-tick)
            with (out/"noise/training.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(common.clean(metrics),allow_nan=False)+"\n")
            if update in cfg["diagnostics"]["trajectory_updates"]:
                pre,post=generate_diagnostic(model,latents,training_rng=rng,sampler=sampler,context=context)
                entry,cp,cq=trajectory_entry(update,model,pre,post,reference,cfg)
                entries.append(entry);pre_cov.append(cp);post_cov.append(cq)
                with (out/"diagnostics/trajectory.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(common.clean(entry),allow_nan=False)+"\n")
                save_trajectory(out/"diagnostics/trajectory.npz",entries,pre_cov,post_cov,reference)
            if update<=5 or update%25==0:
                guard=common.resource_check(out,cfg,start,d)
                write_json(out/"noise/status.json",dict(status="engineering" if engineering else "training005",**common.clean(metrics),**guard))
                print(json.dumps(dict(update=update,old_total=metrics["old_total"],acf20_loss=metrics["acf20_loss"],
                    max_acf20_error=float(error.detach().abs().max()),**guard)),flush=True)
            if update%1000==0:common.atomic_save(checkpoint(),out/"noise/latest.pt")
        state=checkpoint()
        if not engineering and [e["update"] for e in entries]!=cfg["diagnostics"]["trajectory_updates"]:
            raise RuntimeError("Every registered fixed trajectory state must be retained")
        if not engineering and (int(sampler.visits.sum())!=5000*512 or not (sampler.visits>0).all()):
            raise RuntimeError("CompleteTRAIN coverage/update draw count required")
        common.atomic_save(state,out/"noise/engineering.pt" if engineering else out/"noise/final.pt")
        # Endpoint outputs use the fixed bank even for bounded artificial engineering.
        final_pre,final_post=generate_diagnostic(model,latents,training_rng=rng,sampler=sampler,context=context)
        np.savez_compressed(out/"diagnostics/paired_populations.npz",
            initial_pre_adc=p0,initial_post_adc=q0,v004_pre_adc=p4,v004_post_adc=q4,
            v005_pre_adc=final_pre,v005_post_adc=final_post)
        common.resource_check(out,cfg,start,d)
        write_json(out/"noise/TRAINING_COMPLETE.json",dict(status="engineering_discarded" if engineering else "fixed005_endpoint",
            updates=update,real_unique=int((sampler.visits>0).sum()),real_total=len(real),
            all_rows_visited=bool((sampler.visits>0).all()),validation_decoded=False,
            acf20_target_sha256=target_hash,initial_state_matches004=True,
            noise_training_updates=0 if engineering else update,response_training_updates=0))
        return model,state,dict(initial_state_matches004=True,fixed_target_verified=True,
            diagnostic_rng_isolation_verified=True,trajectory_updates=[e["update"] for e in entries])
    except BaseException as error:
        trace=traceback.format_exc();capture=None
        try:common.atomic_save(dict(state=checkpoint(invalid=True),last_noise=last_noise,last_rows=last_rows,
                optimizer_step_attempted=attempted,optimizer_step_completed=completed),out/"noise/failure_state.pt")
        except BaseException as exc:capture=repr(exc)
        write_json(out/"noise/FAILURE.json",dict(status="failed",update=update,error=repr(error),traceback=trace,
            optimizer_step_attempted=attempted,optimizer_step_completed=completed,
            update_count_status="unknown_partial_optimizer_step" if attempted and not completed else "known_completed_optimizer_steps",
            capture_error=capture,response_training_updates=0,validation_decoded=False))
        raise


def verify_classifier_replay(out,device,*,expected_updates=200,expected_per_class=256):
    """Reopen six fixed classifiers; original512-row forward, no optimizer/refit."""
    from .noise_validation import BaselineClassifier,_view
    out=Path(out);d=torch.device(device);rng_before=_token(common.global_rng_state(d))
    with np.load(out/"baseline_populations.npz",allow_pickle=False) as z:
        fake=z["generated"].copy();real=z["REAL_validation"].copy()
        groups=z["real_validation_groups"].copy()
        rows0=z["real_validation_test0"].copy();rows1=z["real_validation_test1"].copy()
    n=int(expected_per_class)
    if fake.ndim!=2 or real.ndim!=2 or fake.shape[1]!=100 or real.shape[1]!=100 or len(fake)<n or len(groups)!=len(real):
        raise ValueError("Classifier replay population support mismatch")
    if not np.isfinite(fake).all() or not np.isfinite(real).all():raise ValueError("Nonfinite replay population")
    if rows0.shape!=(n,) or rows1.shape!=(n,) or not np.issubdtype(rows0.dtype,np.integer) or not np.issubdtype(rows1.dtype,np.integer):
        raise ValueError("Classifier replay row order/quota mismatch")
    if len(np.unique(rows0))!=n or len(np.unique(rows1))!=n or np.intersect1d(rows0,rows1).size or not ((rows0>=0)&(rows0<len(real))).all() or not ((rows1>=0)&(rows1<len(real))).all():
        raise ValueError("Classifier replay requires disjoint legal original test rows")
    labels=np.r_[np.zeros(n),np.ones(n)];results={}
    for view in ("raw","demean","detrend"):
        results[view]={}
        for domain in ("generated_REAL","REAL_REAL"):
            path=out/"classifiers"/view/domain
            checkpoint=path/"classifier.pt";prediction=path/"predictions.npz"
            cs,ps=sha256(checkpoint),sha256(prediction)
            state=torch.load(checkpoint,map_location="cpu",weights_only=False)
            if state["updates"]!=expected_updates:raise ValueError("Classifier replay fixed update mismatch")
            center=np.asarray(state["center"]);scale=np.asarray(state["scale"])
            if center.shape!=(100,) or scale.shape!=(100,) or center.dtype!=np.float32 or scale.dtype!=np.float32 or not np.isfinite(center).all() or not np.isfinite(scale).all() or not (scale>0).all():
                raise ValueError("Malformed saved fit-only classifier normalization")
            if domain=="generated_REAL":
                a,b=fake[:n],real[rows0];g0,g1=np.arange(n),groups[rows0]
            else:a,b=real[rows0],real[rows1];g0,g1=groups[rows0],groups[rows1]
            # Preserve the legacy order: view in float64, concatenate/cast32,
            # saved fit-only center/scale, and one full512-row forward.
            values=(np.concatenate([_view(a,view),_view(b,view)]).astype(np.float32)-center)/scale
            with np.load(prediction,allow_pickle=False) as z:
                saved={k:z[k].copy() for k in ("labels","scores","groups0","groups1")}
            if not np.array_equal(saved["labels"],labels) or not np.array_equal(saved["groups0"],g0) or not np.array_equal(saved["groups1"],g1):
                raise RuntimeError("Classifier replay labels/rows/groups differ:"+view+"/"+domain)
            cuda_devices=[d.index] if d.type=="cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                classifier=BaselineClassifier().to(d);classifier.load_state_dict(state["model"],strict=True)
                classifier.eval().requires_grad_(False)
                with torch.no_grad():scores=classifier(torch.as_tensor(values,device=d)).cpu().numpy()
            if saved["scores"].shape!=(2*n,) or scores.dtype!=saved["scores"].dtype or not np.isfinite(saved["scores"]).all() or np.ascontiguousarray(scores).tobytes()!=np.ascontiguousarray(saved["scores"]).tobytes():
                raise RuntimeError("Classifier score bitwise replay mismatch:"+view+"/"+domain)
            if sha256(checkpoint)!=cs or sha256(prediction)!=ps:raise RuntimeError("Classifier replay input changed")
            results[view][domain]=dict(scores_bitwise_equal=True,test_rows=2*n,checkpoint_sha256=cs,
                predictions_sha256=ps,scores_sha256=hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest(),
                scores_digest_format="contiguous_native_float32_bytes",normalization="saved_fit_only_center_scale_no_refit")
    if rng_before!=_token(common.global_rng_state(d)):raise RuntimeError("Classifier replay advanced global RNG")
    return dict(status="passed",classifiers_reopened=6,classifier_training_updates=0,
                optimizer_steps=0,global_RNG_preserved=True,exact_score_replay=results)


def validate_endpoint(state,cfg,bindings):
    if state["schema_version"]!=5 or state["engineering"] is not False or state["update"]!=5000 or state["config"]!=cfg:
        raise ValueError("Only unique5000 scientific005 endpoint may decodeVAL")
    if state["model_type"]!="ObservedBaselineNoiseV3" or state["model_scope"]!="observed_baseline100_only" or state["response_integration_allowed"] or state["response_training_updates"]!=0:
        raise ValueError("Bounded scope changed")
    if common.state_digest(state["initial_model_state"])!=state["initial004_model_state_sha256"]:
        raise ValueError("Reopened initial state digest mismatch")
    if state["initialization"]["length"]!=100 or state["initialization"]["metadata"]["events"]!=state["acf20_target_rows"]:
        raise ValueError("FullTRAIN initialization support/row count changed")
    if state["acf20_target"].shape!=(20,) or state["acf20_target"].dtype!=torch.float64:
        raise ValueError("Fixed float64 twenty-lag target required")
    if common.state_digest(state["model"])!=state["model_state_sha256"] or not state["initial_state_matches004"]:
        raise ValueError("Endpoint/initial state digest mismatch")
    for key,value in bindings.items():
        if state[key]!=value:raise ValueError("Endpoint binding mismatch:"+key)
    if array_digest(state["acf20_target"])!=state["acf20_target_sha256"]:raise ValueError("TRAIN target digest changed")
    visits=state["sampler"]["visits"]
    if len(visits)!=state["acf20_target_rows"] or int(visits.sum())!=5000*512 or not bool((visits>0).all()):
        raise ValueError("CompleteTRAIN coverage mismatch")


def merge_runner_checks(receipt,checks):
    implementation=receipt["implementation_checks"]
    if not implementation.get("checks") or not all(v is True for v in implementation["checks"].values()):
        raise RuntimeError("Evaluation implementation checks did not pass")
    if not all(v is True for v in checks.values()):raise RuntimeError("Runner checks did not pass")
    required=implementation.get("runner_assertions_required",[])
    if not required or any(checks.get(key) is not True for key in required):
        raise RuntimeError("Required runner assertions are missing or not passed")
    implementation.update(runner=checks,pending_runner_binding=False,status="passed")
    return receipt


def run(config_path,container,identities,out,*,source_manifest=None,device="cuda:0",root=ROOT):
    manifest=source_manifest or Path(root)/SOURCE_MANIFEST
    cfg,bindings=bind(config_path,manifest,root=root)
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError("Preserve existing005 output")
    start=time.monotonic();evaluation_started=False;noise_updates=0
    try:
        train,train_groups=common.load_training_baselines(container,identities,cfg)
        model,state,assertions=fit005(train,train_groups,cfg,out,bindings,root=root,device=device)
        noise_updates=state["update"]
        reopened=torch.load(out/"noise/final.pt",map_location="cpu",weights_only=False)
        validate_endpoint(reopened,cfg,bindings);model.load_state_dict(reopened["model"],strict=True)
        if common.state_digest(common.snapshot(model))!=reopened["model_state_sha256"]:raise RuntimeError("Reopened005 model mismatch")
        replay=common.save_replay(model,cfg,out/"noise/replay.pt",bindings)
        common.verify_replay(model,torch.load(out/"noise/replay.pt",map_location="cpu",weights_only=False))
        _,after_bindings=bind(config_path,manifest,root=root)
        if after_bindings!=bindings:raise ValueError("Bindings changed during005 fit")
        if sha256(container)!=cfg["train_container_sha256"] or sha256(identities)!=cfg["identities_sha256"]:raise ValueError("Inputs changed beforeVAL")
        if not torch.equal(reopened["acf20_target"],mean_acf20(torch.as_tensor(train,dtype=torch.float64)).cpu()):
            raise RuntimeError("Stored target not allTRAIN ACF20")
        validation,val_groups=common.load_endpoint_validation(container,identities,train_groups)
        from .noise_validation import evaluate_noise
        before=common.state_digest(common.snapshot(model));evaluation_started=True
        gate=evaluate_noise(model,train,validation,train_groups,val_groups,cfg["noise"],out/"noise_validation",device)
        if common.state_digest(common.snapshot(model))!=before:raise RuntimeError("Legacy evaluation mutated generator")
        write_json(out/"noise_validation/LEGACY_GATE.json",common.clean(copy.deepcopy(gate)))
        scoped=copy.deepcopy(gate)
        scoped.update(original_next_action=gate["next_action"],next_action="stop_no_validated_480point_response",
            model_scope="observed_baseline100_only",model_state_sha256=before,
            noise_checkpoint_sha256=sha256(out/"noise/final.pt"),config_sha256=bindings["config_sha256"],response_training_updates=0,response_integration_allowed=False)
        write_json(out/"noise_validation/NOISE_GATE.json",common.clean(scoped))
        from .noise_validation_acf005 import evaluate_acf005
        receipt=evaluate_acf005(gate,train,validation,train_groups,val_groups,cfg,out/"noise_validation",
                              root=root,diagnostic_dir=out/"diagnostics")
        if common.state_digest(common.snapshot(model))!=before:raise RuntimeError("005 evaluation mutated generator")
        if receipt["target_ACF20_sha256"]!=reopened["acf20_target_sha256"]:
            raise RuntimeError("Evaluation and fixed TRAIN target digests differ")
        _,final_bindings=bind(config_path,manifest,root=root)
        if final_bindings!=bindings or sha256(container)!=cfg["train_container_sha256"] or sha256(identities)!=cfg["identities_sha256"]:
            raise RuntimeError("Final source/input binding changed")
        validate_endpoint(reopened,cfg,bindings)
        common.verify_replay(model,replay)
        classifier_replay=verify_classifier_replay(out/"noise_validation",device,
            expected_updates=cfg["noise"]["classifier_updates"],expected_per_class=cfg["noise"]["classifier_test_per_class"])
        write_json(out/"noise_validation/CLASSIFIER_REPLAY.json",classifier_replay)
        checks=dict(initial_state_equals004=assertions["initial_state_matches004"],
            immutable_config_source_input=final_bindings==bindings,
            all_TRAIN_coverage=bool((reopened["sampler"]["visits"]>0).all()),
            fixed_TRAIN20_target=array_digest(reopened["acf20_target"])==reopened["acf20_target_sha256"],
            fixed5000_checkpoint_reopened=reopened["update"]==5000,
            frozen_generator_during_evaluation=common.state_digest(common.snapshot(model))==before,
            explicit1024_replay=True,exact_classifier_replay=classifier_replay["status"]=="passed",diagnostic_RNG_isolation=assertions["diagnostic_rng_isolation_verified"],
            TRAIN_initialization_only=True,VAL_only_after_endpoint=True)
        receipt=merge_runner_checks(receipt,checks)
        write_json(out/"noise_validation/EVALUATION_RECEIPT.json",common.clean(receipt))
        common.resource_check(out,cfg,start,device)
        excluded={"COMPLETE.json","status.json","console.log","resource_status.json","launch.json","exit.json","run.lock"}
        files={str(p.relative_to(out)).replace("\\","/"):sha256(p) for p in out.rglob("*") if p.is_file() and p.name not in excluded}
        complete=dict(status="complete",native_result="success",noise_training_updates=5000,classifier_training_updates=1200,
            response_training_updates=0,response_integration_allowed=False,model_scope="observed_baseline100_only",
            measurement_status="measurement_uncertain_single_exposed_acquisition",scientifically_qualified=False,
            legacy_gate_passed=gate["passed"],implementation_checks=receipt["implementation_checks"],
            final_checkpoint_sha256=sha256(out/"noise/final.pt"),replay_sha256=sha256(out/"noise/replay.pt"),
            noise_gate_sha256=sha256(out/"noise_validation/NOISE_GATE.json"),
            legacy_gate_sha256=sha256(out/"noise_validation/LEGACY_GATE.json"),
            evaluation_receipt_sha256=sha256(out/"noise_validation/EVALUATION_RECEIPT.json"),
            classifier_replay_sha256=sha256(out/"noise_validation/CLASSIFIER_REPLAY.json"),
            diagnostic_latents_sha256=sha256(out/"diagnostics/latents.pt"),
            paired_populations_sha256=sha256(out/"diagnostics/paired_populations.npz"),
            acf20_target_sha256=reopened["acf20_target_sha256"],
            model_state_sha256=before,real_unique=len(train),TRAIN_rows=len(train),
            TRAIN_original_groups=len(np.unique(train_groups)),all_rows_visited=True,
            validation_rows=len(validation),validation_original_groups=len(np.unique(val_groups)),
            next_action="stop_no_validated_480point_response",files=files,**bindings)
        write_json(out/"COMPLETE.json",complete);return complete
    except BaseException as error:
        out.mkdir(parents=True,exist_ok=True);finished=0
        for path in (out/"noise_validation/classifiers").glob("*/*/classifier.pt"):
            try:finished+=int(torch.load(path,map_location="cpu",weights_only=False).get("updates")==200)
            except BaseException:pass
        nf={}
        if (out/"noise/FAILURE.json").exists():nf=json.loads((out/"noise/FAILURE.json").read_text())
        partial=evaluation_started and finished<6
        write_json(out/"FAILURE.json",dict(status="failed",error=repr(error),traceback=traceback.format_exc(),
            noise_training_updates_known=noise_updates or nf.get("update",0),
            optimizer_partial_update_possible=nf.get("update_count_status")=="unknown_partial_optimizer_step",
            classifier_training_updates_known=finished*200,partial_current_classifier_updates_unknown=partial,
            possible_classifier_update_range=[finished*200,finished*200+(200 if partial else 0)],
            response_training_updates=0,response_integration_allowed=False,**bindings))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True);parser.add_argument("--container",required=True)
    parser.add_argument("--identities",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--source-manifest",default=str(ROOT/SOURCE_MANIFEST))
    parser.add_argument("--device",choices=["cuda:0"],default="cuda:0")
    args=parser.parse_args()
    value=run(args.config,args.container,args.identities,args.out,source_manifest=args.source_manifest,device=args.device)
    print(json.dumps(common.clean(value),indent=2),flush=True)


if __name__=="__main__":main()