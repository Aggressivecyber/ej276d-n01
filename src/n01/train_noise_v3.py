"""Fixed noise improvement004: observed100 baseline only; never pulse training."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime,timezone

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
for _thread_key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_thread_key]=str(min(12,max(1,int(os.environ.get(_thread_key,"12")))))
sys.dont_write_bytecode=True
import numpy as np
import torch

from .data import FullCoverageSampler,canonical_real,sha256,write_json
from .losses_v2 import BaselineDistributionLoss,fit_baseline_statistics
from .model import digitize

REGISTERED_CONFIG_SHA256="231fc902df4025e4389ede3abce5d7adec2cf591e8c3f3271c5169b68024ef7e"
ROOT=Path(__file__).resolve().parents[2]
REQUIRED_SOURCE=("src/n01/train_noise_v3.py","src/n01/electronics_v3.py",
                 "src/n01/losses_v2.py","src/n01/losses.py","src/n01/model.py",
                 "src/n01/data.py","src/n01/noise_validation.py","src/n01/evaluate.py")


def utc():return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,torch.Tensor):return clean(value.detach().cpu().numpy())
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def atomic_save(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    torch.save(value,temporary);temporary.replace(path)


def snapshot(model):
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


def state_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        v=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def fresh_output(out):
    out=Path(out).resolve()
    if os.name=="nt" and out.drive.upper() not in ("E:","D:"):raise ValueError("Artifacts must stay E/D")
    if out.exists() and any(out.iterdir()):raise FileExistsError("Preserve any existing outputs;no silent rerun")
    out.mkdir(parents=True,exist_ok=True)
    return out


def resource_check(out,config,started,device):
    if time.monotonic()-started>float(config["wall_limit_s"]):raise TimeoutError("Registered noise004 wall budget exceeded")
    output_bytes=0
    for p in Path(out).rglob("*"):
        try:
            if p.is_file():output_bytes+=p.stat().st_size
        except FileNotFoundError:
            pass  # Concurrent atomic status/checkpoint replacement is not a resource failure.
    if output_bytes>int(config["output_limit_MiB"]*2**20):raise RuntimeError("Owned output limit exceeded")
    d=torch.device(device)
    allocated=reserved=peak_allocated=peak_reserved=0
    if d.type=="cuda":
        allocated=torch.cuda.memory_allocated(d);reserved=torch.cuda.memory_reserved(d)
        peak_allocated=torch.cuda.max_memory_allocated(d);peak_reserved=torch.cuda.max_memory_reserved(d)
        if max(allocated,reserved,peak_allocated,peak_reserved)>config["gpu_memory_limit_GiB"]*2**30:
            raise RuntimeError("Noise004 CUDA allocation/reservation limit exceeded")
    return dict(output_bytes=output_bytes,gpu_allocated_bytes=allocated,gpu_reserved_bytes=reserved,
                gpu_peak_allocated_bytes=peak_allocated,gpu_peak_reserved_bytes=peak_reserved)


def validate_bindings(config_path,manifest_path,*,root=ROOT,expected_config_sha=REGISTERED_CONFIG_SHA256):
    config_path,manifest_path=Path(config_path),Path(manifest_path)
    config_hash=sha256(config_path)
    if config_hash!=expected_config_sha:raise ValueError("Frozen noise004 config SHA mismatch")
    config=json.loads(config_path.read_text(encoding="utf-8"))
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"]!=config_hash:raise ValueError("Execution manifest config mismatch")
    if config["run_id"]!="noise_improvement_004_20261012" or not config.get("registered_UTC"):
        raise ValueError("Unregistered noise004 run")
    if config["response_updates"]!=0 or config["response_integration_allowed"] or config["old_gates_changed"]:
        raise ValueError("Noise100 scope/gate contract changed")
    if config["noise"]["updates"]!=5000 or config["model"]["length"]!=100:
        raise ValueError("Fixed5000/100-point contract mismatch")
    root=Path(root).resolve()
    if not set(REQUIRED_SOURCE).issubset(manifest["files"]):raise ValueError("Missing executed source bindings")
    for name,expected in manifest["files"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha256(path)!=expected:raise ValueError("Executed source hash mismatch: "+name)
    for role in ("config","protocol","preregistration"):
        path=(root/manifest[role+"_path"]).resolve()
        if not path.is_relative_to(root) or sha256(path)!=manifest[role+"_sha256"]:
            raise ValueError("Registered document hash mismatch: "+role)
    return config,dict(config_sha256=config_hash,source_manifest_sha256=sha256(manifest_path),
        protocol_sha256=manifest["protocol_sha256"],preregistration_sha256=manifest["preregistration_sha256"],
        source_files=manifest["files"])


def load_training_baselines(container,identities,config,*,expected_rows=21047,expected_groups=422):
    if sha256(container)!=config["train_container_sha256"] or sha256(identities)!=config["identities_sha256"]:
        raise ValueError("TRAIN input SHA mismatch")
    with np.load(container,allow_pickle=False) as z:
        target=z["target"]
        if target.shape!=(expected_rows,480):raise ValueError("Complete legal TRAIN shape mismatch")
        real=canonical_real(target[:,:100])
    with np.load(identities,allow_pickle=False) as z:
        ids=z["real_train_original_index"].copy();groups=z["real_train_group"].copy()
    if len(ids)!=len(real) or len(np.unique(ids))!=len(real) or len(groups)!=len(real):
        raise ValueError("Complete TRAIN identity mismatch")
    if len(np.unique(groups))!=expected_groups:raise ValueError("Complete TRAIN group count mismatch")
    return real,groups


def load_endpoint_validation(container,identities,train_groups):
    # The caller may invoke only after reopening and validating the final checkpoint.
    with np.load(container,allow_pickle=False) as z:
        values=z["target_val"]
        if values.shape!=(599,480):raise ValueError("Existing599-row validation shape mismatch")
        real=canonical_real(values[:,:100])
    with np.load(identities,allow_pickle=False) as z:
        ids=z["real_val_original_index"].copy();groups=z["real_val_group"].copy()
    if len(ids)!=599 or len(np.unique(ids))!=599 or len(groups)!=599 or len(np.unique(groups))!=12:
        raise ValueError("Existing validation identities/groups mismatch")
    if np.intersect1d(train_groups,groups).size:raise ValueError("TRAIN/VAL original group overlap")
    return real,groups


def configure_runtime(config,device):
    threads=int(config.get("cpu_threads",2))
    if threads<1 or threads>12:raise ValueError("Registered CPU cap")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]=str(threads)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed=int(config["noise"]["seed"])
    random.seed(seed);np.random.seed(seed%2**32);torch.manual_seed(seed)
    d=torch.device(device)
    if d.type=="cuda":
        if str(d)!="cuda:0":raise ValueError("Use explicit cuda:0")
        torch.cuda.set_device(d);torch.cuda.manual_seed_all(seed)
        fraction=min(1.,config["gpu_memory_limit_GiB"]*2**30/torch.cuda.get_device_properties(d).total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction,device=d)
    return d


def global_rng_state(device):
    result=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
    if torch.device(device).type=="cuda":result["cuda"]=torch.cuda.get_rng_state_all()
    return result


def fit_noise_baselines(real,groups,config,out,bindings,*,device="cuda:0",engineering=False):
    from .electronics_v3 import ObservedBaselineNoiseV3,fit_baseline_initialization_v3
    out=fresh_output(out)
    started=time.monotonic();d=configure_runtime(config,device)
    if not engineering and str(d)!="cuda:0":raise ValueError("Scientific004 requires explicit cuda:0")
    real=np.asarray(real,dtype=np.float64)
    if real.ndim!=2 or real.shape[1]!=100 or len(groups)!=len(real):raise ValueError("Only complete100-point TRAIN baselines")
    cfg=config["noise"]
    initialization=fit_baseline_initialization_v3(real,**config["model"])
    statistics=fit_baseline_statistics(torch.from_numpy(real))
    model=ObservedBaselineNoiseV3(initialization).to(d)
    initial_state=snapshot(model)
    objective=BaselineDistributionLoss(statistics,seed=cfg["projection_seed"],
        evaluation_seed=cfg["evaluation_projection_seed"]).to(d)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["initial_learning_rate"])
    sampler=FullCoverageSampler(len(real),cfg["seed"]+101)
    rng=torch.Generator(device=d).manual_seed(cfg["seed"]+102)
    target=torch.as_tensor(real,dtype=torch.float64,device=d)
    update=0;last_rows=None;last_noise=None;context=None
    optimizer_step_attempted=False;optimizer_step_completed=False
    end=3 if engineering else 5000
    initial=dict(schema_version=3,model=initial_state,initial_model_state=initial_state,
        initialization=initialization,loss_statistics=statistics,config=config,**bindings,
        update=0,engineering=engineering,model_state_sha256=state_digest(initial_state),
        model_type="ObservedBaselineNoiseV3",model_scope="observed_baseline100_only",
        response_training_updates=0,response_integration_allowed=False)
    atomic_save(initial,out/"initial_state.pt")
    write_json(out/"fit_statistics.json",clean(dict(initialization=initialization,loss_statistics=statistics,
        fit_split="TRAIN",TRAIN_rows=len(real),TRAIN_original_groups=len(np.unique(groups)),
        row_selection="none",validation_decoded=False,model_scope="observed_baseline100_only",**bindings)))

    def checkpoint(allow_invalid_context=False):
        projection=None
        if context is not None:
            try:projection=context.state_dict()
            except RuntimeError:
                if not allow_invalid_context:raise
                projection=dict(**context.metadata(),banks={key:value.detach().cpu().clone() for key,value in context.banks.items()},
                                invalid_context=True,reason="original_context_validation_failed")
        state=snapshot(model)
        return dict(schema_version=3,model=state,initial_model_state=initial_state,
            initialization=initialization,loss_statistics=statistics,config=config,**bindings,
            optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),
            noise_rng=rng.get_state().cpu(),global_rng=global_rng_state(d),
            projection_context=projection,
            update=update,engineering=engineering,model_state_sha256=state_digest(state),
            model_type="ObservedBaselineNoiseV3",model_scope="observed_baseline100_only",
            response_training_updates=0,response_integration_allowed=False,saved_UTC=utc())
    try:
        while update<end:
            tick=time.monotonic()
            optimizer_step_attempted=False;optimizer_step_completed=False
            lr=cfg["low_learning_rate"] if update>=cfg["low_learning_rate_start"] else cfg["initial_learning_rate"]
            for group in optimizer.param_groups:group["lr"]=lr
            last_rows=sampler.draw(cfg["batch_size"])
            last_noise=model.sample_noise(cfg["batch_size"],device=d,generator=rng,length=100)
            context=objective.create_projection_context(update+1)
            optimizer.zero_grad(set_to_none=True)
            output=digitize(model(last_noise,length=100))[:,0]
            components=objective.components(output,target[torch.as_tensor(last_rows,device=d)],
                                            projection_context=context)
            loss=objective.total(components)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite noise004 objective")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"],error_if_nonfinite=True)
            optimizer_step_attempted=True
            optimizer.step()
            optimizer_step_completed=True;update+=1
            context.validate()
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError("Nonfinite noise004 parameter")
            metrics=dict(utc=utc(),update=update,target_updates=end,
                learning_rate=lr,total=float(loss.detach()),
                components={k:float(v.detach()) for k,v in components.items()},
                gradient_preclip_norm=float(norm),projection_context=context.metadata(),
                real_unique=int((sampler.visits>0).sum()),real_total=len(real),
                real_epochs=sampler.completed_epochs,update_seconds=time.monotonic()-tick,
                elapsed_seconds=time.monotonic()-started,noise_training_updates=0 if engineering else update,
                engineering_updates=update if engineering else 0,response_training_updates=0)
            with (out/"training.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(metrics,allow_nan=False)+"\n")
            if update<=5 or update%25==0:
                guard=resource_check(out,config,started,d)
                write_json(out/"status.json",dict(status="engineering" if engineering else "training_noise100",**metrics,**guard))
                print(json.dumps(dict(**metrics,**guard)),flush=True)
            if update%1000==0:atomic_save(checkpoint(),out/"latest.pt")
        state=checkpoint()
        if not engineering and not bool((sampler.visits>0).all()):raise RuntimeError("Incomplete legal TRAIN coverage")
        atomic_save(state,out/"engineering.pt" if engineering else out/"final.pt")
        resource_check(out,config,started,d)
        write_json(out/"TRAINING_COMPLETE.json",dict(status="engineering_discarded" if engineering else "fixed_noise100_endpoint",
            updates=update,real_unique=int((sampler.visits>0).sum()),real_total=len(real),
            all_rows_visited=bool((sampler.visits>0).all()),
            model_state_sha256=state["model_state_sha256"],validation_decoded=False,
            noise_training_updates=0 if engineering else update,response_training_updates=0))
        return model,state
    except BaseException as error:
        trace=traceback.format_exc();capture_error=None
        try:
            atomic_save(dict(state=checkpoint(allow_invalid_context=True),last_rows=last_rows,last_noise=last_noise,
                optimizer_step_attempted=optimizer_step_attempted,optimizer_step_completed=optimizer_step_completed,
                update_count_status="unknown_partial_optimizer_step" if optimizer_step_attempted and not optimizer_step_completed else "known_completed_optimizer_steps"),out/"failure_state.pt")
        except BaseException as capture:
            capture_error=repr(capture)
        write_json(out/"FAILURE.json",dict(status="failed",error=repr(error),traceback=trace,
            update=update,engineering=engineering,response_training_updates=0,validation_decoded=False,
            optimizer_step_attempted=optimizer_step_attempted,optimizer_step_completed=optimizer_step_completed,
            update_count_status="unknown_partial_optimizer_step" if optimizer_step_attempted and not optimizer_step_completed else "known_completed_optimizer_steps",
            failure_state_capture_error=capture_error))
        raise


def validate_endpoint(state,config,bindings):
    if state["engineering"] is not False or state["update"]!=5000 or state["config"]!=config:
        raise ValueError("Only fixed5000 scientific endpoint may decodeVAL/evaluate")
    if (state["schema_version"]!=3 or state["model_type"]!="ObservedBaselineNoiseV3"
            or state["initialization"]["length"]!=100 or state["initialization"]["metadata"]["samples"]!=100
            or state["response_integration_allowed"] is not False
            or state["model_scope"]!="observed_baseline100_only" or state["response_training_updates"]!=0):
        raise ValueError("Noise100 scope mismatch")
    if state_digest(state["model"])!=state["model_state_sha256"]:raise ValueError("Endpoint state digest mismatch")
    for name in ("config_sha256","source_manifest_sha256","protocol_sha256","preregistration_sha256"):
        if state[name]!=bindings[name]:raise ValueError("Endpoint binding mismatch: "+name)
    if state["source_files"]!=bindings["source_files"]:raise ValueError("Endpoint source files mismatch")
    visits=state["sampler"]["visits"]
    if visits.ndim!=1 or len(visits)!=state["initialization"]["metadata"]["events"]:
        raise ValueError("Endpoint coverage shape mismatch")
    if int(visits.sum())!=5000*config["noise"]["batch_size"]:raise ValueError("Endpoint coverage draw accounting mismatch")
    if not bool((visits>0).all()):raise ValueError("Incomplete endpoint TRAIN coverage")
    return state


def save_replay(model,config,path,bindings):
    seed=int(config["noise"]["generation_seed"])+10000
    d=next(model.parameters()).device
    generator=torch.Generator(device=d).manual_seed(seed)
    chunks=[]
    model.eval()
    with torch.no_grad():
        for _ in range(4):
            noise=model.sample_noise(256,device=d,generator=generator,length=100)
            pre=model(noise,length=100);post=digitize(pre,ste=False)
            chunks.append(dict(noise={key:value.detach().cpu().clone() for key,value in noise.items()},
                               pre_adc=pre.detach().cpu().clone(),post_adc=post.detach().cpu().clone()))
    value=dict(schema_version=3,seed=seed,seed_rule="int(config.noise.generation_seed)+10000",
               rows=1024,chunk_size=256,chunks=chunks,**bindings,
               model_state_sha256=state_digest(snapshot(model)),
               model_scope="observed_baseline100_only",response_training_updates=0)
    atomic_save(value,path)
    return value


def verify_replay(model,replay):
    d=next(model.parameters()).device
    if state_digest(snapshot(model))!=replay["model_state_sha256"]:raise ValueError("Replay state digest mismatch")
    with torch.no_grad():
        for chunk in replay["chunks"]:
            noise={key:value.to(d) for key,value in chunk["noise"].items()}
            pre=model(noise,length=100);post=digitize(pre,ste=False)
            if not torch.equal(pre.cpu(),chunk["pre_adc"]) or not torch.equal(post.cpu(),chunk["post_adc"]):
                raise RuntimeError("Same-latent pre/postADC replay mismatch")
    return True


def descriptive_metrics(model,initial_state,train,validation,populations):
    from .noise_validation import baseline_statistics
    with torch.no_grad():k=model.covariance().cpu().numpy()
    generated=np.asarray(populations,dtype=np.float64)
    cgen=np.cov(generated,rowvar=False,ddof=1);creal=np.cov(validation,rowvar=False,ddof=1)
    ctrain=np.cov(train,rowvar=False,ddof=1)
    scale=np.maximum(np.std(train,axis=0,ddof=1),1/16383)
    matrix_scale=scale[:,None]*scale[None,:]
    ratios=generated.std(0,ddof=1)/validation.std(0,ddof=1)
    rho={}
    for name,value in (("TRAIN",train),("REAL_validation",validation),("generated",generated)):
        c=np.asarray(baseline_statistics(value)["ols_covariance_V2"])
        rho[name]=float(c[0,1]/np.sqrt(c[0,0]*c[1,1]))
    final=snapshot(model)
    change={key:dict(initial_norm=float(value.double().norm()),
                     delta_norm=float((final[key]-value).double().norm()))
            for key,value in initial_state.items() if value.is_floating_point()}
    return dict(status="descriptive_not_an_acceptance_gate",
        generated_rows=len(generated),validation_rows=len(validation),
        covariance_scaling="fixed_TRAIN_timeSD_ddof1_floor_ADCstep",
        covariance_relative_error_validation=float(np.linalg.norm((cgen-creal)/matrix_scale)/np.linalg.norm(creal/matrix_scale)),
        covariance_relative_error_TRAIN=float(np.linalg.norm((cgen-ctrain)/matrix_scale)/np.linalg.norm(ctrain/matrix_scale)),
        timeSD_ratio_validation_min_median_max=[float(ratios.min()),float(np.median(ratios)),float(ratios.max())],
        mean_slope_correlations=rho,preADC_covariance_min_eigenvalue_V2=float(np.linalg.eigvalsh(k)[0]),
        parameter_changes=change,model_scope="observed_baseline100_only",
        physical_upstream_electronics_identified=False,response_integration_allowed=False)


def run(config_path,container,identities,out,*,source_manifest=None,device="cuda:0"):
    # Bind source/input/config before decoding any private member.
    config,bindings=validate_bindings(config_path,source_manifest or ROOT/"EXECUTION_SOURCE_MANIFEST.json")
    out=fresh_output(out);started=time.monotonic();evaluation_started=False
    try:
        train,train_groups=load_training_baselines(container,identities,config)
        model,state=fit_noise_baselines(train,train_groups,config,out/"noise",bindings,device=device)
        reopened=torch.load(out/"noise/final.pt",map_location="cpu",weights_only=False)
        validate_endpoint(reopened,config,bindings)
        model.load_state_dict(reopened["model"],strict=True)
        if state_digest(snapshot(model))!=reopened["model_state_sha256"]:
            raise RuntimeError("Reloaded endpoint differs from saved state")
        replay=save_replay(model,config,out/"noise/replay.pt",bindings)
        verify_replay(model,torch.load(out/"noise/replay.pt",map_location="cpu",weights_only=False))
        resource_check(out,config,started,device)
        _,new_bindings=validate_bindings(config_path,source_manifest or ROOT/"EXECUTION_SOURCE_MANIFEST.json")
        if new_bindings!=bindings:raise ValueError("Execution bindings changed during fit")
        if sha256(container)!=config["train_container_sha256"] or sha256(identities)!=config["identities_sha256"]:
            raise ValueError("Inputs changed during fitting;no validation decode permitted")
        validation,val_groups=load_endpoint_validation(container,identities,train_groups)
        from .noise_validation import evaluate_noise
        before=state_digest(snapshot(model))
        evaluation_started=True
        gate=evaluate_noise(model,train,validation,train_groups,val_groups,config["noise"],out/"noise_validation",device)
        after=state_digest(snapshot(model))
        if before!=after:raise RuntimeError("Evaluation mutated frozen generator")
        gate["original_next_action"]=gate["next_action"]
        gate.update(next_action="stop_no_validated_480point_response",model_scope="observed_baseline100_only",
                    model_state_sha256=before,noise_checkpoint_sha256=sha256(out/"noise/final.pt"),
                    config_sha256=bindings["config_sha256"],noise_training_updates=5000,
                    response_training_updates=0,response_integration_allowed=False,
                    unchanged_legacy_gate=True,generator_state_unchanged_during_evaluation=True)
        write_json(out/"noise_validation/NOISE_GATE.json",clean(gate))
        with np.load(out/"noise_validation/baseline_populations.npz",allow_pickle=False) as z:
            generated=z["generated"].copy()
        descriptive=descriptive_metrics(model,reopened["initial_model_state"],train,validation,generated)
        write_json(out/"noise_validation/DESCRIPTIVE_METRICS.json",clean(descriptive))
        resource_check(out,config,started,device)
        excluded={"COMPLETE.json","status.json","console.log","resource_status.json","resource_status",
                  "launch.json","launch","exit.json","exit","run.lock"}
        payload={str(p.relative_to(out)).replace("\\","/"):sha256(p)
                 for p in out.rglob("*") if p.is_file() and p.name not in excluded}
        complete=dict(status="complete",native_result="success",files=payload,
            train_container_sha256=config["train_container_sha256"],identities_sha256=config["identities_sha256"],
            noise_training_updates=5000,classifier_training_updates=1200,
            response_training_updates=0,response_integration_allowed=False,
            model_scope="observed_baseline100_only",noise_gate_passed=gate["passed"],
            noise_gate_status=gate["status"],model_state_sha256=before,
            final_checkpoint_sha256=sha256(out/"noise/final.pt"),
            noise_gate_sha256=sha256(out/"noise_validation/NOISE_GATE.json"),
            descriptive_sha256=sha256(out/"noise_validation/DESCRIPTIVE_METRICS.json"),
            TRAIN_rows=len(train),TRAIN_original_groups=len(np.unique(train_groups)),
            real_unique=int((reopened["sampler"]["visits"]>0).sum()),all_rows_visited=True,
            validation_rows=len(validation),validation_original_groups=len(np.unique(val_groups)),
            validation_previously_used_for_diagnostics=True,replay_verified=True,
            replay_seed=replay["seed"],replay_sha256=sha256(out/"noise/replay.pt"),
            generator_state_before_evaluation=before,generator_state_after_evaluation=after,
            next_action="stop_no_validated_480point_response",elapsed_seconds=time.monotonic()-started,**bindings)
        write_json(out/"COMPLETE.json",complete)
        return complete
    except BaseException as error:
        completed_classifiers=0
        for path in (out/"noise_validation/classifiers").glob("*/*/classifier.pt"):
            try:
                saved=torch.load(path,map_location="cpu",weights_only=False)
                completed_classifiers+=int(saved.get("updates")==config["noise"]["classifier_updates"])
            except BaseException:
                pass
        known_classifier_updates=completed_classifiers*config["noise"]["classifier_updates"]
        partial=evaluation_started and completed_classifiers<6
        noise_failure={}
        if (out/"noise/FAILURE.json").exists():
            noise_failure=json.loads((out/"noise/FAILURE.json").read_text())
        write_json(out/"FAILURE.json",dict(status="failed",error=repr(error),traceback=traceback.format_exc(),
            response_training_updates=0,response_integration_allowed=False,
            noise_training_updates_known=state["update"] if "state" in locals() else noise_failure.get("update",0),
            optimizer_partial_update_possible=noise_failure.get("update_count_status")=="unknown_partial_optimizer_step",
            evaluation_started=evaluation_started,completed_classifier_checkpoints=completed_classifiers,
            classifier_training_updates_known=known_classifier_updates,
            partial_current_classifier_updates_unknown=partial,
            possible_classifier_update_range=[known_classifier_updates,known_classifier_updates+(200 if partial else 0)],
            **bindings))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True);parser.add_argument("--container",required=True)
    parser.add_argument("--identities",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--source-manifest",default=str(ROOT/"EXECUTION_SOURCE_MANIFEST.json"))
    parser.add_argument("--device",choices=["cuda:0"],default="cuda:0")
    args=parser.parse_args()
    result=run(args.config,args.container,args.identities,args.out,source_manifest=args.source_manifest,device=args.device)
    print(json.dumps(clean(result),indent=2),flush=True)


if __name__=="__main__":main()