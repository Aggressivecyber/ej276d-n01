"""Fixed N01 experiment, with exact same-version two-pass distribution gradients."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from .data import (FullCoverageSampler, charge_np, fit_model_statistics,
                   load_training, sha256, write_json)
from .losses import (N01Loss, PopulationWeights, WaveformDiscriminator,
                     d_hinge, fit_loss_statistics, g_hinge)
from .model import N01Response, digitize

WEIGHTS = dict(wave=1., QPSD=2., noise=1., adv=.1, regular=.01, trend=.1)
ARMS = (("full", "full", True), ("deterministic", "deterministic", True),
        ("simple", "simple", True), ("full_uniform", "full", False))


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gradient_summary(gradients):
    norms = {key: float(value.double().norm()) for key, value in gradients.items()}
    cosines = {}
    keys = list(gradients)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            denom = norms[a] * norms[b]
            cosines[a + "/" + b] = (float(torch.dot(gradients[a].flatten().double(),
                                                      gradients[b].flatten().double()) / denom)
                                         if denom > 0 else None)
    return dict(norms=norms, cosine=cosines, zero_norm_angle=None)


def exact_two_pass_backward(model, batches, endpoint_gradient, regular_weight=.01,
                            trend_weight=.1, verify_waveforms=None):
    """Parameter VJP of one joint batch without retaining four U-Net graphs.

    Caller must not mutate model parameters, forward buffers, or explicit noise
    between the snapshot forwards and these re-forwards. GroupNorm has no state.
    """
    start = 0
    for index, (source, noise) in enumerate(batches):
        output = model(source, noise=noise)
        count = len(source)
        if verify_waveforms is not None and not torch.equal(output["waveform"].detach(),
                                                             verify_waveforms[start:start + count]):
            raise RuntimeError("same-parameter/same-noise reforward changed numerical response")
        penalty = (regular_weight * output["regular"] + trend_weight * output["trend"]) / len(batches)
        torch.autograd.backward((output["waveform"], penalty),
                                (endpoint_gradient[start:start + count], None))
        start += count
    if start != len(endpoint_gradient):
        raise ValueError("VJP batch size mismatch")


class Engine:
    def __init__(self, data, model_stats, loss_stats, config, *, variant="full",
                 arm="full", learn_population=True, baseline_state=None, device="cuda"):
        self.config, self.device, self.data = config, torch.device(device), data
        self.seed = int(config["seed"])
        self.variant, self.arm, self.learn_population = variant, arm, learn_population
        self.model_stats, self.loss_stats = model_stats, loss_stats
        seed_all(self.seed)
        self.model = N01Response(model_stats, variant=variant).to(self.device)
        if baseline_state is not None:
            self.model.load_shared_baseline(baseline_state)
        self.loss = N01Loss(loss_stats, seed=config["projection_seed"]).to(self.device)
        # D initialization and stochastic streams are identical across arms.
        torch.manual_seed(self.seed + 1)
        self.discriminator = WaveformDiscriminator(loss_stats).to(self.device)
        self.population = PopulationWeights(torch.from_numpy(charge_np(data["source"])),
                                            torch.from_numpy(data["h"])).to(self.device)
        self.source = torch.from_numpy(data["source"]).to(self.device, dtype=torch.float32)
        self.real = torch.from_numpy(data["real"]).to(self.device, dtype=torch.float64)
        self.source_sampler = FullCoverageSampler(len(self.source), self.seed + 2)
        self.real_sampler = FullCoverageSampler(len(self.real), self.seed + 3)
        self.noise_rng = torch.Generator(device=self.device).manual_seed(self.seed + 4)
        self.g_optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-4)
        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-4, weight_decay=0.)
        self.p_optimizer = torch.optim.Adam(self.population.parameters(), lr=1e-3)
        self.stage = None
        self.update = 0
        self.last_context = {}
        self.last_metrics = {}
        self.started = time.monotonic()

    def set_stage(self, stage):
        self.model.set_stage(stage)
        self.stage = stage
        self.model.train()
        low = stage == "low_learning_rate"
        for group in self.g_optimizer.param_groups:
            group["lr"] = 5e-5 if low else 2e-4
        for group in self.d_optimizer.param_groups:
            group["lr"] = 2.5e-5 if low else 1e-4

    def _parameter_audit(self, batches, endpoint_gradients):
        params = [p for p in self.model.parameters() if p.requires_grad]
        counts = [p.numel() for p in params]
        vectors = {key: torch.zeros(sum(counts), device=self.device) for key in WEIGHTS}
        start = 0
        for source, noise in batches:
            output = self.model(source, noise=noise)
            for key in WEIGHTS:
                if key in ("regular", "trend"):
                    objective = output[key] * WEIGHTS[key] / len(batches)
                    gradients = torch.autograd.grad(objective, params, allow_unused=True, retain_graph=True)
                else:
                    gradients = torch.autograd.grad(output["waveform"], params,
                        grad_outputs=endpoint_gradients[key][start:start + len(source)] * WEIGHTS[key],
                        allow_unused=True, retain_graph=True)
                offset = 0
                for count, gradient in zip(counts, gradients):
                    if gradient is not None:
                        vectors[key][offset:offset + count].add_(gradient.detach().flatten().float())
                    offset += count
            start += len(source)
            del output, gradients
        result = gradient_summary(vectors)
        result.update(scope="all currently trainable response parameters, weighted objective components",
                      parameter_count=sum(counts), parameter_names=[n for n,p in self.model.named_parameters() if p.requires_grad])
        return result

    def step(self, *, audit=False, parameter_audit=False, verify_replay=False):
        if self.stage is None:
            raise RuntimeError("set_stage required")
        start_time = time.monotonic()
        joint = self.stage in ("joint_stochastic", "low_learning_rate")
        baseline = self.stage == "baseline"
        groups = int(self.config["distribution_microbatches"])
        parents = int(self.config["source_parents_per_microbatch"])
        repeats = int(self.config["responses_per_parent"])
        real_count = int(self.config["real_rows_per_microbatch"])
        batches, real_batches, source_indices, real_indices = [], [], [], []
        snapshots, penalties, saturations, identity_errors = [], [], [], []
        self.g_optimizer.zero_grad(set_to_none=True)
        # During baseline, electronics is fitted ONLY by independent baseline
        # noise statistics. Pulse/population losses may not alter its parameters.
        if baseline:
            for p in self.model.electronics.parameters():
                p.requires_grad_(False)
        with torch.no_grad():
            for _ in range(groups):
                si = self.source_sampler.draw(parents)
                ri = self.real_sampler.draw(real_count)
                repeated = np.repeat(si, repeats)
                source_indices.append(repeated)
                real_indices.append(ri)
                source = self.source[torch.as_tensor(repeated, device=self.device)]
                real_batches.append(self.real[torch.as_tensor(ri, device=self.device)])
                noise = self.model.sample_noise(len(source), self.device, self.noise_rng)
                batches.append((source, noise))
                # Save explicit RNG inputs and identities even if forward fails.
                self.last_context = dict(stage=self.stage, attempted_update=self.update + 1,
                    source_indices=np.concatenate(source_indices), real_indices=np.concatenate(real_indices),
                    noise=[{k:v.detach().cpu() for k,v in n.items()} for _,n in batches])
                out = self.model(source, noise=noise)
                snapshots.append(out["waveform"])
                penalties.append({k:out[k].detach() for k in ("regular", "trend")})
                saturations.append(out["saturation_fraction"])
                identity_errors.append(out["charge_identity_residual"].abs().max())
        fake = torch.cat(snapshots)
        real = torch.cat(real_batches)
        indices = torch.as_tensor(np.concatenate(source_indices), device=self.device)
        weights = self.population(indices).detach()
        self.last_context.update(source_parent_ids=self.data["source_ids"][indices.cpu().numpy()],
                                 real_original_ids=self.data["real_ids"][np.concatenate(real_indices)])
        dvalue = None
        if joint:
            self.discriminator.requires_grad_(True)
            self.discriminator.train()
            self.d_optimizer.zero_grad(set_to_none=True)
            # One D optimizer update for one G update; no old G samples.
            dloss = d_hinge(self.discriminator(fake.detach().float()),
                           self.discriminator(real.float()), weights)
            if not torch.isfinite(dloss):
                raise FloatingPointError("nonfinite discriminator objective")
            dloss.backward()
            dnorm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5., error_if_nonfinite=True)
            self.d_optimizer.step()
            dvalue = dict(loss=float(dloss.detach()), preclip_gradient_norm=float(dnorm))
        self.discriminator.requires_grad_(False)
        self.discriminator.eval()  # spectral normalization buffers fixed for G
        endpoint = fake.detach().requires_grad_(True)
        components = self.loss.components(dict(waveform=endpoint), real, weights, stage=self.stage)
        diagnostics = copy.deepcopy(self.loss.last_diagnostics)
        components["adv"] = g_hinge(self.discriminator(endpoint.float()), weights) if joint else endpoint.sum() * 0
        if baseline:
            components["noise"] = endpoint.sum() * 0
        endpoint_keys = ("wave", "QPSD", "noise", "adv")
        need_components = audit or parameter_audit
        endpoint_gradients = {}
        if need_components:
            for key in endpoint_keys:
                endpoint_gradients[key] = torch.autograd.grad(components[key], endpoint, retain_graph=True)[0]
        objective = sum(components[key] * WEIGHTS[key] for key in endpoint_keys)
        if not torch.isfinite(objective):
            raise FloatingPointError("nonfinite response distribution objective")
        gradient = torch.autograd.grad(objective, endpoint)[0].detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite waveform endpoint gradient")
        parameter_metrics = self._parameter_audit(batches, endpoint_gradients) if parameter_audit and not baseline else None
        exact_two_pass_backward(self.model, batches, gradient,
                                verify_waveforms=fake if verify_replay else None)
        independent_noise = None
        if baseline:
            for p in self.model.electronics.parameters():
                p.requires_grad_(True)
            electronics = torch.cat([digitize(self.model.electronics(noise)).squeeze(1) for _,noise in batches])
            independent_noise = self.loss.noise_loss(electronics, real)
            if not torch.isfinite(independent_noise):
                raise FloatingPointError("nonfinite independent electronics objective")
            independent_noise.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5., error_if_nonfinite=True)
        self.g_optimizer.step()
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                raise FloatingPointError("nonfinite parameter after update: " + name)
        pvalue = None
        if joint and self.learn_population and (self.update + 1) % int(self.config["population_update_interval"]) == 0:
            # Response sample positions are detached. The fit describes this
            # same-version snapshot, not a mixed-parameter history cache.
            self.p_optimizer.zero_grad(set_to_none=True)
            pfit = self.loss.population_objective(fake.detach(), real, self.population(indices))
            ppenalty = .01 * self.population.penalty()
            pobj = pfit + ppenalty
            if not torch.isfinite(pobj):
                raise FloatingPointError("nonfinite population objective")
            pobj.backward()
            torch.nn.utils.clip_grad_norm_(self.population.parameters(), 5., error_if_nonfinite=True)
            self.p_optimizer.step()
            pvalue = dict(QPSD=float(pfit.detach()), regularizer=float(ppenalty.detach()))
        self.update += 1
        values = {key:float(value.detach()) for key,value in components.items()}
        for key in ("regular", "trend"):
            values[key] = float(torch.stack([x[key] for x in penalties]).mean())
        if independent_noise is not None:
            values["noise"] = float(independent_noise.detach())
        population_weights = self.population().detach()
        metrics = dict(utc=utc(), arm=self.arm, variant=self.variant, stage=self.stage,
            update=self.update, loss=values, weighted_total=sum(values[k]*WEIGHTS[k] for k in WEIGHTS),
            response_gradient_preclip_norm=float(gnorm), discriminator=dvalue, population_update=pvalue,
            population_theta=self.population.theta.detach().cpu().tolist(),
            comparison_weight_range=[float(population_weights.min()), float(population_weights.max())],
            saturation_fraction=float(torch.stack(saturations).mean()),
            max_preADC_charge_identity_residual=float(torch.stack(identity_errors).max()),
            condition_support=diagnostics,
            coverage=dict(source_unique=int((self.source_sampler.visits>0).sum()), source_total=len(self.source),
                          real_unique=int((self.real_sampler.visits>0).sum()), real_total=len(self.real),
                          source_epochs=self.source_sampler.completed_epochs, real_epochs=self.real_sampler.completed_epochs),
            update_seconds=time.monotonic()-start_time, elapsed_seconds=time.monotonic()-self.started,
            gpu_allocated_bytes=torch.cuda.memory_allocated() if self.device.type=="cuda" else 0,
            gpu_reserved_bytes=torch.cuda.memory_reserved() if self.device.type=="cuda" else 0,
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if self.device.type=="cuda" else 0,
            same_version_reforward_verified=verify_replay)
        if audit:
            metrics["endpoint_gradients"] = gradient_summary({k:endpoint_gradients[k]*WEIGHTS[k] for k in endpoint_keys})
            metrics["endpoint_gradients"]["scope"] = "canonical-waveform endpoint; excludes direct parameter penalties"
        if parameter_metrics is not None:
            metrics["parameter_gradients"] = parameter_metrics
        self.last_metrics = metrics
        return metrics

    def checkpoint(self, path):
        atomic_torch_save(dict(schema_version=1, model=self.model.state_dict(), population=self.population.state_dict(),
            discriminator=self.discriminator.state_dict(), loss_state=self.loss.state_dict(),
            model_statistics=self.model_stats, loss_statistics=self.loss_stats,
            g_optimizer=self.g_optimizer.state_dict(), d_optimizer=self.d_optimizer.state_dict(),
            p_optimizer=self.p_optimizer.state_dict(), source_sampler=self.source_sampler.state_dict(),
            real_sampler=self.real_sampler.state_dict(), noise_rng=self.noise_rng.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=="cuda" else [],
            seed=self.seed, variant=self.variant, arm=self.arm, learn_population=self.learn_population,
            stage=self.stage, update=self.update, config=self.config, last_metrics=self.last_metrics,
            saved_utc=utc()), path)

    def restore(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["config"] != self.config or state["variant"] != self.variant or state["arm"] != self.arm:
            raise ValueError("resume configuration/arm mismatch")
        self.model.load_state_dict(state["model"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.population.load_state_dict(state["population"])
        self.loss.load_state_dict(state["loss_state"])
        self.g_optimizer.load_state_dict(state["g_optimizer"])
        self.d_optimizer.load_state_dict(state["d_optimizer"])
        self.p_optimizer.load_state_dict(state["p_optimizer"])
        self.source_sampler.load_state_dict(state["source_sampler"])
        self.real_sampler.load_state_dict(state["real_sampler"])
        self.noise_rng.set_state(state["noise_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type=="cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        self.update = state["update"]
        self.last_metrics = state["last_metrics"]
        self.set_stage(state["stage"])


def stage_for(update):
    if update < 1000:
        return "baseline"
    if update < 4000:
        return "mean_response"
    if update < 16000:
        return "joint_stochastic"
    return "low_learning_rate"


def train_until(engine, end, directory, status_path):
    directory.mkdir(parents=True, exist_ok=True)
    while engine.update < end:
        stage = stage_for(engine.update)
        changed = engine.stage != stage
        if changed:
            engine.set_stage(stage)
        next_update = engine.update + 1
        audit = next_update <= 5 or next_update % 50 == 0 or changed
        parameter_audit = (next_update % 4000 == 0 or changed) and stage != "baseline"
        metrics = engine.step(audit=audit, parameter_audit=parameter_audit,
                              verify_replay=next_update <= 5 or changed)
        # Every update is retained, including explicit sparse-bin diagnostics.
        with (directory/"training.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False)+"\n")
        if audit or next_update % 10 == 0:
            write_json(status_path, dict(status="training", pid=os.getpid(), **metrics))
        if audit:
            print(json.dumps({k:metrics[k] for k in ("utc","arm","stage","update","weighted_total","update_seconds","coverage")}), flush=True)
        if next_update % 1000 == 0 or next_update == end:
            engine.checkpoint(directory/"latest.pt")
        if next_update in (1000, 4000, 16000, 20000):
            engine.checkpoint(directory/f"stage_{next_update:05d}.pt")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("engineering", "train"), default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.set_num_interop_threads(2)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:config["cpu_threads"]])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this registered run")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(config["gpu_memory_limit_GiB"]*1024**3/total)
    engine = None
    try:
        write_json(out/"status.json", dict(status="preparing", pid=os.getpid(), started_utc=utc(), mode=args.mode))
        actual = sha256(args.container)
        if actual != config["train_container_sha256"]:
            raise ValueError("training container SHA256 changed")
        if sha256(args.identities) != config["identities_sha256"]:
            raise ValueError("identity file SHA256 changed")
        data = load_training(args.container, args.identities)
        model_stats = fit_model_statistics(data)
        loss_stats = fit_loss_statistics(torch.from_numpy(data["real"]), torch.from_numpy(data["source"]))
        write_json(out/"fit_statistics.json", dict(model=model_stats, loss=loss_stats))
        np.savez(out/"original_parent_manifest.npz", source_ids=data["source_ids"],
                 real_original_ids=data["real_ids"], real_groups=data["real_groups"],
                 original_source_weights=data["original_source_weights"])
        write_json(out/"run_binding.json", dict(config=config, container_sha256=actual,
            identity_sha256=sha256(args.identities), started_utc=utc(),
            torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
            GPU=torch.cuda.get_device_name() if args.device=="cuda" else None,
            fit_split="TRAIN", selected_source_rows=len(data["source"]), selected_real_rows=len(data["real"]),
            mode=args.mode, scientific_acceptance_evaluated=False))
        if args.mode == "engineering":
            # Throw-away initialization and optimizer state. Nothing is reused
            # in science; no measured metric selects architecture or weights.
            engine = Engine(data, model_stats, loss_stats, config, device=args.device, arm="engineering")
            metrics = []
            for stage in ("baseline", "mean_response", "joint_stochastic"):
                engine.set_stage(stage)
                metrics.append(engine.step(audit=True, verify_replay=True, parameter_audit=stage=="joint_stochastic"))
                print(json.dumps(dict(stage=stage,update_seconds=metrics[-1]["update_seconds"],loss=metrics[-1]["weighted_total"])),flush=True)
            write_json(out/"engineering_receipt.json", dict(status="passed", utc=utc(),
                discarded_updates=len(metrics), warm_start_used=False, results=metrics,
                scientific_acceptance_evaluated=False))
            write_json(out/"status.json",dict(status="engineering_passed",pid=os.getpid(),utc=utc()))
            return
        baseline_dir = out/"shared_baseline"
        baseline_path = baseline_dir/"stage_01000.pt"
        if not baseline_path.exists():
            engine = Engine(data,model_stats,loss_stats,config,device=args.device,arm="shared_baseline")
            if args.resume and (baseline_dir/"latest.pt").exists():
                engine.restore(baseline_dir/"latest.pt")
            train_until(engine,1000,baseline_dir,out/"status.json")
            del engine
            engine = None
        elif not args.resume:
            raise FileExistsError("existing science baseline: use --resume to retain completed work")
        baseline_state = torch.load(baseline_path,map_location="cpu",weights_only=False)["model"]
        for arm,variant,learn_population in ARMS:
            directory = out/arm
            if (directory/"training_complete.json").exists():
                if not args.resume:
                    raise FileExistsError(arm + " already complete")
            else:
                engine = Engine(data,model_stats,loss_stats,config,variant=variant,arm=arm,
                    learn_population=learn_population,baseline_state=baseline_state,device=args.device)
                # Each arm gets the same post-baseline sampler/noise start and
                # new optimizer. Shared 1000 updates are disclosed separately.
                engine.update = 1000
                if args.resume and (directory/"latest.pt").exists():
                    engine.restore(directory/"latest.pt")
                train_until(engine,20000,directory,out/"status.json")
                with torch.no_grad():
                    weights = engine.population().cpu().numpy()
                np.savez(directory/"comparison_weights.npz", source_ids=data["source_ids"],
                    comparison_weights=weights, original_source_weights=data["original_source_weights"])
                write_json(directory/"training_complete.json",dict(status="fixed_endpoint_reached",utc=utc(),
                    shared_baseline_updates=1000,arm_updates=19000,nominal_updates=20000,
                    checkpoint_sha256=sha256(directory/"stage_20000.pt"),
                    coverage=engine.last_metrics["coverage"],scientific_acceptance_evaluated=False))
                del engine
                engine = None
                if args.device=="cuda":
                    torch.cuda.empty_cache()
            from .evaluate import evaluate_checkpoint
            if not (directory/"evaluation"/"EVALUATION_COMPLETE.json").exists():
                write_json(out/"status.json",dict(status="evaluating",arm=arm,pid=os.getpid(),utc=utc()))
                evaluate_checkpoint(directory/"stage_20000.pt",args.container,args.identities,
                                    directory/"evaluation",device=args.device)
        write_json(out/"status.json",dict(status="fixed_runs_complete",utc=utc(),pid=os.getpid(),
            arms=[x[0] for x in ARMS],unique_updates=77000,scientific_conclusion="requires_joint_review"))
    except BaseException as error:
        receipt = dict(status="failed",utc=utc(),pid=os.getpid(),type=type(error).__name__,
                       message=str(error),traceback=traceback.format_exc(),mode=args.mode)
        if engine is not None:
            receipt.update(arm=engine.arm,stage=engine.stage,completed_updates=engine.update,
                           last_completed_metrics=engine.last_metrics)
            atomic_torch_save(engine.last_context,out/"failure_batch.pt")
            engine.checkpoint(out/"failure_state.pt")
        write_json(out/"failure.json",receipt)
        write_json(out/"status.json",receipt)
        raise


if __name__ == "__main__":
    main()
