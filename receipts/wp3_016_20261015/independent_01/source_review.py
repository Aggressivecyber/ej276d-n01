"""Close016 source review using static source and artificial receipts only."""
from __future__ import annotations
import ast
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path("E:/GAN/research/N01")
OUT=ROOT/"receipts/wp3_016_20261015/independent_01"
ENG=ROOT/"receipts/wp3_016_20261015/engineering_01"
EXPECTED={
    "scripts/wp3_016_controller.py":"f1d5eceadb145d6a921c41055dd7a7e4d9fb05c36a150e99689e50e21358d02a",
    "scripts/run_wp3_016.py":"4729208d66c61c3b082827535719e72d8d7bb2ee60f390b688111ae3b4f12972",
    "src/n01/model016.py":"04c63f91ea5405937a940a9aee6eaed876db79149c857bfbe6ed8cfe6dcb138b",
}


def require(condition,message):
    if not condition:raise AssertionError(message)


def record(path):
    require(path.suffix.lower() not in {".npz",".npy",".pt",".pth",".ckpt",".safetensors"},"No numeric/checkpoint payload reads")
    raw=path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def load(path):return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    target=OUT/"SOURCE_REVIEW.json";require(not target.exists(),"Preserve prior source review")
    ready=load(ENG/"ENGINEERING_READY.json")
    require(record(ENG/"ENGINEERING_READY.json")["sha256"]=="d66bf5c4bb3215622bc2b6aee09686468452cf409f31e096540421de95c1f3de","Assigned READY SHA")
    sources={n:record(ROOT/n) for n in EXPECTED}
    for n,digest in EXPECTED.items():
        require(sources[n]["sha256"]==digest and sources[n]==ready["sources"][n],"Exact final source binding: "+n)
        ast.parse((ROOT/n).read_bytes(),filename=n)
    design=load(ROOT/"wp3_016_design.json")
    require(record(ROOT/"wp3_016_design.json")==ready["design"],"Reviewed design unchanged")
    require(record(OUT/"DESIGN_REVIEW.json")==ready["independent_design_review"],"Original design review binding")
    for field in ["numeric_engineering","stdlib_controller_engineering","settings_draft"]:
        item=ready[field];require(record(ROOT/item["path"])==item,"Engineering artifact binding: "+field)
    numeric=load(ROOT/ready["numeric_engineering"]["path"])
    stdlib=load(ROOT/ready["stdlib_controller_engineering"]["path"])
    independent=load(OUT/"artificial_stdlib_01/ENGINEERING_RECEIPT.json")
    require(ready["passed"] and numeric["passed"] and stdlib["passed"] and independent["passed"],"All engineering passes")
    require(numeric["artificial_only"] and stdlib["artificial_only"] and independent["artificial_only"],"Artificial evidence only")
    for n in ["src/n01/model016.py","scripts/run_wp3_016.py"]:
        require(numeric["sources"][n]==sources[n]["sha256"],"Numeric source unchanged after controller fix")
    require(stdlib["source_sha256"]==independent["source_sha256"]==sources["scripts/wp3_016_controller.py"]["sha256"],"Independent/final controller test binding")
    require(len(numeric["checks"])==7 and len(stdlib["checks"])==len(independent["checks"])==5,"7 numeric plus5 stdlib groups")
    require(stdlib["checks"][2]["negative_cases"]==independent["checks"][2]["negative_cases"]
            and len(independent["checks"][2]["negative_cases"])==15,"15 independently rerun health negative cases")

    tree=ast.parse((ROOT/"scripts/run_wp3_016.py").read_bytes())
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in {"seed_set016","settings016","budget_caps016"}]
    scope=dict(RUN_ID="wp3_016_20261015",HYPOTHESES=("fast","slow","common_level"))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<independent-stdlib-factories>","exec"),scope)
    settings=scope["settings016"]();caps=scope["budget_caps016"](settings)
    require(caps==design["budget_caps_design"]==ready["budget_caps"]==stdlib["pure_factory_design_budget"]
            ==independent["pure_factory_design_budget"],"All independent budget derivations equal")
    require(load(ROOT/ready["settings_draft"]["path"])["budget_caps"]==caps,"Draft exact budget")
    manifest=load(ROOT/"WP3_SOURCE_MANIFEST.json")
    require(record(ROOT/"WP3_SOURCE_MANIFEST.json")["sha256"]=="f1255dbff0e47edb7ec20e76c7d90057201f9a56f1184b7896f0c8cb0a67825a","015 protected manifest")
    require(len(manifest["files"])==105,"105 original sources")
    for n,item in manifest["files"].items():
        actual=record(ROOT/n)
        require(actual["bytes"]==item["bytes"] and actual["sha256"]==item["sha256"],"Protected source105: "+n)
    require(record(ROOT/"wp3_015_config.json")["sha256"]==manifest["config_sha256"],"015 frozen config preserved")

    result=dict(
        schema="n01/wp3-016-independent-source-review/v1",
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_id="wp3_016_20261015",
        status="PASS_SOURCE_REVIEW_PENDING_FROZEN_BINDING",passed=True,blockers=[],sources=sources,budget_caps=caps,
        reviewer=dict(agent="/root/wp3_015_solmax_review",assigned_model="GPT-6.1-Sol",assigned_effort="Max",
                      assignment_evidence="Root assigned independent source review under latest user all-future-work6.1solmax instruction.",runtime_model_metadata_independently_exposed=False),
        conclusion_zh="016源码审查通过。电荷分支、初始化、冻结/优化器/随机流、共享评价与精确预算符合设计；首次更新判定的同快照健康缺口已修复，并通过15项异常用例。本回执供根代理冻结配置与源码包，实际运行仍以冻结绑定和原生执行回执为准。",
        checks=[
            dict(check="actual_A_strictly_positive_canonical_Q_and_mean_corrected_law",passed=True,
                 evidence="model016 calibration_statistics016/apply_charge use canonical_charge008(actual A), require all Q finite and >0, and introduce no floor. softplus coefficients are strictly positive and finite. s2=log1p(a/Q+b), q=Q*exp(sqrt(s2)*z-s2/2) yields analytical mean Q and variance aQ+bQ². Nonfinite/underflow values fail without row deletion or clipping.",
                 numeric_evidence="Artificial law quadrature and actual-module Q/no-floor/nonfinite tests passed; no actual dataset/CP was evaluated."),
            dict(check="actual_A_384_one_LSB_analytic_initialization",passed=True,
                 evidence="Runtime016.calibrate_charge runs A once across all6593 legal parents, including zero prefix100 in the384-coordinate denominator, before heads/noise/ADC/new random/AD. a0=.5*ADC_STEP²/Sminus and b0=.5*ADC_STEP²/S0 imply a0*Sminus+b0*S0=ADC_STEP². Stable inverse softplus sets the common positive pair for both seeds; realized MC is reported without retuning.",
                 actual_native_calibration_verified=False),
            dict(check="off_exact_bytes_and_no_stochastic_charge_shape_product",passed=True,
                 evidence="The off path returns the original M0 tensor object. Active signal=original_M0 + actual_A*expm1(log_multiplier), while original014 B amplitude remains .05*max_abs(A). No new mean adapter exists; prefix100 remains exactly zero. Canonical numerical residual retains the inherited finite projection error.",
                 numeric_evidence="Artificial byte equality against original014 head with matching bytes, unchanged random_shape/basis_amplitude, inactive compensation bypass and canonical identity passed."),
            dict(check="frozen_M0_A_005_and_exact_two_scalar_AdamW",passed=True,
                 evidence="Model016 freezes/evals its M0 head and deep-hashes its state; inherited guards protect A and005. Exactly theta_a/theta_b are returned to a fresh AdamW with lr=.01, weight_decay=0, clip5. Parameter identity/whitelist and fixed optimizer options are asserted. The driver takes only C1 integral loss with frozen statistics plus explicit charge regularization and the inherited three-hypothesis saturation mean.",
                 numeric_evidence="Both artificial scalar gradients matched finite differences; wrong optimizer membership/lr and protected-head mutations were checked."),
            dict(check="registered_independent_charge_stream_preserves_old015_prefixes",passed=True,
                 evidence="ChargeStreams016 wraps the unchanged Streams010 and draws one float64 scalar per row from a separate CPU generator. Old SOURCE/REAL samplers and shape/noise/extension/projection addresses match corresponding015 seeds. Saved checkpoints carry both samplers, old stream state and charge state; evaluation banks reset fixed prefixes.",
                 numeric_evidence="Every old draw and generator state plus global RNG were byte-equal in the artificial prefix test."),
            dict(check="single_shared_initial_A_M0_initial_and_final_evaluation",passed=True,
                 evidence="Each parent/bank/chunk has one shared A/head/charge forward and one005 realization. The initial three signals select base A, original M0 and charge-on from that result; final has only the charge-on signal. Each arm/hyp still executes the original hard ADC/crop. Full four-axis evaluation, all flags/lowQ/undefined rows, separate baseline100 and charge moments/RNG states are saved.",
                 numeric_evidence="Two artificial engine updates exercised48hard+48REPORT,8baseline and8charge-moment files with exact derived counters and preserved lineage."),
            dict(check="exact_scientific_budget_and_controller_two_trial_collection",passed=True,
                 evidence="Independent pure factory extraction equals design/draft/READY/test caps. Training row calls2*2000*512*2=4096000, shared eval calls2seeds*2snapshots*2banks*6593*4=210976; A adds only6593 calibration rows. Hard calls3*4096000+2*2*3*6593*4*(3+1)=13553856. CP4, G/AD4000, optimizer2, calibration fit1. Archive candidate/seed indexing uses len(candidates)=1, preserving seed1_Q1 identity; collection256MiB excludes raw numeric arrays.",
                 reused_algorithm="Pinned014 archive/journal/owned-tree/failure/partial-collection/locking code;016 label/policy and exact index adaptation."),
            dict(check="firstG1_requires_same_snapshot_current_health_or_native0",passed=True,
                 evidence="Final controller embeds process_identity and native_snapshot. Launch/start record boot and supervisor/child start ticks; child CPU set is bound to12. Active first_valid requires matching boot/PID/parent/startticks, command/cwd, frozen config SHA, correct supervisor mode/worker config, CPU affinity, fresh same-snapshot owned-tree RSS/GPU/output/wall and durable completed work at least as far as firstG1. Terminal branch requires native0, full two-trial caps, COMPLETE and exact journal with resource caps; it allows already exited owned processes.",
                 fix_scope="The pinned supervisor resource/termination loop is preserved; AST adaptations add only start/launch identities and child preexec CPU binding. Native snapshot/resources are embedded, adding no missing remote helper.",
                 negative_cases=independent["checks"][2]["negative_cases"],
                 native_Linux_proc_validation_executed=False),
            dict(check="failures_partial_evidence_and_no_restart",passed=True,
                 evidence="Failed charge calls retain entered cost and unknown in-flight status rather than completed work. Driver records FAILED and preserves failure checkpoint metadata; native controller owns only its locked project process tree, refuses reused roots/restarts, and partial collection requires stopped owned PIDs. Same-snapshot failure and incomplete native0 cannot satisfy firstG1 verification.",
                 artificial_evidence="Injected charge failure, dead-without-exit, PID reuse, missing/stale identity/resources, resource excess, durable failure and malformed terminal states passed the focused tests."),
            dict(check="fixed_research_selection_scope_and_original_roles",passed=True,
                 evidence="Only two charge scalars and two2000G trials are opened. Original TRAIN/DEV roles,013T and C1 scales remain fixed. Comparison explicitly uses frozen015 M0 as replacement reference while reporting charge-on initialization separately. Final research selection uses dual-reference ALL6 positive-defined-T noninferiority and mean12wave improvement>1e-12;2% usefulness/absoluteT remain separate reports. Weights/scientific acceptance stay false."),
        ],
        resolved_findings=[dict(id="016-firstG1-current-health",severity="required_fix",prior_controller_sha256="e98446770b4e1e4404fa03f3c0217464f935e657da005308ac6b234000096b02",
                                issue="Historical firstG1 and known-failure exclusion did not prove current process health.",
                                resolution="Embedded fresh owned-process identity/resource snapshot and separate native0 terminal path;15 negative cases independently rerun against finalf1d5 controller.",status="resolved")],
        engineering_ready=record(ENG/"ENGINEERING_READY.json"),
        engineering_receipts=dict(numeric=record(ROOT/ready["numeric_engineering"]["path"]),
                                  controller=record(ROOT/ready["stdlib_controller_engineering"]["path"]),
                                  independent_controller_rerun=record(OUT/"artificial_stdlib_01/ENGINEERING_RECEIPT.json")),
        engineering_test_sources=[record(ENG/"check_wp3_016.py"),record(ENG/"check_controller016.py")],
        numeric_evidence_reuse=dict(model_and_driver_hashes_unchanged=True,numeric_groups=7,synthetic_engine_updates=2,
                                   synthetic_functional_AD_queries=1,full112second_numeric_test_repeated=False,
                                   numeric_receipt_controller_is_prehealthfix="Its old controller hash is historical; numeric model/driver are unchanged. Final controller has separate exact-SHA stdlib coverage."),
        independent_current_tests=dict(stdlib_groups=5,health_negative_cases=15,passed=True,
                                       actual_scientific_CP_NN_AD_random_training=0),
        documents=[record(ROOT/"wp3_016_design.json"),record(ROOT/"docs/WP3_016_CONFIGURATION.md"),record(ROOT/"WP3_016_DATA_ROLES.json"),
                   record(OUT/"DESIGN_REVIEW.json"),record(ROOT/"OBSERVATION_CONTRACT.json"),record(ROOT/"CALIBRATED_TOLERANCES_WP1_013.json"),
                   record(ROOT/"receipts/wp3_015_20261015/postrun_independent_02/FINAL_DEVELOPMENT_SELECTION.json")],
        original105_source_files_SHA_and_bytes_verified=True,original015_config_and_manifest_unchanged=True,
        source_freeze_pending=True,frozen_package_binding_review_required=True,
        native_scientific_start_or_execution_claimed=False,
        actual_local_review_operations=dict(stdlib_only=True,original_NPZ_NPY_decodes=0,checkpoint_tensor_loads=0,
                                           SOURCE_REAL_DEV_waveform_reads=0,NN_rows=0,AD_queries=0,training_updates=0,
                                           new_random_draws=0,new_calibration_fits=0,SSH_calls=0,GPU_work=0,
                                           model_driver_controller_or_design_edits=0,root_meta_edits=0,
                                           written_paths=[OUT.as_posix()]),
        reviewer_script=record(Path(__file__)),weights_promoted=False,scientific_acceptance=False,
        scientific_status="response_match_not_established")
    with target.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    require(load(target)["passed"],"Reopened source review")
    require(all(record(ROOT/n)==v for n,v in sources.items()),"Sources unchanged during review")
    print(json.dumps(dict(status=result["status"],review=record(target),sources=sources,blockers=[]),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
