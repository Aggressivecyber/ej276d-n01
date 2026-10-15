"""Only016 bootstrap-loader delta; source and receipt stdlib checks."""
from __future__ import annotations
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path("E:/GAN/research/N01")
FIX=ROOT/"receipts/wp3_016_20261015/launch_fix01"


def require(value,message):
    if not value:raise AssertionError(message)


def record(path):
    require(path.suffix.lower() not in {".npz",".npy",".pt",".pth",".ckpt",".safetensors"},"No scientific payload read")
    raw=path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def load(path):return json.loads(path.read_text(encoding="utf-8-sig"))


def dump(node):return ast.dump(node,include_attributes=False)


def main():
    target=FIX/"INDEPENDENT_FIX_REVIEW.json";require(not target.exists(),"Preserve existing fix review")
    ready=load(FIX/"CONTROLLER_FIX_READY.json")
    require(record(FIX/"CONTROLLER_FIX_READY.json")["sha256"]=="3dea5b57b0d5961de2669226cc0311deffedf461de41ca4c4e339e66ae4f7b62","Assigned fix READY hash")
    require(ready["passed"],"Engineering fix ready pass")
    for key in ["previous_controller","controller","previous_ENGINEERING_READY","test","test_script","diff","reused_numeric_engineering","reused_health_engineering"]:
        require(record(ROOT/ready[key]["path"])==ready[key],"Ready evidence binding: "+key)
    before=ast.parse((ROOT/ready["previous_controller"]["path"]).read_bytes())
    after=ast.parse((ROOT/ready["controller"]["path"]).read_bytes())
    require([n.name for n in before.body if isinstance(n,ast.FunctionDef)]==[n.name for n in after.body if isinstance(n,ast.FunctionDef)],"Function set unchanged")
    old_ops=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name=="operations")
    new_ops=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=="operations")
    require(dump(ast.Module(body=old_ops.body[:2],type_ignores=[]))==dump(ast.Module(body=new_ops.body[:2],type_ignores=[])),"Operations global/cache prefix unchanged")
    require(dump(ast.Module(body=old_ops.body[4:],type_ignores=[]))==dump(ast.Module(body=new_ops.body[5:],type_ignores=[])),"Operations remainder AST exact")
    old_rest=copy.deepcopy(before);new_rest=copy.deepcopy(after)
    next(n for n in old_rest.body if isinstance(n,ast.FunctionDef) and n.name=="operations").body=[]
    next(n for n in new_rest.body if isinstance(n,ast.FunctionDef) and n.name=="operations").body=[]
    require(dump(old_rest)==dump(new_rest),"All other module AST exact")
    changed=ast.unparse(ast.Module(body=new_ops.body[2:5],type_ignores=[]))
    require("/home/rylan/projects/ej276d-n01/wp2_014_20261015" in changed and "BASE_SHA" in changed
            and "path is not None" in changed and "next" in changed,"Only fixed two-candidate SHA-pinned loader")
    old_review=load(ROOT/"receipts/wp3_016_20261015/independent_01/SOURCE_REVIEW.json")
    sources={n:record(ROOT/n) for n in ready["sources"]}
    require(sources==ready["sources"] and sources["scripts/wp3_016_controller.py"]["sha256"]=="ae9afe444a90acb4fabddf1d2bf643550a4cbf29c4b9130408b040abcc843b5b","Final three-source binding")
    for n in ["scripts/run_wp3_016.py","src/n01/model016.py"]:require(sources[n]==old_review["sources"][n],"Numerical source unchanged")
    original_mf=load(ROOT/"WP3_016_SOURCE_MANIFEST.json")
    others=[]
    for n,row in original_mf["files"].items():
        if n=="scripts/wp3_016_controller.py":continue
        actual=record(ROOT/n);require(dict(bytes=actual["bytes"],sha256=actual["sha256"])==row,"Other frozen source unchanged: "+n);others.append(n)
    require(len(others)==114,"Exactly114 other frozen sources unchanged")
    old_pin_records={
        "wp3_016_config.json":"20884dc1a3916c2ac23820fa1ff7c47da21861a465a3f08fa5988cb71220281d",
        "WP3_016_SOURCE_MANIFEST.json":"9151ca1a9503cde85786c0623effe431e10fa706ec5b92895bc448714ed93789",
        "receipts/wp3_016_20261015/runtime_package_01/wp3_016_source.tar.gz":"9d054aeb744cf35c0f50080ddebc0397574a66ac404dd2f68a5d4e9221cf6fb9",
        "receipts/wp3_016_20261015/native_execution_01/NATIVE_START_SUMMARY.json":"d69bcaa07b2baa8f842361c26b6bb362cd79f29e65f9793b32c4919493929d90",
    }
    for n,digest in old_pin_records.items():require(record(ROOT/n)["sha256"]==digest,"Historical original pin/evidence retained")
    failure=load(ROOT/"receipts/wp3_016_20261015/native_execution_01/NATIVE_START_SUMMARY.json")
    require(failure["status"]=="DEPLOY_BOOTSTRAP_FAILED_BEFORE_INSTALL_OR_START" and failure["scientific_cost_from_this_failed_bootstrap"]==0
            and all(failure["actions"][k]==0 for k in ["SCP_source_transfers","install_calls","start_calls","collect_calls"]),"Original failure stopped before install/start/science")
    test=load(ROOT/ready["test"]["path"]);independent=load(FIX/"independent_bootstrap_stdlib_01/ENGINEERING_RECEIPT.json")
    require(test["passed"] and independent["passed"] and test["source_sha256"]==independent["source_sha256"]==sources["scripts/wp3_016_controller.py"]["sha256"],"Independent bootstrap test current source")
    require(len(test["checks"])==len(independent["checks"])==5 and test["checks"]==independent["checks"],"Same five loader cases passed independently")
    design=load(ROOT/"wp3_016_design.json");p=load(ROOT/"wp3_016_config.json")
    require(ready["budget_caps"]==old_review["budget_caps"]==design["budget_caps_design"]==p["budget_caps"],"Scientific budgets untouched")
    result=dict(schema="n01/wp3-016-bootstrap-fix-independent-review/v1",UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_id="wp3_016_20261015",status="PASS_MINIMAL_BOOTSTRAP_FIX_PENDING_REFREEZE",passed=True,blockers=[],sources=sources,budget_caps=ready["budget_caps"],
        reviewer=dict(agent="/root/wp3_015_solmax_review",assigned_model="GPT-6.1-Sol",assigned_effort="Max",runtime_model_metadata_independently_exposed=False),
        fix_READY=record(FIX/"CONTROLLER_FIX_READY.json"),previous_source_review=record(ROOT/"receipts/wp3_016_20261015/independent_01/SOURCE_REVIEW.json"),
        actual_delta=dict(previous_controller=ready["previous_controller"],controller=ready["controller"],changed_operations_loader_source=changed,
                          all_other_controller_AST_identical=True,operations_prefix_and_remainder_AST_identical=True,
                          other_frozen016_sources_unchanged=114,model_driver_design_roles_contracts_and_budget_unchanged=True,
                          candidate_order=["SELF_ROOT/scripts/wp2_controller.py","/home/rylan/projects/ej276d-n01/wp2_014_20261015/scripts/wp2_controller.py"],
                          pinned_original014_sha256="3480825e94e954a46c2c633e5df6ddef514e2b7def1332bb2e7688e78ae041a3",
                          selection_semantics="Select the first existing candidate, then require its exact BASE_SHA. A wrong installed primary never falls through to a valid fallback."),
        independently_rerun_tests=dict(receipt=record(FIX/"independent_bootstrap_stdlib_01/ENGINEERING_RECEIPT.json"),passed=True,cases=independent["checks"],
                                       execution_mode=independent["executed_controller_bytes_mode"],private_payload_reads=0,native_process_launches=0,SSH_calls=0),
        preserved_original_failure=dict(receipt=record(ROOT/"receipts/wp3_016_20261015/native_execution_01/NATIVE_START_SUMMARY.json"),
                                        original_config_manifest_package_unchanged=True,bootstrap_prepare_attempts=1,SCP_install_start_calls=0,scientific_cost=0,
                                        fresh_remote_state_after_failure_not_queried=True),
        reused_evidence=dict(numeric=ready["reused_numeric_engineering"],health=ready["reused_health_engineering"],
                             reuse_reason="Model/driver and all health/supervision/archive methods are AST-identical; only bootstrap file selection changed.",numeric_tests_repeated=False),
        refreeze_required=True,old_runtime_package_01_is_historical_not_new_controller_binding=True,
        root_next_action="Refreeze runtime_package_fix01 with the new exact controller hash, preserve scientific settings/costs and the original job identity, then perform a thin frozen binding review before any new native attempt.",
        actual_review_operations=dict(stdlib_only=True,original_NPZ_NPY_decodes=0,checkpoint_loads=0,NN_rows=0,AD_queries=0,new_random_draws=0,
                                      training_updates=0,new_calibration_fits=0,SSH_calls=0,GPU_work=0,source_or_meta_edits=0,
                                      new_output_root=FIX.as_posix()),
        evidence=[record(ROOT/ready[k]["path"]) for k in ["previous_controller","previous_ENGINEERING_READY","test","test_script","diff"]]+[record(Path(__file__))],
        cumulative_training_before_and_after_review=175667,scientific_start_performed=False,weights_promoted=False,scientific_acceptance=False,
        scientific_status="response_match_not_established",conclusion_zh="最小bootstrap修复独审通过：仅恢复成熟015的两候选加载路径，所有候选仍强制原014源码SHA。其余AST、114份其它冻源、数值模型和科学条件均不变；5例stdin bootstrap测试独立复跑通过。原失败与旧包完整保留，待新包重新冻结及薄绑定核对。")
    with target.open("x",encoding="utf-8",newline="\n") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    require(load(target)["passed"],"Review reopened")
    require(all(record(ROOT/n)==row for n,row in sources.items()),"Sources stable during review")
    print(json.dumps(dict(status=result["status"],review=record(target),sources=sources),indent=2))


if __name__=="__main__":main()
