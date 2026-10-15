"""Thin016 frozen package binding; no numeric payload decode or model work."""
from __future__ import annotations
import ast
import datetime
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile

ROOT=Path("E:/GAN/research/N01")
STAGE=ROOT/"receipts/wp3_016_20261015"
OUT=STAGE/"independent_01"
PACKAGE=STAGE/"runtime_package_01"
REMOTE="/home/rylan/projects/ej276d-n01/wp3_016_20261015"
EXPECTED_CONFIG="20884dc1a3916c2ac23820fa1ff7c47da21861a465a3f08fa5988cb71220281d"
EXPECTED_MF="9151ca1a9503cde85786c0623effe431e10fa706ec5b92895bc448714ed93789"
EXPECTED_PACKAGE="9d054aeb744cf35c0f50080ddebc0397574a66ac404dd2f68a5d4e9221cf6fb9"
FORBIDDEN={".npz",".npy",".pt",".pth",".ckpt",".safetensors",".bin"}


def require(value,message):
    if not value:raise AssertionError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def record(path):
    require(path.suffix.lower() not in FORBIDDEN,"No scientific input payload read")
    raw=path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=len(raw),sha256=digest(raw))


def load(path):return json.loads(path.read_text(encoding="utf-8-sig"))


def pair(row):return dict(bytes=row["bytes"],sha256=row["sha256"])


def main():
    target=OUT/"FROZEN_PACKAGE_REVIEW.json";require(not target.exists(),"Preserve prior review")
    cp=ROOT/"wp3_016_config.json";mp=ROOT/"WP3_016_SOURCE_MANIFEST.json";archive=PACKAGE/"wp3_016_source.tar.gz"
    config_record,mf_record,package_record=record(cp),record(mp),record(archive)
    require(config_record["sha256"]==EXPECTED_CONFIG and mf_record["sha256"]==EXPECTED_MF and package_record["sha256"]==EXPECTED_PACKAGE,"Assigned final hashes")
    p,mf=load(cp),load(mp)
    require(cp.read_bytes()==(PACKAGE/cp.name).read_bytes() and mp.read_bytes()==(PACKAGE/mp.name).read_bytes(),"Canonical byte-exact adoption")
    source_review=load(OUT/"SOURCE_REVIEW.json");design_review=load(OUT/"DESIGN_REVIEW.json")
    adoption=load(STAGE/"FROZEN_CONTROL_ADOPTION.json");freeze=load(PACKAGE/"FREEZE_RECEIPT.json")
    prep=load(STAGE/"CONTROL_PREPARATION_RECEIPT.json")
    for document in [adoption,freeze]:
        require(document["config_sha256"]==EXPECTED_CONFIG and document["source_manifest_sha256"]==EXPECTED_MF and not document["scientific_started"],"Freeze/adoption identity and no start")
    require(freeze["source_package_sha256"]==EXPECTED_PACKAGE and freeze["source_package_bytes"]==package_record["bytes"],"Package freeze record")
    require(mf["config_sha256"]==EXPECTED_CONFIG and p["status"]=="frozen" and p["run_id"]==mf["run_id"]=="wp3_016_20261015","Frozen configuration identity")
    require(source_review["passed"] and not source_review["blockers"] and design_review["passed"],"Prior independent approvals")
    require(record(ROOT/prep["source_review"]["path"])==prep["source_review"] and record(ROOT/prep["design_review"]["path"])==prep["design_review"],"Preparation binds exact independent reviews")
    require(len(mf["files"])==len(p["source_files"])==adoption["source_files"]==115 and adoption["source_controls"]==117,"115 source117 controls")
    require(set(mf["files"])==set(p["source_files"]),"Config/MF source names")
    local_source_records={}
    for name,row in mf["files"].items():
        actual=record(ROOT/name)
        require(pair(actual)==row and pair(p["source_files"][name])==row and p["source_files"][name]["path"]==REMOTE+"/"+name,"Source config/MF/local binding: "+name)
        local_source_records[name]=actual
    for name,row in source_review["sources"].items():require(local_source_records[name]==row,"Reviewed three-source hash unchanged")

    expected_members={**mf["files"],cp.name:pair(config_record),mp.name:pair(mf_record)}
    archive_members={}
    with tarfile.open(archive,"r:gz") as bundle:
        for member in bundle.getmembers():
            path=PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts and path.suffix.lower() not in FORBIDDEN,"Source-only regular safe archive member")
            require(member.name not in archive_members,"No duplicate member")
            stream=bundle.extractfile(member);require(stream is not None,"Archive member readable")
            raw=stream.read();archive_members[member.name]=dict(bytes=len(raw),sha256=digest(raw))
    require(archive_members==expected_members==freeze["source_inventory"] and len(archive_members)==117,"Every package byte matches115 sources plus2 controls")
    require(mf["before_scientific_decode"] and mf["public_source_only"] and not mf["scientific_inputs_copied"],"Source-only manifest declaration")

    oldmf=load(ROOT/"WP3_SOURCE_MANIFEST.json");oldp=load(ROOT/"wp3_015_config.json")
    require(record(ROOT/"WP3_SOURCE_MANIFEST.json")["sha256"]==mf["prior_WP3_manifest_sha256"]=="f1255dbff0e47edb7ec20e76c7d90057201f9a56f1184b7896f0c8cb0a67825a","Original015 manifest preserved")
    require(record(ROOT/"wp3_015_config.json")["sha256"]==oldmf["config_sha256"] and len(oldmf["files"])==mf["prior_source_file_count"]==105,"Original015 config105 preserved")
    for name,row in oldmf["files"].items():require(mf["files"][name]==row,"Old105 unchanged in new manifest")
    expected_protections=dict(oldp["protection_files"])
    oldremote="/home/rylan/projects/ej276d-n01/wp3_015_20261015"
    for name,row in oldmf["files"].items():expected_protections["wp3_015:"+name]=dict(path=oldremote+"/"+name,local_path=str(ROOT/name),**row)
    for name in ["WP3_SOURCE_MANIFEST.json","wp3_015_config.json"]:
        row=record(ROOT/name);expected_protections["wp3_015:"+name]=dict(path=oldremote+"/"+name,local_path=str(ROOT/name),**pair(row))
    require(p["protection_files"]==expected_protections and len(expected_protections)==freeze["protected_source_records"]==adoption["protected_source_records"]==475,"368 inherited plus107 exact015 protections")
    require(sum(k.startswith("wp3_015:") for k in expected_protections)==107,"105sources plus015 config/MF protected")

    require(len(p["inputs"])==prep["inputs"]==prep["local_input_SHA_verified"]==11 and prep["remote_only_inputs"]==0,"11 locally SHA-bound inputs")
    newheads={"M0_head_seed0","M0_head_seed1"}
    require(set(p["inputs"])=={k for k in oldp["inputs"] if not k.startswith(("C1_head_","mean_seed"))} | newheads,"Original9 plusM0heads input roles")
    inventory=load(ROOT/"receipts/wp3_015_20261015/collection_01/REMOTE_FULL_INVENTORY.json")["inventory"]
    input_bindings={}
    for key,row in p["inputs"].items():
        local=Path(row["local_path"])
        require(local.drive.upper() in ("E:","D:") and local.is_file() and local.stat().st_size==row["bytes"],"Input explicit E/D presence and size: "+key)
        require(any(PurePosixPath(row["path"]).is_relative_to(PurePosixPath(r)) for r in p["allowed_input_roots"]),"Input allowed remote root")
        if key in newheads:
            s=key[-1];name=f"runs/primary/artifacts/seed{s}_M0/final.pt"
            require(pair(row)==inventory[name] and row["path"]==oldremote+"/"+name,"Per-seed015 M0 final hash binding")
        else:require(row==oldp["inputs"][key],"Original9 input unchanged: "+key)
        input_bindings[key]=dict(row,binding_verified=True,payload_SHA_rehashed_this_review=False)
    require(len(p["contracts"])==prep["contracts"]==12,"12 contracts")
    for key,row in p["contracts"].items():
        actual=record(Path(row["local_path"]));require(pair(actual)==pair(row),"Contract local SHA and bytes: "+key)
        name=PurePosixPath(row["path"]).relative_to(PurePosixPath(REMOTE)).as_posix()
        require(mf["files"][name]==pair(row),"Contract package binding: "+key)
    require(p["contracts"]["acceptance"]==p["contracts"]["configuration"] and p["contracts"]["configuration"]["path"].endswith("docs/WP3_016_CONFIGURATION.md"),"016 acceptance/configuration refer to intended shared stage contract")
    require(p["contracts"]["tolerance_table"]["sha256"]=="f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b","Original013T")
    policy=load(Path(p["contracts"]["agent_model_policy"]["local_path"]))
    require("Sol" in json.dumps(policy) or "sol" in json.dumps(policy),"Latest Sol policy bound")
    require(p["budget_caps"]==source_review["budget_caps"]==prep["budget_caps"]==load(ROOT/"wp3_016_design.json")["budget_caps_design"],"Approved exact scientific budget")
    require(p["cumulative_training_before_run"]==prep["cumulative_training_updates"]==175667 and p["G_updates"]==2000
            and p["candidate_trial_order"]==["seed0_Q1","seed1_Q1"] and p["WP3_reserved_short_trials_this_block"]==2,"Cumulative and two-trial budget")
    require(p["development_selection"]["relative_wave_usefulness"].endswith("report_separately")
            and p["development_selection"]["replace_M0_only_if_eligible_and_score_below_reference_by_more_than"]==1e-12
            and not p["development_selection"]["weights_promoted"],"Selection distinct from usefulness/scientific promotion")
    result=dict(schema="n01/wp3-016-frozen-package-independent-review/v1",UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_id=p["run_id"],status="PASS_FROZEN_BINDING",passed=True,blockers=[],
        reviewer=dict(agent="/root/wp3_015_solmax_review",assigned_model="GPT-6.1-Sol",assigned_effort="Max",runtime_model_metadata_independently_exposed=False),
        sources=source_review["sources"],configSHA=EXPECTED_CONFIG,config_sha256=EXPECTED_CONFIG,source_manifest_sha256=EXPECTED_MF,source_package_sha256=EXPECTED_PACKAGE,
        config=config_record,manifest=mf_record,package=package_record,budget_caps=p["budget_caps"],
        checks=dict(canonical_controls_match_frozen_package_exact_bytes=True,all115_source_files_SHA_and_bytes_match=True,
                    all117_archive_members_regular_safe_public_source_only=True,exact_reviewed_three_sources_unchanged=True,
                    original105_sources_preserved=True,original015_107_protections_exact=True,total475_protections_preserve_inherited_records=True,
                    original9_inputs_unchanged_and_two015_M0_final_inputs_bound=True,input_explicit_local_E_D_size_verified=11,
                    root_local_SHA_verified_input_receipt_bound=11,all12_contracts_local_SHA_and_package_bound=True,
                    config_budget_and_selection_match_approved_design=True,latest_SolMax_policy_in_package=True),
        input_bindings=input_bindings,contracts=p["contracts"],source_files=115,archive_controls=117,protected_source_records=475,
        payload_review_boundary="Thin binding uses the root's11-input local-SHA preparation/freeze evidence, unchanged original input identities, M0 collection inventory hashes and current local byte sizes. This reviewer did not re-read raw NPZ or checkpoint payloads; native immutable gate rechecks all bytes before any future numeric decode.",
        numeric_tests_repeated=False,native_scientific_execution_verified=False,source_config_manifest_package_or_design_mutated=False,
        evidence=[record(STAGE/"CONTROL_PREPARATION_RECEIPT.json"),record(STAGE/"FROZEN_CONTROL_ADOPTION.json"),record(PACKAGE/"FREEZE_RECEIPT.json"),
                  record(OUT/"SOURCE_REVIEW.json"),record(OUT/"DESIGN_REVIEW.json"),record(Path(__file__))],
        actual_review_operations=dict(stdlib_only=True,original_NPZ_NPY_reads_or_decodes=0,checkpoint_payload_reads_or_loads=0,NN_rows=0,AD_queries=0,
                                      new_random_draws=0,training_updates=0,new_calibration_fits=0,SSH_calls=0,GPU_work=0,
                                      new_data_source_cache_temp_output_root=OUT.as_posix()),
        cumulative_training_before_and_after_review=175667,scientific_started_by_review=False,weights_promoted=False,scientific_acceptance=False,
        scientific_status="response_match_not_established",conclusion_zh="016最终冻结绑定通过：canonical配置/manifest与源码包逐字一致，115源码117控制、11输入与12合同、原105文件及107新增保护均绑定正确。未重复数值或读取科学payload；可由根代理继续登记与原生执行。")
    with target.open("x",encoding="utf-8",newline="\n") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    require(load(target)["passed"],"Review reopened")
    require(record(cp)==config_record and record(mp)==mf_record and record(archive)==package_record,"Frozen controls unchanged during review")
    print(json.dumps(dict(status=result["status"],review=record(target),config_sha256=EXPECTED_CONFIG,manifest_sha256=EXPECTED_MF,package_sha256=EXPECTED_PACKAGE),indent=2))


if __name__=="__main__":main()
