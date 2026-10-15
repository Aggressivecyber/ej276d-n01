"""Thin re-frozen016 launch-fix pin/delta binding, stdlib only."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile

ROOT=Path("E:/GAN/research/N01")
STAGE=ROOT/"receipts/wp3_016_20261015"
FIX=STAGE/"launch_fix01"
NEW=STAGE/"runtime_package_fix01"
OLD=STAGE/"runtime_package_01"
PINS=dict(config_sha256="c0fe6408b53969da9f0bd70c01bf7649cfabfbf1864752dc5696bb8c9b6d5da9",
          source_manifest_sha256="1488217be443fb49c60db4fcf3d852fcda8ba33507ce7e5ba2ee2f3d1e4d1327",
          source_package_sha256="524ba471ac24d376211349280237e02562dfe797b2921fc25b01c46aac8e688e")


def require(value,message):
    if not value:raise AssertionError(message)


def record(path):
    require(path.suffix.lower() not in {".npz",".npy",".pt",".pth",".ckpt",".safetensors",".bin"},"No scientific payload read")
    raw=path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def load(path):return json.loads(path.read_text(encoding="utf-8-sig"))


def pair(row):return dict(bytes=row["bytes"],sha256=row["sha256"])


def main():
    target=FIX/"FROZEN_PACKAGE_FIX_REVIEW.json";require(not target.exists(),"Preserve prior review")
    cp=ROOT/"wp3_016_config.json";mp=ROOT/"WP3_016_SOURCE_MANIFEST.json";archive=NEW/"wp3_016_source.tar.gz"
    cfg,mf,package=record(cp),record(mp),record(archive)
    require(cfg["sha256"]==PINS["config_sha256"] and mf["sha256"]==PINS["source_manifest_sha256"] and package["sha256"]==PINS["source_package_sha256"],"New assigned pins")
    require(cp.read_bytes()==(NEW/cp.name).read_bytes() and mp.read_bytes()==(NEW/mp.name).read_bytes(),"Canonical new controls exact adoption")
    p,m=load(cp),load(mp);oldp,oldm=load(OLD/cp.name),load(OLD/mp.name)
    independent=load(FIX/"INDEPENDENT_FIX_REVIEW.json");prep=load(STAGE/"CONTROL_PREPARATION_RECEIPT_FIX01.json")
    adoption=load(STAGE/"FROZEN_CONTROL_ADOPTION_FIX01.json");freeze=load(NEW/"FREEZE_RECEIPT.json")
    require(independent["passed"] and not independent["blockers"],"Independent minimal fix pass")
    require(record(ROOT/prep["independent_fix_review"]["path"])==prep["independent_fix_review"],"Preparation exact independent fix review")
    require(record(ROOT/prep["previous_failed_deploy"]["path"])==prep["previous_failed_deploy"],"Preparation original failure")
    for document in [adoption,freeze]:
        require(all(document[k]==v for k,v in PINS.items()) and not document["scientific_started"],"Refreeze/adoption pins and no start")
    require(p["status"]=="frozen" and m["config_sha256"]==PINS["config_sha256"] and p["run_id"]==m["run_id"]=="wp3_016_20261015","Frozen same job identity")
    added={"launch_revision","previous_launch_configuration_sha256","launch_fix_reason","previous_deployment_failure"}
    require(set(p)-set(oldp)==added and not set(oldp)-set(p),"Only approved launch metadata added")
    changed=[k for k in oldp if oldp[k]!=p[k]]
    require(set(changed)=={"frozen_UTC","source_files"},"No scientific setting/input/contract/budget change")
    require(p["launch_revision"]=="fix01" and p["previous_launch_configuration_sha256"]==record(OLD/cp.name)["sha256"]
            and p["previous_deployment_failure"]==prep["previous_failed_deploy"],"Launch revision links previous exact config/failure")
    changed_sources=[k for k in oldp["source_files"] if oldp["source_files"][k]!=p["source_files"][k]]
    require(changed_sources==["scripts/wp3_016_controller.py"] and set(p["source_files"])==set(oldp["source_files"]),"Only controller source binding changed")
    require(p["inputs"]==oldp["inputs"] and len(p["inputs"])==11 and p["contracts"]==oldp["contracts"] and len(p["contracts"])==12
            and p["protection_files"]==oldp["protection_files"] and len(p["protection_files"])==475,"11 inputs12 contracts475 protections unchanged")
    require(p["budget_caps"]==independent["budget_caps"]==oldp["budget_caps"] and p["cumulative_training_before_run"]==175667
            and p["candidate_trial_order"]==["seed0_Q1","seed1_Q1"],"Same two trials and cost; no new candidate/job")
    require(len(m["files"])==len(p["source_files"])==115 and set(m["files"])==set(oldm["files"]),"Same115 source names")
    for name,row in m["files"].items():
        actual=record(ROOT/name)
        require(pair(actual)==row==pair(p["source_files"][name]),"Frozen source binding: "+name)
    for name,row in independent["sources"].items():require(record(ROOT/name)==row,"Exact reviewed fix source hashes")
    mf_changed=[k for k in oldm if oldm[k]!=m[k]]
    require(set(mf_changed)=={"config_sha256","frozen_UTC","files"},"Manifest only new config/time/controller")
    require([k for k in oldm["files"] if oldm["files"][k]!=m["files"][k]]==["scripts/wp3_016_controller.py"],"114 other source hashes unchanged")
    members={}
    with tarfile.open(archive,"r:gz") as bundle:
        for member in bundle.getmembers():
            path=PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts and path.suffix.lower() not in {".npz",".npy",".pt",".pth",".bin"},"Regular safe source-only archive")
            require(member.name not in members,"No duplicate archive member")
            raw=bundle.extractfile(member).read();members[member.name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    expected={**m["files"],cp.name:pair(cfg),mp.name:pair(mf)}
    require(members==expected==freeze["source_inventory"] and len(members)==117,"117 archive controls exact")
    oldfreeze=load(OLD/"FREEZE_RECEIPT.json")
    for name,key in [(cp.name,"config_sha256"),(mp.name,"source_manifest_sha256"),("wp3_016_source.tar.gz","source_package_sha256")]:
        require(record(OLD/name)["sha256"]==oldfreeze[key],"Original frozen package retained: "+name)
    package_changed=[name for name in oldfreeze["source_inventory"] if oldfreeze["source_inventory"][name]!=members[name]]
    require(set(package_changed)=={"scripts/wp3_016_controller.py",cp.name,mp.name},"Exactly3 package members changed")
    result=dict(schema="n01/wp3-016-frozen-launch-fix-independent-review/v1",UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_id=p["run_id"],status="PASS_REFROZEN_LAUNCH_FIX_BINDING",passed=True,blockers=[],sources=independent["sources"],budget_caps=p["budget_caps"],**PINS,
        reviewer=dict(agent="/root/wp3_015_solmax_review",assigned_model="GPT-6.1-Sol",assigned_effort="Max",runtime_model_metadata_independently_exposed=False),
        config=cfg,manifest=mf,package=package,source_files=115,archive_controls=117,protected_source_records=475,
        config_delta=dict(added_launch_metadata=sorted(added),changed_existing_keys=changed,only_controller_source_record_changed=True,
                          all_scientific_settings_inputs_contracts_budgets_and_roles_unchanged=True,candidate_trial_order_unchanged=True,
                          job_identity_unchanged=True,new_candidate_trials=0),
        package_delta=dict(changed_members=package_changed,other114_sources_match_current_local_bytes=True,all117_members_match_SHA_and_bytes=True,
                           regular_members_only=True,scientific_payload_copied=False,original_runtime_package_01_retained_and_SHA_verified=True),
        original105_and107_protection_evidence="Original105 source subset and all475 protection records are unchanged from the independently passed original frozen binding; exact protection dictionary equality was checked in this delta review.",
        input_and_contract_evidence="All11 input and12 contract records are JSON-identical to the original independently passed package. No raw input/checkpoint payload was re-read.",
        evidence=[record(FIX/"INDEPENDENT_FIX_REVIEW.json"),record(STAGE/"CONTROL_PREPARATION_RECEIPT_FIX01.json"),record(STAGE/"FROZEN_CONTROL_ADOPTION_FIX01.json"),
                  record(NEW/"FREEZE_RECEIPT.json"),record(STAGE/"independent_01/FROZEN_PACKAGE_REVIEW.json"),record(Path(__file__))],
        actual_review_operations=dict(stdlib_only=True,numeric_tests_repeated=False,original_NPZ_NPY_reads_or_decodes=0,checkpoint_payload_reads_or_loads=0,
                                      NN_rows=0,AD_queries=0,new_random_draws=0,training_updates=0,new_calibration_fits=0,SSH_calls=0,GPU_work=0,
                                      canonical_source_config_manifest_or_meta_edits=0,new_output_root=FIX.as_posix()),
        cumulative_training_before_and_after_review=175667,scientific_started_by_review=False,weights_promoted=False,scientific_acceptance=False,
        scientific_status="response_match_not_established",conclusion_zh="修复版冻结绑定通过：新canonical配置和manifest与117成员包逐字一致，仅controller及两个控制文件更换；114其它源码、11输入、12合同、475保护、科学设置和两trial预算不变。旧包与首次失败证据保留，可由根代理完成同job源码修订登记、备份与单次部署。")
    with target.open("x",encoding="utf-8",newline="\n") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    require(load(target)["passed"],"Review reopened")
    require(record(cp)==cfg and record(mp)==mf and record(archive)==package,"Pins stable during review")
    print(json.dumps(dict(status=result["status"],review=record(target),**PINS),indent=2))


if __name__=="__main__":main()
