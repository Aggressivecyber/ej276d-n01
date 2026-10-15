"""Bounded016 pre-G failure diagnosis from local JSON/journal/source only."""
from __future__ import annotations
import ast
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import tarfile

ROOT=Path("E:/GAN/research/N01")
STAGE=ROOT/"receipts/wp3_016_20261015"
OUT=STAGE/"launch_fix02"
COLLECTED=ROOT/"runs/wp3_016_20261015/failed_collected_01"
ART=COLLECTED/"runs/primary/artifacts"


def require(value,message):
    if not value:raise AssertionError(message)


def record(path):
    require(path.suffix.lower() not in {".npz",".npy",".pt",".pth",".ckpt",".safetensors",".bin"},"No raw array/checkpoint reads")
    raw=path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def load(path):return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    target=OUT/"FAILURE_DIAGNOSIS_REVIEW.json";require(not target.exists(),"Preserve prior diagnosis")
    state=load(STAGE/"start_verification_01/REMOTE_STATE.json");summary=load(STAGE/"native_execution_02/NATIVE_START_SUMMARY.json")
    failed=state["runs/primary/artifacts/FAILED.json"];native=state["runs/primary/supervisor/exit_receipt.json"]
    require(summary["producer_error"]==failed["error"]=="016 per-parent A canonical Q changed across draws","Exact failure")
    require(native["child_native_exit_code"]==native["exit_code"]==1 and failed["counters"]["G_updates"]==failed["counters"]["new_AD_queries"]==0,"PreG native1")
    require(failed["counters"]==failed["attempted_counters"]==summary["actual_completed_counters"],"Counter closure")
    require(failed["operations_entered"]==failed["operations_completed"]==2929 and not failed["inflight"],"2929 operations complete despite later uncharged assertion")
    require(state["runs/primary/artifacts/FIRST_UPDATE_PROOF.json"] is None and all(p["alive"] is False for p in state["live_processes"].values()),"No first update and owned processes exited")
    collection=load(STAGE/"failure_collection_01/COLLECTION_RECEIPT.json");inventory=load(STAGE/"failure_collection_01/REMOTE_FULL_INVENTORY.json")
    require(collection["selected_file_count"]==21 and collection["whole_remote_file_count"]==22 and collection["partial_evidence_only"],"Partial21of22")
    require(set(inventory["omitted_remote_inventory"])=={"runs/primary/artifacts/seed0_Q1/initial_evaluation/bank0_BASELINE100.npz"},"Only baseline raw NPZ omitted")
    require(load(ART/"FAILED.json")==failed,"Partial failed JSON equals only snapshot")
    cal=load(ART/"CHARGE_INITIALIZATION.json")
    require(cal["parents"]==6593 and cal["source_parent_rows"]==list(range(6593)) and cal["all_Q_mu_finite_strictly_positive"],"Calibration full positiveQ")
    actual_rms=cal["actual_A_expected_increment_RMS_V"];target_rms=cal["target_RMS_V"]
    require(abs(actual_rms/target_rms-1)<1e-14,"Saved analytical1LSB calibration")
    log=(COLLECTED/"runs/primary/console.log").read_text(encoding="utf-8")
    require("line 312, in evaluate" in log and "np.all(mu==mu[:,:1])" in log,"Failure traceback at reducer exact equality")
    package=STAGE/"runtime_package_fix01/wp3_016_source.tar.gz"
    with tarfile.open(package,"r:gz") as archive:
        raw_driver=archive.extractfile("scripts/run_wp3_016.py").read()
    driver_sha=hashlib.sha256(raw_driver).hexdigest()
    require(driver_sha=="4729208d66c61c3b082827535719e72d8d7bb2ee60f390b688111ae3b4f12972","Diagnose exact failed driver, independent of future edits")
    ast.parse(raw_driver)
    original013=(ROOT/"scripts/run_wp1_013.py").read_text(encoding="utf-8-sig")
    require("BASELINE_NUMERICAL_TOLERANCE_V=1e-7" in original013 and "broadcast_or_mean_substitution=False" in original013,"Existing013 engineering gate with preservation")
    p013=load(ROOT/"wp1_013_config.json")
    def find_registered(x):
        if isinstance(x,dict):
            if x.get("numerical_A_replay_max_abs_V")==1e-7 and x.get("A_storage")=="all_actual_float64_draws_per_bank_no_broadcast":return True
            return any(find_registered(v) for v in x.values())
        if isinstance(x,list):return any(find_registered(v) for v in x)
        return False
    require(find_registered(p013),"Registered013 tolerance/storage provenance")
    P,D,C=6593,4,128;rows=P*D
    parents=[i//D for i in range(rows)];draws=[i%D for i in range(rows)]
    chunks=[(i,min(i+C,rows)) for i in range(0,rows,C)]
    require(all(i%D==0 and (j-i)%D==0 for i,j in chunks),"No parent crosses chunk")
    require(all(parents[i*D:(i+1)*D]==[i]*D and draws[i*D:(i+1)*D]==list(range(D)) for i in range(P)),"Manufactured full integer lineage")
    counts=failed["counters"]
    require(counts["old_A_rows"]==P+rows and counts["head_total_kernel_rows"]==counts["noise005_rows"]==counts["charge_module_rows"]==counts["charge_latent_scalar_draws"]==rows
            and counts["hard_ADC_rows"]==counts["all_path_gather_rows"]==rows*3*3,"Exactly one complete initial bank before reducer")
    init_ops=math.ceil(P/256)+1+3+1;eval_ops=math.ceil(rows/C)*(5+9)
    require(init_ops+eval_ops==2929,"31 initialization plus2898 bank operation arithmetic")
    pending={};complete=collections.Counter();entered=finished=0
    for line in (ART/"operations.jsonl").read_text(encoding="utf-8").splitlines():
        j=json.loads(line);op=j["operation_id"]
        if j["event"]=="entered":require(op==entered,"Sequential unique journal enter");pending[op]=(j["name"],j["charges"]);entered+=1
        else:
            require(j["event"]=="completed" and pending.pop(op)==(j["name"],j["charges"]),"Matching operation completion")
            complete.update(j["charges"]);finished+=1
    require(not pending and entered==finished==2929 and all(complete[k]==v for k,v in counts.items()),"Collected journal matches every actual count")
    u=2**-53;gamma255=255*u/(1-255*u);Vtol=1e-7;Qbound=256*Vtol
    base=[0.0]*100+[0.05]*284
    tiny=base.copy();large=base.copy();cancel=base.copy()
    for i in range(110,366):tiny[i]+=0.5e-7;large[i]+=2e-7
    cancel[110]+=2e-7;cancel[111]-=2e-7
    def delta(a):return max(abs(x-y) for x,y in zip(a,base))
    def q(a):return math.fsum(a[110:366])
    artificial=[dict(case="exact_A",A_residual=delta(base),gate=True),
                dict(case="sub_tolerance_actual_A_but_Q_not_bit_equal",A_residual=delta(tiny),Q_difference=q(tiny)-q(base),gate=delta(tiny)<=Vtol),
                dict(case="A_over_registered_gate",A_residual=delta(large),gate=delta(large)<=Vtol),
                dict(case="large_cancelling_shape_error_not_caught_by_Q_only",A_residual=delta(cancel),Q_difference=q(cancel)-q(base),gate=delta(cancel)<=Vtol)]
    require(artificial[1]["gate"] and artificial[1]["Q_difference"]!=0 and not artificial[2]["gate"] and not artificial[3]["gate"],"Manufactured scalar waveform counterexamples")
    previous_caps=load(COLLECTED/"wp3_016_config.json")["budget_caps"]
    replacement_aggregate={k:counts[k]+v for k,v in previous_caps.items()}
    result=dict(schema="n01/wp3-016-preG-failure-independent-diagnosis/v1",UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_id="wp3_016_20261015",status="DIAGNOSIS_COMPLETE_MEASURED_REPEAT_MAGNITUDE_UNKNOWN",passed=True,
        reviewer=dict(agent="/root/wp3_015_solmax_review",assigned_model="GPT-6.1-Sol",assigned_effort="Max",runtime_model_metadata_independently_exposed=False),
        conclusion_zh="真实失败定位于初始bank0完成后的报告reducer严格Q相等检查，尚未G/AD。父/draw顺序和chunk划分静态正确；实际差值未落盘，不能确认仅为浮点舍入。可用013已登记的1e-7V实际A重复残差门作最小工程修复，保留逐draw实际A/mu/v，先落盘再门判，禁止替代或广播。",
        confirmed_execution=dict(source_commit=summary["source_commit"],pins=summary["pins"],worker_pid=failed["pid"],supervisor_pid=native["supervisor_pid"],
                                 failure_UTC=failed["UTC"],native_exit_UTC=native["UTC"],snapshot_UTC=state["UTC"],native_child_exit=1,native_supervisor_exit=1,
                                 G_updates=0,AD_queries=0,optimizer_steps=0,actual_counts=counts,scientific_compute_and_RNG_cost_nonzero=True,
                                 operations_entered=2929,operations_completed=2929,inflight=[],failed_operation_ids=[],truncated_tail=False,
                                 explanation="The failing require is outside Store.call, after all charged bank operations completed. Thus native failure with an all-completed journal is consistent."),
        lineage_and_batch_analysis=dict(source="failed frozen driver evaluate lines282,287-314",parent_order="repeat(arange6593,4)",draw_order="tile(arange4,6593)",
                                       append_and_reshape="Each sequential chunk appends q/mu, then concatenate and reshape(6593,4).",
                                       evaluation_chunk_rows=128,full_chunks=206,last_chunk_rows=4,parents_per_full_chunk=32,parent_crosses_chunk=False,
                                       actual_array_lineage_replayed=False,
                                       inference_zh="没有静态错位或父跨chunk证据。因而不能把本次失败直接解释为同父跨不同batch大小；真实设备行位置/算子数值差异仍属候选解释。"),
        model_path_analysis=dict(Q_mu="canonical_charge008(actual frozen A signal_pre) in float64 before charge multiplier",
                                 actual_A_path="Frozen deterministic baseline with zero physical/local latents, sample-wise GroupNorm/Conv/FiLM and no observed BatchNorm/Dropout path; new shape/charge latents do not setQ_mu.",
                                 mean_rows_preserved_in_model=True,charge_denominator_floor=False,
                                 exact_mu_comparison_implies="At least one saved-in-memory per-draw Q differs from its parent first draw; finite positive output guards were already passed.",
                                 does_not_imply="No magnitude, affected parents, waveform residual, ULP profile or CUDA-kernel cause follows from the boolean failure."),
        saved_calibration=dict(parents=cal["parents"],S0_V2=cal["S0_V2"],Sminus_V2_per_Vns=cal["Sminus_V2_per_Vns"],
                               a_initial_Vns=cal["a_initial_Vns"],b_initial=cal["b_initial"],Q_mu_min_Vns=cal["Q_mu_min_Vns"],Q_mu_max_Vns=cal["Q_mu_max_Vns"],
                               actual_A_expected_increment_RMS_V=actual_rms,target_RMS_V=target_rms,
                               inference="One-parent-per-row calibration passed positive Q and analytical1LSB initialization. It does not quantify the later duplicate-draw A/Q residual."),
        partial_evidence=dict(selected_files=21,whole_files=22,omitted=list(inventory["omitted_remote_inventory"]),
                              actual_A_or_mu_repeat_arrays_saved=False,original_bad_repeat_magnitude="unknown_not_recoverable_from_collected_metadata",
                              save_order="bank0 BASELINE100 was written before the strict check; CHARGE_PREADC_MOMENTS, CHARGE_SEMANTICS, HARD_ROWS and REPORT saves follow the check and were not reached.",
                              original_failure_preserved=True),
        engineering_gate_recommendation=dict(reuse_existing_gate=True,numerical_A_max_abs_residual_V=Vtol,
            provenance=["scripts/run_wp1_013.py:21 and25-63 A_numerical_residual", "wp1_013_config.json:3502-3504 original numerical storage repair"],
            applicability="Same frozen A on the same exact SOURCE-parent/draw identities, full384 actual float64 voltage rows. Reuse the helper unchanged; the limit is execution replay tolerance, separate from scientific response T.",
            required_save_before_gate=["Actual A384 rows for every parent/draw, actual model Q_mu per draw and q_sig per draw, per-draw v=a*mu+b*mu², and parent/draw/ID metadata", "A repeat diagnostic summary with max/RMS/bit differences and affected identities, input byte-identity and exact lineage status"],
            exact_guards="Verify source IDs unique, repeat/tile order, count/dtype/finite-positive fields, and exact same-parent SOURCE input bytes separately; failure remains hard. Use all actual outputs without substitution.",
            numeric_guard="Write evidence first, then fail if any full-A point residual exceeds the existing1e-7V gate. A Q-only allclose is insufficient: cancelling waveform errors can hide in the integral.",
            prohibited=["No default np.allclose or unregistered tolerance", "No change to actual A, no broadcasting first draw or mean A", "No substitution of mu[:,0] or mean(mu) into q law or its denominator", "No claim that the already failed attempt was within tolerance"],
            extra_NN_or_random_needed_for_diagnostics=0,scientific_law_seeds_loss_T_and_fixed4000G_design_changed=False),
        canonical_Q_bound=dict(canonical_half_open_slice=[110,366],terms=256,software_dt_ns=1,
            exact_real_arithmetic="|Q_d-Q_ref| <= sum_[110:366)|A_d-A_ref| <=256*max_t|A_d-A_ref|",
            derived_max_from_A_gate_Vns=Qbound,
            finite_sum_note="For stored float64 A and a normal IEEE round-to-nearest sum without overflow/subnormal exceptions, include a conservative sum-rounding term gamma255*(L1(A_d)+L1(A_ref)); gamma255=255u/(1-255u),u=2^-53. Record actual Q residual and actual reduction-consistency residual. Do not turn the derived integral bound into an unrelated learned/scientific tolerance.",
            double_unit_roundoff=u,gamma255=gamma255,failed_run_actual_Q_error="unknown"),
        correct_draw_specific_statistics=dict(conditioning="Condition on every retained actual mu_d and independent charge z_d, D=4.",
            row_moments="E(q_d)=mu_d; v_d=Var(q_d)=a*mu_d+b*mu_d².",
            report_mean_only="E(mean_d q_d)=mean_d mu_d; this descriptive report center never replaces actual per-row modelmu.",
            expected_sample_variance_ddof1="E[s_q²]=mean_d(v_d)+samplevar_d(mu_d,ddof=1). The second term is deterministic numerical row-mean spread and must not be labelled physical charge variance.",
            theoretical_sample_mean_MCSE="sqrt(sum_d v_d)/D; variance of sample mean=sum_d v_d/D². No samplevar(mu) term is added when conditioning on the retained deterministicmu layout.",
            draw_specific_residual_second_moment_ddof0="mean_d((q_d-mu_d)²) has expectation mean_d(v_d); do not subtract the realized residual mean or relabel the numericmu spread as physical variation.",
            empirical_MCSE="Keep sqrt(samplevar(q,ddof=1)/D) if reported, label it the empirical draw-based value and distinguish it from the analytical conditionalMCSE.",
            inference_limit="Repeated model draws remain Monte Carlo samples of the generator. Unpaired REAL does not supply same-input conditional variance truth."),
        artificial_checks_executed=dict(method="stdlib manufactured integer lineage, count arithmetic and non-neural hand-written384-point arrays only",
                                        lineage_all6593=True,chunk_count=207,initialization_operations=init_ops,evaluation_operations=eval_ops,
                                        total_operations=init_ops+eval_ops,waveform_counterexamples=artificial,new_NN_or_random_or_original_arrays=0),
        focused_artificial_tests_for_next_patch=["Exact repeats, sub1e-7 A perturbations, above-gate A perturbations and equal-Q cancelling shape errors; persist diagnostic before any rejection", "Parent/draw permutation, input-byte mismatch, missing row, wrongdtype and nonfinite/nonpositive fields must reject exactly", "Keep actual per-drawmu and v untouched, verify mean(v)+samplevar(mu) and sqrt(sum(v))/D with controlled unequalmu arrays", "Run evaluator save path on manufactured arrays without CP/real data/newNN; verify saved A/mu/v and failure evidence order"],
        restart_review_boundary="This is a diagnosis review, not a repaired-source/frozen-package or restart approval. Root has separately authorized a minimal driver storage/reducer and new runtime-root repair; its final diff/tests/pins require independent review. No retrospective rerun of the failed private arrays is needed to choose the already registered engineering limit.",
        replacement_accounting=dict(existing_failure_counts=counts,planned_replacement_caps=previous_caps,
                                    failure_plus_one_full_replacement_aggregate_caps=replacement_aggregate,
                                    original_scientific_seed_trials=2,total_G_budget=4000,
                                    old_nontraining_compute_RNG_CP_not_erased=True,cumulative_G_before_and_after_failure=175667),
        evidence=[record(STAGE/"start_verification_01/REMOTE_STATE.json"),record(STAGE/"native_execution_02/NATIVE_START_SUMMARY.json"),
                  record(STAGE/"failure_collection_01/COLLECTION_RECEIPT.json"),record(STAGE/"failure_collection_01/REMOTE_FULL_INVENTORY.json"),
                  record(ART/"CHARGE_INITIALIZATION.json"),record(ART/"FAILED.json"),record(ART/"operations.jsonl"),record(COLLECTED/"runs/primary/console.log"),
                  record(ROOT/"scripts/run_wp1_013.py"),record(ROOT/"wp1_013_config.json"),record(ROOT/"src/n01/model008.py"),record(ROOT/"src/n01/model010.py"),
                  record(ROOT/"src/n01/model016.py"),record(ROOT/"src/n01/model.py"),record(Path(__file__))],
        failed_driver_source=dict(package_path=package.relative_to(ROOT).as_posix(),member="scripts/run_wp3_016.py",sha256=driver_sha,source_read_only=True),
        actual_review_operations=dict(stdlib_only=True,original_NPZ_NPY_reads_or_decodes=0,checkpoint_payload_reads_or_loads=0,new_NN_rows=0,new_AD_queries=0,
                                      new_random_draws=0,training_updates=0,new_calibration_fits=0,SSH_calls=0,GPU_work=0,source_or_meta_or_oldpins_edits=0,
                                      output_root=OUT.as_posix()),
        scientific_acceptance=False,weights_promoted=False,scientific_status="response_match_not_established")
    with target.open("x",encoding="utf-8",newline="\n") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    require(load(target)["passed"],"Diagnosis reopened")
    print(json.dumps(dict(status=result["status"],review=record(target),cause="unknown",existing_A_tolerance_V=Vtol,Q_bound_from_A_Vns=Qbound),indent=2))


if __name__=="__main__":main()
