"""Read-only 016 design review; stdlib arithmetic and source/JSON hashing only."""
from __future__ import annotations

import ast
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path("E:/GAN/research/N01")
OUT = ROOT / "receipts/wp3_016_20261015/independent_01"
DESIGN = ROOT / "wp3_016_design.json"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def record(path):
    require(path.suffix.lower() not in {".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors"}, "No raw array or checkpoint read")
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    destination = OUT / "DESIGN_REVIEW.json"
    require(not destination.exists(), "Preserve prior reviews")
    d = json.loads(DESIGN.read_text(encoding="utf-8-sig"))
    require(d["status"] == "implementation_design_not_frozen_no_scientific_job_started", "Design is not frozen or running")
    require(d["model_policy"] == {"model": "gpt-6.1-sol", "reasoning_effort": "max"}, "Latest assigned model policy")
    require(d["trials"]["seeds"] == 2 and d["trials"]["G_updates_each"] == 2000
            and d["trials"]["total_G_updates"] == 4000 and d["charge"]["trainable_scalars"] == 2, "One two-parameter candidate, two trials")
    require(d["evaluation"]["initial_arms"] == ["A", "M0", "initial"] and d["evaluation"]["final_arms"] == ["final"], "Initial three arms and final one arm")
    require(d["training"]["learning_rate"] == .01 and d["training"]["weight_decay"] == 0
            and d["training"]["new_AD_per_update"] == 1 and d["training"]["probe_AD"] == 0, "Fixed optimizer/AD design")
    require(d["charge"]["latent_shape_charge_independent"] and d["charge"]["electronic_constant_variance"] == 0, "Independent charge latent and no repeated electronic variance")
    require(d["initialization"]["target_expected_preADC_384_RMS_ADC_LSB"] == 1
            and d["initialization"]["zero_prefix100_in_RMS_denominator"]
            and d["initialization"]["actual_output_Q_positive_all_parents_required"], "Analytical full-A initialization scope")
    require(d["selection"]["weights_promoted"] is False and d["selection"]["scientific_status"] == "response_match_not_established", "Research reference only")
    require(d["frozen"]["mean_adapter"] == "disabled" and all(d["frozen"][k] for k in ["A", "noise005", "per_seed015_M0_final_shape_head", "C1_integral_statistics", "observation_contract"]), "Protected components stay frozen")

    p, draws, banks, hypotheses, seeds, updates, minibatch, train_draws = 6593, 4, 2, 3, 2, 2000, 512, 2
    train_rows = seeds * updates * minibatch * train_draws
    eval_shared_rows = seeds * banks * p * draws * 2
    reused_forward_rows = train_rows + eval_shared_rows
    hard_train_rows = train_rows * hypotheses
    hard_eval_rows = seeds * banks * hypotheses * p * draws * (3 + 1)
    expected = {
        "calibration_A_rows": p, "old_A_rows": reused_forward_rows + p,
        "head_actual_parameter_forward_rows": reused_forward_rows, "head_total_kernel_rows": reused_forward_rows,
        "noise005_rows": reused_forward_rows, "charge_module_rows": reused_forward_rows,
        "charge_latent_scalar_draws": reused_forward_rows, "charge_initialization_fits": 1,
        "hard_ADC_rows": hard_train_rows + hard_eval_rows, "all_path_gather_rows": hard_train_rows + hard_eval_rows,
        "checkpoint_loads": 4, "optimizer_created": 2, "optimizer_steps": 4000, "G_updates": 4000,
        "training_backward_queries": 4000, "new_AD_queries": 4000, "probe_AD_queries": 0,
        "integral_statistics_fits": 0, "mean_adapter_rows": 0, "D_updates": 0, "noise_updates": 0,
        "population_updates": 0, "C2ST_updates": 0, "DEV_model_rows": 0,
    }
    require(expected == d["budget_caps_design"], "Every design budget entry matches independent count derivation")
    source_paths = [
        ROOT / "src/n01/model008.py", ROOT / "src/n01/model010.py", ROOT / "src/n01/model014.py",
        ROOT / "src/n01/model015.py", ROOT / "src/n01/losses008.py", ROOT / "src/n01/losses014.py",
        ROOT / "src/n01/response008.py", ROOT / "scripts/run_wp3_015.py",
    ]
    trees = {p.relative_to(ROOT).as_posix(): ast.parse(p.read_text(encoding="utf-8-sig")) for p in source_paths}
    require(len(trees) == 8, "Current source parsed without importing scientific modules")
    adc_tree = trees["src/n01/losses008.py"]
    adc_assignment = [n for n in adc_tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "ADC_STEP" for t in n.targets)]
    require(len(adc_assignment) == 1 and ast.unparse(adc_assignment[0].value) == "1.0 / (2 ** 14 - 1)", "Bound ADC_STEP expression")
    review = {
        "schema": "n01/wp3-016-independent-design-review/v1",
        "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": d["run_id"], "status": "PASS_IMPLEMENTATION_DESIGN", "passed": True,
        "reviewer": {"agent": "/root/wp3_015_solmax_review", "assigned_model": "GPT-6.1-Sol", "assigned_effort": "Max",
                     "assignment_evidence": "Parent assigned an independent read-only design review under the latest user all-future-work 6.1solmax instruction.",
                     "runtime_model_metadata_independently_exposed": False},
        "reviewed_design": record(DESIGN), "design_frozen": False, "scientific_job_started_by_this_review": False,
        "recommendation": "Proceed to one bounded implementation using mature014/015 engineering; freeze exact source/config/data roles before any scientific start.",
        "conclusion_zh": "016设计通过实现前独立审查：两参数正值、解析均值校正的q_sig可以补足现有零canonical-Q形状分支未提供的电荷自由度；初值校准与全部设计计数一致。该设计接受后可施工，科学效果由冻结硬观察评价决定。",
        "checks": [
            {"check": "positive_mean_corrected_lognormal_law", "passed": True,
             "proof": ["For Q>0, a,b>=0 and s2=log1p(a/Q+b), E[exp(sZ-s2/2)]=1.",
                       "E[q|x]=Q; Var(q|x)=Q^2*(exp(s2)-1)=aQ+bQ^2.",
                       "The result concerns preADC conditional expectation; the nonlinear hard observation can alter output means."],
             "finite_draw_note": "Neither four/eight draws nor antithetic Gaussian Z pairs exactly enforce the lognormal sample mean. Retain MCSE and the analytical correction; no sample-mean subtraction."},
            {"check": "actual_canonical_A_charge_and_units", "passed": True,
             "source": "src/n01/model008.py: canonical_charge008 is the signal-only linear sum over [110:366) on the bound1ns grid.",
             "implementation_requirement": "Use actual frozen A signal_pre in float64; check every one of6593 legal parents for finite strictly positive canonical Q. Do not substitute qbar, observed Q, a surrogate denominator or a floor. Fail with preserved evidence on violation.",
             "current_runtime_Q_positive_verified": False,
             "units": "Q and a are V*software_ns; b and a/Q+b are dimensionless; q_reference only scales the positive parameterization."},
            {"check": "assembly_shape_independence_and_off_exact_bytes", "passed": True,
             "proof": "Let S0=A+B*eta. The proposed S=S0+A*expm1(sZ-s2/2) equals (q/Q)*A+B*eta; the B term retains original A/input amplitude and is not multiplied by q.",
             "source_facts": ["model010 amplitude=.05*max_abs(original A signal_pre); canonical projection and zero prefix100 remain the mature014/015 path.",
                              "model015 M0 keeps the mean adapter disabled and returns its inherited C1 signal path."],
             "implementation_requirement": "The charge-off branch must select/return the original M0 tensor bytes directly; adding a numerical zero is insufficient for signed-zero byte identity. Keep old stream addresses unchanged, with no extra charge draw for A/M0 reference arms.",
             "canonical_Q_numerical_note": "Report floating-point residuals in canonical charge and expected mean, including the inherited finite projection residual; do not promise bitwise equality of an active random sum to the analytical identity."},
            {"check": "one_full_SOURCE_analytic_one_LSB_initialization", "passed": True,
             "derivation": "E[(S-S0)^2|x,t]=A(x,t)^2*(a/Q(x)+b). With S0=mean_parent,384(A^2) and Sminus=mean_parent,384(A^2/Q), expected squared RMS=a*Sminus+b*S0. a0=.5*ADC_STEP^2/Sminus and b0=.5*ADC_STEP^2/S0 therefore give ADC_STEP^2 exactly in algebra.",
             "calibration_scope": "One frozen A forward over6593 parents, including the zero prefix100 in the384 denominator; no head/noise/ADC/random/AD in calibration. Reuse this same A-dependent initial pair across both per-seed M0 heads.",
             "implementation_requirement": "Require finite strictly positive S0, Sminus, a0, b0 and actual positive finite softplus coefficients. Use stable inverse softplus and float64. Report expected RMS, per-input CV and realized initial evaluation RMS; do not tune to realized Monte Carlo draws.",
             "actual_calibration_executed": False},
            {"check": "two_softplus_parameters_and_fixed_optimizer", "passed": True,
             "parameterization": d["charge"]["parameters"], "learning_rate": .01, "weight_decay": 0., "clip_norm": 5.,
             "reason": "A strictly positive analytical initialization avoids the sqrt derivative singularity at a=b=0 and the zero gradient of a squared zero initializer. Independent charge latent and exactly theta_a/theta_b keep the intended factor isolated.",
             "implementation_requirement": "Freeze all A/005/M0 parameters and buffers, whitelist exactly the two scalar parameters in AdamW, and guard finite law/objective/gradients/updates without silent clipping of q or row removal."},
            {"check": "unchanged_loss_and_regularization_meaning", "passed": True,
             "objective": d["training"]["objective"],
             "source_resolution_of_mean_saturation_penalty": "run_wp3_015.py step forms torch.stack([saturation_penalty(signal+noise[h]) for h in HYPOTHESES]).mean(). The016 phrase mean_saturation_penalty should preserve this arithmetic across the same three hypotheses.",
             "implementation_requirement": "frozen_shape_regular is the inherited regularization constant with no opened shape gradient; new mean(a/Q+b) is the explicit dimensionless charge variance regularizer. Keep the existing C1 integral scales/projection and original hard output metrics."},
            {"check": "single_factor_trials_evaluation_reuse_and_exact_budget", "passed": True,
             "trials": d["trials"], "initial_arms": ["A", "M0", "initial"], "final_arms": ["final"],
             "derivation": {"training_source_row_calls": {"formula": "2*2000*512*2", "value": train_rows},
                            "shared_evaluation_source_row_calls": {"formula": "2seeds*2snapshots*2banks*6593parents*4draws", "value": eval_shared_rows},
                            "A_head_noise_charge_common_rows": reused_forward_rows,
                            "old_A_with_unique_calibration": reused_forward_rows+p,
                            "training_hard_ADC_gather_rows": {"formula": "4096000*3hyp", "value": hard_train_rows},
                            "evaluation_hard_ADC_gather_rows": {"formula": "2seeds*2banks*3hyp*6593*4draws*(3initial_arms+1final_arm)", "value": hard_eval_rows},
                            "total_hard_ADC_gather_rows": hard_train_rows+hard_eval_rows,
                            "CP": "A+005+one015-M0-head_per_seed=4; all loads shared/cached once", "G_AD": "2*2000=4000 each"},
             "budget_all_fields_independently_equal": True,
             "implementation_requirement": "Initial A/M0/charge-on must reuse the same single A/head/noise realization per parent/bank, not run three independent source forwards. Final has only the charge-on arm. Each of three hypotheses still requires its own hard ADC/gather. Extra forwards, charge draws or calibration passes require corrected counts before freezing."},
            {"check": "research_selection_and_usefulness_are_separate", "passed": True,
             "fixed_rule": "For each seed/hyp cell, average the two separate bank distances; every positive-defined required T must satisfy final-A<=T and final-M0<=T. Missing/undefined required components are pending and ineligible; zero/undefined T remains unresolved. An eligible candidate replaces the M0 research reference only if mean12wave improves over M0 by more than1e-12; ties retain q-off M0.",
             "ALL6_2percent_vs_A_usefulness": "Report separately; this design does not make it an additional replacement gate.",
             "absolute_T": "Report separately; no scientific promotion from this selection.",
             "clarification_of_prior015_summary": "The015 Chinese next-step phrase about use/noninferiority was advisory shorthand. It does not change016's proposed rule or add ALL6 usefulness as an eligibility gate. This review accepts the016 rule exactly as written.",
             "optional_scientific_advice": "If Q1 loses one of M0's current6/6 usefulness cells while improving its mean12 score, explicitly report that tradeoff beside the research-reference choice; no automatic promotion to scientific acceptance."},
            {"check": "identification_and_failed_residual_boundaries", "passed": True,
             "evidence": "015 final independent selection/review: M0 usage6/6, absolute T fails; canonical-zeroQ shape misses a charge direction, while unpaired population Q width cannot identify actual conditional variance.",
             "scope": "Global two-coefficient q only. Keep peak/rise/FWHM and electronics DC as separately reported residuals. No time/local-noise/population/mean expansion, no electronics refit, no trained extra control, no DEV read, and no extension beyond the remaining two WP3 slots."},
        ],
        "budget_independent_derivation": expected,
        "findings": [], "blockers": [], "necessary_design_mutations": [],
        "implementation_freeze_items_already_in_scope": [
            "Bind exact selected015 M0 checkpoint/state hash per seed and immutable A/005 states without changing015 artifacts.",
            "Bind original SOURCE/REAL samplers, shape/noise/extension/projection addresses and the already named independent charge seeds; save RNG states and first real update proof using the mature controller.",
            "Keep off-byte QA, positive actual canonical A validation, analytical initialization receipt and exact operation charges as implementation semantics, not extra scientific gates.",
            "Use015 saturation penalty arithmetic, and resolve every planned evaluator/forward reuse exactly before frozen config/source/data-roles registration.",
            "On a completed two-trial run, close negative results without near-threshold extra training. After the first correct real update and same-snapshot health proof, set one future n01 wake and rest.",
        ],
        "implementation_or_native_validation_claimed": False,
        "actual_review_operations": {"method": "Read local design/JSON/source; stdlib AST parsing, SHA256 hashing and integer budget arithmetic; symbolic moment derivation.",
                                    "scientific_module_imports": 0, "original_NPZ_or_NPY_reads": 0, "checkpoint_tensor_loads": 0,
                                    "SOURCE_REAL_or_DEV_waveform_reads": 0, "A_or_NN_forwards": 0, "new_AD_queries": 0,
                                    "new_optimizer_or_sampler": 0, "training_updates": 0, "new_random_draws": 0,
                                    "new_calibration_fit": 0, "new_scientific_numeric_computation": 0,
                                    "SSH_or_remote_status_calls": 0, "GPU_work": 0, "design_or_model_source_edits": 0,
                                    "root_metadata_edits": 0, "new_output_root": OUT.as_posix()},
        "evidence": [record(DESIGN), record(ROOT / "USER_MASTER_PLAN_20261015.txt"),
                     record(ROOT / "receipts/wp3_015_20261015/postrun_independent_02/FINAL_INDEPENDENT_REVIEW.json"),
                     record(ROOT / "receipts/wp3_015_20261015/postrun_independent_02/FINAL_DEVELOPMENT_SELECTION.json")]
                    + [record(p) for p in source_paths] + [record(Path(__file__))],
        "weights_promoted": False, "scientific_acceptance": False, "scientific_status": "response_match_not_established",
    }
    with destination.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(review, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    reopened = json.loads(destination.read_text(encoding="utf-8"))
    require(reopened["passed"] and not reopened["blockers"], "Review reopened")
    require(record(DESIGN) == review["reviewed_design"], "Design unchanged during review")
    print(json.dumps({"status": review["status"], "review": record(destination), "design": review["reviewed_design"],
                      "checks": len(review["checks"]), "budget": expected}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
