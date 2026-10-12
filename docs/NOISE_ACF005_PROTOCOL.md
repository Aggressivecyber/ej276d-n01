# N01 005: ACF objective coverage and bounded integration decision

This is the prospective execution protocol for the user's task “005 ACF 目标覆盖修复与后续整合边界”, reviewed against `0f914546564e37bd4bdfdb58302d6becbef1f02f`. It is not a completion receipt. Register this file and the configuration before any new scientific array decode, initialization or fit. Preserve all 002/003/004 files and negative conclusions.

## One scientific question and scope

With the exact 004 ObservedBaselineNoiseV3, canonical input, empirical TRAIN initialization, optimizer, sampler, batch size, random streams and 5000-update budget, does one additional ACF1..20 objective reduce the previously observed longer-lag residuals without losing the existing distributional/OLS gains?

No U-Net replacement, additional noise architecture, continuation from 004 final weights, seed search, adaptive checkpoint selection, rerun of the 003 raw origin audit, or rerun of the 128-fit null calibration is part of this run. Scientific noise updates: exactly 5000 if execution succeeds. Formal response updates: zero under every outcome. The model supports exactly 100 samples and continues to reject 480.

## Shared ACF definition

For each finite 100-point waveform, `b = x - mean(x)`, `v = max(mean(b^2), (1/16383)^2/12)`, and

`a_k(x) = mean(b[0:100-k] * b[k:100]) / v`, for k=1,...,20.

The numerator has denominator 100-k. Average these ratios across events; never replace this by a ratio of pooled covariance and pooled variance. The ADC floor, per-event demeaning, fixed 100-point support and lag convention are identical to the preserved NumPy `noise_validation.baseline_statistics`.

Add a new shared differentiable function and use it in the added objective and 005 diagnostics. Keep old `losses.py`, `losses_v2.py`, all 32-dimensional statistics buffers, and `noise_validation.py` unchanged. Artificial-data tests compare to the preserved NumPy implementation: quantized traces, finite white/colored noise, exact and near constants around the numerical floor, mixed amplitudes, offsets, per-event versus pooled averaging, float32/float64, finite gradients and gradcheck. No white-noise finite-window ACF=0 assertion. Fix max aggregation to scalar `torch.max(e.square())`, including its defined treatment of tied maxima.

## 005A: no-gradient TRAIN mechanism comparison

Decode all 21,047 legal canonical TRAIN baselines and their 422 original artificial record groups; do not decode target_val. Recompute the 004 initializer with unchanged full-TRAIN mean/covariance ddof1, ridge 1e-6 of mean variance, positive diagonal floor 1e-6, and initial scale spread .02. Bind and verify the saved 004 initial and final checkpoint hashes. Assert the newly constructed model state is exactly the saved 004 initial model state before fitting.

Use 8192 explicit latent events from a separate CUDA generator, seed `noise.generation_seed + 20000`, sampled in chunks of 256 with the unchanged model sampler. Save the actual Gaussian and scale latents. Apply the same explicit values to the reconstructed untrained initializer and frozen 004 final model. Save both pre-ADC and post-ADC outputs. This is a mechanism comparison; no alternative initializer is selected. Statistics always compare to unchanged canonical TRAIN in volts.

Save ACF1..20, full 100x100 empirical covariance (ddof1), time means/SD, mean scatter, centered-time OLS mean/slope covariance and TRAIN-scaled OLS relative error, event RMS quantiles and mean/slope correlation. Normalize full-covariance descriptive errors by the outer product of full-TRAIN per-time SD, each floored at one ADC step. No new covariance acceptance gate is introduced.

TRAIN is observed post-ADC data only; no analog TRAIN counterpart is available or manufactured. The initializer and 004-final comparisons are finite generated samples (8192 each) against 21047 observed TRAIN events, not exact population equality.

## 005B: one new objective and fixed endpoint

Keep `BaselineDistributionLoss` and all its weights unchanged. Compute one detached 20-value target `rho_R` from **all** legal TRAIN using the shared eventwise definition. Store it, its array digest and the input binding; never use validation values to fit or replace it. For the same quantized STE generator batch used by the old objective, define

`e = (mean_event_ACF20(generated_batch) - rho_R) / 0.015`

`L_acf20 = 0.5 * (mean(e^2) + max(e^2))`

`L_new = L_004 + 1.0 * L_acf20`.

The 0.015 denominator is a fixed existing error scale, not an altered gate. Optimize all 5151 original float64 parameters. Start from the newly recomputed initializer, with a new empty Adam state; do not load the 004 optimizer or final weights into the candidate.

Retain 004 noise configuration byte-equivalent as a nested object: batch 512, seed 2026101219, full-coverage sampler seed +101, explicit training-noise generator seed +102, projection seed 2026101220 with one context per update, initial LR .001, LR .0002 for updates 4001..5000, clip norm 5.0. All 5000 updates are mandatory; only the final checkpoint is evaluated. Every TRAIN event must have nonzero visits. New diagnostics use their own fixed latents and never advance the training sampler/RNG/global RNG or change the projection context.

For every update, record all five old objective components, their weighted total, added objective, total, all 20 signed ACF errors before scaling, ACF20 and old-objective parameter-gradient norms, their dot product and cosine/angle before clipping, and the combined preclip norm. Use the same batch, noise and context for both gradient calculations. A zero norm gives an explicitly undefined cosine/angle, not fabricated zero. Gradient logging does not rescale or project gradients. The actual optimizer gradient remains the exact sum of the two objectives, followed by the unchanged global clipping.

The primary angle is between the old objective gradient and the added ACF20 gradient, as requested. Also record the angle between old and total gradients to describe the actual update-direction change. Flatten the original model parameters in their unchanged named-parameter order; angles are in degrees. Loss/gradient entries for update u refer to the forward state before optimizer update u; a trajectory entry at u is after that update.

At updates 0,250,...,5000, use the exact same 8192 diagnostic latents as 005A to save pre/post-ADC full covariance and all the descriptive statistics above. Save the candidate's endpoint pre/post outputs for replay, but intermediate diagnostic outputs need not be duplicated. Record update 0 as untrained. These trajectories do not select weights or alter updates. Save model/optimizer/RNG/sampler states, initial state, loss statistics, diagnostics, all step logs and explicit replay values. The observation is post-selection/alignment baseline noise, not identified hardware electronics.

## Evaluation, control calibration and uncertainty

Reopen and validate the fixed final checkpoint and all bindings before decoding the existing 599-row development validation set. Its 12 artificial groups come from the same previously viewed acquisition. It is not blind validation or 12 independent acquisitions.

1. **Implementation/role isolation:** verify identical initialization, immutable input/source hashes, TRAIN-only targets/scalers/normalizers, group disjointness, all-row coverage, unchanged frozen generator during evaluation, and exact saved-randomness replay. These are implementation checks, not distributional acceptance.
2. **Legacy gate:** run the unchanged 004 evaluation once with its fixed seeds/counts/thresholds and all six fixed classifiers, 200 updates each (1200 evaluation-only updates). Save its full original fields and result as `LEGACY_GATE.json`; preserve its original next-action field only as historical output, never execute it. Copy the gate into a separately scoped receipt with response integration false. Verify the three newly reproduced REAL–REAL scores, labels and group assignments against the hash-bound 004 outputs. No candidate can optimize these controls.
3. **Candidate screening:** separately report the ten legacy checks other than `negative_control_point` and `negative_control_interval_contains_half`, unchanged. The raw generated–REAL AUC point range [.4,.6] and 95% clustered interval fully inside [.35,.65] are a predeclared practical approximate-equivalence screen for this fixed classifier and development population, not proof that the distributions are statistically equal. Demean/detrend classifier views remain diagnostics. Keep all failed component checks and uncertainty intervals.
4. **Control calibration:** reuse the bound 003 128-refit report: legacy false alarms 9/128 and conditional permutation false alarms 7/128, with their saved Wilson intervals. Do not claim either method is superior or universally valid. Its calibration is conditional on fixed TRAIN/pseudo-test groups in one historical acquisition. Independently report the legacy raw REAL–REAL point/interval and whether its practical interval is contained in [.35,.65].
5. **One conditional group-randomization diagnostic:** for the fixed raw REAL–REAL saved scores, labels and parents, perform 9999 label permutations with fixed seed 2026131221, preserving each original test group's equal nonzero class quotas. Use `T=abs(AUC-.5)`, `>=` including ties and `p=(1+tail)/(9999+1)`. Save every null AUC, group quotas, random stream and score/input hashes. Report a Wilson interval for the Monte Carlo tail proportion separately from acquisition uncertainty. No classifier refits occur in this step. Alpha .05 is descriptive; p>.05 does not certify the measurement or erase the old interval failure. No other view is tested and no seed is retried.

The 003 permutation reference used 255 draws per control, whereas this single 005 diagnostic uses 9999. Its 7/128 rate is reference calibration of the former finite rule, not an empirically established false-positive rate for the latter. Wilson uncertainty for the current K/9999 is named `conditional_tail_probability_MC_Wilson95`, not a cross-acquisition or candidate-equivalence interval.

The control status remains `measurement_uncertain_single_exposed_acquisition` after successful implementation checks. `candidate_screen_met` may be reported independently; `scientifically_qualified`, `independent_acquisition_validated`, and `response_training_authorized` remain false. If implementation/role isolation or exact control reproduction fails, the measurement is invalid and the task stops for engineering investigation. Calibration weakness never triggers further candidate training.

Save a new `EVALUATION_RECEIPT.json` with separate `implementation_checks`, `legacy_gate`, `candidate_checks`, `control_calibration`, `decision`, and support-scope fields. Also save the legacy and new per-lag errors against both TRAIN and development validation. Provide group bootstrap uncertainty for the fixed endpoint ACF differences: 1000 resamples, seed 2026131222; resample all TRAIN/VAL original groups with replacement and generated events separately, use precomputed per-event ACF20 sums/counts, and report marginal 95% intervals and simultaneous 95% centered-max bands. These intervals are descriptive, do not introduce a new gate, and are conditional on this fixed trained model/one acquisition. They do not include training randomness.

The TRAIN comparison uses the fixed 8192-event diagnostic population and seed 2026131222; the development comparison uses the unchanged legacy 8192-event evaluation population and seed 2026131223. These generated populations and their generation streams remain distinct and are labeled in all per-lag tables.

## Decision exits and 100/480 boundary

For the same fixed 8192 diagnostic latents, define TRAIN match as max absolute ACF1..20 difference <= .015. Define improvement over 004 as a lower ACF20 objective and no greater maximum absolute TRAIN ACF error. Both are declared before 005A results; no covariance/OLS trajectory gate is added.

Use strict `<` for objective decrease and `<=` for the maximum comparison, without an added numerical tolerance or adaptive threshold.

- TRAIN match plus improvement plus all ten development candidate checks: report only improved 100-point observed-noise candidate; proceed to a **read-only 480-point integration design check**, with measurement uncertainty disclosed. No formal pulse training is released.
- TRAIN mismatch: stop this candidate and use untrained/004/005 comparisons, gradients and covariance trajectory to assess competing explanations such as objective conflict or model-family limits. Do not automatically increase steps or size.
- TRAIN match but development candidate failure: report TRAIN–VAL shift, generated-sample uncertainty and original-group intervals; no validation target replacement.
- Only the legacy control interval fails: preserve that failure, keep candidate and measurement status separate under the rules above; never retry the candidate to repair an unrelated control statistic.

Any read-only integration check first inspects already-bound acquisition metadata and the preserved origin audit's window geometry for actual long silent/pretrigger support. It does not repeat the raw origin audit, select quieter events, relabel a pulse tail as noise, tile/repeat the 100x100 covariance, or fabricate missing silence. If no defensible long-silent truth exists, document that fact and outline explicit long-time covariance hypotheses and Q/PSD/variance sensitivity checks for a future protocol. No new 480-point law or sensitivity simulation is silently adopted here. Observed noise, effective full response and physical noise decomposition are distinct claims; exploratory full-response research may use explicit hypotheses under a separately frozen protocol without claiming physical identification.

## Execution and delivery

One scientific process at a time on the verified server, CPU threads 12, CUDA allocation 10 GiB, owned process-tree RSS 8 GiB, wall time 1800 s, scientific output 512 MiB. Preflight process identity/resources before launch; stop on hash mismatch, nonfinite values, resource violation or unsupported lengths. Preserve failure states and count completed updates accurately. Artificial tests never read REAL arrays or count as scientific training.

Snapshot registries on E before editing. Keep all source/data/logs/caches/reports on E/D, plus a separately bound remote run directory. Preserve originals. Reopen final checkpoints, reproduce generator outputs/classifier scores and reconstruct metrics. Render and visually inspect the report. Deliver shared ACF definition/tests, 005A arrays and comparisons, unique 005B endpoint/all logs, per-lag table, separated gate/calibration receipts, explicit unexecuted work and 100/480 limits. Back up privately to E/D with reopened-member/SHA256 verification; publish only source and derived results to the already authorized N01 repository, excluding REAL waveform arrays.

Method context: [Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545), [Multi-level block permutation](https://pubmed.ncbi.nlm.nih.gov/26074200/). The finite implementation, retained role boundary and limited uncertainty interpretation above are the run's operative rules.
