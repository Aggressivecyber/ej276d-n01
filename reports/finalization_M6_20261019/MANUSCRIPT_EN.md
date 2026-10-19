# Readout-aware conditional waveform calibration: conserved reference integrals do not ensure native pulse-shape agreement

**Research manuscript draft — 19 October 2026.** The evaluated S candidate failed its native-readout protection criteria. Results below describe a bounded development experiment and its reproducible negative finding. Author information and a target journal have not been assigned.

## Abstract

Simulation-to-measurement waveform calibration must preserve detector observables after digitization, baseline subtraction, peak selection and dynamic windowing. We study a conditional calibration operator that couples a constrained mean correction with a linear transformation of a frozen stochastic residual. A cubic-spline correction preserves reference front and tail integrals, support and the deterministic peak, while a quadratic penalty discourages changes in neighboring readout windows. On a fixed development population of 6,593 source events and 21,047 measured records, the operator reduces a multiview waveform distance by 28.58–30.01% and a joint morphology distance by 58.61–59.12% across two frozen heads and three noise hypotheses. Nevertheless, all six conditions fail native charge–pulse-shape, front, tail, pulse-shape-ratio and negative-lobe protection. The mean correction compresses the leading peak gap below one ADC step for 5,374 source events; digitization alone reselects the peak for 4,460 events. Reference conservation therefore coexists with substantial changes in native readout. Complete saved outputs reproduce the original metrics, and a research API passes 523 byte-exact golden and replay checks. The result establishes a concrete limitation of this frozen constrained family and motivates treating native peak stability as a separate scientific requirement.

## 1. Problem and related work

An effective stochastic response maps a simulated event x to a distribution of observed pulses Y conditional on x. The measured marginal distribution mixes response variability with the source population. Matching that marginal can improve analysis outputs, but a conditional response claim additionally requires evidence about input dependence and repeated outputs for the same input. In this study, no measured event is paired with a simulated microscopic event. The objective is useful calibration under the existing acquisition and software observation definition.

CPU-Net is a direct detector-waveform comparator: Bhimani and colleagues use a CycleGAN framework for simulated-to-measured HPGe pulse translation. Its official implementation contains stochastic reparameterization, so a deterministic ordinary U-Net would not reproduce that reference. No CPU-Net adaptation was trained under the present protocol, and its reported HPGe performance is not used as a numerical baseline here. [Bhimani et al.](https://arxiv.org/abs/2507.09106v2), [fixed official implementation](https://github.com/aobol/CPU-Net/tree/c74144a2743eac8f1fc12c91d6fae8a036d018b2).

Scoring-rule training provides a separate nonadversarial route for generative prediction; its forecasting guarantees do not identify an unpaired detector kernel in this experiment. Classifier two-sample tests can test detectable marginal differences using held-out classification. A new frozen-endpoint test was planned but was not run for S. [Pacchiardi et al.](https://arxiv.org/abs/2112.08217v3), [Lopez-Paz and Oquab](https://arxiv.org/abs/1610.06545v4).

The present contribution is an implemented conditional calibration construction and a measured failure mechanism linking near-tied mean peaks to nonlinear hard readout. Comparative superiority and recovery of a physical stochastic kernel remain outside the available evidence.

## 2. Population and observation contract

The complete development population contains 6,593 source parents and 21,047 measured records from 422 logical groups in one historical acquisition. Source and measured populations are unpaired. Their original membership and weights are retained. Q denotes a voltage–software-time integral in V ns; it is not calibrated incident-particle energy. Parent identifiers track output ancestry and are excluded from neural conditioning.

The input is a prepared 384-sample clean source carrier. The final output contains 256 samples. The frozen observation chain applies the original 14-bit ADC, subtracts the mean of the first 100 digitized samples, selects the first maximum in carrier indices [112,144), and crops [p−16,p+240). Front and tail integrals use [p−16,p+30) and [p+30,p+240), respectively. The sampling interval is one software ns. Undefined timing features and censored observations retain their original states and denominators.

Two frozen response heads and three noise-extension hypotheses—fast, slow and common_level—define six sensitivity conditions. They share the measured acquisition and are not six independent experiments. M0 supplies the main historical reference. BASE054 supplies the deterministic mean used by the new correction and remains a limited-benefit reference. Candidate 070 supplies a prior morphology-improving comparison with failed protection. All four variants use the original FIT sampling convention and the same complete source population.

## 3. Conditional mean and random transport

Let m(x) be the BASE054 deterministic mean and p its reference peak. A frozen shared conditional profile and crossing-level time map define the target deformation. Target construction uses the source event and training-derived shared statistics, never an event-specific measured target. The active carrier interval [110,176) has 66 coordinates. Twenty-four cubic B-spline columns define a smooth local representation; they do not imply 24 free coordinates after constraints.

The correction vanishes outside the active interval, on excluded support, and at the two boundary values and adjacent samples. The latter conditions preserve discrete endpoint values and first differences. Reference front and tail sums are zero. An SVD of the scaled equality system defines a basis G for the legal subspace. A rank threshold of 10⁻¹² is fixed before candidate scoring. Thus d=Gz expresses the correction without redundant integral equations.

For target local change v, the mean problem is

\[
\min_z \frac12\|D(Gz-v)\|^2
+\lambda_s\|S_2Gz\|^2
+\lambda_r\sum_jw_j\|R_{p+j}Gz\|^2.
\]

D normalizes waveform displacement, S₂ penalizes discrete curvature, and R extracts neighboring-window Q and tail changes. The frozen scales are 0.15554293597289215 V, 3.66896364775571 V ns and 0.6947215402546918 V ns. A pre-candidate rule based on saved reference peak offsets selects radius four; feasible neighboring peaks remain within the original search interval. The fixed weights are λs=0.01 and λr=1.

Linear inequalities preserve positive mean support and deterministic peak order. Small scale-relative margins are bounded by the original gaps, retaining feasibility of zero correction. They preserve the floating-point peak but are much smaller than one ADC step. OSQP 1.1.3 solves the resulting convex problem with absolute and relative tolerances of 10⁻¹², polishing enabled and warm starting disabled. The assembled waveform is checked for integral closure, nonnegative allowed support and exact first-argmax preservation. This use follows the quadratic-program structure described by [Stellato et al.](https://arxiv.org/abs/1711.08013v4).

The original residual r is transported with the same input-conditioned map. The desired local change (W_x−I)r is projected into the legal smooth space in the frozen positive-definite metric. This gives r′=L_xr, where L_x depends on x and the deterministic construction, not on the realization of r. Consequently, conditional zero mean is preserved when it holds for the original residual, and Cov(r′|x)=L_x Cov(r|x)L_xᵀ. The mean problem's inequality active set is not recomputed for each random draw. These identities apply before electronics and ADC. The complete signal m′+r′ then enters the original electronics and hard observation chain.

## 4. Evaluation and reproducibility

The registered waveform distance combines frozen multiview W1 terms with weights 0.50 for sliced, 0.25 for coordinate and 0.25 for short-lag joint distances. Its views include voltage, charge-normalized shape, differences, curvature and smoothed differences. It is distinct from an earlier energy-based training objective. The joint morphology distance uses 21 original observables and frozen projection directions. Scalar distances retain V, V ns, ns or dimensionless ratio units as appropriate. The machine-readable [24-condition metric table](data/METRICS_ALL_CONDITIONS.csv) contains distances, whereas the [conditional table](data/CONDITIONAL_MEANS_AND_DENOMINATORS.csv) contains means and their actual denominators.

Useful point improvements require at least 2% in waveform and joint distances and at least one research-scale unit in average morphology. Native protection is evaluated separately. Research scales T were produced by earlier artificial perturbations and are not hardware tolerances. Protected distance increases relative to M0 are limited to 0.05T, with original anomaly and negative-lobe rules retained. Improvement in one distance cannot offset a failed protected component.

There is one frozen S structure, two original heads and no new neural training. The completed evaluation solves 13,186 mean QPs and stores 24 native result sets totaling 158,232 rows. A separate saved-output evaluation exactly reproduces all complete metric JSON files and verifies native waveform and ADC bytes and peak indices. It performs no response inference or new candidate search. The maximum reported reference-integral residual is 6.83×10⁻¹⁶ V ns.

## 5. Results

Across the six conditions, S improves waveform distance by 28.58–30.01%, joint morphology distance by 58.61–59.12%, and average morphology by 3.660–3.679T relative to M0. Every condition nevertheless fails QPSD, front, tail, guarded PSD and negative-lobe-area protection. The negative-lobe distance increase is 0.278–0.340 mV ns. The [protection table](data/PROTECTION_FAILURE_CARD.csv) retains raw distances, D/T and paired increments; the [negative-lobe table](data/NEGATIVE_LOBE_FAILURE_CARD.csv) records its separate zero-worsening rule.

S preserves the reference floating-point peak, but 5,374 of 6,593 mean waveforms have a top-two gap smaller than one ADC code, compared with 294 for BASE054. In a bounded deterministic diagnostic with both stochastic residual and electronics disabled, ADC alone changes the selected peak in 4,460 S events and 92 BASE054 events. Thus a large portion of the readout sensitivity already exists before random covariance or electronics is introduced.

Under complete stochastic readout, the final peak differs between BASE054 and S for 3,421–3,455 events, depending on condition. In the changed-peak group, the mean correction contributes an average competitive-peak advantage of +13.40 to +13.86 mV, while the random correction contributes −0.23 to −0.28 mV. Signed decompositions identify window movement as the main positive term in the mean tail increment. These are identities along a fixed ordered path; overlapping peak-change counts and absolute terms are not additive causal fractions.

For a fixed waveform, shifting the tail window one sample gives

\[
T_{p+1}(y)-T_p(y)=y_{p+240}-y_{p+30}.
\]

This identity explains how conserving a reference integral can coexist with a changed native tail. A soft penalty at neighboring peaks reduces an objective; it does not certify invariance under the selected native window.

## 6. Software delivery and interpretation

A common research interface accepts prepared 384-point float64 carriers, parent labels, draw count, sampling seed, model-head index and noise hypothesis. It preserves the original 028 loader and offers the closed S candidate as a separate mode. Whole-batch replay semantics are explicit: identical batch order and sampling seed reproduce outputs; parent labels are not random-number keys. Native outputs include Q, front, tail, raw and guarded PSD, parent/draw mapping and observation flags. The S output records `model_accepted=false`.

The formal API test covers legacy 028 on CPU and S/CLI on CUDA. All 523 tensor checks are byte-exact, encompassing 396 legacy checks, 96 S golden checks and 31 replay checks. Additional checks cover input rejection, unchanged model and global RNG states, seed sensitivity, actual CLI execution and lossless reopening. The test produces 1,094 rows and uses 1,032 QPs. The public demonstration contains one synthetic event and the two frozen mathematical modules; private weights, event waveforms and calibration profiles are excluded.

The present evidence supports a reproducible negative method result. It does not satisfy the planned repeated-parent kernel study, new-endpoint classifier test, four-system comparison, three-fold full-pipeline validation or alternative-tail-window task. Within-output-Q population covariance is displayed only as population covariance. It cannot supply a same-parent response target. No new confirmation banks were opened because no candidate met the FIT contract.

The conditional three-coefficient random repair was not triggered: the deterministic ADC-only diagnostic does not isolate the remaining failure to that random subspace. The next scientific design must decide whether to change the frozen mean constraints or objective and register a new bounded evaluation. Current outputs preserve the measured improvement and the failed protection together. They show that reference conservation, smooth representation and software reproducibility are insufficient acceptance conditions for an effective native stochastic response.

## Figures and availability

Six [English captions](FIGURE_CAPTIONS_EN.md) distinguish measured results from missing experiments. PNG and SVG figures are supplied with aggregate reconstruction data and a plotting script. The [related-work table](RELATED_WORK.md) identifies direct comparisons that remain unexecuted. This package supports external reproduction of the synthetic mathematics and aggregate figures; full neural replay requires the authorized private environment and its frozen dependencies.
