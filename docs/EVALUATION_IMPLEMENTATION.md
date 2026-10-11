# N01 fixed-endpoint descriptive evaluation

evaluate_checkpoint(checkpoint_path, container, identity_path, output_dir,
device="cuda") evaluates a caller-declared completed response arm. It loads the
existing 512 SOURCE and 599 REAL validation parents through load_validation only
when invoked at the endpoint. It uses the checkpoint's model statistics, loss
statistics, complete frozen loss state, response parameters, population state,
variant, arm, and seed. It does not select checkpoints or fit validation values
into model/loss scales. Baseline checkpoints are rejected. On Windows, outputs must remain
on E: or D:. Linux callers use the registered remote run directory. CPU thread limits are inherited from the caller, and every
response/classifier GPU batch is at most 128.

The function returns its report dictionary and writes endpoint_report.json.
It writes EVALUATION_COMPLETE.json only after every requested computation and
artifact save finishes. That marker includes checkpoint/input/report hashes,
artifact hashes, and the evaluator source hash. A matching completed evaluation
can be reused; different endpoint/input hashes require a new directory to
preserve the previous result. A report without its completion marker is not
treated as complete. Errors do not silently drop invalid generated events.

## Frozen population weights and preserved identities

SOURCE parents retain their original IDs, selected-interface count weight one,
and h metadata. Calibrated physical weights are unknown and are not present in
this selected data interface.
REAL parents retain their original IDs and original group membership. IDs/groups
are lineage and split metadata, and never enter the response or classifier as
features. Complete identity arrays are saved in endpoint_populations.npz.

Validation comparison weights use the checkpoint's TRAIN median/IQR scales,
TRAIN centered quadratic feature, saved theta, and full TRAIN exponential
normalizer. The current population state stores centered composition features
without their original tanh mean. Therefore the evaluator opens only TRAIN
container field h to reconstruct that mean using the saved TRAIN h scale. It
does not open TRAIN waveforms or refit any validation statistic. Validation mean
comparison weight need not equal one: the family retains its TRAIN normalizer.
Empirical distribution calculations normalize the selected validation population
weights to total probability one. Raw and comparison-weighted results are
reported separately using the same response draws.

## Complete waveform, QPSD, noise, and controls

Four responses per SOURCE validation parent produce 2048 observed waveforms.
The evaluation RNG seed is fixed across arms (20261012 plus recorded offsets).
All observations use the common frozen 14-bit inclusive [-0.5,0.5] V nearest-even
ADC with 1 ns samples. SOURCE inputs are not preclipped.

The report gives raw/weighted scalar distributions, QPSD covariance, the full
frozen TRAIN-scale waveform/QPSD/noise distribution distances, and explicit
conditional support diagnostics. The numeric population artifact retains every
waveform, all Q/PSD values, pre-ADC charge diagnostics, comparison weights,
complete voltage/peak-normalized shape mean and standard-deviation curves,
pointwise 5/50/95 percent curves, and baseline FFT power/frequency curves.
Normalization uses first-100-sample means and a one-ADC-step peak floor.

Charge and PSD use the documented 100:400 and 150:400 windows with baseline
subtraction. Nonpositive Q and guarded PSD counts remain explicit, with every row
retained globally. Near-zero/nonpositive-Q guarded PSD is a numerical descriptive
feature, not a physical PSD claim. Undefined half-rise times are saved as NaN in
numeric artifacts and as null in JSON, with invalid counts/denominators reported.

Noise descriptions include baseline RMS, ACF lags 1..16, five frequency fractions,
baseline amplitude tails, and local variances. Summary-mean uncertainty resamples
SOURCE parent IDs or original REAL groups as clusters, with 400 bootstrap draws.
It describes finite-population sampling, not independent acquisition uncertainty.

Eight frozen REAL TRAIN output-logQ bins compare generated output Q with REAL Q.
Each raw/weighted/REAL bin receives a two-Gaussian PSD fit only with at least 16
rows. Fits use weighted EM, three fixed initial quantile pairs, at most 200
iterations, and a stated 1e-4 PSD standard-deviation numerical floor. Reports
include lower/upper peak positions, widths in original PSD coordinates, fractions,
effective count, convergence, initialization variation, one-versus-two-component
BIC, and overlap separation. Repeated generated rows share parents; their raw
count is not a claim of equally many independent measured events. Mixture names
are descriptive peaks and never particle truth labels or pseudolabel accuracy.
BIC/overlap diagnostics do not impose a scientific pass criterion.

REAL-REAL controls use disjoint existing original REAL groups and the same frozen
statistics. Insufficient groups are reported as not estimable. Common observation
controls verify that a second ADC application is exactly idempotent on canonical
REAL and generated values, and record the first-application numerical change.
There is no stochastic forward-REAL identity requirement.

## Independent domain classifier

A separate CNN with channels 8 and 16 and an MLP of width 32 is fitted on endpoint
validation's designated classifier-fit subset. No training discriminator weight,
architecture, or score is reused. The first 60 percent of fixed-seed shuffled
unique SOURCE parent IDs and unique original REAL groups form the fit portion;
all remaining groups form the test portion. Every response of a SOURCE parent
stays on one side. REAL group overlap is prevented. Fewer than two original
groups makes this diagnostic not estimable.

The classifier uses frozen TRAIN voltage scales, fixed Adam learning rate 1e-3,
200 updates, batch at most 128, and balanced domain contribution. Test scores do
not select classifier updates, response checkpoints, response arms, seeds, or
architecture. Test raw and comparison-weighted ROC AUC use the same fitted model.
AUC handles ties with half credit. A 400-draw separate-domain cluster bootstrap
resamples test SOURCE parents and original REAL groups, yielding percentile 95
intervals and sampling SE. Few REAL groups are explicitly flagged as limiting
the uncertainty estimate. This diagnostic's labels mean generated versus REAL,
not neutron/gamma truth. Parameters, split identities, test logits, labels, row
references, and weights are saved. Operators use fixed convolution/linear
operations and do not require adaptive pooling or linear upsampling backward.

## Repeated parents and paired input interventions

A fixed, value-independent RNG choice preselects 128 validation parents, sorted
by row only for stable storage. Each receives 64 explicit independent latent
draws. repeated_parent_responses.npz stores all waveforms, original parent IDs,
Q/PSD, signal mean/stochastic charges, time scales, electronics charge, ADC
changes, saturation, conditional variances and QPSD covariance, relative
half-rise/peak times, source-relative time differences, and baseline RMS.
Relative time is also reported around the frozen 120 ns anchor.

Conditional mean SE is sample SD/sqrt(64). Conditional variance and QPSD
covariance SE use 400 within-parent draw bootstrap resamples. Selected waveform
covariances at samples 100/120/150/200/300/399/479 have an explicitly labeled
cross-product asymptotic sampling SE. A joint Q/PSD/time/noise covariance is also
saved. These model-draw results do not uniquely identify physical noise without
repeated measured truth for the same physical input.

Eight preselected parents receive 32 draws for each input intervention. The same
explicit noise seed, batch order, and latent tensors are reused across original
inputs, replacement with one common normalized reference shape at the original
Q, and same-shape scales 0.8/1.2. Reports contain input-change magnitudes, actual
input Q ratios, paired output changes, per-parent response means, paired SE,
and electronics-charge invariance under identical latent electronics draws.
Small reference-shape changes can be uninformative and are reported as such.
There is no global-shuffle-marginal criterion. Intervention waveforms and paired
numeric differences are preserved in their own NPZ artifact.

## Cross-arm aggregation and interpretation

summarize_arm_comparison(run_root) reads only hash-verified completed evaluation
reports under that bounded run directory and writes ARM_COMPARISON.json. The
evaluator invokes aggregation after each successful completion or verified resume
reuse using output_dir.parent.parent as the run root. It
lists full/deterministic/simple/full_uniform metrics side by side without
ranking or selecting them. Missing endpoints remain pending. Duplicate endpoints
for one arm raise an ambiguity error rather than selecting a favorable result.

When both full and full_uniform are available with matching data provenance, the
aggregator checks frozen checkpoint hashes, compares shared response parameter
groups, compares Q/PSD/charge-mean/time and conditional variances/covariance on
common original repeated parents, and describes changes in their mean voltage
responses. It records training-seed differences as a possible confound and
bootstrap uncertainty across common parents. Separate retraining is necessary
for this comparison; raw versus weighted statistics from one frozen response
alone do not establish response stability. The per-arm evaluator can also find
a counterpart in the standard sibling arm/evaluation-directory-name layout;
otherwise this comparison remains pending until aggregation.

No historical acceptance thresholds, best-checkpoint selection, independent
acquisition claim, physical calibration claim, or unique response/population
identification claim is introduced. All distances and uncertainty intervals are
descriptive. Scientific acceptance remains undetermined by these artifacts.

## Implementation verification

The source is parsed/imported without reading scientific arrays. Synthetic checks
cover weighted AUC/ties, mixture fit finiteness/support reporting, saved-TRAIN
new-row population weights, and within-parent bootstrap tensor broadcasting.
The separate loss implementation's 14 artificial tests have passed. These are
static/numerical implementation checks; no endpoint evaluation or scientific
training is claimed by creating this file.
