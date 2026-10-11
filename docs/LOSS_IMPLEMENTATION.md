# N01 loss implementation

This implementation follows `USER_PROPOSAL.md`; it imports no earlier loss, bin,
population tilt, generated-output cache, or scientific acceptance criterion.
Loss coefficients remain the proposal's initial values, not validated optima.

## Interfaces and TRAIN statistics

`fit_loss_statistics(real_train, source_train)` takes `(N,480)` or `(N,1,480)`
voltage waveforms and returns an ordinary torch-save-compatible dictionary.
Only legal TRAIN tensors may be supplied. The caller owns the split audit.
The dictionary explicitly records `fit_split=TRAIN`, both full population counts,
charge extrema, invalidity counts, numerical floors, seven conditional bin
boundaries, and all standardized feature centers/scales. Statistics are fitted
on CPU with no array selection, except a positive-REAL-Q subset used solely to
define the positive reference charge when nonpositive observed charges exist.
That subset is never used as a training or loss population.

Nonfinite rows, nonpositive waveform peaks, or SOURCE Q below the integration
resolution fail explicitly with counts. No event is silently dropped. SOURCE
peak or waveform values above the ADC range are not preclipped by these losses.
All source amplitude values remain available to the response model.

Q is `sum(x[100:400])-3*sum(x[:100])`, evaluated in float64. PSD has tail window
150:400, subtracts 2.5 times the first-100-sample sum, and divides by the same Q.
The units are voltage and 1 ns samples, so Q is V ns. `q_floor` is the larger of
one millionth of the positive REAL TRAIN median Q and the RMS resolution of a
14-bit, 1 V range, baseline-subtracted sum: ADC step times sqrt(1200/12).
Peak normalization uses a one-ADC-step denominator floor; this floor is not an
event selection threshold. Standardization is median and IQR/1.349, with
documented numerical floors rather than a held-out fitted scale.

For positive Q above the floor, the charge feature is exactly log(Q/Qref).
Below it, a C1, strictly monotone asinh continuation keeps zero and negative
generated charges finite and represented. PSD uses a signed denominator whose
magnitude is at least `q_floor`. This guarded ratio is a numerical objective
feature; physical PSD near zero is not asserted. REAL near-zero/nonpositive
counts are reported in fit metadata, and generated/REAL guarded counts are
reported on each objective call. The current verified REAL TRAIN population
has positive Q well above that floor; its eight bins are ordinary logQ bins.

## Distribution distance and fixed directions

`weighted_wasserstein_1d(x,y,weights=None,real_weights=None)` exactly integrates
the absolute difference of the two weighted empirical CDFs over their merged
sorted support. It accepts unequal population counts and multiple coordinates.
It differentiates through sample positions and normalized empirical weights,
piecewise through sort/search membership. It uses neither a quantile grid nor
SciPy, and retains every sample including underflow/overflow populations.
Each probability vector is normalized within the empirical population being
compared. Nonnegative finite weights with positive total mass are required.

`N01Loss(stats,seed=20261012)` returns five non-adversarial scalar components
through `.components(outputs,real,comparison_weights=None,stage=...)` or its
ordinary module call: `wave`, `QPSD`, `noise`, `regular`, and `trend`. The sixth
proposal component, `adv`, is provided separately by `g_hinge`. `regular` and
`trend` are scalar generator outputs, passed through without inherited rules.
For endpoint-waveform/VJP training, the caller can pass differentiable zero
regular/trend placeholders and accumulate those parameter penalties during
same-version reforward. The loss does not cache any old generated response.

The waveform term averages projected empirical W1 over six views: complete
voltage, baseline-subtracted peak-normalized shape, first difference, second
difference, and first differences after width-5 and width-17 moving averages.
Padding for smoothing replicates edge values. Every view uses REAL TRAIN
coordinate scales and 32 unit-norm Gaussian projection directions. QPSD uses
16 joint projection directions. Noise features use 32 directions. Directions
are generated once from a local CPU RNG and stored as buffers; they never
refresh during training, move with the module, and are saved in its state dict.
Construction does not change the global RNG. All runs must record that seed.

QPSD's global term equally averages coordinate-wise empirical W1 and joint
projected W1. Conditional PSD uses generated output Q and REAL output Q,
never raw SOURCE Q versus REAL Q. Seven REAL TRAIN quantile boundaries form
eight equal-probability bins with first/last bins extending to minus/plus
infinity. Boundary membership uses `torch.bucketize(...,right=False)`.
Repeated boundaries are retained and their support remains visible.
Each conditional term requires at least 16 current rows from each side.
The global and mean supported-conditional terms receive equal weights when
any conditional bin has sufficient support. If none does, only the global
term is returned; omitted terms are never reported as zero matched distances.

`.last_diagnostics` reports every bin's two counts, support/status, bins used,
coverage completeness, rebatch need, nonpositive/guarded Q counts, and global
row counts. The trainer can combine a larger batch of forwards from the same
parameter version. Incomplete coverage is an explicit limitation rather than
a reason to reject sparse upper-tail events or remove them from the global
objective. Conditional membership is discrete; the global joint objective
supplies charge gradients through all rows. Diagnostics do not by themselves
establish scientific equivalence or acceptance.

Generator components detach comparison weights, preventing response updates
from optimizing population parameters. `.population_objective(fake,real,weights)`
returns only the QPSD distribution term with fake/REAL positions detached,
retaining gradients through comparison weights for independent phi updates.

## Stage meaning and baseline noise

In `mean_response`, waveform comparison uses absolute differences of population
means in all six standardized views. Global QPSD compares feature means and
conditional PSD compares bin means. It therefore does not ask the deterministic
mean response to reproduce full Q/PSD conditional widths. The raw mean waveform,
normalized mean waveform, and their derivatives supply the relative time and
mean-shape structure objective. These are unpaired population means, not an
eventwise SOURCE/REAL MSE. Bin changes can still alter the conditional objective.
Noise remains a baseline feature distribution objective because the electronics
branch retains event noise. `joint_stochastic` and `low_learning_rate` use the
full empirical distribution distances. Stage freezing/activation belongs to
the trainer. The standalone baseline stage should train the electronics branch
with `.noise_loss(fake_electronics,real_baselines)` directly; calling components
at `baseline` does not implicitly disable the other terms.

`.noise_features(x)` accepts full 480-point waveforms or first-100-point segments
and returns 32 float64 features: baseline mean and linear slope on t in [-1,1],
log RMS after mean subtraction, ACF at lags 1 through 16, centered amplitude
quantiles 1/50/99 percent, five positive-frequency power fractions, and log local
variance in five adjacent 20-point segments. Frequency groups are FFT bins
[1:3], [3:6], [6:11], [11:21], [21:51] on a 100-point real FFT; DC is excluded.
Quantization variance is the numerical RMS/ACF/variance/power floor. The noise
term equally averages coordinate and sliced W1 with REAL TRAIN scales.
The full-waveform and multiscale derivative terms also observe pulse-region
fluctuations; no paired noiseless REAL signal is available for isolating them.

## Discriminator and comparison weights

`WaveformDiscriminator(stats)` applies the same frozen REAL TRAIN voltage
standardization, then spectral-normalized 1D convolutions with 32/64/128 channels,
kernel 7, stride 2, and LeakyReLU(0.2). It returns global `(B,)` and local `(B,60)`
heads. Both heads are spectral normalized. `d_hinge(fakepred,realpred,weights)`
is mean relu(1+Dfake) plus mean relu(1-Dreal), averaged equally over heads.
`g_hinge(fakepred,weights)` is negative mean Dfake over both heads. Optional
comparison weights apply only to fake row means. The caller detaches fake
waveforms for D training and freezes D parameters for G training as appropriate.
This training critic is not the independent final domain classifier.

`PopulationWeights(q_source,h)` contains exactly three trainable theta entries.
Features are bounded standardized log SOURCE Q, its centered squared value, and
centered bounded standardized known source deposition composition h. No ID,
filename, row, batch position, real particle label, or measured-output feature
is supplied. SOURCE TRAIN fixes feature centers/scales and the full-population
normalizer. Index inputs to `.forward(indices)` only select already computed
weights and preserve duplicated-parent response weights.

The full-population comparison weight is
`0.5 + 0.5*exp(0.5*tanh(F@theta))/mean(exp(0.5*tanh(F@theta)))`.
It has full-population mean 1 and positive ratios bounded by
`0.5+0.5*exp(-1)` and `0.5+0.5*exp(1)` (approximately 0.684 and 1.859).
The normalizer is never a minibatch normalizer. Theta zero gives weight one.
Equal-population runs freeze that zero parameter. `.penalty()` returns theta
mean square plus mean w*log(w), the KL to uniform for weights with mean 1.
A trainer coefficient of 0.01 gives L2 and KL coefficients 0.01 each. It must
be logged separately from the QPSD objective. Original parents and original
physical weights are preserved; these ratios are comparison weights and do
not establish physical calibration or identifiability of response/population.

## Verification limits

`tests/test_losses.py` uses only synthetic tensors to check exact W1 values,
sample/weight gradients, identical-distribution zeros, validity failures,
generated-weight detachment and phi-only updates, out-of-range population
coverage, explicit sparse-bin diagnostics, seeded directions, baseline feature
dimensions, population bounds/global normalization, and gradients of both
discriminator heads. These are implementation checks, not training success or
physical-response acceptance. No scientific training is launched by this file.
