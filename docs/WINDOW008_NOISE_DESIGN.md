# Window008 noise and common observation contract

Status: implemented in `src/n01/observation008.py`; eleven synthetic CPU checks passed. These checks establish code and algebra behavior only. This document does not claim scientific training, CUDA execution, detector-response acceptance, a measured 384-sample quiet record, or a native 256-sample acquisition.

## Scope and coordinate binding

The final waveform used for training, evaluation and delivery is **256 voltage samples at the existing N01 1 ns software spacing**. A **384-sample internal carrier** retains the original baseline and enough signal/noise context for the registered peak-dependent crop. The neural signal network therefore operates on 384 samples internally. This is an effective observed response within a registered 256-sample pulse window; it does not identify the complete approximately 500 ns acquisition or an intrinsic hardware impulse response.

The source audit in `receipts/planning_008_20261013/gamma_256_reference.json` identifies Gamma's 256 as its software network crop, peak minus 16 through peak plus 239. Gamma and the bound GAN archive have the same recorded SHA256, but their preprocessing is different. Window008 retains N01's existing 20-sample block mean, voltage units and ADC convention. It does not import Gamma's additional noise, gain, peak normalization, 11-sample baseline or hardware trigger.

The internal sample coordinates remain the existing aligned N01 coordinates. In the registered TRAIN audit, pristine SOURCE peaks are at 126 and REAL peaks span 120 through 137. These observations support the fixed software search `[112,144)`; they are not a hardware trigger certificate. With a search peak `p` in 112 through 143, the crop `[p-16,p+240)` lies inside the carrier, including the rightmost possible sample 382. Neither zero padding nor copied edge values is needed for the observation crop.

## Common observation order

All intervals below are half open. Generated and REAL observations use this same implementation and ordering:

1. Assemble the generated 384-sample carrier as pure response signal plus frozen noise, then apply the original inclusive 14-bit ADC over `[-0.5,0.5]` V. The step is `1/16383` V with the existing rounding implementation. REAL enters as verified canonical float64 ADC values and is not digitized again.
2. Save the actual carrier values `[0,100)` and subtract their mean from the entire carrier.
3. Find the **first argmax** within carrier `[112,144)`. Subtracting a common baseline does not change that argmax.
4. Gather `[p-16,p+240)` to form 256 samples. The peak is at output index 16. Do not quantize again after baseline subtraction.
5. Compute `Q = sum(window[0:256]) * 1 ns`, `tail = sum(window[46:256]) * 1 ns` and `PSD = tail / Q` under the explicit denominator policy below. Charge units are V ns. The tail begins 30 samples after the selected peak.

The baseline always comes from the **original carrier's first 100 ADC values**, never from the first 100 samples of the 256-sample pulse. A crop starting before carrier sample 100 can overlap a few baseline samples; the formula includes that overlap and does not remove rows. The model's pure signal is exactly zero over the first 100 carrier samples, so the generated baseline is from noise/ADC and carries no response-parameter gradient.

Pristine SOURCE follows a separate `crop_existing_values` path: original voltage values, baseline subtraction and the same peak/crop geometry, **without ADC, clipping or ADC-grid validation**. This preserves clean SOURCE values. Its observation metadata explicitly says `clean_values_no_ADC`, and its `baseline100_adc` and ADC-carrier fields are `None`.

## Frozen noise law and its limits

The original 005 model learns the effective pre-ADC latent distribution of the observed 100-sample prefix. The available prefix does not establish that this latent is pure hardware electronics. Its complete nonstationary mean vector and covariance are retained; a stationary kernel never replaces its covariance.

Let the frozen prefix be `n_B = mu_B + S L z`, with `C_B = L L^T`, `z` standard Gaussian and `S = exp(eta u - eta^2)` from the same 005 latent draw. Then `E[S^2] = 1`. Let `v = mean(diag(C_B))` and `mu_0 = mean(mu_B)`. The original three reference kernels are unchanged, now constructed only on 384 samples:

| Hypothesis | Kernel before fixed ridge, for sample distance d |
| --- | --- |
| `fast` | `v * exp(-d/4)` |
| `slow` | `v * (0.75 exp(-d/4) + 0.25 exp(-d/80))` |
| `common_level` | `v * (0.75 exp(-d/4) + 0.25)` |

Add the original `v * 1e-6 * I` to the full reference kernel before extracting blocks. With prefix block B and remaining 284-sample block T, calculate `A = K_TB K_BB^-1` by Cholesky solve and `R = K_TT - A K_BT`. Symmetrize R for floating-point roundoff and require its Cholesky to succeed. There is no adaptive ridge or eigenvalue clipping.

For each hypothesis, the extension is

```text
mu_T = mu_0 + A (mu_B - mu_0)
n_T  = mu_T + A (n_B - mu_B) + S chol(R) epsilon
```

The prefix, its exact scale S and the independent 284-dimensional epsilon are shared across all three hypotheses. The output prefix is unchanged. The actual joint covariance is

```text
[ C_B          C_B A^T           ]
[ A C_B        A C_B A^T + R     ]
```

It is **not** the reference kernel K unless additional equalities happen to hold. The 384-sample law is the principal marginal of the already registered 006 480-sample law; changing its support adds no new fit. A shorter carrier avoids drawing or evaluating unused samples, while preserving the retained coordinates' law.

The class deep-copies and freezes the supplied 005 state, disables training and checks state versions, float64 and gradient flags. Its forwards accept explicit latents and draw no randomness. The driver must keep response and electronics latent streams independent, reuse electronics across arms, and bind the single prefix/S draw used by `extend_prefix`; the low-level function cannot prove the provenance of arbitrary supplied tensors.

The three kernels remain fixed **sensitivity hypotheses**. They are not a confidence set, identified tail electronics, or a finite upper/lower bound over all possible covariances compatible with the prefix. Holding only C_B fixed leaves an unbounded positive-semidefinite covariance-completion family. A quiet long-tail measurement would provide additional evidence, but a new 480-sample quiet record is not a prerequisite to this bounded implementation and experiment. The pulse tail is not silently treated as quiet.

## Readout decomposition and peak migration

For a fixed selected peak p, define `P_p(v)` as the 256 gathered carrier values minus `mean(v[0:100])`. This is a linear operator. For `x = s + n`, digitized `a = D(x)` and ADC residual `e = a - x`, the exact observed waveform identity is

```text
P_p(a) = P_p(s) + P_p(n) + P_p(e)
```

Every term must use the **same final observed peak p**. The residual includes clipping and rounding, with their corresponding baseline contribution. Total-charge and tail-charge sums of this identity give exact readout decompositions. Applying separate signal-only/noise-only peak searches and adding their readouts is not this identity.

The signal model uses the fixed canonical charge `C(s) = sum(s[110:366]) = qsig`, as specified in `WINDOW008_MODEL_DESIGN.md`. Consequently the final charge is

```text
Q_observed = qsig + [sum(s[p-16:p+240]) - C(s)]
             + sum(P_p(n)) + sum(P_p(e))
```

The bracketed term is signal charge moved into or out of the window by peak migration. It must not be attributed to electronics. There is no claim that summing all 384 signal samples yields qsig. PSD is a ratio: exact saved readouts are required at low charge; a first-order ratio approximation does not replace them. Unquantized `theoretical()` moments concern the carrier before ADC and peak selection, so they are not reported as moments of the final 256-sample observation.

## Interfaces and output policy

```python
from n01.observation008 import (
    FrozenObservedNoise008, observe_generated_carrier,
    observe_existing_carrier, crop_existing_values,
)

noise_model = FrozenObservedNoise008(frozen005)
common = noise_model.sample_prefix(noise005)
noises = {
    h: noise_model.extend_prefix(common['prefix_pre'], common['scale'],
                                 epsilon284, h)['noise_pre']
    for h in noise_model.hypotheses
}
generated = observe_generated_carrier(signal_pre + noises['fast'],
                                      ste=True, q_floor=registered_q_floor)
real = observe_existing_carrier(canonical_real384)
source = crop_existing_values(pristine_source384)
```

`ObservedBaselineNoise005` aliases the original `ObservedBaselineNoiseV3`; `ConditionalExtension008` aliases `FrozenObservedNoise008`. Prefix/scale/noise outputs have shapes `B x 100`, `B x 1` and `B x 384`. Extension epsilon accepts `B x 284` or `B x 1 x 284`; the 005 latent dictionary retains its original shapes and keys. `sample_all` performs one prefix draw and returns all three noises with that common prefix/scale.

`observe_torch` and `observe_numpy` share the exact Torch core; the NumPy wrapper does not maintain a second observation implementation. The generated helper takes a Torch carrier. The existing and clean helpers dispatch Torch/NumPy inputs. All carriers must be finite, nonempty `B x 384` floating arrays/tensors. REAL stored as float32 must first be restored using the existing verified canonicalization; `already_digitized=True` is an exact grid check, not permission to ignore representation error.

Returned fields include `waveform256`/`window_V`, `baseline100`, `baseline100_adc`, `baseline_mean_V`, `peak_index`, `global_peak_index`, `crop_start`, `crop_stop`, `Q`, `tail`, `PSD_raw`, `PSD`, `flags` and explicit input/ADC metadata. Helpers return `crop_indices`/`indices` by default. Optional `return_carrier=True` adds the actual carrier/ADC fields; existing REAL has `pre_adc_carrier=None`, because its physical pre-ADC values are unknown. Low-level `crop256_torch/numpy` performs only raw gathering and is not a substitute for the complete observation chain.

With `q_floor=None`, `PSD` equals the raw ratio and `Q==0` yields an explicit NaN plus `PSD_undefined`. An explicitly registered positive `q_floor` gives a signed denominator guard and a `PSD_guarded` flag; `PSD_raw` is still retained. Negative or zero charge is flagged and retained. The observation module does not silently filter events.

Search-boundary and global-peak-outside-search flags are report-only under the frozen root protocol. They neither discard rows, change search limits, nor by themselves stop a candidate. Rail-hit and generated pre-ADC-out-of-range flags are also returned. Malformed shape, nonfinite input, noncanonical declared REAL ADC input, unsupported peak indices, or mutated frozen noise state stop the call. The driver owns any additional predeclared physical and finite-output gates.

Argmax indices do not receive gradients. Selected sample values and the actual pre-crop baseline retain their normal gradients. Training can request the existing straight-through ADC; evaluation uses actual digitization. A straight-through gradient is an optimization convention and does not differentiate the discrete peak choice.

## Alternative noise routes and experiment interpretation

Resampling empirical TRAIN prefixes and concatenating or tiling them would add stationarity/periodicity, seam and long-range dependence assumptions. Estimating a stationary spectrum from a prefix also requires an explicit stationarity model; the frozen nonstationary C_B is not interchangeable with such a spectrum. Either route could be a separately registered TRAIN-only noise candidate with controlled preprocessing and held-out TRAIN diagnostics. Neither is needed for this minimal extension, and neither can be introduced silently as an equivalent implementation or be selected using DEV.

The bounded Window008 experiment must retain all three fixed noise hypotheses and the registered paired arm comparison. Report each hypothesis rather than choosing the most favorable one or presenting an average as electronics uncertainty. A 256-sample common observation makes the comparison internally consistent; it does not remove noise identifiability limits or prove response matching. Old 480-window Q/PSD values are not numerically interchangeable with the new gates. Engineering checks, completion of the fixed training schedule, TRAIN evaluation, DEV evaluation and scientific response acceptance remain separate claims.

## Synthetic CPU evidence

The native CPU command exited 0 using PyTorch 2.10.0+cpu and NumPy 2.2.6. It decoded **zero science arrays** and performed **zero training updates**. Eleven checks passed:

1. Registered peak limits, crop geometry and flag-only row retention.
2. Bitwise identity of generated-ADC and existing-canonical-ADC observations.
3. NumPy/Torch identity and an independent baseline/crop formula, with error below `1e-15` V.
4. Clean SOURCE values above ADC range remain unclipped and unquantized.
5. Straight-through charge gradients include the actual pre-crop baseline coefficient `-2.56` and selected-sample coefficient `+1`, within `2e-15` for a nonoverlapping crop fixture.
6. First-argmax behavior for tied peaks.
7. Hard rejection of wrong shapes, nonfinite inputs and noncanonical declared ADC inputs.
8. Explicit retention and marking of zero-Q raw/guarded PSD.
9. Exact preservation of a synthetic nonstationary 005 prefix and its shared S.
10. Equality to the 006 law's 384-sample principal marginal: maximum mean error `0`, covariance error `2.0679515313825692e-25` V squared and paired-sample error `9.75781955236954e-19` V.
11. Detection of ordinary mutation of frozen noise state.

The tested model and inputs were synthetic. No statement above validates hardware noise, TRAIN distribution fit, CUDA behavior or learned detector response.
