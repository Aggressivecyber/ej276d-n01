# Window008 signal model and effective-window contract

Status: implemented; synthetic CPU checks passed. This document does not claim scientific training, detector-response acceptance, CUDA execution, or DEV validation. Existing model/checkpoint/report files remain separate from this new implementation.

## Window and observation binding

The delivered, trained-on, and evaluated waveform contains **256 voltage samples** at the existing 1 ns spacing. The signal network uses an explicitly registered **384-sample carrier** so that the original baseline samples and peak-search guard context remain available. The U-Net therefore processes 384 samples internally; it is not described as a 256-sample U-Net.

SOURCE input is the pristine original 384-sample voltage carrier. SOURCE is not passed through ADC before the signal model. Its original first 100 samples estimate the SOURCE baseline; all registered SOURCE peaks are at carrier index 126 using first-argmax in `[112,144)`. SOURCE effective data and the new model statistics use its centered `[110,366)` window, which contains 256 samples. No old 480-sample charge statistics or trained signal weights are imported.

Generated output is assembled outside this module as signal plus frozen-005 electronics on the carrier. The common observation module applies the registered ADC, subtracts the mean of the same carrier's original ADC samples `[0,100)`, finds the first maximum in `[112,144)`, and takes `[p-16,p+240)`. Final total charge is the sum of all 256 samples and tail charge is the sum of `[46,256)`. The baseline is never estimated from the 16 samples at the start of the cropped pulse. No padding is permitted. Search-boundary and global-peak-outside-search flags are retained for reporting under the root protocol.

The signal module has no electronics or ADC branch. Its first 100 output samples are exactly zero, making the generated observation baseline an electronics-only measurement. The signal onset uses a smooth gate from carrier sample 100 to 115. There is no forced zero value or taper at carrier sample 383, because a valid observation crop can extend through sample 382.

## Public API

```python
from n01.model008 import N01Response008, canonical_charge008

model = N01Response008(new_train_statistics['model'], variant='full')
model.set_stage('mean_response')
latents = model.sample_noise(batch, device=source_carrier384.device,
                             generator=response_generator)
result = model.signal_forward(source_carrier384, latents)
```

`source_carrier384` is a floating tensor of shape `B x 384` or `B x 1 x 384`. `forward` and `signal_forward` are equivalent. `SignalResponse008` is an explicit alias of `N01Response008`. Statistics must provide new `q_ref`, `a_ref`, `log_q_min`, and `log_q_max`; the optional detector condition defaults to zero for this single-condition implementation.

All three arms have the same explicit latent dictionary: `phys: B x 10` and `local_white: B x 1 x 400`. Neither forward path samples randomness internally. The external paired stream must draw these shapes for every arm even when an arm ignores a component. The 400 local-white samples supply the 384-sample signal-local term and the 16-sample history required by its 17-tap causal filter; they are not baseline/electronics noise.

Selected returned fields:

| Field | Shape | Meaning |
| --- | --- | --- |
| `signal_pre` | B x 384 | Pure signal voltage carrier before electronics and ADC |
| `q_source`, `qbar`, `qsig` | B | Canonical SOURCE charge, mean response charge, stochastic signal charge |
| `unit_shape`, `signal_noise` | B x 384 | Canonical-unit-charge shape and zero-canonical-charge local signal term |
| `basis` | B x 8 x 384 | Full-arm stochastic basis; zeros when inactive |
| `time_scale` | B | Positive relative time scale about carrier index 126 |
| `physical_cholesky`, `physical_eta` | B x 10 x 10, B x 10 | Response covariance factor and paired physical latent realization |
| `base_regular`, `regular`, `trend` | scalar | Signal regularization; fixed-shape charge monotonicity is structural, so the trend penalty is zero |
| `canonical_charge`, `canonical_local_charge` | B | Charge measurements in the registered fixed signal window |
| `canonical_charge_identity_residual` | B | `canonical_charge - qsig` |
| `query_outside_carrier`, `beyond_extension` | B | Number of warp queries outside the supplied carrier / defined extended interval |
| `warp_left_outside_carrier`, `warp_right_outside_carrier` | B | Separate left and right carrier-query counts |

## Signal charge and stochastic structure

For an already centered 384-sample signal, define `C(v) = sum(v[110:366])` in V ns. This is a linear **canonical signal** charge, distinct from the final noisy-peak-dependent observation charge. The shape is normalized to `C(unit_shape)=1`. The local term uses a fixed smooth projection with `C(projection)=1`, giving `C(signal_noise)=0`. The final pure signal is `qsig * unit_shape + signal_noise`, so `C(signal_pre)=qsig` to numerical precision.

The mean charge map is a shape-conditioned monotone rational-quadratic spline on log SOURCE charge. Monotonicity applies while amplitude-free SOURCE shape features remain fixed. The physical response uses `eta = L z`; `qsig = qbar * exp(eta_Q - 0.5 * ||L_Q||^2)`. For standard normal response latents, this preserves conditional mean charge `qbar`. The local signal term is conditioned on pulse shape and charge, unlike the separately frozen electronics.

The final observed peak can move when the response, electronics and ADC are applied. Consequently, the observed charge is not asserted to equal `qsig + canonical_noise_charge`. Any exact observed-charge decomposition must use the **same final observed crop** for signal and electronics, and must include ADC and baseline-subtraction contributions. The training/evaluation adapter owns this decomposition.

## Three arms and initialization

| Arm | Mean response | Stochastic response |
| --- | --- | --- |
| `simple` | Charge spline, causal response filter, scalar time response | Two correlated physical coordinates for charge and time; no U-Net, basis or local term |
| `deterministic` | Same filter/charge/time path plus U-Net mean residual | Response latents ignored; electronics remains external and shared |
| `full` | Same mean path as deterministic | Ten physical coordinates, eight basis coordinates, and a conditioned local signal term |

The U-Net uses lengths `384 -> 192 -> 96 -> 48 -> 24` and widths `32,48,64,96,128`, followed by matched decoder lengths. Basis and envelope heads reduce 384 samples to 48 controls by non-overlapping averages of eight samples and expand with a fixed cubic matrix. Length-independent neural primitives are imported from `model.py`; its 480-sample globals, charge function, waveform constructor, old warp, support gate and electronics branch are not used or modified.

All model construction is new. For the paired comparison, root copies every shape-compatible full-arm state tensor to deterministic and simple. Full and deterministic share every active mean-stage tensor. The simple physical head additionally copies the first three output rows of `physical_head.base.2.weight` and `.bias`, which correspond to the 2 x 2 lower triangle. Full's additional refinement and stochastic components remain specific to its architecture. This partial sharing does not assert equal joint stochastic behavior across arms.

Stages are `baseline`, `mean_response`, `joint_stochastic`, and `low_learning_rate`. The mean stage freezes and disables physical/basis/local stochastic modules. Joint and low-rate stages activate the permitted stochastic path for each arm. The root schedule binds 3,000 mean, 12,000 joint and 4,000 low-rate updates per arm, with no DEV-based selection. This model work did not run that schedule.

## Warp boundary convention

Relative warp queries use `126 + (t-126)/scale`. The filtered SOURCE carrier is followed by a 64-sample external cubic Hermite continuation matching its last value and finite-difference slope and reaching zero at external sample 447. This is an explicitly registered numerical boundary convention, not measured SOURCE context. The supplied carrier endpoint itself remains unmodified. Queries left of sample 0 or right of sample 447 return zero and are counted in `beyond_extension`; left and right carrier exits are also reported separately. `query_outside_carrier` includes both sides. In particular, a compressed waveform can query negative indices in the baseline region; this is visible in the counters.

Malformed shapes, missing/nonfinite latents, nonfinite SOURCE, a SOURCE search peak different from the registered 126, nonpositive source/response/shape charge, and a degenerate full-arm basis stop the call. Rows are never silently discarded.

## Synthetic CPU evidence

Receipt: `receipts/planning_008_20261013/model008_synthetic_cpu.json`. Four analytic synthetic SOURCE pulses were run on PyTorch 2.10.0+cpu with deterministic algorithms and explicit paired latents. No science data was read, no optimizer step ran, and no checkpoint was created.

- Full/deterministic mean outputs were bitwise equal after copying 178 shared state tensors. The simple arm copied 27 shape-compatible tensors plus the two partial physical-output tensors.
- Every arm returned `4 x 384`, preserved a strictly zero first-100 prefix, and retained nonzero carrier endpoints in this fixture.
- Maximum canonical-charge residual was `1.7763568394002505e-15` V ns; maximum absolute local charge was `3.469446951953614e-18` V ns.
- Repeating the same input and explicit latents was bitwise identical. Maximum batch-permutation difference was `2.7755575615628914e-17` V. The deterministic arm ignored changed response latents bitwise.
- Every present charge, time, response-filter, mean, physical, basis, envelope and local-filter module had finite nonzero gradients for the synthetic scalar objective.
- A unit-scale warp reproduced its input bitwise. Nonunit-scale fixtures exercised and checked boundary counters.

These checks establish the listed CPU algebra/interface behaviors only. They do not establish learned response quality, scientific acceptance, GPU performance, or physical detector fidelity.

The separate `model008_observation008_synthetic_cpu.json` receipt exercises the complete signal-to-observation path with artificial electronics and two deliberate impulses. The observed peaks were `126,128,124,126`; the final output was `4 x 256`. Charge reconstructed from signal and electronics at the same final crop, ADC error at that crop, and the original-100 ADC baseline agreed with observed charge to `3.1086244689504383e-15` V ns. STE gradients reached the signal mean head. These artificial impulses test crop migration; they do not describe the frozen electronics law.

## Reproducible engine engineering check

`scripts/check_window008_engine.py` checks the integrated `response008.py` with analytic pulses and an artificial observed population/baseline. It reads the registered real noise005 weights but no TRAIN/DEV container, prepared scientific arrays or trained signal checkpoint. It uses the same objectives and three hypotheses, with engineering batches reduced to two microbatches, four source parents per microbatch and two responses per parent. Conditional-support thresholds are retained; unsupported small-batch bins remain reported by the existing objective.

```powershell
python -B scripts/check_window008_engine.py --device cpu --out E:\GAN\research\N01\receipts\planning_008_20261013\engine_cpu_NEW
```

On the prepared remote workspace, the same script accepts `--device cuda:0 --out /absolute/new/output`. It never silently substitutes CPU for an unavailable CUDA device. Output directories must be new; Windows output is restricted to E or D. The receipt binds the script, imported project sources, configuration, fixed noise checkpoint, artificial fixture, device and engineering counters.

CPU receipt: `receipts/planning_008_20261013/engine_cpu_01/ENGINEERING_RECEIPT.json`; status `passed_synthetic_engineering_only`, PyTorch 2.10.0+cpu. It performed exactly one mean and one joint optimizer step per arm: **six engineering G updates, nine engineering D updates, zero scientific updates**, with no noise or population updates.

- Each arm saved after its mean update, restored all model/D/optimizer/sampler/explicit-stream state and counters exactly, and continued for its one joint update. A negative test changing only a loss statistic was rejected with `Resume statistics/schema mismatch`.
- Full and deterministic shared state remained bitwise identical after the paired mean update.
- Explicit latent waveform replay passed in all engine updates. The independent two-pass replay/direct-graph comparison had exactly zero gradient difference for all three arms on this CPU fixture; these probes made no optimizer updates.
- Observed-charge decomposition maximum error was `1.1206313654810174e-15` V ns. The common observed baseline was identical across the three noise hypotheses, and its noise-loss gradient to the response waveform was exactly zero.
- Required signal branches had finite nonzero joint-stage gradients, including full-arm physical, basis, envelope and local-filter branches. The noise005 state stayed unchanged and received no parameter gradients.

The independent engine review identified two integration defects before this passing run: simple-arm physical-head constant buffers require their registered prefix correspondence, and checkpoint restore must bind loss statistics as well as model/noise initialization. Root fixed both in `response008.py`; this engineering run exercised the fixes. Native CUDA execution and production-batch capacity are separate checks.
