# N01 stationary electronics v2 implementation

Status: candidate implementation and artificial-process checks only. Scientific training updates: 0. REAL TRAIN/VAL rows read during implementation: 0. No frozen run_001 weights/results or existing model/data/trainer source were modified. This implementation follows NOISE_REPAIR_SCOPE.md; noise validation and its registered stop rule precede any new pulse training.

## Generator and parameterization

For each event b and sample t, the pre-ADC electronics are

    e[b,t] = mu0 + sigma0*z_offset[b]
             + sigma[b] * sum(k=0..64, h[k]*white[b,t+64-k])
    sigma[b] = exp(log_rms + s*z_scale[b] - s^2/2)
    s = softplus(raw_log_scale_std)
    h = filter.raw / ||filter.raw||_2
    mu0 = voltage_scale * offset_mean_units
    sigma0 = exp(log_offset_std)

All latent draws are independent standard normals. Offset/amplitude distributions and the free filter are optimized jointly. Offset and amplitude are independent in this first candidate; no extra non-Gaussian offset mechanism was added. There is no source waveform, charge, particle, event ID or acquisition group input. There are no slope/drift parameters or time-coordinate drift buffers. Random baseline sample slopes may still arise from the stationary process.

The free signed FIR has 65 trainable coefficients. Its initial coefficients are exactly the unit impulse, with no bound on the size of subsequent tail coefficients. L2 normalization separates sample RMS from covariance shape. The raw FIR's scale is redundant under normalization. A nonfinite or effectively zero norm (<=1e-12) fails explicitly. There is no DC normalization, amplitude clipping, circular convolution, zero-charge projection or final-waveform Q projection. A 64-sample independent white history precedes every valid crop, so the first generated sample has the same stationary variance as the last.

All arithmetic used to form the output and normalized filter is float64; optimized parameters use float32, matching N01's existing optimizer/network conventions. Arithmetic E[sigma]=exp(log_rms) is mean-corrected; E[sigma^2]=exp(2*log_rms+s^2). No ceiling is imposed on correlation or logscale spread. Nonfinite output fails without dropping rows. Canonical ADC remains the caller/N01Response responsibility.

The mean parameter is dimensionless. voltage_scale is a fixed buffer set to the initial TRAIN difference-RMS estimate, or an explicitly supplied positive initialization scale. Under a normal Adam step of 1e-3, the physical mean step is of order voltage_scale*1e-3, avoiding a direct 1 mV-sized step. Standard deviations use log parameters. There are 69 trainable numbers (65 FIR plus offset_mean_units/log_offset_std/log_rms/raw_log_scale_std); unit normalization removes one filter scale degree of freedom.

## Explicit API

`ElectronicsNoiseV2(initialization=None, length=480)` is directly callable as `model.electronics(noise)`. It returns Bx1x480 float64 by default. `model.electronics(noise,length=100)` returns Bx1x100. Accepted initialization keys are rms, log_scale_std, offset_mean, offset_std, voltage_scale and metadata. Missing scalar keys use artificial engineering defaults rms=1e-3 V, log_scale_std=.02, offset_mean=0 V, offset_std=1e-4 V. rms/log_scale_std/offset_std/voltage_scale must be positive. Legacy `electronics_offset_std` / `electronics_slope_std` dictionaries are rejected explicitly; there is no silent OLS-statistics reuse.

Only three explicit noise keys are consumed:

| Key | 480-point generation | 100-point generation |
| --- | --- | --- |
| electronics_white | Bx1x544 | Bx1x164 |
| electronics_scale | Bx1x1 | Bx1x1 |
| electronics_offset | Bx1x1 | Bx1x1 |

Extra keys, including electronics_slope/phys/local_white, are ignored by this electronics module. The original N01Response.sample_noise returns these three consumed shapes correctly within its complete six-key dictionary. Its generator order remains unchanged when replacing the module. Standalone `sample_noise(batch,device=None,generator=None,length=None)` draws only the three consumed keys in the displayed order as float32; a dedicated 100-point fit therefore avoids unnecessary 480-point convolution. The two samplers have different RNG consumption; replay is specified within a chosen sampler and explicit dictionary, not between the two streams.

When length=100 and white has the original 544 points, only its first164 points are used. These produce exactly the same first100 electronics values as a480-point call with the same complete white/scale/offset tensors. Directly supplied164-point white is therefore valid. Used inputs require finite floating tensors with a common device and batch dimension; missing/invalid data fail explicitly. Explicitly supplied dictionaries reproduce deterministic outputs for fixed parameters and execution environment, with no batch statistics or forward-state mutation.

`theoretical_stats()` returns differentiable scalar tensors: offset_mean, offset_std, rms_arithmetic_mean, log_scale_std, sigma_squared_mean, sample_variance_stationary_component, sample_variance_total, voltage_scale. `filter.taps()` returns the effective normalized coefficients.

`autocovariance(max_lag,include_offset=True)` returns ensemble covariance at integer lags. For lag l<65 it is

    Cov(e[t],e[t+l]) = sigma0^2 + E[sigma^2]*sum_k h[k]*h[k+l].

For lag>=65 only the event-offset covariance remains. Excluding offset removes its common-event term. `linear_statistic_variance(weights,include_offset=True)` evaluates the exact finite-window variance of a weighted sample sum, including offset contribution sigma0^2*(sum weights)^2. This supplies baseline-mean and Q variance checks without assuming white noise after training. For N01 Q weights (-3 on0:100, +1 on100:400, 0 elsewhere), offsets cancel and stationary electronics retain nonzero Q fluctuations.

## Transparent TRAIN initialization

`fit_electronics_initialization(train_baselines,log_scale_std=.02,std_floor=1e-7)` accepts explicitly supplied finite NxL or Nx1xL baselines and reads no files. The caller must ensure that only registered TRAIN baselines reach this function. It does not inspect IDs, validation rows, pulse charge, OLS slopes or OLS midpoint-intercept distributions.

For each event, estimate diff_sigma2=mean(diff(baseline)^2)/2. Its square root gives a rough white-initialization RMS estimate; rms is the arithmetic average across events. Positive short-lag correlation can bias this rough value downward. It initializes the actual unit-impulse filter; it does not identify latent noise amplitude under an already correlated process.

Initialize offset_mean to the mean of event baseline sample means. Initialize offset_std^2 to max(sampleVar(eventmeans,ddof=1) - rms^2*exp(.02^2)/L, std_floor^2). The subtraction is the stationary sample-mean variance at the actual unit-impulse initialization. As the FIR/amplitude distribution learn, the correct baseline-mean noise contribution is recalculated by the analytical interface. This initialization is a starting estimate, not an assertion that the remaining variance is identified physical offset variance.

log_scale_std starts at the fixed near-zero .02, allowing training to separate actual event heterogeneity from finite-window RMS scatter. Observed finite-window log difference-RMS SD is reported only as metadata and is never copied into that latent parameter. The metadata records sample count, ddof, raw difference second moment, observed baseline-mean variance, subtracted variance, residual before floor, floor usage, observed unused log spread, and the assumptions. Identically constant baselines fail rather than receiving an invented positive noise estimate.

## N01 integration

`make_n01_response_v2(stats=None,variant='full',electronics_initialization=None,electronics=None,stage='joint_stochastic')` creates a new N01Response and replaces only its electronics. The old model source and run_001 stay intact. This factory reads no checkpoint or scientific arrays and loads no legacy electronics. Supply either explicit initialization or an already trained ElectronicsNoiseV2 module. Existing sample_noise, stage behavior, ADC, signal charge identity and physical heads are retained. set_stage runs after replacement, so electronics are enabled for baseline and frozen by the original later stages. A standalone newly constructed electronics module is trainable by default for noise-only fitting.

The factory does not make a noise-only fit equivalent to the old baseline phase: root orchestration must register its5000-step noise-only budget and validate the trained electronics before deciding whether to train pulse responses. New parameter/state names differ from old electronics. Copying complete v2 state_dict between v2 candidates is supported; an old N01 electronics state must not be loaded into v2. The root trainer remains untouched. Passing the same trained module to multiple response objects shares its parameters; deep-copy it if independent module identity is required. No adapter silently loads frozen physics weights.

## Verification and evidence boundary

Ten CPU-only artificial-process tests verify:

- Free-FIR achievable lag1 correlation cos(pi/66)=.998867..., plus negative -.48, and exact normalized energy.
- Monte Carlo covariance/mean for a known positive MA1 process; matching variance at the first/middle/last valid sample.
- Lognormal arithmetic mean and second moment against ensemble theory.
- Original complete six-key shapes, explicit replay, ignored unused keys, batch reorder invariance and consistent100/480-point crops.
- Installed N01 electronics output independence under changed source amplitude and pulse shape, retaining original stage freeze rules.
- Nonzero positive/negative Q fluctuations and ensemble Q variance agreement; event-offset variance cancels Q.
- Actual prehistory contributions, causal behavior and no circular wrap.
- Finite nonzero gradients in every parameter group, an Adam offset-mean step at the intended voltage scale, and continued L2 normalization after a step.
- Rough initialization against a known iid Gaussian process, variance subtraction, fixed small spread and explicit rejection of legacy OLS dictionaries.
- Explicit invalid-data/zero-filter rejection with no clipping or row removal.

All tests use fabricated data. Test success establishes implementation semantics and attainable covariance, not empirical electronics fit or a repaired physical response. No REAL data fit, scientific generator run, CUDA execution or new training was performed by this implementation task. Formal evaluation must still check full100-point joint structure, baseline mean/RMS/ACF/spectrum and OLS joint statistics with registered parent/group isolation, matched sample counts and independent classifier directions, including randomized REAL-REAL controls. Previously examined validation data must be labeled as reused diagnostic evidence. The root owns those protocols and scientific execution.

CPU test command (E/D temporary files; thread limits must be set in the invoking shell):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:TEMP='E:\GAN\runtime_tmp'
$env:TMP='E:\GAN\runtime_tmp'
& 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe' -B -m unittest discover -s 'E:\GAN\research\N01\tests' -p test_electronics_v2.py -v
```
