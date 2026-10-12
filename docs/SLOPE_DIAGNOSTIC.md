# Frozen N01 random slope diagnostic

This is a bounded mechanism diagnostic of frozen full and deterministic stage-20000 endpoints. It performs no training, introduces no new source parents, changes no model parameters, and is not evidence that an intervened model is physically qualified. The only paired intervention is `noise["electronics_slope"] = 0`; learned `slope_mean` is retained. All other explicit noise tensors, input samples, ADC operations and frozen weights are shared within a pair.

## Algebra and variance

For sample k=0,...,479, t_k=2k/479-1. Q(w)=sum(w[100:400])-3sum(w[0:100]); tail charge is sum(w[150:400])-2.5sum(w[0:100]); PSD=tail/Q. Constant offsets cancel both charge operators.

For n_k=b0+b1*t_k, Q(n)=C_Q*b1 with C_Q=120000/479=250.52192066805844. C_tail=112500/479=234.86430062630478, C_fast=7500/479=15.65762004175366 and C_tail/C_Q=15/16. Given paired off/on preADC outputs and delta_b1=slope_std*z_slope, the first-order PSD increment about off is C_Q*(15/16-PSDoff)*delta_b1/Qoff. The exact finite increment replaces Qoff in the denominator by Qon. Both formulas and large abs(deltaQ/Qoff) are saved; the linearization is not assumed accurate for a large perturbation or near-zero charge.

The first-100 coordinate mean is -380/479=-0.7933194154488519 and centered sum of squares Sxx=333300/229441=1.4526610326837832. Under iid white noise with variance E[sigma_e^2], an OLS slope estimate has variance E[sigma_e^2]/Sxx. Its estimated midpoint intercept is correlated with the slope: covariance is -tmean*E[sigma_e^2]/Sxx. These are estimation statistics, not automatically independent physical offset/drift variances.

The implemented electronics draws offset and random slope independently. Its baseline sample-mean variance is sigma_b0^2+tmean^2*sigma_b1^2+E[sigma_e^2]*wmean^T Cov_h wmean. The script computes the causal FIR covariance exactly from frozen coefficients, including valid extended white input. Corresponding colored OLS estimation variance uses the OLS slope weights. Random-slope charge variance is C_Q^2*sigma_b1^2; the offset contributes zero Q variance.

CPU inspection of the frozen endpoints, with zero model forwards, found the same shared electronics for full/deterministic: random-slope charge variance 0.02118525632847886 V^2 ns^2 and filtered-white charge variance 0.0005028693971503941 V^2 ns^2, totaling 0.021688125725629257. Predicted baseline-mean variance is 3.6168719306470706e-7 V^2 with random slope, and 1.4924615043746065e-7 with it off. Saved REAL validation sample-mean variance (ddof=1) is 2.354801089673536e-8 V^2. Saved REAL OLS midpoint-intercept/slope covariance is 2.0107536006119453e-7 V^2. These numbers locate a mechanism and do not establish a repaired distribution. OLS iid/colored slope estimation variance is respectively 2.791986295158821e-7 / 2.7818829574688e-7 V^2.

## Replay and paired design

The published repeated responses archived 128 legal validation parent IDs, their order and source waveforms, but no explicit noise dictionary. This diagnostic restores the published CPU RNG declaration seed 20261034 and generation batch 128, in exactly the model.sample_noise key/draw order: phys(B,10), electronics_white(B,1,544), electronics_scale(B,1,1), electronics_offset(B,1,1), electronics_slope(B,1,1), local_white(B,1,496). Parent-major ordering is retained (64 consecutive draws per parent).

The baseline is re-generated once and compared with the archived waveform, reporting bitwise equality, changed samples, and maximum LSB difference. Old RTX5070/CUDA results and local hardware need not be bitwise identical; the intervention reference is the newly paired baseline. Restoring a declared RNG stream does not assert an archived sample-level replay before comparison. The explicit six noise tensors plus CPU RNG states, parent IDs, source rows/waveforms, seed and flattening indices are saved in noise_inputs.npz. The off dictionary is defined completely by replacing only electronics_slope with zeros.

Each arm uses the same 128 parents x 64 draws for on and off. Full and deterministic therefore require 32768 total model-forward rows, without gradients. Fourteen-bit canonical ADC outputs and float64 preADC outputs are both preserved. PreADC differences must match delta_b1*t within numerical tolerance, including Q/tail/baseline mean identities. Unsaturated paired ADC Q residual is bounded by 600/16383 V ns; saturated draws are counted separately. No nonpositive-Q or nonfinite-PSD rows are silently removed: raw division values, endpoint-compatible guarded PSD, and validity flags are all saved. Saturation and preADC charge flags are also saved.

Per-parent Q/PSD means, sample variances/covariances, paired changes, baseline sample-mean statistics, and variance decomposition Var(on)-Var(off)=Var(delta)+2Cov(off,delta) use ddof=1. Summaries report across-parent mean and median separately. Baseline ACF and power spectra are pooled descriptive comparisons, with REAL saved validation as a reference. Any nonfinite raw-PSD input leaves corresponding moments null plus explicit counts rather than producing a selected-population estimate.

The published full variance_Q median was 0.7146650152275655 with ddof=1, versus 0.7034983743646349 with ddof=0. Deterministic was 0.02172364099013191 with ddof=1, versus 0.0213842090996611 with ddof=0. Comparing the two models requires a common variance convention.

## Execution and artifacts

Inspect mode is the default and performs zero model forwards. Frozen SHA256 checks cover all six src/n01 files and selected endpoint checkpoint/report/population/repeated inputs before execution. Source hashes and frozen model state hashes are checked again after diagnosis. Completed output directories reject overwrite. If a run fails, FAILED.json retains the exception and source hashes; a failure batch archive is saved where applicable. Choose a fresh output directory for an intentional retry.

Formal coordinated local CUDA command (do not launch until the parent confirms GPU availability):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:TEMP='E:\GAN\runtime_tmp'
$env:TMP='E:\GAN\runtime_tmp'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe' -B `
  'E:\GAN\research\N01\scripts\diagnose_n01_slope.py' --mode paired `
  --arms full deterministic --device cuda --forward-batch 128 --gpu-memory-limit-gib 2 `
  --output-dir 'E:\GAN\research\N01\runs\noise_audit_20261012\slope\paired_cuda'
```

Artifacts: top-level binding.json, algebra_and_variance.json, noise_inputs.npz, report.json and COMPLETE.json; per-arm paired_outputs.npz, per_parent_metrics.npz and report.json. All data/cache/temporary writes are confined to E/D. CUDA allocation is limited to 2 GiB through the PyTorch allocator; external GPU use is outside that limit. Forward progress is printed every batch128. Estimated CUDA inference plus compressed storage is 1-5 minutes, not an observed benchmark; use the final elapsed-time receipt. Uncompressed paired scientific arrays are approximately 250 MB across two arms, plus approximately 35 MB explicit noise; compressed size depends on values.

## CPU verification

Seven artificial unit tests passed: exact slope charge algebra, PSD finite derivative and finite increment, iid OLS variance, independent-drift baseline-mean variance, exact synthetic sample_noise RNG reproduction, explicit zero/negative-charge retention, and paired full/deterministic forward invariants plus sample-variance decomposition. Frozen inspect mode completed successfully in approximately 4.61 seconds with zero forwards. The formal 128x64 paired CUDA run is delegated to the parent; no scientific paired run was executed during implementation.
