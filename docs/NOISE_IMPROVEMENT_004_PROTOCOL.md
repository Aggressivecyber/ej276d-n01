# N01 noise improvement 004 — bounded observed baseline

User authorization: `改进`, after completion of noise-origin audit 003. This is a new noise-only candidate. Original runs, the v2 failure, and the origin audit remain intact.

## Model and evidence boundary

Use the existing aligned, 14-bit ADC observation interface in volts. Train on all 21,047 legal REAL TRAIN baselines, samples 0:100 at 1 ns. The model represents the distribution of **observed baselines after the existing acquisition, selection and alignment chain**. It does not reconstruct an identified upstream electronics process or replay CFD on a pulse-free baseline. Audit 003 bound the target representation; changing that representation here would change the question.

For independent standard Gaussian inputs z (100 values) and u (one value), generate

`y = mean_100 + exp(tau*u - tau^2) * L*z`, followed by the unchanged inclusive 14-bit quantizer on [-0.5, 0.5] V.

L is a full lower-triangular 100 by 100 factor with a strictly positive softplus diagonal. Its 5,050 entries, 100 mean values and one positive scale-spread parameter are trainable. The scale has E[S^2]=1, so unconditional pre-quantizer covariance is exactly L L^T. This avoids independent OLS offset/slope additions and their variance duplication. Initial tau is 0.02, not copied from finite-window RMS scatter.

Initialize mean and sample covariance from every TRAIN row (ddof=1). Add the fixed numerical ridge `1e-6 * trace(C)/100 * I`, then take Cholesky. The voltage normalization is `sqrt(trace(C)/100)`. The normalized positive diagonal floor is 1e-6. There is no subtraction of presumed ADC noise and no claim that this empirical initializer recovers a latent analog covariance; the quantized distribution is fitted afterward.

The supported output length is **exactly 100**. Both noise sampling and forward calls must reject 480 and other lengths. No response-model factory accepts this candidate. Even if every 100-point check passes, this run cannot release pulse training: a full 480-point joint noise law is still unvalidated.

A Gaussian global scale mixture remains centrally symmetric before quantization and cannot express arbitrary skew, multimodality or different standardized tail shapes by time position. Failure of this single candidate closes the run; no adaptive mixture size, seed, window or checkpoint search is allowed.

## Fixed optimization and evaluation

Keep BaselineDistributionLoss and its registered v2 weights unchanged. Use the same training batch size (512), projection refresh per optimizer update, uniform full-coverage sampler, Adam learning rates (0.001 for updates 1–4000, 0.0002 for 4001–5000), and gradient clip norm 5.0. Run exactly 5,000 scientific noise updates, then use only the final checkpoint. All TRAIN rows must have nonzero visits. Engineering smoke uses synthetic data only; it is not part of the scientific fit.

Retain repair_002's frozen evaluation seeds, sample counts, classifiers, bootstrap procedure and all thresholds **without alteration**, including its REAL–REAL confidence-interval condition. The previously observed control failure is not repaired by changing labels, seeds or gates. Report candidate checks separately from control validity, but the original all-check gate stays authoritative for comparison. There are six evaluation classifiers, 200 updates each; these 1,200 updates are separately accounted and never counted as noise or pulse optimization.

Validation waveforms are decoded only after the final scientific checkpoint is saved and validated. Their 599 rows and 12 artificial record groups come from the same previously viewed acquisition, not 12 independent acquisitions. No source events, particle labels, IDs, group numbers or filenames are model inputs.

Additional **descriptive, non-selecting** endpoint measures: full100 covariance relative error after fixed TRAIN time-SD normalization, time-SD ratio min/median/max, mean–slope correlation, pre-quantizer covariance eigenvalue minimum, and parameter changes. Report v2 versus v3 using the original v2 frozen payload and the same validation data. No additional threshold is introduced after seeing results.

## Execution, replay and delivery

One scientific process at a time on the verified server, CPU threads 12, GPU allocation at most 10 GiB, owned process-tree RSS at most 8 GiB, wall limit 1,800 seconds, new run output at most 512 MiB. Bind config, source and input hashes before initialization. Stop on input mismatch, nonfinite values, resource failure, unsupported length or interrupted state; preserve failure evidence. Never modify another workload.

Save initialization, optimizer/RNG/sampler states, loss statistics, all per-update records, explicit replay latents, pre/post-quantizer replay outputs, final model, evaluation populations/classifier outputs and a complete manifest. Reopen the final checkpoint and replay saved generator outputs; reconstruct classifier predictions and compare saved scores. Numerical replay verifies reproducibility, not distributional acceptance.

Keep all local source, data, caches and reports on E/D. Preserve an E snapshot before editing registries. Package private payload locally and on D with SHA256 verification; publish source and derived results under the already authorized GitHub repository without raw REAL waveform arrays. Render and inspect the report before delivery. Report noise updates, classifier updates and zero response updates distinctly.

## Method references

- [PyTorch 2.12 multivariate normal documentation](https://docs.pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal): lower-triangular scale factors represent positive-definite covariance. This is a standard parameterization, not a novelty claim.
- [Lopez-Paz and Oquab, Revisiting Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545): held-out classifier discrimination as a two-sample diagnostic. This does not establish equivalence or hardware identification.

The configuration and registration receipt are frozen before any new scientific initialization or fit.
