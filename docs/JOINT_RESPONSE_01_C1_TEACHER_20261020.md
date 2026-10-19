# JOINT_RESPONSE_01: C1 coupled-teacher revalidation

The C1 interpolation engineering contrast passed all four fixed-direction gates and the complete saved-output SciPy audit. It has not established a correct conditional response. This protocol repeats stage 3 before using the changed generator for REAL calibration.

## Models, targets and roles

Both recovery targets are the unchanged original joint01 synthetic teachers, including their simultaneous gain, time, tail and correlated latent changes. The C1 neural students start at their exact M0 identity, and the C1 physical controls use their stated stochastic initializations. Thus the recovery target does not silently adopt the student's new interpolation. All original parameter bounds, joint feature objectives, finite-difference scale, ridge, radius, CHECK acceptance rules and three noise hypotheses remain. Each neural and physical track receives 12 joint directions and at most three fixed backtracking trials per direction. No paired waveform, teacher parameter label or conditional-moment target enters recovery fitting.

Reuse the original 6465 fit parents and 128 posthoc reporting parents, the original unpaired cycling streams and feature projections. The four main tracks must cover every fit parent on both SOURCE and teacher-target sides. A0/M0/005/PCA already used full TRAIN; these are frozen-system posthoc roles, not full-pipeline unseen validation. There is no REAL response fit, population optimization, DEV decoding or classifier fit.

For the separate population stress, use the same known C1 response for teacher and student, warm-started at that known parameter vector. Fit the fixed high-h population under the nominal and correct high-h profiles, three directions each. This isolates population misspecification without confounding it with the original-versus-C1 interpolation difference. Preserve the original fixed-response nominal/low-h/high-h reports for both recovery teachers. Population labels and the warm start in this separate diagnostic are not recovery evidence.

## Prespecified acceptance and saved evidence

Repeat the original 36 hard-observation gates on two independent 64-draw libraries: joint excess-risk remaining fraction at most 0.65 and native wave/QPSD limits `max(1.02 * M0, 1.20 * teacher oracle)`. Retain the complete original 64-draw conditional reports as diagnostics.

Conditional acceptance is defined prospectively at 2048 draws per each of the 128 reporting parents, using two new banks per head: seeds 2026102031–2026102034. Compute actual generated means, 16-cosine conditional covariance, canonical Q variance and wave variance for the unchanged teacher, M0 and C1 endpoint. Mean error must be at most `min(0.05, 0.65 * M0 error)`; covariance error at most `min(0.50, 0.75 * M0 error)`; Q variance ratio must be within [0.5, 2]. Every teacher/M0/candidate covariance estimate must agree across the two banks within 0.10 using the symmetric relative Frobenius difference. These are the same previously registered high-precision criteria, fixed here before any new fit. All 36 hard gates and 18 conditional/precision gates must pass. A failed gate closes this run without opening REAL use.

Persist each iteration's fit and CHECK row streams/seeds, pre-update parameters, residual/Jacobian, solved direction, tested proposal parameters, CHECK residuals and final parameters. Save feature-map Cholesky factors and block slices for independent reconstruction of reported risks and acceptance. Save all high-precision moment arrays. These augment the unchanged teacher fitting algorithm's original trace and coverage files. They do not alter the solver or add candidate opportunities. Private matrices, moments, identities and waveforms stay off GitHub; publish source/configuration and aggregate audit receipts.

## Bounds and accounting

One scientific job, 4 CPU threads, GPU allocator 2.75 GiB, RSS 12 GiB, internal 1500 seconds, external 1800 seconds plus 20 seconds termination grace, outputs 128 MiB, no automatic restart. The original complete teacher run took 143.65 seconds; C1 generation and the fresh conditional precision evaluation add work. Correct startup and observed progress determine the single wake time only if a long wait remains after all ready work.

Maximums: six checkpoints, one TRAIN archive and one PCA archive; A0 52 calls/6593 rows; M0 104 calls/13186 rows; 232 banks; 16225296 standard-normal values; 228 noise005 calls/979586 rows; 2938758 extension rows; 7416 joint generator calls/27447938 rows; 17556 hard observations/72218502 rows; 2724 finite-difference columns; 54 joint solves, 162 candidate trials, at most 54 teacher updates; 180224 original conditional-report rows; 983040 native metric comparison rows; 18 feature maps. The 1536 high-precision moment generator calls/3145728 rows are included in the joint totals, not counted twice. New REAL updates, AD queries and classifier fits remain zero.
