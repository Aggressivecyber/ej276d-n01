# N01 implementation binding for run_001_20261012

This binding fills details absent from USER_PROPOSAL.md. It is fixed before
formal updates and does not inherit another experiment's science protocol.
The original proposal and initial extracted config are retained separately.

## Data and observation

All 6593 SOURCE TRAIN parents and 21047 REAL TRAIN records are sampled through
without-replacement epochs, carrying the last rows across batch boundaries.
Source IDs, REAL original record indices and uniform source count weights are saved.
REAL validation groups are disjoint from TRAIN groups; ID values never enter
the model. No validation waveform is decoded for fitting or engineering checks.
The existing 512 SOURCE / 599 REAL validation roles remain held out.

Q is the float64 baseline-subtracted sum from samples 100:400; baseline is the
first 100 samples. PSD uses the baseline-subtracted tail 150:400 / Q. The common
ADC operator clamps to [-0.5,0.5] V and rounds once in float64 onto 16384 endpoint
levels. REAL is canonicalized directly to that same grid. Generated pre-ADC
saturation and Q changes are recorded; SOURCE input peaks are never preclipped.
The sample coordinate is 1 ns as defined by the existing software data chain.
Physical calibration remains unknown. The source alignment anchor is 120.

Electronics initialization uses detrended REAL baseline residual RMS, and
intercept/slope statistics in the full-window t coordinate [-1,1]. The midpoint
intercept and slope are initially independent; no unmeasured correlation is
claimed. Only legal TRAIN statistics define scales and bins.

## Fixed training and comparisons

One seed is 2026101201, with fixed loss-projection seed 2026101202. There is no
seed search. A common 1000-update simple response/electronics baseline precedes
four fixed runs: full stochastic U-Net, deterministic U-Net with the same
electronics, simple stochastic charge/time/filter baseline, and full U-Net with
uniform comparison weights. The fourth run evaluates response sensitivity to the
population assumption; it is not a fourth architecture candidate. The first
three each learn their separate bounded population weights. Electronics is copied
from the same baseline and frozen identically across all arms.

Each arm then receives 3000 mean-response, 12000 joint-distribution and 4000
low-LR updates. Nominal budget is 20000 per arm, with shared first 1000 updates;
actual distinct optimizer updates total 77000. Each arm starts with identical
initialization seed, shared physics, sampler and noise-stream seeds, and fresh
optimizers after the common baseline. The nominal update index resumes at 1000.
No held-out score chooses checkpoints, weights or comparators.

The baseline trains the simple physical response from distribution losses and
the independent electronics branch ONLY from real baseline noise statistics.
Electronics is detached during baseline response losses, then trained separately
with the same objective coefficient 1. Mean-response training disables physical
stochastic heads and adversarial loss; wave/QPSD objectives compare means while
baseline noise retains its distribution objective. Joint training enables all
specified stochastic branches and one D update per G update. The simple baseline
has correlated random Q/time broadening; the deterministic comparator retains
the same stochastic electronics but has no physical stochastic head.

G uses AdamW lr=2e-4, weight decay=1e-4; D uses AdamW lr=1e-4, weight decay=0;
both use default betas=(0.9,0.999), eps=1e-8. Low stage rates are 5e-5/2.5e-5.
All active gradients are clipped to norm 5, with nonfinite gradients fatal.
The six proposal coefficients are 1,2,1,0.1,0.01,0.1. Monotonic charge is enforced
by architecture, making its redundant trend violation penalty exactly zero.
Engineering saturation regularization is included in regular, not hidden in Q.

Population parameters use Adam lr=1e-3, once per 16 joint/low-LR response updates.
The three smooth features and bounded normalized formula are described in
LOSS_IMPLEMENTATION.md. Phi optimizes detached QPSD positions plus 0.01 times
its L2+KL penalty. G and D use detached comparison weights. Uniform original TRAIN row-count
weights remain one; learned ratios are saved in a separate file. Calibrated
physical weights are unknown and are not reconstructed from count weights.
The trainer field original_source_weights denotes this count-weight reference. Uniform-arm
theta stays zero. Response/population identifiability is not assumed.

## Joint distribution batch and gradient implementation

Four microbatches each contain 128 SOURCE parents with two independent response
draws and 256 REAL records. A distribution update thus uses 512 parent draws,
1024 generated responses and 1024 REAL records at one parameter version. There
is no old response cache. Eight REAL TRAIN quantile bins extend to infinite
outer bounds; a conditional term needs 16 current rows on each side. All rows
remain in global terms; every missing conditional term is logged explicitly.

The first forward pass records explicit random inputs and canonical waveforms
without retaining model graphs. Losses are jointly differentiated at the 1024
waveform endpoints; each identical-noise microbatch is then re-forwarded to
apply the parameter VJP and its 1/4 direct regular/trend penalty. G parameters
are unchanged throughout both passes. D updates first, then its parameters and
spectral-normalization buffers are frozen during the G objective. GroupNorm has
no running-state mutation. Exact equality of snapshot/reforward is checked in
engineering, first updates and stage transitions. A synthetic direct-joint
backward comparison verifies the two-pass gradient with relative error <1e-6.

Logs preserve all six component losses, clipping norms, conditional support,
coverage, saturation and charge identity residuals each update. Weighted endpoint
gradient norms/angles are logged every 50 updates; all active response parameter
gradients for all six terms are measured at stage transitions and every 4000
updates. The zero trend vector has undefined angle, reported null.

## Execution and failures

The selected server is rylan-GTi15 / 100.82.231.102. The isolated run root is
/home/rylan/projects/ej276d-n01/run_001_20261012. Existing training data is read
from its verified immutable file; no old trained model or loss is loaded.
Only one N01 process uses the GPU. CPU affinity/threads are limited to 12;
PyTorch CUDA allocation limit is 10 GiB; supervisor RSS tree limit is 48 GiB.
The supervisor can signal only its own child process group. It preserves native
exit code and peak RSS. No unrelated process is stopped or reconfigured.

Each complete update is logged. Latest checkpoints every 1000 updates contain
models, optimizers, samplers, loss directions and random states. Stage endpoints
are retained. Invalid Q(v), nonfinite values or other exceptions stop execution
and save failure details, parent identities, explicit noise and model state;
there is no silent rejection, clipping of invalid Q(v), automatic restart or
parameter scan. Resume requires explicit --resume and matching configuration.

The engineering run uses three discarded real-TRAIN updates (one per active
stage family) solely to verify CUDA execution, fixed-size batches, finite
gradients, same-noise replay and memory. Science reinitializes from its fixed
seed, without an engineering checkpoint. These checks do not establish model
accuracy, superiority, calibration, conditional identifiability or reuse across
independent acquisitions. Endpoint evaluation is described separately.

## Verified launch

Engineering attempt 001 stopped at a deterministic CUDA interpolation backward;
attempt 002 passed after mathematically equivalent fixed interpolation/averaging.
Its three updates were discarded. Formal science started 2026-10-11T16:51:53Z
(2026-10-12 00:51:53 Asia/Singapore), PID 478257, supervisor 478255.
The response/sampler/noise streams in each arm are reset to the same prechosen
seed after baseline; they do not continue the baseline RNG stream.
31 artificial numerical tests and the complete evaluator on artificial data
passed. These checks did not read scientific validation waveforms.
