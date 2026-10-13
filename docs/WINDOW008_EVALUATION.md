# Window008 fixed endpoint evaluation

Status: implemented in `src/n01/evaluate_window008.py`; synthetic CPU generation, replay, readout, distance and bootstrap checks passed. The complete scientific evaluator has not been run. The tests used artificial waveforms and synthetic model/noise initializations, with zero science-array decodes and zero optimizer updates.

## Callable contract and freeze boundary

```python
from n01.evaluate_window008 import evaluate_run

summary = evaluate_run(
    out=run_directory,
    cfg=config_path_or_dict,
    dataset=prepared_DATASET008_npz,
    statistics=final_FIT_STATISTICS008_json_or_dict,
    raw_container=bound_TRAIN_PRIVATE_npz,
    identities=bound_original_identity_npz,
    device='cuda:0',
)
```

The caller supplies the run directory containing `train/deterministic/final.pt`, `train/full/final.pt` and `train/simple/final.pt`. Outputs go into a **new** `evaluation/` directory; an existing directory is not overwritten. For exact replay, the keyword-only `destination` or CLI `--destination` may instead specify a fresh descendant of `run.parent/evaluation_jobs/`; the training run and native-receipt inputs remain unchanged. The corresponding CLI is `python -m n01.evaluate_window008` with `--out`, `--config`, `--dataset`, `--statistics`, `--raw-container`, `--identities` and optional `--device`/`--destination`.

`DATASET008.npz` is the saved TRAIN-only preparation. Its directory must also contain `FIT_DATA_STATISTICS.json`. The latter supplies the already fitted SOURCE TRAIN Q8 edges; its SHA256 must equal `data_statistics_sha256` in the final statistics, and the dataset must match `prepared_dataset_sha256`. The evaluator does not refit bins or scales. Final checkpoint statistics remain unchanged; separately bound preparation summaries are used only for evaluation grouping.

Before opening any DEV waveform member, the evaluator first requires the adjacent `../supervisor/exit_receipt.json` to report native exit0 with no resource-stop reason. The associated `launch.json` must identify the registered `n01.train_window008` command and output directory, with matching child/supervisor PIDs. `TRAINING_COMPLETE.json` must report57000 G/144000 D, all three19000-G/16000-D-per-hypothesis endpoints, full parent coverage, no noise/population/C2ST updates, and unchanged final checkpoint hashes. Its config/source-manifest hashes must match the training binding and actual registered files; every training-manifest file is rechecked.

The evaluator then verifies all three checkpoint contents: schema/arm, fixed update counts, final low-rate stage, identical registered config/statistics, shared common initializer and frozen005 state identity. It loads and freezes all response models and frozen005, hashes checkpoint/data/noise/source files, and writes `FROZEN_INPUTS.json`. Only then can it call the DEV loader. Native/completion/launch/binding receipts are part of this freeze, and all final states and bound file hashes are checked again after evaluation.

The freeze receipt contains the resolved evaluation settings, config and statistics digests, every `src/n01/*.py` source hash, the training source manifest, runtime versions and deterministic-execution settings. The training manifest excludes this separately developed evaluator. Its runtime `FROZEN_INPUTS.json` binds sources before DEV decoding and scientific forward calls; this alone is not proof of an external evaluator freeze before process launch. The separate evaluation launcher must supply that prelaunch source manifest. The evaluator requests deterministic algorithms, disables cuDNN benchmarking and TF32, and records its CUDA workspace setting. Cross-device bitwise identity is not assumed.

Before loading models onto CUDA, the evaluator verifies that the selected device has at least the registered `gpu_memory_limit_GiB` and applies that byte limit through the PyTorch CUDA allocator fraction. The freeze receipt records requested bytes, physical device memory and the applied fraction. This allocator limit does not include every CUDA context/driver allocation; the separate supervisor enforces its process/GPU/RSS/time/output bounds. CPU synthetic fixtures remain supported and explicitly record that no CUDA allocator limit was applied.

If an exception occurs, `FAILED.json` records the error and whether DEV decoding started/completed; partial artifacts are preserved. Missing/incomplete endpoints, changing bindings, malformed data, nonfinite outputs and broken engineering contracts stop evaluation. Search-boundary flags, nonpositive observed Q and guarded PSD do not remove rows or choose another setting.

## Populations, pairing and observation

TRAIN retains all **6,593 SOURCE parents** and **21,047 REAL rows**. DEV retains the existing **512 SOURCE parents** and **599 REAL rows**, and is transformed through the same Window008 clean-SOURCE or canonical-REAL observation. TRAIN/DEV source IDs, REAL IDs and REAL original groups must remain disjoint; DEV IDs must match the identity-only preparation binding. Previously exposed DEV is reported separately and never selects settings or endpoints.

Each SOURCE parent receives four draws in fixed parent-major order, with 128 generated rows per chunk: 26,372 TRAIN rows and 2,048 DEV rows per arm/hypothesis. Response, original005 and extension streams use the separately registered seeds. Every arm receives the same response latents and same electronics realizations. Every hypothesis shares one unchanged005 prefix, its S and the same 284 extension Gaussian values. Randomness is drawn once per chunk; sample artifacts require no extra forward or draw.

The model receives the pristine SOURCE 384-sample voltage carrier. Generated signal plus noise passes through the common ADC, original100 baseline subtraction, first argmax in `[112,144)` and `[p-16,p+240)` crop. REAL retains its canonical ADC values; SOURCE is not digitized. Waveform comparisons use 256 samples, total Q uses all 256, and tail starts at output sample46. No old 480-point charge function or old charge bins enter this module.

The explicit generated ADC baseline100 is checked for bitwise equality across arms and hypotheses. Noise distances consume that baseline, never the first100 of the cropped pulse. Model strict-zero signal prefix and canonical-charge residual are recorded independently of final noisy-window readout.

## Saved observables and exact decomposition

`TRAIN_observables.npz` and `DEV_observables.npz` save every parent/draw identity and all generated rows. Per arm, fields include `qsig`, `qbar`, `time_scale`, canonical charge/local-charge checks and warp counters. Per arm/hypothesis, fields include Q, tail, guarded PSD, raw PSD, baseline mean, selected/global peak, crop endpoints, ADC/search/Q flags and the exact same-crop readout decomposition.

For the final observed peak p, let `P_p(v)` gather256 values and subtract `mean(v[0:100])`. For `pre = signal + noise` and `ADC_delta = ADC(pre) - pre`, the module saves:

```text
Q_observed = Q_signal + Q_noise + Q_ADC_delta
tail_observed = tail_signal + tail_noise + tail_ADC_delta
Q_signal_crop_migration = Q_signal - qsig
```

All terms use the final observed crop and each term's own original100 baseline contribution. ADC residual includes clipping and rounding. It does not use separate signal-only and noise-only peak searches. Residuals of the two additive identities are saved for every row.

PSD is not clipped to `[0,1]`. `PSD_raw` preserves signed ratios and NaN when Q is exactly zero. The fixed TRAIN loss `q_floor` gives a signed denominator guard for the finite numerical PSD feature; guard/nonpositive/undefined counts remain explicit. Raw-PSD summaries count undefined values rather than silently dropping their parent rows. NPZ preserves numerical NaNs; JSON uses null with counts or defined-bootstrap-replicate metadata.

`*_curves.npz` contains population and conditional mean/std curves, original REAL curves and explicit baseline100 mean/std. Standard deviations use population `ddof=0`, matching empirical distribution widths. `*_sample_waveforms.npz` saves the first128 generated rows, their parent/draw identities, original SOURCE carriers, pure signal carriers, electronics carriers and ADC baselines. These are 128 draw rows, not necessarily128 distinct parents.

Each generation JSON records per-chunk latent hashes, selected parent-row hashes, output-waveform/observable hashes, common baseline hashes, summaries and engineering checks. Initial/final explicit CPU generator states are saved in `*_stream_states.npz`. Exact CPU synthetic replay has been checked; actual CPU/CUDA scientific replay remains separate evidence.

## Matched population distances and REAL/REAL reference

Matching uses **2,048 TRAIN** or **256 DEV** SOURCE parents without replacement, selected only by the frozen seed, and their first generated draw. The DEV count defaults to the root-specified256 when the config omits `matched_DEV_per_class`; it is made explicit in the freeze receipt. This is a population comparison, not a paired SOURCE-to-REAL target assignment.

REAL A and B are disjoint fixed random selections with equal quotas per original group. A seeded remainder rule allocates any extra rows before inspecting outcomes. Inadequate group support stops evaluation rather than changing quotas. The original full REAL population remains in the separate population summaries.

An independent, frozen evaluation projection bank drives the Window008-v2 waveform, Q/PSD and explicit-baseline100 noise distances. All arms and hypotheses use the same bank. The REAL B versus REAL A comparison supplies the matched empirical reference. The evaluator trains no classifier and makes zero C2ST or other optimizer updates.

`*_matched_waveforms.npz` saves each arm/hypothesis's matched256 waveforms, shared generated baseline100, REAL A/B waveforms and baselines, and parent IDs. `MATCHED_SELECTIONS.json` additionally preserves original groups. `DISTRIBUTION_DISTANCES.json` records each hypothesis separately and reports full-minus-deterministic/full-minus-simple distances and relative distance reductions when the reference denominator is positive. There is no favorable-hypothesis selection or averaging presented as electronics uncertainty.

## Conditional widths and parent bootstrap

Two distinct grouping rules are retained:

1. **Output Q in REAL TRAIN bins:** the frozen Window008 loss bins use the extended-log Q coordinate. All rows, including guarded/nonpositive Q, receive a bin. Membership can differ across arms and hypotheses because it follows their output.
2. **Fixed SOURCE Q in SOURCE TRAIN bins:** the original clean SOURCE Q selects a bin using the separately frozen preparation quantiles. A given parent keeps that bin for every draw, arm and hypothesis.

The same bin number in these two schemes does not imply the same physical population. Each generated conditional summary gives rows, unique parents, Q/PSD means/widths/quantiles and invalid/guard counts; source-to-output migration counts and per-bin waveform curves are also saved. REAL has output-Q conditional summaries, without inventing a paired SOURCE bin.

The 1,000 bootstrap replicates resample whole SOURCE parents and keep all four of each parent's draws together. A single parent weight is shared across every arm/hypothesis. Bin boundaries and frozen endpoint memberships are not refitted. Means and standard deviations for Q and guarded PSD are saved with percentile95 intervals. The full-versus-deterministic and full-versus-simple comparisons include paired mean differences, width differences and width ratios. Ratios with zero reference width are explicitly undefined.

Variance is computed from each parent/bin's centered mean and within-parent M2, adding the weighted between-parent squared deviations. It does not subtract two nearly equal raw second moments. If the retained numerical observations have exactly identical minimum and maximum, their width is exactly zero; this is a numerical identity, not a scientific width threshold. Independent synthetic review caught the earlier cancellation error before any scientific evaluation, and that implementation was replaced.

The same paired replicates also give each arm's range of means across the three hypotheses and its maximum/minimum conditional width ratio. Mean ranges are provided both in original units and divided by the fixed REAL population standard deviation. This reports sensitivity to the three specified laws; it is not a bound over unidentified electronics models.

The registered support flag is at least32 distinct parents per bin. Unsupported bins and their numerical results remain visible and marked; they do not become a success condition or a row-selection rule. REAL is held fixed in this SOURCE-parent bootstrap, so these intervals do not cover REAL acquisition uncertainty. Output-Q conditional differences can include different membership; the common SOURCE grouping provides the separate fixed-parent view.

## Interpretation and synthetic evidence

No new scientific pass threshold is added. Full's narrower/wider conditional PSD or lower population distance alone does not establish the correct stochastic detector response. The evidence is unpaired, from one acquisition, with previously exposed DEV and three finite noise assumptions. `SUMMARY.json` therefore separates fixed-endpoint evaluation completion from `response_match_not_established` and records the absence of independent-acquisition or paired-event physical truth.

The native CPU test used PyTorch2.10.0+cpu, NumPy2.2.6 and deterministic algorithms. Eight checks passed: signed/zero-Q retention, balanced disjoint REAL sampling, fresh-window Q bins, actual three-model common observation and decomposition, artifact payload types, bitwise paired replay, 1,000 parent bootstrap replicates including empty/support-limited bins, and frozen-projection distances using explicit baseline100. It performed96 artificial model row-forwards across two identical runs, zero science-array decodes, zero scientific training updates and no disk-data writes. Maximum Q identity residual was `1.4617484739992692e-15` V ns; maximum tail residual was `1.787307281342354e-16` V ns.

Five further CPU checks passed: rejection of a changed endpoint budget; bootstrap point widths against direct population standard deviations; known paired hypothesis-width and arm-width ratios; round-trip reopening of every bootstrap NPZ array in memory with `allow_pickle=False`; and an injected freeze failure that prevented the DEV loader from being called and preserved false DEV-read flags. These additional checks also decoded zero science arrays and wrote no disk data.

The native completion gate additionally accepted a fully synthetic valid chain and rejected eleven altered metadata cases covering native exit/resource-stop reason, PID binding, update counts, config/manifest bindings, missing arms, both final-checkpoint hash fields, launch module and parent coverage. These were metadata-only mocks with zero scientific checkpoint or array reads.

Eight resource/output checks used CPU and mocked CUDA metadata: CPU bypass; invalid limit rejection; correct 10/24 allocation fraction; undersized-device rejection; unavailable/unsupported-device rejection; alternate-output boundary rejection; default/fresh replay acceptance; and exclusion of the edited evaluator/docs from the frozen training manifest. These checks made no GPU allocation or science-array read and wrote no disk data.

After the stable-variance repair, a separate synthetic 11-parent/four-draw fixture checked all 1,000 paired bootstrap replicates against explicit row resampling: 288,000 mean/std positions, including 3,844 exact zero widths and 104,732 empty positions. Every exact zero was preserved and every zero reference-width ratio was undefined. Maximum absolute mean error was `2.842170943040401e-14`; maximum standard-deviation error was `7.105427357601002e-15`. This fixture decoded no science arrays and wrote no disk data; independent review is recorded separately.

Independent review then passed nine checks with native exit0, including means/stds, intervals, paired ratios, hypothesis sensitivity, undefined-value masks and seventeen native-completion rejection cases. Its receipt is `receipts/window008_20261013/evaluation_review_01/REVIEW_AFTER_VARIANCE_FIX.json` (SHA256 `7f72b70609e534fbec0e99fe1d8e4a29cc28e598ff902cd3c39d1078df644ab9`). This is synthetic/metadata review, not scientific endpoint execution.

These checks did not open a scientific endpoint checkpoint, execute the full 6,593/512-parent evaluation, measure GPU performance, or validate detector response. The complete evaluator still requires the frozen three-arm19k endpoint and its independent execution receipt.
