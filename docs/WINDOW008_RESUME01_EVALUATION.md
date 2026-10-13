# Window008 resume01 endpoint evaluation

This adapter evaluates the three fixed Window008 endpoints after resume01's own supervisor has produced native exit 0. It preserves the original numerical evaluator and the two historical native exit 1 records. Evaluation, exact replay, and physical detector-response identity remain separate outcomes; `scientific_status=response_match_not_established` is retained.

The entry point is `src/n01/evaluate_window008_resume01.py`. The immutable numerical core is `src/n01/evaluate_window008.py`, SHA256 `75f8988415a181a31a7c2b5890418b1f7685e0d8fb141cdea84f4174af783cc0`. The adapter imports the original generation, readout, matching, bootstrap, distance, split-validation, and post-evaluation freeze-check functions directly. It supplies new completion, source/policy, checkpoint-loader, and input-freeze gates. It does not call or bypass the original evaluator's original-run completion gate.

## Endpoints and native execution roles

| Arm | Endpoint path recorded at the registered runtime root | Checkpoint schema | Model implementation | Origin |
|---|---|---|---|---|
| deterministic | `provenance/window008_fix01_reference/deterministic_final.pt` | `response008/v1` | `n01.model008.N01Response008` | `retained_original_endpoint` |
| full | `runs/science/train/full/final.pt` | `response008_resume01/v1` | `n01.model008_fix01.N01Response008Fix01` | `fix01_periodic6000_resume` |
| simple | `runs/science/train/simple/final.pt` | `response008_resume01/v1` | `n01.model008_fix01.N01Response008Fix01` | `fresh_common_initialization` |

Each endpoint has G19000, D16000 for each of `fast`, `slow`, and `common_level`, final stage `low_learning_rate`, and complete coverage of 6593 SOURCE / 21047 REAL TRAIN parents. The retained deterministic checkpoint keeps its original bytes and schema. New checkpoints must be complete final boundaries, have `automatic_resume_allowed=false`, and carry the bound resume01 recovery identity and parent fix01 identity.

The source chain verifies the original 68-file manifest, the fix01 114-file manifest, the resume01 manifest, configuration bytes, and all 26 historical reference files. Original native1 represents the numerical failure; fix01 native1 represents the verified user SIGTERM before server shutdown. The gate checks launch arguments, timeout/supervisor/trainer PID roles, input bindings, chronological receipts, and independent resume01 native0. The trainer's pending-native completion fields remain unchanged.

The resume gate uses schema `n01/window008-resume01-training-complete/v1` and status `fixed_resumed_endpoints_complete_pending_native_exit_and_evaluation`. It verifies each indexed completion file, each arm's separate completion JSON, endpoint-map fields and paths, strict model states, common initialization, frozen005 noise, TRAIN statistics, sampler coverage, all named AdamW slots and phase-specific steps, and the first/new/low-learning-rate boundary metadata.

The full restore proof compares every model, discriminator, G/D optimizer, sampler and random-stream byte in `restored_start.pt` with the hash-bound fix01 full6000 periodic checkpoint. It cross-checks the saved import receipt's state and named moment digests. Mean-branch AdamW steps are 6000 and stochastic-branch steps are 3000 at import; final steps are 19000 / 16000. The simple initial model and untrained D state are checked against the original common initializer. Forensic failure checkpoints are hashed as provenance and are never loaded or restored.

| Accounting scope | G updates | D updates across three hypotheses |
|---|---:|---:|
| New resume01 execution | 32000 | 87000 |
| Inherited effective endpoints/state | 25000 | 57000 |
| Effective three-endpoint comparison | 57000 | 144000 |
| Historical Window008 actual total | 57667 | 146001 |
| Discarded original numerical-failure portion | 269 | 807 |
| Discarded user-stop portion after full6000 | 398 | 1194 |

The completion gate rechecks all frozen bytes after CPU metadata inspection and reports zero module forwards, optimizer constructions/updates, and DEV decodes. It returns resolved endpoint paths for either the registered Linux run or a local collected mirror. Immutable provenance resolves against the explicit project source root; new full/simple endpoints resolve against the supplied science directory. No aliases or paths escaping those roots are accepted.

## Independent evaluation freeze and numerical policy

Before the first scientific forward or TRAIN/DEV waveform decode, the runtime must carry `WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json` (`n01/window008-resume01-evaluation-source/v1`) and `window008_resume01_evaluation_policy.json` (`n01/window008-resume01-evaluation-policy/v1`). The additive manifest binds the unchanged training manifests/configurations, the new evaluation policy, every added source, and the exact staged module inventory. The evaluator requires its own source, QA script, this document and policy in the additive file inventory. The staged runtime excludes the unfrozen historical continuation draft. The main freeze can include addon sources and policies before either job uses DEV.

The independently frozen evaluation resource policy is 12 CPU threads, 10 GiB GPU allocator limit, 8 GiB job-tree RSS, 21600 seconds wall limit and 3072 MiB output. It records the primary/replay jobs and their exact replay policy separately from the training receipts. The operational supervisor enforces RSS, wall and disk limits; the evaluator applies deterministic backend settings and the CUDA allocator limit before loading GPU models.

The numerical policy is exactly the original Window008 recipe: four draws per SOURCE parent, chunks of 128, matched TRAIN 2048 / DEV 256, 128 sample waveform rows, 1000 paired SOURCE-parent bootstrap replicates and minimum bin support 32. Seeds retain their original roles. The same 384-to-256 observation path, baseline100, frozen noise005 and three separate noise hypotheses are used. Both conditional schemes remain `output_Q_in_REAL_TRAIN_bins` and `fixed_SOURCE_Q_in_SOURCE_TRAIN_bins`. Nonpositive-Q and guarded/undefined raw-PSD rows remain in the observation population and diagnostics. Statistics and source Q8 edges are fitted on TRAIN and are hash-bound before DEV opens.

`FROZEN_INPUTS.json` has schema `n01/window008-resume01-evaluation-freeze/v1` and status `all_endpoints_frozen_before_DEV_numeric_read`. It records the mixed endpoint map, checkpoint/noise tensor digests, every input/source hash, resolved settings, backend state, recovery identity, historical native chain, successful own native completion and the distinct ledgers. It is written before the first prepared dataset decode and before `_load_dev`. The original freeze-verification function rechecks tensors, files and sources after generation.

`SUMMARY.json` has schema `n01/window008-resume01-evaluation/v1` and status `fixed_resumed_endpoint_evaluation_complete`. Original numeric artifact names and semantics remain unchanged. Evaluation performs zero G/D/noise/population/C2ST updates, refits no statistics or bins, and selects no hypothesis or training setting from DEV.

## Public interfaces and execution

`training_completion_gate(run, cfg, resume_cfg, *, root=None)` is the CPU-only gate. `root` defaults to the source checkout, and `run` must name the collected/registered `runs/science` directory. It can be called independently before any numerical evaluation and returns the complete gate receipt including `endpoint_map`, `endpoint_paths`, `checkpoint_state_sha256`, `files_sha256`, `accounting`, and `native_completion`.

`load_endpoints(completion, cfg, statistics, device='cpu')` loads the strictly verified mixed endpoint mapping in evaluation mode with all gradients disabled. It creates no optimizer and calls no neural forward.

`_freeze(run, cfg, resume_cfg, statistics, dataset_path, statistics_path, raw_container, identities, device, destination, *, source_manifest, policy_path, root=None)` verifies independent evaluation sources and policy, calls the completion gate, freezes model/noise/input state and writes `FROZEN_INPUTS.json`. It returns `(models, extension, freeze, settings, data_statistics)`. Addon jobs may call this interface only after their own preregistered policy gate, with the main frozen evaluation manifest/policy and a fresh addon destination. They must freeze their own policy before their first DEV decode.

The CLI is:

```text
python -B -m n01.evaluate_window008_resume01
  --out <resume-runtime-root>/runs/science
  --config <resume-runtime-root>/window008_config.json
  --resume-config <resume-runtime-root>/window008_resume01_config.json
  --dataset <resume-runtime-root>/data/window008/DATASET008.npz
  --statistics <resume-runtime-root>/data/window008/FIT_STATISTICS008.json
  --raw-container <unchanged-registered-private-container>
  --identities <resume-runtime-root>/data/PARENT_IDENTITIES.npz
  --source-manifest <resume-runtime-root>/WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json
  --policy <resume-runtime-root>/window008_resume01_evaluation_policy.json
  --destination <resume-runtime-root>/runs/evaluation_jobs/primary/artifacts
  --device cuda:0
```

The Python `evaluate_run` takes the original input arguments with explicit `resume_cfg` plus keyword-only `destination`, `source_manifest`, `policy_path`, and optional `root`. Its destinations are exactly fresh `runs/evaluation_jobs/{primary,replay01}/artifacts`. Existing folders are rejected, and failure receipts preserve every partial artifact. Replay01 starts only after separately validated primary evaluator and supervisor native0, uses identical inputs/endpoints/seeds/chunking/backends, and adds no unique draws.

Exact replay compares all 13 NPZ artifacts and all 66 members by dtype, shape, values and raw-array SHA256, together with chunk/parent/draw order, initial/final stream states, selections, distances, bootstrap outputs and JSON semantics. Each side's own archive/artifact hashes must pass. Only the already-declared volatile exclusions and archive-container differences are allowed; tolerances and exclusions are not expanded.

## Engineering evidence

`scripts/check_window008_resume01_evaluation.py --out <fresh E/D receipt directory>` runs the actual collected training-completion gate and mixed CPU loader, plus artificial malformed-metadata/source-policy checks. It blocks `numpy.load`, `nn.Module._call_impl`, AdamW construction and AdamW steps. It rejects forensic failure-checkpoint loads, keeps checkpoint loads on CPU, and performs no scientific dispatch or GPU work. Its artificial source root is labelled and never executed.

The retained receipts under `receipts/window008_resume01_20261013/evaluation_cpu_*` distinguish CPU engineering QA from scientific numerical evaluation. Primary/replay/native/comparison receipts and numerical results are supplied by the separately frozen operational jobs after independent review.
