# Window008 fix01 evaluation adapter handoff

Status: **static handoff complete; adapter implementation and evaluation pending**.
Run: `window008_fix01_20261013`; registered remote root:
`/home/rylan/projects/ej276d-n01/window008_fix01_20261013`.
This handoff adds no source module and performs no checkpoint load, scientific forward, optimizer update, GPU work, or job polling.

## 1. Minimal adapter boundary

Keep `src/n01/evaluate_window008.py` and the three fix01 model/engine/driver files unchanged.
Implement a separately versioned evaluator with a fix01 completion gate, mixed-model endpoint loader, input-freeze receipt, and orchestration entry point.
The original `training_completion_gate`, `_freeze`, and `evaluate_run` assume the original run, checkpoint schema, and model class; do not call them as the fix01 gate or bypass their checks.
Reuse these original numerical interfaces without changing their definitions:

- `_fixed_settings(cfg)`, `_validate_split(data, split)`, `_load_dev(container, identities, spec, q_floor)`.
- `balanced_real_domains(groups, count, seed)` and `generate_split(split, data, models, extension, statistics, cfg, settings, destination, selected, device)`.
- `parent_bootstrap(generated, statistics, cfg, settings, split)` and `distance_receipt(objective, fake, real, fake_baseline, real_baseline, context)`.
- `save_json`, `save_npz`, `_array`, `_clean`, and the existing readout/binning helpers used by generation.
- `_verify_frozen(models, extension, receipt)`, after supplying its required `noise_state_sha256`, `checkpoint_state_sha256`, `files_sha256`, and `sources_sha256` fields.

Pass a mixed `models` mapping to generation. Load `deterministic` with `n01.model008.N01Response008`; load `full` and `simple` with `n01.model008_fix01.N01Response008Fix01`.
Use strict state loading and the checkpoint's recorded arm/statistics/configuration; freeze every model and the existing noise005 extension in evaluation mode.

## 2. Existing fix01 completion contract

The driver writes schema `n01/window008-fix01-training-complete/v1`, status `fixed_recovered_endpoints_complete_pending_native_exit_and_evaluation`, `run_id`, `base_run_id`, `recovery`, `arms`, `files`, and `endpoint_map`.
Each `endpoint_map` value has exactly `{path, sha256, schema, model_implementation, origin}`; paths are relative to the registered run root.

| Arm | Endpoint path | Schema | Origin |
| --- | --- | --- | --- |
| deterministic | `provenance/window008_fix01_reference/deterministic_final.pt` | `response008/v1` | `retained_original_endpoint` |
| full | `runs/science/train/full/final.pt` | `response008_fix01/v1` | `stage3000_recovery` |
| simple | `runs/science/train/simple/final.pt` | `response008_fix01/v1` | `fresh_common_initialization` |

`recovery` binds five fields: `run_id`, `config_sha256`, `source_manifest_sha256`, `base_configuration_sha256`, and `base_source_manifest_sha256`.
The two new checkpoints also record `model_implementation`, `parameter_signature`, `lineage`, `new_G_updates`, `new_D_updates`, and `automatic_resume_allowed=false`.
The retained deterministic checkpoint keeps its original bytes and original schema; do not resave or relabel it.

## 3. Gate before DEV or endpoint evaluation

1. Validate the frozen base configuration/manifest, fix01 configuration/manifest, reference inputs, completion receipt, indexed completion files, endpoint paths, and SHA256 values. Reject missing, extra, aliased, or out-of-root endpoints.
2. Cross-check all three arm receipts against their checkpoints: `G_updates=19000`, each of `fast/slow/common_level` has `D_updates=16000`, final stage `low_learning_rate`, coverage `6593` SOURCE and `21047` REAL, and the registered common-initialization/noise005 bindings.
3. Validate full lineage from the frozen original stage3000 checkpoint and simple lineage from the common initialization. Cross-check `endpoint_map`, arm rows, checkpoint schema/class, and state digests; additionally match the recovery identity on both new checkpoints.
4. Preserve and verify the old run's original native **exit 1**, failure receipt, launch binding, and stage3000/deterministic provenance. A completed deterministic arm does not change that failed job's exit status.
5. Independently verify the new supervisor's native **exit 0**, null termination reason, exact registered command/module/arguments, wrapper/trainer PID binding, and completion-file hashes. The driver's pending-native status and `awaiting_own_supervisor_native_exit` are not substitutes for this receipt.
6. Write and hash-bind the complete evaluation input/source/endpoint freeze before the first DEV decode. Recheck frozen model states, files, and sources after evaluation.

Validate accounting as separate quantities; all values below describe the successful fixed endpoint contract, not a claim that the current run has finished:

| Accounting field prefix | G updates | D updates |
| --- | ---: | ---: |
| `new_execution_` | 35000 | 96000 |
| `effective_comparison_` | 57000 | 144000 |
| `historical_actual_` | 57269 | 144807 |
| `inherited_effective_` | 22000 | 48000 |
| `discarded_legacy_joint_` | 269 | 807 |

## 4. Preserve the frozen numerical evaluation

Use the original base recipe and its exact seed roles: four draws per SOURCE parent, chunk size 128, matched TRAIN 2048 / DEV 256, 1000 parent-bootstrap replicates, and minimum bin count 32.
Retain the original 384-to-256 observation path, baseline100, frozen noise005, fixed SOURCE population weights, three separate hypotheses, and `N01Loss008` evaluation projection context.
Preserve both conditional schemes (`output_Q_in_REAL_TRAIN_bins`, `fixed_SOURCE_Q_in_SOURCE_TRAIN_bins`), Q/PSD summaries, all valid observation rows including nonpositive Q, parent/draw order, matched SOURCE selection, and balanced disjoint REAL A/B selection.
Use TRAIN-fitted statistics and the separately frozen `FIT_DATA_STATISTICS.json` domain summaries. Keep the existing TRAIN/DEV identity checks and prohibit overlap or DEV-derived fitting.
Evaluation performs zero G/D/noise/population/C2ST updates. Keep `scientific_status=response_match_not_established`; evaluation completion and exact replay do not establish detector-response identity.
Use a versioned fix01 evaluation summary/freeze schema that exposes the mixed endpoint map, recovery identity, old failure provenance, new native completion, and the three accounting totals above. Preserve numerical artifact names and semantics.

## 5. Primary and exact replay operations

Register separate fresh job directories under `runs/evaluation_jobs/primary/artifacts` and `runs/evaluation_jobs/replay01/artifacts` in the fix01 run root.
Create each job's start-attempt marker exclusively; preserve every partial or failed attempt and prohibit overwrite or automatic restart.
Adapt the existing operational policy/helpers for the new root/module/manifest/schema and actual launch chain. Freeze the evaluation resource policy separately from the recovery training time limit.
Require the primary evaluator and its supervisor to finish with native exit 0 and valid artifact/completion hashes before starting replay01.
Replay uses identical endpoints, input bytes, seeds, chunking, backend settings, and projection/noise streams in a fresh destination; it adds no unique draws or independent samples.
Reuse the comparison logic in `scripts/verify_window008_evaluation.py` after providing a fix01-aware artifact-integrity gate. Its existing CLI and integrity helper assume the old run/schema.
Compare all 13 NPZ files and all 66 members exactly by dtype, shape, values, and raw-array SHA256; also compare chunk/parent/draw order, stream initial/final states, selections, distances, bootstrap outputs, and JSON semantics.
Retain only the already-declared volatile-field exclusions and archive-container differences; validate each side's own file hashes. Do not broaden tolerances or exclusions to force a pass.

## 6. Next implementation evidence

Implement and independently review the adapter plus operational bindings, then perform artificial-data engineering checks before any scientific evaluation.
Read actual endpoint/native receipts after training completion, execute the gate, and retain explicit primary/replay/native/comparison receipts.
After a correctly started long job, schedule the requested wakeup and stop active monitoring.
