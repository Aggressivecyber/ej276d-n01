"""WP1 boundary localization using only frozen saved mechanism012 outputs.

No source/REAL pulse container, checkpoint, Torch, ADC observation, neural model,
loss projection, CDF distance, gradient or new direction is evaluated here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from wp1_input_information import (finite_summary, frozen_inputs, fresh_output,
    json_exclusive, json_read, load_members, npz_exclusive, npz_headers,
    require, sha256, unchanged)


HYPOTHESES = ("fast", "slow", "common_level")
PROBE_PATHS = ("S_clamp_surrogate_fixed_actual_crop", "H_fixed_hard_ADC_fixed_actual_crop", "H_moving_same_hard_ADC_first_argmax")
STEPS = (.1, 1, 10)
SIGNS = (-1, 1)
ADC_LEVELS = 16383


def saved_prefix(step, sign):
    return "probe_"+str(step).replace(".", "p")+"_"+("minus" if sign < 0 else "plus")


def scalar(value):
    import numpy as np
    value = np.asarray(value).item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def selected_members(inventory):
    """Whitelist 012 SOURCE identity and boundary fields; no REAL wave reads."""
    names = set(inventory)
    wanted = {"source_parent_ids", "source_parent_rows", "source_parent_row"}
    if "probe_carrier_direction" in names:
        wanted.add("probe_carrier_direction")
    for hypothesis in HYPOTHESES:
        for key in ("actual_peak_index", "actual_crop_start", "actual_crop_stop", "actual_Q", "actual_tail", "actual_PSD",
                    "actual_window_V", "first_runnerup_margin_V", "peak_tie_count", "peak_tied",
                    "signed_influence_wave", "signed_influence_QPSD"):
            wanted.add(hypothesis+"_"+key)
        wanted.update(name for name in names if name.startswith(hypothesis+"_actual_flag_"))
    wanted.update(name for name in names if name.startswith("probe_") and name != "probe_carrier_direction")
    require(wanted <= names, "012 saved boundary members are incomplete")
    return sorted(wanted)


def row_identities(rows, draws_per_parent):
    import numpy as np
    ordinal = np.asarray(rows["source_parent_row"])
    parent_rows = np.asarray(rows["source_parent_rows"])
    ids = np.asarray(rows["source_parent_ids"])
    require(ordinal.ndim == 1 and len(ordinal) == len(parent_rows)*draws_per_parent and len(ids) == len(parent_rows), "012 SOURCE parent/draw structure changed")
    require(np.issubdtype(ordinal.dtype, np.integer) and np.array_equal(ordinal, np.repeat(np.arange(len(parent_rows)), draws_per_parent)), "012 local parent ordinal/draw grouping changed")
    require(len(np.unique(parent_rows)) == len(parent_rows) and len(np.unique(ids)) == len(ids), "012 repeated parent identities malformed")
    return dict(saved_row=np.arange(len(ordinal), dtype=np.int64), source_parent_ordinal=ordinal.copy(),
        source_original_TRAIN_row=parent_rows[ordinal], source_id=ids[ordinal],
        draw_index=np.tile(np.arange(draws_per_parent, dtype=np.int64), len(parent_rows)),
        source_role=np.full(len(ordinal), "TRAIN", dtype="<U5"))


def comparison_summary(value, changed, parent):
    import numpy as np
    return dict(all_draws=finite_summary(value), changed_draws=finite_summary(np.asarray(value)[changed]),
        unchanged_draws=finite_summary(np.asarray(value)[~changed]),
        changed_draw_count=int(changed.sum()), changed_parent_count=int(len(np.unique(parent[changed]))),
        draw_denominator=len(changed), parent_denominator=int(len(np.unique(parent))),
        summary_is_an_observable_change_not_a_row_loss=True)


def boundary_saved_analysis(rows, probe, point, settings):
    """Pure algebra on saved arrays; an artificial fixture can supply tiny rows."""
    import numpy as np
    require(tuple(settings["steps_LSB"]) == STEPS and tuple(settings["signs"]) == SIGNS,
            "all six registered signed steps must be retained")
    require(tuple(settings["paths"]) == PROBE_PATHS, "012 saved three-path contract changed")
    identities = row_identities(rows, settings["draws_per_parent"])
    n = len(identities["saved_row"])
    parent = identities["source_original_TRAIN_row"]
    original_comparisons = point["comparisons"]
    original_global = float(point["weighted_terms"]["wave"]+point["weighted_terms"]["QPSD"])
    require(np.isfinite(original_global), "012 saved original global proxy is nonfinite")
    report = dict(schema="n01/wp1-boundary-saved/v1", original_global_wave_plus_2QPSD=original_global,
        reference_direction_AD_loss_per_V=probe.get("d_AD_loss_per_V"), direction_defined=probe["direction_defined"],
        all_six_signed_probes_reported=True, new_probe_or_best_step_selection=False, hypotheses=list(HYPOTHESES),
        original_SOURCE_parents=int(len(np.unique(parent))), original_SOURCE_draws=n, signed_probes=[], central_differences=[],
        interpretation=dict(global_losses_are_saved_whole_population_distributions=True,
            row_Q_tail_changes_are_not_row_losses=True, endpoint_signed_influence_is_not_causal_contribution=True,
            saved_anchor_signed_influence_does_not_describe_the_signed_probe=True,
            probe_ADC_or_window_arrays_not_saved=True, perturbed_search_runnerup_margin="undefined_not_saved",
            perturbed_search_peak_tie_count="undefined_not_saved",
            probe_window_shape_peak_tie_flag_is_not_the_search_peak_tie_count=True,
            quantizer_half_grid_ties_are_distinct_from_peak_ADC_ties=True,
            no_NEW_ADC_or_NN_replay_no_loss_or_CDF_recompute=True))
    output = dict(identities)
    changed_events = []
    if not probe["direction_defined"]:
        require(len(probe["signed_probes"]) == 6 and all(not item["defined"] for item in probe["signed_probes"]), "012 undefined direction entries missing")
        report["signed_probes"] = probe["signed_probes"]
        report["reason"] = probe["reason"]
        return report, output, changed_events
    direction = np.asarray(rows["probe_carrier_direction"], dtype=np.float64)
    require(direction.shape == (n, 384) and np.isfinite(direction).all(), "012 saved carrier direction malformed")
    lookup = {(float(item["step_LSB"]), int(item["sign"])): item for item in probe["signed_probes"]}
    require(set(lookup) == {(step, sign) for step in STEPS for sign in SIGNS}, "012 saved signed probes missing/duplicated")
    for step in STEPS:
        for sign in SIGNS:
            saved = lookup[(step, sign)]
            delta = float(saved["delta_V"])
            require(abs(delta-sign*step/ADC_LEVELS) <= 1e-18 and saved["defined"], "012 saved probe step mismatch")
            prefix = saved_prefix(step, sign)
            record = dict(step_LSB=step, sign=sign, delta_V=delta, path_global_losses={}, hypotheses={})
            for path in PROBE_PATHS:
                loss = float(saved["paths"][path]["mean_h_proxy"])
                require(np.isfinite(loss), "012 saved signed global loss nonfinite")
                score_from_parts = np.mean([float(saved["paths"][path]["hypotheses"][h]["wave"])
                    +2*float(saved["paths"][path]["hypotheses"][h]["QPSD_fixed"]) for h in HYPOTHESES])
                require(abs(loss-score_from_parts) <= 1e-12, "012 stored whole-proxy combination differs from its parts")
                record["path_global_losses"][path] = dict(value=loss, delta_from_saved_original=loss-original_global,
                    per_hypothesis={h: dict(wave=float(saved["paths"][path]["hypotheses"][h]["wave"]),
                        delta_wave=float(saved["paths"][path]["hypotheses"][h]["wave"])-float(original_comparisons[h]["actual_wave"]),
                        QPSD_fixed=float(saved["paths"][path]["hypotheses"][h]["QPSD_fixed"]),
                        delta_2QPSD=2*(float(saved["paths"][path]["hypotheses"][h]["QPSD_fixed"])-float(original_comparisons[h]["actual_QPSD"]))) for h in HYPOTHESES})
            values = {path: record["path_global_losses"][path]["value"] for path in PROBE_PATHS}
            record["global_H_fixed_minus_S"] = values[PROBE_PATHS[1]]-values[PROBE_PATHS[0]]
            record["global_H_moving_minus_H_fixed"] = values[PROBE_PATHS[2]]-values[PROBE_PATHS[1]]
            for h in HYPOTHESES:
                original_peak = np.asarray(rows[h+"_actual_peak_index"])
                original_start = np.asarray(rows[h+"_actual_crop_start"])
                require(original_peak.shape == (n,) and np.array_equal(original_start, original_peak-16)
                    and np.array_equal(rows[h+"_actual_crop_stop"], original_peak+240), "012 original crop geometry differs")
                moving_stem = prefix+"_"+PROBE_PATHS[2]+"_"+h
                fixed_stem = prefix+"_"+PROBE_PATHS[1]+"_"+h
                surrogate_stem = prefix+"_"+PROBE_PATHS[0]+"_"+h
                moving_peak = np.asarray(rows[moving_stem+"_peak_index"])
                changed = moving_peak != original_peak
                require(np.array_equal(changed, rows[moving_stem+"_peak_changed"]), "012 stored peak-change mask differs")
                require(np.array_equal(rows[fixed_stem+"_peak_index"], original_peak)
                    and np.array_equal(rows[surrogate_stem+"_peak_index"], original_peak), "fixed path crop was not fixed")
                require(int(changed.sum()) == saved["paths"][PROBE_PATHS[2]]["hypotheses"][h]["peak_changed_draws"], "012 JSON peak counts disagree with stored arrays")
                observable_deltas = {}
                for observable in ("Q", "tail", "PSD"):
                    reference = np.asarray(rows[h+"_actual_"+observable], dtype=np.float64)
                    for path in PROBE_PATHS:
                        name = prefix+"_"+path+"_"+h
                        value = np.asarray(rows[name+"_"+observable], dtype=np.float64)
                        require(value.shape == reference.shape == (n,), "012 saved observable shape changed")
                        output[name+"_"+observable] = value
                        output[name+"_delta_"+observable] = value-reference
                    for label, left, right in (("Hmoving_minus_Hfixed", moving_stem, fixed_stem),
                                              ("Hfixed_minus_S", fixed_stem, surrogate_stem)):
                        difference = np.asarray(rows[left+"_"+observable])-np.asarray(rows[right+"_"+observable])
                        output[prefix+"_"+h+"_"+label+"_"+observable] = difference
                        observable_deltas[label+"_"+observable] = comparison_summary(difference, changed, parent)
                    # Identical ADC and identical gather for unchanged rows imply exact equal observables.
                    require(np.array_equal(np.asarray(rows[moving_stem+"_"+observable])[~changed],
                                           np.asarray(rows[fixed_stem+"_"+observable])[~changed]), "same-ADC unchanged crop observables differ")
                output[prefix+"_"+h+"_peak_changed"] = changed
                output[prefix+"_"+h+"_peak_old"] = original_peak
                output[prefix+"_"+h+"_peak_new"] = moving_peak
                output[prefix+"_"+h+"_crop_shift_samples"] = moving_peak-original_peak
                for path in PROBE_PATHS:
                    stem = prefix+"_"+path+"_"+h
                    for name in rows:
                        if name.startswith(stem+"_flag_"):
                            output[name] = np.asarray(rows[name])
                reference_margin = np.asarray(rows[h+"_first_runnerup_margin_V"])
                info = dict(changed_draws=int(changed.sum()), changed_parents=int(len(np.unique(parent[changed]))),
                    draw_denominator=n, parent_denominator=int(len(np.unique(parent))),
                    reference_runnerup_margin_V=comparison_summary(reference_margin, changed, parent),
                    reference_peak_tied_changed_draws=int(np.asarray(rows[h+"_peak_tied"])[changed].sum()),
                    reference_quantizer_halfgrid_tied_changed_draws=int(np.asarray(rows[h+"_actual_flag_ADC_halfgrid_tied"])[changed].sum()),
                    probe_quantizer_halfgrid_tied_changed_draws=int(np.asarray(rows[moving_stem+"_flag_ADC_halfgrid_tied"])[changed].sum()),
                    probe_window_shape_peak_tied_changed_draws=int(np.asarray(rows[moving_stem+"_flag_shape_peak_tied"])[changed].sum()),
                    perturbation_search_runnerup_margin=None, perturbation_search_runnerup_margin_reason="full_probe_ADC_search_not_saved",
                    observable_deltas=observable_deltas)
                window = np.asarray(rows[h+"_actual_window_V"])
                require(window.shape == (n, 256), "saved actual waveform shape changed")
                for index in np.flatnonzero(changed):
                    old, new = int(original_peak[index]), int(moving_peak[index])
                    candidate_position = new-int(original_start[index])
                    candidate_in_saved_crop = 0 <= candidate_position < 256
                    event = dict(step_LSB=step, sign=sign, delta_V=delta, hypothesis=h,
                        **{name: scalar(value[index]) for name, value in identities.items()},
                        peak_old=old, peak_new=new, crop_start_old=old-16, crop_stop_old=old+240,
                        crop_start_new=new-16, crop_stop_new=new+240, crop_shift_samples=new-old,
                        reference_runnerup_margin_V=scalar(reference_margin[index]),
                        reference_peak_search_tie_count=scalar(rows[h+"_peak_tie_count"][index]),
                        reference_ADC_halfgrid_tie_cells=scalar(rows[h+"_actual_flag_ADC_halfgrid_tie_cells"][index]),
                        reference_ADC_halfgrid_min_distance=scalar(rows[h+"_actual_flag_ADC_halfgrid_min_distance"][index]),
                        probe_ADC_halfgrid_tie_cells=scalar(rows[moving_stem+"_flag_ADC_halfgrid_tie_cells"][index]),
                        probe_ADC_halfgrid_min_distance=scalar(rows[moving_stem+"_flag_ADC_halfgrid_min_distance"][index]),
                        probe_window_shape_peak_tied=scalar(rows[moving_stem+"_flag_shape_peak_tied"][index]),
                        probe_search_runnerup_margin_V=None, probe_search_runnerup_margin_reason="full_probe_ADC_search_not_saved",
                        reference_new_candidate_in_saved_actual_crop=bool(candidate_in_saved_crop),
                        reference_ADC_old_peak_minus_new_candidate_V=None if not candidate_in_saved_crop else float(window[index, 16]-window[index, candidate_position]),
                        pre_clamp_direction_delta_new_minus_old_V=float(delta*(direction[index, new]-direction[index, old])),
                        Qtail={observable: dict(actual=scalar(rows[h+"_actual_"+observable][index]),
                            S=scalar(rows[surrogate_stem+"_"+observable][index]), H_fixed=scalar(rows[fixed_stem+"_"+observable][index]),
                            H_moving=scalar(rows[moving_stem+"_"+observable][index]),
                            H_moving_minus_H_fixed=scalar(output[prefix+"_"+h+"_Hmoving_minus_Hfixed_"+observable][index]))
                            for observable in ("Q", "tail", "PSD")},
                        saved_A_anchor_endpoint_signed_influence={name: scalar(rows[h+"_signed_influence_"+name][index]) for name in ("wave", "QPSD")},
                        influence_scope="different saved A-anchor/current comparison; local linearization, not probe row loss or causal share")
                    changed_events.append(event)
                record["hypotheses"][h] = info
            report["signed_probes"].append(record)
    for step in STEPS:
        minus = next(item for item in report["signed_probes"] if item["step_LSB"] == step and item["sign"] == -1)
        plus = next(item for item in report["signed_probes"] if item["step_LSB"] == step and item["sign"] == 1)
        difference = {path: (plus["path_global_losses"][path]["value"]-minus["path_global_losses"][path]["value"])/(2*step/ADC_LEVELS) for path in PROBE_PATHS}
        original = next(item for item in probe["central_differences"] if item["step_LSB"] == step)
        require(all(abs(difference[path]-original["paths"][path]["central_difference_loss_per_V"]) <= 1e-10 for path in PROBE_PATHS), "saved central difference arithmetic mismatch")
        report["central_differences"].append(dict(step_LSB=step, loss_per_V=difference,
            saved_AD_loss_per_V=probe["d_AD_loss_per_V"], Hmoving_minus_Hfixed=difference[PROBE_PATHS[2]]-difference[PROBE_PATHS[1]]))
    report["changed_event_records"] = len(changed_events)
    report["changed_events_are_all_reported_not_only_smallest_or_best_step"] = True
    return report, output, changed_events


def run_saved_boundary(config_path, expected_sha256, output):
    started = time.monotonic()
    config, settings, paths, checked = frozen_inputs(config_path, expected_sha256, "boundary_analysis")
    decoded = []
    output = fresh_output(output)
    try:
        complete = json_read(paths["complete"])
        require(complete["engineering_checks_passed"] and complete["inputs_unchanged"]
                and complete["head_tensors_unchanged"], "012 saved input integrity completion missing")
        require(sha256(paths["row_mechanisms"]) == complete["artifacts"]["ROW_MECHANISMS.npz"]["sha256"]
            and sha256(paths["directional_probe"]) == complete["artifacts"]["DIRECTIONAL_PROBE.json"]["sha256"]
            and sha256(paths["point_replay"]) == complete["artifacts"]["POINT_REPLAY.json"]["sha256"], "012 saved artifact chain differs")
        inventory = npz_headers(paths["row_mechanisms"])["members"]
        rows = load_members(paths["row_mechanisms"], selected_members(inventory), decoded)
        report, arrays, events = boundary_saved_analysis(rows, json_read(paths["directional_probe"]), json_read(paths["point_replay"]), settings)
        require(report["original_SOURCE_draws"] == settings["source_draws"] and report["original_SOURCE_parents"] == settings["source_parents"], "012 saved population count changed")
        json_exclusive(output/"BOUNDARY_SAVED_AUDIT.json", report)
        json_exclusive(output/"CHANGED_EVENTS.json", dict(rows=events, row_count=len(events), row_loss_or_causal_share_established=False))
        npz_exclusive(output/"BOUNDARY_ROWS.npz", arrays)
        unchanged(checked)
        artifact_inventory = {path.name: dict(bytes=path.stat().st_size, sha256=sha256(path)) for path in output.iterdir() if path.is_file()}
        require(sum(item["bytes"] for item in artifact_inventory.values()) <= settings["output_limit_bytes"], "WP1 boundary output budget exceeded")
        require(time.monotonic()-started <= settings["wall_limit_s"], "WP1 boundary wall budget exceeded")
        json_exclusive(output/"COMPLETE.json", dict(schema="n01/wp1-boundary-saved-complete/v1", complete=True,
            artificial_only=bool(config.get("artificial_only", False)),
            elapsed_s=time.monotonic()-started, immutable_files_sha256=checked, inputs_unchanged=True,
            decoded_named_members=decoded, artifacts=artifact_inventory,
            original_SOURCE_or_REAL_waveform_decodes=0, DEV_numeric_reads=0,
            model_forwards=0, autograd_queries=0, checkpoint_loads=0, ADC_evaluations=0,
            new_loss_or_CDF_evaluations=0, optimizer_created=0, optimizer_steps=0, scientific_training_updates=0))
        return report
    except BaseException as error:
        json_exclusive(output/"FAILED.json", dict(error_type=type(error).__name__, error=str(error),
            elapsed_s=time.monotonic()-started, decoded_named_members=decoded, complete=False,
            automatic_restart=False, model_forwards=0, autograd_queries=0, checkpoint_loads=0, optimizer_steps=0))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_saved_boundary(args.config, args.config_sha256, args.out)
    print(json.dumps(dict(status="saved_boundary_complete", direction_defined=report["direction_defined"], model_forwards=0, autograd_queries=0)))


if __name__ == "__main__":
    main()
