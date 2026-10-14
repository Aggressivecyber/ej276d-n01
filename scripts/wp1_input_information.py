"""WP1 SOURCE information audit. Header inspection is standard-library only.

Scientific array reads require an explicitly frozen work-package configuration.
This module never imports Torch, loads a checkpoint, or runs a response model.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_exclusive(path, value):
    path = Path(path)
    if path.drive:
        require(path.drive.upper() in ("E:", "D:"), "WP1 outputs must be on E or D")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def npz_headers(path):
    """Read ZIP inventory and only NPY headers, never an array payload."""
    members = {}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            name = entry.filename
            require(name.endswith(".npy") and "/" not in name and "\\" not in name,
                    "NPZ has unexpected member path")
            require(name[:-4] not in members, "NPZ duplicate member")
            with archive.open(entry) as stream:
                require(stream.read(6) == b"\x93NUMPY", "NPZ member is not NPY")
                version = tuple(stream.read(2))
                require(version in ((1, 0), (2, 0), (3, 0)), "unsupported NPY header version")
                size_bytes = 2 if version == (1, 0) else 4
                size = struct.unpack("<H" if size_bytes == 2 else "<I", stream.read(size_bytes))[0]
                require(size <= 1024 * 1024, "oversized NPY header")
                raw = stream.read(size)
                require(len(raw) == size, "truncated NPY header")
                header = ast.literal_eval(raw.decode("utf-8" if version == (3, 0) else "latin1").strip())
                require(set(header) == {"descr", "fortran_order", "shape"}, "unexpected NPY header fields")
            members[name[:-4]] = dict(dtype=header["descr"], shape=list(header["shape"]),
                fortran_order=header["fortran_order"], ZIP_bytes=entry.file_size,
                compressed_bytes=entry.compress_size, array_payload_decoded=False)
    return dict(path=str(Path(path).resolve()), bytes=Path(path).stat().st_size,
                sha256=sha256(path), members=members, array_payload_decodes=0)


def frozen_inputs(config_path, expected_sha256, section_name):
    """Bind the common WP1 config, listed source bytes and inputs before NumPy."""
    require(expected_sha256 and sha256(config_path) == expected_sha256, "WP1 configuration identity mismatch")
    config = json_read(config_path)
    aliases = {"information_analysis": "input_information", "boundary_analysis": "boundary_saved"}
    alias = aliases.get(section_name)
    present = section_name in config
    legacy_present = alias is not None and alias in config
    require(present != legacy_present if alias is not None else present, "missing or ambiguous WP1 analysis configuration section")
    section = config[section_name if present else alias]
    require(config.get("scientific_training_updates", 0) == 0, "information work package cannot train")
    require(config.get("source_files"), "frozen WP1 source inventory missing")
    records = dict(config["source_files"])
    for name, entry in section["inputs"].items():
        records["input:"+name] = entry
    checked = {}
    paths = {}
    for name, entry in records.items():
        path = Path(entry["path"]).resolve(strict=True)
        require(path.is_file(), "bound WP1 input must be a regular file")
        key = str(path)
        actual = checked.setdefault(key, None)
        if actual is None:
            actual = checked[key] = sha256(path)
        require(actual == entry["sha256"], "bound WP1 file changed: "+name)
        if "bytes" in entry:
            require(path.stat().st_size == entry["bytes"], "bound WP1 size changed: "+name)
        if name.startswith("input:"):
            paths[name[6:]] = path
    required_scripts = {"wp1_input_information.py", "wp1_boundary_saved.py"}
    require(required_scripts <= {Path(name).name for name in checked}, "WP1 owned analysis scripts missing from frozen source inventory")
    return config, section, paths, checked


def unchanged(checked):
    for name, digest in checked.items():
        require(sha256(name) == digest, "WP1 immutable file changed: "+name)


def fresh_output(path):
    path = Path(path).resolve()
    if path.drive:
        require(path.drive.upper() in ("E:", "D:"), "WP1 outputs must be on E or D")
    require(not path.exists(), "WP1 output directory already exists; no overwrite")
    path.mkdir(parents=True, exist_ok=False)
    return path


def npz_exclusive(path, arrays):
    import numpy as np
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def load_members(path, members, decoded):
    """A member whitelist prevents REAL or unrelated historical array reads."""
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        require(set(members) <= set(archive.files), "required WP1 named member missing")
        result = {}
        for name in members:
            result[name] = archive[name].copy()
            require(not result[name].dtype.hasobject, "WP1 object arrays are forbidden")
            decoded.append(dict(path=str(path), member=name, shape=list(result[name].shape), dtype=str(result[name].dtype)))
        return result


def canonical_charge_source_np(source):
    """Original A: float64 prefix centering then fixed canonical [110:366]."""
    import numpy as np
    value = np.asarray(source, dtype=np.float64)
    require(value.ndim == 2 and value.shape[1] in (384, 480), "expected full480 or carrier384 SOURCE")
    return (value-value[:, :100].mean(1, keepdims=True))[:, 110:366].sum(1)


def analytic_source480(ee, had):
    """The bound original R2 clean_expectation formula; audit only, no NN."""
    import numpy as np
    ee, had = np.asarray(ee, dtype=np.float64), np.asarray(had, dtype=np.float64)
    require(ee.ndim == had.ndim == 1 and ee.shape == had.shape, "deposition metadata shape mismatch")
    require(np.isfinite(ee).all() and np.isfinite(had).all() and (ee >= 0).all() and (had >= 0).all(),
            "invalid deposition metadata")
    u = np.maximum(np.arange(512)-128., 0)[None, :]
    out = np.zeros((len(ee), 512), dtype=np.float64)
    for energy, taus, fractions in ((ee, (13, 35, 270), (.7, .15, .15)), (.15*had, (13, 59, 460), (.5, .1, .4))):
        for tau, fraction in zip(taus, fractions):
            out += energy[:, None]*fraction*(np.exp(-u/tau)-np.exp(-u/5))/(tau-5)
    visible = ee+.15*had
    require((visible > 0).all(), "zero deposition SOURCE cannot be reconstructed as a legal pulse")
    nominal = (visible+.03918)/.000132506/1000
    out *= nominal[:, None]/out.sum(1, keepdims=True)
    peak = out.argmax(1)
    threshold = out.max(1)*.5
    clock = np.arange(512)[None, :]
    below = (clock < peak[:, None]) & (out < threshold[:, None])
    half_rise = np.where(below, clock, -1).max(1)+1
    indices = (half_rise[:, None]-120+np.arange(480)[None, :]).clip(0, 511)
    return np.take_along_axis(out, indices, axis=1), nominal, half_rise


def finite_summary(value):
    import numpy as np
    value = np.asarray(value, dtype=np.float64)
    defined = value[np.isfinite(value)]
    return dict(rows=int(value.size), defined_rows=int(defined.size), undefined_rows=int(value.size-defined.size),
        minimum=None if not len(defined) else float(defined.min()), maximum=None if not len(defined) else float(defined.max()),
        mean=None if not len(defined) else float(defined.mean()),
        quantile_levels=[0, .01, .5, .99, 1], quantiles=None if not len(defined) else np.quantile(defined, [0, .01, .5, .99, 1]).tolist())


def average_ranks(value):
    import numpy as np
    value = np.asarray(value)
    order = np.argsort(value, kind="stable")
    ordered = value[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1])+1]
    stops = np.r_[starts[1:], len(value)]
    result = np.empty(len(value), dtype=np.float64)
    result[order] = np.repeat((starts+stops-1)/2, stops-starts)
    return result


def association(x, y):
    import numpy as np
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(defined=False, reason="constant_or_insufficient", n=int(len(x)), Pearson=None, Spearman=None)
    return dict(defined=True, n=int(len(x)), Pearson=float(np.corrcoef(x, y)[0, 1]),
                Spearman=float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1]))


def pca_information(matrix, thresholds=(.95, .99, .999), components=16):
    """Centered descriptive PCA with an absolute variance and energy report."""
    import numpy as np
    matrix = np.asarray(matrix, dtype=np.float64)
    require(matrix.ndim == 2 and len(matrix) > 0 and np.isfinite(matrix).all(), "PCA requires defined retained rows")
    mean = matrix.mean(0)
    centered = matrix-mean
    covariance = centered.T @ centered/max(len(matrix)-1, 1)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    bound = np.finfo(np.float64).eps*matrix.shape[1]*max(float(np.abs(eigenvalues).max()), np.finfo(np.float64).tiny)*32
    require(float(eigenvalues.min()) >= -bound, "PCA covariance has more than roundoff negative eigenvalues")
    negative = int((eigenvalues < 0).sum())
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    vectors = vectors[:, ::-1]
    for column in range(vectors.shape[1]):
        pivot = int(np.abs(vectors[:, column]).argmax())
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    total = float(eigenvalues.sum())
    ratios = eigenvalues/total if total else np.zeros_like(eigenvalues)
    dimensions = {str(level): (int(np.searchsorted(np.cumsum(ratios), level, side="left"))+1 if total else 0)
                  for level in thresholds}
    energy = float(np.square(matrix).sum()/len(matrix))
    report = dict(rows=len(matrix), coordinates=matrix.shape[1], centered=True, thresholds=list(thresholds),
        explained_variance_dimensions=dimensions, total_variance=total, mean_row_squared_norm=energy,
        variance_to_row_energy=None if not energy else total/energy,
        eigenvalues=eigenvalues.tolist(), explained_variance_ratios=ratios.tolist(),
        roundoff_negative_eigenvalues_clipped=negative, negative_roundoff_bound=bound,
        dimensions_do_not_by_themselves_distinguish_float32_rounding_from_physical_information=True)
    return report, mean, vectors[:, :min(components, len(eigenvalues))]


def normalized_views(source480, *, charge_floor=1e-8):
    import numpy as np
    raw = np.asarray(source480, dtype=np.float64)
    require(raw.ndim == 2 and raw.shape[1] == 480 and np.isfinite(raw).all(), "expected finite complete SOURCE480")
    prefix = raw[:, :100].mean(1)
    centered = raw-prefix[:, None]
    q = centered[:, 110:366].sum(1)
    peaks = centered[:, 112:144].argmax(1)+112
    indices = peaks[:, None]-16+np.arange(256)[None, :]
    representations = dict(full480=centered, carrier384=centered[:, :384],
                           clean_window256=np.take_along_axis(centered, indices, axis=1))
    views, valid = {}, {}
    for name, value in representations.items():
        peak = value.max(1)
        for normalization, denominator, usable in (("canonical_Q", q, q > charge_floor), ("positive_peak", peak, peak > charge_floor)):
            key = name+"_"+normalization
            result = np.full(value.shape, np.nan, dtype=np.float64)
            result[usable] = value[usable]/denominator[usable, None]
            views[key], valid[key] = result, usable
    return representations, views, valid, dict(prefix_mean_V=prefix, canonical_Q_Vns=q,
        peak_index=peaks, crop_start=peaks-16, crop_stop=peaks+240,
        amplitude480_V=np.abs(centered).max(1), amplitude384_V=np.abs(centered[:, :384]).max(1),
        peak_amplitude480_V=centered.max(1), peak_amplitude384_V=centered[:, :384].max(1),
        Q_clean_window256_Vns=representations["clean_window256"].sum(1),
        tail_clean_window256_Vns=representations["clean_window256"][:, 46:].sum(1))


def numeric_pattern_counts(matrix, decimals=(12, 9, 6)):
    """Exact byte patterns and fixed rounding counts are not template classes."""
    import numpy as np
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    require(np.isfinite(matrix).all(), "numeric pattern census requires defined rows")
    def count(value):
        return len({hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest() for row in value})
    return dict(exact_float64_normalized_byte_patterns=count(matrix),
        fixed_decimal_pattern_counts={str(digits): count(np.round(matrix, digits)) for digits in decimals},
        rounding_decimals=list(decimals), numeric_patterns_are_not_analytic_template_counts=True)


def same_charge_pairs(shape, q, roles, *, relative_q_max=.01, shape_rms_min=.001, chunk=64, max_output=128):
    """All pairs are searched; exact real input vectors verify retained strengths.

    Only same original role parents and different identities are eligible.
    The squared-distance Gram is streamed, so no NxN matrix is retained.
    """
    import numpy as np
    shape, q, roles = np.asarray(shape, dtype=np.float64), np.asarray(q, dtype=np.float64), np.asarray(roles)
    require(shape.ndim == 2 and len(shape) == len(q) == len(roles), "same-Q pair dimensions disagree")
    require(0 < relative_q_max <= .1 and 0 < shape_rms_min < 1 and chunk > 0 and max_output > 0, "invalid frozen pair audit settings")
    valid = np.isfinite(shape).all(1) & np.isfinite(q) & (q > 1e-8)
    work = np.where(valid[:, None], shape, 0)
    square = np.square(work).sum(1)
    eligible_count = np.zeros(len(q), dtype=np.int64)
    best = np.full(len(q), -1, dtype=np.int64)
    strength = np.full(len(q), np.nan, dtype=np.float64)
    for start in range(0, len(q), chunk):
        stop = min(start+chunk, len(q))
        relative = np.abs(q[start:stop, None]-q[None, :])/np.maximum(np.abs(q[start:stop, None]), np.abs(q[None, :])).clip(1e-300)
        eligible = (relative <= relative_q_max) & (roles[start:stop, None] == roles[None, :]) & valid[start:stop, None] & valid[None, :]
        eligible[np.arange(stop-start), np.arange(start, stop)] = False
        eligible_count[start:stop] = eligible.sum(1)
        distance = np.maximum(square[start:stop, None]+square[None, :]-2*(work[start:stop] @ work.T), 0)
        distance[~eligible] = -1
        chosen = distance.argmax(1)
        has = eligible_count[start:stop] > 0
        selected_rows = np.arange(start, stop)[has]
        selected = chosen[has]
        best[selected_rows] = selected
        strength[selected_rows] = np.sqrt(np.square(shape[selected_rows]-shape[selected]).mean(1))
    feasible = np.isfinite(strength) & (strength >= shape_rms_min)
    candidates = np.flatnonzero(feasible)
    order = sorted(candidates.tolist(), key=lambda i: (-float(strength[i]), i))
    pairs, seen = [], set()
    for left in order:
        right = int(best[left])
        identity = tuple(sorted((left, right)))
        if identity in seen:
            continue
        seen.add(identity)
        pairs.append(dict(left_cache_row=int(left), right_cache_row=right, role=str(roles[left]),
            relative_canonical_Q_difference=float(abs(q[left]-q[right])/max(abs(q[left]), abs(q[right]))),
            peak_normalized_shape_RMS_difference=float(strength[left]),
            canonical_Q_left_Vns=float(q[left]), canonical_Q_right_Vns=float(q[right]),
            real_existing_inputs=True, rescaling_or_synthetic_waveform_intervention=False))
        if len(pairs) == max_output:
            break
    report = dict(search="complete same-original-role pair census, streamed Gram then exact selected-vector difference",
        relative_Q_max=relative_q_max, required_peak_normalized_RMS=shape_rms_min, normalization_coordinates=shape.shape[1],
        rows=len(q), rows_with_same_Q_candidates=int((eligible_count > 0).sum()), eligible_ordered_pairs=int(eligible_count.sum()),
        rows_with_testable_shape_contrast=int(feasible.sum()), largest_verified_strength=finite_summary(strength)["maximum"],
        feasible=bool(feasible.any()), undefined_normalization_rows=int((~valid).sum()), pairs=pairs,
        source_shape_feasibility_is_not_response_dependence_or_causal_validation=True,
        no_input_or_row_ID_becomes_a_neural_feature=True)
    return report, dict(same_Q_candidate_count=eligible_count, strongest_same_Q_partner_cache_row=best,
                        strongest_same_Q_shape_RMS=strength, same_Q_shape_contrast_testable=feasible)


def audit_source_information(source480, metadata, training_container, prepared, settings):
    """Pure NumPy analysis; callers bind file bytes and enforce one WP1 run."""
    import numpy as np
    source480 = np.asarray(source480)
    ids = np.asarray(metadata["event_ids"])
    train = np.asarray(metadata["internal_training"], dtype=bool)
    require(source480.shape == (len(ids), 480) and source480.dtype == np.dtype("float32"), "original SOURCE cache dtype/length changed")
    require(len(np.unique(ids)) == len(ids) and train.shape == (len(ids),), "SOURCE parent identity malformed")
    by_id = {str(value): i for i, value in enumerate(ids)}
    train_ids, dev_ids = training_container["source_ids"], training_container["source_val_ids"]
    train_rows = np.asarray([by_id[str(value)] for value in train_ids], dtype=np.int64)
    dev_rows = np.asarray([by_id[str(value)] for value in dev_ids], dtype=np.int64)
    require(np.array_equal(train_rows, np.flatnonzero(train)), "full original SOURCE TRAIN membership/order changed")
    require(not train[dev_rows].any() and len(np.unique(dev_rows)) == len(dev_rows), "fixed DEV identities changed")
    require(np.array_equal(source480[train_rows], training_container["source"])
            and np.array_equal(source480[dev_rows], training_container["source_val"]), "SOURCE cache and existing interface waveform bytes disagree")
    require(np.array_equal(prepared["source_ids"], train_ids) and np.array_equal(prepared["source_val_ids"], dev_ids), "008 parent order changed")
    require(np.array_equal(source480[train_rows, :384].astype(np.float64), prepared["source_carrier384"]), "008 internal carrier changed")
    require(np.array_equal(prepared["original_source_weights"], np.ones(len(train_rows))), "original count weights changed")
    ee, had = metadata["edep_ee_MeV"], metadata["edep_hadron_MeV"]
    expected64, nominal, half_rise = analytic_source480(ee, had)
    visible = ee+.15*had
    h = .15*had/np.maximum(visible, 1e-15)
    require(np.array_equal(visible, metadata["visible_energy_proxy_MeV"])
            and np.array_equal(h, metadata["visible_hadron_fraction"]), "stored visible deposition proxies changed")
    require(np.allclose(h[train_rows], training_container["h"], rtol=0, atol=1e-12)
            and np.allclose(h[dev_rows], training_container["hval"], rtol=0, atol=1e-12)
            and np.array_equal(training_container["h"], prepared["h"]), "h interface provenance mismatch")
    fixed_dev = np.zeros(len(ids), dtype=bool); fixed_dev[dev_rows] = True
    roles = np.where(train, "TRAIN", "INTERNAL_VAL")
    masks = dict(ALL_LEGAL=np.ones(len(ids), dtype=bool), TRAIN=train, INTERNAL_VAL=~train, EXISTING_FIXED_DEV=fixed_dev)
    representations, views, usable, measures = normalized_views(source480)
    expected_representations, expected_views, expected_usable, _ = normalized_views(expected64)
    observed_crop_error = float(np.abs(representations["clean_window256"][train_rows]-prepared["source"]).max())
    require(observed_crop_error <= 1e-10, "008 clean source crop differs from original input")
    rounding = source480.astype(np.float64)-expected64
    formula_equal = (expected64.astype(np.float32) == source480).all(1)
    rows = dict(source_id=ids.copy(), cache_row=np.arange(len(ids), dtype=np.int64), role=roles,
        existing_fixed_DEV=fixed_dev, original_count_weight=np.ones(len(ids)),
        physical_weight_known=np.zeros(len(ids), dtype=bool),
        edep_ee_MeV=ee.copy(), edep_hadron_MeV=had.copy(), visible_energy_proxy_MeV=visible,
        h_visible=h, source_PDG_audit=metadata["pdg"].copy(), analytic_half_rise_index512=half_rise,
        analytic_float32_waveform_equal=formula_equal, generator_nominal_Q512_Vns=nominal,
        float32_voltage_RMS_residual=np.sqrt(np.square(rounding).mean(1)),
        float32_voltage_absmax_residual=np.abs(rounding).max(1), **measures)
    rows["amplitude_384_V"] = measures["amplitude384_V"]
    rows["amplitude_480_V"] = measures["amplitude480_V"]
    for name, mask in usable.items():
        rows[name+"_normalization_defined"] = mask.copy()
    summary = dict(schema="n01/wp1-input-information-audit/v1", counts={name: int(mask.sum()) for name, mask in masks.items()},
        baseline="original100 prefix mean", canonical_charge="centered source[110:366].sum at software dt1ns",
        full_source480_rank_is_reported_separately_from_internal384_and_clean256=True,
        source_roles_preserved=True, remaining_internal_validation_is_not_new_blind=True,
        original_count_weights="uniform ones", calibrated_original_physical_weights="unknown",
        generator_provenance=dict(ee_decay_ns=[13, 35, 270], ee_weights=[.7, .15, .15],
            had_decay_ns=[13, 59, 460], had_weights=[.5, .1, .4], common_rise_ns=5, had_visible_coefficient=.15,
            nominal_start512=128, aligned_half_rise480=120, physical_calibration="unknown",
            microscopic_deposition_timing_available=False, nominal512_charge_equals_canonical_charge=False),
        metadata_definition="h=.15*edep_hadron/(edep_ee+.15*edep_hadron); SOURCE audit proxy, not REAL particle truth",
        metadata_formula_verified=True, generator_float32_exact_rows=int(formula_equal.sum()),
        generator_float32_relation_verified=bool(formula_equal.all()),
        rounding_voltage_residual=finite_summary(rows["float32_voltage_RMS_residual"]),
        prepared008_clean_crop_absmax_error_V=observed_crop_error, PCA={}, measurements={}, associations={}, templates={})
    require(np.array_equal(nominal.astype(np.float32), metadata["Qsource_Vns"]), "stored nominal full512 charge changed")
    summary["nominal512_minus_canonical_charge_Vns"] = finite_summary(nominal-measures["canonical_Q_Vns"])
    banks = {}
    for role, mask in masks.items():
        summary["measurements"][role] = {name: finite_summary(value[mask]) for name, value in measures.items()}
        summary["associations"][role] = {name: association(h[mask], value[mask]) for name, value in measures.items()
            if name in ("canonical_Q_Vns", "amplitude480_V", "amplitude384_V", "Q_clean_window256_Vns", "tail_clean_window256_Vns")}
        composition = np.where(had == 0, "pure_EE", np.where(ee == 0, "pure_had", "mixed"))
        classes = np.char.add(np.char.add(composition, "|half_rise="), half_rise.astype(str))
        keys, counts = np.unique(classes[mask], return_counts=True)
        summary["templates"][role] = dict(composition_and_integer_alignment_categories={str(key): int(count) for key, count in zip(keys, counts)},
            distinct_visible_h_values=int(len(np.unique(h[mask]))), mixed_shape_condition_is_continuous_h=True,
            analytic_categories_are_not_float_rounding_categories=True,
            peaknorm_numeric_patterns=numeric_pattern_counts(views["full480_positive_peak"][mask & usable["full480_positive_peak"]]))
        for name, value in views.items():
            chosen = mask & usable[name]
            if not chosen.any():
                summary["PCA"][role+"_"+name] = dict(defined=False, reason="no_defined_normalization", rows=0, retained_undefined_rows=int(mask.sum()))
                continue
            report, mean, axes = pca_information(value[chosen], settings["PCA_variance_thresholds"], settings["saved_PCA_components"])
            report["undefined_normalization_rows_retained_outside_descriptive_PCA"] = int((mask & ~usable[name]).sum())
            residual = value[chosen]-expected_views[name][chosen]
            report["analytic_to_float32_normalized_residual_RMS"] = finite_summary(np.sqrt(np.square(residual).mean(1)))
            analytic_centered = expected_views[name][chosen]-expected_views[name][chosen].mean(0)
            residual_centered = residual-residual.mean(0)
            denominator = max(int(chosen.sum())-1, 1)
            report["analytic_float64_normalized_total_variance"] = float(np.square(analytic_centered).sum()/denominator)
            report["float32_difference_centered_total_variance"] = float(np.square(residual_centered).sum()/denominator)
            report["analytic_rounding_twice_cross_covariance"] = float(2*(analytic_centered*residual_centered).sum()/denominator)
            report["rounding_variance_is_not_additive_without_the_reported_cross_term"] = True
            summary["PCA"][role+"_"+name] = report
            banks[role+"_"+name+"_mean"] = mean
            banks[role+"_"+name+"_components"] = axes
            if role == "TRAIN":
                rows[name+"_TRAIN_PCA_scores"] = (value-mean) @ axes
                summary["associations"][role][name+"_PC1_vs_h"] = association(h[chosen], rows[name+"_TRAIN_PCA_scores"][chosen, 0])
    pairs, pair_rows = same_charge_pairs(views["full480_positive_peak"], measures["canonical_Q_Vns"], roles,
        relative_q_max=settings["pair_relative_Q_max"], shape_rms_min=settings["pair_peaknorm_RMS_min"],
        chunk=settings["pair_chunk_rows"], max_output=settings["saved_pair_max"])
    for pair in pairs["pairs"]:
        pair["left_source_id"] = str(ids[pair["left_cache_row"]]); pair["right_source_id"] = str(ids[pair["right_cache_row"]])
    summary["same_Q_real_input_shape_feasibility"] = pairs
    rows.update(pair_rows)
    rows["PC_scores_384_Qnorm"] = rows["carrier384_canonical_Q_TRAIN_PCA_scores"]
    rows["PC_valid_384_Qnorm"] = usable["carrier384_canonical_Q"].copy()
    summary["neural_inputs_added"] = []; summary["response_shape_intervention_executed"] = False
    return summary, rows, banks


def run_information(config_path, expected_sha256, output):
    started = time.monotonic()
    config, settings, paths, checked = frozen_inputs(config_path, expected_sha256, "information_analysis")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = str(settings.get("CPU_threads", 12))
    import numpy as np
    decoded = []
    output = fresh_output(output)
    try:
        cache = load_members(paths["g4_cache"], ("clean_expectation_V", "event_ids", "internal_training", "edep_ee_MeV",
            "edep_hadron_MeV", "visible_energy_proxy_MeV", "visible_hadron_fraction", "pdg", "Qsource_Vns"), decoded)
        original = load_members(paths["training_container"], ("source", "source_ids", "h", "source_val", "source_val_ids", "hval"), decoded)
        prepared = load_members(paths["dataset008"], ("source", "source_carrier384", "source_ids", "source_val_ids", "h", "original_source_weights"), decoded)
        require(len(cache["event_ids"]) == settings["counts"]["all_legal_SOURCE"]
            and int(cache["internal_training"].sum()) == settings["counts"]["TRAIN"]
            and len(original["source_val_ids"]) == settings["counts"]["existing_fixed_DEV"], "frozen legal SOURCE counts changed")
        report, rows, banks = audit_source_information(cache.pop("clean_expectation_V"), cache, original, prepared, settings)
        npz_exclusive(output/"PARENT_INFORMATION.npz", rows)
        npz_exclusive(output/"PCA_BANK.npz", banks)
        json_exclusive(output/"INPUT_INFORMATION_AUDIT.json", report)
        unchanged(checked)
        inventory = {path.name: dict(bytes=path.stat().st_size, sha256=sha256(path)) for path in output.iterdir() if path.is_file()}
        require(sum(item["bytes"] for item in inventory.values()) <= settings["output_limit_bytes"], "WP1 information output budget exceeded")
        require(time.monotonic()-started <= settings["wall_limit_s"], "WP1 information wall budget exceeded")
        json_exclusive(output/"COMPLETE.json", dict(schema="n01/wp1-input-information-complete/v1", complete=True,
            artificial_only=bool(config.get("artificial_only", False)),
            elapsed_s=time.monotonic()-started, immutable_files_sha256=checked, inputs_unchanged=True,
            decoded_named_members=decoded, artifacts=inventory, source_parent_counts=report["counts"],
            descriptive_PCA_decompositions=len(report["PCA"]),
            INTERNAL_VAL_SOURCE_information_values_read=report["counts"]["INTERNAL_VAL"],
            existing_fixed_DEV_SOURCE_information_values_read=report["counts"]["EXISTING_FIXED_DEV"],
            DEV_values_used_for_training_or_model_selection=False, REAL_waveform_decodes=0,
            model_forwards=0, autograd_queries=0, checkpoint_loads=0, optimizer_created=0, optimizer_steps=0,
            scientific_training_updates=0, physical_information_or_response_acceptance_not_inferred=True))
        return report
    except BaseException as error:
        json_exclusive(output/"FAILED.json", dict(error_type=type(error).__name__, error=str(error),
            elapsed_s=time.monotonic()-started, decoded_named_members=decoded, complete=False, automatic_restart=False,
            model_forwards=0, autograd_queries=0, checkpoint_loads=0, optimizer_steps=0))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", nargs="+", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require((args.headers is None) != (args.config is None), "choose header inspection or the frozen WP1 scientific config")
    if args.headers is not None:
        json_exclusive(args.out, dict(schema="n01/wp1-source-header-inventory/v1",
            files=[npz_headers(path) for path in args.headers], scientific_array_decodes=0,
            checkpoint_loads=0, model_forwards=0, autograd_queries=0, optimizer_steps=0))
        print(json.dumps(dict(status="header_inventory_complete", scientific_array_decodes=0)))
    else:
        result = run_information(args.config, args.config_sha256, args.out)
        print(json.dumps(dict(status="SOURCE_information_complete", counts=result["counts"], model_forwards=0, autograd_queries=0)))


if __name__ == "__main__":
    main()
