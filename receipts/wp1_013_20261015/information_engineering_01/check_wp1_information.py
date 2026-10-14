"""Only small manufactured arrays. Scientific NN/AD/CP/array reads are zero."""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TEMP"] = os.environ["TMP"] = "E:/GAN/runtime_tmp"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"scripts"))
import wp1_input_information as info
import wp1_boundary_saved as boundary
import numpy as np


def binding(path):
    return dict(path=str(path.resolve()), sha256=info.sha256(path), bytes=path.stat().st_size)


def settings():
    return dict(PCA_variance_thresholds=[.95, .99, .999], saved_PCA_components=16,
        pair_relative_Q_max=.01, pair_peaknorm_RMS_min=.001, pair_chunk_rows=8,
        saved_pair_max=8, CPU_threads=2, wall_limit_s=600, output_limit_bytes=64*1024*1024)


def source_fixture():
    n = 32
    visible = np.tile(np.linspace(.2, 2, 8), 4)
    h = np.repeat([0, 1, .3, .7], 8)
    ee, had = visible*(1-h), visible*h/.15
    expected64, nominal, _ = info.analytic_source480(ee, had)
    wave = expected64.astype(np.float32)
    original_source = ROOT.parent/"sim2real_two_papers_20261008_0305/work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/run.py"
    tree = ast.parse(original_source.read_text("utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "clean_expectation")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(original_source)+":pure-clean-expectation", "exec"), namespace)
    oracle, oracle_nominal = namespace["clean_expectation"](ee, had)
    np.testing.assert_array_equal(wave, oracle)
    np.testing.assert_array_equal(nominal.astype(np.float32), oracle_nominal)
    ids = np.asarray(["ARTIFICIAL_SOURCE|"+str(index) for index in range(n)])
    train = np.arange(n) < 24
    dev = np.arange(24, 28)
    visible = ee+.15*had
    h = .15*had/np.maximum(visible, 1e-15)
    cache = dict(clean_expectation_V=wave, event_ids=ids, internal_training=train, edep_ee_MeV=ee,
        edep_hadron_MeV=had, visible_energy_proxy_MeV=visible, visible_hadron_fraction=h,
        pdg=np.where(h > .5, 2112, 22).astype(np.int32), Qsource_Vns=nominal.astype(np.float32))
    original = dict(source=wave[train], source_val=wave[dev], source_ids=ids[train], source_val_ids=ids[dev], h=h[train], hval=h[dev])
    representation, _, _, _ = info.normalized_views(wave)
    prepared = dict(source=representation["clean_window256"][train], source_carrier384=wave[train, :384].astype(np.float64),
        source_ids=ids[train], source_val_ids=ids[dev], h=h[train], original_source_weights=np.ones(24))
    return cache, original, prepared


def boundary_fixture():
    n = 4
    rows = dict(source_parent_ids=np.asarray(["ARTIFICIAL_PARENT_A", "ARTIFICIAL_PARENT_B"]),
        source_parent_rows=np.asarray([17, 29]), source_parent_row=np.asarray([0, 0, 1, 1]))
    native_paths=tuple(info.json_read(ROOT/"mechanism012_policy.json")["probe"]["paths"])
    assert boundary.PROBE_PATHS==native_paths
    v = np.zeros((n, 384)); v[:, 127] = .8; v[:, 300] = -.8
    rows["probe_carrier_direction"] = v
    actual_q = np.asarray([.02, .03, .04, .05])
    actual_tail = actual_q*.2
    for h in boundary.HYPOTHESES:
        wave = np.zeros((n, 256)); wave[:, 16] = .01; wave[:, 17] = .01
        for key, value in dict(actual_peak_index=np.full(n, 126), actual_crop_start=np.full(n, 110), actual_crop_stop=np.full(n, 366),
                actual_Q=actual_q, actual_tail=actual_tail, actual_PSD=actual_tail/actual_q, actual_window_V=wave,
                first_runnerup_margin_V=np.zeros(n), peak_tie_count=np.full(n, 2), peak_tied=np.ones(n, bool),
                actual_flag_ADC_halfgrid_tied=np.zeros(n, bool), actual_flag_ADC_halfgrid_tie_cells=np.zeros(n, int),
                actual_flag_ADC_halfgrid_min_distance=np.full(n, .2),
                signed_influence_wave=np.asarray([.1, .2, .3, .4]), signed_influence_QPSD=np.asarray([.01, .02, .03, .04])).items():
            rows[h+"_"+key] = value
    point = dict(weighted_terms=dict(wave=.2, QPSD=.1, RMS=.0002),
        comparisons={h: dict(actual_wave=.2, actual_QPSD=.05) for h in boundary.HYPOTHESES})
    signed = []
    for step in boundary.STEPS:
        for sign in boundary.SIGNS:
            delta = sign*step/boundary.ADC_LEVELS
            prefix = boundary.saved_prefix(step, sign)
            paths = {}
            for path_index, path in enumerate(native_paths):
                hypotheses = {}
                for h_index, h in enumerate(boundary.HYPOTHESES):
                    changed = np.zeros(n, bool)
                    if path_index == 2 and ((step == .1 and sign == 1) or step > .1):
                        changed[(h_index+int(step)) % n] = True
                    peak = np.full(n, 126); peak[changed] = 127
                    stem = prefix+"_"+path+"_"+h
                    q = actual_q+delta*.01
                    tail = actual_tail+delta*.002
                    q[changed] += .001; tail[changed] -= .0001
                    for key, value in dict(peak_index=peak, Q=q, tail=tail, PSD=tail/q, peak_changed=changed,
                        flag_ADC_halfgrid_tied=np.zeros(n, bool), flag_ADC_halfgrid_tie_cells=np.zeros(n, int),
                        flag_ADC_halfgrid_min_distance=np.full(n, .25), flag_shape_peak_tied=~changed).items():
                        rows[stem+"_"+key] = value
                    wave_value = .2-delta*.2+(float(changed.sum())*.00001 if path_index == 2 else 0)
                    hypotheses[h] = dict(wave=wave_value, QPSD_fixed=.05, peak_changed_draws=int(changed.sum()))
                paths[path] = dict(hypotheses=hypotheses, mean_h_proxy=float(np.mean([item["wave"]+2*item["QPSD_fixed"] for item in hypotheses.values()])))
            signed.append(dict(step_LSB=step, sign=sign, delta_V=delta, defined=True, paths=paths))
    central = []
    for step in boundary.STEPS:
        minus = next(item for item in signed if item["step_LSB"] == step and item["sign"] == -1)
        plus = next(item for item in signed if item["step_LSB"] == step and item["sign"] == 1)
        central.append(dict(step_LSB=step, paths={path: dict(central_difference_loss_per_V=
            (plus["paths"][path]["mean_h_proxy"]-minus["paths"][path]["mean_h_proxy"])/(2*step/boundary.ADC_LEVELS)) for path in native_paths}))
    return rows, dict(direction_defined=True, d_AD_loss_per_V=-.2, signed_probes=signed, central_differences=central), point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = info.fresh_output(args.out)
    began = time.monotonic()
    results = []
    def passed(name, **data):
        results.append(dict(name=name, passed=True, **data))
    cache, original, prepared = source_fixture()
    passed("original_bound_pure_generator_AST_exact_float32_parity_artificial_inputs_only")
    q = info.canonical_charge_source_np(cache["clean_expectation_V"])
    shifted = cache["clean_expectation_V"].astype(np.float64)+.017
    np.testing.assert_allclose(info.canonical_charge_source_np(shifted), q, rtol=0, atol=2e-13)
    passed("canonical_source_Q_preserves_nonzero_prefix_baseline_removal")
    known = np.asarray([[1,0,0,0],[-1,0,0,0],[0,.2,0,0],[0,-.2,0,0],[0,0,.05,0],[0,0,-.05,0]], float)
    pca, _, _ = info.pca_information(known)
    assert pca["explained_variance_dimensions"] == {"0.95":1,"0.99":2,"0.999":3}
    constant, _, _ = info.pca_information(np.ones((4,8)))
    assert set(constant["explained_variance_dimensions"].values()) == {0}
    passed("centered_PCA_known_95_99_999_dimensions_and_exact_constant_zero")
    tail = np.zeros((4, 480)); tail[:,126] = 1; tail[:,400] = [-.2,-.1,.1,.2]
    _, views, _, _ = info.normalized_views(tail)
    full, _, _ = info.pca_information(views["full480_positive_peak"])
    short, _, _ = info.pca_information(views["carrier384_positive_peak"])
    assert full["explained_variance_dimensions"]["0.95"] == 1 and short["explained_variance_dimensions"]["0.95"] == 0
    passed("complete480_has_information_absent_from384_no_rank_overclaim")
    shapes = np.asarray([[1,0,0],[0,1,0],[1,0,0],[0,0,1]], float)
    pairs, pair_rows = info.same_charge_pairs(shapes, [1,1.002,2,2.003], ["TRAIN","TRAIN","INTERNAL_VAL","INTERNAL_VAL"], chunk=2)
    np.testing.assert_array_equal(pair_rows["strongest_same_Q_partner_cache_row"], [1,0,3,2])
    np.testing.assert_allclose(pair_rows["strongest_same_Q_shape_RMS"], np.sqrt(2/3))
    assert pairs["eligible_ordered_pairs"] == 4 and pairs["feasible"]
    passed("streamed_all_sameQ_real_pair_search_matches_literal_distance_and_role_mask")
    files = out/"artificial_inputs"; files.mkdir()
    for name, value in (("cache",cache),("original",original),("prepared",prepared)):
        info.npz_exclusive(files/(name+".npz"), value)
    cfg = dict(artificial_only=True, scientific_training_updates=0,
        source_files={name: binding(ROOT/"scripts"/name) for name in ("wp1_input_information.py","wp1_boundary_saved.py")},
        information_analysis=dict(settings(), inputs={key: binding(files/(name+".npz")) for key, name in
            (("g4_cache","cache"),("training_container","original"),("dataset008","prepared"))},
            counts=dict(all_legal_SOURCE=32, TRAIN=24, existing_fixed_DEV=4)))
    rows, probe, point = boundary_fixture()
    for name, value in (("rows", rows),):
        info.npz_exclusive(files/(name+".npz"), value)
    for name, value in (("probe",probe),("point",point)):
        info.json_exclusive(files/(name+".json"), value)
    complete = dict(engineering_checks_passed=True, inputs_unchanged=True, head_tensors_unchanged=True,
        artifacts={key:binding(files/name) for key,name in (("ROW_MECHANISMS.npz","rows.npz"),("DIRECTIONAL_PROBE.json","probe.json"),("POINT_REPLAY.json","point.json"))})
    info.json_exclusive(files/"complete.json",complete)
    cfg["boundary_analysis"] = dict(steps_LSB=list(boundary.STEPS),signs=list(boundary.SIGNS),paths=list(boundary.PROBE_PATHS),
        draws_per_parent=2,source_draws=4,source_parents=2,wall_limit_s=600,output_limit_bytes=64*1024*1024,
        inputs={key:binding(files/name) for key,name in (("row_mechanisms","rows.npz"),("directional_probe","probe.json"),
                ("point_replay","point.json"),("complete","complete.json"))})
    cfg_path=out/"ARTIFICIAL_CONFIG.json"; info.json_exclusive(cfg_path,cfg)
    digest=info.sha256(cfg_path)
    report=info.run_information(cfg_path,digest,out/"information_producer")
    assert report["counts"] == dict(ALL_LEGAL=32,TRAIN=24,INTERNAL_VAL=8,EXISTING_FIXED_DEV=4)
    assert report["generator_float32_exact_rows"] == 32 and report["source_roles_preserved"]
    with np.load(out/"information_producer/PARENT_INFORMATION.npz",allow_pickle=False) as archive:
        assert archive["PC_scores_384_Qnorm"].shape == (32,16)
        assert not archive["physical_weight_known"].any()
    header=info.npz_headers(out/"information_producer/PARENT_INFORMATION.npz")
    assert header["array_payload_decodes"] == 0 and header["members"]["source_id"]["shape"] == [32]
    passed("actual_information_hashgate_writer_roles_aliases_and_header_inventory")
    result=boundary.run_saved_boundary(cfg_path,digest,out/"boundary_producer")
    events=info.json_read(out/"boundary_producer/CHANGED_EVENTS.json")["rows"]
    assert len(result["signed_probes"]) == 6 and len(result["central_differences"]) == 3
    assert any(event["step_LSB"]==.1 for event in events) and any(event["step_LSB"]==10 for event in events)
    assert all(event["probe_search_runnerup_margin_V"] is None and event["reference_peak_search_tie_count"]==2 for event in events)
    assert all("not probe row loss" in event["influence_scope"] for event in events)
    assert all(event["source_original_TRAIN_row"]==[17,29][event["source_parent_ordinal"]] for event in events)
    assert result["interpretation"]["quantizer_half_grid_ties_are_distinct_from_peak_ADC_ties"]
    passed("actual_saved_boundary_writer_all_steps_exact_crop_difference_unknown_margin_and_influence_scope",changed_events=len(events))
    zero=dict(direction_defined=False,reason="all_zero_random_shape",signed_probes=[dict(step_LSB=step,sign=sign,defined=False,paths=None) for step in boundary.STEPS for sign in boundary.SIGNS])
    zero_result,_,events=boundary.boundary_saved_analysis(rows,zero,point,cfg["boundary_analysis"])
    assert not zero_result["direction_defined"] and len(zero_result["signed_probes"])==6 and not events
    passed("zero_direction_keeps_six_undefined_entries_without_evaluations")
    try:
        info.frozen_inputs(cfg_path,"0"*64,"information_analysis")
    except ValueError:
        pass
    else:
        raise AssertionError("configuration mismatch did not reject")
    try:
        info.fresh_output(out/"boundary_producer")
    except ValueError:
        pass
    else:
        raise AssertionError("existing output was accepted")
    passed("hash_mismatch_and_existing_output_refused_without_decode_or_overwrite")
    receipt=dict(schema="n01/wp1-information-engineering/v1",artificial_only=True,native_exit_code=0,
        passed=len(results),total=len(results),results=results,elapsed_s=time.monotonic()-began,
        source_sha256={name:info.sha256(ROOT/"scripts"/name) for name in ("wp1_input_information.py","wp1_boundary_saved.py")},
        original_SOURCE_or_REAL_array_decodes=0,scientific_DEV_array_decodes=0,checkpoint_loads=0,model_forwards=0,
        autograd_queries=0,optimizer_created=0,optimizer_steps=0, scientific_training_updates=0,
        original_generator_only_pure_AST_function_with_artificial_depositions=True)
    info.json_exclusive(out/"ENGINEERING_RECEIPT.json",receipt)
    print(json.dumps(dict(native_exit_code=0,passed=len(results),total=len(results),scientific_array_decodes=0)))


if __name__=="__main__":
    main()
