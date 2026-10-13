"""Prepare the frozen-parent TRAIN256 voltage interface; never decode DEV pulses."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from n01.data import sha256, write_json
from n01.data008 import (INPUT_HASHES, fit_statistics008, load_training008,
                         self_test, window_spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--identities', default=str(ROOT / 'data/PARENT_IDENTITIES.npz'))
    parser.add_argument('--spec', help='JSON data-interface spec only; omit to use documented defaults')
    parser.add_argument('--out', default=str(ROOT / 'runs/window008_20261013/prepared'), help='Fresh output directory on E/D')
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    if os.name == 'nt' and out.drive.upper() not in ('E:', 'D:'):
        raise ValueError('All data and receipts must be on E/D')
    paths = dict(data=out / 'DATASET008.npz', statistics=out / 'FIT_DATA_STATISTICS.json',
                 receipt=out / 'PREPARATION_RECEIPT.json')
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Preserve original outputs; choose a fresh prefix')
    out.mkdir(parents=True, exist_ok=True)
    spec = window_spec(json.loads(Path(args.spec).read_text(encoding='utf-8-sig')) if args.spec else None)
    start = time.monotonic()
    tests = self_test()
    data = load_training008(args.container, args.identities, spec)
    statistics = fit_statistics008(data, spec)
    with paths['data'].open('xb') as stream:
        np.savez_compressed(stream, **data)
    # Reopen every prepared array and compare exactly. This file contains only
    # TRAIN waveforms and authorized DEV identity/group metadata.
    with np.load(paths['data'], allow_pickle=False) as saved:
        if set(saved.files) != set(data):
            raise RuntimeError('Prepared members changed')
        for name, value in data.items():
            np.testing.assert_array_equal(saved[name], value, err_msg=name)
    write_json(paths['statistics'], statistics)
    receipt = dict(status='TRAIN256_prepared_reopened_verified', data_spec=spec,
        elapsed_seconds=time.monotonic() - start, self_test=tests,
        container=str(Path(args.container).resolve()), identities=str(Path(args.identities).resolve()),
        input_sha256=INPUT_HASHES, output_data_sha256=sha256(paths['data']),
        output_statistics_sha256=sha256(paths['statistics']),
        source_rows=len(data['source']), real_rows=len(data['real']), source_DEV_identity_rows=len(data['source_val_ids']),
        real_DEV_identity_rows=len(data['real_val_original_index']), rows_removed=0,
        no_padding=True, no_peak_normalization=True, no_postcrop_ADC=True,
        common_ADC_scope='REAL_and_GEN_only; SOURCE_pristine_expected_voltage',
        source_ADC=False, real_already_digitized=True, source_and_real_carrier_samples=384,
        global_peak_and_search_edge_flags_retained=True,
        no_extra_noise_or_gain=True, original_voltage_unit_retained=True,
        DEV_waveforms_decoded=False, metadata_features_are_not_network_inputs=True,
        original_480_files_modified=False, scientific_training_updates=0, scientific_model_forwards=0,
        code_sha256={name: sha256(ROOT / name) for name in
                     ('src/n01/data008.py', 'src/n01/observation008.py', 'scripts/prepare_window008.py', 'docs/WINDOW008_DATA_INTERFACE.md')},
        output_paths={name: str(path) for name, path in paths.items()})
    write_json(paths['receipt'], receipt)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
