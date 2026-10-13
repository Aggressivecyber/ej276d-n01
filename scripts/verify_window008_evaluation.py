"""Compare every saved array and numerical receipt of the fixed Window008 stream.

This module never loads a model, draws a random value, or decodes a data input.
ZIP-container timestamps are irrelevant; each artifact's own file binding is
checked, while replay equality uses array dtype, shape, values, and raw bytes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np

from window008_evaluation_common import (
    ROOT, artifact_integrity, completed_job, json_sha, read, require, sha, write_new,
)

NPZ_NAMES = {'evaluation_projection_bank.npz'} | {
    split + '_' + suffix + '.npz'
    for split in ('TRAIN', 'DEV')
    for suffix in ('observables', 'curves', 'sample_waveforms', 'stream_states',
                   'parent_bootstrap', 'matched_waveforms')
}
JSON_IGNORED = {
    'FROZEN_INPUTS.json': {'UTC'}, 'TRAIN_generation.json': {'UTC'},
    'DEV_generation.json': {'UTC'}, 'SUMMARY.json': {'UTC', 'artifact_sha256'},
    'TRAIN_PARENT_BOOTSTRAP.json': set(), 'DEV_PARENT_BOOTSTRAP.json': set(),
    'MATCHED_SELECTIONS.json': set(), 'DISTRIBUTION_DISTANCES.json': set(),
}


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def semantic_json(path, ignored):
    value = read(path)
    require(isinstance(value, dict), 'Expected JSON object: ' + str(path))
    return {key: item for key, item in value.items() if key not in ignored}


def numeric_count(value):
    if isinstance(value, dict):
        return sum(numeric_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(numeric_count(item) for item in value)
    return int(type(value) in (int, float))


def stream_contract(destination):
    """Read generated identities and require the registered parent-major stream."""
    receipt = {}
    for split in ('TRAIN', 'DEV'):
        generation = read(destination / (split + '_generation.json'))
        require(generation['split'] == split and generation['draws_per_parent'] == 4
                and generation['rows_removed'] == generation['extra_sample_forwards'] == 0,
                'Generation changed the registered stream: ' + split)
        n, rows = generation['parents'], generation['rows']
        require(type(n) is int and n > 0 and rows == n * 4
                and generation['response_forward_rows_per_arm'] == rows
                and generation['noise005_forward_rows'] == rows, 'Wrong stream row accounting: ' + split)
        with np.load(destination / (split + '_observables.npz'), allow_pickle=False) as archive:
            ids = archive['source_parent_ids']
            parents, draws = archive['source_parent_row'], archive['draw']
            require(ids.shape == (n,) and len(np.unique(ids)) == n, 'Source parent identity mismatch: ' + split)
            require(np.array_equal(parents, np.repeat(np.arange(n, dtype=np.int64), 4))
                    and np.array_equal(draws, np.tile(np.arange(4), n)), 'Parent/draw order changed: ' + split)
            identity_hash = array_sha(ids)
            parent_hash, draw_hash = array_sha(parents), array_sha(draws)
        expected_ranges = [(start, min(start + 128, rows)) for start in range(0, rows, 128)]
        chunks = generation['chunks']
        require([(item['start'], item['stop']) for item in chunks] == expected_ranges,
                'Chunk boundaries changed: ' + split)
        for item in chunks:
            require(item['source_parent_rows_sha256'] == array_sha(parents[item['start']:item['stop']]),
                    'Chunk parent binding differs: ' + split)
            require(set(item['outputs']) == {'deterministic', 'full', 'simple'}, 'Chunk arm inventory differs')
            for arm in item['outputs'].values():
                require(set(arm) == {'fast', 'slow', 'common_level'}, 'Chunk hypothesis inventory differs')
                for hypothesis in arm.values():
                    require(isinstance(hypothesis['waveform256_sha256'], str)
                            and bool(hypothesis['observables_sha256']), 'Chunk output binding missing')
            require(bool(item['response_sha256']) and bool(item['noise_latents_sha256'])
                    and isinstance(item['extension_epsilon_sha256'], str)
                    and isinstance(item['baseline100_ADC_sha256'], str), 'Chunk latent/baseline binding missing')
        receipt[split] = dict(parents=n, draws_per_parent=4, rows=rows, chunk_rows=128,
            chunks=len(chunks), source_parent_ids_sha256=identity_hash,
            source_parent_row_sha256=parent_hash, draw_sha256=draw_hash,
            all_chunk_records_sha256=json_sha(chunks))
    return receipt


def compare_artifacts(reference, candidate):
    """Return a complete comparison report; status never hides a mismatch."""
    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    require(reference != candidate, 'Replay must live in a separate fresh directory')
    report = dict(schema='n01/window008-exact-replay/v1', UTC=datetime.now(timezone.utc).isoformat(),
        status='exact_replay_failed', reference=str(reference), candidate=str(candidate),
        optimizer_updates=0, additional_unique_draws=0, replay_is_additional_independent_sample=False,
        comparison='exact_dtype_shape_values_and_member_sha256',
        ignored_fields={key: sorted(value) for key, value in JSON_IGNORED.items() if value},
        npz_files={}, json_files={}, stream_contract={}, mismatches=[])
    for name, destination in (('reference', reference), ('candidate', candidate)):
        try:
            artifact_integrity(destination)
            require({item.name for item in destination.iterdir()} == NPZ_NAMES | set(JSON_IGNORED),
                    'Unexpected or missing evaluation artifact')
            require(all(item.is_file() and not item.is_symlink() for item in destination.iterdir()),
                    'Evaluation artifacts must be regular files')
            report['stream_contract'][name] = stream_contract(destination)
        except Exception as error:
            report['mismatches'].append(dict(kind='artifact_integrity_or_stream', side=name, error=repr(error)))
    # The archive is opened one file at a time, so no complete output copy is held.
    for filename in sorted(NPZ_NAMES):
        details = dict(members={}, member_inventory_equal=False, exact=False)
        report['npz_files'][filename] = details
        try:
            with np.load(reference / filename, allow_pickle=False) as first, \
                    np.load(candidate / filename, allow_pickle=False) as second:
                details['member_inventory_equal'] = sorted(first.files) == sorted(second.files) and bool(first.files)
                require(details['member_inventory_equal'], 'NPZ member inventory differs: ' + filename)
                for name in sorted(first.files):
                    a, b = first[name], second[name]
                    require(a.dtype.kind != 'O' and b.dtype.kind != 'O', 'Object arrays are forbidden')
                    dtype_equal, shape_equal = a.dtype == b.dtype, a.shape == b.shape
                    value_equal = False
                    if dtype_equal and shape_equal:
                        value_equal = bool(np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc'
                                           else np.array_equal(a, b))
                    hash_a, hash_b = array_sha(a), array_sha(b)
                    exact = bool(dtype_equal and shape_equal and value_equal and hash_a == hash_b)
                    details['members'][name] = dict(reference_dtype=str(a.dtype), candidate_dtype=str(b.dtype),
                        reference_shape=list(a.shape), candidate_shape=list(b.shape),
                        reference_array_sha256=hash_a, candidate_array_sha256=hash_b,
                        dtype_equal=bool(dtype_equal), shape_equal=bool(shape_equal),
                        values_equal=value_equal, raw_array_hash_equal=hash_a == hash_b, exact=exact)
                    if not exact:
                        report['mismatches'].append(dict(kind='NPZ_member', file=filename, member=name))
                details['exact'] = all(item['exact'] for item in details['members'].values())
        except Exception as error:
            report['mismatches'].append(dict(kind='NPZ_read', file=filename, error=repr(error)))
    for filename, ignored in JSON_IGNORED.items():
        try:
            a = semantic_json(reference / filename, ignored)
            b = semantic_json(candidate / filename, ignored)
            digest_a, digest_b = json_sha(a), json_sha(b)
            exact = digest_a == digest_b
            report['json_files'][filename] = dict(reference_semantic_sha256=digest_a,
                candidate_semantic_sha256=digest_b, numeric_values_checked=numeric_count(a), exact=exact)
            if not exact:
                report['mismatches'].append(dict(kind='JSON_semantics', file=filename))
        except Exception as error:
            report['mismatches'].append(dict(kind='JSON_read', file=filename, error=repr(error)))
    report['NPZ_files_checked'] = len(report['npz_files'])
    report['NPZ_members_checked'] = sum(len(item['members']) for item in report['npz_files'].values())
    report['JSON_files_checked'] = len(report['json_files'])
    report['generation_chunk_records_checked'] = sum(
        item['chunks'] for item in report['stream_contract'].get('reference', {}).values())
    report['reference_summary_sha256'] = sha(reference / 'SUMMARY.json') if (reference / 'SUMMARY.json').is_file() else None
    report['candidate_summary_sha256'] = sha(candidate / 'SUMMARY.json') if (candidate / 'SUMMARY.json').is_file() else None
    if not report['mismatches']:
        report['status'] = 'exact_replay_verified'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New report path; never replaces the original job report')
    args = parser.parse_args()
    from window008_evaluation_common import allow_output
    target = allow_output(args.output)
    require(not target.exists() and target.parent.is_dir(), 'Verification report must be a new file in an existing E/D or remote directory')
    first, second = completed_job('primary', ROOT), completed_job('replay01', ROOT)
    report = compare_artifacts(first['paths']['artifacts'], second['paths']['artifacts'])
    write_new(target, report)
    require(report['status'] == 'exact_replay_verified', 'Exact replay verification failed; inspect ' + str(target))
    print(str(target))


if __name__ == '__main__':
    main()
