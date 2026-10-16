"""One saved artificial-array equality check; no NN, CP or scientific input."""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]

import numpy as np
import torch
from n01.data import sha256, write_json
from n01.losses008 import fit_loss_statistics
from n01.losses008_v2 import N01Loss008, empirical_wasserstein_1d
from n01.losses014 import fit_integral_statistics014
from run_response032 import NativeMetricLoss032, column_W1, hard_metrics

torch.set_num_threads(4)
torch.set_grad_enabled(False)
out = Path(__file__).resolve().parent
ast.parse((ROOT/'scripts/run_response032.py').read_text(encoding='utf8'))
rng = np.random.default_rng(2026101732)
t = np.arange(256)
pulse = np.where(t < 16, np.exp(-.5*((t-16)/3.)**2), np.exp(-(t-16)/40.))
arrays = dict(x=rng.normal(size=(7, 37)), y=rng.normal(size=(11, 37)),
    fake=np.linspace(.03, .11, 7)[:, None]*pulse+rng.normal(0, .0002, (7, 256)),
    real=np.linspace(.035, .105, 11)[:, None]*pulse+rng.normal(0, .0002, (11, 256)),
    baseline=rng.normal(0, .001, (11, 100)), fake_baseline=rng.normal(0, .001, (7, 100)))
fixture = out/'W1_ENGINEERING_INPUTS.npz'
np.savez_compressed(fixture, **arrays)
with np.load(fixture, allow_pickle=False) as saved:
    assert all(np.array_equal(value, saved[key]) for key, value in arrays.items())
    tensors = {key: torch.from_numpy(saved[key].copy()) for key in saved.files}
original = empirical_wasserstein_1d(tensors['x'], tensors['y'])
chunked = column_W1(tensors['x'], tensors['y'], 16)
stats = fit_loss_statistics(tensors['real'], tensors['fake'], tensors['baseline'])
native = N01Loss008(stats, seed=2026101426, evaluation_seed=2026101430)
bounded = NativeMetricLoss032(stats, fit_integral_statistics014(tensors['real']),
    seed=2026101426, evaluation_seed=2026101430, integral_seed=2026101732300)
context = native.create_evaluation_context()
original_parts = native.wave_components(tensors['fake'], tensors['real'], None, context)
chunked_parts = bounded.wave_components(tensors['fake'], tensors['real'], None, context)
errors = {name: abs(float(value)-float(chunked_parts[name])) for name, value in original_parts.items()}
report = hard_metrics(arrays['fake'], arrays['fake_baseline'], arrays['real'], arrays['baseline'], bounded, context)
assert abs(float(original-chunked)) <= 1e-12 and max(errors.values()) <= 1e-12
assert 'fake_bin_per_row' not in report['output_Q_conditions'] and 'real_bin_per_row' not in report['output_Q_conditions']
assert all(value['denominator'] == 11 for value in report['real_flags'].values())
receipt = dict(schema='n01/response032-native-W1-engineering/v1', status='passed',
    fixture_sha256=sha256(fixture), runner_sha256=sha256(ROOT/'scripts/run_response032.py'),
    scalar_matrix_shapes=[[7, 37], [11, 37]], column_chunk=16, final_chunk_width=5,
    original_W1=float(original), chunked_W1=float(chunked), scalar_abs_error=abs(float(original-chunked)),
    original_wave_components={name: float(value) for name, value in original_parts.items()},
    chunked_wave_components={name: float(value) for name, value in chunked_parts.items()},
    wave_component_abs_errors=errors, hard_metric_report_smoke='passed',
    JSON_per_row_Q_bins_removed=True, REAL_flags_with_denominators=True,
    artificial_fixture_saved_and_reopened=True, original_031_source_sha256=sha256(ROOT/'scripts/run_response031.py'),
    counts=dict(NN_forwards=0, AD_queries=0, checkpoint_loads=0, raw_scientific_arrays_opened=0,
                optimizer_updates=0, artificial_array_archive_reopens=1),
    scientific_status='not_response_fit_evidence')
write_json(out/'W1_ENGINEERING.json', receipt)
print(json.dumps(receipt, indent=2))
