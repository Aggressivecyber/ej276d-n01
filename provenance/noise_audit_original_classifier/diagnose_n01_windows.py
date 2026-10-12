"""Predeclared frozen-endpoint C2ST views with matched REAL-REAL controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from n01.data import sha256, write_json
from n01.evaluate import _split_groups, _weighted_auc, _auc_bootstrap, _clean

SEED = 2026101203
FIT_PER_CLASS = 128
TEST_PER_CLASS = 96
UPDATES = 200
VIEWS = ('full', 'baseline_raw', 'baseline_demean', 'baseline_detrend', 'pulse_demean', 'pulse_shape')


def view_array(wave: np.ndarray, name: str) -> np.ndarray:
    y = np.asarray(wave, dtype=np.float64)
    baseline = y[:, :100]
    mean = baseline.mean(1, keepdims=True)
    if name == 'full':
        result = y
    elif name == 'baseline_raw':
        result = baseline
    elif name == 'baseline_demean':
        result = baseline - mean
    elif name == 'baseline_detrend':
        t = np.arange(100, dtype=np.float64) - 49.5
        b = baseline - mean
        result = b - (b @ t / (t @ t))[:, None] * t
    elif name in ('pulse_demean', 'pulse_shape'):
        result = y[:, 100:] - mean
        if name == 'pulse_shape':
            result = result / np.maximum(result.max(1, keepdims=True), 1 / 16383)
    else:
        raise ValueError(name)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite view; no rows removed')
    return result.astype(np.float32)


def choose_rows(rows: np.ndarray, count: int, rng, groups=None) -> np.ndarray:
    rows = np.asarray(rows)
    if groups is not None:
        # Uniform parent selection, followed by one of its saved four draws.
        unique = np.unique(groups[rows])
        if len(unique) < count:
            raise ValueError('Too few distinct parents')
        selected = rng.choice(unique, count, replace=False)
        return np.asarray([rng.choice(rows[groups[rows] == g]) for g in selected])
    if len(rows) < count:
        raise ValueError('Insufficient prespecified control support')
    return rng.choice(rows, count, replace=False)


def build_splits(fake_ids, real_groups) -> dict:
    ft, fe = _split_groups(fake_ids, 20261113)
    rt, re = _split_groups(real_groups, 20261114)
    rng = np.random.default_rng(SEED)
    mtf = choose_rows(ft, FIT_PER_CLASS, rng, fake_ids)
    mef = choose_rows(fe, TEST_PER_CLASS, rng, fake_ids)
    mtr = choose_rows(rt, FIT_PER_CLASS, rng)
    mer = choose_rows(re, TEST_PER_CLASS, rng)
    rfit = rng.permutation(np.unique(real_groups[rt]))
    rtest = rng.permutation(np.unique(real_groups[re]))
    f0, f1 = np.array_split(rfit, 2)
    e0, e1 = np.array_split(rtest, 2)
    rr = {}
    for role, groups, count in [('fit0', f0, FIT_PER_CLASS), ('fit1', f1, FIT_PER_CLASS),
                                 ('test0', e0, TEST_PER_CLASS), ('test1', e1, TEST_PER_CLASS)]:
        rr[role] = choose_rows(np.flatnonzero(np.isin(real_groups, groups)), count, rng)
    cases = {
        'matched_SOURCE_REAL': dict(domain0='SOURCE', fit0=mtf, test0=mef, fit1=mtr, test1=mer),
        'matched_REAL_REAL': dict(domain0='REAL', **rr),
        'all_SOURCE_REAL': dict(domain0='SOURCE', fit0=ft, test0=fe, fit1=rt, test1=re),
    }
    for name, case in cases.items():
        g0 = fake_ids if case['domain0'] == 'SOURCE' else real_groups
        if np.intersect1d(g0[case['fit0']], g0[case['test0']]).size:
            raise ValueError('Domain0 group leakage')
        if np.intersect1d(real_groups[case['fit1']], real_groups[case['test1']]).size:
            raise ValueError('Domain1 group leakage')
        if case['domain0'] == 'REAL':
            partitions = [set(real_groups[case[key]].tolist()) for key in ('fit0', 'test0', 'fit1', 'test1')]
            if any(a & b for i, a in enumerate(partitions) for b in partitions[i + 1:]):
                raise ValueError('REAL-REAL four-way group leakage')
    return cases


class WindowClassifier(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(1, 8, 9, stride=4, padding=4), nn.SiLU(),
                                      nn.Conv1d(8, 16, 9, stride=4, padding=4), nn.SiLU())
        n = (int(length) + 3) // 4
        n = (n + 3) // 4
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * n, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.features(x[:, None]))[:, 0]


def fit_classifier(x0, x1, g0, g1, case, view, device, output):
    a, b, c, d = [case[k] for k in ('fit0', 'fit1', 'test0', 'test1')]
    train_x = np.concatenate((x0[a], x1[b]))
    test_x = np.concatenate((x0[c], x1[d]))
    train_y = np.r_[np.zeros(len(a)), np.ones(len(b))].astype(np.float32)
    test_y = np.r_[np.zeros(len(c)), np.ones(len(d))].astype(np.float32)
    center = train_x.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(train_x.std(0, ddof=0, dtype=np.float64), 1e-5 if view == 'pulse_shape' else 1 / 16383).astype(np.float32)
    train_x = torch.as_tensor((train_x - center) / scale, device=device)
    test_x = torch.as_tensor((test_x - center) / scale, device=device)
    target = torch.as_tensor(train_y, device=device)
    cw = np.where(train_y == 1, len(train_y) / (2 * len(b)), len(train_y) / (2 * len(a)))
    weights = torch.as_tensor(cw, dtype=torch.float32, device=device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 1)
        model = WindowClassifier(train_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED + 2)
    losses = []
    model.train()
    for step in range(UPDATES):
        index = rng.choice(len(train_x), min(128, len(train_x)), replace=False)
        loss = (F.binary_cross_entropy_with_logits(model(train_x[index]), target[index], reduction='none') * weights[index]).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite classifier loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        scores = torch.cat([model(test_x[s:s + 128]).cpu() for s in range(0, len(test_x), 128)]).numpy()
    uncertainty = _auc_bootstrap(test_y, scores, g0[c], g1[d], np.ones(len(test_y)), SEED + 3)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model={k: v.cpu() for k, v in model.state_dict().items()}, center=center, scale=scale,
                    architecture='Conv1d8_16_flatten_MLP32', input_length=train_x.shape[1], seed=SEED + 1,
                    updates=UPDATES, view=view), output / 'classifier.pt')
    np.savez_compressed(output / 'predictions_and_splits.npz', labels=test_y, logits=scores,
                        test_groups0=g0[c], test_groups1=g1[d], fit_groups0=g0[a], fit_groups1=g1[b],
                        fit_rows0=a, fit_rows1=b, test_rows0=c, test_rows1=d, fit_loss=np.asarray(losses))
    report = dict(AUC=_weighted_auc(test_y, scores), uncertainty=uncertainty,
                  n_fit=[len(a), len(b)], n_test=[len(c), len(d)],
                  n_fit_groups=[len(np.unique(g0[a])), len(np.unique(g1[b]))],
                  n_test_groups=[len(np.unique(g0[c])), len(np.unique(g1[d]))],
                  fit_group_overlap_test=[0, 0], updates=UPDATES, seed=SEED + 1,
                  classifier_parameters=sum(p.numel() for p in model.parameters()),
                  final_fit_BCE=losses[-1], response_model_updates=0,
                  normalization_fit_scope='classifier_fit_rows_only',
                  label_1='REAL' if case['domain0'] == 'SOURCE' else 'random_REAL_domain_1',
                  orientation_flipped=False, no_heldout_selection=True)
    write_json(output / 'result.json', _clean(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', type=Path, default=ROOT / 'runs/run_001_20261012/final_review_20261012_0900/train/full/evaluation/endpoint_populations.npz')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/noise_audit_20261012/windows')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.24)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'WINDOW_DIAGNOSTIC_COMPLETE.json').exists():
        raise RuntimeError('Completed output exists; preserve it, do not refit silently')
    with np.load(args.endpoint, allow_pickle=False) as z:
        fake = z['generated_waveform'].copy()
        real = z['REAL_waveform'].copy()
        fake_ids = z['generated_source_parent_ids'].copy()
        real_groups = z['real_original_groups'].copy()
    splits = build_splits(fake_ids, real_groups)
    write_json(out / 'PREFIT_RECEIPT.json', dict(started_UTC=datetime.now(timezone.utc).isoformat(),
        endpoint_sha256=sha256(args.endpoint), script_sha256=sha256(__file__),
        protocol_sha256=sha256(ROOT / 'docs/NOISE_AUDIT_PROTOCOL.md'), views=VIEWS,
        cases=splits, updates_per_classifier=UPDATES, response_training_updates=0,
        device=str(device), torch_version=torch.__version__, seed=SEED,
        caveats=['Same acquisition only', 'REAL-REAL has few test groups',
                 'Diagnostic transformed AUC is not full-response acceptance',
                 'Independently fitted classifiers do not reveal the original discriminator feature attribution']))
    result = {}
    started = time.perf_counter()
    for view in VIEWS:
        f, r = view_array(fake, view), view_array(real, view)
        result[view] = {}
        for name, case in splits.items():
            x0, g0 = (f, fake_ids) if case['domain0'] == 'SOURCE' else (r, real_groups)
            value = fit_classifier(x0, r, g0, real_groups, case, view, device, out / view / name)
            result[view][name] = value
            print(json.dumps(dict(view=view, case=name, AUC=value['AUC'], CI=value['uncertainty']['raw']['percentile_95_interval'].tolist(), elapsed_s=time.perf_counter()-started)), flush=True)
            write_json(out / 'PROGRESS.json', _clean(result))
    report = dict(status='fixed_window_diagnostics_complete', results=result,
                  elapsed_s=time.perf_counter()-started, classifier_updates=UPDATES * len(VIEWS) * len(splits),
                  response_model_updates=0, endpoint_sha256=sha256(args.endpoint),
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0,
                  finished_UTC=datetime.now(timezone.utc).isoformat())
    write_json(out / 'WINDOW_DIAGNOSTIC_COMPLETE.json', _clean(report))


if __name__ == '__main__':
    main()
