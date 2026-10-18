"""Manufactured engineering checks; no data archive, CP, training or DEV."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import wasserstein_distance
import torch

from n01.data import write_json
from n01.model008 import canonical_charge008
from n01.model038 import N01Response038, TIME_KNOTS038
from n01.observation008 import observe_generated_carrier
from run_response039 import (HYPOTHESES, MORPH, PROTECTED, column_W1, joint_embedding,
                             objective039, projection039, confirmation039, seed039)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(39001)
    t = torch.arange(384, dtype=torch.float64)
    amp = torch.linspace(.03, .12, 32)[:, None]
    width = torch.linspace(3., 6., 32)[:, None]
    tail = torch.linspace(26., 50., 32)[:, None]
    source = amp*torch.where(t < 126, torch.exp(-.5*((t-126)/width).square()),
                             torch.exp(-(t-126)/tail))
    source[:, :100] = 0
    adapter = torch.nn.Linear(2, 8)
    torch.nn.init.zeros_(adapter.weight)
    torch.nn.init.zeros_(adapter.bias)
    model = SimpleNamespace(mean_adapter=adapter, time_knots=torch.tensor(TIME_KNOTS038))
    encoding = dict(original=dict(signal_pre=source), condition=torch.zeros(32, 2))
    with torch.no_grad():
        identity = N01Response038._mean_response(model, encoding, enabled=True)
        assert torch.equal(identity['mean_signal_pre_V'], source)
        truth = torch.tensor([.125, -.125, .0625, .125, -.0625, 0., 0., 0.])
        adapter.bias.copy_(truth)
        changed = N01Response038._mean_response(model, encoding, enabled=True)
        assert torch.equal(changed['mean_signal_pre_V'][:, :100], source[:, :100])
        q_error = float((canonical_charge008(changed['mean_signal_pre_V'])-canonical_charge008(source)).abs().max())
        assert q_error < 1e-12
        assert bool((changed['time_queries'].diff(dim=1) > 0).all())
        y0 = observe_generated_carrier(source, ste=False, q_floor=.001)
        yt = observe_generated_carrier(changed['mean_signal_pre_V'], ste=False, q_floor=.001)
    scale = dict(Q_Vns=.14, peak_voltage_V=.0055, FWHM_ns=.51,
                 rise10_90_ns=.21, tail_Vns=.031, PSD_guarded=.0043, q_floor=.001)
    def embedding(observation):
        return joint_embedding(*(observation[k].numpy() for k in ('window_V', 'baseline100', 'peak_index')), scale)
    z0, zt = embedding(y0), embedding(yt)
    projection = projection039(3901, 'cpu')
    distance_before = float(column_W1(torch.tensor(z0)@projection, torch.tensor(zt)@projection, 16))
    distance_truth = float(column_W1(torch.tensor(zt)@projection, torch.tensor(zt)@projection, 16))
    assert distance_before > 0 and distance_truth == 0
    permutation_distance = float(column_W1(torch.tensor(zt[::-1].copy())@projection, torch.tensor(zt)@projection, 16))
    assert permutation_distance < 1e-12
    # A censored front retains its row, masks and finite observable embedding.
    clipped = yt['window_V'].numpy().copy()
    clipped[0, :16] = clipped[0, 16]*.95
    z_missing = joint_embedding(clipped, yt['baseline100'].numpy(), yt['peak_index'].numpy(), scale)
    assert np.isfinite(z_missing).all() and len(z_missing) == 32 and z_missing[0, 1] == 1
    x, y = torch.randn(23, 5, dtype=torch.float64), torch.randn(31, 5, dtype=torch.float64)
    expected = np.mean([wasserstein_distance(x[:, i], y[:, i]) for i in range(5)])
    assert abs(float(column_W1(x, y, 2))-expected) < 1e-12
    contract = dict(fit_primary_relative_worsening_max=.01, protected_allowance_T=.05,
                    morphology_individual_allowance_T=.05, useful_wave_relative_gain=.02,
                    useful_joint_relative_gain=.02, useful_morphology_mean_T_gain=1.)
    tolerances = {h: dict.fromkeys(PROTECTED+MORPH, 1.) for h in HYPOTHESES}
    baseline = {h: dict(values=dict.fromkeys(PROTECTED+MORPH+('wave', 'joint_morphology', 'negative_lobe_area_Vns'), 10.),
                         fake_flags={'missing': dict(count=0, denominator=32)},
                         observation_flags={'ADC': dict(count=0, denominator=32)},
                         conditional_coverage=dict(conditional_coverage_complete=True)) for h in HYPOTHESES}
    good = copy.deepcopy(baseline)
    for h in HYPOTHESES:
        for key in ('wave', 'joint_morphology')+MORPH:
            good[h]['values'][key] = 8.
    assert objective039(good, baseline, tolerances, contract)['feasible']
    assert objective039(good, baseline, tolerances, contract)['score'] < 1
    bad = copy.deepcopy(good)
    bad['slow']['values']['QPSD'] += .051
    assert not objective039(bad, baseline, tolerances, contract)['feasible']
    bad = copy.deepcopy(good)
    bad['fast']['observation_flags']['ADC']['count'] = 1
    assert not objective039(bad, baseline, tolerances, contract)['feasible']
    success = confirmation039([(good, baseline)]*4, tolerances, contract)
    assert success['useful_mean_correction'] and not success['full_random_response_scientific_acceptance']
    assert not confirmation039([(baseline, baseline)]*4, tolerances, contract)['useful_mean_correction']
    assert seed039(1, 'FIT', 0, 'shape') != seed039(1, 'CONFIRM', 0, 'shape')
    write_json(args.out, dict(status='PASS', manufactured_rows=32, checkpoint_loads=0, dataset_loads=0,
               AD_queries=0, optimizer_steps=0, canonical_Q_max_error=q_error,
               joint_before=distance_before, joint_truth=distance_truth, missing_rows_retained=True,
               unequal_population_W1_matches_scipy=True, row_permutation_distance=permutation_distance,
               row_permutation_tolerance=1e-12,
               protected_QPSD_and_ADC_failures_rejected=True, unchanged_endpoint_not_useful=True,
               exact_synthetic_target_passes_stage_but_not_full_scientific_goal=True,
               scientific_data_validity_established=False))
    print('response039 manufactured engineering: PASS')


if __name__ == '__main__':
    main()
