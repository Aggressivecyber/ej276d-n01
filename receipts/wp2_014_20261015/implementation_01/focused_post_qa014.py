"""Focused artificial regressions for the final accounting/flag amendments."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'src'))
import torch
import numpy as np
import run_wp2_014 as d
from n01.observation008 import observe_generated_carrier
from wp1_ops import write_new, sha
from wp1_evaluation import observable_features

out = Path(sys.argv[1]).resolve()
assert out.drive.upper() in ('E:', 'D:') and not out.exists()
out.mkdir(parents=True)
p = d.settings014()
p['budget_caps'] = d.budget_caps014(p)
caps = p['budget_caps']
assert caps['head_actual_parameter_forward_rows'] == 8616512
assert caps['calibration_map_rows'] == 36864
assert caps['head_total_kernel_rows'] == 8653376
assert caps['head_total_kernel_rows'] == caps['head_actual_parameter_forward_rows']+caps['calibration_map_rows']
s = d.Store014(out/'ARTIFICIAL_CHARGE_BRIDGE', p, {}, artificial=True)
s.call('actual_head_only', dict(head_actual_parameter_forward_rows=4), lambda: True)
s.call('calibration_map_only', dict(calibration_map_rows=7), lambda: True)
s.call('explicit_both', dict(head_actual_parameter_forward_rows=2, calibration_map_rows=3), lambda: True)
assert s.counts['head_actual_parameter_forward_rows'] == 6
assert s.counts['calibration_map_rows'] == 10
assert s.counts['head_total_kernel_rows'] == 16
assert s.entered == s.completed == 3 and not s.inflight
def fail():
    raise RuntimeError('artificial unknown native completion')
try:
    s.call('failed_map', dict(calibration_map_rows=5), fail)
except RuntimeError:
    pass
else:
    raise AssertionError('missing injected failure')
assert s.counts['calibration_map_rows'] == 10 and s.counts['head_total_kernel_rows'] == 16
assert s.attempted['calibration_map_rows'] == 15 and s.attempted['head_total_kernel_rows'] == 21
assert s.entered == 4 and s.completed == 3 and s.inflight

carrier = torch.zeros(4, 384, dtype=torch.float64)
carrier[1, 100:] = -.0001
carrier[2, 112:200] = .0001
carrier[3, 126:170] = .02
q_floor = .01
raw = observe_generated_carrier(carrier, ste=False, q_floor=None, return_carrier=True)
protected = observe_generated_carrier(carrier, ste=False, q_floor=q_floor, return_carrier=True)
for n in ('window_V', 'baseline100', 'baseline_mean_V', 'adc_carrier', 'Q', 'tail', 'peak_index'):
    assert torch.equal(raw[n], protected[n]), n
features, flags = observable_features(protected['window_V'].numpy(), protected['baseline100'].numpy(), q_floor)
assert np.array_equal(protected['flags']['PSD_guarded'].numpy(), flags['PSD_guarded'])
assert np.array_equal(protected['flags']['PSD_undefined'].numpy(), flags['PSD_undefined'])
assert len(features['Q_Vns']) == 4 and (features['Q_Vns'] <= 0).any() and flags['PSD_guarded'].any()
rss = d.current_rss014()
assert rss is not None and rss > 0
for rel in d.OWNED_FILES:
    compile((ROOT/rel).read_text('utf-8-sig'), rel, 'exec')
write_new(out/'POST_QA_BRIDGE.json', dict(schema='n01/wp2-post-QA-bridge/v1', status='PASS',
    artificial_only=True, private_data_reads=0, trained_checkpoint_loads=0, real_NN_rows=0,
    real_AD_queries=0, scientific_G_updates=0, SSH_calls=0,
    tested_final_amendments=['head_total=actual_head+calibration_maps for separate/combined/durable failed charges',
        'explicit frozen observation q_floor preserves ADC/window/Q/tail/peak bytes and fixes guarded flags',
        'native Windows RSS fallback returns actual positive RSS', 'all three final source files compile'],
    counters=s.counts, attempted=s.attempted, operations_entered=s.entered, operations_completed=s.completed,
    unknown_inflight=s.inflight, fake_signed_Q_rows=features['Q_Vns'], fake_guarded_rows=flags['PSD_guarded'],
    resource_probe_RSS_bytes=rss, final_budget_caps=caps,
    source_SHA256={rel:sha(ROOT/rel) for rel in d.OWNED_FILES},
    prior_full_manual_receipt='manual_02/MANUAL_SEMANTICS.json',
    post_manual_metadata_only_changes=['resource poll/grace fields', 'per-trial COMPLETE native_status',
        'explicit REAL unknown ADC flags', 'CPU BLAS thread variables before numeric import',
        'original TRAIN-statistic DEV-decoded flags asserted']))
print(json.dumps(dict(status='PASS', total_cap=caps['head_total_kernel_rows'], RSS_bytes=rss)))
