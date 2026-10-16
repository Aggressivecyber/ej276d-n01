"""One artificial control-only015 -> actual hard008 -> original codec fixture."""
from pathlib import Path
import hashlib
import json
import sys
import time
from types import SimpleNamespace

root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root/'scripts'), str(root/'src')]
import run_wp3_024 as driver
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
import numpy as np
import torch
from n01.model008 import canonical_charge008, _onset
from n01.model015 import FrozenC1MeanResponse015, _ExactZeroMeanAddition015
from n01.model017 import assemble_cached017, deterministic_loading017
from n01.response008 import fixed_noises, make_extension, state_digest
from n01.train_shape010 import Streams010
from n01.observation008 import observe_generated_carrier
from run_wp1_013 import lossless_window_codes

started = time.monotonic()
torch.set_num_threads(2)
before = torch.get_rng_state().clone()
t = torch.arange(384, dtype=torch.float64)
gate = _onset(t)
projection = gate*torch.exp(-.5*((t-180.)/65.)**2)
projection = projection/canonical_charge008(projection)
p = driver.settings024()
source = torch.stack([scale*(torch.exp(-.5*((t-126.)/width)**2)+.09*torch.exp(-(t-126.).clamp_min(0)/35.))*gate
    for scale, width in ((.02, 5.), (.035, 8.), (.08, 11.))])
a = source*torch.tensor([1.1, .8, 1.3])[:, None]+.0003*gate[None, :]*torch.exp(-(t-140.).clamp_min(0)/40.)
source_centered = source-source[:, :100].mean(1, keepdim=True)
q = canonical_charge008(source_centered)
tail = source_centered[:, 156:366].sum(1)
assert not torch.equal(q, canonical_charge008(a))
u = torch.arange(-6, 7, dtype=torch.float64)
gaussian = torch.exp(-.5*(u/2.)**2)
gaussian = gaussian/gaussian.sum()
causal = torch.exp(-torch.arange(64, dtype=torch.float64)/16.)
causal = causal/causal.sum()
holder = SimpleNamespace(mean_q_reference=torch.tensor(p['q_reference'], dtype=torch.float64),
    mean_q_floor=torch.tensor(p['q_floor'], dtype=torch.float64),
    mean_gaussian_kernel=gaussian, mean_causal_kernel=causal,
    baseline=SimpleNamespace(gate=gate[None, None], projection=projection[None, None]))
carrier = a+.0001*torch.sin(t/7.)[None, :]
points = p['theta_points']+[[-3., 0., 3., 0., 3., 0.]]
checks, clipped = [], 0
with torch.no_grad():
    for index, point in enumerate(points):
        holder.mean_theta = torch.tensor(point, dtype=torch.float64)
        original = FrozenC1MeanResponse015.mean_adapter(holder, source, a)
        control = driver.mean_adapter024(a, q, tail, holder.mean_theta, p['q_reference'], p['q_floor'])
        for key in original:
            torch.testing.assert_close(control[key], original[key], rtol=0., atol=0.)
        old_carrier = _ExactZeroMeanAddition015.apply(carrier, original['delta_mean_pre_V'])
        new_carrier = _ExactZeroMeanAddition015.apply(carrier, control['delta_mean_pre_V'])
        old = observe_generated_carrier(old_carrier, ste=False, q_floor=p['q_floor'], return_carrier=True, return_indices=True)
        new = observe_generated_carrier(new_carrier, ste=False, q_floor=p['q_floor'], return_carrier=True, return_indices=True)
        for key in ('window_V', 'baseline100', 'baseline_mean_V', 'peak_index', 'adc_carrier', 'crop_indices'):
            torch.testing.assert_close(new[key], old[key], rtol=0., atol=0.)
        if index == 0:
            assert torch.equal(new_carrier.view(torch.int64), carrier.view(torch.int64))
        for obs in (old, new):
            fields = {key: obs[key].numpy() for key in ('window_V', 'baseline100', 'peak_index', 'adc_carrier')}
            fields['diagnostic__baseline_mean_V'] = obs['baseline_mean_V'].numpy()
            coded = lossless_window_codes(fields)
            decoded = coded['ADC_crop_codes'].astype(np.float64)/16383-.5-coded['baseline_mean_V'][:, None]
            if 'IEEE754_reconstruction_XOR_uint64' in coded:
                decoded.view(np.uint64)[:] ^= coded['IEEE754_reconstruction_XOR_uint64']
            assert np.array_equal(decoded.view(np.uint64), fields['window_V'].view(np.uint64))
        clipped += int((control['mean_trust_scale'] < 1).sum())
        checks.append(dict(theta=point, scientific_profile_point=index < 13, all_adapter_fields_exact=True,
            original_hard_ADC_argmax_crop_exact=True, codec_float64_bits_exact=True,
            maximum_trust=float(control['mean_RMS_relative_A'].max()),
            clipped_parents=int((control['mean_trust_scale'] < 1).sum())))
assert clipped > 0
assert torch.equal(before, torch.get_rng_state())
sources = sorted({str(Path(module.__file__).resolve().relative_to(root)).replace('\\', '/')
    for module in tuple(sys.modules.values()) if getattr(module, '__file__', None)
    and Path(module.__file__).is_file() and Path(module.__file__).resolve().is_relative_to(root)
    and Path(module.__file__).suffix == '.py' and Path(module.__file__).resolve() != Path(__file__).resolve()})
expected = set(driver.SCIENCE_SOURCES024)
assert set(sources) <= expected, sorted(set(sources)-expected)
record = dict(schema='n01/wp3-024-composed-artificial-fixture/v1', status='PASS', artificial_only=True,
    source_driver_sha256=hashlib.sha256((root/'scripts/run_wp3_024.py').read_bytes()).hexdigest(),
    reference_model015_sha256=hashlib.sha256((root/'src/n01/model015.py').read_bytes()).hexdigest(),
    artificial_parents=3, scientific_profile_points=13, engineering_only_trust_fixture_points=1,
    source_controls_from_SOURCE_Q_tail_not_A=True, global_CPU_RNG_unchanged=True,
    mean_adapter_parent_rows=84, hard_ADC_argmax_rows=84, lossless_codec_rows=84,
    NN_forward_rows=0, checkpoint_loads=0, AD_queries=0, G_updates=0, RNG_requests=0,
    private_NPZ_inputs=0, full_old_suites_repeated=False, checks=checks,
    earlier_attempts_file='ARTIFICIAL_ATTEMPTS.json', cumulative_artificial_hard_rows=252,
    imported_local_sources=sources, scientific_source_closure=list(driver.SCIENCE_SOURCES024),
    elapsed_seconds=time.monotonic()-started,
    actual_scientific_execution=False, python_executable=sys.executable, torch_version=torch.__version__)
out = Path(__file__).with_name('COMPOSED_ADAPTER_HARD_FIXTURE_RECEIPT.json')
with out.open('x', encoding='utf-8') as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(dict(status='PASS', receipt=str(out), artificial_hard_rows=84,
    imported_source_count=len(sources), closure_count=len(expected)), ensure_ascii=False))
