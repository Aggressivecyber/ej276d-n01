"""One bounded public-fixture check for the020 recovery and formerly missing lazy import."""
from pathlib import Path
import copy
import functools
import hashlib
import json
import os
import sys
import types

here = Path(__file__).resolve().parent
repo = here.parents[2]
pkg = here/'source_package'
for name in ('tmp', 'cache'):
    (here/name).mkdir(exist_ok=True)
for name in ('TEMP', 'TMP', 'TMPDIR'):
    os.environ[name] = str(here/'tmp')
for name in ('TORCH_HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR'):
    os.environ[name] = str(here/'cache')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
assert sys.flags.isolated and sys.dont_write_bytecode
sys.path[:] = [str(pkg/'scripts'), str(pkg/'src')] + [p for p in sys.path if not Path(p).resolve().is_relative_to(repo)]
package = types.ModuleType('n01')
package.__path__ = [str(pkg/'src/n01')]
sys.modules['n01'] = package
import numpy as np
import torch
import run_wp4_020 as driver
import run_wp4_017 as legacy
from run_wp1_013 import observation_rows, lossless_window_codes
from n01.model017 import CachedFrozenQ1Charge017

torch.set_num_threads(1)
blocked_calls = []
def prohibited(*args, **kwargs):
    blocked_calls.append('prohibited scientific operation')
    raise AssertionError('Fixture forbids CP, NN forward, AD, optimizer and random draws')
for name in ('rand', 'randn', 'randint', 'randperm', 'load'):
    setattr(torch, name, prohibited)
torch.optim.AdamW = prohibited
torch.Tensor.backward = prohibited
torch.autograd.backward = prohibited
torch.autograd.grad = prohibited
torch.nn.Module._call_impl = prohibited
legacy.Engine017 = prohibited
legacy.ParentSampler017 = prohibited
rng_before = torch.get_rng_state().clone()
p = driver.settings020()
p['budget_caps'] = driver.budget_caps020(p)
driver.validate_settings020(p, frozen=False)
assert p['budget_caps']['G_updates'] == 2000
assert p['budget_caps']['checkpoint_loads'] == 2
assert p['budget_caps']['noise005_rows'] == 4939904
assert p['budget_caps']['total_scalar_normal_requests'] == 1946322176
assert p['budget_caps']['hard_ADC_rows'] == 14819712
assert p['budget_caps']['unpaired_training_energy_pair_distances'] == 12582912000
assert p['budget_caps']['evaluation_energy_pair_distances'] == 162029568
assert p['budget_caps']['evaluation_observable_pair_distances'] == 405073920
coverage = {role: dict(all_covered=True, unique=5274, fit_parents=5274)
            for role in ('student0', 'student1', 'teacher0', 'teacher1')}
state = dict(schema='n01/wp4-019-checkpoint/v1', run_id='wp4_019_20261016', trial=driver.TRIALS[0],
    teacher='T1', mode='unpaired', seed=0, G_updates=2000, complete_step_boundary=True, inflight={},
    cache_manifest_sha256='public-fixed-cache', frozen_state_sha256={'noise005': 'fixture', 'extension': 'fixture'},
    parameters=dict(theta_a=torch.tensor(-5., dtype=torch.float64), theta_b=torch.tensor(-8., dtype=torch.float64),
                    q_reference=torch.tensor(p['qref'], dtype=torch.float64)),
    last_metrics=dict(G_updates=2000, teacher='T1', mode='unpaired', seed=0, theta_a=-5., theta_b=-8., parent_coverage=coverage),
    optimizer='NOT_USED', student_streams='NOT_USED', teacher_streams='NOT_USED', parent_sampler_states='NOT_USED')
model, metrics, recovered_coverage = driver.restore_seed0_model020(state, p, 'public-fixed-cache', state['frozen_state_sha256'], 'cpu')
assert all(not value.requires_grad for value in model.parameters())
assert set(dict(model.named_parameters())) == {'theta_a', 'theta_b'}
assert all(torch.equal(value, model.state_dict()[key]) for key, value in state['parameters'].items())
assert recovered_coverage == coverage
rejected = []
for key, bad in [('G_updates', 1999), ('mode', 'paired'), ('complete_step_boundary', False), ('run_id', 'wp4_020_20261016')]:
    changed = dict(state); changed[key] = bad
    try:
        driver.restore_seed0_model020(changed, p, 'public-fixed-cache', state['frozen_state_sha256'], 'cpu')
    except ValueError:
        rejected.append(key)
    else:
        raise AssertionError('Invalid source checkpoint accepted: '+key)
# Run the original kernel evaluator on four public fixed parents,2 native banks,32 draws.
# Only its signals provider is replaced; observation/diagnostics/codec/moments are actual frozen code.
fixture = here/'fixture_artifacts'
cache = here/'fixed_cache'; cache.mkdir()
(cache/'CACHE_MANIFEST.json').write_text('{"public_fixture":true}\n', encoding='utf-8')
p['evaluation'] = dict(p['evaluation'], parents=4)
store = legacy.Store017(fixture, p, {}, artificial=True)
legacy.write_new = functools.partial(driver.write_generated020, output=store.root, store=store)
cache_rows = dict(Q=np.array([1.,1.1,1.2,1.3]), ids=np.array(['public0','public1','public2','public3']),
    directory=cache, manifest={'T2_fixed_B0_scope': 'public fixture; not an evaluation of T2'})
rt = legacy.Runtime017(p, cache_rows, types.SimpleNamespace(assert_frozen=lambda: None),
    {'loss': {'q_floor': 1e-12, 'peak_floor': 1e-12}}, {}, store, 'cpu')
time_index = np.arange(384)
fixed = np.stack([.04*np.maximum(0.,1.-np.abs(time_index-peak)/10.) for peak in (126,127,128,129)])
provided = []
def public_fixed_signals(rows, streams, *, model=None, teacher=None, training=False):
    assert not training and teacher is None
    provided.append(dict(rows=len(rows), namespace=copy.deepcopy(streams.namespace)))
    carrier = torch.as_tensor(fixed[rows], dtype=torch.float64)
    q = torch.as_tensor(cache_rows['Q'][rows], dtype=torch.float64)
    out = dict(signal_pre=carrier, qsig=q, Q_mu_Vns=q, theory_variance_Vns2=.01*q+.0025*q.square())
    return out, {h: torch.zeros_like(carrier) for h in driver.HYPOTHESES}
rt.signals = public_fixed_signals
legacy.evaluate_kernel017(rt, store.root/'evaluation', model=model)
assert provided == [dict(rows=128,namespace=dict(kind='evaluation',role='student',bank=bank)) for bank in (0,1)]
assert store.counts['evaluation_signal_rows'] == 256
assert store.counts['hard_ADC_rows'] == 768
assert store.counts['evaluation_kernel_snapshots'] == 1
hard_receipts = {}
for bank in (0,1):
    for hypothesis in driver.HYPOTHESES:
        path = store.root/'evaluation'/f'bank{bank}_{hypothesis}_HARD_ROWS.npz'
        wave, flags, peak = legacy.decode_hard017(path)
        assert wave.shape == (128,256) and peak.shape == (128,) and len(flags)>0
        assert np.isfinite(wave).all()
        with np.load(path,allow_pickle=False) as ar:
            assert len(ar['source_parent_row']) == 128 and len(np.unique(ar['source_parent_row'])) == 4
        hard_receipts[path.name] = dict(rows=128, flags=len(flags), raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
with np.load(store.root/'evaluation/EVALUATION_RANDOM_STATES.npz',allow_pickle=False) as ar:
    for bank in (0,1):
        for stream in ('shape','charge','noise005','extension284','sign','PIT'):
            assert np.array_equal(ar[f'{bank}_initial_{stream}'],ar[f'{bank}_final_{stream}'])
assert torch.equal(rng_before,torch.get_rng_state()) and blocked_calls == []
project_modules = {}
for name, module in sorted(sys.modules.items()):
    path = getattr(module,'__file__',None)
    if path and (name.startswith('n01.') or name in ('run_wp4_020','run_wp4_019','run_wp4_017','run_wp2_014','run_wp4_018','run_wp1_013','wp1_ops','mechanism012_losses')):
        actual = Path(path).resolve()
        assert actual.is_relative_to(pkg), (name,str(actual))
        project_modules[name] = str(actual.relative_to(pkg))
assert Path(project_modules['mechanism012_losses']).as_posix() == 'scripts/mechanism012_losses.py'
manifest = json.loads((here/'SOURCE_PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
for rel, record in manifest['files'].items():
    raw=(pkg/rel).read_bytes()
    assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
    assert raw==(repo/rel).read_bytes(), rel
receipt = dict(schema='n01/wp4-020-isolated-engineering-check/v1', status='PASS', artificial_only=True,
    executable=sys.executable, python=sys.version, numpy=np.__version__, torch=torch.__version__,
    source_package_manifest_sha256=hashlib.sha256((here/'SOURCE_PACKAGE_MANIFEST.json').read_bytes()).hexdigest(),
    source_package_files=manifest['files'], source_package_count=32, project_modules=project_modules,
    original_repository_import_fallback=False, python_isolated=True, pycache_disabled=True,
    all_TEMP_TMP_and_requested_library_cache_paths_under_E_engineering=True,
    restore=dict(parameters_restored=['theta_a','theta_b'],qref_fixed=True,requires_grad=False,
        source019_G=2000,new_G=0,optimizer_created=False,Engine_created=False,parent_samplers_created=False,
        rejected_checkpoint_metadata=rejected),
    actual_evaluator=dict(parents=4,banks=2,draws=32,signal_rows=256,hard_ADC_rows=768,
        hypotheses=list(driver.HYPOTHESES),hard_files=hard_receipts,
        observation_rows_numeric_flags_peak_diagnostics_and_codec_executed=True,native32_report_generated=True),
    NN_forward_rows=0, CP_loads=0, original_arrays_or_waveforms_read=0, G=0, AD=0, random_draws=0,
    scalar_parent_draws=0, global_CPU_RNG_unchanged=True, all_evaluation_stream_states_unchanged=True,
    prohibited_calls_seen=blocked_calls,scientific_qualification_inferred=False)
(here/'ISOLATED_CHECK_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='PASS',source_package_count=32,hard_ADC_rows=768,NN=0,CP=0,G=0,AD=0,RNG=0)),flush=True)
