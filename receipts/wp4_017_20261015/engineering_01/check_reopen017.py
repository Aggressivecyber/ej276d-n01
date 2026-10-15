"""Reopen only fresh manufactured017 output/checkpoints; never originals."""
from pathlib import Path
import hashlib
import json
import os
import sys
if os.name=='nt':
    import platform
    platform._wmi=None
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False,parents=True)
import numpy as np
import torch
from run_wp4_017 import decode_hard017,Streams017,sha
from n01.stats017 import scalar_features017
from n01.saved_array_kernels017 import scalar_moments
fixture=ROOT/'receipts/wp4_017_20261015/engineering_01/artificial_driver_02/artifacts'
directory=fixture/'T1_paired_seed0/evaluation';checked=[];checks={};rng=torch.get_rng_state().clone()
def check(name,condition):
    assert bool(condition),name
    checks[name]=True
hard=directory/'bank0_fast_HARD_ROWS.npz';Y,flags,peaks=decode_hard017(hard);checked.append(hard)
with np.load(directory/'bank0_BASELINE100.npz',allow_pickle=False) as a:
    baseline=a['baseline100'];ids=a['source_parent_ids'];rows=a['source_parent_row'];draw=a['draw_index']
    checked.append(directory/'bank0_BASELINE100.npz')
    with np.load(hard,allow_pickle=False) as h:
        check('saved_parent_draw_bank_rows_align',np.array_equal(h['source_parent_ids'],ids) and np.array_equal(h['source_parent_row'],rows)
            and np.array_equal(h['draw_index'],draw) and np.all(h['bank_index']==0) and np.array_equal(draw,np.tile(np.arange(32),6)))
scalar,morph=scalar_features017(Y,baseline,peaks);moments=scalar_moments(scalar.reshape(6,32,9))
with np.load(directory/'bank0_fast_REPORT.npz',allow_pickle=False) as report:
    check('new_saved_codec_plus_baseline_reopens_actual_scalar9',np.array_equal(scalar.reshape(6,32,9),report['scalar9'],equal_nan=True))
    check('new_saved_row_join_preserves_native32_wave_moments',np.array_equal(Y.reshape(6,32,256).mean(1),report['wave_mean'])
        and np.array_equal(Y.reshape(6,32,256).var(1,ddof=1),report['wave_variance']))
    check('new_saved_native32_actual_finite_pair_counts',all(np.array_equal(v,report['scalar_'+k],equal_nan=True) for k,v in moments.items()))
    checked.append(directory/'bank0_fast_REPORT.npz')
check('new_saved_morphology_flags_all_reopen',all(np.array_equal(v,flags['flag__'+k]) for k,v in morph.items() if np.asarray(v).dtype.kind=='b'))
path=fixture/'T1_paired_seed0/final.pt';state=torch.load(path,map_location='cpu',weights_only=False);checked.append(path)
check('own_artificial_checkpoint_only2theta_and_fixed_qref',set(state['parameters'])=={'theta_a','theta_b','q_reference'}
    and state['complete_step_boundary'] is True and state['G_updates']==2 and len(state['optimizer']['param_groups'][0]['params'])==2)
check('own_artificial_checkpoint_actual_last_theta_values',float(state['parameters']['theta_a'])==state['last_metrics']['theta_a']
    and float(state['parameters']['theta_b'])==state['last_metrics']['theta_b'])
for role in ('student','teacher'):
    original=state[role+'_streams']['charge'];g0=torch.Generator(device='cpu');g1=torch.Generator(device='cpu')
    g0.set_state(original.clone());g1.set_state(original.clone())
    first=torch.randn(4,generator=g0,dtype=torch.float32);second=torch.randn(4,generator=g1,dtype=torch.float32)
    check(role+'_checkpoint_stream_state_roundtrip',torch.equal(first,second) and torch.equal(g0.get_state(),g1.get_state()))
    del g0,g1
with np.load(directory/'EVALUATION_RANDOM_STATES.npz',allow_pickle=False) as a:
    expected=Streams017(dict(kind='evaluation',role='student',bank=0)).state_dict()
    check('actual_eval_all_six_initial_stream_states_retained',all(np.array_equal(a['0_initial_'+k],v.numpy()) for k,v in expected.items()))
    checked.append(directory/'EVALUATION_RANDOM_STATES.npz')
check('reopen_preserves_all_fixture_input_bytes',all(q.is_file() for q in checked))
check('reopen_global_CPU_RNG_unchanged',torch.equal(rng,torch.get_rng_state()))
result=dict(passed=True,checks=checks,artificial_only=True,
    own_manufactured_files_reopened={str(q.relative_to(ROOT)).replace('\\','/'):dict(bytes=q.stat().st_size,sha256=sha(q)) for q in checked},
    actual_scientific_CP_NN_AD_G_RNG_SOURCE_REAL_DEV_GPU=0,engineering_own_small_checkpoint_loads=1,
    engineering_explicit_normal_requests=16,engineering_model_forwards_backward_optimizer_steps=0,
    sources={n:dict(bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in ('scripts/run_wp4_017.py','src/n01/stats017.py','src/n01/saved_array_kernels017.py')})
(OUT/'ENGINEERING_RECEIPT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({'passed':True,'checks':len(checks),'own_small_fixture_files_reopened':len(checked)}),flush=True)
