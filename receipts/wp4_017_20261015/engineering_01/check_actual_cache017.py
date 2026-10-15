"""One new cache adapter check with manufactured original-class NN weights."""
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
from n01.model008 import N01Response008,canonical_charge008
from n01.model014 import FrozenDeterministicVoltageShape014
from n01.model017 import deterministic_loading017,assemble_cached017
from n01.model010 import tensor_state_digest010
from run_wp4_017 import build_frozen_cache017,sha
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);rng=torch.get_rng_state().clone()
with deterministic_loading017():
    base=N01Response008(dict(q_ref=3.7103998046755806,a_ref=.04,log_q_min=-3.,log_q_max=3.),'deterministic')
    base.set_stage('low_learning_rate');base.eval();base.requires_grad_(False)
    model=FrozenDeterministicVoltageShape014(base,seed=2026101540)
with torch.no_grad():
    model.basis_head.bias.copy_(torch.linspace(.003,.03,8))
model.eval();model.requires_grad_(False)
before=dict(base=tensor_state_digest010(base),head=tensor_state_digest010(model.basis_head))
t=torch.arange(384,dtype=torch.float64);source=np.stack([(.03*torch.exp(-((t-126)/12)**2)).numpy(),(.05*torch.exp(-((t-126)/15)**2)).numpy()]);source[:,:100]=0
charged={}
def call(name,charges,fn):
    for k,v in charges.items():charged[k]=charged.get(k,0)+v
    return fn()
cache=build_frozen_cache017(model,source,np.asarray([101,102]),OUT/'cache',call,chunk=2,heldout=1)
z=torch.tensor([[.2,-.4,.7,.1,-.3,.6,-.2,.8],[-.1,.6,-.2,.9,-.4,.2,.1,-.7]],dtype=torch.float32)
zq=torch.tensor([-.7,.9],dtype=torch.float32)
with torch.no_grad():
    mature=model(torch.as_tensor(source),z)
    A,B,Q=[torch.as_tensor(np.array(cache[k],copy=True)) for k in ('A','B','Q')]
    off=assemble_cached017(A,B,Q,z,None,A.new_tensor(.01*3.7103998046755806),A.new_tensor(.0025),enabled=False)
    on=assemble_cached017(A,B,Q,z,zq,A.new_tensor(.01*3.7103998046755806),A.new_tensor(.0025))
    s2=torch.log1p(.01*3.7103998046755806/Q+.0025)
    expected=mature['signal_pre']+mature['base_signal_pre_V']*torch.expm1(s2.sqrt()*zq.double()-.5*s2)[:,None]
checks=dict(actual_original_classes_constructed_without_RNG=torch.equal(rng,torch.get_rng_state()),
    one_registered_cache_A_and_head_pass_rows2=charged==dict(A_original_forward_rows=2,original_U_Net_decoded_rows=2,fixed_M0_head_actual_kernel_rows=2),
    actual_cache_A_bits_equal_mature=torch.equal(A,mature['base_signal_pre_V']),
    actual_cache_B_bits_equal_mature=torch.equal(B,mature['voltage_basis_V']),
    actual_positive_Q_matches_canonical=torch.equal(Q,canonical_charge008(A)) and bool((Q>0).all()),
    actual_decoded_float32_32x384=cache['manifest']['decoded_dtype']=='float32' and cache['manifest']['decoded_shape']==[2,32,384],
    cache_off_exact_mature_M0_bits=np.array_equal(off['signal_pre'].numpy().view(np.uint64),mature['signal_pre'].numpy().view(np.uint64)),
    cache_on_actual_charge_formula_bits=np.array_equal(on['signal_pre'].numpy().view(np.uint64),expected.numpy().view(np.uint64)),
    A_and_1288_head_weights_unchanged=before==dict(base=tensor_state_digest010(base),head=tensor_state_digest010(model.basis_head)),
    exactly_frozen_1288_head=sum(p.numel() for p in model.basis_head.parameters())==1288 and not any(p.requires_grad for p in model.parameters()))
assert all(checks.values()),checks
result=dict(passed=True,checks=checks,artificial_only=True,manufactured_original_class_forward_rows=4,
    manufactured_head_and_decoded_rows=4,engineering_registered_cache_rows=2,engineering_comparator_rows=2,
    actual_scientific_CP_NN_AD_G_RNG_SOURCE_REAL_DEV_GPU=0,explicit_random_sample_requests=0,
    original_checkpoint_files_opened=[],source_synthetic_manual=True,
    sources={n:dict(bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in ('scripts/run_wp4_017.py','src/n01/model017.py')})
(OUT/'ENGINEERING_RECEIPT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({'passed':True,'checks':len(checks),'cache_and_comparator_artificial_rows':4}),flush=True)
