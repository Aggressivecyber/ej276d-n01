"""Artificial CPU-only literals; no original arrays, checkpoints or GPU calls."""
from pathlib import Path
import hashlib
import json
import sys
import math
import faulthandler
faulthandler.dump_traceback_later(30,repeat=False)
import os
if os.name=='nt':
    # Python3.13's optional WMI query hung before Torch DLL selection. Use
    # stdlib's own supported fallback; no library files or OS service changed.
    import platform
    platform._wmi=None

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
print('artificial fixture: importing NumPy',flush=True)
import numpy as np
print('artificial fixture: importing Torch CPU',flush=True)
import torch
print('artificial fixture: Torch import complete',flush=True)
faulthandler.cancel_dump_traceback_later()
from n01.model017 import CachedFrozenQ1Charge017,deterministic_loading017,energy_score_U017,teacher_shape_coefficients017
from n01.model008 import canonical_charge008
from n01.observation008 import observe_generated_carrier
from n01.stats017 import weighted017,coverage017,support_cell017

torch.set_num_threads(2)
checks={}
def check(name,condition):
    assert bool(condition),name
    checks[name]=True
def reject(name,fn):
    try:fn()
    except (ValueError,RuntimeError):checks[name]=True
    else:raise AssertionError(name)

rng=torch.get_rng_state().clone();original=torch.nn.init.normal_
with deterministic_loading017():
    head=torch.nn.Conv1d(32,8,5,padding=2)
    linear=torch.nn.Linear(5,3)
check('placeholder_zero_no_rng',torch.equal(rng,torch.get_rng_state()) and not head.weight.any() and not linear.bias.any())
try:
    with deterministic_loading017():raise RuntimeError('manufactured constructor exception')
except RuntimeError:pass
check('placeholder_restore_on_exception',torch.nn.init.normal_ is original)

S=torch.zeros((3,2,256),dtype=torch.float64,requires_grad=True);T=torch.zeros_like(S)
loss=energy_score_U017(S,T,paired=True).mean();loss.backward()
check('zero_distance_finite_zero_gradient',float(loss)==0 and torch.isfinite(S.grad).all() and not S.grad.any())
X=torch.tensor([[0.]*256,[0.]*256,[.01]*256,[-.02]*256],dtype=torch.float64,requires_grad=True)
Y=torch.tensor([[.02]*256,[0.]*256,[0.]*256,[-.01]*256],dtype=torch.float64)
score=energy_score_U017(X,Y,block_rows=2);score.backward()
check('overlapping_offdiagonal_zero_distance_stable',torch.isfinite(score) and torch.isfinite(X.grad).all())
from n01.model017 import metric_features017
f,g=metric_features017(X.detach()),metric_features017(Y)
brute=sum(torch.linalg.vector_norm(a-b) for a in f for b in g)/16-sum(torch.linalg.vector_norm(f[i]-f[j]) for i in range(4) for j in range(4) if i!=j)/24
check('exact_U_terms_against_literal',torch.allclose(score.detach(),brute,atol=1e-12,rtol=1e-13))

t=torch.arange(384,dtype=torch.float64);base=torch.exp(-.5*((t-126)/7)**2)*.03;base[:100]=0
A=torch.stack([base*.5,base,base*1.4,base*2])
B=torch.zeros((4,8,384),dtype=torch.float64)
B[:,0,145:155]=.0001;B[:,0,155:165]=-.0001
Q=canonical_charge008(A);z=torch.zeros((4,8));z[:,0]=torch.tensor([-.4,.7,-1.,.2])
zq=torch.tensor([-.8,.3,1.2,-1.4]);model=CachedFrozenQ1Charge017()
before_rng=torch.get_rng_state().clone();out=model(A,B,Q,z,zq)
check('cached_model_no_rng',torch.equal(before_rng,torch.get_rng_state()))
off=model(A,B,Q,z,None,enabled=False)
check('charge_off_exact_M0_bits',torch.equal(off['signal_pre'].view(torch.int64),out['M0_reference_signal_pre_V'].view(torch.int64)))
check('cache_not_modified',torch.equal(Q,canonical_charge008(A)) and not A.requires_grad and not B.requires_grad)
reject('invalid_mu_rejected',lambda:model(A,B,torch.zeros_like(Q),z,zq))
sign=torch.tensor([-1,1,-1,1],dtype=torch.int8)
mix=teacher_shape_coefficients017(z,teacher='T2',sign=sign)
check('T2_only_first_coordinate',torch.equal(mix[:,1:],z[:,1:].double()) and torch.allclose(mix[:,0],.9*sign.double()+math.sqrt(.19)*z[:,0].double()))
reject('T1_sign_not_allowed',lambda:teacher_shape_coefficients017(z,teacher='T1',sign=sign))
target=torch.stack([base*.4,base*1.1,base*1.65,base*1.7])
optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=.01,weight_decay=0.)
prediction=observe_generated_carrier(out['signal_pre'],ste=True)['window_V']
target=observe_generated_carrier(target,ste=False)['window_V']
loss=energy_score_U017(prediction,target,block_rows=2);optimizer.zero_grad();loss.backward()
grads=[float(p.grad) for p in model.trainable_parameters()];before=[p.detach().clone() for p in model.trainable_parameters()]
model.assert_optimizer_membership(optimizer);optimizer.step()
check('actual_both_theta_gradient_step',[g!=0 and math.isfinite(g) for g in grads]==[True,True] and all(not torch.equal(a,b) for a,b in zip(before,model.trainable_parameters())))

check('weighted_HC1_and_ESS',weighted017([1.,3.],[1.,3.])==dict(mean=2.5,SE=.75,ESS=1.6,parents=2))
draw=np.tile(np.arange(32,dtype=np.float64)[None,None,:,None],(70,2,1,9))
draw[:,0,24:,0]=np.nan;draw[:,1,24:,0]=np.nan
U=np.full((70,2,32,6),.5);cov=coverage017(draw,draw,U)
check('PIT_actual48finite_denominator',cov['supported'][:,0].all() and (cov['n_student_finite'][:,0]==48).all())
empty=draw.copy();empty[0,:,:,0]=np.nan;cov=coverage017(empty,draw,U)
check('PIT_zerofinite_pending_not0',not cov['supported'][0,0] and np.isnan(cov['rates'][0,0]).all() and (cov['hits'][0,0]==-1).all())
cell=support_cell017(np.ones(100,bool),np.ones(100),np.arange(100)<94)
check('original_group95percent_support_rule',cell['status']=='pending' and cell['original_parents']==100 and cell['supported_parents']==94)

result=dict(artificial_only=True,passed=True,checks=checks,groups=len(checks),
    artificial_work=dict(original_CP_loads=0,original_SOURCE_REAL_DEV_reads=0,GPU_calls=0,new_random_scalar_requests=0,
        artificial_backwards=3,artificial_optimizer_steps=1,artificial_2theta_forwards=2),
    sources={name:dict(bytes=(ROOT/name).stat().st_size,sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
        for name in ('src/n01/model017.py','src/n01/stats017.py','src/n01/saved_array_kernels017.py')})
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
with (out/'ENGINEERING_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,groups=len(checks),receipt=str(out/'ENGINEERING_RECEIPT.json'))))
