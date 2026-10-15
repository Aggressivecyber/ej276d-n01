"""Exactly one artificial full1024 unpaired CPU Engine update, no original IO."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import ctypes
if os.name=='nt':
    import platform
    platform._wmi=None
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False,parents=True)
import numpy as np
import torch
from run_wp4_017 import settings017,budget_caps017,Store017,Runtime017,Engine017,sha,read_json
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
source=ROOT/'receipts/wp4_017_20261015/engineering_01/artificial_driver_01/artifacts/cache'
cache={k:np.load(source/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('A','B','Q')}
with np.load(source/'PARENT_SPLIT.npz',allow_pickle=False) as a:cache.update(fit=a['fit_rows'],test=a['test_rows'],ids=a['source_ids'])
cache.update(directory=source,manifest=read_json(source/'CACHE_MANIFEST.json'))
class FakeExtension:
    def assert_frozen(self):return True
class ManualRuntime(Runtime017):
    def noises(self,nz,epsilon):
        def value():
            prefix=nz['baseline_gaussian'][:,0,:].double()*1e-4;tail=epsilon[:,0,:].double()*1e-4
            return {h:torch.cat((prefix,tail*scale),1) for h,scale in zip(('fast','slow','common_level'),(1.,1.1,.9))}
        return self.call('artificial_electronics_no_NN',dict(noise005_rows=len(epsilon),all_three_extensions_rows=3*len(epsilon)),value)
    def verify(self,*,files=False):
        assert all(sha(source/name)==row['sha256'] for name,row in self.cache['manifest']['cache_files'].items())

p=settings017();p['CPU_threads']=2;p['budget_caps']=budget_caps017(p)
store=Store017(OUT/'artifacts',p,{},artificial=True);store.status()
rt=ManualRuntime(p,cache,FakeExtension(),{'loss':{'q_floor':1e-6,'peak_floor':1e-5}},dict(artificial_fixed_base=True),store,'cpu')
engine=Engine017(rt,'T1','unpaired',0,store.root/'T1_unpaired_seed0')
shapes={};storages={}
def pack(x):
    key=str(tuple(x.shape));shapes[key]=shapes.get(key,0)+1
    storages[(x.untyped_storage().data_ptr(),x.untyped_storage().nbytes())]=True
    assert not (x.ndim>=3 and x.shape[-3:]==(1024,1024,257)),'full broadcast difference retained'
    return x
def unpack(x):return x
started=time.monotonic();before=torch.get_rng_state().clone()
print('one artificial1024 Engine forward/backward, three hypotheses',flush=True)
with torch.autograd.graph.saved_tensors_hooks(pack,unpack):metrics=engine.step()
elapsed=time.monotonic()-started
if os.name=='nt':
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    c=Counters();c.cb=ctypes.sizeof(c);kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=ctypes.c_void_p
    psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    assert psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb)
    memory=dict(measurement='owned Windows GetProcessMemoryInfo native peak',peak_RSS_bytes=c.PeakWorkingSetSize,
        current_RSS_bytes=c.WorkingSetSize,peak_commit_bytes=c.PeakPagefileUsage)
else:
    import resource
    memory=dict(measurement='owned getrusage ru_maxrss',peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
assert memory['peak_RSS_bytes']<8*1024**3
assert store.counts['new_AD_queries']==store.counts['training_backward_queries']==store.counts['G_updates']==store.counts['optimizer_steps']==1
assert store.counts['training_student_rows']==store.counts['training_teacher_rows']==1024
assert store.counts['unpaired_training_energy_pair_distances']==3*2*1024**2
assert store.counts['unpaired_training_self_ordered_retained_pairs']==3*1024*1023
assert store.counts['unpaired_training_self_diagonal_computed_then_masked']==3*1024
assert all(v!=0 for v in metrics['preclip_gradients']) and all(v!=0 for v in metrics['actual_parameter_deltas'])
assert torch.equal(before,torch.get_rng_state())
result=dict(passed=True,artificial_only=True,elapsed_seconds=elapsed,memory=memory,
    saved_tensor_shapes_with_multiplicity=shapes,unique_saved_storage_bytes=sum(k[1] for k in storages),
    proper_exact_U_statistic=True,one_backward=True,no_detach_no_objective_change=True,
    full1024x1024x257_difference_not_materialized=True,three_hypotheses=True,metrics=metrics,
    counters=store.counts,attempted=store.attempted,inflight=store.inflight,
    actual_scientific_CP_NN_AD_G_RNG_SOURCE_REAL_DEV_GPU=0,engineering_backward_queries=1,engineering_optimizer_steps=1,
    GPU_memory_acceptance='not measured; native owned supervisor cap remains10GiB',
    sources={n:dict(bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in ('scripts/run_wp4_017.py','src/n01/model017.py')})
(OUT/'ENGINEERING_RECEIPT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps(dict(passed=True,elapsed_seconds=elapsed,memory=memory,unique_saved_storage_bytes=result['unique_saved_storage_bytes'],gradients=metrics['preclip_gradients'])),flush=True)
