"""One-line bit-algebra regression with deterministic artificial arrays only."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;POST=HERE.parent/'postrun_independent_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def record(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))


def main():
    out=Path(sys.argv[1]).resolve();assert out.drive.upper() in ('E:','D:') and not out.exists();out.mkdir(parents=True)
    (out/'tmp').mkdir();os.environ.update(TEMP=str(out/'tmp'),TMP=str(out/'tmp'),TMPDIR=str(out/'tmp'))
    before=HERE/'before_saved_residual_bits_fix01/readback_saved_arrays.py';after=POST/'readback_saved_arrays.py'
    assert sha(before)=='bc8b6b5f15080049fe6d9da275392f08b33a6fc5b484975c8724ac47d96e5d09'
    oldline=b"    equal(mu-saved_cpu,saved_difference,'charge saved canonical residual algebra',atol=0.,rtol=64*np.finfo(np.float64).eps)"
    newline=b"    require(np.array_equal((mu-saved_cpu).view(np.uint64),saved_difference.view(np.uint64)),'charge saved canonical residual algebra bits')"
    assert before.read_bytes().count(oldline)==1 and before.read_bytes().replace(oldline,newline,1)==after.read_bytes()
    old=load_module(before,'residual_before016');new=load_module(after,'residual_after016')
    f=load_module(HERE/'check_charge016.py','independent_scalar_fixture016')
    for w in (old,new):w.np=np;w.P,w.R,w.N=3,4,12
    def verify(w,case):
        holder=SimpleNamespace(errors={})
        def equal(*args,**kwargs):return w.Audit.equal(holder,*args,**kwargs)
        return w.verify_charge_arrays(*case,bank=0,equal=equal)
    base=f.fixture();verify(new,base)
    alternate=copy.deepcopy(base)
    alternate[0]['Q_mu_CPU_recomputed_draws_Vns'][0,0]=np.nextafter(alternate[0]['Q_mu_CPU_recomputed_draws_Vns'][0,0],np.inf)
    alternate[0]['Q_mu_GPU_minus_CPU_draws_Vns']=alternate[0]['Q_mu_draws_Vns']-alternate[0]['Q_mu_CPU_recomputed_draws_Vns']
    alternate[1]['Q_mu_CPU_GPU_absmax_Vns']=float(np.abs(alternate[0]['Q_mu_GPU_minus_CPU_draws_Vns']).max())
    verify(new,alternate)
    bad=copy.deepcopy(alternate);original=float(bad[0]['Q_mu_GPU_minus_CPU_draws_Vns'][0,0])
    bad[0]['Q_mu_GPU_minus_CPU_draws_Vns'][0,0]=np.nextafter(original,np.inf)
    assert struct_bits(original)!=struct_bits(float(bad[0]['Q_mu_GPU_minus_CPU_draws_Vns'][0,0]))
    assert abs(struct_bits(original)-struct_bits(float(bad[0]['Q_mu_GPU_minus_CPU_draws_Vns'][0,0])))==1
    verify(old,bad) # Reproduce the permissive relative tolerance in the prior line.
    negative=[]
    for name,case in [('one_ULP_saved_residual_tamper',bad),('negative_zero_saved_residual',copy.deepcopy(base))]:
        if name.startswith('negative_zero'):case[0]['Q_mu_GPU_minus_CPU_draws_Vns'][0,0]=-0.
        try:verify(new,case)
        except ValueError as error:
            assert str(error)=='charge saved canonical residual algebra bits';negative.append(name)
        else:raise AssertionError('bit mismatch was accepted '+name)
    assert not {'torch','n01'}&set(sys.modules)
    result=dict(schema='n01/wp3-016-readback-saved-residual-bits-focused/v1',UTC=datetime.now(timezone.utc).isoformat(),passed=True,artificial_only=True,
        exact_one_line_bytes_changed=True,prior_worker=record(before),worker=record(after),
        positive_cases=['unchanged_deterministic_scalar_reference','legal_one_ULP_alternate_CPU_sum_with_exact_saved_residual'],
        negative_cases=negative,prior_relative_tolerance_accepts_one_ULP_tamper=True,new_bit_guard_rejects_one_ULP_tamper=True,
        A_point_gate_canonical_double_reduction_bounds_and_all_other_worker_bytes_unchanged=True,
        prior_full_charge_suite=record(HERE/'artificial_charge_02/ENGINEERING_RECEIPT.json'),
        prior_full_stdlib_suite=record(HERE/'artificial_stdlib_02/ENGINEERING_RECEIPT.json'),
        full_suites_rerun=False,actual_private_NPZ_CP_reads=0,NN_AD_RNG_training_GPU_SSH=0,
        controller=record(HERE/'readback_control.py'),kernel=record(POST/'saved_array_kernels.py'),checker=record(Path(__file__)))
    path=out/'ENGINEERING_RECEIPT.json'
    with path.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,receipt_sha256=sha(path),worker_sha256=sha(after),worker_bytes=after.stat().st_size,negative_cases=negative)))


def struct_bits(value):
    import struct
    return struct.unpack('<Q',struct.pack('<d',value))[0]


if __name__=='__main__':main()
