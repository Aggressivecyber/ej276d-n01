"""One predeclared fixed real-point replay; no solver fit or point selection."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,resource,sys,time
import numpy as np
import scipy
from scipy.optimize import NonlinearConstraint
from scipy._lib.cobyqa.problem import NonlinearConstraints
import torch
from finite_native_goal01 import FiniteNativeGoal01
import solver_cached_violation01 as repair

parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True);parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();started=time.monotonic()
assert hashlib.sha256(args.config.read_bytes()).hexdigest()==args.config_sha256
p=json.loads(args.config.read_bytes());args.out.mkdir(exist_ok=False)
assert scipy.__version__=='1.16.3' and Path(scipy.__file__).resolve().is_relative_to(Path(p['solver_runtime']).resolve())
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
counts=dict(parameter_fits=0,candidate_selections=0,fixed_point_evaluations=0,completed_fixed_points=0,cached_observation_rows=0,mean_head_parent_rows=0,constraint_interface_calls=0,optimizer_updates=0,new_stochastic_draws=0,original_backbone_calls=0)
def progress(stage,**extra):
    if stage=='actual_hard_guards_complete':counts['cached_observation_rows']+=103440;counts['mean_head_parent_rows']+=6465
    assert time.monotonic()-started<p['resources']['internal_seconds']
    (args.out/'PROGRESS.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),stage=stage,execution_counts=counts,**extra))+'\n')
try:
    (args.out/'START.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),argv=sys.argv,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()[19],config_sha256=args.config_sha256,automatic_restart=False),indent=2)+'\n')
    for name,rec in {**p['inputs'],**p['frozen_source_pins'],**p['new_source_pins']}.items():
        path=Path(name)
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert path.stat().st_size==rec['bytes'] and digest==rec['sha256'],name
    prior=json.loads(Path(p['P0_result']).read_bytes());assert prior['status']=='PASS_EXACT_FINITE_MODEL_ACTUAL_GUARDS_AND_COMPLETE_NATIVE_CDF'
    model=FiniteNativeGoal01(p,progress);point=model.warmstart/max(1.,float(abs(model.warmstart).sum()))
    counts['fixed_point_evaluations']=1;record=model.evaluate(point,keep_arrays=True);counts['completed_fixed_points']=1
    arrays=record.pop('readout');record.pop('memory_means')
    with np.load(p['known_point_readout'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['global_generated_indices'],model.cache['generated_indices'])
        for key,value in arrays.items():np.testing.assert_array_equal(value,z[key])
    expected=prior['cases'][1];assert expected['case']=='fixed_zero_warmstart'
    for key in ['wave','QPSD']:assert abs(record['values'][key]-expected['values'][key])<1e-9
    for bank in range(2):
        assert record['flags'][bank]==expected['flags'][bank]
        for name in p['risk_names']:assert abs(record['metrics_by_bank'][bank][name]-expected['metrics_by_bank'][bank][name])<1e-9
    values=[];original=model.source['records']['original'];reference=model.source['native']['original']['values']
    for bank in range(2):
        for name in p['risk_names']:
            old=original['metrics_by_bank'][bank][name];values.append((record['metrics_by_bank'][bank][name]-old)/max(1.,abs(old)))
        values.extend(record['flags'][bank][name]-original['flags'][bank][name] for name in p['flag_names'])
    constraints=np.r_[values,record['values']['QPSD']/reference['QPSD']-1,record['mean_relative_RMS_max']/.01-1]
    assert constraints.shape==(34,)
    def fixed_constraints(x):
        counts['constraint_interface_calls']+=1
        np.testing.assert_array_equal(x,point)
        return constraints
    nonlinear=NonlinearConstraints([NonlinearConstraint(fixed_constraints,-np.inf,0.)],False,False)
    cub,ceq=nonlinear(point);old=repair.original_violation(nonlinear,point,cub,ceq)
    before=counts['constraint_interface_calls'];shifted=point.copy();shifted[0]=np.nextafter(shifted[0],np.inf)
    cached=nonlinear.violation(shifted,cub,ceq)
    assert counts['constraint_interface_calls']==before
    np.testing.assert_array_equal(cached,old)
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='PASS_FIXED_TRUE_NATIVE_POINT_AND_CACHED_CONSTRAINT_INTERFACE',job_number=p['job_number'],execution_counts=counts,values=record['values'],point_seconds=record['seconds'],elapsed_seconds=time.monotonic()-started,actual_constraint_count=34,cached_violation_exact=True,cached_violation_added_constraint_calls=0,ADC_prefix_baseline_crop_flags_and_indices_exact=True,actual18risks_and14flags_match_P0=True,full_wave_columns=8776,full_QPSD_columns=26,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(0),parameter_fits=0,candidate_selections=0,installed_library_files_modified=False,scientific_acceptance=False,restricted_average_response_acceptance=False,NAS_access=False)
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');(args.out/'COMPLETE.json').write_text('{"status":"COMPLETE"}\n');print(json.dumps(result),flush=True)
finally:
    error=sys.exc_info()[1]
    if error is not None:(args.out/'FAILED.json').write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),execution_counts=counts,elapsed_seconds=time.monotonic()-started,automatic_restart=False))+'\n')
