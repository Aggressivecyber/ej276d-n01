"""One bounded fit to the actual full hard-objective and actual constraints."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np
import scipy
from scipy.optimize import Bounds,NonlinearConstraint,minimize
import torch
from finite_native_goal01 import FiniteNativeGoal01
import solver_cached_violation01

parser=argparse.ArgumentParser()
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--config-sha256',required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();started=time.monotonic()
assert hashlib.sha256(args.config.read_bytes()).hexdigest()==args.config_sha256
p=json.loads(args.config.read_bytes());args.out.mkdir(exist_ok=False)
assert scipy.__version__=='1.16.3'
assert Path(scipy.__file__).resolve().is_relative_to(Path(p['solver_runtime']).resolve())
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
counts=dict(parameter_fits=0,candidate_selections=0,solver_calls=0,distinct_solver_points=0,new_finite_point_evaluations=0,completed_finite_point_evaluations=0,reused_P0_points=0,objective_interface_calls=0,constraint_interface_calls=0,memo_reuses=0,cached_observation_rows=0,mean_head_parent_rows=0,inflight_cached_observation_rows_upper=0,inflight_mean_head_parent_rows_upper=0,original_backbone_calls=0,new_stochastic_draws=0,optimizer_updates=0,REPORT_rows_used_in_metrics=0)

def progress(stage,**extra):
    if stage=='actual_hard_guards_complete':
        counts['cached_observation_rows']+=103440;counts['mean_head_parent_rows']+=6465
        counts['inflight_cached_observation_rows_upper']=0;counts['inflight_mean_head_parent_rows_upper']=0
    assert time.monotonic()-started<p['resources']['internal_seconds']
    assert counts['new_finite_point_evaluations']<=p['operation_budget']['new_finite_point_evaluations']
    (args.out/'PROGRESS.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),stage=stage,elapsed_seconds=time.monotonic()-started,execution_counts=counts,**extra))+'\n')

try:
    (args.out/'START.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),argv=sys.argv,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()[19],config_sha256=args.config_sha256,automatic_restart=False),indent=2)+'\n')
    for name,rec in {**p['inputs'],**p['frozen_source_pins'],**p['new_source_pins']}.items():
        path=Path(name)
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        assert path.stat().st_size==rec['bytes'] and digest==rec['sha256'],name
    prerequisite=json.loads(Path(p['P0_result']).read_bytes())
    assert prerequisite['status']=='PASS_EXACT_FINITE_MODEL_ACTUAL_GUARDS_AND_COMPLETE_NATIVE_CDF'
    interface=json.loads(Path(p['solver_interface_verification']).read_bytes())
    assert interface['status']=='PASS_FIXED_TRUE_NATIVE_POINT_AND_CACHED_CONSTRAINT_INTERFACE'
    startup=json.loads(Path(p['solver_startup_verification']).read_bytes())
    assert startup['status']=='PASS_REAL_WARMSTART_ACTUAL_EVALUATOR_CACHE_AND_BUDGET_RETURN'
    model=FiniteNativeGoal01(p,progress)
    reference=model.source['native']['original']['values']
    original=model.source['records']['original']
    def constraints_of(record):
        values=[]
        for bank in range(2):
            for name in p['risk_names']:
                old=original['metrics_by_bank'][bank][name]
                values.append((record['metrics_by_bank'][bank][name]-old)/max(1.,abs(old)))
            values.extend(record['flags'][bank][name]-original['flags'][bank][name] for name in p['flag_names'])
        return np.r_[values,record['values']['QPSD']/reference['QPSD']-1,record['mean_relative_RMS_max']/.01-1]
    warm=model.warmstart/max(1.,float(abs(model.warmstart).sum()))
    warm_record=prerequisite['cases'][1]
    assert warm_record['case']=='fixed_zero_warmstart'
    memo={}
    def evaluate(raw):
        point=np.asarray(raw)/max(1.,float(abs(raw).sum()));key=point.tobytes()
        if key in memo:
            counts['memo_reuses']+=1
            return memo[key]
        assert counts['distinct_solver_points']<p['solver_options']['maxfev']
        counts['distinct_solver_points']+=1
        counts['new_finite_point_evaluations']+=1
        counts['inflight_cached_observation_rows_upper']=103440;counts['inflight_mean_head_parent_rows_upper']=6465
        progress('finite_solver_point_start',point_index=counts['distinct_solver_points'])
        record=model.evaluate(point)
        counts['completed_finite_point_evaluations']+=1
        if counts['distinct_solver_points']==1:
            for metric_name in ['wave','QPSD']:assert abs(record['values'][metric_name]-warm_record['values'][metric_name])<1e-9
            (args.out/'P0_RUNTIME_REPLAY.json').write_text(json.dumps(dict(status='PASS_FIRST_FIT_POINT_WITH_ISOLATED_SOLVER_RUNTIME',values=record['values'],source_P0_values=warm_record['values'],solver=scipy.__version__,solver_path=scipy.__file__),indent=2)+'\n')
        item=dict(point=point.copy(),record=record,constraints=constraints_of(record));memo[key]=item
        row=dict(UTC=datetime.now(timezone.utc).isoformat(),point_index=counts['distinct_solver_points'],coefficients=point.tolist(),values=record['values'],mean_relative_RMS_max=record['mean_relative_RMS_max'],max_constraint=float(item['constraints'].max()),point_seconds=record['seconds'],elapsed_seconds=time.monotonic()-started,execution_counts=counts.copy())
        with (args.out/'POINTS.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        progress('finite_solver_point_complete',**{key:value for key,value in row.items() if key not in ['UTC','execution_counts','elapsed_seconds','coefficients']})
        return item
    def objective(raw):
        counts['objective_interface_calls']+=1
        return evaluate(raw)['record']['values']['wave']/reference['wave']
    def constraint(raw):
        counts['constraint_interface_calls']+=1
        return evaluate(raw)['constraints']
    counts['parameter_fits']=1;counts['solver_calls']=1
    (args.out/'FIT_STARTED.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),parameter_fits=1,candidate_selections=0,coordinate_count=163,optimizer='COBYQA',target='actual_complete_native_hard_wave',source_P0_job=prerequisite['job_number'],all_original_backbone_and54response_coordinates_frozen=True,solver_options=p['solver_options']),indent=2)+'\n')
    progress('solver_start',max_distinct_solver_points=p['solver_options']['maxfev'])
    solution=minimize(objective,warm,method='COBYQA',bounds=Bounds(-np.ones(163),np.ones(163)),constraints=NonlinearConstraint(constraint,-np.inf,0.),options=p['solver_options'])
    point=np.asarray(solution.x)/max(1.,float(abs(solution.x).sum()));counts['candidate_selections']=1
    with np.load(p['source_joint_solution'],allow_pickle=False) as z:reference_native_gradients=z['native'].copy()
    np.savez_compressed(args.out/'FINITE_NATIVE_POINT.npz',primal=point,raw_solver_point=solution.x,coordinate_steps=model.steps,native=reference_native_gradients,head_delta=point[:161]*model.steps[:161],original_head=model.mean['initial_head'],NN_coefficients=point[161:]*model.steps[161:],NN_physical_steps=model.additive_physical_steps,original_theta=model.cache['original_theta'],theta=model.cache['original_theta'],source_FIT_rows=model.mean['source_FIT_rows'],REAL_FIT_rows=model.mean['REAL_FIT_rows'])
    (args.out/'POINT_FROZEN.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),parameter_fits=1,candidate_selections=1,point_sha256=hashlib.sha256((args.out/'FINITE_NATIVE_POINT.npz').read_bytes()).hexdigest(),selection='Only the single COBYQA returned point, regardless of solver status; no seed/factor/model selection',scientific_acceptance=False),indent=2)+'\n')
    progress('frozen_final_point_replay')
    counts['new_finite_point_evaluations']+=1
    counts['inflight_cached_observation_rows_upper']=103440;counts['inflight_mean_head_parent_rows_upper']=6465
    replay=model.evaluate(point,keep_arrays=True);counts['completed_finite_point_evaluations']+=1
    arrays=replay.pop('readout');means=replay.pop('memory_means')
    final_constraints=constraints_of(replay)
    np.savez_compressed(args.out/'FIT_POINT_HARD_READOUT.npz',**arrays,global_generated_indices=model.cache['generated_indices'])
    np.savez_compressed(args.out/'FIT_POINT_MEMORY.npz',candidate=means)
    violation=max(0.,float(final_constraints.max()))
    changes={key:replay['values'][key]/reference[key]-1 for key in ['wave','QPSD']}
    eligible=bool(changes['wave']<=-.02 and changes['QPSD']<=0 and violation<=1e-10 and replay['mean_relative_RMS_max']<=.01+1e-12)
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='COMPLETE_ONE_ACTUAL_FINITE_NATIVE_FIT',job_number=p['job_number'],solver_success=bool(solution.success),solver_status=int(solution.status),solver_message=str(solution.message),solver_nfev=int(solution.nfev),mapped_L1=float(abs(point).sum()),actual_constraints=final_constraints.tolist(),solver_interface_job=interface['job_number'],max_actual_constraint_violation=violation,actual_relative_changes=changes,final_record=replay,eligible_for_one_native_readback=eligible,execution_counts=counts,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(0),parameter_fits=1,candidate_selections=1,scientific_acceptance=False,restricted_average_response_acceptance=False,endpoint_adopted=False,NAS_access=False)
    progress('complete');(args.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    assert sum(path.stat().st_size for path in args.out.iterdir() if path.is_file())<128*1024**2
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE'))+'\n')
    print(json.dumps(result),flush=True)
finally:
    error=sys.exc_info()[1]
    if error is not None:(args.out/'FAILED.json').write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started,execution_counts=counts,automatic_restart=False))+'\n')
