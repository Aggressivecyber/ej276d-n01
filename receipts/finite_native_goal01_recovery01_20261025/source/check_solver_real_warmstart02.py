"""Execute actual fitter callback AST with a saved-record test double; no model."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import argparse,ast,copy,hashlib,json,time
import numpy as np
import scipy
from scipy.optimize import Bounds,NonlinearConstraint,minimize
import solver_cached_violation01

parser=argparse.ArgumentParser();parser.add_argument('--source-config',type=Path,required=True);parser.add_argument('--fitter-source',type=Path,required=True);parser.add_argument('--probe-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
p=json.loads(args.source_config.read_bytes());p['solver_options']['maxfev']=2;p['solver_options']['maxiter']=4
with np.load(p['source_joint_solution'],allow_pickle=False) as z:raw=z['primal'].copy()
warm=raw/max(1.,float(abs(raw).sum()));assert warm.shape==(163,)
source=json.loads(Path(p['source_native_result']).read_bytes());prior=json.loads(Path(p['P0_result']).read_bytes());warm_record=prior['cases'][1]
text=args.fitter_source.read_text();tree=ast.parse(text)
functions={v.name:v for v in ast.walk(tree) if isinstance(v,ast.FunctionDef)}
assert all(k in functions for k in ['progress','constraints_of','evaluate','objective','constraint'])
assert "for metric_name in ['wave','QPSD']" in text and 'key=point.tobytes()' in text
assert 'evaluate(solution.x)' not in text
assign=[v for v in tree.body if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='counts' for t in v.targets)][0]
args.probe_root.mkdir(exist_ok=False)
namespace=dict(np=np,json=json,time=time,datetime=datetime,timezone=timezone,scipy=scipy,p=p,started=time.monotonic(),memo={},warm_record=warm_record,reference=source['native']['original']['values'],original=source['records']['original'],model=SimpleNamespace(evaluate=lambda point:copy.deepcopy(warm_record)))
exec(compile(ast.Module(body=[assign]+[functions[k] for k in ['progress','constraints_of','evaluate','objective','constraint']],type_ignores=[]),str(args.fitter_source),'exec'),namespace)
repeat=args.probe_root/'repeat';repeat.mkdir();namespace['args']=SimpleNamespace(out=repeat)
a=namespace['evaluate'](warm);b=namespace['evaluate'](warm.copy());assert a is b
assert namespace['counts']['distinct_solver_points']==1 and namespace['counts']['memo_reuses']==1
repeat_counts=namespace['counts'].copy()
exec(compile(ast.Module(body=[assign],type_ignores=[]),str(args.fitter_source),'exec'),namespace);namespace['memo']={};budget=args.probe_root/'budget';budget.mkdir();namespace['args']=SimpleNamespace(out=budget)
solution=minimize(namespace['objective'],warm,method='COBYQA',bounds=Bounds(-np.ones(163),np.ones(163)),constraints=NonlinearConstraint(namespace['constraint'],-np.inf,0.),options=p['solver_options'])
counts=namespace['counts'];assert counts['distinct_solver_points']==counts['completed_finite_point_evaluations']==solution.nfev==2
assert counts['objective_interface_calls']==2 and solution.status==6,(counts,solution.status,str(solution.message))
record=dict(UTC=datetime.now(timezone.utc).isoformat(),status='PASS_REAL_WARMSTART_ACTUAL_EVALUATOR_CACHE_AND_BUDGET_RETURN',source_config_sha256=hashlib.sha256(args.source_config.read_bytes()).hexdigest(),fitter_source_sha256=hashlib.sha256(args.fitter_source.read_bytes()).hexdigest(),repeat_counts=repeat_counts,budget_counts=counts,solver_nfev=int(solution.nfev),solver_status=int(solution.status),solver_message=str(solution.message),solver_nit=int(solution.nit),probe_limits=dict(maxfev=2,maxiter=4),parameter_dimension=163,constraint_dimension=34,no_rounding_or_numeric_point_changes=True,actual_fitter_functions_executed=['progress','constraints_of','evaluate','objective','constraint'],scientific_N01_model_calls=0,scientific_N01_parameter_fits=0,scientific_N01_candidate_selections=0,software_solver_calls=1,scope='Actual callback code with saved true-point record as test double; not a physical fit or new physical trial.',NAS_access=False)
args.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))