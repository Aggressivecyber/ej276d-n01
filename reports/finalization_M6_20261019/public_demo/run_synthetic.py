"""One synthetic event exercises the frozen mean QP and linear random map."""
import argparse, hashlib, importlib.metadata, json, platform, sys
from pathlib import Path
import numpy as np
from smooth_mean import constraint_space, fit_mean, SOLVER_SETTINGS
from linear_random_transport import raw_warp_delta, project_random

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
grid=np.arange(384);peak=126
mean=.16*np.where(grid<=peak,np.exp(-((grid-peak)/5.)**2),np.exp(-(grid-peak)/20.))
scales=(float(mean[peak]),float(mean[peak-16:peak+240].sum()),float(mean[peak+30:peak+240].sum()))
space=constraint_space(mean,scales,radius=4,lambda_s=.01,lambda_r=1.)
local=np.arange(110,176)
target=-.035*np.exp(-((local-peak)/3.)**2)+.012*np.exp(-((local-peak-8)/6.)**2)
delta,_,info=fit_mean(mean,target,space)
query=local+.8*np.sin(np.pi*np.arange(66)/65)
operator=np.column_stack([project_random(raw_warp_delta(v,query,1.,space['allowed']),space) for v in np.eye(66)])
transport=np.eye(66)+operator
rng=np.random.default_rng(2026101908);a,b=rng.normal(size=(2,66))*.001
linearity=float(np.max(np.abs(operator@(a+b)-operator@a-operator@b)))
factor=np.column_stack([.001*np.sin((np.arange(66)+1)*k*np.pi/67) for k in (1,2,3)])
cov0=factor@factor.T;cov1=(transport@factor)@(transport@factor).T
cov_error=float(np.max(np.abs(cov1-transport@cov0@transport.T)))
mean_area=max(abs(delta[space['front']].sum()),abs(delta[space['tail']].sum()))
random_area=max(float(np.abs(operator[space['front']].sum(0)).max()),float(np.abs(operator[space['tail']].sum(0)).max()))
adapted=mean.copy();adapted[110:176]+=delta
assert int(adapted[112:144].argmax()+112)==peak
assert max(mean_area,random_area,linearity,cov_error)<1e-12
np.savez_compressed(args.output/'SYNTHETIC_ARRAYS.npz',mean_V=mean,adapted_mean_V=adapted,
                   target_local_delta_V=target,local_delta_V=delta,query_absolute=query,random_map=transport,
                   covariance_before_V2=cov0,covariance_after_V2=cov1)
record={'status':'PASS_SYNTHETIC_MATHEMATICS','scope':'one constructed event; no neural inference or measured-response validation',
    'event_count':1,'QP_solves':1,'checkpoint_loads':0,'SOURCE_REAL_data_decodes':0,
    'scientific_acceptance':False,'sampling_seed':2026101908,'solver':info,'solver_settings':SOLVER_SETTINGS,
    'query_absolute_min_max':[float(query.min()),float(query.max())],
    'mean_reference_area_residual_Vns':float(mean_area),'linear_map_area_residual':random_area,
    'linearity_error_V':linearity,'covariance_identity_error_V2':cov_error,
    'python':sys.version,'platform':platform.platform(),
    'dependencies':{k:importlib.metadata.version(k) for k in ('numpy','scipy','osqp','jinja2','joblib')},
    'module_sha256':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ('smooth_mean.py','linear_random_transport.py')},
    'arrays_sha256':hashlib.sha256((args.output/'SYNTHETIC_ARRAYS.npz').read_bytes()).hexdigest()}
(args.output/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:record[k] for k in ('status','QP_solves','mean_reference_area_residual_Vns','linearity_error_V','covariance_identity_error_V2')}))
