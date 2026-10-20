from pathlib import Path
import ast, textwrap

root=Path('E:/GAN/research/N01'); here=Path(__file__).resolve().parent
old=(root/'receipts/joint_response_01_20261020/wake_20261020_0218_01/AUDIT_COMPLETED_SEED0.py').read_text('utf-8')
header='''from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,numpy as np,torch
from scipy.optimize import minimize

torch.set_num_threads(4)
ROOT=Path('E:/GAN/research/N01');BASE=Path(__file__).resolve().parent
HERE=BASE/'terminal_01';OUT=HERE/'collected/run_01'
p=json.loads((ROOT/'configs/joint_response_01_original_precision_REAL_pilot_20261020.json').read_bytes())
manifest=json.loads((HERE/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
source=json.loads((BASE/'SOURCE_MANIFEST.json').read_bytes())
result=json.loads((OUT/'RESULT.json').read_bytes())
complete=json.loads((OUT/'COMPLETE.json').read_bytes())
assert hashlib.sha256((ROOT/'configs/joint_response_01_original_precision_REAL_pilot_20261020.json').read_bytes()).hexdigest()==source['config_sha256']==result['config_sha256']
assert hashlib.sha256((OUT/'RESULT.json').read_bytes()).hexdigest()==complete['result_sha256']
assert manifest['source_files_verified']==len(source['files'])==819
assert manifest['native_exit']['exit_code']==0 and not manifest['processes']
assert manifest['complete_present'] and not manifest['failed_present']
local_hashes=0
for name,item in source['files'].items():
    if (ROOT/name).is_file():
        raw=(ROOT/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
        local_hashes+=1
for name,digest in p['source_pins'].items():assert source['files'][name]['sha256']==digest,name
for name,item in manifest['outputs'].items():
    raw=(HERE/'collected'/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
raw=(HERE/'terminal.tar').read_bytes();assert len(raw)==manifest['archive_bytes'] and hashlib.sha256(raw).hexdigest()==manifest['archive_sha256']
assert manifest['native_exit']['peak_rss_bytes']<=p['resources']['RSS_GiB']*1024**3
assert result['elapsed_seconds']<=p['resources']['internal_seconds']
assert result['peak_GPU_reserved_bytes']<=p['resources']['GPU_GiB']*1024**3
assert sum(v['bytes'] for v in manifest['outputs'].values())<=p['resources']['output_MiB']*1024**2
with np.load(OUT/'REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as z:roles={k:z[k].copy() for k in z.files}
original_roles=ROOT/'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz'
assert hashlib.sha256(original_roles.read_bytes()).hexdigest()==p['source_pins'][p['roles_file']]
with np.load(original_roles,allow_pickle=False) as z:
    for k in z.files:np.testing.assert_array_equal(roles[k],z[k])
counts=Counter();comparisons=[];tracks={};details={};max_solver_error=0.
hypotheses=('fast','slow','common_level');blocks=('wave','charge_morphology','peak_codes','baseline','joint');fit_labels=('FIT0','FIT1','FIT2')
limits=np.array([1.01,1.01,1.10])[None,:]

def independent_solve(r,J,N,a,p,radius,step):
    operators,offsets=[],[]
    for il,label in enumerate(fit_labels):
        L=a[label+'_cholesky'];scale=np.sqrt(np.square(L).sum(1))
        for ih in range(3):
            sl=slice((3*il+ih)*len(L),(3*il+ih+1)*len(L))
            center=L@(r[sl]*3)/scale
            derivative=(L@(J[sl]*3*radius))/scale[:,None]
            for ib in range(4):
                take=slice(56*ib,56*(ib+1));cap=1.15*np.square(center[take]).mean()+2/4096
                offsets.append(center[take]/np.sqrt(56*cap));operators.append(derivative[take]/np.sqrt(56*cap))
    aa,bb=np.stack(operators),np.stack(offsets)
    scaled=J*radius;gram=scaled.T@scaled
    hh=gram+p['solver']['ridge_fraction']*np.diag(gram).mean()*np.eye(len(step))
    gg=scaled.T@r; nn=N*radius/np.concatenate([a[label+'_baseline_native'] for label in fit_labels])[:,None]
    def objective(u):return float((.5*u@hh@u+gg@u)/(r@r))
    def gradient(u):return (hh@u+gg)/(r@r)
    def constraints(u):return np.r_[1-np.square(bb+aa@u).sum(1),-nn@u,1-u@u]
    def constraint_jacobian(u):return np.vstack((-2*np.einsum('mni,mn->mi',aa,bb+aa@u),-nn,-2*u))
    solved=minimize(objective,np.zeros(len(step)),jac=gradient,method='SLSQP',
        bounds=list(zip((a['lower']-a['theta'])/radius,(a['upper']-a['theta'])/radius)),
        constraints={'type':'ineq','fun':constraints,'jac':constraint_jacobian},
        options={'maxiter':p['constrained_solver']['max_iterations'],'ftol':p['constrained_solver']['ftol'],'disp':False})
    assert solved.success,solved.message
    difference=float(np.max(np.abs(solved.x*radius-step)))
    assert difference<2e-7,difference
    assert constraints(solved.x).min()>=-1e-6
    return difference

'''
body=old[old.index("for kind,dim in"):old.index("assert counts['FD_columns_recomputed']==1200")]
body=body.replace("track=f'seed0_{kind}'","track=f'seed{seed}_{kind}'")
body=body.replace("len(trace['iterations'])==12","len(trace['iterations'])==p['iterations']")
body=body.replace('visits=12*3*number',"visits=p['iterations']*3*number")
body=body.replace('iterations=12,',"iterations=p['iterations'],")
body=body.replace("p['training_seed']+offset+", "p['training_seed']+100000*seed+offset+")
body=body.replace("assert trust['denominator'] == 4096","assert trust['denominator'] == 32768")
body=body.replace('max(4, .005 * 4096)','max(4, .005 * 32768)')
body=body.replace("            assert report['native_projection_contexts'][label]['seed']", """            bank_seed=p['training_seed']+100000*seed+(1000+100*iteration+10*il if label in fit_labels else 70000+100*iteration)
            assert int(a[label+'_random_seed'])==bank_seed
            counts['random_bank_seeds_verified']+=1
            assert report['native_projection_contexts'][label]['seed']""")
body=body.replace("            assert slack >= -1e-6", """            assert slack >= -1e-6
            max_solver_error=max(max_solver_error,independent_solve(r,J,N,a,p,radius,step))
            counts['independent_joint_solves']+=1""")
body=body.replace("elif i + 1 < len(report['trials']):", "elif i + 1 < len(report['trials']) or report['stop_reason']=='zero_joint_direction':")
body=body.replace("assert report['selected_trial'] is None and report['stop_reason']=='trial_budget' and len(report['trials'])==3", """assert report['selected_trial'] is None
            if report['stop_reason']=='zero_joint_direction':
                prefix=f"trial{len(report['trials'])}_";radius=p['initial_radius']*p['rejection_radius_factor']**len(report['trials'])
                np.testing.assert_allclose(a[prefix+'jacobian'],J,rtol=1e-10,atol=1e-11)
                np.testing.assert_allclose(a[prefix+'native_jacobian'],N,rtol=1e-10,atol=1e-11)
                max_solver_error=max(max_solver_error,independent_solve(r,J,N,a,p,radius,np.zeros(dim)))
                g=report['zero_direction_geometry'];assert g['step_norm']==0 and g['unit_step_norm']==0
                np.testing.assert_array_equal(g['original_native_values'],n)
                np.testing.assert_array_equal(g['predicted_native_values'],n)
                assert g['predicted_joint_remaining_fraction']==1 and g['normalized_minimum_slack']>=-1e-6
                counts['independent_joint_solves']+=1;counts['zero_directions_recomputed']+=1
            else:
                assert report['stop_reason']=='trial_budget' and len(report['trials'])==3""")
body += """    for name in ('source','real'):
        assert tracks[track]['coverage'][name]['unique']==len(roles[name+'_train_rows'])
        assert tracks[track]['coverage'][name]['visits']==p['iterations']*3*(1024 if name=='source' else 4096)
    original_summary=result['tracks'][track]
    assert original_summary['accepted_updates']==tracks[track]['accepted_updates']
    assert original_summary['source_parents_visited']==6465 and original_summary['real_records_visited']==18948
"""
evaluation=old[old.index('folder=OUT;'):old.index("assert len(summaries['0']['failed_gates'])")]
evaluation=evaluation.replace("folder/'seed0_EVALUATION.json'","folder/f'seed{seed}_EVALUATION.json'")
evaluation=evaluation.replace("folder/'seed0_DECISION.json'","folder/f'seed{seed}_DECISION.json'")
evaluation=evaluation.replace(';summaries={};kernels={}','')
evaluation += "assert decision==result['decisions'][str(seed)]\n"
ending='''
c=result['execution_counts']
for name,value in c.items():assert value<=p['operation_budget'][name],name
assert c['finite_difference_columns']==counts['FD_columns_recomputed']==600
assert c['FIT_bank_finite_difference_columns']==1800
assert c['joint_candidate_trials']==counts['joint_candidates']
assert c['joint_direction_solves']==c['joint_direction_solve_attempts']==counts['independent_joint_solves']
assert c['secant_matrix_corrections']==counts['secant_corrections_recomputed']
assert c['REAL_response_updates']==sum(v['accepted_updates'] for v in tracks.values())
train_calls=12*8+600*6+4*c['joint_candidate_trials']
assert c['joint_response_calls']==train_calls+16+2048
assert c['joint_response_rows']==train_calls*32768+16*8192+2048*1024
assert c['generated_observation_calls']==3*(train_calls+16)
assert c['generated_observation_rows']==3*(train_calls*32768+16*8192)
assert c['native_component_rows']==3*train_calls*(32768+4096)
assert c['native_metric_rows']==16*3*(8192+2099)
assert c['allocated_banks']==56 and c['noise005_calls']==52
assert c['noise005_rows']==1605632 and c['extension_rows']==4816896
assert c['shape_standard_normal_values']==17039360
assert c['kernel_signal_calls']==2048 and c['kernel_signal_rows']==2097152
assert c['REAL_fit_feature_rows']==147456 and c['REAL_CHECK_feature_rows']==49152 and c['REAL_evaluation_feature_rows']==4198
assert c['checkpoint_loads']==6 and c['TRAIN_archives_decoded']==c['PCA_archives_decoded']==1
assert c['old_A_calls']==52 and c['old_A_rows']==6593 and c['M0_head_calls']==104 and c['M0_head_rows']==13186
assert c['REAL_observable_replay_calls']==1 and c['REAL_observable_replay_rows']==21047
assert c['feature_maps_constructed']==c['new_native_projection_contexts']==5
assert result['native_timing']['calls']==train_calls
passed=all(not v['failed_gates'] for v in summaries.values())
assert result['REAL_calibration_candidate_pass_all']==passed and not result['scientific_acceptance']
assert result['stage5_validation_pending'] and result['final_all_TRAIN_refit_pending']
rejects={track:dict(Counter((x['label']+'/'+reason) for x in comparisons if x['track']==track and not x['accepted'] for reason,failed in [('native',not x['native_pass']),('joint',not x['original_joint_accept']),('trust',not x['trust_pass']),('macro',not x['macro_pass'])] if failed)) for track in tracks}
review=dict(status='PASS_SAVED_ORIGINAL_PRECISION_REAL_PILOT_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),
    source_files_verified=819,local_source_hashes=local_hashes,source_pins=len(p['source_pins']),
    outputs_rehashed=len(manifest['outputs']),output_bytes=sum(x['bytes'] for x in manifest['outputs'].values()),
    native_exit=manifest['native_exit'],elapsed_seconds=result['elapsed_seconds'],config_sha256=result['config_sha256'],
    result_sha256=complete['result_sha256'],execution_counts=c,audit_counts=dict(counts),
    tracks=tracks,iteration_details=details,max_independent_solver_delta_error=max_solver_error,
    final_gates_recomputed=192,final_evidence=summaries,kernel_evidence_recomputed=kernels,
    acceptance_checks=comparisons,rejection_summary=rejects,REAL_calibration_candidate_pass_all=passed,
    scientific_acceptance=False,stage5_pending=True,
    evidence_limit='Saved matrix and vector arithmetic, complete solves including any zero directions, reported gate summaries and kernel moments audited. Original trial waveforms, trust RMS/flag extraction and per-draw kernel accumulation are not regenerated. Reporting roles are historically exposed frozen-system evaluation.')
(HERE/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\\n',encoding='utf-8')
print(json.dumps(dict(status=review['status'],audit_counts=dict(counts),tracks=tracks,final=summaries,execution_counts=c,max_solver_error=max_solver_error)))
'''
source=header+'for seed in range(2):\n'+textwrap.indent(body,'    ')+'\nsummaries={};kernels={}\nfor seed in range(2):\n'+textwrap.indent(evaluation,'    ')+ending
ast.parse(source)
(here/'AUDIT_COMPLETE.py').write_text(source,encoding='utf-8',newline='\n')
print('Prepared complete four-track matrix, solver, final-gate, kernel and operation audit; terminal execution pending.')
