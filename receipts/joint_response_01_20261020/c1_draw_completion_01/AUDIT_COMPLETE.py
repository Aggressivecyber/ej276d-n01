"""Run after terminal collection; saved arrays only, no generator replay."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, time
import numpy as np
import torch

started=time.monotonic()
root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
out=here/'terminal_01';run=out/'collected/run_01'
p=json.loads((root/'configs/joint_response_01_c1_draw_completion_20261020.json').read_bytes())
manifest=json.loads((here/'SOURCE_MANIFEST.json').read_bytes())
remote=json.loads((out/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
result=json.loads((run/'RESULT.json').read_bytes())
complete=json.loads((run/'COMPLETE.json').read_bytes())
assert remote['native_exit']['exit_code']==0 and remote['processes']==[]
assert remote['complete_present'] and remote['result_present'] and not remote['failed_present']
assert complete['result_sha256']==hashlib.sha256((run/'RESULT.json').read_bytes()).hexdigest()
assert result['config_sha256']==hashlib.sha256((root/'configs/joint_response_01_c1_draw_completion_20261020.json').read_bytes()).hexdigest()
assert (out/'terminal.tar').stat().st_size==remote['archive_bytes']
assert hashlib.sha256((out/'terminal.tar').read_bytes()).hexdigest()==remote['archive_sha256']
local=0
for name,record in manifest['files'].items():
    if (root/name).is_file():
        raw=(root/name).read_bytes();assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],name
        local+=1
assert len(manifest['files'])==remote['source_files_verified']==775
for name,digest in p['source_pins'].items():assert manifest['files'][name]['sha256']==digest,name
for name,record in remote['outputs'].items():
    raw=(out/'collected'/name).read_bytes();assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],name
assert result['elapsed_seconds']<p['resources']['internal_seconds']
assert remote['native_exit']['peak_rss_bytes']<=p['resources']['RSS_GiB']*1024**3
assert result['peak_GPU_allocated_bytes']<=p['resources']['GPU_GiB']*1024**3
assert sum(v['bytes'] for k,v in remote['outputs'].items() if k.startswith('run_01/'))<p['resources']['output_MiB']*1024**2
old=root/'receipts/joint_response_01_20261020/c1_matched_control_01/terminal_01/collected/run_01'
roles=np.load(old/'TEACHER_ROLES.npz',allow_pickle=False);pool=roles['train_rows']
assert len(pool)==6465 and len(roles['heldout_rows'])==128 and np.intersect1d(pool,roles['heldout_rows']).size==0
hypotheses=('fast','slow','common_level')
slices={'wave':(0,56),'charge_morphology':(56,112),'peak_codes':(112,168),'baseline':(168,224),'joint':(224,272)}
streams=[];reports={};maximum_fd=maximum_solve=maximum_risk=maximum_trust=0.
for seed in (1,):
    banks=np.load(run/f'seed{seed}_BANKS.npz',allow_pickle=False)
    before=np.load(old/f'seed{seed}_joint_ITER11.npz',allow_pickle=False)
    theta=banks['theta'];np.testing.assert_array_equal(theta,before['endpoint']);np.testing.assert_array_equal(banks['pool'],pool)
    for role in range(6):
        order=torch.randperm(len(pool),generator=torch.Generator().manual_seed(p['order_seed']+100*seed+role)).numpy()
        np.testing.assert_array_equal(banks['parent_orders'][role],pool[order])
        base=p['bank_seed']+1000000*seed+1000*role
        assert banks['bank_base_seeds'][role]==base
        streams.extend(base+3*c+o for c in range(102) for o in range(3))
    factors=np.load(run/f'seed{seed}_FEATURE_MAPS.npz',allow_pickle=False)
    original_factors=np.load(old/f'seed{seed}_FEATURE_MAPS.npz',allow_pickle=False)
    assert len(factors.files)==18//2
    for key in factors.files:np.testing.assert_array_equal(factors[key],original_factors[key])
    native=result['reports'][str(seed)]
    assert native==json.loads((run/f'seed{seed}_REPORT.json').read_bytes())
    directions={};geometry={}
    for draws in (4,32):
        a=np.load(run/f'seed{seed}_draw{draws}_DIRECTION.npz',allow_pickle=False)
        assert all(np.isfinite(a[k]).all() for k in a.files)
        np.testing.assert_array_equal(a['theta'],theta);assert int(a['draws'])==draws
        upper,lower=a['upper'],a['lower']
        spans=np.minimum(upper,theta+p['solver']['finite_difference_step'])-np.maximum(lower,theta-p['solver']['finite_difference_step'])
        np.testing.assert_array_equal(spans,a['FD_parameter_spans'])
        J=((a['FD_high']-a['FD_low'])/spans[:,None]).T
        maximum_fd=max(maximum_fd,float(np.max(np.abs(J-a['jacobian']))))
        np.testing.assert_allclose(J,a['jacobian'],atol=2e-14,rtol=2e-11)
        gram=J.T@J;ridge=p['solver']['ridge_fraction']*np.diag(gram).mean()
        delta=np.linalg.solve(gram+ridge*np.eye(54),-(J.T@a['residual']))
        delta*=min(1.,p['solver']['radius']/np.linalg.norm(delta))
        maximum_solve=max(maximum_solve,float(np.max(np.abs(delta-a['delta']))))
        np.testing.assert_allclose(delta,a['delta'],atol=2e-12,rtol=2e-9)
        np.testing.assert_allclose(ridge,native['directions'][str(draws)]['ridge'],atol=2e-14,rtol=2e-11)
        singular=np.linalg.svd(J,compute_uv=False)
        np.testing.assert_allclose(singular,native['directions'][str(draws)]['singular_values'],atol=2e-12,rtol=2e-9)
        assert int((singular>.01*singular[0]).sum())==native['directions'][str(draws)]['effective_rank_at_1pct']
        directions[draws]=a['delta']
        geometry[str(draws)]=dict(FIT_linear_risk_ratio=float(np.square(a['residual']+J@a['delta']).sum()/np.square(a['residual']).sum()),
            mean_step_norm=float(np.linalg.norm(a['delta'][:24])),random_step_norm=float(np.linalg.norm(a['delta'][24:53])))
    summaries={}
    for view,map_role in (('CHECK','CHECK'),('VERIFY','FINAL')):
        a=np.load(run/f'seed{seed}_{view}.npz',allow_pickle=False)
        np.testing.assert_array_equal(a['theta'],theta)
        recorded=native['validation'][view]
        names=['old']+[f'draw{d}_trial{i}' for d in (4,32) for i in range(3)]
        risk_views=[]
        for index,name in enumerate(names):
            white=a['old_residual'] if name=='old' else a[name+'_residual']
            white=white.reshape(3,272)*np.sqrt(3.)
            all_risks={}
            for hi,h in enumerate(hypotheses):
                factor=factors[map_role+'_'+h+'_cholesky'];raw=factor@white[hi];variance=np.square(factor).sum(1)
                calculated={'joint_whitened':float(np.square(white[hi]).mean())}
                calculated.update({block:float(np.mean(np.square(raw[lo:hi_index])/variance[lo:hi_index])) for block,(lo,hi_index) in slices.items()})
                reference=recorded['old_risks'][h] if index==0 else recorded['trials'][index-1]['risks'][h]
                for key,value in calculated.items():
                    maximum_risk=max(maximum_risk,abs(value-reference[key]));np.testing.assert_allclose(value,reference[key],atol=2e-14,rtol=2e-10)
                all_risks[h]=calculated
            risk_views.append(all_risks)
        summaries[view]=[];baseline=risk_views[0]
        for index,(name,trial,candidate) in enumerate(zip(names[1:],recorded['trials'],risk_views[1:],strict=True)):
            draws=(4,32)[index//3];fraction=p['solver']['backtracking_fractions'][index%3]
            assert trial['draws']==draws and trial['fraction']==fraction
            proposal=np.clip(theta+fraction*directions[draws],lower,upper)
            np.testing.assert_array_equal(a[name+'_proposal'],proposal)
            relative=a[name+'_relative_RMS'];trust=trial['trust']
            assert relative.shape==(3*6465*32,) and (relative>=0).all() and trust['denominator']==6465*32
            for key,value in (('mean_relative_RMS',relative.mean()),('p99_relative_RMS',np.quantile(relative,.99)),('max_relative_RMS',relative.max())):
                maximum_trust=max(maximum_trust,abs(value-trust[key]));np.testing.assert_allclose(value,trust[key],atol=2e-13,rtol=2e-10)
            trust_pass=(trust['mean_relative_RMS']<=p['solver']['output_TR_mean_RMS_fraction'] and
                trust['p99_relative_RMS']<=p['solver']['output_TR_p99_RMS_fraction'] and
                max(v for d in trust['flag_count_changes'].values() for v in d.values())<=max(4,6465*32*p['solver']['flag_increase_fraction']))
            assert trust_pass==trust['pass']
            ratio=np.mean([candidate[h]['joint_whitened'] for h in hypotheses])/np.mean([baseline[h]['joint_whitened'] for h in hypotheses])
            gain=ratio<=1-p['solver']['minimum_CHECK_gain']
            macro=all(candidate[h][b]<=p['solver']['macro_degradation_factor']*baseline[h][b]+2/(6465*32)
                for h in hypotheses for b in ('wave','charge_morphology','peak_codes','baseline'))
            passed=bool(trust_pass and gain and macro)
            assert passed==trial['passes_original_candidate_rule']
            summaries[view].append(dict(draws=draws,fraction=fraction,joint_risk_ratio=float(ratio),pass_all=passed,trust_pass=bool(trust_pass),macro_pass=bool(macro),gain_pass=bool(gain)))
    joint_pass={str(d):[f for f in p['solver']['backtracking_fractions'] if all(any(t['draws']==d and t['fraction']==f and t['pass_all'] for t in summaries[v]) for v in ('CHECK','VERIFY'))] for d in (4,32)}
    assert joint_pass==native['fractions_passing_both_independent_banks']
    reports[str(seed)]=dict(direction_geometry=geometry,validation=summaries,fractions_passing_both=joint_pass,
        direction_cosine=float(directions[4]@directions[32]/(np.linalg.norm(directions[4])*np.linalg.norm(directions[32]))))
assert len(streams)==len(set(streams))==1836
for key,value in p['operation_budget'].items():assert result['execution_counts'].get(key,0)==value,key
assert set(result['execution_counts'])<=set(p['operation_budget'])
assert result['parameter_updates']==result['REAL_response_updates']==0
assert not result['scientific_acceptance'] and not result['automatic_REAL_promotion'] and not result['report_parents_used']
prior=here.parent/'c1_draw_precision_01/terminal_01'
old_remote=json.loads((prior/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert hashlib.sha256((prior/'terminal.tar').read_bytes()).hexdigest()==old_remote['archive_sha256']
for name,record in old_remote['outputs'].items():
    raw=(prior/'collected'/name).read_bytes();assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],name
old_review=json.loads((prior/'TERMINAL_REVIEW.json').read_bytes())
assert old_review['status']=='PASS_PARTIAL_DRAW_PRECISION_AUDIT_RESOURCE_STOP'
old_low=prior/'collected/run_01/seed1_draw4_DIRECTION.npz'
assert hashlib.sha256(old_low.read_bytes()).hexdigest()==hashlib.sha256((run/'seed1_draw4_DIRECTION.npz').read_bytes()).hexdigest()
low=np.load(old_low,allow_pickle=False);rep=np.load(run/'seed1_draw4_REPLAY.npz',allow_pickle=False)
columns=np.array(p['replay_columns']);np.testing.assert_array_equal(rep['FD_parameter_indices'],columns)
checks=0
for key in ('theta','lower','upper','residual','delta'):
    np.testing.assert_array_equal(rep[key],low[key]);checks+=1
for key in ('FD_high','FD_low','FD_parameter_spans'):
    np.testing.assert_array_equal(rep[key],low[key][columns]);checks+=len(columns)
np.testing.assert_array_equal(rep['jacobian'],low['jacobian'])
np.testing.assert_allclose(((rep['FD_high']-rep['FD_low'])/rep['FD_parameter_spans'][:,None]).T,rep['jacobian'][:,columns],atol=2e-14,rtol=2e-11)
restored=json.loads((run/'RESTORE_REPLAY.json').read_bytes())
assert checks==restored['exact_array_checks']==17 and restored['pass_all'] and restored['feature_factors_exact']==9 and restored['parent_bank_arrays_exact']==4
for name in ('MEMORY_FIT_BANKS.json','MEMORY_AFTER_DIRECTIONS.json'):
    memory=json.loads((run/name).read_bytes());assert memory['full_bank_copies']==0 and memory['peak_RSS_bytes']<=12*1024**3
combined={'0':old_review['reports']['0'],'1':reports['1']}
combined_streams=[p['bank_seed']+1000000*seed+1000*role+3*chunk+offset for seed in (0,1) for role in range(6) for chunk in range(102) for offset in range(3)]
assert len(combined_streams)==len(set(combined_streams))==3672
combined_cost={}
for key in ('finite_difference_columns','joint_response_calls','joint_response_rows','generated_observation_calls','generated_observation_rows'):
    old_key={'finite_difference_columns':'FD_total','joint_response_calls':'generator_calls','joint_response_rows':'generator_rows','generated_observation_calls':'observation_calls','generated_observation_rows':'observation_rows'}[key]
    combined_cost[key]={side:old_review['unobserved_execution_bounds'][old_key+'_'+side]+result['execution_counts'][key] for side in ('lower','upper')}
combined_cost['joint_direction_solves']={'lower':5,'upper':5};combined_cost['joint_candidate_trials']={'lower':24,'upper':24}
review=dict(status='PASS_SAVED_DRAW_COMPLETION_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),
    source_files_verified=775,local_source_files_rehashed=local,pinned_inputs=116,output_files_verified=len(remote['outputs']),
    feature_factors_exact=9,parent_permutations_replayed=6,independent_stream_seeds=1836,
    FD_columns_recomputed=108,joint_solves_recomputed=2,candidate_rules_recomputed=12,
    maximum_FD_error=maximum_fd,maximum_solve_error=maximum_solve,maximum_risk_error=maximum_risk,maximum_trust_error=maximum_trust,
    reports=reports,combined_reports=combined,combined_unique_FD_columns=216,combined_unique_joint_directions=4,combined_unique_candidates=24,combined_execution_cost_bounds=combined_cost,low_draw_replay_exact_checks=checks,high_draw_copy_removed_bytes=3829762560,old_resource_failure_preserved=True,execution_counts=result['execution_counts'],raw_generator_replayed=False,native_flag_counts_not_raw_replayed=True,
    parameter_updates=0,scientific_acceptance=False,elapsed_seconds=time.monotonic()-started)
(out/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(review))
